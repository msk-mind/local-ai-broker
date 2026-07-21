#!/usr/bin/env python3
"""Reproducible performance proof for repo_inspection_v2 hot paths.

This harness is intentionally CPU-runnable. It exercises the real inspection
pipeline with deterministic fake GPU clients and reports whether the expected
warm-path fast paths actually trigger:

- repeated identical query skips chunk loading, lexical index setup, lexical
  search, semantic retrieval, and rerank
- partial-dirty run reuses unchanged file chunks and invalidates the query cache

It prints a JSON summary and exits non-zero if an invariant fails.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
WORKER_DIR = REPO_ROOT / "workers" / "rag-compression"
UNIT_DIR = REPO_ROOT / "tests" / "unit"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))
if str(UNIT_DIR) not in sys.path:
    sys.path.insert(0, str(UNIT_DIR))

import inspection_pipeline  # noqa: E402
from test_repo_inspection_v2 import SemanticDiagnosticsFactory, all_services  # noqa: E402


QUERY = "Trace the retry_job service call chain"
FRESH_QUERY = "Trace the submit_job service call chain"
LOCAL_TIMING_BUDGETS_MS = {
    "cold_build_syntax_chunks_ms": 100.0,
    "warm_local_repository_fingerprint_ms": 5.0,
    "fresh_local_repository_fingerprint_ms": 60.0,
    "fresh_local_write_query_stage_cache_ms": 5.0,
    "partial_dirty_repository_fingerprint_ms": 60.0,
    "partial_dirty_build_syntax_chunks_ms": 80.0,
    "partial_dirty_ensure_lexical_index_ms": 10.0,
    "partial_dirty_write_query_stage_cache_ms": 5.0,
}
LOCAL_WALL_BUDGETS_S = {
    "warm_seconds": 0.05,
    "fresh_local_seconds": 0.10,
    "partial_dirty_seconds": 0.10,
}


def _stage_repo(root: Path) -> None:
    (root / "service.py").write_text(
        "def retry_job(job_id):\n    return submit_job(job_id)\n\ndef submit_job(job_id):\n    return job_id\n",
        encoding="utf-8",
    )
    (root / "mcp.go").write_text(
        "package mcp\n\nfunc InspectRepo(query string) string {\n\treturn query\n}\n",
        encoding="utf-8",
    )


def _git_init_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)


def _run_with_counts(discovered, output_dir: Path, query: str = QUERY, *, task_params: dict | None = None):
    factory = SemanticDiagnosticsFactory()
    counts = {"build_syntax_chunks": 0, "ensure_lexical_index": 0, "lexical_search": 0}
    original_build = inspection_pipeline.build_syntax_chunks
    original_ensure = inspection_pipeline.ensure_lexical_index
    original_lexical = inspection_pipeline.lexical_search

    def counted_build(*args, **kwargs):
        counts["build_syntax_chunks"] += 1
        return original_build(*args, **kwargs)

    def counted_ensure(*args, **kwargs):
        counts["ensure_lexical_index"] += 1
        return original_ensure(*args, **kwargs)

    def counted_lexical(*args, **kwargs):
        counts["lexical_search"] += 1
        return original_lexical(*args, **kwargs)

    started = time.perf_counter()
    with (
        mock.patch.object(inspection_pipeline, "build_syntax_chunks", side_effect=counted_build),
        mock.patch.object(inspection_pipeline, "ensure_lexical_index", side_effect=counted_ensure),
        mock.patch.object(inspection_pipeline, "lexical_search", side_effect=counted_lexical),
    ):
        payload = inspection_pipeline.run_inspection(
            discovered,
            query,
            mode="evidence",
            task_params={
                "index_cache_dir": str(output_dir / "repo-inspection-v2-cache"),
                **(task_params or {}),
            },
            services=all_services(),
            client_factory=factory,
            output_dir=output_dir,
        )["payload"]
    elapsed = time.perf_counter() - started
    return {
        "seconds": round(elapsed, 4),
        "payload": payload,
        "counts": counts,
        "semantic_search_calls": list(factory.semantic_search_calls),
        "rerank_calls": list(factory.rerank_calls),
        "ensure_semantic_index_calls": list(factory.ensure_semantic_index_calls),
    }


def parse_json_output(stdout: str):
    text = stdout.strip()
    if not text:
        raise ValueError("command produced no JSON output")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
                if isinstance(value, (dict, list)):
                    return value
            except json.JSONDecodeError:
                continue
        for line in reversed(text.splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise ValueError("command output did not contain JSON")


def unwrap_result(raw):
    if not isinstance(raw, dict):
        raise ValueError("result must be a JSON object")
    if isinstance(raw.get("content"), list):
        for item in raw["content"]:
            if isinstance(item, dict) and item.get("type") == "text":
                try:
                    return unwrap_result(json.loads(item.get("text", "")))
                except (json.JSONDecodeError, ValueError):
                    continue
    if isinstance(raw.get("result"), dict) and "schema_name" not in raw:
        return unwrap_result(raw["result"])
    if raw.get("schema_name"):
        payload = raw.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("schema payload must be an object")
        return payload
    if "mode" in raw and "evidence" in raw:
        return raw
    raise ValueError("could not unwrap repo inspection payload")


def _run_via_command(command_template: str, repo_root: Path, timeout_seconds: int):
    tokens = shlex.split(command_template)
    proof_run_nonce = repo_root.name

    def invoke(query: str, *, request_label: str, extra_task_params: dict | None = None):
        values = {"repo": str(repo_root), "query": query, "mode": "evidence"}
        command = [token.format(**values) for token in tokens]
        env = os.environ.copy()
        env.update(
            {
                "INSPECT_REPO_PERF_REPO": str(repo_root),
                "INSPECT_REPO_PERF_QUERY": query,
                "INSPECT_REPO_PERF_MODE": "evidence",
            }
        )
        merged_extra_task_params = {
            "proof_request_nonce": f"{proof_run_nonce}:{request_label}",
        }
        if extra_task_params:
            merged_extra_task_params.update(extra_task_params)
        env["INSPECT_REPO_EXTRA_TASK_PARAMS"] = json.dumps(
            merged_extra_task_params, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            raise RuntimeError(
                f"command failed with exit {completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
            )
        payload = unwrap_result(parse_json_output(completed.stdout))
        return {"seconds": round(elapsed, 4), "payload": payload}

    cold = invoke(QUERY, request_label="cold")
    warm = invoke(QUERY, request_label="warm")
    worker_warm = invoke(QUERY, request_label="worker-warm", extra_task_params={"client_nonce": "worker-warm"})
    (repo_root / "service.py").write_text(
        "def retry_job(job_id):\n    value = submit_job(job_id)\n    return value\n\ndef submit_job(job_id):\n    return job_id\n",
        encoding="utf-8",
    )
    partial_dirty = invoke(QUERY, request_label="partial-dirty")
    return {
        "mode": "command",
        "cold": cold,
        "warm": warm,
        "worker_warm": worker_warm,
        "partial_dirty": partial_dirty,
    }


def _attempt_reason(payload: dict, operation: str) -> str:
    for attempt in payload.get("runtime", {}).get("attempts", []):
        if attempt.get("operation") == operation:
            return str(attempt.get("escalation_reason") or "")
    return ""


def _run_inprocess(repo_root: Path, work_root: Path):
    discovered = [{"id": "input_0", "type": "repo", "classification": "internal", "path": repo_root}]
    shared_cache = work_root / "shared-cache"
    cold_out = work_root / "out-cold"
    fresh_out = work_root / "out-fresh"
    hinted_out = work_root / "out-hinted"
    env = {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache)}
    with mock.patch.dict(os.environ, env, clear=False):
        cold = _run_with_counts(discovered, cold_out)
        warm = _run_with_counts(discovered, cold_out)
        fresh_local = _run_with_counts(discovered, fresh_out, FRESH_QUERY)
        fresh_hinted = _run_with_counts(
            discovered,
            hinted_out,
            FRESH_QUERY,
            task_params={
                "_broker_repository_state_fingerprint": "git:broker-hint-fingerprint",
                "_broker_repository_state_fingerprint_source": "request_cache",
            },
        )
    (repo_root / "service.py").write_text(
        "def retry_job(job_id):\n    value = submit_job(job_id)\n    return value\n\ndef submit_job(job_id):\n    return job_id\n",
        encoding="utf-8",
    )
    with mock.patch.dict(os.environ, env, clear=False):
        partial_dirty = _run_with_counts(discovered, cold_out)
    return {
        "mode": "inprocess",
        "cold": cold,
        "warm": warm,
        "fresh_local": fresh_local,
        "fresh_hinted": fresh_hinted,
        "partial_dirty": partial_dirty,
    }


def _has_fingerprint_source(payload: dict, expected_source: str) -> bool:
    sources = payload.get("retrieval", {}).get("fingerprint_sources") or []
    expected = str(expected_source or "").strip()
    return expected != "" and expected in {str(item).strip() for item in sources}


def _repository_fingerprint_ms(payload: dict) -> float:
    return float(payload.get("retrieval", {}).get("setup_timings_ms", {}).get("repository_fingerprint_ms") or 0.0)


def prove(
    *,
    command_template: str | None = None,
    timeout_seconds: int = 180,
    git_init: bool = False,
    expect_fingerprint_source: str | None = None,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="inspect-repo-perf-proof-") as temp_dir:
        root = Path(temp_dir)
        repo_root = root / "repo"
        repo_root.mkdir()
        _stage_repo(repo_root)
        if git_init:
            _git_init_repo(repo_root)
        if command_template:
            run = _run_via_command(command_template, repo_root, timeout_seconds)
        else:
            run = _run_inprocess(repo_root, root)

    checks = {
        "cold_query_stage_cache_miss": run["cold"]["payload"]["retrieval"]["query_stage_cache_hit"] is False,
        "warm_query_stage_cache_hit": run["warm"]["payload"]["retrieval"]["query_stage_cache_hit"] is True,
        "partial_dirty_invalidates_query_cache": run["partial_dirty"]["payload"]["retrieval"]["query_stage_cache_hit"] is False,
        "partial_dirty_reuses_one_file": run["partial_dirty"]["payload"]["retrieval"]["chunk_cache_reused_files"] >= 1,
    }
    if run["mode"] == "inprocess":
        checks.update(
            {
                "partial_dirty_rebuilds_one_file": run["partial_dirty"]["payload"]["retrieval"]["chunk_cache_rebuilt_files"] >= 1,
                "fresh_local_reuses_all_files": run["fresh_local"]["payload"]["retrieval"]["chunk_cache_reused_files"] >= 2,
                "fresh_local_rebuilds_zero_files": run["fresh_local"]["payload"]["retrieval"]["chunk_cache_rebuilt_files"] == 0,
                "fresh_local_skips_chunk_rebuild": run["fresh_local"]["counts"]["build_syntax_chunks"] == 0,
                "fresh_hinted_skips_repository_fingerprint": (
                    float(
                        run["fresh_hinted"]["payload"]["retrieval"]["setup_timings_ms"].get("repository_fingerprint_ms")
                        or 0.0
                    )
                    == 0.0
                ),
                "fresh_hinted_uses_broker_hint_source": _has_fingerprint_source(
                    run["fresh_hinted"]["payload"], "broker_hint"
                ),
                "warm_skips_chunk_loading": run["warm"]["counts"]["build_syntax_chunks"] == 0,
                "warm_skips_lexical_index": run["warm"]["counts"]["ensure_lexical_index"] == 0,
                "warm_skips_lexical_search": run["warm"]["counts"]["lexical_search"] == 0,
                "warm_skips_semantic_search": run["warm"]["semantic_search_calls"] == [],
                "warm_skips_rerank": run["warm"]["rerank_calls"] == [],
                "cold_build_syntax_chunks_budget": (
                    float(
                        run["cold"]["payload"]["retrieval"]["stage_timings_ms"].get("build_syntax_chunks_ms") or 0.0
                    )
                    <= LOCAL_TIMING_BUDGETS_MS["cold_build_syntax_chunks_ms"]
                ),
                "fresh_local_repository_fingerprint_budget": (
                    float(
                        run["fresh_local"]["payload"]["retrieval"]["setup_timings_ms"].get("repository_fingerprint_ms")
                        or 0.0
                    )
                    <= LOCAL_TIMING_BUDGETS_MS["fresh_local_repository_fingerprint_ms"]
                ),
                "warm_local_repository_fingerprint_budget": (
                    float(
                        run["warm"]["payload"]["retrieval"]["setup_timings_ms"].get("repository_fingerprint_ms")
                        or 0.0
                    )
                    <= LOCAL_TIMING_BUDGETS_MS["warm_local_repository_fingerprint_ms"]
                ),
                "fresh_local_write_query_stage_cache_budget": (
                    float(
                        run["fresh_local"]["payload"]["retrieval"]["stage_timings_ms"].get("write_query_stage_cache_ms")
                        or 0.0
                    )
                    <= LOCAL_TIMING_BUDGETS_MS["fresh_local_write_query_stage_cache_ms"]
                ),
                "partial_dirty_build_syntax_chunks_budget": (
                    float(
                        run["partial_dirty"]["payload"]["retrieval"]["stage_timings_ms"].get("build_syntax_chunks_ms")
                        or 0.0
                    )
                    <= LOCAL_TIMING_BUDGETS_MS["partial_dirty_build_syntax_chunks_ms"]
                ),
                "partial_dirty_repository_fingerprint_budget": (
                    float(
                        run["partial_dirty"]["payload"]["retrieval"]["setup_timings_ms"].get("repository_fingerprint_ms")
                        or 0.0
                    )
                    <= LOCAL_TIMING_BUDGETS_MS["partial_dirty_repository_fingerprint_ms"]
                ),
                "partial_dirty_ensure_lexical_index_budget": (
                    float(
                        run["partial_dirty"]["payload"]["retrieval"]["stage_timings_ms"].get("ensure_lexical_index_ms")
                        or 0.0
                    )
                    <= LOCAL_TIMING_BUDGETS_MS["partial_dirty_ensure_lexical_index_ms"]
                ),
                "partial_dirty_write_query_stage_cache_budget": (
                    float(
                        run["partial_dirty"]["payload"]["retrieval"]["stage_timings_ms"].get("write_query_stage_cache_ms")
                        or 0.0
                    )
                    <= LOCAL_TIMING_BUDGETS_MS["partial_dirty_write_query_stage_cache_ms"]
                ),
                "warm_wall_budget": float(run["warm"]["seconds"]) <= LOCAL_WALL_BUDGETS_S["warm_seconds"],
                "fresh_local_wall_budget": (
                    float(run["fresh_local"]["seconds"]) <= LOCAL_WALL_BUDGETS_S["fresh_local_seconds"]
                ),
                "partial_dirty_wall_budget": (
                    float(run["partial_dirty"]["seconds"]) <= LOCAL_WALL_BUDGETS_S["partial_dirty_seconds"]
                ),
            }
        )
    else:
        partial_dirty_total = int(run["partial_dirty"]["payload"]["retrieval"].get("chunk_cache_total_files") or 0)
        partial_dirty_reused = int(run["partial_dirty"]["payload"]["retrieval"].get("chunk_cache_reused_files") or 0)
        partial_dirty_rebuilt = int(run["partial_dirty"]["payload"]["retrieval"].get("chunk_cache_rebuilt_files") or 0)
        checks.update(
            {
                "partial_dirty_avoids_full_rebuild": (
                    partial_dirty_total <= 0 or partial_dirty_rebuilt < partial_dirty_total
                ),
                "partial_dirty_changed_state_reused_or_rebuilt": (
                    partial_dirty_rebuilt >= 1 or partial_dirty_reused >= partial_dirty_total
                ),
                "partial_dirty_lexical_index_delta_update": (
                    (
                        float(run["partial_dirty"]["payload"]["retrieval"].get("lexical_index_sqlite_rebuild_ms") or 0.0)
                        == 0.0
                        and float(run["partial_dirty"]["payload"]["retrieval"].get("lexical_index_sqlite_update_ms") or 0.0)
                        > 0.0
                        and int(run["partial_dirty"]["payload"]["retrieval"].get("lexical_index_updated_files") or 0) >= 1
                    )
                    or (
                        partial_dirty_total > 0
                        and partial_dirty_rebuilt == 0
                        and partial_dirty_reused >= partial_dirty_total
                    )
                ),
                "warm_skips_lexical_search": run["warm"]["payload"]["retrieval"]["lexical_candidates"] == 0,
                "warm_skips_semantic_search": run["warm"]["payload"]["retrieval"]["semantic_candidates"] == 0,
                "warm_skips_rerank": (
                    _attempt_reason(run["warm"]["payload"], "rerank") == "query_stage_cache_hit"
                    or run["warm"]["payload"]["quality"].get("reranking") != "gpu"
                ),
                "worker_warm_query_stage_cache_hit": (
                    run["worker_warm"]["payload"]["retrieval"]["query_stage_cache_hit"] is True
                ),
                "worker_warm_skips_lexical_search": (
                    run["worker_warm"]["payload"]["retrieval"]["lexical_candidates"] == 0
                ),
                "worker_warm_skips_semantic_search": (
                    run["worker_warm"]["payload"]["retrieval"]["semantic_candidates"] == 0
                ),
                "worker_warm_skips_rerank": (
                    _attempt_reason(run["worker_warm"]["payload"], "rerank") == "query_stage_cache_hit"
                    or run["worker_warm"]["payload"]["quality"].get("reranking") != "gpu"
                ),
            }
        )
    if expect_fingerprint_source:
        checks.update(
            {
                "cold_uses_expected_fingerprint_source": _has_fingerprint_source(
                    run["cold"]["payload"], expect_fingerprint_source
                ),
                "warm_uses_expected_fingerprint_source": _has_fingerprint_source(
                    run["warm"]["payload"], expect_fingerprint_source
                ),
                "worker_warm_uses_expected_fingerprint_source": _has_fingerprint_source(
                    run["worker_warm"]["payload"], expect_fingerprint_source
                )
                if "worker_warm" in run
                else True,
                "partial_dirty_uses_expected_fingerprint_source": _has_fingerprint_source(
                    run["partial_dirty"]["payload"], expect_fingerprint_source
                ),
            }
        )
    if expect_fingerprint_source:
        checks.update(
            {
                "cold_hint_skips_repository_fingerprint": _repository_fingerprint_ms(run["cold"]["payload"]) == 0.0,
                "warm_hint_skips_repository_fingerprint": _repository_fingerprint_ms(run["warm"]["payload"]) == 0.0,
                "worker_warm_hint_skips_repository_fingerprint": _repository_fingerprint_ms(
                    run["worker_warm"]["payload"]
                )
                == 0.0
                if "worker_warm" in run
                else True,
                "partial_dirty_hint_skips_repository_fingerprint": _repository_fingerprint_ms(
                    run["partial_dirty"]["payload"]
                )
                == 0.0,
            }
        )
    ok = all(checks.values())
    notes: list[str] = []
    if run["mode"] == "command":
        retrieval = run["warm"]["payload"].get("quality", {}).get("retrieval")
        reranking = run["warm"]["payload"].get("quality", {}).get("reranking")
        if retrieval != "gpu" or reranking != "gpu":
            notes.append(
                "command path reused the persisted query-stage cache in lexical-fallback mode; answer-ready GPU retrieval+rereank were still unavailable"
            )
        if int(run["partial_dirty"]["payload"]["retrieval"].get("chunk_cache_rebuilt_files") or 0) == 0:
            notes.append(
                "command path reused the changed-file chunk state from the shared broker cache instead of rebuilding it locally"
            )
    return {
        "ok": ok,
        "checks": checks,
        "notes": notes,
        "mode": run["mode"],
        "cold": {
            "seconds": run["cold"]["seconds"],
            "retrieval": run["cold"]["payload"]["retrieval"],
            "quality": run["cold"]["payload"].get("quality"),
            **({"counts": run["cold"]["counts"]} if "counts" in run["cold"] else {}),
        },
        "warm": {
            "seconds": run["warm"]["seconds"],
            "retrieval": run["warm"]["payload"]["retrieval"],
            "quality": run["warm"]["payload"].get("quality"),
            **({"counts": run["warm"]["counts"]} if "counts" in run["warm"] else {}),
        },
        **(
            {
                "worker_warm": {
                    "seconds": run["worker_warm"]["seconds"],
                    "retrieval": run["worker_warm"]["payload"]["retrieval"],
                    "quality": run["worker_warm"]["payload"].get("quality"),
                }
            }
            if "worker_warm" in run
            else {}
        ),
        **(
            {
                "fresh_local": {
                    "seconds": run["fresh_local"]["seconds"],
                    "retrieval": run["fresh_local"]["payload"]["retrieval"],
                    "quality": run["fresh_local"]["payload"].get("quality"),
                    **({"counts": run["fresh_local"]["counts"]} if "counts" in run["fresh_local"] else {}),
                }
            }
            if "fresh_local" in run
            else {}
        ),
        **(
            {
                "fresh_hinted": {
                    "seconds": run["fresh_hinted"]["seconds"],
                    "retrieval": run["fresh_hinted"]["payload"]["retrieval"],
                    "quality": run["fresh_hinted"]["payload"].get("quality"),
                    **({"counts": run["fresh_hinted"]["counts"]} if "counts" in run["fresh_hinted"] else {}),
                }
            }
            if "fresh_hinted" in run
            else {}
        ),
        "partial_dirty": {
            "seconds": run["partial_dirty"]["seconds"],
            "retrieval": run["partial_dirty"]["payload"]["retrieval"],
            "quality": run["partial_dirty"]["payload"].get("quality"),
            **({"counts": run["partial_dirty"]["counts"]} if "counts" in run["partial_dirty"] else {}),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command",
        help="Command template to invoke the real broker path; supports {repo}, {query}, and {mode}",
    )
    parser.add_argument("--timeout-seconds", type=int, default=180, help="Per-invocation timeout for --command mode")
    parser.add_argument("--git-init", action="store_true", help="Initialize the staged temp repo as a real git repository")
    parser.add_argument(
        "--expect-fingerprint-source",
        help="Assert that cold, warm, and partial-dirty retrieval report this fingerprint source",
    )
    parser.add_argument("--output", type=Path, help="Optional path to write the JSON proof summary")
    args = parser.parse_args()

    summary = prove(
        command_template=args.command,
        timeout_seconds=args.timeout_seconds,
        git_init=args.git_init,
        expect_fingerprint_source=args.expect_fingerprint_source,
    )
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
