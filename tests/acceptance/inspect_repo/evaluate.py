#!/usr/bin/env python3
"""Evaluate repo_inspection_v2 retrieval and citation quality.

The evaluator has no third-party dependencies and does not contact a GPU. It can
read saved result fixtures, invoke an arbitrary command once per query, or stage
the broker's worker CLI contract directly.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_GOLDEN = HERE / "golden_queries.json"
DEFAULT_RESULTS = HERE / "fixtures" / "cpu_results.json"
DEFAULT_REPO = HERE.parents[2]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_path(value: str) -> str:
    value = value.replace("\\", "/").strip()
    while value.startswith("./"):
        value = value[2:]
    return value


def paths_match(candidate: str, expected: str) -> bool:
    candidate = normalize_path(candidate)
    expected = normalize_path(expected)
    return candidate == expected or candidate.endswith("/" + expected)


def load_suite(path: Path) -> dict[str, Any]:
    suite = load_json(path)
    queries = suite.get("queries", [])
    if len(queries) != 30:
        raise ValueError(f"golden suite must contain exactly 30 queries, found {len(queries)}")
    identifiers = [query.get("id", "") for query in queries]
    if any(not identifier for identifier in identifiers):
        raise ValueError("every golden query must have a non-empty id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("golden query ids must be unique")
    for query in queries:
        if not str(query.get("query", "")).strip():
            raise ValueError(f"{query['id']}: query must be non-empty")
        if query.get("mode") not in {"auto", "evidence", "answer"}:
            raise ValueError(f"{query['id']}: invalid mode {query.get('mode')!r}")
        if not query.get("relevant_paths"):
            raise ValueError(f"{query['id']}: relevant_paths must be non-empty")
    return suite


def parse_json_output(stdout: str) -> Any:
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
    raise ValueError("command output did not contain a JSON object")


def compact_fixture_result(record: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    """Expand a hand-authored ranked-path snapshot into the public v2 contract."""
    ranked_paths = [normalize_path(path) for path in record.get("ranked_paths", [])]
    evidence = [
        {
            "id": f"ev_{index:03d}",
            "source_refs": [{"path": path, "line_start": 1, "line_end": 1}],
        }
        for index, path in enumerate(ranked_paths[:12], start=1)
    ]
    answer_ready = bool(record.get("answer_ready"))
    payload: dict[str, Any] = {
        "mode": query["mode"],
        "query": query["query"],
        "findings": [],
        "evidence": evidence,
        "quality": {
            "result": "answer_ready" if answer_ready else "evidence_only",
            "retrieval": "gpu" if answer_ready else "lexical_degraded",
            "reranking": "gpu" if answer_ready else "unavailable",
            "synthesis": "gpu" if answer_ready else "not_requested",
            "answer_ready": answer_ready,
        },
        "warnings": [] if answer_ready else ["GPU_RETRIEVAL_UNAVAILABLE"],
        "provenance": {"fixture": "cpu_results"},
        "retrieval": {
            "ranked_candidates": [{"path": path, "rank": rank} for rank, path in enumerate(ranked_paths, 1)]
        },
        "runtime": {"attempts": []},
    }
    if answer_ready:
        cited = [item["id"] for item in evidence[:2]]
        payload["answer"] = record.get("answer", "Fixture answer grounded in the ranked evidence.")
        payload["findings"] = [
            {"summary": "Fixture finding grounded in released evidence.", "evidence_refs": cited}
        ]
        payload["runtime"]["attempts"] = [
            {
                "operation": "semantic_retrieval",
                "tier": "p40-retrieval",
                "status": "succeeded",
                "gpu_count": 1,
                "model_profile": "fixture-retrieval",
                "slurm_job_id": "fixture-retrieval-job",
            },
            {
                "operation": "rerank",
                "tier": "p40-retrieval",
                "status": "succeeded",
                "gpu_count": 1,
                "model_profile": "fixture-reranker",
                "slurm_job_id": "fixture-retrieval-job",
            },
            {
                "operation": "synthesis",
                "tier": "p40-synthesis",
                "status": "succeeded",
                "gpu_count": 1,
                "model_profile": "fixture-synthesis",
                "slurm_job_id": "fixture-synthesis-job",
            },
        ]
    return {"schema_name": "repo_inspection_v2", "schema_version": "2.0.0", "payload": payload}


def load_result_records(path: Path, queries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if path.is_dir():
        records: list[Any] = [load_json(item) for item in sorted(path.glob("*.json"))]
    elif path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raw = load_json(path)
        if isinstance(raw, dict) and "results" in raw:
            records = raw["results"]
        elif isinstance(raw, list):
            records = raw
        elif isinstance(raw, dict):
            records = [{"id": key, "result": value} for key, value in raw.items()]
        else:
            raise ValueError(f"unsupported result fixture shape in {path}")

    results: dict[str, Any] = {}
    for record in records:
        query_id = str(record.get("id", ""))
        if query_id not in queries:
            raise ValueError(f"result fixture has unknown query id {query_id!r}")
        if "ranked_paths" in record:
            results[query_id] = compact_fixture_result(record, queries[query_id])
        else:
            results[query_id] = record.get("result", record)
    return results


def invoke_command(template: str, query: dict[str, Any], repo: Path, timeout: int) -> Any:
    tokens = shlex.split(template)
    values = {
        "id": query["id"],
        "query": query["query"],
        "mode": query["mode"],
        "repo": str(repo),
    }
    command = [token.format(**values) for token in tokens]
    env = os.environ.copy()
    env.update(
        {
            "INSPECT_REPO_EVAL_QUERY_ID": query["id"],
            "INSPECT_REPO_EVAL_QUERY": query["query"],
            "INSPECT_REPO_EVAL_MODE": query["mode"],
            "INSPECT_REPO_EVAL_REPO": str(repo),
        }
    )
    completed = subprocess.run(
        command,
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed for {query['id']} with exit {completed.returncode}: {completed.stderr.strip()}"
        )
    return parse_json_output(completed.stdout)


def invoke_worker_cli(worker: Path, query: dict[str, Any], repo: Path, timeout: int) -> Any:
    with tempfile.TemporaryDirectory(prefix=f"inspect-repo-eval-{query['id']}-") as temp:
        run_dir = Path(temp)
        job_id = f"eval_{query['id']}"
        job_spec = {
            "job_id": job_id,
            "task_type": "inspect_repo",
            "task_params": {
                "query": query["query"],
                "mode": query["mode"],
                # Avoid retrieving the golden query text itself when the suite
                # lives inside the repository under test.
                "exclude_dirs": ["acceptance"],
            },
            "constraints": {
                "retrieval_token_budget": 16000,
                "evidence_token_budget": 4000,
                "final_pack_token_budget": 2048,
                "synthesis_context_token_budget": 16000,
            },
            "output_schema": {"name": "repo_inspection_v2"},
        }
        execution_plan = {
            "job_id": job_id,
            "task_type": "inspect_repo",
            "execution_profile": {"tier": "cpu-rag-indexing", "runtime": "deterministic"},
            "selected_model": "",
            "runtime_backend": "deterministic",
            "resource_tier": "cpu-rag-indexing",
            "runtime_connection": {},
            "gpu_services": [],
        }
        input_manifest = {
            "job_id": job_id,
            "input_refs": [
                {"id": "input_0", "type": "repo", "uri": repo.resolve().as_uri(), "classification": "internal"}
            ],
        }
        for name, value in (
            ("job.json", job_spec),
            ("plan.json", execution_plan),
            ("input.json", input_manifest),
        ):
            (run_dir / name).write_text(json.dumps(value), encoding="utf-8")

        command = [str(worker)]
        if worker.suffix == ".py":
            command.insert(0, sys.executable)
        command.extend(
            [
                "--job-spec",
                str(run_dir / "job.json"),
                "--execution-plan",
                str(run_dir / "plan.json"),
                "--input-manifest",
                str(run_dir / "input.json"),
                "--output-dir",
                str(run_dir / "out"),
            ]
        )
        env = os.environ.copy()
        env["BROKER_GPU_SERVICE_ENABLED"] = "0"
        completed = subprocess.run(
            command,
            cwd=repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"worker failed for {query['id']} with exit {completed.returncode}: {completed.stderr.strip()}"
            )
        result_path = run_dir / "out" / "result.json"
        if result_path.exists():
            return load_json(result_path)
        return parse_json_output(completed.stdout)


def unwrap_result(raw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
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
            raise ValueError("schema envelope payload must be an object")
        return raw, payload
    if "mode" in raw and "evidence" in raw:
        return {"schema_name": "", "schema_version": "", "payload": raw}, raw
    raise ValueError("could not find a repo inspection payload")


def item_path(item: Any) -> str:
    if isinstance(item, str):
        return normalize_path(item)
    if not isinstance(item, dict):
        return ""
    for key in ("path", "file", "uri"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return normalize_path(value)
    refs = item.get("source_refs") or item.get("sources")
    if isinstance(refs, list):
        for ref in refs:
            path = item_path(ref)
            if path:
                return path
    for key in ("source_ref", "metadata"):
        path = item_path(item.get(key))
        if path:
            return path
    return ""


def nested_candidate_lists(value: Any, key: str = "") -> Iterable[list[Any]]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key in {"ranked_candidates", "candidates", "top_candidates", "results"} and isinstance(child, list):
                yield child
            yield from nested_candidate_lists(child, child_key)
    elif isinstance(value, list) and key not in {"evidence", "findings", "source_refs"}:
        for child in value:
            yield from nested_candidate_lists(child)


def ranked_paths(payload: dict[str, Any]) -> list[str]:
    candidates: list[Any] = []
    for candidate_list in nested_candidate_lists(payload.get("retrieval", {})):
        if candidate_list:
            candidates = candidate_list
            break
    diagnostics = payload.get("diagnostics", {})
    retrieval_diagnostics = diagnostics.get("retrieval", {}) if isinstance(diagnostics, dict) else {}
    if not candidates:
        for candidate_list in nested_candidate_lists(retrieval_diagnostics):
            if candidate_list:
                candidates = candidate_list
                break
    if not candidates:
        for candidate_list in nested_candidate_lists(diagnostics):
            if candidate_list:
                candidates = candidate_list
                break
    if not candidates:
        for key in ("ranked_candidates", "candidates", "retrieval"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates = value
                break
    if not candidates:
        candidates = payload.get("evidence", [])
    paths: list[str] = []
    for candidate in candidates:
        path = item_path(candidate)
        if path and path not in paths:
            paths.append(path)
    if not paths:
        for candidate in payload.get("evidence", []):
            path = item_path(candidate)
            if path and path not in paths:
                paths.append(path)
    return paths


def best_prefix_rank(paths: list[str], prefix: str) -> int | None:
    prefix = normalize_path(prefix)
    for rank, path in enumerate(paths, start=1):
        path = normalize_path(path)
        if path == prefix or path.startswith(prefix) or ("/" + prefix) in path:
            return rank
    return None


def contract_and_citations(
    envelope: dict[str, Any], payload: dict[str, Any], query: dict[str, Any]
) -> tuple[list[str], int, int, int]:
    errors: list[str] = []
    if envelope.get("schema_name") != "repo_inspection_v2":
        errors.append(f"schema_name is {envelope.get('schema_name')!r}, expected 'repo_inspection_v2'")
    if envelope.get("schema_version") != "2.0.0":
        errors.append(f"schema_version is {envelope.get('schema_version')!r}, expected '2.0.0'")
    if payload.get("mode") != query["mode"]:
        errors.append(f"mode is {payload.get('mode')!r}, expected {query['mode']!r}")
    if str(payload.get("query", "")).strip() != query["query"]:
        errors.append("result did not preserve the original query")

    quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
    if not quality:
        errors.append("quality must be a non-empty object")
    quality_values = {
        "result": {"answer_ready", "evidence_only", "failed"},
        "retrieval": {"gpu", "lexical_degraded", "failed"},
        "reranking": {"gpu", "unavailable", "failed"},
        "synthesis": {"gpu", "not_requested", "failed"},
    }
    for key, allowed in quality_values.items():
        if quality.get(key) not in allowed:
            errors.append(f"quality.{key} must be one of {sorted(allowed)!r}")
    if not isinstance(quality.get("answer_ready"), bool):
        errors.append("quality.answer_ready must be boolean")
    answer_present = "answer" in payload
    answer = payload.get("answer")
    has_answer = isinstance(answer, str) and bool(answer.strip())
    if answer_present and not has_answer:
        errors.append("answer must be omitted or be a non-empty string")
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
    if not isinstance(payload.get("findings"), list):
        errors.append("findings must be an array")
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), list) else []
    if not isinstance(payload.get("evidence"), list):
        errors.append("evidence must be an array")
    retrieval = payload.get("retrieval")
    runtime = payload.get("runtime")
    if not isinstance(retrieval, dict):
        errors.append("retrieval must be an object")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("attempts"), list):
        errors.append("runtime must be an object with an attempts array")
    if not isinstance(payload.get("warnings"), list):
        errors.append("warnings must be an array")
    if not isinstance(payload.get("provenance"), dict):
        errors.append("provenance must be an object")
    if len(evidence) > 12:
        errors.append(f"released {len(evidence)} evidence chunks; maximum is 12")
    evidence_ids_seen: set[str] = set()
    for index, item in enumerate(evidence):
        if not isinstance(item, dict) or not str(item.get("id", "")).strip():
            errors.append(f"evidence[{index}] must have a non-empty id")
            continue
        evidence_id = str(item["id"])
        if evidence_id in evidence_ids_seen:
            errors.append(f"duplicate evidence id {evidence_id!r}")
        evidence_ids_seen.add(evidence_id)
        refs = item.get("source_refs")
        if not isinstance(refs, list) or not refs:
            errors.append(f"evidence {evidence_id!r} must have source_refs")

    valid_citations = 0
    invalid_citations = 0
    missing_references = 0
    if has_answer:
        if query["mode"] == "evidence":
            errors.append("mode=evidence must not return a synthesized answer")
        expected_quality = {
            "result": "answer_ready",
            "retrieval": "gpu",
            "reranking": "gpu",
            "synthesis": "gpu",
            "answer_ready": True,
        }
        for key, expected in expected_quality.items():
            if quality.get(key) != expected:
                errors.append(f"answer-ready result requires quality.{key}={expected!r}")
        attempts = runtime.get("attempts", []) if isinstance(runtime, dict) else []
        successful: dict[str, int] = {}
        seen_p40_synthesis = False
        seen_v100_synthesis = False
        for index, attempt in enumerate(attempts):
            if not isinstance(attempt, dict):
                errors.append(f"runtime.attempts[{index}] must be an object")
                continue
            operation = str(attempt.get("operation", ""))
            tier = str(attempt.get("tier", ""))
            status = str(attempt.get("status", ""))
            if operation == "synthesis":
                if tier == "p40-synthesis":
                    seen_p40_synthesis = True
                elif tier == "v100-reasoning":
                    if not seen_p40_synthesis:
                        errors.append("V100 synthesis attempt appears before P40")
                    seen_v100_synthesis = True
                elif tier in {"a100-single", "a100-multigpu"} and not (
                    seen_p40_synthesis and seen_v100_synthesis
                ):
                    errors.append("A100 synthesis attempt appears before P40 and V100")
            if status != "succeeded" or operation in successful:
                continue
            expected_count = 4 if tier in {"v100-reasoning", "a100-multigpu"} else 1
            if attempt.get("gpu_count") != expected_count:
                errors.append(f"successful {operation} attempt has the wrong GPU count")
            if not str(attempt.get("model_profile", "")).strip():
                errors.append(f"successful {operation} attempt is missing model_profile")
            if not str(attempt.get("slurm_job_id", "")).strip():
                errors.append(f"successful {operation} attempt is missing slurm_job_id")
            successful[operation] = index
        required_operations = {"semantic_retrieval", "rerank", "synthesis"}
        if set(successful) != required_operations:
            errors.append("answer-ready result must record successful GPU retrieval, reranking, and synthesis")
        elif not (
            successful["semantic_retrieval"] < successful["rerank"] < successful["synthesis"]
        ):
            errors.append("successful GPU attempts must be ordered retrieval, reranking, synthesis")
        if not findings:
            errors.append("answer-ready result has no findings")
            missing_references += 1
        evidence_ids = {
            str(item.get("id")) for item in evidence if isinstance(item, dict) and item.get("id")
        }
        for finding in findings:
            refs = finding.get("evidence_refs", []) if isinstance(finding, dict) else []
            refs = [str(ref) for ref in refs if str(ref)] if isinstance(refs, list) else []
            if not refs:
                missing_references += 1
                continue
            for ref in refs:
                if ref in evidence_ids:
                    valid_citations += 1
                else:
                    invalid_citations += 1
    else:
        if answer_present:
            errors.append("evidence-only or failed result must omit answer")
        if findings:
            errors.append("evidence-only result must not contain synthesized findings")
        if quality.get("result") == "failed":
            attempts = payload.get("runtime", {}).get("attempts", [])
            if query["mode"] != "answer":
                errors.append("quality.result=failed is only valid when mode=answer")
            if quality.get("answer_ready") is not False:
                errors.append("failed answer-mode result must report answer_ready=false")
            if not isinstance(attempts, list) or (
                not attempts and not (not evidence and quality.get("retrieval") == "failed")
            ):
                errors.append("failed answer-mode result must preserve the complete runtime attempt history")
        elif quality.get("result") != "evidence_only" or quality.get("answer_ready") is not False:
            errors.append("result without an answer must report evidence_only and answer_ready=false")
        if query["mode"] == "answer" and quality.get("result") != "failed":
            errors.append("mode=answer must return either an answer-ready result or structured failed result")
    return errors, valid_citations, invalid_citations, missing_references


def evaluate(suite: dict[str, Any], raw_results: dict[str, Any], selected_ids: set[str] | None = None) -> dict[str, Any]:
    queries = [query for query in suite["queries"] if not selected_ids or query["id"] in selected_ids]
    if not queries:
        raise ValueError("no golden queries selected")
    per_query: list[dict[str, Any]] = []
    recall_total = 0.0
    reciprocal_rank_total = 0.0
    valid_citations = 0
    invalid_citations = 0
    missing_references = 0
    contract_errors: list[str] = []
    ordering_failures: list[str] = []

    for query in queries:
        query_id = query["id"]
        if query_id not in raw_results:
            contract_errors.append(f"{query_id}: missing result")
            per_query.append({"id": query_id, "recall_at_10": 0.0, "reciprocal_rank": 0.0})
            continue
        try:
            envelope, payload = unwrap_result(raw_results[query_id])
        except ValueError as exc:
            contract_errors.append(f"{query_id}: {exc}")
            per_query.append({"id": query_id, "recall_at_10": 0.0, "reciprocal_rank": 0.0})
            continue
        paths = ranked_paths(payload)
        top_ten = paths[:10]
        relevant = {normalize_path(path) for path in query["relevant_paths"]}
        hits = [path for path in relevant if any(paths_match(candidate, path) for candidate in top_ten)]
        recall = len(hits) / len(relevant)
        first_rank = next(
            (rank for rank, path in enumerate(paths, 1) if any(paths_match(path, target) for target in relevant)),
            None,
        )
        reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
        recall_total += recall
        reciprocal_rank_total += reciprocal_rank

        errors, valid, invalid, missing = contract_and_citations(envelope, payload, query)
        contract_errors.extend(f"{query_id}: {error}" for error in errors)
        valid_citations += valid
        invalid_citations += invalid
        missing_references += missing

        for rule in query.get("ordering_rules", []):
            lower_rank = best_prefix_rank(paths, rule["lower_prefix"])
            for prefix in rule["higher_prefixes"]:
                higher_rank = best_prefix_rank(paths, prefix)
                if higher_rank is None or (lower_rank is not None and higher_rank >= lower_rank):
                    ordering_failures.append(
                        f"{query_id}/{rule['name']}: {prefix!r} rank {higher_rank} is not above "
                        f"{rule['lower_prefix']!r} rank {lower_rank}"
                    )

        per_query.append(
            {
                "id": query_id,
                "recall_at_10": round(recall, 6),
                "reciprocal_rank": round(reciprocal_rank, 6),
                "ranked_paths": paths[:10],
            }
        )

    count = len(queries)
    citation_denominator = valid_citations + invalid_citations + missing_references
    citation_precision = valid_citations / citation_denominator if citation_denominator else 1.0
    metrics = {
        "query_count": count,
        "recall_at_10": recall_total / count,
        "mrr": reciprocal_rank_total / count,
        "citation_precision": citation_precision,
        "missing_evidence_refs": missing_references + invalid_citations,
    }
    thresholds = suite["thresholds"]
    failures = list(contract_errors) + list(ordering_failures)
    if metrics["recall_at_10"] < thresholds["recall_at_10"]:
        failures.append(
            f"Recall@10 {metrics['recall_at_10']:.4f} is below {thresholds['recall_at_10']:.4f}"
        )
    if metrics["mrr"] < thresholds["mrr"]:
        failures.append(f"MRR {metrics['mrr']:.4f} is below {thresholds['mrr']:.4f}")
    if metrics["citation_precision"] < thresholds["citation_precision"]:
        failures.append(
            f"citation precision {metrics['citation_precision']:.4f} is below "
            f"{thresholds['citation_precision']:.4f}"
        )
    if metrics["missing_evidence_refs"] > thresholds["missing_evidence_refs"]:
        failures.append(
            f"missing evidence refs {metrics['missing_evidence_refs']} exceeds "
            f"{thresholds['missing_evidence_refs']}"
        )
    return {
        "suite": suite["name"],
        "passed": not failures,
        "metrics": metrics,
        "thresholds": thresholds,
        "ordering_failures": ordering_failures,
        "contract_errors": contract_errors,
        "failures": failures,
        "queries": per_query,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--results", type=Path, help="JSON/JSONL result fixture or directory")
    source.add_argument("--command", help="command template with {id}, {query}, {mode}, and {repo} placeholders")
    source.add_argument("--worker-cli", type=Path, help="stage and invoke the broker worker CLI directly")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--query-id", action="append", dest="query_ids", help="evaluate only this query id; repeatable")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        suite = load_suite(args.golden)
        query_map = {query["id"]: query for query in suite["queries"]}
        selected = set(args.query_ids or []) or None
        if selected:
            unknown = selected.difference(query_map)
            if unknown:
                raise ValueError(f"unknown query ids: {', '.join(sorted(unknown))}")
        selected_queries = [query for query in suite["queries"] if not selected or query["id"] in selected]
        if args.command:
            results = {
                query["id"]: invoke_command(args.command, query, args.repo.resolve(), args.timeout_seconds)
                for query in selected_queries
            }
        elif args.worker_cli:
            results = {
                query["id"]: invoke_worker_cli(args.worker_cli.resolve(), query, args.repo.resolve(), args.timeout_seconds)
                for query in selected_queries
            }
        else:
            results = load_result_records(args.results or DEFAULT_RESULTS, query_map)
        report = evaluate(suite, results, selected)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"inspect_repo evaluation error: {exc}", file=sys.stderr)
        return 2

    metrics = report["metrics"]
    status = "PASS" if report["passed"] else "FAIL"
    print(
        f"{status} {report['suite']}: queries={metrics['query_count']} "
        f"Recall@10={metrics['recall_at_10']:.3f} MRR={metrics['mrr']:.3f} "
        f"citation_precision={metrics['citation_precision']:.3f} "
        f"missing_refs={metrics['missing_evidence_refs']}"
    )
    if args.verbose or not report["passed"]:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
