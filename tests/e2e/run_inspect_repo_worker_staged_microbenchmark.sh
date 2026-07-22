#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_DIR="$(mktemp -d /tmp/local-ai-broker-worker-staged-microbench.XXXXXX)"

SAMPLES="${SAMPLES:-8}"

python3 - <<'PY' "${REPO_ROOT}" "${BASE_DIR}" "${SAMPLES}"
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
base_dir = Path(sys.argv[2]).resolve()
samples = int(sys.argv[3])

sys.path.insert(0, str(repo_root / "workers" / "rag-compression"))
import inspection_index  # noqa: E402


def summarize(values):
    ordered = sorted(float(value) for value in values)
    return {
        "min_ms": round(min(ordered), 3) if ordered else None,
        "mean_ms": round(statistics.fmean(ordered), 3) if ordered else None,
        "median_ms": round(statistics.median(ordered), 3) if ordered else None,
        "p90_ms": round(ordered[min(len(ordered) - 1, max(0, int(len(ordered) * 0.9) - 1))], 3) if ordered else None,
        "samples_ms": [round(value, 3) for value in ordered],
    }


def git_init_repo(root: Path):
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)


def stage_repo(root: Path, *, variant: int):
    root.mkdir(parents=True, exist_ok=True)
    (root / "service.py").write_text(
        "\n".join(
            [
                "def retry_job(job_id):",
                f"    seed = {variant}",
                "    return submit_job(job_id + seed)",
                "",
                "def submit_job(job_id):",
                "    return job_id",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "helper.py").write_text(
        "\n".join(
            [
                "def helper():",
                f"    return {variant}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_build(repo: Path, cache_dir: Path):
    discovered = [{"id": "repo", "type": "repo", "classification": "internal", "path": repo}]
    fp_started = time.perf_counter()
    fingerprint, fingerprint_state = inspection_index.repository_fingerprint(
        discovered,
        cache_dir=cache_dir,
        excluded_paths={cache_dir},
    )
    fingerprint_ms = (time.perf_counter() - fp_started) * 1000.0
    build_started = time.perf_counter()
    chunks, stats = inspection_index.build_syntax_chunks(
        discovered,
        cache_dir=cache_dir,
        excluded_paths={cache_dir},
        repository_state_fingerprint=fingerprint,
        repository_fingerprint_state=fingerprint_state,
        return_diagnostics=True,
    )
    build_ms = (time.perf_counter() - build_started) * 1000.0
    lexical_fp = inspection_index.inspection_index_fingerprint(fingerprint, chunks)
    lexical_started = time.perf_counter()
    _path, _hit, lexical_stats = inspection_index.ensure_lexical_index(
        chunks,
        cache_dir,
        lexical_fp,
        build_config_digest=(
            inspection_index._load_file_chunk_working_manifest(
                inspection_index._file_chunk_manifest_path(cache_dir)
            )
            or {}
        ).get("build_config_digest", ""),
    )
    lexical_ms = (time.perf_counter() - lexical_started) * 1000.0
    chunk_substage = dict(getattr(chunks, "_chunk_build_substage_timings", {}) or {})
    return {
        "fingerprint_ms": round(fingerprint_ms, 3),
        "build_total_ms": round(build_ms, 3),
        "lexical_total_ms": round(lexical_ms, 3),
        "chunk_stats": stats,
        "lexical_stats": lexical_stats,
        "chunk_substage": chunk_substage,
    }


repo = base_dir / "staged-repo"
cache_dir = base_dir / "staged-cache"
stage_repo(repo, variant=100)
git_init_repo(repo)
run_build(repo, cache_dir)

records = []
for i in range(samples):
    (repo / "service.py").write_text(
        "\n".join(
            [
                "def retry_job(job_id):",
                f"    value = {i + 1}",
                "    return submit_job(job_id + value)",
                "",
                "def submit_job(job_id):",
                "    return job_id",
                "",
            ]
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(repo), "add", "service.py"], check=True)
    result = run_build(repo, cache_dir)
    records.append(
        {
            "fingerprint_ms": result["fingerprint_ms"],
            "build_total_ms": result["build_total_ms"],
            "lexical_total_ms": result["lexical_total_ms"],
            "discover_source_files_ms": float(result["chunk_substage"].get("discover_source_files_ms") or 0.0),
            "previous_snapshot_load_ms": float(result["chunk_substage"].get("previous_snapshot_load_ms") or 0.0),
            "file_chunk_bundle_load_ms": float(result["chunk_substage"].get("file_chunk_bundle_load_ms") or 0.0),
            "working_manifest_write_ms": float(result["chunk_substage"].get("working_manifest_write_ms") or 0.0),
            "snapshot_write_ms": float(result["chunk_substage"].get("snapshot_write_ms") or 0.0),
            "git_dirty_manifest_keys_ms": float(result["chunk_substage"].get("git_dirty_manifest_keys_ms") or 0.0),
            "ensure_lexical_index_ms": result["lexical_total_ms"],
            "reused_files": int(result["chunk_stats"].get("reused_files") or 0),
            "rebuilt_files": int(result["chunk_stats"].get("rebuilt_files") or 0),
            "lexical_working_cache_hit": bool(result["lexical_stats"].get("working_cache_hit")),
        }
    )

summary = {
    "samples": samples,
    "worker_staged_microbenchmark": {
        "staged_dirty": {
            key: summarize([record[key] for record in records])
            for key in (
                "fingerprint_ms",
                "build_total_ms",
                "lexical_total_ms",
                "discover_source_files_ms",
                "previous_snapshot_load_ms",
                "file_chunk_bundle_load_ms",
                "working_manifest_write_ms",
                "snapshot_write_ms",
                "git_dirty_manifest_keys_ms",
                "ensure_lexical_index_ms",
                "reused_files",
                "rebuilt_files",
            )
        }
    },
}
summary["worker_staged_microbenchmark"]["staged_dirty"]["lexical_working_cache_hit_count"] = sum(
    1 for record in records if record["lexical_working_cache_hit"]
)
summary["worker_staged_microbenchmark"]["staged_dirty"]["records"] = records
print(json.dumps(summary, indent=2))
PY
