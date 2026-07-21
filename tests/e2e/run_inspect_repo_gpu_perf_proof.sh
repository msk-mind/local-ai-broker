#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ADAPTER="${REPO_ROOT}/tests/acceptance/inspect_repo/broker_perf_adapter.py"
PROOF="${REPO_ROOT}/tests/acceptance/inspect_repo/perf_proof.py"

BASE_URL=""
OUTPUT_DIR=""
TIMEOUT_SECONDS=180

while [ "$#" -gt 0 ]; do
  case "$1" in
    --base-url)
      BASE_URL="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS="${2:-}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "usage: $0 --base-url URL [--output-dir DIR] [--timeout-seconds N]" >&2
      exit 2
      ;;
  esac
done

if [ -z "${BASE_URL}" ]; then
  echo "--base-url is required" >&2
  exit 2
fi

if [ -z "${OUTPUT_DIR}" ]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  OUTPUT_DIR="${REPO_ROOT}/.broker-live-tests/gpu-perf-proof-${stamp}"
fi

mkdir -p "${OUTPUT_DIR}"

set +e
python3 "${PROOF}" \
  --git-init \
  --expect-fingerprint-source input_manifest \
  --command "python3 ${ADAPTER} --base-url ${BASE_URL} --repo {repo} --query {query} --mode {mode}" \
  --timeout-seconds "${TIMEOUT_SECONDS}" \
  --output "${OUTPUT_DIR}/summary.json"
proof_status=$?
set -e

set +e
python3 - <<'PY' "${OUTPUT_DIR}/summary.json" "${OUTPUT_DIR}/validation.json"
import json, sys
from pathlib import Path

summary_path = Path(sys.argv[1])
validation_path = Path(sys.argv[2])
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
checks = summary.get("checks") or {}
notes = summary.get("notes") or []
errors = []

def compact_record(record: dict) -> dict:
    quality = dict(record.get("quality") or {})
    retrieval = dict(record.get("retrieval") or {})
    return {
        "quality": quality,
        "semantic_candidates": retrieval.get("semantic_candidates"),
        "reranked_candidates": retrieval.get("reranked_candidates"),
        "query_stage_cache_hit": retrieval.get("query_stage_cache_hit"),
        "fingerprint_sources": retrieval.get("fingerprint_sources"),
        "setup_timings_ms": retrieval.get("setup_timings_ms"),
        "stage_timings_ms": retrieval.get("stage_timings_ms"),
    }

required_checks = [
    "cold_hint_skips_repository_fingerprint",
    "warm_hint_skips_repository_fingerprint",
    "partial_dirty_hint_skips_repository_fingerprint",
    "warm_query_stage_cache_hit",
    "partial_dirty_invalidates_query_cache",
]
missing = [name for name in required_checks if checks.get(name) is not True]
if missing:
    errors.append(f"missing required broker-hint checks: {missing}")

if any("lexical-fallback mode" in str(note) for note in notes):
    errors.append(f"fell back to lexical mode: {notes}")

def require_gpu_quality(label: str, record: dict):
    quality = record.get("quality") or {}
    retrieval = quality.get("retrieval")
    reranking = quality.get("reranking")
    if retrieval != "gpu" or reranking != "gpu":
        errors.append(f"{label} did not report gpu retrieval/reranking: {compact_record(record)}")

def require_positive_candidates(label: str, record: dict):
    retrieval = record.get("retrieval") or {}
    if int(retrieval.get("semantic_candidates") or 0) <= 0:
        errors.append(f"{label} did not produce semantic candidates: {compact_record(record)}")
    if int(retrieval.get("reranked_candidates") or 0) <= 0:
        errors.append(f"{label} did not produce reranked candidates: {compact_record(record)}")

require_gpu_quality("cold", summary.get("cold") or {})
require_gpu_quality("warm", summary.get("warm") or {})
require_gpu_quality("partial_dirty", summary.get("partial_dirty") or {})
require_positive_candidates("cold", summary.get("cold") or {})
require_positive_candidates("partial_dirty", summary.get("partial_dirty") or {})

validation = {
    "ok": not errors,
    "errors": errors,
}
validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
if errors:
    raise SystemExit("\n".join(errors))
PY
validation_status=$?
set -e

python3 - <<'PY' "${OUTPUT_DIR}" "${BASE_URL}" "${proof_status}" "${validation_status}"
import json, sys
from datetime import datetime, timezone
from pathlib import Path

output_dir = Path(sys.argv[1])
base_url = sys.argv[2]
proof_status = int(sys.argv[3])
validation_status = int(sys.argv[4])
summary_path = output_dir / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
validation_path = output_dir / "validation.json"
validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
metadata = {
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "base_url": base_url,
    "runner": "tests/e2e/run_inspect_repo_gpu_perf_proof.sh",
    "proof_script": "tests/acceptance/inspect_repo/perf_proof.py",
    "adapter_script": "tests/acceptance/inspect_repo/broker_perf_adapter.py",
    "expected_fingerprint_source": "input_manifest",
    "require_zero_repository_fingerprint_ms": True,
    "require_gpu_retrieval_and_reranking": True,
    "proof_ok": bool(summary.get("ok")),
    "proof_exit_status": proof_status,
    "validation_ok": bool(validation.get("ok")),
    "validation_exit_status": validation_status,
    "validation_errors": list(validation.get("errors") or []),
    "notes": list(summary.get("notes") or []),
}
(output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
PY

if [ -n "${BROKER_JOB_STORE_PATH:-}" ] && [ -f "${BROKER_JOB_STORE_PATH}" ]; then
  cp "${BROKER_JOB_STORE_PATH}" "${OUTPUT_DIR}/jobs.json"
fi

if [ -n "${BROKER_AUDIT_LOG_PATH:-}" ] && [ -f "${BROKER_AUDIT_LOG_PATH}" ]; then
  cp "${BROKER_AUDIT_LOG_PATH}" "${OUTPUT_DIR}/audit.jsonl"
fi

if [ -n "${BROKER_GPU_SERVICE_REGISTRY_PATH:-}" ] && [ -f "${BROKER_GPU_SERVICE_REGISTRY_PATH}" ]; then
  cp "${BROKER_GPU_SERVICE_REGISTRY_PATH}" "${OUTPUT_DIR}/gpu-services.json"
fi

cat "${OUTPUT_DIR}/summary.json"
if [ "${proof_status}" -ne 0 ]; then
  exit "${proof_status}"
fi
exit "${validation_status}"
