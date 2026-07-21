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
  OUTPUT_DIR="${REPO_ROOT}/.broker-live-tests/perf-proof-${stamp}"
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

python3 - <<'PY' "${OUTPUT_DIR}/summary.json"
import json, sys
from pathlib import Path

summary_path = Path(sys.argv[1])
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
checks = summary.get("checks") or {}
required = [
    "cold_hint_skips_repository_fingerprint",
    "warm_hint_skips_repository_fingerprint",
    "partial_dirty_hint_skips_repository_fingerprint",
]
missing = [name for name in required if checks.get(name) is not True]
if missing:
    raise SystemExit(f"saved perf proof missing required broker-hint checks: {missing}")
PY

python3 - <<'PY' "${OUTPUT_DIR}" "${BASE_URL}"
import json, sys
from datetime import datetime, timezone
from pathlib import Path

output_dir = Path(sys.argv[1])
base_url = sys.argv[2]
summary_path = output_dir / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
metadata = {
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "base_url": base_url,
    "runner": "tests/e2e/run_inspect_repo_perf_proof.sh",
    "proof_script": "tests/acceptance/inspect_repo/perf_proof.py",
    "adapter_script": "tests/acceptance/inspect_repo/broker_perf_adapter.py",
    "expected_fingerprint_source": "input_manifest",
    "require_zero_repository_fingerprint_ms": True,
    "proof_ok": bool(summary.get("ok")),
    "proof_checks": {
        "cold_hint_skips_repository_fingerprint": bool(
            (summary.get("checks") or {}).get("cold_hint_skips_repository_fingerprint")
        ),
        "warm_hint_skips_repository_fingerprint": bool(
            (summary.get("checks") or {}).get("warm_hint_skips_repository_fingerprint")
        ),
        "partial_dirty_hint_skips_repository_fingerprint": bool(
            (summary.get("checks") or {}).get("partial_dirty_hint_skips_repository_fingerprint")
        ),
    },
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
exit "${proof_status}"
