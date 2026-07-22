#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SAMPLES="${SAMPLES:-5}"
MAX_STAGED_FINGERPRINT_MS="${MAX_STAGED_FINGERPRINT_MS:-8.0}"
MAX_STAGED_BUILD_MS="${MAX_STAGED_BUILD_MS:-5.0}"

SUMMARY_JSON="$(SAMPLES="${SAMPLES}" bash "${SCRIPT_DIR}/run_inspect_repo_worker_staged_microbenchmark.sh")"

SUMMARY_JSON="${SUMMARY_JSON}" \
MAX_STAGED_FINGERPRINT_MS="${MAX_STAGED_FINGERPRINT_MS}" \
MAX_STAGED_BUILD_MS="${MAX_STAGED_BUILD_MS}" \
python3 - <<'PY'
import json
import os

summary = json.loads(os.environ["SUMMARY_JSON"])
staged = ((summary.get("worker_staged_microbenchmark") or {}).get("staged_dirty") or {})
fingerprint_mean = float(((staged.get("fingerprint_ms") or {}).get("mean_ms")) or 0.0)
build_mean = float(((staged.get("build_total_ms") or {}).get("mean_ms")) or 0.0)
reused_mean = float(((staged.get("reused_files") or {}).get("mean_ms")) or 0.0)
rebuilt_mean = float(((staged.get("rebuilt_files") or {}).get("mean_ms")) or 0.0)

assert fingerprint_mean > 0.0, summary
assert build_mean > 0.0, summary
assert fingerprint_mean <= float(os.environ["MAX_STAGED_FINGERPRINT_MS"]), summary
assert build_mean <= float(os.environ["MAX_STAGED_BUILD_MS"]), summary
assert rebuilt_mean >= 1.0, summary
assert reused_mean >= 1.0, summary
PY

echo "${SUMMARY_JSON}"
