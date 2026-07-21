#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_DIR="$(mktemp -d /tmp/local-ai-broker-warm-daemon-lexical-perf.XXXXXX)"
source "${SCRIPT_DIR}/smoke_lib.sh"

cleanup() {
  kill_pid_if_running "${BROKER_PID:-}"
  rm -rf "${BASE_DIR}"
}
trap cleanup EXIT

INPUT_REPO="${BASE_DIR}/repo"
mkdir -p "${INPUT_REPO}"
cat > "${INPUT_REPO}/service.py" <<'EOF'
def retry_job(job_id):
    return submit_job(job_id)

def submit_job(job_id):
    return job_id
EOF
cat > "${INPUT_REPO}/helper.py" <<'EOF'
def helper():
    return retry_job(1)
EOF

git -C "${INPUT_REPO}" init -q
git -C "${INPUT_REPO}" config user.email test@example.invalid
git -C "${INPUT_REPO}" config user.name Test
git -C "${INPUT_REPO}" add .
git -C "${INPUT_REPO}" commit -qm initial

export BROKER_LISTEN_ADDR="$(pick_free_loopback_addr)"
export BROKER_JOB_STORE_PATH="${BASE_DIR}/jobs.json"
export BROKER_RUN_ROOT_PATH="${BASE_DIR}/runs"
export BROKER_REPO_ROOT_PATH="${REPO_ROOT}"
export BROKER_REPO_INSPECTION_SHARED_CACHE_DIR="${BASE_DIR}/shared-cache"
export BROKER_BACKEND="local"
export BROKER_LOCAL_MODE="command"
export BROKER_LOCAL_SCRIPT_PATH="${REPO_ROOT}/deploy/local/broker_worker.sh"
export BROKER_AUDIT_LOG_PATH="${BASE_DIR}/audit.jsonl"
export BROKER_AUDIT_VERIFY_MODE="warn"
export BROKER_LOG_PATH="${BASE_DIR}/broker.log"
export CGO_ENABLED=0

start_broker_server "${REPO_ROOT}"

python3 "${REPO_ROOT}/tests/acceptance/inspect_repo/broker_perf_adapter.py" \
  --base-url "http://${BROKER_LISTEN_ADDR}" \
  --repo "${INPUT_REPO}" \
  --query "Trace the retry_job service call chain" \
  --mode evidence >/dev/null

SECOND_JSON="$(
python3 "${REPO_ROOT}/tests/acceptance/inspect_repo/broker_perf_adapter.py" \
  --base-url "http://${BROKER_LISTEN_ADDR}" \
  --repo "${INPUT_REPO}" \
  --query "Trace how helper reaches retry_job" \
  --mode evidence
)"

SECOND_JSON="${SECOND_JSON}" python3 - <<'PY'
import json
import os

payload = (json.loads(os.environ["SECOND_JSON"]).get("result") or {}).get("payload") or {}
runtime = payload.get("runtime") or {}
retrieval = payload.get("retrieval") or {}
stage_timings = retrieval.get("stage_timings_ms") or {}

assert runtime.get("warm_daemon_active") is True, runtime
assert runtime.get("broker_result_source") == "preferred_inline_release", runtime
assert float(stage_timings.get("ensure_lexical_index_ms") or 0.0) <= 8.0, stage_timings
assert float(runtime.get("worker_phase_timings_ms", {}).get("total") or 0.0) <= 20.0, runtime
PY

echo "${SECOND_JSON}"
