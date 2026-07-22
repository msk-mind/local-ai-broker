#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_DIR="$(mktemp -d /tmp/local-ai-broker-inspect-repo-warm-snapshot-hint.XXXXXX)"
source "${SCRIPT_DIR}/smoke_lib.sh"

cleanup() {
  kill_pid_if_running "${BROKER_PID:-}"
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
export BROKER_BACKEND="local"
export BROKER_LOCAL_MODE="command"
export BROKER_LOCAL_SCRIPT_PATH="${REPO_ROOT}/deploy/local/broker_worker.sh"
unset BROKER_LOCAL_INSPECT_REPO_WARM_ENABLED
export BROKER_AUDIT_LOG_PATH="${BASE_DIR}/audit.jsonl"
export BROKER_AUDIT_VERIFY_MODE="warn"
export CGO_ENABLED=0

GO_BIN="${GO_BIN:-$(command -v go)}"
env -u GOROOT CGO_ENABLED="${CGO_ENABLED}" GOCACHE=/tmp/local-ai-broker-gocache GOPATH=/tmp/local-ai-broker-gopath \
  "${GO_BIN}" run "${REPO_ROOT}/broker/cmd/broker-server" &
BROKER_PID=$!
wait_for_http_ok "http://${BROKER_LISTEN_ADDR}/healthz" 800 0.1

FIRST_JSON="$(python3 "${REPO_ROOT}/tests/acceptance/inspect_repo/broker_perf_adapter.py" \
  --base-url "http://${BROKER_LISTEN_ADDR}" \
  --repo "${INPUT_REPO}" \
  --query "Trace the retry_job service call chain" \
  --mode evidence)"

SECOND_JSON="$(python3 "${REPO_ROOT}/tests/acceptance/inspect_repo/broker_perf_adapter.py" \
  --base-url "http://${BROKER_LISTEN_ADDR}" \
  --repo "${INPUT_REPO}" \
  --query "Trace how helper reaches retry_job" \
  --mode evidence)"

FIRST_JSON="${FIRST_JSON}" SECOND_JSON="${SECOND_JSON}" python3 - <<'PY'
import json
import os

first = json.loads(os.environ["FIRST_JSON"])
second = json.loads(os.environ["SECOND_JSON"])

first_payload = (first.get("result") or {}).get("payload") or {}
second_payload = (second.get("result") or {}).get("payload") or {}
first_runtime = first_payload.get("runtime") or {}
second_runtime = second_payload.get("runtime") or {}
second_retrieval = second_payload.get("retrieval") or {}
second_chunk_substages = second_retrieval.get("chunk_build_substage_timings_ms") or {}

assert first_runtime.get("warm_daemon_active") is True, first_runtime
assert second_runtime.get("warm_daemon_active") is True, second_runtime
assert second_runtime.get("result_source") != "broker_cache_hit", second_runtime
assert second_runtime.get("prefetch_state_source") == "snapshot_metadata", second_runtime
assert second_runtime.get("prefetch_state_cache_hit") is False, second_runtime
assert float(second_chunk_substages.get("discover_source_files_ms") or 0.0) == 0.0, second_chunk_substages
assert int(second_retrieval.get("chunk_cache_reused_files") or 0) >= 1, second_retrieval
assert int(second_retrieval.get("chunk_cache_rebuilt_files") or 0) == 0, second_retrieval
PY

echo "${SECOND_JSON}"
