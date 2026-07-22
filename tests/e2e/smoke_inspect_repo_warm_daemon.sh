#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_DIR="$(mktemp -d /tmp/local-ai-broker-inspect-repo-warm-daemon.XXXXXX)"
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

RESULT_JSON="$(python3 "${REPO_ROOT}/tests/acceptance/inspect_repo/broker_perf_adapter.py" \
  --base-url "http://${BROKER_LISTEN_ADDR}" \
  --repo "${INPUT_REPO}" \
  --query "Trace the retry_job service call chain" \
  --mode evidence)"

SECOND_JSON="$(python3 "${REPO_ROOT}/tests/acceptance/inspect_repo/broker_perf_adapter.py" \
  --base-url "http://${BROKER_LISTEN_ADDR}" \
  --repo "${INPUT_REPO}" \
  --query "Trace the retry_job service call chain" \
  --mode evidence)"

RESULT_JSON="${RESULT_JSON}" SECOND_JSON="${SECOND_JSON}" BASE_DIR="${BASE_DIR}" python3 - <<'PY'
import json
import os
from pathlib import Path

first = json.loads(os.environ["RESULT_JSON"])
second = json.loads(os.environ["SECOND_JSON"])
payload = first["result"]["payload"]
runtime = payload.get("runtime") or {}
assert runtime.get("local_backend_mode") == "warm_daemon", runtime
assert runtime.get("warm_daemon_active") is True, runtime

base_dir = Path(os.environ["BASE_DIR"])
run_root = base_dir / "runs"
daemon_pid = run_root / ".inspect-repo-warm" / "daemon.pid"
assert daemon_pid.exists(), daemon_pid

job_dirs = [path for path in run_root.iterdir() if path.is_dir() and path.name.startswith("job_")]
assert len(job_dirs) == 1, [p.name for p in job_dirs]

first_payload = (first.get("result") or {}).get("payload") or {}
second_payload = (second.get("result") or {}).get("payload") or {}
assert (first_payload.get("runtime") or {}).get("warm_daemon_active") is True, first_payload
assert (second_payload.get("runtime") or {}).get("warm_daemon_active") is True, second_payload
PY

echo "${SECOND_JSON}"
