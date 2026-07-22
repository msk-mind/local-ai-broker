#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_DIR="$(mktemp -d /tmp/local-ai-broker-path-breakdown.XXXXXX)"
source "${SCRIPT_DIR}/smoke_lib.sh"

cleanup() {
  kill_pid_if_running "${BROKER_PID:-}"
  rm -rf "${BASE_DIR}"
}
trap cleanup EXIT

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

start_broker_server "${REPO_ROOT}"

python3 - <<'PY' "${REPO_ROOT}" "${BROKER_LISTEN_ADDR}" "${INPUT_REPO}"
import json
import subprocess
import sys
import time

repo_root, broker_addr, input_repo = sys.argv[1:4]
adapter = f"{repo_root}/tests/acceptance/inspect_repo/broker_perf_adapter.py"
base_url = f"http://{broker_addr}"

def run(query: str) -> dict:
    started = time.perf_counter()
    output = subprocess.check_output(
        [
            "python3",
            adapter,
            "--base-url",
            base_url,
            "--repo",
            input_repo,
            "--query",
            query,
            "--mode",
            "evidence",
        ],
        text=True,
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    release = json.loads(output)
    payload = (release.get("result") or {}).get("payload") or {}
    runtime = payload.get("runtime") or {}
    retrieval = payload.get("retrieval") or {}
    return {
        "query": query,
        "wall_ms": round(wall_ms, 3),
        "broker_result_source": runtime.get("broker_result_source"),
        "broker_phase_timings_ms": runtime.get("broker_phase_timings_ms"),
        "worker_total_ms": (runtime.get("worker_phase_timings_ms") or {}).get("total"),
        "query_stage_cache_hit": retrieval.get("query_stage_cache_hit"),
    }

rows = [
    {"path": "cold_preferred_inline", "data": run("Trace the retry_job service call chain")},
    {"path": "same_query_followup", "data": run("Trace the retry_job service call chain")},
    {"path": "new_query_inline_release", "data": run("Trace how helper reaches retry_job")},
    {"path": "new_query_followup", "data": run("Trace how helper reaches retry_job")},
]

print(json.dumps(rows, indent=2))
PY
