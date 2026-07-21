#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_DIR="$(mktemp -d /tmp/local-ai-broker-inspect-repo-warm-prefetch.XXXXXX)"

cleanup() {
  if [ -n "${DAEMON_PID:-}" ] && kill -0 "${DAEMON_PID}" 2>/dev/null; then
    kill "${DAEMON_PID}" 2>/dev/null || true
    wait "${DAEMON_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

INPUT_REPO="${BASE_DIR}/repo"
SPOOL_DIR="${BASE_DIR}/spool"
SHARED_CACHE_DIR="${BASE_DIR}/shared-cache"
QUERY="Trace the retry_job service call chain"

mkdir -p "${INPUT_REPO}" "${SPOOL_DIR}/requests" "${SHARED_CACHE_DIR}"
cat > "${INPUT_REPO}/service.py" <<'EOF'
def retry_job(job_id):
    return submit_job(job_id)

def submit_job(job_id):
    return job_id
EOF
cat > "${INPUT_REPO}/mcp.go" <<'EOF'
package mcp

func InspectRepo(query string) string {
	return query
}
EOF

git -C "${INPUT_REPO}" init -q
git -C "${INPUT_REPO}" config user.email test@example.invalid
git -C "${INPUT_REPO}" config user.name Test
git -C "${INPUT_REPO}" add .
git -C "${INPUT_REPO}" commit -qm initial

export BROKER_REPO_INSPECTION_SHARED_CACHE_DIR="${SHARED_CACHE_DIR}"

python3 -S "${REPO_ROOT}/workers/rag-compression/inspect_repo_worker.py" \
  --daemon-spool-dir "${SPOOL_DIR}" \
  --repo-root "${REPO_ROOT}" &
DAEMON_PID=$!

python3 - <<'PY' "${SPOOL_DIR}"
import json
import sys
import time
from pathlib import Path

heartbeat = Path(sys.argv[1]) / "daemon-heartbeat.json"
deadline = time.time() + 30
while time.time() < deadline:
    if heartbeat.exists():
        payload = json.loads(heartbeat.read_text(encoding="utf-8"))
        if payload.get("state") == "running":
            raise SystemExit(0)
    time.sleep(0.05)
raise SystemExit("timed out waiting for warm daemon heartbeat")
PY

python3 - "${BASE_DIR}" "${INPUT_REPO}" "${SHARED_CACHE_DIR}" "${QUERY}" > "${BASE_DIR}/summary.json" <<'PY'
import json
import socket
import sys
import time
from pathlib import Path

base_dir = Path(sys.argv[1])
repo = Path(sys.argv[2]).resolve()
shared_cache = Path(sys.argv[3]).resolve()
query = sys.argv[4]
spool = base_dir / "spool"
socket_path = spool / "daemon.sock"

def write_request(job_name: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    job_spec = {
        "job_id": job_name,
        "task_params": {"query": query, "mode": "evidence"},
        "constraints": {
            "retrieval_token_budget": 16000,
            "evidence_token_budget": 4000,
            "final_pack_token_budget": 2048,
            "synthesis_context_token_budget": 16000,
        },
    }
    input_manifest = {
        "input_refs": [
            {
                "id": "input_0",
                "type": "repo",
                "uri": repo.as_uri(),
                "classification": "internal",
                "content_hash": "git:warm-daemon-prefetch-proof",
            }
        ]
    }
    execution_plan = {
        "runtime_backend": "local",
        "execution_profile": {"backend": "local"},
        "repo_inspection_cache_path": str(shared_cache),
        "repo_inspection_shared_cache_path": str(shared_cache),
    }
    (output_dir / "job_spec.json").write_text(json.dumps(job_spec), encoding="utf-8")
    (output_dir / "input_manifest.json").write_text(json.dumps(input_manifest), encoding="utf-8")
    (output_dir / "execution_plan.json").write_text(json.dumps(execution_plan), encoding="utf-8")
    request = {
        "job_id": job_name,
        "job_spec_path": str(output_dir / "job_spec.json"),
        "execution_plan_path": str(output_dir / "execution_plan.json"),
        "input_manifest_path": str(output_dir / "input_manifest.json"),
        "output_dir": str(output_dir),
        "heartbeat_path": str(output_dir / "heartbeat.json"),
    }
    request_path = spool / "requests" / f"{job_name}.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        client.connect(str(socket_path))
        client.send(job_name.encode("utf-8"))
    finally:
        client.close()

def wait_result(output_dir: Path):
    result_path = output_dir / "result.json"
    deadline = time.time() + 60
    while time.time() < deadline:
        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        time.sleep(0.05)
    raise SystemExit(f"timed out waiting for {result_path}")

first_dir = base_dir / "job-one"
second_dir = base_dir / "job-two"

write_request("job_one", first_dir)
first = wait_result(first_dir)

write_request("job_two", second_dir)
second = wait_result(second_dir)

print(json.dumps({"first": first, "second": second}))
PY

SUMMARY_JSON="$(cat "${BASE_DIR}/summary.json")"

SUMMARY_JSON="${SUMMARY_JSON}" BASE_DIR="${BASE_DIR}" python3 - <<'PY'
import json
import os
from pathlib import Path

summary = json.loads(os.environ["SUMMARY_JSON"])
first_payload = summary["first"]["payload"]
second_payload = summary["second"]["payload"]

first_runtime = first_payload.get("runtime") or {}
second_runtime = second_payload.get("runtime") or {}

assert first_runtime.get("local_backend_mode") == "warm_daemon", first_runtime
assert second_runtime.get("local_backend_mode") == "warm_daemon", second_runtime
assert first_runtime.get("warm_daemon_active") is True, first_runtime
assert second_runtime.get("warm_daemon_active") is True, second_runtime

assert first_runtime.get("prefetch_state_source") == "fresh", first_runtime
assert first_runtime.get("prefetch_state_cache_hit") is False, first_runtime

assert second_runtime.get("prefetch_state_source") == "early_process_cache", second_runtime
assert second_runtime.get("prefetch_state_cache_hit") is True, second_runtime

assert second_runtime.get("result_source") != "broker_cache_hit", second_runtime
assert (second_payload.get("retrieval") or {}).get("query_stage_cache_hit") is True, second_payload

base_dir = Path(os.environ["BASE_DIR"])
assert (base_dir / "job-one" / "result.json").exists()
assert (base_dir / "job-two" / "result.json").exists()
assert str(base_dir / "job-one") != str(base_dir / "job-two")
PY

cat "${BASE_DIR}/summary.json"
