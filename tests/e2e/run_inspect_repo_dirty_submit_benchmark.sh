#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_DIR="$(mktemp -d /tmp/local-ai-broker-inspect-repo-dirty-submit-bench.XXXXXX)"
source "${SCRIPT_DIR}/smoke_lib.sh"

FILES="${FILES:-300}"
SAMPLES="${SAMPLES:-5}"

cleanup() {
  kill_pid_if_running "${BROKER_PID:-}"
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
export CGO_ENABLED=0

start_broker_server "${REPO_ROOT}"

python3 - <<'PY' "http://${BROKER_LISTEN_ADDR}" "${BASE_DIR}" "${FILES}" "${SAMPLES}"
import json
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

base_url = sys.argv[1].rstrip("/")
base_dir = Path(sys.argv[2]).resolve()
files = int(sys.argv[3])
samples = int(sys.argv[4])

repo = Path(tempfile.mkdtemp(prefix="broker-dirty-submit-repo-"))
subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
src = repo / "src"
src.mkdir()
for i in range(files):
    (src / f"f{i:04d}.py").write_text(f"def f{i:04d}():\n    return {i}\n", encoding="utf-8")
subprocess.run(["git", "add", "."], cwd=repo, check=True, stdout=subprocess.DEVNULL)
subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)

target = src / "f0000.py"
target.write_text("def f0000():\n    return 'dirty-0'\n", encoding="utf-8")


def submit_and_wait(current_query: str):
    body = {
        "task_type": "inspect_repo",
        "input_refs": [{"type": "repo", "uri": repo.as_uri(), "classification": "internal"}],
        "task_params": {"query": current_query, "mode": "evidence"},
        "constraints": {
            "retrieval_token_budget": 16000,
            "evidence_token_budget": 4000,
            "final_pack_token_budget": 2048,
            "synthesis_context_token_budget": 16000,
        },
        "execution_profile": {"backend": "local", "tier": "cpu-rag-indexing"},
        "output_schema": {"name": "repo_inspection_v2"},
    }
    request = urllib.request.Request(
        base_url + "/v1/jobs",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Broker-Actor": "alice",
            "X-Broker-Role": "user",
        },
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as response:
        submit = json.loads(response.read().decode("utf-8"))
    submit_elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    release = submit.get("released_result")
    if not (isinstance(release, dict) and release.get("state") == "succeeded" and isinstance(release.get("result"), dict)):
        job_id = str(submit["job_id"])
        deadline = time.time() + 180
        while time.time() < deadline:
            poll = urllib.request.Request(
                base_url + f"/v1/jobs/{urllib.parse.quote(job_id)}/result?wait_ms=1000&poll_interval_ms=10",
                headers={"X-Broker-Actor": "alice", "X-Broker-Role": "user"},
                method="GET",
            )
            with urllib.request.urlopen(poll, timeout=65) as response:
                release = json.loads(response.read().decode("utf-8"))
            if release.get("state") == "succeeded" and isinstance(release.get("result"), dict):
                break
        else:
            raise SystemExit(f"timed out waiting for {job_id}")
    payload = release["result"]["payload"]
    runtime = payload.get("runtime") or {}
    retrieval = payload.get("retrieval") or {}
    broker_phase_timings = runtime.get("broker_phase_timings_ms") or {}
    setup_timings = retrieval.get("setup_timings_ms") or {}
    return {
        "submit_http_ms": submit_elapsed_ms,
        "hash_input_path_ms": float(broker_phase_timings.get("hash_input_path_ms") or 0.0),
        "cache_key_ms": float(broker_phase_timings.get("cache_key_ms") or 0.0),
        "total_submit_ms": float(broker_phase_timings.get("total_submit_ms") or 0.0),
        "repository_fingerprint_ms": float(setup_timings.get("repository_fingerprint_ms") or 0.0),
        "broker_result_source": str(runtime.get("broker_result_source") or ""),
    }


def summarize(records):
    summary = {}
    for key in ["submit_http_ms", "hash_input_path_ms", "cache_key_ms", "total_submit_ms", "repository_fingerprint_ms"]:
        values = [float(item[key]) for item in records]
        summary[key] = {
            "mean_ms": round(statistics.fmean(values), 3),
            "median_ms": round(statistics.median(values), 3),
            "samples_ms": [round(v, 3) for v in values],
        }
    return summary


warmup = submit_and_wait("trace retry_job warmup")

unchanged_dirty = [submit_and_wait(f"trace retry_job unchanged {i}") for i in range(samples)]

same_dirty_edit = []
for i in range(samples):
    target.write_text(f"def f0000():\n    return 'dirty-{i+1}'\n", encoding="utf-8")
    same_dirty_edit.append(submit_and_wait(f"trace retry_job same-dirty-edit {i}"))

print(json.dumps({
    "repo": str(repo),
    "files": files,
    "samples": samples,
    "warmup": warmup,
    "unchanged_dirty": {
        "summary": summarize(unchanged_dirty),
        "records": unchanged_dirty,
    },
    "same_dirty_edit": {
        "summary": summarize(same_dirty_edit),
        "records": same_dirty_edit,
    },
}, indent=2))
PY
