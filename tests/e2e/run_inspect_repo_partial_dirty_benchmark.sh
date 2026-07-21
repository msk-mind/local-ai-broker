#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_DIR="$(mktemp -d /tmp/local-ai-broker-inspect-repo-partial-dirty-bench.XXXXXX)"
source "${SCRIPT_DIR}/smoke_lib.sh"

SAMPLES="${SAMPLES:-6}"
MODE="${MODE:-evidence}"
QUERY="${QUERY:-Trace the retry_job service call chain}"

cleanup() {
  kill_pid_if_running "${BROKER_PID:-}"
}
trap cleanup EXIT

INPUT_REPO="${BASE_DIR}/repo"
mkdir -p "${INPUT_REPO}"
printf '%s\n' \
  'def retry_job(job_id):' \
  '    return submit_job(job_id)' \
  '' \
  'def submit_job(job_id):' \
  '    return job_id' > "${INPUT_REPO}/service.py"
printf '%s\n' \
  'def helper():' \
  '    return 1' > "${INPUT_REPO}/helper.py"

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
export BROKER_AUDIT_LOG_PATH="${BASE_DIR}/audit.jsonl"
export BROKER_AUDIT_VERIFY_MODE="warn"
export CGO_ENABLED=0

start_broker_server "${REPO_ROOT}"

python3 - <<'PY' "http://${BROKER_LISTEN_ADDR}" "${INPUT_REPO}" "${QUERY}" "${MODE}" "${SAMPLES}"
import json
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

base_url = sys.argv[1].rstrip("/")
repo = Path(sys.argv[2]).resolve()
query = sys.argv[3]
mode = sys.argv[4]
samples = int(sys.argv[5])


def submit_and_wait(current_query: str):
    body = {
        "task_type": "inspect_repo",
        "input_refs": [{"type": "repo", "uri": repo.as_uri(), "classification": "internal"}],
        "task_params": {"query": current_query, "mode": mode},
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
    with urllib.request.urlopen(request, timeout=30) as response:
        submit = json.loads(response.read().decode("utf-8"))
    job_id = str(submit["job_id"])
    deadline = time.time() + 180
    release = None
    while time.time() < deadline:
        poll = urllib.request.Request(
            base_url + f"/v1/jobs/{urllib.parse.quote(job_id)}/result?wait_ms=1000&poll_interval_ms=10",
            headers={"X-Broker-Actor": "alice", "X-Broker-Role": "user"},
            method="GET",
        )
        with urllib.request.urlopen(poll, timeout=35) as response:
            release = json.loads(response.read().decode("utf-8"))
        state = str(release.get("state") or "")
        if state == "succeeded" and release.get("result") is not None:
            break
        if state in {"failed", "cancelled", "timed_out", "preempted"}:
            raise SystemExit(f"job {job_id} ended in state {state}: {json.dumps(release)}")
    else:
        raise SystemExit(f"timed out waiting for job {job_id}: {json.dumps(release)}")

    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    payload = release["result"]["payload"]
    return elapsed_ms, payload


records = []

# Establish the clean baseline so later runs are genuinely partial-dirty.
baseline_ms, baseline_payload = submit_and_wait(query)
records.append(
    {
        "phase": "clean_baseline",
        "ms": baseline_ms,
        "retrieval": baseline_payload.get("retrieval"),
        "quality": baseline_payload.get("quality"),
    }
)

dirty_file = repo / "service.py"
partial_dirty_ms = []
fingerprint_ms = []

for i in range(samples):
    dirty_file.write_text(
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
    elapsed_ms, payload = submit_and_wait(query)
    retrieval = payload.get("retrieval") or {}
    partial_dirty_ms.append(elapsed_ms)
    fingerprint_ms.append(float((retrieval.get("setup_timings_ms") or {}).get("repository_fingerprint_ms") or 0.0))
    records.append(
        {
            "phase": "partial_dirty",
            "iteration": i,
            "ms": elapsed_ms,
            "retrieval": retrieval,
            "quality": payload.get("quality"),
        }
    )

ordered = sorted(partial_dirty_ms)
summary = {
    "samples": samples,
    "mode": mode,
    "query": query,
    "baseline_ms": baseline_ms,
    "partial_dirty_ms": partial_dirty_ms,
    "partial_dirty_min_ms": min(partial_dirty_ms) if partial_dirty_ms else None,
    "partial_dirty_mean_ms": round(statistics.fmean(partial_dirty_ms), 3) if partial_dirty_ms else None,
    "partial_dirty_median_ms": round(statistics.median(partial_dirty_ms), 3) if partial_dirty_ms else None,
    "partial_dirty_p90_ms": ordered[min(len(ordered) - 1, max(0, int(len(ordered) * 0.9) - 1))] if ordered else None,
    "repository_fingerprint_ms": fingerprint_ms,
    "repository_fingerprint_mean_ms": round(statistics.fmean(fingerprint_ms), 3) if fingerprint_ms else None,
    "records": records,
}
print(json.dumps(summary, indent=2))
PY
