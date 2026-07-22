#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_DIR="$(mktemp -d /tmp/local-ai-broker-inspect-repo-answer-cache-hit.XXXXXX)"
source "${SCRIPT_DIR}/smoke_lib.sh"

cleanup() {
  kill_pid_if_running "${BROKER_PID:-}"
}
trap cleanup EXIT

QUERY="trace routing"
MODE="answer"
INPUT_REPO="${BASE_DIR}/repo"
mkdir -p "${INPUT_REPO}"
printf '# demo\n' > "${INPUT_REPO}/README.md"

CACHE_KEY_JSON="$(
  CACHE_KEY_HELPER_REPO="${INPUT_REPO}" \
  CACHE_KEY_HELPER_QUERY="${QUERY}" \
  CACHE_KEY_HELPER_MODE="${MODE}" \
  CACHE_KEY_HELPER_CLASSIFICATION="internal" \
  CACHE_KEY_HELPER_TIER="cpu-rag-indexing" \
  CACHE_KEY_HELPER_RUNTIME="deterministic" \
  go run "${REPO_ROOT}/tests/acceptance/inspect_repo/cache_key_helper.go"
)"

python3 - <<'PY' "${BASE_DIR}" "${INPUT_REPO}" "${QUERY}" "${MODE}" "${CACHE_KEY_JSON}"
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

base_dir = Path(sys.argv[1])
input_repo = Path(sys.argv[2])
query = sys.argv[3]
mode = sys.argv[4]
cache_info = json.loads(sys.argv[5])
now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

job = {
    "job_id": "job_seed_answer_ready",
    "task_type": "inspect_repo",
    "state": "succeeded",
    "submitted_by": "alice",
    "request": {
        "task_type": "inspect_repo",
        "input_refs": [
            {
                "type": "repo",
                "uri": input_repo.resolve().as_uri(),
                "classification": "internal",
            }
        ],
        "task_params": {"query": query, "mode": mode},
        "execution_profile": {"tier": "cpu-rag-indexing", "runtime": "deterministic"},
        "output_schema": {"name": "repo_inspection_v2"},
    },
    "result": {
        "schema_name": "repo_inspection_v2",
        "schema_version": "2.0.0",
        "payload": {
            "mode": mode,
            "query": query,
            "answer": "done",
            "findings": [{"summary": "done", "evidence_refs": ["ev_1"]}],
            "evidence": [
                {
                    "id": "ev_1",
                    "path": "README.md",
                    "source_refs": [{"path": "README.md", "line_start": 1, "line_end": 1}],
                }
            ],
            "quality": {
                "result": "answer_ready",
                "retrieval": "gpu",
                "reranking": "gpu",
                "synthesis": "gpu",
                "answer_ready": True,
            },
            "warnings": [],
            "provenance": {"index_fingerprint": "sha256:test"},
            "retrieval": {
                "lexical_candidates": 1,
                "semantic_candidates": 1,
                "reranked_candidates": 1,
            },
            "runtime": {
                "attempts": [
                    {"operation": "semantic_retrieval", "status": "succeeded"},
                    {"operation": "rerank", "status": "succeeded"},
                    {"operation": "synthesis", "status": "succeeded"},
                ]
            },
        },
    },
    "artifacts": [{"artifact_id": "artifact_1", "artifact_type": "evidence_pack"}],
    "created_at": now,
    "updated_at": now,
    "submitted_at": now,
    "started_at": now,
    "completed_at": now,
    "backend_kind": "local",
    "backend_state": "COMPLETED",
    "cache_key": cache_info["cache_key"],
}

(base_dir / "jobs.json").write_text(json.dumps({job["job_id"]: job}) + "\n", encoding="utf-8")
PY

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

RESPONSE_JSON="$(
python3 - <<'PY' "http://${BROKER_LISTEN_ADDR}" "${INPUT_REPO}" "${QUERY}" "${MODE}"
import json
import sys
import urllib.request
from pathlib import Path

base_url = sys.argv[1].rstrip("/")
repo = Path(sys.argv[2]).resolve()
query = sys.argv[3]
mode = sys.argv[4]

body = {
    "task_type": "inspect_repo",
    "input_refs": [{"type": "repo", "uri": repo.as_uri(), "classification": "internal"}],
    "task_params": {"query": query, "mode": mode},
    "execution_profile": {"tier": "cpu-rag-indexing", "runtime": "deterministic"},
    "output_schema": {"name": "repo_inspection_v2"},
}
req = urllib.request.Request(
    base_url + "/v1/jobs",
    data=json.dumps(body).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "X-Broker-Actor": "alice",
        "X-Broker-Role": "user",
    },
    method="POST",
)
with urllib.request.urlopen(req, timeout=30) as response:
    print(response.read().decode("utf-8"))
PY
)"

RESPONSE_JSON="${RESPONSE_JSON}" python3 - <<'PY'
import json
import os

response = json.loads(os.environ["RESPONSE_JSON"])
assert response["cache"]["status"] == "hit", response
released = response.get("released_result")
assert isinstance(released, dict), response
assert released["state"] == "succeeded", response
result = released["result"]
assert result["schema_name"] == "repo_inspection_v2", response
payload = result["payload"]
assert payload["quality"]["result"] == "answer_ready", response
assert payload["quality"]["retrieval"] == "gpu", response
assert payload["quality"]["reranking"] == "gpu", response
assert payload["quality"]["synthesis"] == "gpu", response
assert payload["answer"] == "done", response
assert payload["retrieval"]["query_stage_cache_hit"] is True, response
print(json.dumps(response))
PY
