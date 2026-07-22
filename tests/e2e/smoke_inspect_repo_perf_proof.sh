#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_DIR="$(mktemp -d /tmp/local-ai-broker-inspect-repo-perf-proof.XXXXXX)"
source "${SCRIPT_DIR}/smoke_lib.sh"

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

SUMMARY_JSON="$(python3 "${REPO_ROOT}/tests/acceptance/inspect_repo/perf_proof.py" \
  --git-init \
  --command "python3 ${REPO_ROOT}/tests/acceptance/inspect_repo/broker_perf_adapter.py --base-url http://${BROKER_LISTEN_ADDR} --repo {repo} --query {query} --mode {mode}" \
  --timeout-seconds 180 || true)"

SUMMARY_JSON="${SUMMARY_JSON}" python3 - <<'PY'
import json
import os

summary = json.loads(os.environ["SUMMARY_JSON"])
assert summary["mode"] == "command", summary
assert summary["checks"]["cold_query_stage_cache_miss"] is True, summary
assert summary["checks"]["warm_query_stage_cache_hit"] is True, summary
assert summary["checks"]["worker_warm_query_stage_cache_hit"] is True, summary
assert summary["checks"]["partial_dirty_invalidates_query_cache"] is True, summary
assert summary["checks"]["partial_dirty_reuses_one_file"] is True, summary
assert summary["checks"]["partial_dirty_lexical_index_delta_update"] is True, summary
assert summary["checks"]["warm_skips_lexical_search"] is True, summary
assert summary["checks"]["warm_skips_semantic_search"] is True, summary
assert summary["checks"]["warm_skips_rerank"] is True, summary
assert summary["checks"]["worker_warm_skips_lexical_search"] is True, summary
assert summary["checks"]["worker_warm_skips_semantic_search"] is True, summary
assert summary["checks"]["worker_warm_skips_rerank"] is True, summary
assert summary["checks"]["partial_dirty_avoids_full_rebuild"] is True, summary
assert summary["checks"]["partial_dirty_changed_state_reused_or_rebuilt"] is True, summary
assert summary["notes"], summary
assert "lexical-fallback mode" in summary["notes"][0], summary
PY

echo "${SUMMARY_JSON}"
