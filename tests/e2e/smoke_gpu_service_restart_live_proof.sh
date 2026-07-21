#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SUMMARY_JSON="$(python3 "${REPO_ROOT}/tests/acceptance/gpu_service_restart_live_proof.py" "$@")"

SUMMARY_JSON="${SUMMARY_JSON}" python3 - <<'PY'
import json
import os

summary = json.loads(os.environ["SUMMARY_JSON"])
assert summary["ok"] is True, summary
assert summary["checks"]["cluster_submit_succeeded"] is True, summary
assert summary["checks"]["first_index_upsert_succeeded"] is True, summary
assert summary["checks"]["restart_status_ready"] is True, summary
assert summary["checks"]["restart_search_result_preserved"] is True, summary
assert summary["checks"]["persisted_status_sidecar_present"] is True, summary
assert summary["checks"]["persisted_matrix_or_faiss_sidecar_present"] is True, summary
assert summary["checks"]["restart_succeeds_without_full_cache_file"] is True, summary
assert summary["checks"]["restart_avoids_document_reembedding"] is True, summary
assert summary["notes"], summary
PY

echo "${SUMMARY_JSON}"
