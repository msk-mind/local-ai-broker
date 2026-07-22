#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_DIR="$(mktemp -d /tmp/local-ai-broker-inspect-repo-discovery-bench.XXXXXX)"
source "${SCRIPT_DIR}/smoke_lib.sh"

SAMPLES="${SAMPLES:-6}"
MODE="${MODE:-evidence}"
QUERY="${QUERY:-Trace the retry_job service call chain}"

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

python3 - <<'PY' "http://${BROKER_LISTEN_ADDR}" "${BASE_DIR}" "${QUERY}" "${MODE}" "${SAMPLES}"
import json
import statistics
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

base_url = sys.argv[1].rstrip("/")
base_dir = Path(sys.argv[2]).resolve()
query = sys.argv[3]
mode = sys.argv[4]
samples = int(sys.argv[5])


def submit_and_wait(repo: Path, current_query: str):
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
    submit_started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=30) as response:
        submit = json.loads(response.read().decode("utf-8"))
    submit_elapsed_ms = round((time.perf_counter() - submit_started) * 1000.0, 3)
    job_id = str(submit["job_id"])
    release = None
    released_result = submit.get("released_result")
    if isinstance(released_result, dict):
        state = str(released_result.get("state") or "")
        if state == "succeeded" and released_result.get("result") is not None:
            release = released_result
    deadline = time.time() + 180
    wait_started = time.perf_counter()
    if release is None:
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
    wait_elapsed_ms = round((time.perf_counter() - wait_started) * 1000.0, 3)

    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    payload = release["result"]["payload"]
    heartbeat_payload = {}
    heartbeat_path = base_dir / "runs" / job_id / "heartbeat.json"
    try:
        if heartbeat_path.exists():
            heartbeat_payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    except Exception:
        heartbeat_payload = {}
    retrieval = payload.get("retrieval") or {}
    runtime = payload.get("runtime") or {}
    lifecycle = runtime.get("broker_lifecycle") or {}
    broker_phase_timings = runtime.get("broker_phase_timings_ms") or {}
    broker_result_source = runtime.get("broker_result_source") or ""
    heartbeat_metrics = heartbeat_payload.get("metrics") or {}
    worker_phase_timings = runtime.get("worker_phase_timings_ms") or {}
    chunk_substages = retrieval.get("chunk_build_substage_timings_ms") or {}
    stage_timings = retrieval.get("stage_timings_ms") or {}
    tail_timings = retrieval.get("tail_timings_ms") or {}
    enqueue_to_claim_ms = 0.0
    enqueue_to_written_ms = 0.0
    written_to_wakeup_ms = 0.0
    wakeup_to_claim_ms = 0.0
    claim_to_result_write_ms = 0.0
    result_write_to_client_visible_ms = 0.0
    result_write_started_to_completed_ms = 0.0
    result_write_completed_to_notify_ms = 0.0
    result_write_completed_to_client_visible_ms = 0.0
    broker_result_wait_ms = 0.0
    broker_result_initial_fetch_to_release_ms = 0.0
    broker_result_release_to_response_ms = 0.0
    broker_submit_response_ready_to_client_visible_ms = 0.0
    enqueued_unix_ns = lifecycle.get("broker_request_enqueued_unix_ns")
    written_unix_ns = lifecycle.get("broker_request_written_unix_ns")
    wakeup_unix_ns = lifecycle.get("worker_wakeup_received_unix_ns")
    wakeup_monotonic_ns = lifecycle.get("worker_wakeup_received_monotonic_ns")
    claimed_unix_ns = lifecycle.get("worker_claimed_unix_ns")
    claimed_monotonic_ns = lifecycle.get("worker_claimed_monotonic_ns")
    result_write_started_unix_ns = lifecycle.get("worker_result_write_started_unix_ns") or heartbeat_metrics.get("worker_result_write_started_unix_ns")
    result_write_started_monotonic_ns = lifecycle.get("worker_result_write_started_monotonic_ns") or heartbeat_metrics.get("worker_result_write_started_monotonic_ns")
    result_written_unix_ns = lifecycle.get("worker_result_written_unix_ns") or heartbeat_metrics.get("worker_result_written_unix_ns")
    result_written_monotonic_ns = lifecycle.get("worker_result_written_monotonic_ns") or heartbeat_metrics.get("worker_result_written_monotonic_ns")
    completion_notified_unix_ns = lifecycle.get("worker_completion_notified_unix_ns") or heartbeat_metrics.get("worker_completion_notified_unix_ns")
    completion_notified_monotonic_ns = lifecycle.get("worker_completion_notified_monotonic_ns") or heartbeat_metrics.get("worker_completion_notified_monotonic_ns")
    broker_result_request_started_unix_ns = lifecycle.get("broker_result_request_started_unix_ns")
    broker_result_initial_fetch_unix_ns = lifecycle.get("broker_result_initial_fetch_unix_ns")
    broker_result_release_observed_unix_ns = lifecycle.get("broker_result_release_observed_unix_ns")
    broker_result_response_ready_unix_ns = lifecycle.get("broker_result_response_ready_unix_ns")
    broker_submit_response_ready_unix_ns = lifecycle.get("broker_submit_response_ready_unix_ns")
    try:
        if enqueued_unix_ns is not None and claimed_unix_ns is not None:
            enqueue_to_claim_ms = max(0.0, (int(claimed_unix_ns) - int(enqueued_unix_ns)) / 1_000_000.0)
    except (TypeError, ValueError):
        enqueue_to_claim_ms = 0.0
    try:
        if enqueued_unix_ns is not None and written_unix_ns is not None:
            enqueue_to_written_ms = max(0.0, (int(written_unix_ns) - int(enqueued_unix_ns)) / 1_000_000.0)
    except (TypeError, ValueError):
        enqueue_to_written_ms = 0.0
    try:
        if written_unix_ns is not None and wakeup_unix_ns is not None:
            written_to_wakeup_ms = max(0.0, (int(wakeup_unix_ns) - int(written_unix_ns)) / 1_000_000.0)
    except (TypeError, ValueError):
        written_to_wakeup_ms = 0.0
    try:
        if wakeup_monotonic_ns is not None and claimed_monotonic_ns is not None:
            wakeup_to_claim_ms = max(0.0, (int(claimed_monotonic_ns) - int(wakeup_monotonic_ns)) / 1_000_000.0)
    except (TypeError, ValueError):
        wakeup_to_claim_ms = 0.0
    try:
        if claimed_monotonic_ns is not None and result_written_monotonic_ns is not None:
            claim_to_result_write_ms = max(0.0, (int(result_written_monotonic_ns) - int(claimed_monotonic_ns)) / 1_000_000.0)
    except (TypeError, ValueError):
        claim_to_result_write_ms = 0.0
    try:
        if result_written_unix_ns is not None:
            client_visible_unix_ns = int(time.time_ns())
            result_write_to_client_visible_ms = max(0.0, (client_visible_unix_ns - int(result_written_unix_ns)) / 1_000_000.0)
    except (TypeError, ValueError):
        result_write_to_client_visible_ms = 0.0
    try:
        if result_write_started_monotonic_ns is not None and result_written_monotonic_ns is not None:
            result_write_started_to_completed_ms = max(0.0, (int(result_written_monotonic_ns) - int(result_write_started_monotonic_ns)) / 1_000_000.0)
    except (TypeError, ValueError):
        result_write_started_to_completed_ms = 0.0
    try:
        if result_written_monotonic_ns is not None and completion_notified_monotonic_ns is not None:
            result_write_completed_to_notify_ms = max(0.0, (int(completion_notified_monotonic_ns) - int(result_written_monotonic_ns)) / 1_000_000.0)
    except (TypeError, ValueError):
        result_write_completed_to_notify_ms = 0.0
    try:
        if result_written_unix_ns is not None:
            client_visible_unix_ns = int(time.time_ns())
            result_write_completed_to_client_visible_ms = max(0.0, (client_visible_unix_ns - int(result_written_unix_ns)) / 1_000_000.0)
    except (TypeError, ValueError):
        result_write_completed_to_client_visible_ms = 0.0
    try:
        if broker_result_request_started_unix_ns is not None and broker_result_response_ready_unix_ns is not None:
            broker_result_wait_ms = max(0.0, (int(broker_result_response_ready_unix_ns) - int(broker_result_request_started_unix_ns)) / 1_000_000.0)
    except (TypeError, ValueError):
        broker_result_wait_ms = 0.0
    try:
        if broker_result_initial_fetch_unix_ns is not None and broker_result_release_observed_unix_ns is not None:
            broker_result_initial_fetch_to_release_ms = max(0.0, (int(broker_result_release_observed_unix_ns) - int(broker_result_initial_fetch_unix_ns)) / 1_000_000.0)
    except (TypeError, ValueError):
        broker_result_initial_fetch_to_release_ms = 0.0
    try:
        if broker_result_release_observed_unix_ns is not None and broker_result_response_ready_unix_ns is not None:
            broker_result_release_to_response_ms = max(0.0, (int(broker_result_response_ready_unix_ns) - int(broker_result_release_observed_unix_ns)) / 1_000_000.0)
    except (TypeError, ValueError):
        broker_result_release_to_response_ms = 0.0
    try:
        if broker_submit_response_ready_unix_ns is not None:
            broker_submit_response_ready_to_client_visible_ms = max(0.0, (client_visible_unix_ns - int(broker_submit_response_ready_unix_ns)) / 1_000_000.0)
    except (TypeError, ValueError, UnboundLocalError):
        broker_submit_response_ready_to_client_visible_ms = 0.0
    return {
        "wall_ms": elapsed_ms,
        "submit_http_ms": submit_elapsed_ms,
        "result_wait_http_ms": wait_elapsed_ms,
        "payload": payload,
        "worker_phase_total_ms": float(worker_phase_timings.get("total") or 0.0),
        "broker_control_overhead_ms": max(0.0, elapsed_ms - float(worker_phase_timings.get("total") or 0.0)),
        "enqueue_to_claim_ms": round(enqueue_to_claim_ms, 3),
        "enqueue_to_written_ms": round(enqueue_to_written_ms, 3),
        "written_to_wakeup_ms": round(written_to_wakeup_ms, 3),
        "wakeup_to_claim_ms": round(wakeup_to_claim_ms, 3),
        "claim_to_result_write_ms": round(claim_to_result_write_ms, 3),
        "result_write_to_client_visible_ms": round(result_write_to_client_visible_ms, 3),
        "result_write_started_to_completed_ms": round(result_write_started_to_completed_ms, 3),
        "result_write_completed_to_notify_ms": round(result_write_completed_to_notify_ms, 3),
        "result_write_completed_to_client_visible_ms": round(result_write_completed_to_client_visible_ms, 3),
        "broker_result_wait_ms": round(broker_result_wait_ms, 3),
        "broker_result_initial_fetch_to_release_ms": round(broker_result_initial_fetch_to_release_ms, 3),
        "broker_result_release_to_response_ms": round(broker_result_release_to_response_ms, 3),
        "broker_submit_response_ready_to_client_visible_ms": round(broker_submit_response_ready_to_client_visible_ms, 3),
        "broker_result_source": str(broker_result_source),
        "cache_key_ms": float(broker_phase_timings.get("cache_key_ms") or 0.0),
        "resolve_input_path_ms": float(broker_phase_timings.get("resolve_input_path_ms") or 0.0),
        "hash_input_path_ms": float(broker_phase_timings.get("hash_input_path_ms") or 0.0),
        "hash_input_file_ms": float(broker_phase_timings.get("hash_input_file_ms") or 0.0),
        "serialize_cache_key_payload_ms": float(broker_phase_timings.get("serialize_cache_key_payload_ms") or 0.0),
        "hash_cache_key_payload_ms": float(broker_phase_timings.get("hash_cache_key_payload_ms") or 0.0),
        "build_inline_bundle_ms": float(broker_phase_timings.get("build_inline_bundle_ms") or 0.0),
        "stage_bundle_ms": float(broker_phase_timings.get("stage_bundle_ms") or 0.0),
        "backend_submit_ms": float(broker_phase_timings.get("backend_submit_ms") or 0.0),
        "store_create_job_ms": float(broker_phase_timings.get("store_create_job_ms") or 0.0),
        "total_submit_ms": float(broker_phase_timings.get("total_submit_ms") or 0.0),
        "inline_release_initial_probe_ms": float(broker_phase_timings.get("inline_release_initial_probe_ms") or 0.0),
        "inline_release_waiter_wait_ms": float(broker_phase_timings.get("inline_release_waiter_wait_ms") or 0.0),
        "inline_release_post_wait_release_build_ms": float(broker_phase_timings.get("inline_release_post_wait_release_build_ms") or 0.0),
        "inline_release_total_ms": float(broker_phase_timings.get("inline_release_total_ms") or 0.0),
        "worker_phase_process_bootstrap_ms": float(worker_phase_timings.get("process_bootstrap") or 0.0),
        "worker_phase_load_job_inputs_ms": float(worker_phase_timings.get("load_job_inputs") or 0.0),
        "worker_phase_import_prefetch_helpers_ms": float(worker_phase_timings.get("import_prefetch_helpers") or 0.0),
        "worker_phase_prefetch_cache_context_ms": float(worker_phase_timings.get("prefetch_cache_context") or 0.0),
        "worker_phase_import_run_inspection_ms": float(worker_phase_timings.get("import_run_inspection") or 0.0),
        "worker_phase_run_inspection_ms": float(worker_phase_timings.get("run_inspection") or 0.0),
        "worker_phase_write_artifacts_ms": float(worker_phase_timings.get("write_artifacts") or 0.0),
        "worker_phase_finalize_ms": float(worker_phase_timings.get("finalize") or 0.0),
        "manifest_load_ms": float(chunk_substages.get("manifest_load_ms") or 0.0),
        "shared_manifest_load_ms": float(chunk_substages.get("shared_manifest_load_ms") or 0.0),
        "shared_state_manifest_load_ms": float(chunk_substages.get("shared_state_manifest_load_ms") or 0.0),
        "snapshot_probe_ms": float(chunk_substages.get("snapshot_probe_ms") or 0.0),
        "previous_snapshot_load_ms": float(chunk_substages.get("previous_snapshot_load_ms") or 0.0),
        "discover_source_files_ms": float(chunk_substages.get("discover_source_files_ms") or 0.0),
        "shared_state_lookup_ms": float(chunk_substages.get("shared_state_lookup_ms") or 0.0),
        "file_chunk_bundle_load_ms": float(chunk_substages.get("file_chunk_bundle_load_ms") or 0.0),
        "file_chunk_bundle_write_ms": float(chunk_substages.get("file_chunk_bundle_write_ms") or 0.0),
        "source_read_ms": float(chunk_substages.get("source_read_ms") or 0.0),
        "chunk_build_ms": float(chunk_substages.get("chunk_build_ms") or 0.0),
        "chunk_content_hash_ms": float(chunk_substages.get("chunk_content_hash_ms") or 0.0),
        "chunk_identity_hash_ms": float(chunk_substages.get("chunk_identity_hash_ms") or 0.0),
        "chunk_token_estimate_ms": float(chunk_substages.get("chunk_token_estimate_ms") or 0.0),
        "manifest_records_ms": float(chunk_substages.get("manifest_records_ms") or 0.0),
        "semantic_signature_ms": float(chunk_substages.get("semantic_signature_ms") or 0.0),
        "symbol_marker_ms": float(chunk_substages.get("symbol_marker_ms") or 0.0),
        "symbol_marker_write_ms": float(chunk_substages.get("symbol_marker_write_ms") or 0.0),
        "working_manifest_write_ms": float(chunk_substages.get("working_manifest_write_ms") or 0.0),
        "shared_state_manifest_flush_ms": float(chunk_substages.get("shared_state_manifest_flush_ms") or 0.0),
        "snapshot_write_ms": float(chunk_substages.get("snapshot_write_ms") or 0.0),
        "git_file_signatures_ms": float(chunk_substages.get("git_file_signatures_ms") or 0.0),
        "git_dirty_manifest_keys_ms": float(chunk_substages.get("git_dirty_manifest_keys_ms") or 0.0),
        "build_syntax_chunks_ms": float(stage_timings.get("build_syntax_chunks_ms") or 0.0),
        "ensure_lexical_index_ms": float(stage_timings.get("ensure_lexical_index_ms") or 0.0),
        "query_stage_cache_ms": float(stage_timings.get("query_stage_cache_ms") or 0.0),
        "write_query_stage_cache_ms": float(stage_timings.get("write_query_stage_cache_ms") or 0.0),
        "artifact_payloads_ms": float(tail_timings.get("artifact_payloads_ms") or 0.0),
        "chunk_cache_reused_files": int(retrieval.get("chunk_cache_reused_files") or 0),
        "chunk_cache_rebuilt_files": int(retrieval.get("chunk_cache_rebuilt_files") or 0),
    }


def stage_repo(root: Path, *, variant: int):
    root.mkdir(parents=True, exist_ok=True)
    (root / "service.py").write_text(
        "\n".join(
            [
                "def retry_job(job_id):",
                f"    seed = {variant}",
                "    return submit_job(job_id + seed)",
                "",
                "def submit_job(job_id):",
                "    return job_id",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "helper.py").write_text(
        "\n".join(
            [
                "def helper():",
                f"    return {variant}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def git_init_repo(root: Path):
    import subprocess

    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)


def summarize(records, field):
    values = [float(record[field]) for record in records]
    ordered = sorted(values)
    return {
        "min_ms": round(min(values), 3) if values else None,
        "mean_ms": round(statistics.fmean(values), 3) if values else None,
        "median_ms": round(statistics.median(values), 3) if values else None,
        "p90_ms": round(ordered[min(len(ordered) - 1, max(0, int(len(ordered) * 0.9) - 1))], 3) if ordered else None,
        "samples_ms": [round(value, 3) for value in values],
    }


cold_records = []
for i in range(samples):
    repo = base_dir / f"cold-repo-{i}"
    stage_repo(repo, variant=i)
    git_init_repo(repo)
    result = submit_and_wait(repo, query)
    cold_records.append(
        {
            "iteration": i,
            "wall_ms": result["wall_ms"],
            "submit_http_ms": result["submit_http_ms"],
            "result_wait_http_ms": result["result_wait_http_ms"],
            "worker_phase_total_ms": result["worker_phase_total_ms"],
            "broker_control_overhead_ms": result["broker_control_overhead_ms"],
            "enqueue_to_claim_ms": result["enqueue_to_claim_ms"],
            "enqueue_to_written_ms": result["enqueue_to_written_ms"],
            "written_to_wakeup_ms": result["written_to_wakeup_ms"],
            "wakeup_to_claim_ms": result["wakeup_to_claim_ms"],
            "claim_to_result_write_ms": result["claim_to_result_write_ms"],
            "result_write_to_client_visible_ms": result["result_write_to_client_visible_ms"],
            "result_write_started_to_completed_ms": result["result_write_started_to_completed_ms"],
            "result_write_completed_to_notify_ms": result["result_write_completed_to_notify_ms"],
            "result_write_completed_to_client_visible_ms": result["result_write_completed_to_client_visible_ms"],
            "broker_result_wait_ms": result["broker_result_wait_ms"],
            "broker_result_initial_fetch_to_release_ms": result["broker_result_initial_fetch_to_release_ms"],
            "broker_result_release_to_response_ms": result["broker_result_release_to_response_ms"],
            "broker_submit_response_ready_to_client_visible_ms": result["broker_submit_response_ready_to_client_visible_ms"],
            "broker_result_source": result["broker_result_source"],
            "cache_key_ms": result["cache_key_ms"],
            "resolve_input_path_ms": result["resolve_input_path_ms"],
            "hash_input_path_ms": result["hash_input_path_ms"],
            "hash_input_file_ms": result["hash_input_file_ms"],
            "serialize_cache_key_payload_ms": result["serialize_cache_key_payload_ms"],
            "hash_cache_key_payload_ms": result["hash_cache_key_payload_ms"],
            "build_inline_bundle_ms": result["build_inline_bundle_ms"],
            "stage_bundle_ms": result["stage_bundle_ms"],
            "backend_submit_ms": result["backend_submit_ms"],
            "store_create_job_ms": result["store_create_job_ms"],
            "total_submit_ms": result["total_submit_ms"],
            "inline_release_initial_probe_ms": result["inline_release_initial_probe_ms"],
            "inline_release_waiter_wait_ms": result["inline_release_waiter_wait_ms"],
            "inline_release_post_wait_release_build_ms": result["inline_release_post_wait_release_build_ms"],
            "inline_release_total_ms": result["inline_release_total_ms"],
            "worker_phase_process_bootstrap_ms": result["worker_phase_process_bootstrap_ms"],
            "worker_phase_load_job_inputs_ms": result["worker_phase_load_job_inputs_ms"],
            "worker_phase_import_prefetch_helpers_ms": result["worker_phase_import_prefetch_helpers_ms"],
            "worker_phase_prefetch_cache_context_ms": result["worker_phase_prefetch_cache_context_ms"],
            "worker_phase_import_run_inspection_ms": result["worker_phase_import_run_inspection_ms"],
            "worker_phase_run_inspection_ms": result["worker_phase_run_inspection_ms"],
            "worker_phase_write_artifacts_ms": result["worker_phase_write_artifacts_ms"],
            "worker_phase_finalize_ms": result["worker_phase_finalize_ms"],
            "manifest_load_ms": result["manifest_load_ms"],
            "shared_manifest_load_ms": result["shared_manifest_load_ms"],
            "shared_state_manifest_load_ms": result["shared_state_manifest_load_ms"],
            "snapshot_probe_ms": result["snapshot_probe_ms"],
            "discover_source_files_ms": result["discover_source_files_ms"],
            "shared_state_lookup_ms": result["shared_state_lookup_ms"],
            "file_chunk_bundle_load_ms": result["file_chunk_bundle_load_ms"],
            "file_chunk_bundle_write_ms": result["file_chunk_bundle_write_ms"],
            "source_read_ms": result["source_read_ms"],
            "chunk_build_ms": result["chunk_build_ms"],
            "chunk_content_hash_ms": result["chunk_content_hash_ms"],
            "chunk_identity_hash_ms": result["chunk_identity_hash_ms"],
            "chunk_token_estimate_ms": result["chunk_token_estimate_ms"],
            "manifest_records_ms": result["manifest_records_ms"],
            "semantic_signature_ms": result["semantic_signature_ms"],
            "symbol_marker_ms": result["symbol_marker_ms"],
            "symbol_marker_write_ms": result["symbol_marker_write_ms"],
            "working_manifest_write_ms": result["working_manifest_write_ms"],
            "shared_state_manifest_flush_ms": result["shared_state_manifest_flush_ms"],
            "snapshot_write_ms": result["snapshot_write_ms"],
            "git_file_signatures_ms": result["git_file_signatures_ms"],
            "build_syntax_chunks_ms": result["build_syntax_chunks_ms"],
            "ensure_lexical_index_ms": result["ensure_lexical_index_ms"],
            "query_stage_cache_ms": result["query_stage_cache_ms"],
            "write_query_stage_cache_ms": result["write_query_stage_cache_ms"],
            "artifact_payloads_ms": result["artifact_payloads_ms"],
            "chunk_cache_reused_files": result["chunk_cache_reused_files"],
            "chunk_cache_rebuilt_files": result["chunk_cache_rebuilt_files"],
        }
    )

partial_repo = base_dir / "partial-dirty-repo"
stage_repo(partial_repo, variant=100)
git_init_repo(partial_repo)
baseline = submit_and_wait(partial_repo, query)

partial_dirty_records = []
for i in range(samples):
    (partial_repo / "service.py").write_text(
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
    result = submit_and_wait(partial_repo, query)
    partial_dirty_records.append(
        {
            "iteration": i,
            "wall_ms": result["wall_ms"],
            "submit_http_ms": result["submit_http_ms"],
            "result_wait_http_ms": result["result_wait_http_ms"],
            "worker_phase_total_ms": result["worker_phase_total_ms"],
            "broker_control_overhead_ms": result["broker_control_overhead_ms"],
            "enqueue_to_claim_ms": result["enqueue_to_claim_ms"],
            "enqueue_to_written_ms": result["enqueue_to_written_ms"],
            "written_to_wakeup_ms": result["written_to_wakeup_ms"],
            "wakeup_to_claim_ms": result["wakeup_to_claim_ms"],
            "claim_to_result_write_ms": result["claim_to_result_write_ms"],
            "result_write_to_client_visible_ms": result["result_write_to_client_visible_ms"],
            "result_write_started_to_completed_ms": result["result_write_started_to_completed_ms"],
            "result_write_completed_to_notify_ms": result["result_write_completed_to_notify_ms"],
            "result_write_completed_to_client_visible_ms": result["result_write_completed_to_client_visible_ms"],
            "broker_result_wait_ms": result["broker_result_wait_ms"],
            "broker_result_initial_fetch_to_release_ms": result["broker_result_initial_fetch_to_release_ms"],
            "broker_result_release_to_response_ms": result["broker_result_release_to_response_ms"],
            "broker_submit_response_ready_to_client_visible_ms": result["broker_submit_response_ready_to_client_visible_ms"],
            "broker_result_source": result["broker_result_source"],
            "cache_key_ms": result["cache_key_ms"],
            "resolve_input_path_ms": result["resolve_input_path_ms"],
            "hash_input_path_ms": result["hash_input_path_ms"],
            "hash_input_file_ms": result["hash_input_file_ms"],
            "serialize_cache_key_payload_ms": result["serialize_cache_key_payload_ms"],
            "hash_cache_key_payload_ms": result["hash_cache_key_payload_ms"],
            "build_inline_bundle_ms": result["build_inline_bundle_ms"],
            "stage_bundle_ms": result["stage_bundle_ms"],
            "backend_submit_ms": result["backend_submit_ms"],
            "store_create_job_ms": result["store_create_job_ms"],
            "total_submit_ms": result["total_submit_ms"],
            "inline_release_initial_probe_ms": result["inline_release_initial_probe_ms"],
            "inline_release_waiter_wait_ms": result["inline_release_waiter_wait_ms"],
            "inline_release_post_wait_release_build_ms": result["inline_release_post_wait_release_build_ms"],
            "inline_release_total_ms": result["inline_release_total_ms"],
            "worker_phase_process_bootstrap_ms": result["worker_phase_process_bootstrap_ms"],
            "worker_phase_load_job_inputs_ms": result["worker_phase_load_job_inputs_ms"],
            "worker_phase_import_prefetch_helpers_ms": result["worker_phase_import_prefetch_helpers_ms"],
            "worker_phase_prefetch_cache_context_ms": result["worker_phase_prefetch_cache_context_ms"],
            "worker_phase_import_run_inspection_ms": result["worker_phase_import_run_inspection_ms"],
            "worker_phase_run_inspection_ms": result["worker_phase_run_inspection_ms"],
            "worker_phase_write_artifacts_ms": result["worker_phase_write_artifacts_ms"],
            "worker_phase_finalize_ms": result["worker_phase_finalize_ms"],
            "manifest_load_ms": result["manifest_load_ms"],
            "shared_manifest_load_ms": result["shared_manifest_load_ms"],
            "shared_state_manifest_load_ms": result["shared_state_manifest_load_ms"],
            "snapshot_probe_ms": result["snapshot_probe_ms"],
            "previous_snapshot_load_ms": result["previous_snapshot_load_ms"],
            "discover_source_files_ms": result["discover_source_files_ms"],
            "shared_state_lookup_ms": result["shared_state_lookup_ms"],
            "file_chunk_bundle_load_ms": result["file_chunk_bundle_load_ms"],
            "file_chunk_bundle_write_ms": result["file_chunk_bundle_write_ms"],
            "source_read_ms": result["source_read_ms"],
            "chunk_build_ms": result["chunk_build_ms"],
            "chunk_content_hash_ms": result["chunk_content_hash_ms"],
            "chunk_identity_hash_ms": result["chunk_identity_hash_ms"],
            "chunk_token_estimate_ms": result["chunk_token_estimate_ms"],
            "manifest_records_ms": result["manifest_records_ms"],
            "semantic_signature_ms": result["semantic_signature_ms"],
            "symbol_marker_ms": result["symbol_marker_ms"],
            "symbol_marker_write_ms": result["symbol_marker_write_ms"],
            "working_manifest_write_ms": result["working_manifest_write_ms"],
            "shared_state_manifest_flush_ms": result["shared_state_manifest_flush_ms"],
            "snapshot_write_ms": result["snapshot_write_ms"],
            "git_dirty_manifest_keys_ms": result["git_dirty_manifest_keys_ms"],
            "git_file_signatures_ms": result["git_file_signatures_ms"],
            "build_syntax_chunks_ms": result["build_syntax_chunks_ms"],
            "ensure_lexical_index_ms": result["ensure_lexical_index_ms"],
            "query_stage_cache_ms": result["query_stage_cache_ms"],
            "write_query_stage_cache_ms": result["write_query_stage_cache_ms"],
            "artifact_payloads_ms": result["artifact_payloads_ms"],
            "chunk_cache_reused_files": result["chunk_cache_reused_files"],
            "chunk_cache_rebuilt_files": result["chunk_cache_rebuilt_files"],
        }
    )

summary = {
    "samples": samples,
    "mode": mode,
    "query": query,
    "cold": {
        "wall_ms": summarize(cold_records, "wall_ms"),
        "submit_http_ms": summarize(cold_records, "submit_http_ms"),
        "result_wait_http_ms": summarize(cold_records, "result_wait_http_ms"),
        "worker_phase_total_ms": summarize(cold_records, "worker_phase_total_ms"),
        "broker_control_overhead_ms": summarize(cold_records, "broker_control_overhead_ms"),
        "enqueue_to_claim_ms": summarize(cold_records, "enqueue_to_claim_ms"),
        "enqueue_to_written_ms": summarize(cold_records, "enqueue_to_written_ms"),
        "written_to_wakeup_ms": summarize(cold_records, "written_to_wakeup_ms"),
        "wakeup_to_claim_ms": summarize(cold_records, "wakeup_to_claim_ms"),
        "claim_to_result_write_ms": summarize(cold_records, "claim_to_result_write_ms"),
        "result_write_to_client_visible_ms": summarize(cold_records, "result_write_to_client_visible_ms"),
        "result_write_started_to_completed_ms": summarize(cold_records, "result_write_started_to_completed_ms"),
        "result_write_completed_to_notify_ms": summarize(cold_records, "result_write_completed_to_notify_ms"),
        "result_write_completed_to_client_visible_ms": summarize(cold_records, "result_write_completed_to_client_visible_ms"),
        "broker_result_wait_ms": summarize(cold_records, "broker_result_wait_ms"),
        "broker_result_initial_fetch_to_release_ms": summarize(cold_records, "broker_result_initial_fetch_to_release_ms"),
        "broker_result_release_to_response_ms": summarize(cold_records, "broker_result_release_to_response_ms"),
        "broker_submit_response_ready_to_client_visible_ms": summarize(cold_records, "broker_submit_response_ready_to_client_visible_ms"),
        "inline_release_initial_probe_ms": summarize(cold_records, "inline_release_initial_probe_ms"),
        "inline_release_waiter_wait_ms": summarize(cold_records, "inline_release_waiter_wait_ms"),
        "inline_release_post_wait_release_build_ms": summarize(cold_records, "inline_release_post_wait_release_build_ms"),
        "inline_release_total_ms": summarize(cold_records, "inline_release_total_ms"),
        "worker_phase_process_bootstrap_ms": summarize(cold_records, "worker_phase_process_bootstrap_ms"),
        "worker_phase_load_job_inputs_ms": summarize(cold_records, "worker_phase_load_job_inputs_ms"),
        "worker_phase_import_prefetch_helpers_ms": summarize(cold_records, "worker_phase_import_prefetch_helpers_ms"),
        "worker_phase_prefetch_cache_context_ms": summarize(cold_records, "worker_phase_prefetch_cache_context_ms"),
        "worker_phase_import_run_inspection_ms": summarize(cold_records, "worker_phase_import_run_inspection_ms"),
        "worker_phase_run_inspection_ms": summarize(cold_records, "worker_phase_run_inspection_ms"),
        "worker_phase_write_artifacts_ms": summarize(cold_records, "worker_phase_write_artifacts_ms"),
        "worker_phase_finalize_ms": summarize(cold_records, "worker_phase_finalize_ms"),
        "manifest_load_ms": summarize(cold_records, "manifest_load_ms"),
        "shared_manifest_load_ms": summarize(cold_records, "shared_manifest_load_ms"),
        "shared_state_manifest_load_ms": summarize(cold_records, "shared_state_manifest_load_ms"),
        "snapshot_probe_ms": summarize(cold_records, "snapshot_probe_ms"),
        "discover_source_files_ms": summarize(cold_records, "discover_source_files_ms"),
        "shared_state_lookup_ms": summarize(cold_records, "shared_state_lookup_ms"),
        "file_chunk_bundle_load_ms": summarize(cold_records, "file_chunk_bundle_load_ms"),
        "file_chunk_bundle_write_ms": summarize(cold_records, "file_chunk_bundle_write_ms"),
        "source_read_ms": summarize(cold_records, "source_read_ms"),
        "chunk_build_ms": summarize(cold_records, "chunk_build_ms"),
        "chunk_content_hash_ms": summarize(cold_records, "chunk_content_hash_ms"),
        "chunk_identity_hash_ms": summarize(cold_records, "chunk_identity_hash_ms"),
        "chunk_token_estimate_ms": summarize(cold_records, "chunk_token_estimate_ms"),
        "manifest_records_ms": summarize(cold_records, "manifest_records_ms"),
        "semantic_signature_ms": summarize(cold_records, "semantic_signature_ms"),
        "symbol_marker_ms": summarize(cold_records, "symbol_marker_ms"),
        "symbol_marker_write_ms": summarize(cold_records, "symbol_marker_write_ms"),
        "working_manifest_write_ms": summarize(cold_records, "working_manifest_write_ms"),
        "shared_state_manifest_flush_ms": summarize(cold_records, "shared_state_manifest_flush_ms"),
        "snapshot_write_ms": summarize(cold_records, "snapshot_write_ms"),
        "git_file_signatures_ms": summarize(cold_records, "git_file_signatures_ms"),
        "build_syntax_chunks_ms": summarize(cold_records, "build_syntax_chunks_ms"),
        "ensure_lexical_index_ms": summarize(cold_records, "ensure_lexical_index_ms"),
        "query_stage_cache_ms": summarize(cold_records, "query_stage_cache_ms"),
        "write_query_stage_cache_ms": summarize(cold_records, "write_query_stage_cache_ms"),
        "artifact_payloads_ms": summarize(cold_records, "artifact_payloads_ms"),
        "records": cold_records,
    },
    "partial_dirty": {
        "baseline_wall_ms": baseline["wall_ms"],
        "baseline_discover_source_files_ms": baseline["discover_source_files_ms"],
        "wall_ms": summarize(partial_dirty_records, "wall_ms"),
        "submit_http_ms": summarize(partial_dirty_records, "submit_http_ms"),
        "result_wait_http_ms": summarize(partial_dirty_records, "result_wait_http_ms"),
        "worker_phase_total_ms": summarize(partial_dirty_records, "worker_phase_total_ms"),
        "broker_control_overhead_ms": summarize(partial_dirty_records, "broker_control_overhead_ms"),
        "enqueue_to_claim_ms": summarize(partial_dirty_records, "enqueue_to_claim_ms"),
        "enqueue_to_written_ms": summarize(partial_dirty_records, "enqueue_to_written_ms"),
        "written_to_wakeup_ms": summarize(partial_dirty_records, "written_to_wakeup_ms"),
        "wakeup_to_claim_ms": summarize(partial_dirty_records, "wakeup_to_claim_ms"),
        "claim_to_result_write_ms": summarize(partial_dirty_records, "claim_to_result_write_ms"),
        "result_write_to_client_visible_ms": summarize(partial_dirty_records, "result_write_to_client_visible_ms"),
        "result_write_started_to_completed_ms": summarize(partial_dirty_records, "result_write_started_to_completed_ms"),
        "result_write_completed_to_notify_ms": summarize(partial_dirty_records, "result_write_completed_to_notify_ms"),
        "result_write_completed_to_client_visible_ms": summarize(partial_dirty_records, "result_write_completed_to_client_visible_ms"),
        "broker_result_wait_ms": summarize(partial_dirty_records, "broker_result_wait_ms"),
        "broker_result_initial_fetch_to_release_ms": summarize(partial_dirty_records, "broker_result_initial_fetch_to_release_ms"),
        "broker_result_release_to_response_ms": summarize(partial_dirty_records, "broker_result_release_to_response_ms"),
        "broker_submit_response_ready_to_client_visible_ms": summarize(partial_dirty_records, "broker_submit_response_ready_to_client_visible_ms"),
        "inline_release_initial_probe_ms": summarize(partial_dirty_records, "inline_release_initial_probe_ms"),
        "inline_release_waiter_wait_ms": summarize(partial_dirty_records, "inline_release_waiter_wait_ms"),
        "inline_release_post_wait_release_build_ms": summarize(partial_dirty_records, "inline_release_post_wait_release_build_ms"),
        "inline_release_total_ms": summarize(partial_dirty_records, "inline_release_total_ms"),
        "worker_phase_process_bootstrap_ms": summarize(partial_dirty_records, "worker_phase_process_bootstrap_ms"),
        "worker_phase_load_job_inputs_ms": summarize(partial_dirty_records, "worker_phase_load_job_inputs_ms"),
        "worker_phase_import_prefetch_helpers_ms": summarize(partial_dirty_records, "worker_phase_import_prefetch_helpers_ms"),
        "worker_phase_prefetch_cache_context_ms": summarize(partial_dirty_records, "worker_phase_prefetch_cache_context_ms"),
        "worker_phase_import_run_inspection_ms": summarize(partial_dirty_records, "worker_phase_import_run_inspection_ms"),
        "worker_phase_run_inspection_ms": summarize(partial_dirty_records, "worker_phase_run_inspection_ms"),
        "worker_phase_write_artifacts_ms": summarize(partial_dirty_records, "worker_phase_write_artifacts_ms"),
        "worker_phase_finalize_ms": summarize(partial_dirty_records, "worker_phase_finalize_ms"),
        "manifest_load_ms": summarize(partial_dirty_records, "manifest_load_ms"),
        "shared_manifest_load_ms": summarize(partial_dirty_records, "shared_manifest_load_ms"),
        "shared_state_manifest_load_ms": summarize(partial_dirty_records, "shared_state_manifest_load_ms"),
        "snapshot_probe_ms": summarize(partial_dirty_records, "snapshot_probe_ms"),
        "previous_snapshot_load_ms": summarize(partial_dirty_records, "previous_snapshot_load_ms"),
        "discover_source_files_ms": summarize(partial_dirty_records, "discover_source_files_ms"),
        "shared_state_lookup_ms": summarize(partial_dirty_records, "shared_state_lookup_ms"),
        "file_chunk_bundle_load_ms": summarize(partial_dirty_records, "file_chunk_bundle_load_ms"),
        "file_chunk_bundle_write_ms": summarize(partial_dirty_records, "file_chunk_bundle_write_ms"),
        "source_read_ms": summarize(partial_dirty_records, "source_read_ms"),
        "chunk_build_ms": summarize(partial_dirty_records, "chunk_build_ms"),
        "chunk_content_hash_ms": summarize(partial_dirty_records, "chunk_content_hash_ms"),
        "chunk_identity_hash_ms": summarize(partial_dirty_records, "chunk_identity_hash_ms"),
        "chunk_token_estimate_ms": summarize(partial_dirty_records, "chunk_token_estimate_ms"),
        "manifest_records_ms": summarize(partial_dirty_records, "manifest_records_ms"),
        "semantic_signature_ms": summarize(partial_dirty_records, "semantic_signature_ms"),
        "symbol_marker_ms": summarize(partial_dirty_records, "symbol_marker_ms"),
        "symbol_marker_write_ms": summarize(partial_dirty_records, "symbol_marker_write_ms"),
        "working_manifest_write_ms": summarize(partial_dirty_records, "working_manifest_write_ms"),
        "shared_state_manifest_flush_ms": summarize(partial_dirty_records, "shared_state_manifest_flush_ms"),
        "snapshot_write_ms": summarize(partial_dirty_records, "snapshot_write_ms"),
        "git_dirty_manifest_keys_ms": summarize(partial_dirty_records, "git_dirty_manifest_keys_ms"),
        "git_file_signatures_ms": summarize(partial_dirty_records, "git_file_signatures_ms"),
        "build_syntax_chunks_ms": summarize(partial_dirty_records, "build_syntax_chunks_ms"),
        "ensure_lexical_index_ms": summarize(partial_dirty_records, "ensure_lexical_index_ms"),
        "query_stage_cache_ms": summarize(partial_dirty_records, "query_stage_cache_ms"),
        "write_query_stage_cache_ms": summarize(partial_dirty_records, "write_query_stage_cache_ms"),
        "artifact_payloads_ms": summarize(partial_dirty_records, "artifact_payloads_ms"),
        "records": partial_dirty_records,
    },
}
print(json.dumps(summary, indent=2))
PY
