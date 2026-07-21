#!/usr/bin/env python3

import json
import os
import select
import socket
import sys
import time

PROCESS_BOOTSTRAP_STARTED = time.perf_counter()
DEFAULT_WARM_DAEMON_POLL_INTERVAL_SECONDS = 0.01
_VALIDATE_REQUEST = None
_PREPARE_PREFETCHED_STATE = None
_CACHED_LEXICAL_FALLBACK_FROM_CONTEXT = None
_RUN_INSPECTION = None

WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKER_DIR not in sys.path:
    sys.path.insert(0, WORKER_DIR)


def parse_args(argv):
    options = {
        "job_spec": None,
        "execution_plan": None,
        "input_manifest": None,
        "output_dir": None,
        "heartbeat_path": None,
        "completion_socket_path": None,
        "daemon_spool_dir": None,
        "repo_root": None,
    }
    index = 0
    while index < len(argv):
        flag = argv[index]
        index += 1
        if not flag.startswith("--"):
            raise ValueError(f"unexpected argument: {flag}")
        if index >= len(argv):
            raise ValueError(f"missing value for {flag}")
        value = argv[index]
        index += 1
        if flag == "--job-spec":
            options["job_spec"] = value
        elif flag == "--execution-plan":
            options["execution_plan"] = value
        elif flag == "--input-manifest":
            options["input_manifest"] = value
        elif flag == "--output-dir":
            options["output_dir"] = value
        elif flag == "--heartbeat-path":
            options["heartbeat_path"] = value
        elif flag == "--completion-socket-path":
            options["completion_socket_path"] = value
        elif flag == "--daemon-spool-dir":
            options["daemon_spool_dir"] = value
        elif flag == "--repo-root":
            options["repo_root"] = value
        else:
            raise ValueError(f"unexpected argument: {flag}")
    if not options["daemon_spool_dir"]:
        missing = [name for name in ("job_spec", "input_manifest", "output_dir") if not options[name]]
        if missing:
            missing_flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            raise ValueError(f"missing required arguments: {missing_flags}")
    return options


def main():
    args = parse_args(sys.argv[1:])

    if args.get("daemon_spool_dir"):
        if args.get("repo_root"):
            os.environ["BROKER_REPO_ROOT"] = str(args["repo_root"])
        return run_warm_daemon(args["daemon_spool_dir"])
    return run_staged_job(
        args["job_spec"],
        args["execution_plan"],
        args["input_manifest"],
        args["output_dir"],
        args.get("heartbeat_path"),
        args.get("completion_socket_path"),
    )


def warm_daemon_poll_interval_seconds():
    raw = os.environ.get("BROKER_LOCAL_INSPECT_REPO_WARM_POLL_INTERVAL_SECONDS", "").strip()
    if not raw:
        return DEFAULT_WARM_DAEMON_POLL_INTERVAL_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_WARM_DAEMON_POLL_INTERVAL_SECONDS
    if value <= 0:
        return DEFAULT_WARM_DAEMON_POLL_INTERVAL_SECONDS
    return value


def run_warm_daemon(spool_dir):
    from pathlib import Path

    spool_path = Path(spool_dir).expanduser().resolve(strict=False)
    request_dir = spool_path / "requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    poll_interval_seconds = warm_daemon_poll_interval_seconds()
    wakeup_socket = open_warm_daemon_wakeup_socket(spool_path)
    preload_warm_daemon_modules()
    recover_warm_requests(request_dir)
    try:
        while True:
            write_warm_daemon_heartbeat(spool_path)
            processed = False
            wake_request_names = []
            for request_path in iter_warm_request_paths(request_dir, wake_request_names):
                if process_claimed_warm_request(request_path):
                    processed = True
            if not processed:
                wake_request_names = wait_for_warm_daemon_wakeup(wakeup_socket, poll_interval_seconds)
                if wake_request_names:
                    wakeup_metadata = build_wakeup_metadata_map(wake_request_names)
                    for request_path in iter_warm_request_paths(request_dir, wake_request_names):
                        if process_claimed_warm_request(
                            request_path,
                            wakeup_metadata.get(request_path.name),
                        ):
                            processed = True
    finally:
        close_warm_daemon_wakeup_socket(wakeup_socket, spool_path)


def recover_warm_requests(request_dir):
    for working_path in sorted(request_dir.glob("*.working")):
        request_path = working_path.with_suffix(".json")
        if request_path.exists():
            working_path.unlink(missing_ok=True)
            continue
        try:
            working_path.rename(request_path)
        except OSError:
            continue


def preload_warm_daemon_modules():
    validate_request_function()
    prefetch_helpers()
    run_inspection_function()


def validate_request_function():
    global _VALIDATE_REQUEST
    if _VALIDATE_REQUEST is None:
        from inspection_contract import validate_request as imported_validate_request

        _VALIDATE_REQUEST = imported_validate_request
    return _VALIDATE_REQUEST


def prefetch_helpers():
    global _PREPARE_PREFETCHED_STATE, _CACHED_LEXICAL_FALLBACK_FROM_CONTEXT
    if _PREPARE_PREFETCHED_STATE is None or _CACHED_LEXICAL_FALLBACK_FROM_CONTEXT is None:
        from inspection_hotpath import (
            cached_lexical_fallback_from_context as imported_cached_lexical_fallback_from_context,
            prepare_prefetched_state as imported_prepare_prefetched_state,
        )

        _PREPARE_PREFETCHED_STATE = imported_prepare_prefetched_state
        _CACHED_LEXICAL_FALLBACK_FROM_CONTEXT = imported_cached_lexical_fallback_from_context
    return _PREPARE_PREFETCHED_STATE, _CACHED_LEXICAL_FALLBACK_FROM_CONTEXT


def run_inspection_function():
    global _RUN_INSPECTION
    if _RUN_INSPECTION is None:
        from inspection_pipeline import run_inspection as imported_run_inspection

        _RUN_INSPECTION = imported_run_inspection
    return _RUN_INSPECTION


def write_warm_daemon_heartbeat(spool_path):
    from pathlib import Path

    heartbeat_path = Path(spool_path) / "daemon-heartbeat.json"
    payload = {
        "state": "running",
        "timestamp": time.time(),
        "pid": os.getpid(),
    }
    write_json(heartbeat_path, payload, durable=False)


def warm_daemon_wakeup_socket_path(spool_path):
    from pathlib import Path

    return Path(spool_path) / "daemon.sock"


def open_warm_daemon_wakeup_socket(spool_path):
    socket_path = warm_daemon_wakeup_socket_path(spool_path)
    socket_path.unlink(missing_ok=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    listener.bind(str(socket_path))
    listener.setblocking(False)
    return listener


def iter_warm_request_paths(request_dir, wake_request_names=None):
    seen = set()
    for raw_name in list(wake_request_names or ()):
        request_name = str(raw_name or "").strip()
        if not request_name or not request_name.endswith(".json"):
            continue
        if "/" in request_name or "\\" in request_name:
            continue
        request_path = request_dir / request_name
        if request_path in seen:
            continue
        seen.add(request_path)
        yield request_path
    for request_path in sorted(request_dir.glob("*.json")):
        if request_path in seen:
            continue
        seen.add(request_path)
        yield request_path


def process_claimed_warm_request(request_path, wakeup_metadata=None):
    claim_path = request_path.with_suffix(".working")
    try:
        request_path.rename(claim_path)
    except OSError:
        return False
    try:
        process_warm_request(claim_path, wakeup_metadata=wakeup_metadata)
    finally:
        claim_path.unlink(missing_ok=True)
    return True


def wait_for_warm_daemon_wakeup(listener, timeout_seconds):
    if listener is None:
        time.sleep(timeout_seconds)
        return []
    ready, _, _ = select.select([listener], [], [], timeout_seconds)
    if not ready:
        return []
    messages = []
    try:
        while True:
            payload = listener.recv(4096)
            if payload:
                messages.append(payload.decode("utf-8", errors="ignore").strip())
    except BlockingIOError:
        return [message for message in messages if message]
    except OSError:
        return [message for message in messages if message]


def build_wakeup_metadata_map(wake_request_names):
    received_unix_ns = int(time.time_ns())
    received_monotonic_ns = int(time.monotonic_ns())
    metadata = {}
    for request_name in list(wake_request_names or ()):
        request_name = str(request_name or "").strip()
        if not request_name:
            continue
        metadata[request_name] = {
            "worker_wakeup_received_unix_ns": received_unix_ns,
            "worker_wakeup_received_monotonic_ns": received_monotonic_ns,
        }
    return metadata


def close_warm_daemon_wakeup_socket(listener, spool_path):
    if listener is not None:
        try:
            listener.close()
        except OSError:
            pass
    warm_daemon_wakeup_socket_path(spool_path).unlink(missing_ok=True)


def warm_daemon_busy_marker_path(spool_path):
    return spool_path / "busy.marker"


def process_warm_request(request_path, wakeup_metadata=None):
    from pathlib import Path

    spool_path = request_path.parent.parent
    busy_marker_path = warm_daemon_busy_marker_path(spool_path)
    request = load_json(request_path)
    if isinstance(wakeup_metadata, dict):
        request.update(wakeup_metadata)
    request["worker_claimed_monotonic_ns"] = int(time.monotonic_ns())
    request["worker_claimed_unix_ns"] = int(time.time_ns())
    write_json(
        busy_marker_path,
        {
            "request": request_path.name,
            "claimed_unix_ns": int(request["worker_claimed_unix_ns"]),
            "claimed_monotonic_ns": int(request["worker_claimed_monotonic_ns"]),
        },
        durable=False,
    )
    output_dir = Path(request["output_dir"])
    cancel_path = output_dir / "cancel.request"
    heartbeat_path = Path(request.get("heartbeat_path") or output_dir / "heartbeat.json")
    job_spec = request.get("job_spec") or load_json(request["job_spec_path"])
    job_id = str(request.get("job_id") or job_spec.get("job_id") or output_dir.name)
    if cancel_path.exists():
        emit_heartbeat(
            heartbeat_path,
            job_id,
            "cancelled",
            "cancelled",
            100,
            "Repository inspection cancelled before execution",
            {},
        )
        cancel_path.unlink(missing_ok=True)
        (output_dir / "warm-request.marker").unlink(missing_ok=True)
        return
    try:
        return run_staged_job(
            request.get("job_spec_path"),
            request.get("execution_plan_path"),
            request.get("input_manifest_path"),
            request["output_dir"],
            request.get("heartbeat_path"),
            request.get("completion_socket_path"),
            inline_job_spec=request.get("job_spec"),
            inline_execution_plan=request.get("execution_plan"),
            inline_input_manifest=request.get("input_manifest"),
            warm_slot_release=lambda: release_warm_request_slot(output_dir, busy_marker_path),
            warm_request=request,
            daemon_mode=True,
        )
    except Exception as exc:
        emit_heartbeat(
            heartbeat_path,
            job_id,
            "failed",
            "failed",
            100,
            f"Repository inspection failed: {exc}",
            {},
        )
        raise
    finally:
        busy_marker_path.unlink(missing_ok=True)
        cancel_path.unlink(missing_ok=True)
        (output_dir / "warm-request.marker").unlink(missing_ok=True)


def release_warm_request_slot(output_dir, busy_marker_path):
    busy_marker_path.unlink(missing_ok=True)
    (output_dir / "warm-request.marker").unlink(missing_ok=True)


def run_staged_job(
    job_spec_path,
    execution_plan_path,
    input_manifest_path,
    output_dir_value,
    heartbeat_path_value=None,
    completion_socket_path_value=None,
    *,
    inline_job_spec=None,
    inline_execution_plan=None,
    inline_input_manifest=None,
    warm_slot_release=None,
    warm_request=None,
    daemon_mode=False,
):
    from pathlib import Path

    worker_started = time.perf_counter()
    process_bootstrap_started = worker_started if daemon_mode else PROCESS_BOOTSTRAP_STARTED
    after_parse_args = worker_started

    job_spec = inline_job_spec or load_json(job_spec_path)
    input_manifest = inline_input_manifest or load_json(input_manifest_path)
    output_dir = Path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = Path(heartbeat_path_value) if heartbeat_path_value else output_dir / "heartbeat.json"
    completion_socket_path = completion_socket_path_value
    execution_plan_path = Path(execution_plan_path) if execution_plan_path else output_dir / "execution_plan.json"
    execution_plan = inline_execution_plan if inline_execution_plan is not None else load_optional_json(execution_plan_path)
    apply_execution_plan_environment(execution_plan)
    after_load_inputs = time.perf_counter()

    task_params = job_spec.get("task_params") or {}
    if daemon_mode:
        task_params = dict(task_params)
        task_params["_broker_skip_shared_lexical_publish"] = True
        task_params["_broker_skip_shared_chunk_publish"] = True
    constraints = job_spec.get("constraints") or {}
    input_refs = input_manifest.get("input_refs") or []
    if not input_refs:
        raise ValueError("inspect_repo requires at least one input ref")

    validate_request = validate_request_function()
    after_import_validate = time.perf_counter()

    query, mode = validate_request(task_params.get("query"), task_params.get("mode", "auto"))
    after_validate = time.perf_counter()
    discovered = discover_inputs(input_refs)
    after_discover = time.perf_counter()
    job_id = str(job_spec["job_id"])

    prepare_prefetched_state, cached_lexical_fallback_from_context = prefetch_helpers()
    after_import_prefetch = time.perf_counter()

    prefetched_state = prepare_prefetched_state(
        discovered,
        query,
        mode=mode,
        constraints=constraints,
        task_params=task_params,
        execution_plan=execution_plan,
        output_dir=output_dir,
    )
    after_prefetch = time.perf_counter()
    cached_run = prefetched_state.get("cached_lexical_fallback_run")
    if (
        cached_run is None
        and prefetched_state.get("cached_query_stage") is not None
        and not bool(prefetched_state.get("prefetched_query_stage_requires_verification"))
    ):
        cached_run = cached_lexical_fallback_from_context(prefetched_state)
    after_cached_probe = time.perf_counter()
    if cached_run is not None:
        payload = cached_run["payload"]
        annotate_runtime_mode(payload, daemon_mode=daemon_mode)
        annotate_prefetch_runtime(payload, prefetched_state)
        record_phase_timings(
            payload,
            process_bootstrap_started=process_bootstrap_started,
            worker_started=worker_started,
            after_parse_args=after_parse_args,
            after_load_inputs=after_load_inputs,
            after_import_validate=after_import_validate,
            after_validate=after_validate,
            after_discover=after_discover,
            after_import_prefetch=after_import_prefetch,
            after_prefetch=after_prefetch,
            after_cached_probe=after_cached_probe,
            after_import_pipeline=after_cached_probe,
            after_run=after_cached_probe,
            after_artifacts=after_cached_probe,
            completed_at=time.perf_counter(),
            cache_hit=True,
        )
        artifacts = write_repo_inspection_artifacts(
            output_dir,
            cached_run.get("artifact_payloads") or {},
            highest_classification(discovered),
            include_full_trace=bool(task_params.get("include_full_trace")),
        )
        annotate_lifecycle_runtime(payload, warm_request=warm_request)
        result = {
            "schema_name": "repo_inspection_v2",
            "schema_version": "2.0.0",
            "payload": payload,
        }
        write_json(output_dir / "artifacts.json", artifacts, durable=False)
        annotate_result_write_started_runtime(payload)
        if daemon_mode and callable(warm_slot_release):
            warm_slot_release()
        write_json(output_dir / "result.json", result, durable=False)
        annotate_result_write_completed_runtime(payload)
        notify_completion(completion_socket_path, job_id)
        annotate_completion_notified_runtime(payload)
        emit_heartbeat(
            heartbeat_path,
            job_id,
            "completed",
            "completed",
            100,
            "Repository inspection completed from persisted lexical-fallback cache",
            {
                "evidence_count": len(payload.get("evidence") or []),
                "quality_result": (payload.get("quality") or {}).get("result", ""),
                "artifact_count": len(artifacts),
                "query_stage_cache_hit": True,
                "worker_phase_timings_ms": ((payload.get("runtime") or {}).get("worker_phase_timings_ms") or {}),
                **completion_lifecycle_metrics(payload),
            },
        )
        return 0

    emit_heartbeat(
        heartbeat_path,
        job_id,
        "running",
        "gpu_first_retrieval",
        30,
        "Building repository indexes and retrieving GPU candidates",
        {},
    )
    run_inspection = run_inspection_function()
    after_import_pipeline = time.perf_counter()

    run = run_inspection(
        discovered,
        query,
        mode=mode,
        constraints=constraints,
        task_params=task_params,
        execution_plan=execution_plan,
        output_dir=output_dir,
        prefetched_state=prefetched_state,
    )
    after_run = time.perf_counter()
    payload = run["payload"]
    annotate_runtime_mode(payload, daemon_mode=daemon_mode)
    annotate_prefetch_runtime(payload, prefetched_state)
    if bool((payload.get("quality") or {}).get("answer_ready")):
        emit_heartbeat(
            heartbeat_path,
            job_id,
            "running",
            "validated_synthesis",
            85,
            "Validating GPU synthesis and evidence citations",
            {
                "evidence_count": len(payload.get("evidence") or []),
                "answer_ready": True,
            },
        )
    artifacts = write_repo_inspection_artifacts(
        output_dir,
        run.get("artifact_payloads") or {},
        highest_classification(discovered),
        include_full_trace=bool(task_params.get("include_full_trace")),
    )
    after_artifacts = time.perf_counter()
    record_phase_timings(
        payload,
        process_bootstrap_started=process_bootstrap_started,
        worker_started=worker_started,
        after_parse_args=after_parse_args,
        after_load_inputs=after_load_inputs,
        after_import_validate=after_import_validate,
        after_validate=after_validate,
        after_discover=after_discover,
        after_import_prefetch=after_import_prefetch,
        after_prefetch=after_prefetch,
        after_cached_probe=after_cached_probe,
        after_import_pipeline=after_import_pipeline,
        after_run=after_run,
        after_artifacts=after_artifacts,
        completed_at=time.perf_counter(),
        cache_hit=False,
    )
    annotate_lifecycle_runtime(payload, warm_request=warm_request)
    result = {
        "schema_name": "repo_inspection_v2",
        "schema_version": "2.0.0",
        "payload": payload,
    }
    write_json(output_dir / "artifacts.json", artifacts, durable=False)
    annotate_result_write_started_runtime(payload)
    if daemon_mode and callable(warm_slot_release):
        warm_slot_release()
    write_json(output_dir / "result.json", result, durable=False)
    annotate_result_write_completed_runtime(payload)
    notify_completion(completion_socket_path, job_id)
    annotate_completion_notified_runtime(payload)
    emit_heartbeat(
        heartbeat_path,
        job_id,
        "completed",
        "completed",
        100,
        "Repository inspection completed",
        {
            "evidence_count": len(payload.get("evidence") or []),
            "quality_result": (payload.get("quality") or {}).get("result", ""),
            "artifact_count": len(artifacts),
            "worker_phase_timings_ms": ((payload.get("runtime") or {}).get("worker_phase_timings_ms") or {}),
            **completion_lifecycle_metrics(payload),
        },
    )
    return 0


def annotate_runtime_mode(payload, *, daemon_mode=False):
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        payload["runtime"] = runtime
    runtime["local_backend_mode"] = "warm_daemon" if daemon_mode else "direct_worker"
    runtime["warm_daemon_active"] = bool(daemon_mode)


def annotate_prefetch_runtime(payload, prefetched_state):
    if not isinstance(prefetched_state, dict):
        return
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        payload["runtime"] = runtime
    runtime["prefetch_state_source"] = str(prefetched_state.get("prefetch_state_source") or "")
    runtime["prefetch_state_cache_hit"] = bool(prefetched_state.get("prefetch_state_cache_hit"))
    timings = prefetched_state.get("prefetch_stage_timings_ms")
    if isinstance(timings, dict) and timings:
        runtime["prefetch_stage_timings_ms"] = dict(timings)


def annotate_lifecycle_runtime(payload, *, warm_request=None):
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        payload["runtime"] = runtime
    lifecycle = runtime.get("broker_lifecycle")
    if not isinstance(lifecycle, dict):
        lifecycle = {}
        runtime["broker_lifecycle"] = lifecycle
    lifecycle["worker_result_payload_ready_monotonic_ns"] = int(time.monotonic_ns())
    lifecycle["worker_result_payload_ready_unix_ns"] = int(time.time_ns())
    if not isinstance(warm_request, dict):
        return
    for source_key, target_key in (
        ("broker_request_enqueued_ns", "broker_request_enqueued_unix_ns"),
        ("broker_request_written_ns", "broker_request_written_unix_ns"),
        ("worker_wakeup_received_unix_ns", "worker_wakeup_received_unix_ns"),
        ("worker_wakeup_received_monotonic_ns", "worker_wakeup_received_monotonic_ns"),
        ("worker_claimed_monotonic_ns", "worker_claimed_monotonic_ns"),
        ("worker_claimed_unix_ns", "worker_claimed_unix_ns"),
    ):
        value = warm_request.get(source_key)
        try:
            if value is not None:
                lifecycle[target_key] = int(value)
        except (TypeError, ValueError):
            continue


def annotate_result_write_started_runtime(payload):
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        payload["runtime"] = runtime
    lifecycle = runtime.get("broker_lifecycle")
    if not isinstance(lifecycle, dict):
        lifecycle = {}
        runtime["broker_lifecycle"] = lifecycle
    lifecycle["worker_result_write_started_monotonic_ns"] = int(time.monotonic_ns())
    lifecycle["worker_result_write_started_unix_ns"] = int(time.time_ns())


def annotate_result_write_completed_runtime(payload):
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        payload["runtime"] = runtime
    lifecycle = runtime.get("broker_lifecycle")
    if not isinstance(lifecycle, dict):
        lifecycle = {}
        runtime["broker_lifecycle"] = lifecycle
    lifecycle["worker_result_written_monotonic_ns"] = int(time.monotonic_ns())
    lifecycle["worker_result_written_unix_ns"] = int(time.time_ns())


def annotate_completion_notified_runtime(payload):
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        payload["runtime"] = runtime
    lifecycle = runtime.get("broker_lifecycle")
    if not isinstance(lifecycle, dict):
        lifecycle = {}
        runtime["broker_lifecycle"] = lifecycle
    lifecycle["worker_completion_notified_monotonic_ns"] = int(time.monotonic_ns())
    lifecycle["worker_completion_notified_unix_ns"] = int(time.time_ns())


def completion_lifecycle_metrics(payload):
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        return {}
    lifecycle = runtime.get("broker_lifecycle")
    if not isinstance(lifecycle, dict):
        return {}
    metrics = {}
    for key in (
        "worker_result_payload_ready_monotonic_ns",
        "worker_result_payload_ready_unix_ns",
        "worker_result_write_started_monotonic_ns",
        "worker_result_write_started_unix_ns",
        "worker_result_written_monotonic_ns",
        "worker_result_written_unix_ns",
        "worker_completion_notified_monotonic_ns",
        "worker_completion_notified_unix_ns",
    ):
        value = lifecycle.get(key)
        try:
            if value is not None:
                metrics[key] = int(value)
        except (TypeError, ValueError):
            continue
    return metrics


def apply_execution_plan_environment(execution_plan):
    execution_plan = execution_plan or {}
    shared_cache_path = execution_plan.get("repo_inspection_shared_cache_path")
    if not shared_cache_path and bool(execution_plan.get("repo_inspection_use_node_local_cache")):
        shared_cache_path = execution_plan.get("repo_inspection_cache_path")
    if shared_cache_path:
        os.environ["BROKER_REPO_INSPECTION_SHARED_CACHE_DIR"] = str(shared_cache_path)


def record_phase_timings(
    payload,
    *,
    process_bootstrap_started,
    worker_started,
    after_parse_args,
    after_load_inputs,
    after_import_validate,
    after_validate,
    after_discover,
    after_import_prefetch,
    after_prefetch,
    after_cached_probe,
    after_import_pipeline,
    after_run,
    after_artifacts,
    completed_at,
    cache_hit,
):
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        payload["runtime"] = runtime
    timings = {
        "process_bootstrap": round((worker_started - process_bootstrap_started) * 1000.0, 3),
        "parse_args": round((after_parse_args - worker_started) * 1000.0, 3),
        "load_job_inputs": round((after_load_inputs - after_parse_args) * 1000.0, 3),
        "import_validate_request": round((after_import_validate - after_load_inputs) * 1000.0, 3),
        "validate_request": round((after_validate - worker_started) * 1000.0, 3),
        "discover_inputs": round((after_discover - after_validate) * 1000.0, 3),
        "import_prefetch_helpers": round((after_import_prefetch - after_discover) * 1000.0, 3),
        "prefetch_cache_context": round((after_prefetch - after_discover) * 1000.0, 3),
        "cached_probe": round((after_cached_probe - after_prefetch) * 1000.0, 3),
        "import_run_inspection": round((after_import_pipeline - after_cached_probe) * 1000.0, 3),
        "run_inspection": round((after_run - after_cached_probe) * 1000.0, 3),
        "write_artifacts": round((after_artifacts - after_run) * 1000.0, 3),
        "finalize": round((completed_at - after_artifacts) * 1000.0, 3),
        "total": round((completed_at - process_bootstrap_started) * 1000.0, 3),
    }
    timings["cache_hit"] = bool(cache_hit)
    runtime["worker_phase_timings_ms"] = timings


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_optional_json(path):
    if not os.path.exists(path):
        return {}
    return load_json(path)


def write_json(path, payload, *, durable=True):
    os.makedirs(os.path.dirname(os.fspath(path)), exist_ok=True)
    path_text = os.fspath(path)
    tmp = f"{path_text}.tmp-{os.getpid()}"
    try:
        os.unlink(tmp)
    except FileNotFoundError:
        pass
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
        handle.flush()
        if durable:
            os.fsync(handle.fileno())
    os.replace(tmp, path_text)


def emit_heartbeat(path, job_id, state, phase, percent, message, metrics):
    if path is None:
        return
    from datetime import datetime, timezone

    payload = {
        "job_id": job_id,
        "state": state,
        "phase": phase,
        "percent": percent,
        "message": message,
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "metrics": metrics,
    }
    write_json(path, payload, durable=False)


def notify_completion(socket_path, job_id):
    if not socket_path or not job_id:
        return
    import socket

    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            client.connect(socket_path)
            client.send(str(job_id).encode("utf-8"))
        finally:
            client.close()
    except OSError:
        return


def discover_inputs(input_refs):
    from pathlib import Path

    discovered = []
    for idx, ref in enumerate(input_refs):
        uri = ref.get("uri", "")
        ref_type = ref.get("type", "")
        classification = ref.get("classification", "unknown")
        metadata = ref.get("metadata") or {}
        if uri.startswith("artifact://"):
            resolved_path = metadata.get("resolved_path", "")
            content = ""
            path = None
            if resolved_path:
                path = Path(resolved_path)
                if path.exists() and path.is_file():
                    content = path.read_text(encoding="utf-8", errors="replace")
            discovered.append(
                {
                    "id": f"input_{idx}",
                    "type": ref_type or "artifact",
                    "uri": uri,
                    "classification": classification,
                    "artifact_id": trim_artifact_prefix(uri),
                    "artifact_type": metadata.get("artifact_type", ""),
                    "source_job_id": metadata.get("source_job_id", ""),
                    "path": path,
                    "content": content,
                    "content_hash": str(ref.get("content_hash") or metadata.get("content_hash") or ""),
                }
            )
            continue
        path = resolve_file_uri(uri)
        if path.is_dir():
            discovered.append(
                {
                    "id": f"input_{idx}",
                    "type": ref_type or "repo",
                    "uri": uri,
                    "classification": classification,
                    "path": path,
                    "content": "",
                    "content_hash": str(ref.get("content_hash") or metadata.get("content_hash") or ""),
                }
            )
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        discovered.append(
            {
                "id": f"input_{idx}",
                "type": ref_type or "file",
                "uri": uri,
                "classification": classification,
                "path": path,
                "content": text,
                "content_hash": str(ref.get("content_hash") or metadata.get("content_hash") or ""),
            }
        )
    return discovered


def resolve_file_uri(uri):
    from pathlib import Path
    from urllib.parse import unquote, urlparse

    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"unsupported input uri: {uri}")
    return Path(unquote(parsed.path))


def write_repo_inspection_artifacts(output_dir, artifact_payloads, classification, *, include_full_trace=True):
    definitions = {
        "evidence_pack": ("artifact_evidence_pack", "evidence_pack"),
        "retrieval_result": ("artifact_retrieval_result", "retrieval_result"),
        "runtime_diagnostics": ("artifact_runtime_diagnostics", "runtime_diagnostics"),
        "chunk_manifest": ("artifact_chunk_manifest", "chunk_manifest"),
    }
    allowed_names = {"evidence_pack"}
    if include_full_trace:
        allowed_names.update({"retrieval_result", "runtime_diagnostics", "chunk_manifest"})
    artifacts = []
    for name, payload in artifact_payloads.items():
        if name not in allowed_names:
            continue
        definition = definitions.get(name)
        if definition is None:
            continue
        path = output_dir / f"{name}.json"
        write_json(path, payload, durable=False)
        artifact_id, artifact_type = definition
        artifacts.append(
            {
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "classification": classification,
                "path": str(path),
            }
        )
    return artifacts


def highest_classification(items):
    order = {"public": 0, "internal": 1, "restricted": 2, "phi": 3, "secret_adjacent": 4, "unknown": -1}
    best = "unknown"
    for item in items:
        value = item.get("classification", "unknown")
        if order.get(value, -1) > order.get(best, -1):
            best = value
    return best


def trim_artifact_prefix(uri):
    prefix = "artifact://"
    if uri.startswith(prefix):
        return uri[len(prefix):]
    return uri


if __name__ == "__main__":
    raise SystemExit(main())
