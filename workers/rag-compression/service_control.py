"""Authenticated demand spool client for scheduler-owned GPU services.

The inspection worker only writes a signed demand file and polls its signed
response.  It never invokes Slurm or mutates the shared service registry.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gpu_client import GPUServiceError


REQUEST_SCHEMA = "gpu_service_demand_v1"
RESPONSE_SCHEMA = "gpu_service_demand_response_v1"
FAILURE_REPORT_SCHEMA = "gpu_service_failure_report_v1"
REPORTABLE_SERVICE_FAILURES = {
    "availability",
    "timeout",
    "service_failure",
    "authentication",
    "oom",
    "heartbeat_lost",
    "endpoint_unhealthy",
}


def _b64url(value):
    return base64.urlsafe_b64encode(str(value or "").encode("utf-8")).decode("ascii").rstrip("=")


def _timestamp(value=None):
    value = (value or datetime.now(timezone.utc)).astimezone(timezone.utc)
    base = value.strftime("%Y-%m-%dT%H:%M:%S")
    fraction = f"{value.microsecond:06d}".rstrip("0")
    return base + (f".{fraction}" if fraction else "") + "Z"


def _sign(token, fields):
    message = "\0".join(str(value) for value in fields).encode("utf-8")
    return hmac.new(str(token).encode("utf-8"), message, hashlib.sha256).hexdigest()


def request_signature(token, payload):
    return _sign(
        token,
        [
            payload.get("schema", ""),
            payload.get("request_id", ""),
            payload.get("tier", ""),
            payload.get("failure_category", ""),
            _b64url(payload.get("reason", "")),
            payload.get("requested_at", ""),
            payload.get("deadline", ""),
            payload.get("nonce", ""),
        ],
    )


def response_signature(token, payload):
    fields = [
        payload.get("schema", ""),
        payload.get("request_id", ""),
        payload.get("demand_id", ""),
        payload.get("state", ""),
        payload.get("failure_category", ""),
        payload.get("updated_at", ""),
        _b64url(payload.get("error", "")),
    ]
    service = payload.get("service")
    if isinstance(service, dict):
        auth = service.get("endpoint_auth") or {}
        gpu = service.get("gpu") or {}
        capabilities = service.get("capabilities") or []
        fields.extend(
            [
                service.get("id", ""),
                service.get("tier", ""),
                service.get("endpoint", ""),
                auth.get("type", ""),
                auth.get("bearer_token", ""),
                service.get("model_profile", ""),
                service.get("model", ""),
                ",".join(sorted(str(value) for value in capabilities)),
                int(service.get("context_limit_tokens") or 0),
                gpu.get("type", ""),
                int(gpu.get("count") or 0),
                service.get("slurm_job_id", ""),
                service.get("heartbeat_at", ""),
                service.get("lease_expires_at", ""),
            ]
        )
    diagnostics = payload.get("service_diagnostics")
    if isinstance(diagnostics, dict):
        gpu = diagnostics.get("gpu") or {}
        fields.extend(
            [
                "service_diagnostics",
                diagnostics.get("tier", ""),
                diagnostics.get("slurm_job_id", ""),
                gpu.get("type", ""),
                int(gpu.get("count") or 0),
                diagnostics.get("model_profile", ""),
            ]
        )
    return _sign(token, fields)


def failure_report_signature(token, payload):
    return _sign(
        token,
        [
            payload.get("schema", ""),
            payload.get("report_id", ""),
            payload.get("service_id", ""),
            payload.get("tier", ""),
            payload.get("failure_category", ""),
            _b64url(payload.get("reason", "")),
            payload.get("reported_at", ""),
            payload.get("nonce", ""),
        ],
    )


def verify_response(token, payload, request_id):
    if not isinstance(payload, dict) or payload.get("schema") != RESPONSE_SCHEMA:
        raise GPUServiceError("service_failure", "GPU demand response has an invalid schema")
    if payload.get("request_id") != request_id:
        raise GPUServiceError("service_failure", "GPU demand response request id does not match")
    supplied = str(payload.get("signature") or "")
    expected = response_signature(token, payload)
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise GPUServiceError("authentication", "GPU demand response signature is invalid", retryable=False)
    return payload


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    tmp = path.parent / f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(6)}"
    descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        tmp.unlink(missing_ok=True)


def report_p40_service_failure(execution_plan, service, failure_category, reason):
    """Emit an authenticated lease-failure action without scheduling or waiting.

    Returns True only after the mode-0600 action file is atomically published.
    Invalid, non-P40, or unconfigured reports return False so reporting never
    masks the original inference failure.
    """

    execution_plan = execution_plan or {}
    service = service or {}
    request_dir = execution_plan.get("gpu_service_request_path") or os.environ.get(
        "BROKER_GPU_SERVICE_CONTROL_REQUEST_DIR"
    )
    token = execution_plan.get("gpu_service_control_token") or os.environ.get(
        "BROKER_GPU_SERVICE_CONTROL_TOKEN"
    )
    service_id = str(service.get("id") or "") if isinstance(service, dict) else ""
    tier = str(service.get("tier") or "") if isinstance(service, dict) else ""
    category = str(failure_category or "service_failure")
    if (
        not request_dir
        or not token
        or not service_id
        or tier not in {"p40-retrieval", "p40-synthesis"}
        or category not in REPORTABLE_SERVICE_FAILURES
    ):
        return False
    report_id = "gpu-failure-" + secrets.token_hex(12)
    payload = {
        "schema": FAILURE_REPORT_SCHEMA,
        "report_id": report_id,
        "service_id": service_id,
        "tier": tier,
        "failure_category": category,
        "reason": str(reason or "request worker reported GPU service failure")[:4096],
        "reported_at": _timestamp(),
        "nonce": secrets.token_hex(16),
    }
    payload["signature"] = failure_report_signature(token, payload)
    try:
        _atomic_write_json(Path(request_dir) / f"{report_id}.failure.json", payload)
    except OSError:
        return False
    return True


def _parse_timestamp(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def validate_ready_service(service, tier, health_interval_seconds):
    if not isinstance(service, dict) or service.get("tier") != tier:
        raise GPUServiceError("service_failure", "GPU demand response returned the wrong tier")
    required = ("endpoint", "model_profile", "model", "heartbeat_at", "lease_expires_at")
    if any(not service.get(key) for key in required):
        raise GPUServiceError("service_failure", "GPU demand response returned an incomplete service record")
    auth = service.get("endpoint_auth") or {}
    if auth.get("type") != "bearer" or not auth.get("bearer_token"):
        raise GPUServiceError("authentication", "GPU demand response is missing bearer authentication", retryable=False)
    now = datetime.now(timezone.utc)
    lease = _parse_timestamp(service.get("lease_expires_at"))
    heartbeat = _parse_timestamp(service.get("heartbeat_at"))
    if lease is None or lease <= now:
        raise GPUServiceError("availability", "GPU demand response service lease is expired")
    if heartbeat is None:
        raise GPUServiceError("service_failure", "GPU demand response heartbeat is invalid")
    max_age = max(1.0, float(health_interval_seconds or 30)) * 3
    if (now - heartbeat).total_seconds() > max_age:
        raise GPUServiceError("service_failure", "GPU demand response heartbeat is stale")
    record = dict(service)
    record["state"] = "ready"
    return record


def request_service(execution_plan, tier, failure_category, reason):
    execution_plan = execution_plan or {}
    request_dir = execution_plan.get("gpu_service_request_path")
    token = execution_plan.get("gpu_service_control_token")
    if not request_dir or not token:
        raise GPUServiceError("availability", "GPU service demand control is not configured")
    timeout_seconds = float(execution_plan.get("gpu_service_startup_timeout_seconds") or 300)
    health_interval = float(execution_plan.get("gpu_service_health_interval_seconds") or 30)
    now = datetime.now(timezone.utc)
    request_id = "req_" + secrets.token_hex(16)
    payload = {
        "schema": REQUEST_SCHEMA,
        "request_id": request_id,
        "tier": str(tier),
        "failure_category": str(failure_category or "availability"),
        "reason": str(reason or "GPU service required"),
        "requested_at": _timestamp(now),
        "deadline": _timestamp(now + timedelta(seconds=timeout_seconds)),
        "nonce": secrets.token_urlsafe(24),
    }
    payload["signature"] = request_signature(token, payload)
    request_path = Path(request_dir) / f"{request_id}.request.json"
    response_path = Path(request_dir) / f"{request_id}.response.json"
    _atomic_write_json(request_path, payload)

    try:
        monotonic_deadline = time.monotonic() + timeout_seconds
        poll_seconds = min(max(0.05, health_interval / 4), 2.0)
        while time.monotonic() < monotonic_deadline:
            if response_path.exists():
                try:
                    response = json.loads(response_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise GPUServiceError("service_failure", "GPU demand response could not be read") from exc
                response = verify_response(token, response, request_id)
                state = str(response.get("state") or "")
                if state == "ready":
                    return validate_ready_service(response.get("service"), tier, health_interval)
                if state == "failed":
                    error = str(response.get("error") or "GPU service demand failed")
                    category = str(response.get("failure_category") or "service_failure")
                    diagnostics = response.get("service_diagnostics")
                    if isinstance(diagnostics, dict) and str(diagnostics.get("tier") or "") != str(tier):
                        raise GPUServiceError(
                            "service_failure",
                            "GPU demand failure diagnostics returned the wrong tier",
                        )
                    raise GPUServiceError(category, error, service_diagnostics=diagnostics)
                if state not in {"pending", "launching"}:
                    raise GPUServiceError("service_failure", "GPU demand response has an invalid state")
            time.sleep(min(poll_seconds, max(0.0, monotonic_deadline - time.monotonic())))
        raise GPUServiceError("queue_delay", f"GPU service demand for {tier} exceeded its startup deadline")
    finally:
        # Requests and responses are single-consumer capability files. Remove
        # both on success, terminal failure, authentication failure, or timeout.
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)


def requester_from_execution_plan(execution_plan):
    execution_plan = execution_plan or {}
    request_path = execution_plan.get("gpu_service_request_path") or os.environ.get(
        "BROKER_GPU_SERVICE_CONTROL_REQUEST_DIR"
    )
    control_token = execution_plan.get("gpu_service_control_token") or os.environ.get(
        "BROKER_GPU_SERVICE_CONTROL_TOKEN"
    )
    if not request_path or not control_token:
        return None
    protected_plan = dict(execution_plan)
    protected_plan["gpu_service_request_path"] = request_path
    protected_plan["gpu_service_control_token"] = control_token

    def requester(tier, failure_category, reason):
        return request_service(protected_plan, tier, failure_category, reason)

    return requester


def failure_reporter_from_execution_plan(execution_plan):
    execution_plan = execution_plan or {}
    request_path = execution_plan.get("gpu_service_request_path") or os.environ.get(
        "BROKER_GPU_SERVICE_CONTROL_REQUEST_DIR"
    )
    control_token = execution_plan.get("gpu_service_control_token") or os.environ.get(
        "BROKER_GPU_SERVICE_CONTROL_TOKEN"
    )
    if not request_path or not control_token:
        return None
    protected_plan = dict(execution_plan)
    protected_plan["gpu_service_request_path"] = request_path
    protected_plan["gpu_service_control_token"] = control_token

    def reporter(service, failure_category, reason):
        return report_p40_service_failure(protected_plan, service, failure_category, reason)

    return reporter
