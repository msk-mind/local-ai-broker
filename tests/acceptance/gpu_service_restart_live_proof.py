#!/usr/bin/env python3

import argparse
import hashlib
import http.client
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_SERVICE_SCRIPT = REPO_ROOT / "deploy" / "slurm" / "gpu_service.slurm"
FAKE_RUNTIME = REPO_ROOT / "tests" / "acceptance" / "fake_gpu_runtime.py"


def utc_now():
    return datetime.now(timezone.utc)


def fmt(ts):
    return ts.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run(cmd, *, cwd=None, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, check=True, capture_output=True, text=True)


def wait_for(predicate, timeout_seconds, interval=2.0):
    deadline = time.monotonic() + timeout_seconds
    last = None
    while time.monotonic() < deadline:
        ok, last = predicate()
        if ok:
            return last
        time.sleep(interval)
    raise TimeoutError(last or "timed out")


def write_registry(path: Path, token: str, *, state="starting"):
    created = utc_now()
    record = {
        "id": "gpu-p40-retrieval-live-proof",
        "tier": "p40-retrieval",
        "role": "retrieval",
        "state": state,
        "model_profile": "retrieval-profile",
        "model": "fake-retrieval-model",
        "capabilities": ["embeddings", "index_status", "index_upsert", "faiss_search", "rerank"],
        "context_limit_tokens": 32768,
        "gpu": {"type": "p40", "count": 1},
        "created_at": fmt(created),
        "startup_deadline": fmt(created + timedelta(minutes=15)),
        "lease_expires_at": fmt(created + timedelta(hours=4)),
        "registration_token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
    }
    payload = {
        "schema": "gpu_service_registry_v1",
        "updated_at": fmt(created),
        "records": [record],
        "demands": [],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def write_spec(path: Path, registry_path: Path, token: str, runtime_log: Path, cache_dir: Path):
    payload = {
        "tier": "p40-retrieval",
        "service_id": "gpu-p40-retrieval-live-proof",
        "registry_path": str(registry_path),
        "registration_token": token,
        "heartbeat_interval_seconds": 15,
        "lease_duration_seconds": 4 * 60 * 60,
        "capabilities": ["embeddings", "index_status", "index_upsert", "faiss_search", "rerank"],
        "deployment": {
            "name": "p40-retrieval-live-proof",
            "model": "fake-retrieval-model",
            "quantization": "none",
            "context_limit_tokens": 32768,
            "runtime": shutil.which("python3") or "python3",
            "runtime_args": [
                str(FAKE_RUNTIME),
                "--port={port}",
                "--token={endpoint_token}",
                f"--log-path={runtime_log}",
                "--model={model}",
            ],
        },
        "placement": {
            "partition": "hpc",
            "gpu": {"type": "p40", "count": 1},
            "nodelist": "",
            "constraint": "",
            "qos": "",
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(cache_dir, 0o700)


def submit_service(spec_path: Path, output_log: Path, cache_dir: Path):
    env = os.environ.copy()
    export_parts = [f"BROKER_GPU_SERVICE_SPEC_PATH={spec_path}", f"BROKER_GPU_SERVICE_INDEX_CACHE_DIR={cache_dir}"]
    cmd = [
        "sbatch",
        "--parsable",
        "--job-name",
        "gpu-live-proof",
        "--partition",
        "hpc",
        "--gres",
        "gpu:p40:1",
        "--output",
        str(output_log),
        "--error",
        str(output_log),
        "--export",
        ",".join(["ALL", *export_parts]),
        str(GPU_SERVICE_SCRIPT),
    ]
    return run(cmd, cwd=REPO_ROOT, env=env).stdout.strip()


def cancel_job(job_id: str):
    subprocess.run(["scancel", job_id], cwd=REPO_ROOT, check=False, capture_output=True, text=True)


def wait_job_gone(job_id: str, timeout_seconds: int):
    def probe():
        result = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        state = result.stdout.strip()
        return state == "", state or "still present"

    wait_for(probe, timeout_seconds, interval=2.0)


def wait_record_ready(registry_path: Path, timeout_seconds: int):
    def probe():
        if not registry_path.exists():
            return False, "registry missing"
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        records = payload.get("records") or []
        if not records:
            return False, "no records"
        record = records[0]
        if record.get("state") != "ready":
            return False, f"state={record.get('state')}"
        endpoint = str(record.get("endpoint") or "")
        token = str(((record.get("endpoint_auth") or {}).get("bearer_token")) or "")
        if not endpoint or not token:
            return False, "record not routable"
        return True, record

    return wait_for(probe, timeout_seconds, interval=2.0)


def request_json(method: str, endpoint: str, token: str, path: str, payload=None):
    parsed = urlparse(endpoint)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=30)
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(body))
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    decoded = json.loads(raw.decode("utf-8")) if raw else {}
    return response.status, decoded


def read_runtime_log(path: Path):
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / ".broker-live-tests" / f"gpu-service-restart-live-proof-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "registry.json"
    spec_path = output_dir / "spec.json"
    runtime_log = output_dir / "fake-runtime.log"
    cache_dir = output_dir / "semantic-index-cache"
    slurm_log_1 = output_dir / "slurm-first.log"
    slurm_log_2 = output_dir / "slurm-second.log"
    token = secrets.token_urlsafe(24)

    write_registry(registry_path, token, state="starting")
    write_spec(spec_path, registry_path, token, runtime_log, cache_dir)

    first_job_id = ""
    second_job_id = ""
    first_record = None
    second_record = None
    first_status = None
    first_search = None
    second_status = None
    second_search = None
    try:
        write_spec(spec_path, registry_path, token, runtime_log, cache_dir)
        first_job_id = submit_service(spec_path, slurm_log_1, cache_dir)
        first_record = wait_record_ready(registry_path, args.timeout_seconds)
        first_endpoint = str(first_record["endpoint"])
        first_token = str(first_record["endpoint_auth"]["bearer_token"])
        status, payload = request_json(
            "POST",
            first_endpoint,
            first_token,
            "/v1/indexes/upsert",
            {
                "model": "retrieval-model",
                "model_profile": "retrieval",
                "index_fingerprint": "fp1",
                "document_count": 1,
                "replace": True,
                "finalize": True,
                "documents": [{"id": "chunk_a", "text": "alpha beta"}],
            },
        )
        first_status = {"status": status, "payload": payload}
        if status != 200 or not payload.get("accepted"):
            raise RuntimeError(f"initial upsert failed: {first_status}")
        status, payload = request_json(
            "POST",
            first_endpoint,
            first_token,
            "/v1/search",
            {
                "model": "retrieval-model",
                "model_profile": "retrieval",
                "index_fingerprint": "fp1",
                "query": "alpha query",
                "limit": 1,
            },
        )
        first_search = {"status": status, "payload": payload}
        cancel_job(first_job_id)
        wait_job_gone(first_job_id, min(args.timeout_seconds, 300))

        removed_full_cache_files = []
        for path in list(cache_dir.glob("semantic-index-*.json.gz")):
            removed_full_cache_files.append(str(path))
            path.unlink()

        write_registry(registry_path, token, state="starting")
        write_spec(spec_path, registry_path, token, runtime_log, cache_dir)
        second_job_id = submit_service(spec_path, slurm_log_2, cache_dir)
        second_record = wait_record_ready(registry_path, args.timeout_seconds)
        second_endpoint = str(second_record["endpoint"])
        second_token = str(second_record["endpoint_auth"]["bearer_token"])

        status_started = time.perf_counter()
        status, payload = request_json(
            "POST",
            second_endpoint,
            second_token,
            "/v1/indexes/status",
            {"model_profile": "retrieval", "index_fingerprint": "fp1", "document_count": 1},
        )
        second_status = {
            "status": status,
            "payload": payload,
            "seconds": round(time.perf_counter() - status_started, 6),
        }

        search_started = time.perf_counter()
        status, payload = request_json(
            "POST",
            second_endpoint,
            second_token,
            "/v1/search",
            {
                "model": "retrieval-model",
                "model_profile": "retrieval",
                "index_fingerprint": "fp1",
                "query": "alpha query",
                "limit": 1,
            },
        )
        second_search = {
            "status": status,
            "payload": payload,
            "seconds": round(time.perf_counter() - search_started, 6),
        }

        runtime_events = read_runtime_log(runtime_log)
        restart_events = runtime_events[2:] if len(runtime_events) >= 2 else []
        restart_inputs = [event.get("input") for event in restart_events]
        faiss_files = list(cache_dir.glob("semantic-index-*.faiss"))
        matrix_files = list(cache_dir.glob("semantic-index-*.npy"))
        checks = {
            "cluster_submit_succeeded": bool(first_job_id and second_job_id),
            "first_index_upsert_succeeded": first_status["status"] == 200 and bool(first_status["payload"].get("accepted")),
            "restart_status_ready": second_status["status"] == 200 and second_status["payload"] == {"ready": True, "document_count": 1},
            "restart_search_result_preserved": second_search["status"] == 200 and [item.get("id") for item in (second_search["payload"].get("results") or [])] == ["chunk_a"],
            "persisted_status_sidecar_present": len(list(cache_dir.glob("semantic-index-status-*.json"))) >= 1,
            "persisted_matrix_or_faiss_sidecar_present": bool(faiss_files or matrix_files),
            "restart_succeeds_without_full_cache_file": not list(cache_dir.glob("semantic-index-*.json.gz")),
            "restart_avoids_document_reembedding": restart_inputs in ([], [["alpha query"]]),
        }
        summary = {
            "ok": all(checks.values()),
            "checks": checks,
            "jobs": {"first": first_job_id, "second": second_job_id},
            "first_run": {
                "record": {"endpoint": first_record.get("endpoint"), "slurm_job_id": first_record.get("slurm_job_id")},
                "upsert": first_status,
                "search": first_search,
            },
            "restart": {
                "record": {"endpoint": second_record.get("endpoint"), "slurm_job_id": second_record.get("slurm_job_id")},
                "status": second_status,
                "search": second_search,
                "runtime_events": restart_events,
                "faiss_sidecar_present": len(faiss_files) >= 1,
                "matrix_sidecar_present": len(matrix_files) >= 1,
                "removed_full_cache_files": removed_full_cache_files,
            },
            "artifacts": {
                "output_dir": str(output_dir),
                "registry_path": str(registry_path),
                "runtime_log": str(runtime_log),
                "cache_dir": str(cache_dir),
                "slurm_log_first": str(slurm_log_1),
                "slurm_log_second": str(slurm_log_2),
            },
            "notes": [
                "live Slurm-backed proof: after the full semantic cache file was removed, the restarted GPU-service job on a P40 node still served search from persisted sidecars without re-embedding stored documents"
            ],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        if not summary["ok"]:
            raise SystemExit(1)
    finally:
        if second_job_id:
            cancel_job(second_job_id)
        elif first_job_id:
            cancel_job(first_job_id)


if __name__ == "__main__":
    main()
