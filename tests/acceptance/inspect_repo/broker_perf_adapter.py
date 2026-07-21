#!/usr/bin/env python3
"""Submit one inspect_repo request through broker-server and print the result JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _request_json(method: str, url: str, *, headers: dict[str, str], body: dict | None = None):
    data = None
    request_headers = dict(headers)
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_succeeded_result(payload: dict | None):
    if not isinstance(payload, dict):
        return None
    if str(payload.get("state") or "") != "succeeded":
        return None
    result = payload.get("result")
    return payload if result is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", default="evidence")
    parser.add_argument("--actor", default="alice")
    parser.add_argument("--role", default="user")
    parser.add_argument("--backend", default="local")
    parser.add_argument("--tier", default="cpu-rag-indexing")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    args = parser.parse_args()

    repo_path = Path(args.repo).expanduser().resolve(strict=False)
    base_url = args.base_url.rstrip("/")
    headers = {
        "X-Broker-Actor": args.actor,
        "X-Broker-Role": args.role,
    }
    submit_body = {
        "task_type": "inspect_repo",
        "input_refs": [
            {
                "type": "repo",
                "uri": repo_path.as_uri(),
                "classification": "internal",
            }
        ],
        "task_params": {
            "query": args.query,
            "mode": args.mode,
        },
        "constraints": {
            "retrieval_token_budget": 16000,
            "evidence_token_budget": 4000,
            "final_pack_token_budget": 2048,
            "synthesis_context_token_budget": 16000,
        },
        "execution_profile": {
            "backend": args.backend,
            "tier": args.tier,
        },
        "output_schema": {
            "name": "repo_inspection_v2",
        },
    }
    extra_task_params_raw = os.environ.get("INSPECT_REPO_EXTRA_TASK_PARAMS", "").strip()
    if extra_task_params_raw:
        try:
            extra_task_params = json.loads(extra_task_params_raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid INSPECT_REPO_EXTRA_TASK_PARAMS: {exc}")
        if not isinstance(extra_task_params, dict):
            raise SystemExit("INSPECT_REPO_EXTRA_TASK_PARAMS must decode to a JSON object")
        submit_body["task_params"].update(extra_task_params)

    try:
        submit = _request_json("POST", base_url + "/v1/jobs", headers=headers, body=submit_body)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"submit failed: {exc.code} {detail}")
    job_id = str(submit.get("job_id") or "")
    if not job_id:
        raise SystemExit(f"submit response missing job_id: {submit!r}")
    released_result = submit.get("released_result")
    if isinstance(released_result, dict) and str(released_result.get("state") or "") == "succeeded" and released_result.get("result") is not None:
        print(json.dumps(released_result))
        return 0

    deadline = time.time() + max(1, args.timeout_seconds)
    last_release = None
    source_job_id = ""
    while time.time() < deadline:
        release = _request_json("GET", base_url + f"/v1/jobs/{urllib.parse.quote(job_id)}/result", headers=headers)
        last_release = release
        succeeded = _extract_succeeded_result(release)
        if succeeded is not None:
            print(json.dumps(succeeded))
            return 0
        state = str(release.get("state") or "")
        if state in {"failed", "cancelled", "timed_out", "preempted"}:
            raise SystemExit(f"job {job_id} ended in state {state}: {json.dumps(release)}")
        if not source_job_id or state == "running":
            job = _request_json("GET", base_url + f"/v1/jobs/{urllib.parse.quote(job_id)}", headers=headers)
            source_job_id = str(job.get("cache_source_job_id") or source_job_id or "")
            if source_job_id:
                source_release = _request_json(
                    "GET",
                    base_url + f"/v1/jobs/{urllib.parse.quote(source_job_id)}/result",
                    headers=headers,
                )
                succeeded = _extract_succeeded_result(source_release)
                if succeeded is not None:
                    print(json.dumps(succeeded))
                    return 0
        time.sleep(max(0.01, args.poll_interval))

    raise SystemExit(f"timed out waiting for job {job_id}: {json.dumps(last_release)}")


if __name__ == "__main__":
    raise SystemExit(main())
