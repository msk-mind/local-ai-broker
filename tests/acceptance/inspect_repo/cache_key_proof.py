#!/usr/bin/env python3
"""Measure broker cache-key behavior for repeated repo requests.

This harness isolates `broker/pkg/cache.KeyForRequest` from the full
inspect_repo pipeline so we can distinguish request-key latency from worker,
indexing, and release-path costs.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _stage_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test User"], check=True)
    (root / "service.py").write_text(
        "def retry_job(job_id):\n    return submit_job(job_id)\n\ndef submit_job(job_id):\n    return job_id\n",
        encoding="utf-8",
    )
    (root / "mcp.go").write_text(
        "package mcp\n\nfunc InspectRepo(query string) string {\n\treturn query\n}\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "init"], check=True)


def _invoke(repo: Path):
    helper = Path(os.environ["CACHE_KEY_HELPER_BIN"])
    started = time.perf_counter()
    completed = subprocess.run(
        [str(helper)],
        cwd=REPO_ROOT,
        env={"CACHE_KEY_HELPER_REPO": str(repo), **dict(**os.environ)},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    payload = json.loads(completed.stdout.strip())
    return {"seconds": round(elapsed, 4), "payload": payload}


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="inspect-repo-cache-key-proof-") as temp_dir:
        root = Path(temp_dir)
        _stage_repo(root)
        helper = root / "cache-key-helper"
        subprocess.run(
            ["go", "build", "-o", str(helper), "tests/acceptance/inspect_repo/cache_key_helper.go"],
            cwd=REPO_ROOT,
            check=True,
        )
        env = os.environ.copy()
        env["CACHE_KEY_HELPER_BIN"] = str(helper)
        old = os.environ.get("CACHE_KEY_HELPER_BIN")
        os.environ["CACHE_KEY_HELPER_BIN"] = str(helper)
        try:
            cold = _invoke(root)
            warm = _invoke(root)
            (root / "service.py").write_text(
                "def retry_job(job_id):\n    value = submit_job(job_id)\n    return value\n\ndef submit_job(job_id):\n    return job_id\n",
                encoding="utf-8",
            )
            partial_dirty = _invoke(root)
        finally:
            if old is None:
                os.environ.pop("CACHE_KEY_HELPER_BIN", None)
            else:
                os.environ["CACHE_KEY_HELPER_BIN"] = old

    summary = {
        "cold": cold,
        "warm": warm,
        "partial_dirty": partial_dirty,
        "checks": {
            "warm_same_key": cold["payload"]["cache_key"] == warm["payload"]["cache_key"],
            "partial_dirty_changes_key": partial_dirty["payload"]["cache_key"] != cold["payload"]["cache_key"],
        },
    }
    summary["ok"] = all(summary["checks"].values())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
