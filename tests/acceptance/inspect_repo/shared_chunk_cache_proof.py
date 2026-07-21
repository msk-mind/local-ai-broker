#!/usr/bin/env python3
"""Proof that inspect_repo can reuse shared file-chunk cache across cold local caches."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
WORKER_DIR = REPO_ROOT / "workers" / "rag-compression"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

import inspection_index  # noqa: E402


def stage_repo(root: Path) -> list[dict[str, object]]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "service.py").write_text(
        "def retry_job(job_id):\n    return submit_job(job_id)\n\ndef submit_job(job_id):\n    return job_id\n",
        encoding="utf-8",
    )
    return [{"id": "repo", "type": "repo", "classification": "internal", "path": root}]


def prove() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="inspect-repo-shared-chunk-cache-") as temp_dir:
        temp_root = Path(temp_dir)
        repo_root = temp_root / "repo"
        shared_cache_dir = temp_root / "shared-cache"
        cold_cache_dir = temp_root / "cache-cold"
        fresh_cache_dir = temp_root / "cache-fresh"
        discovered = stage_repo(repo_root)
        fingerprint = "sha256:shared-proof"

        with mock.patch.dict(os.environ, {"BROKER_REPO_INSPECTION_SHARED_CACHE_DIR": str(shared_cache_dir)}, clear=False):
            first_chunks, first_stats = inspection_index.build_syntax_chunks(
                discovered,
                cache_dir=cold_cache_dir,
                repository_state_fingerprint=fingerprint,
                return_diagnostics=True,
            )
            _, _, _, build_config_digest = inspection_index._snapshot_build_context(
                discovered,
                None,
                cache_dir=fresh_cache_dir,
            )
            shared_snapshot_chunks = inspection_index.load_cached_chunk_snapshot(
                fresh_cache_dir,
                repository_state_fingerprint=fingerprint,
                build_config_digest=build_config_digest,
            )

            original_read_text = Path.read_text

            def guarded_read_text(path_self, *args, **kwargs):
                if path_self == repo_root / "service.py":
                    raise AssertionError("shared chunk cache should avoid reopening source file on fresh local cache")
                return original_read_text(path_self, *args, **kwargs)

            with mock.patch.object(Path, "read_text", autospec=True, side_effect=guarded_read_text):
                second_chunks, second_stats = inspection_index.build_syntax_chunks(
                    discovered,
                    cache_dir=fresh_cache_dir,
                    repository_state_fingerprint=fingerprint,
                    return_diagnostics=True,
                )
            ok = (
                first_stats["rebuilt_files"] >= 1
                and second_stats["reused_files"] >= 1
                and second_stats["rebuilt_files"] == 0
                and first_chunks == second_chunks
                and first_chunks == shared_snapshot_chunks
            )
            first_fp = inspection_index.inspection_index_fingerprint(fingerprint, first_chunks)
            _, _, first_lexical_stats = inspection_index.ensure_lexical_index(first_chunks, cold_cache_dir, first_fp)
            shutil_second = fresh_cache_dir / "lexical-working.sqlite3"
            shutil_second.unlink(missing_ok=True)
            with mock.patch.object(
                inspection_index,
                "_rebuild_working_lexical_index",
                side_effect=AssertionError("shared lexical index should avoid rebuild on fresh local cache"),
            ):
                _, second_lexical_hit, second_lexical_stats = inspection_index.ensure_lexical_index(
                    second_chunks, fresh_cache_dir, first_fp
                )
            ok = ok and bool(second_lexical_hit) and bool(second_lexical_stats.get("shared_restore"))
            return {
                "ok": ok,
                "shared_cache_dir": str(shared_cache_dir),
                "cold_cache_dir": str(cold_cache_dir),
                "fresh_cache_dir": str(fresh_cache_dir),
                "first_stats": first_stats,
                "shared_snapshot_hit": shared_snapshot_chunks is not None,
                "second_stats": second_stats,
                "first_lexical_stats": first_lexical_stats,
                "second_lexical_hit": second_lexical_hit,
                "second_lexical_stats": second_lexical_stats,
                "chunk_count": len(second_chunks),
            }


def main() -> int:
    result = prove()
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
