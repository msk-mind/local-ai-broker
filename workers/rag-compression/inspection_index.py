"""Repository chunking and persistent indexes for ``repo_inspection_v2``.

This module deliberately contains no model inference.  It is safe to run on a
CPU worker: it discovers files, creates symbol-aware chunks, maintains the
lexical index, and reads/writes embedding cache metadata produced by a GPU
service.
"""

from __future__ import annotations

import ast
import bisect
import contextlib
import fnmatch
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import sqlite3
import subprocess
import time
from collections import defaultdict
from pathlib import Path, PurePosixPath


DEFAULT_IGNORE_DIRS = {
    ".git",
    ".broker",
    ".broker-live-tests",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "site-packages",
    "build",
    "dist",
}

DEFAULT_IGNORE_FILE_GLOBS = {
    "slurm-*.out",
}

LANGUAGES = {
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".h": "c",
    ".hpp": "cpp",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".py": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".sh": "shell",
    ".slurm": "shell",
    ".sql": "sql",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".txt": "text",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".ini": "config",
    ".conf": "config",
}

MAX_CHUNK_BYTES = 32 * 1024
INDEX_SCHEMA_VERSION = "repo-inspection-chunks-v2"
FILE_CHUNK_CACHE_SCHEMA = "repo-inspection-file-chunks-v2"
# Prefer inventory-delta refresh for medium repositories too. The cached
# scoped-status snapshot is still cheaper than re-enumerating the full tree for
# content-only edits, and the fallback remains available when inventory changes.
SMALL_REPO_INVENTORY_DELTA_THRESHOLD = 512
SMALL_GIT_FINGERPRINT_FASTPATH_FILE_THRESHOLD = 96
FILE_CHUNK_WORKING_MANIFEST_SCHEMA = "repo-inspection-file-chunk-manifest-v2"
FILE_CHUNK_SNAPSHOT_SCHEMA = "repo-inspection-file-chunk-snapshot-v1"
FILE_CHUNK_SNAPSHOT_METADATA_SCHEMA = "repo-inspection-file-chunk-snapshot-metadata-v1"
SHARED_FILE_CHUNK_STATE_MANIFEST_SCHEMA = "repo-inspection-shared-file-chunk-state-v1"
SYMBOL_MARKER_CACHE_SCHEMA = "repo-inspection-symbol-markers-v1"
DISCOVERY_WORKING_MANIFEST_SCHEMA = "repo-inspection-discovery-manifest-v1"
GIT_FINGERPRINT_MANIFEST_SCHEMA = "repo-inspection-git-fingerprint-v1"
GIT_FILE_SIGNATURE_MANIFEST_SCHEMA = "repo-inspection-git-file-signatures-v1"
METADATA_FINGERPRINT_MANIFEST_SCHEMA = "repo-inspection-metadata-fingerprint-v1"
LEXICAL_INDEX_SCHEMA = "repo-inspection-lexical-v3"
LEXICAL_WORKING_MANIFEST_SCHEMA = "repo-inspection-lexical-manifest-v1"
LEXICAL_HELPER_CACHE_LIMIT = 8
LEXICAL_RESULT_CACHE_LIMIT = 32
QUERY_FEATURE_CACHE_LIMIT = 64
SMALL_CORPUS_LEXICAL_FTS_THRESHOLD = 32
_LEXICAL_HELPER_CACHE = {}
_LEXICAL_RESULT_CACHE = {}
_QUERY_FEATURE_CACHE = {}
_SNAPSHOT_METADATA_CACHE = {}
_FILE_CHUNK_SNAPSHOT_MEMORY_CACHE = {}
_DISCOVERY_WORKING_MANIFEST_CACHE = {}
_FILE_CHUNK_WORKING_MANIFEST_CACHE = {}
_GIT_FILE_SIGNATURE_MANIFEST_CACHE = {}
_GIT_FINGERPRINT_MANIFEST_CACHE = {}
_LEXICAL_WORKING_MANIFEST_CACHE = {}
_DISCOVERY_RECORD_PROCESS_CACHE = {}
_DISCOVERY_EXACT_PROCESS_CACHE = {}
_SHARED_FILE_CHUNK_STATE_ENTRY_MEMORY_CACHE = {}
_FILE_CHUNK_STATE_BUNDLE_MEMORY_CACHE = {}
_FILE_CHUNK_CACHE_PAYLOAD_MEMORY_CACHE = {}
_SYMBOL_MARKER_MEMORY_CACHE = {}
_PATH_SMALL_PAYLOAD_MEMORY_CACHE = {}
_SYMBOL_MARKER_MEMORY_CACHE_LIMIT = 256
_SNAPSHOT_METADATA_CACHE_LIMIT = 64
_FILE_CHUNK_SNAPSHOT_MEMORY_CACHE_LIMIT = 32
_DISCOVERY_WORKING_MANIFEST_CACHE_LIMIT = 64
_FILE_CHUNK_WORKING_MANIFEST_CACHE_LIMIT = 64
_GIT_FILE_SIGNATURE_MANIFEST_CACHE_LIMIT = 128
_GIT_FINGERPRINT_MANIFEST_CACHE_LIMIT = 128
_LEXICAL_WORKING_MANIFEST_CACHE_LIMIT = 64
_DISCOVERY_RECORD_PROCESS_CACHE_LIMIT = 256
_DISCOVERY_EXACT_PROCESS_CACHE_LIMIT = 256
_SHARED_FILE_CHUNK_STATE_ENTRY_MEMORY_CACHE_LIMIT = 256
_FILE_CHUNK_STATE_BUNDLE_MEMORY_CACHE_LIMIT = 256
_FILE_CHUNK_CACHE_PAYLOAD_MEMORY_CACHE_LIMIT = 128
_PATH_SMALL_PAYLOAD_MEMORY_CACHE_LIMIT = 256
_PATH_SMALL_PAYLOAD_CACHE_MAX_BYTES = 1 << 20
_GIT_TOP_CACHE = {}
_GIT_TOP_CACHE_LIMIT = 256
_DEFAULT_GIT_PROBE_CACHE = {}
_PRIVATE_CACHE_DIR_READY = set()
_PROCESS_CACHE_CONTAINERS = (
    _LEXICAL_HELPER_CACHE, _LEXICAL_RESULT_CACHE, _QUERY_FEATURE_CACHE,
    _SNAPSHOT_METADATA_CACHE, _FILE_CHUNK_SNAPSHOT_MEMORY_CACHE,
    _DISCOVERY_WORKING_MANIFEST_CACHE, _FILE_CHUNK_WORKING_MANIFEST_CACHE,
    _GIT_FILE_SIGNATURE_MANIFEST_CACHE, _GIT_FINGERPRINT_MANIFEST_CACHE,
    _LEXICAL_WORKING_MANIFEST_CACHE, _DISCOVERY_RECORD_PROCESS_CACHE,
    _DISCOVERY_EXACT_PROCESS_CACHE, _SHARED_FILE_CHUNK_STATE_ENTRY_MEMORY_CACHE,
    _FILE_CHUNK_STATE_BUNDLE_MEMORY_CACHE, _FILE_CHUNK_CACHE_PAYLOAD_MEMORY_CACHE,
    _SYMBOL_MARKER_MEMORY_CACHE, _PATH_SMALL_PAYLOAD_MEMORY_CACHE,
    _GIT_TOP_CACHE, _DEFAULT_GIT_PROBE_CACHE, _PRIVATE_CACHE_DIR_READY,
)


def reset_process_caches():
    """Clear process-local inspection caches between isolated test runs."""
    for value in _PROCESS_CACHE_CONTAINERS:
        value.clear()


class ChunkList(list):
    pass


SYMBOL_PATTERNS = {
    "go": re.compile(r"^\s*(?:func\s+(?:\([^)]*\)\s*)?|type\s+)([A-Za-z_][\w]*)", re.MULTILINE),
    "javascript": re.compile(
        r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class)\s+([A-Za-z_$][\w$]*)|"
        r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=",
        re.MULTILINE,
    ),
    "typescript": re.compile(
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class|interface|type|enum)\s+([A-Za-z_$][\w$]*)|"
        r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=",
        re.MULTILINE,
    ),
    "rust": re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait|impl)\s+([A-Za-z_][\w]*)", re.MULTILINE),
    "java": re.compile(r"^\s*(?:public|protected|private|static|final|abstract|synchronized|native|\s)+\s*(?:class|interface|enum|[\w<>\[\],?]+)\s+([A-Za-z_][\w]*)\s*(?:\(|\{|extends|implements)", re.MULTILINE),
    "c": re.compile(r"^\s*(?:[A-Za-z_]\w*[\s*]+)+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{", re.MULTILINE),
    "cpp": re.compile(r"^\s*(?:template\s*<[^>]+>\s*)?(?:[A-Za-z_:~]\w*[\s:*&<>]+)+([A-Za-z_:~]\w*)\s*\([^;]*\)\s*(?:const\s*)?\{", re.MULTILINE),
    "ruby": re.compile(r"^\s*(?:def|class|module)\s+([A-Za-z_][\w:!?=]*)", re.MULTILINE),
    "shell": re.compile(r"^\s*(?:function\s+)?([A-Za-z_][\w]*)\s*\(\s*\)\s*\{", re.MULTILINE),
}

PYTHON_TOP_LEVEL_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)(?:(?:async\s+)?def|class)\s+(?P<name>[A-Za-z_][\w]*)",
    re.MULTILINE,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8", errors="replace"))


def estimate_tokens(value: str) -> int:
    return max(1, len(value) // 4)


def semantic_chunk_signature(chunk) -> str:
    return json.dumps(
        {
            "chunk_id": str(chunk.get("chunk_id") or ""),
            "content_hash": str(chunk.get("content_hash") or ""),
            "path": str(chunk.get("path") or ""),
            "line_start": int(chunk.get("line_start") or 0),
            "line_end": int(chunk.get("line_end") or 0),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def language_for_path(path: Path) -> str:
    if path.name == "go.mod":
        return "go-module"
    if path.name == ".env" or path.name.endswith(".env.example"):
        return "config"
    return LANGUAGES.get(path.suffix.lower(), "")


def _should_skip(path: Path, root: Path, ignored: set[str], ignored_paths=None) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    if any(part in ignored for part in rel_parts):
        return True
    basename = path.name
    if basename and any(fnmatch.fnmatch(basename, pattern) for pattern in DEFAULT_IGNORE_FILE_GLOBS):
        return True
    if ignored_paths:
        resolved = path if path.is_absolute() else path.resolve(strict=False)
        for ignored_path in ignored_paths:
            if ignored_path is None:
                continue
            if resolved == ignored_path or ignored_path in resolved.parents:
                return True
    return False


def _rg_file_list(root: Path, ignored: set[str], ignored_paths=None):
    args = ["rg", "--files", "--hidden", "--no-ignore"]
    for name in sorted(ignored):
        args.extend(["-g", f"!{name}/**"])
        args.extend(["-g", f"!**/{name}/**"])
    for ignored_path in sorted({str(path.relative_to(root)) for path in (ignored_paths or []) if root in path.parents}, key=str):
        args.extend(["-g", f"!{ignored_path}/**"])
        args.extend(["-g", f"!**/{ignored_path}/**"])
    args.append(str(root))
    try:
        output = subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            text=True,
        ).stdout
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    candidates = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve(strict=False)
        candidates.append(candidate)
    return candidates


def _git_file_list(root: Path, ignored: set[str], ignored_paths=None, *, git_probe_cache=None, include_tracked_signatures=True):
    top = _git_top(root)
    if top is None:
        return None
    root_resolved = root.resolve(strict=False)
    top_resolved = top.resolve(strict=False)
    try:
        scope_rel = root_resolved.relative_to(top_resolved).as_posix()
    except ValueError:
        return None
    tracked_args = ["git", "-C", str(top), "--no-optional-locks", "ls-files", "--cached"]
    combined_args = ["git", "-C", str(top), "--no-optional-locks", "ls-files", "--cached"]
    if include_tracked_signatures:
        tracked_args.append("-s")
        combined_args.append("-s")
    combined_args.extend(["--others", "--exclude-standard", "-z"])
    tracked_args.append("-z")
    if scope_rel not in {"", "."}:
        tracked_args.extend(["--", scope_rel])
        combined_args.extend(["--", scope_rel])
    cached_status_snapshot = _cached_scoped_status_snapshot(top, scope_rel, git_probe_cache=git_probe_cache)
    cached_status_output = (
        cached_status_snapshot.get("output")
        if isinstance(cached_status_snapshot, dict)
        else _cached_scoped_status_output(top, scope_rel, git_probe_cache=git_probe_cache)
    )
    if cached_status_output is None:
        try:
            tracked_output = subprocess.run(
                combined_args,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=30,
                text=False,
            ).stdout
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            return None
    else:
        try:
            tracked_output = subprocess.run(
                tracked_args,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=30,
                text=False,
            ).stdout
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            return None
    candidates = []
    tracked_signature_bucket = _git_probe_cache_bucket(git_probe_cache, "tracked_blob_signatures")
    tracked_signature_map = {}
    for raw_entry in tracked_output.split(b"\0"):
        if not raw_entry:
            continue
        try:
            if include_tracked_signatures:
                meta, rel_bytes = raw_entry.split(b"\t", 1)
                _mode, blob_oid, _stage = meta.decode("utf-8", errors="replace").split()
                rel = rel_bytes.decode("utf-8", errors="surrogateescape")
            else:
                rel = raw_entry.decode("utf-8", errors="surrogateescape")
            if not language_for_path(Path(rel)):
                continue
            if include_tracked_signatures:
                tracked_signature_map[rel] = f"git:{blob_oid}"
            candidate = top_resolved / rel
            if not _should_skip(candidate, root, ignored, ignored_paths=ignored_paths):
                candidates.append(candidate)
            continue
        except ValueError:
            if cached_status_output is not None:
                continue
            rel = raw_entry.decode("utf-8", errors="surrogateescape")
            if not language_for_path(Path(rel)):
                continue
            candidate = top_resolved / rel
            if not _should_skip(candidate, root, ignored, ignored_paths=ignored_paths):
                candidates.append(candidate)
    if include_tracked_signatures and tracked_signature_bucket is not None and tracked_signature_map:
        tracked_signature_bucket[str(top_resolved)] = tracked_signature_map
    if cached_status_output is not None:
        parsed_entries = (
            cached_status_snapshot.get("parsed_entries")
            if isinstance(cached_status_snapshot, dict)
            else _parse_git_status_entries(cached_status_output)
        )
        for entry in parsed_entries:
            if entry.get("code")[:1] != "?":
                continue
            for rel in entry.get("paths") or ():
                if not language_for_path(Path(rel)):
                    continue
                candidate = top_resolved / rel
                if not _should_skip(candidate, root, ignored, ignored_paths=ignored_paths):
                    candidates.append(candidate)
    return candidates


def _iter_source_candidates(root: Path, ignored: set[str], ignored_paths=None, *, git_probe_cache=None, include_tracked_signatures=True):
    candidates = _git_file_list(
        root,
        ignored,
        ignored_paths=ignored_paths,
        git_probe_cache=git_probe_cache,
        include_tracked_signatures=include_tracked_signatures,
    )
    if candidates is None:
        candidates = _rg_file_list(root, ignored, ignored_paths=ignored_paths)
    if candidates is None:
        candidates = sorted(root.rglob("*"))
    for candidate in candidates:
        if _should_skip(candidate, root, ignored, ignored_paths=ignored_paths) or candidate.is_symlink() or not candidate.is_file():
            continue
        yield candidate


def _discovery_manifest_path(cache_dir: Path):
    return _private_cache_dir(Path(cache_dir)) / "discovery-working-manifest.json"


def _shared_discovery_manifest_path(*, create=False):
    cache_root = _shared_repo_inspection_cache_root(create=create)
    if cache_root is None:
        return None
    return cache_root / "discovery-working-manifest.json"


def _discovery_fingerprint_manifest_path(cache_dir: Path, repository_state_fingerprint: str):
    safe_repository = str(repository_state_fingerprint or "").replace(":", "_")
    cache_root = _private_cache_dir(Path(cache_dir) / "discovery-by-fingerprint")
    return cache_root / f"{safe_repository}.json"


def _shared_discovery_fingerprint_manifest_path(repository_state_fingerprint: str, *, create=False):
    cache_root = _shared_repo_inspection_cache_root(create=create)
    if cache_root is None:
        return None
    cache_root = cache_root / "discovery-by-fingerprint"
    try:
        cache_root = _private_cache_dir(cache_root) if create else cache_root.expanduser().absolute()
    except OSError:
        return None
    safe_repository = str(repository_state_fingerprint or "").replace(":", "_")
    return cache_root / f"{safe_repository}.json"


def _load_discovery_working_manifest(path: Path):
    if path is None:
        return None
    cache_key = _snapshot_metadata_cache_key(path)
    if cache_key is not None:
        cached = _DISCOVERY_WORKING_MANIFEST_CACHE.get(cache_key)
        if cached is not None:
            _DISCOVERY_WORKING_MANIFEST_CACHE.pop(cache_key, None)
            _DISCOVERY_WORKING_MANIFEST_CACHE[cache_key] = cached
            return {
                str(root_key): {
                    "git_top": str(record.get("git_top") or ""),
                    "scope_rel": str(record.get("scope_rel") or ""),
                    "scope_oid": str(record.get("scope_oid") or ""),
                    "repository_state_fingerprint": str(record.get("repository_state_fingerprint") or ""),
                    "filter_key": str(record.get("filter_key") or ""),
                    "files": [str(item) for item in (record.get("files") or ()) if isinstance(item, str)],
                    "dir_signatures": {
                        str(key): str(value)
                        for key, value in dict(record.get("dir_signatures") or {}).items()
                        if isinstance(key, str)
                    },
                }
                for root_key, record in cached.items()
                if isinstance(record, dict)
            }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        path.unlink(missing_ok=True)
        return None
    if not isinstance(payload, dict) or payload.get("schema") != DISCOVERY_WORKING_MANIFEST_SCHEMA:
        return None
    roots = payload.get("roots")
    if not isinstance(roots, dict):
        return None
    normalized = {}
    for root_key, record in roots.items():
        if not isinstance(record, dict):
            return None
        files = record.get("files")
        if not isinstance(files, list):
            return None
        normalized[str(root_key)] = {
            "git_top": str(record.get("git_top") or ""),
            "scope_rel": str(record.get("scope_rel") or ""),
            "scope_oid": str(record.get("scope_oid") or ""),
            "repository_state_fingerprint": str(record.get("repository_state_fingerprint") or ""),
            "filter_key": str(record.get("filter_key") or ""),
            "files": [str(item) for item in files if isinstance(item, str)],
            "dir_signatures": {
                str(key): str(value)
                for key, value in dict(record.get("dir_signatures") or {}).items()
                if isinstance(key, str)
            },
        }
    if cache_key is not None:
        _DISCOVERY_WORKING_MANIFEST_CACHE.pop(cache_key, None)
        _DISCOVERY_WORKING_MANIFEST_CACHE[cache_key] = {
            str(root_key): {
                "git_top": str(record.get("git_top") or ""),
                "scope_rel": str(record.get("scope_rel") or ""),
                "scope_oid": str(record.get("scope_oid") or ""),
                "repository_state_fingerprint": str(record.get("repository_state_fingerprint") or ""),
                "filter_key": str(record.get("filter_key") or ""),
                "files": [str(item) for item in (record.get("files") or ()) if isinstance(item, str)],
                "dir_signatures": {
                    str(key): str(value)
                    for key, value in dict(record.get("dir_signatures") or {}).items()
                    if isinstance(key, str)
                },
            }
            for root_key, record in normalized.items()
        }
        while len(_DISCOVERY_WORKING_MANIFEST_CACHE) > _DISCOVERY_WORKING_MANIFEST_CACHE_LIMIT:
            _DISCOVERY_WORKING_MANIFEST_CACHE.pop(next(iter(_DISCOVERY_WORKING_MANIFEST_CACHE)))
    return normalized


def _load_preferred_discovery_working_manifest(local_path: Path | None, shared_path: Path | None):
    local = _load_discovery_working_manifest(local_path)
    if local is not None:
        return local
    return _load_discovery_working_manifest(shared_path)


def _write_discovery_working_manifest(path: Path, roots):
    payload_bytes = json.dumps(
        {
            "schema": DISCOVERY_WORKING_MANIFEST_SCHEMA,
            "roots": roots,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if not _path_bytes_equal(path, payload_bytes):
        _atomic_private_bytes(path, payload_bytes)


def _write_preferred_discovery_working_manifest(local_path: Path | None, shared_path: Path | None, roots):
    if local_path is not None:
        _write_discovery_working_manifest(local_path, roots)
    if shared_path is not None and shared_path != local_path:
        if local_path is not None and local_path.exists():
            _clone_private_cache_file(local_path, shared_path)
        else:
            _write_discovery_working_manifest(shared_path, roots)


def _git_scope_rel(root: Path, top: Path):
    try:
        rel = root.resolve(strict=False).relative_to(top).as_posix()
    except ValueError:
        return None
    return rel or "."


def _discovery_filter_key(root: Path, ignored: set[str], ignored_paths=None):
    root = root.resolve(strict=False)
    relative_ignored_paths = sorted(
        str(path.relative_to(root))
        for path in (ignored_paths or ())
        if root in path.parents
    )
    payload = {
        "root": str(root),
        "ignored": sorted(str(name) for name in ignored),
        "ignored_paths": relative_ignored_paths,
    }
    return f"sha256:{sha256_text(json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True))}"


def _directory_state_signature(path: Path):
    try:
        stat = path.stat()
    except OSError:
        return "missing"
    if not path.exists() or not path.is_dir():
        return "missing"
    return f"meta:{int(stat.st_mtime_ns)}:{int(getattr(stat, 'st_ctime_ns', 0))}"


def _directory_signatures_for_files(root: Path, relative_paths):
    root = root.resolve(strict=False)
    directories = {root}
    for rel in relative_paths or ():
        try:
            candidate = (root / rel).resolve(strict=False)
        except OSError:
            candidate = root / rel
        current = candidate.parent
        while True:
            directories.add(current)
            if current == root:
                break
            parent = current.parent
            if parent == current:
                break
            current = parent
    signatures = {}
    for directory in sorted(directories):
        try:
            rel = directory.relative_to(root).as_posix()
        except ValueError:
            continue
        signatures[rel or "."] = _directory_state_signature(directory)
    return signatures


def _cached_non_git_source_files(root: Path, ignored: set[str], ignored_paths=None, previous_record=None):
    if not previous_record:
        return None, None
    if str(previous_record.get("git_top") or "").strip():
        return None, None
    filter_key = _discovery_filter_key(root, ignored, ignored_paths=ignored_paths)
    previous_filter_key = str(previous_record.get("filter_key") or "")
    if previous_filter_key and previous_filter_key != filter_key:
        return None, None
    current = [str(item) for item in previous_record.get("files") or () if str(item)]
    dir_signatures = {
        str(key): str(value)
        for key, value in dict(previous_record.get("dir_signatures") or {}).items()
        if isinstance(key, str)
    }
    if not current or not dir_signatures:
        return None, None
    root_resolved = root.resolve(strict=False)
    for rel, previous_signature in dir_signatures.items():
        directory = root_resolved if rel in {"", "."} else (root_resolved / rel)
        if _directory_state_signature(directory) != previous_signature:
            return None, None
    files = [root_resolved / rel for rel in current]
    return files, {
        "git_top": "",
        "scope_rel": "",
        "scope_oid": "",
        "filter_key": filter_key,
        "files": current,
        "dir_signatures": _directory_signatures_for_files(root_resolved, current),
    }


def _git_scope_head_oid(top: Path, scope_rel: str):
    try:
        if scope_rel in {"", "."}:
            return _run_git(top, "rev-parse", "HEAD^{tree}", text=True).strip()
        return _run_git(top, "rev-parse", f"HEAD:{scope_rel}", text=True).strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _git_scope_inventory_identity(top: Path, scope_rel: str, *, git_probe_cache=None):
    if scope_rel in {"", "."}:
        repo_head_bucket = _git_probe_cache_bucket(git_probe_cache, "repo_head")
        repo_head_key = str(top.resolve(strict=False))
        if repo_head_bucket is not None and repo_head_key in repo_head_bucket:
            cached = str(repo_head_bucket[repo_head_key] or "").strip()
            if cached:
                return cached
        try:
            identity = _run_git(top, "rev-parse", "HEAD^{tree}", text=True).strip()
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
        if repo_head_bucket is not None:
            repo_head_bucket[repo_head_key] = identity
        return identity
    head_oid_bucket = _git_probe_cache_bucket(git_probe_cache, "scope_head_oid")
    oid_key = (str(top.resolve(strict=False)), scope_rel)
    if head_oid_bucket is not None and oid_key in head_oid_bucket:
        return head_oid_bucket[oid_key]
    identity = _git_scope_head_oid(top, scope_rel)
    if head_oid_bucket is not None:
        head_oid_bucket[oid_key] = identity
    return identity


def _git_probe_cache_bucket(git_probe_cache, name):
    if git_probe_cache is None:
        return None
    bucket = git_probe_cache.get(name)
    if not isinstance(bucket, dict):
        bucket = {}
        git_probe_cache[name] = bucket
    return bucket


def _effective_git_probe_cache(git_probe_cache):
    if isinstance(git_probe_cache, dict):
        return git_probe_cache
    return _DEFAULT_GIT_PROBE_CACHE


def _git_fastpath_state_cache_key(top: Path, normalized_scope_paths):
    return (str(top.resolve(strict=False)), tuple(normalized_scope_paths))


def _load_git_fastpath_state_cache(git_probe_cache, top: Path, normalized_scope_paths):
    bucket = _git_probe_cache_bucket(git_probe_cache, "git_fastpath_state")
    if bucket is None:
        return None
    cached = bucket.get(_git_fastpath_state_cache_key(top, normalized_scope_paths))
    if not isinstance(cached, dict):
        return None
    return {
        "head": str(cached.get("head") or ""),
        "state": dict(cached.get("state") or {}),
        "aux": dict(cached.get("aux") or {}),
    }


def _store_git_fastpath_state_cache(git_probe_cache, top: Path, normalized_scope_paths, *, head: str, state, aux=None):
    bucket = _git_probe_cache_bucket(git_probe_cache, "git_fastpath_state")
    if bucket is None:
        return
    bucket[_git_fastpath_state_cache_key(top, normalized_scope_paths)] = {
        "head": str(head or ""),
        "state": dict(state or {}),
        "aux": dict(aux or {}),
    }


def _invalidate_git_probe_worktree_caches(git_probe_cache):
    if not isinstance(git_probe_cache, dict):
        return
    for key in (
        "scoped_status_output",
        "scoped_status_snapshot",
        "scoped_inventory_delta_paths",
        "scoped_clean_probe",
    ):
        git_probe_cache.pop(key, None)


def _git_status_scope_cache_key(top: Path, normalized_scope_paths, *, untracked_files="all"):
    return (
        str(top.resolve(strict=False)),
        tuple(normalized_scope_paths),
        str(untracked_files or "all"),
    )


def _ignored_filter_cache_key(top: Path, ignored: set[str], ignored_paths=None):
    relative_ignored_paths = []
    for path in (ignored_paths or ()):
        try:
            relative_ignored_paths.append(str(Path(path).resolve(strict=False).relative_to(top)))
        except ValueError:
            continue
    return (
        tuple(sorted(str(name) for name in (ignored or ()))),
        tuple(sorted(relative_ignored_paths)),
    )


def _metadata_source_candidates_cache_key(root: Path, ignored: set[str], ignored_paths=None):
    root = root.resolve(strict=False)
    relative_ignored_paths = sorted(
        str(path.relative_to(root))
        for path in (ignored_paths or ())
        if root in path.parents
    )
    return (
        str(root),
        tuple(sorted(str(name) for name in ignored)),
        tuple(relative_ignored_paths),
    )


def _normalize_touched_paths_hint(touched_paths):
    if not isinstance(touched_paths, (list, tuple, set)):
        return ()
    normalized = []
    seen = set()
    for raw_path in touched_paths:
        raw_text = str(raw_path or "").strip()
        if not raw_text:
            continue
        path = PurePosixPath(raw_text).as_posix()
        if path in {"", "."}:
            continue
        if path not in seen:
            seen.add(path)
            normalized.append(path)
    return tuple(sorted(normalized))


def _paths_within_scope(paths, normalized_scope_paths):
    normalized_scope_paths = [str(path) for path in (normalized_scope_paths or ()) if str(path or "") not in {"", "."}]
    if not normalized_scope_paths:
        return tuple(
            sorted(
                {
                    PurePosixPath(str(path or "")).as_posix()
                    for path in (paths or ())
                    if str(path or "") not in {"", "."}
                }
            )
        )
    scope_prefixes = tuple(f"{scope.rstrip('/')}/" for scope in normalized_scope_paths)
    selected = set()
    for raw_path in paths or ():
        normalized = PurePosixPath(str(raw_path or "")).as_posix()
        if normalized in {"", "."}:
            continue
        if normalized in normalized_scope_paths or any(normalized.startswith(prefix) for prefix in scope_prefixes):
            selected.add(normalized)
    return tuple(sorted(selected))


def _cached_git_fastpath_dirty_paths(top: Path, normalized_scope_paths=None, *, git_probe_cache=None):
    candidate_scope_keys = []
    normalized_scope_paths = tuple(normalized_scope_paths or ())
    if normalized_scope_paths:
        candidate_scope_keys.append(list(normalized_scope_paths))
    else:
        candidate_scope_keys.append(["."])
    candidate_scope_keys.append([])
    seen_scope_keys = set()
    for scope_key in candidate_scope_keys:
        scope_tuple = tuple(scope_key)
        if scope_tuple in seen_scope_keys:
            continue
        seen_scope_keys.add(scope_tuple)
        cached = _load_git_fastpath_state_cache(git_probe_cache, top, scope_key)
        if not isinstance(cached, dict):
            continue
        state = cached.get("state")
        if not isinstance(state, dict) or str(state.get("kind") or "") != "git":
            continue
        dirty_paths = state.get("dirty_paths")
        if dirty_paths is None:
            if _git_state_is_clean(state):
                return tuple()
            continue
        return _paths_within_scope(dirty_paths, normalized_scope_paths)
    return None


def _touched_paths_hint_from_repository_fingerprint_state(repository_fingerprint_state):
    if not isinstance(repository_fingerprint_state, (list, tuple)):
        return ()
    normalized = []
    seen = set()
    for state in repository_fingerprint_state:
        if not isinstance(state, dict):
            continue
        raw_paths = state.get("dirty_paths")
        if not isinstance(raw_paths, (list, tuple, set)):
            continue
        for raw_path in raw_paths:
            raw_text = str(raw_path or "").strip()
            if not raw_text:
                continue
            path = PurePosixPath(raw_text).as_posix()
            if path in {"", "."} or path in seen:
                continue
            seen.add(path)
            normalized.append(path)
    return tuple(sorted(normalized))


def _clean_worktree_files_hint_from_repository_fingerprint_state(repository_fingerprint_state):
    if not isinstance(repository_fingerprint_state, (list, tuple)):
        return ()
    for state in repository_fingerprint_state:
        if not isinstance(state, dict):
            continue
        raw_files = state.get("clean_worktree_files")
        if not isinstance(raw_files, (list, tuple)):
            continue
        normalized = []
        seen = set()
        for raw_path in raw_files:
            raw_text = str(raw_path or "").strip()
            if not raw_text:
                continue
            path = PurePosixPath(raw_text).as_posix()
            if path in {"", "."} or path in seen:
                continue
            seen.add(path)
            normalized.append(path)
        if normalized:
            return tuple(sorted(normalized))
    return ()


def _discovered_files_from_clean_worktree_hint(
    discovered,
    ignored,
    *,
    ignored_paths=None,
    git_probe_cache=None,
    clean_worktree_files_hint=None,
):
    normalized_hint = _normalize_touched_paths_hint(clean_worktree_files_hint)
    if not normalized_hint or len(discovered or ()) != 1:
        return None
    item = discovered[0] if isinstance(discovered[0], dict) else None
    if not isinstance(item, dict):
        return None
    path_value = item.get("path")
    if path_value is None:
        return None
    root = Path(path_value)
    if not root.is_dir():
        return None
    git_top = _git_top(root)
    if git_top is None:
        return None
    git_top = git_top.resolve(strict=False)
    root_resolved = root.resolve(strict=False)
    try:
        scope_rel = root_resolved.relative_to(git_top).as_posix()
    except ValueError:
        return None
    scope_prefix = "" if scope_rel in {"", "."} else f"{scope_rel.rstrip('/')}/"
    files = []
    for rel in normalized_hint:
        if scope_prefix:
            if rel == scope_rel:
                continue
            if not rel.startswith(scope_prefix):
                continue
            item_rel = rel[len(scope_prefix) :]
        else:
            item_rel = rel
        if item_rel in {"", "."}:
            continue
        if not language_for_path(Path(item_rel)):
            continue
        candidate = git_top / rel
        if _should_skip(candidate, root_resolved, ignored, ignored_paths=ignored_paths):
            continue
        files.append((item, candidate, PurePosixPath(item_rel).as_posix()))
    return files or None


def _discovered_files_from_previous_manifest(
    discovered,
    namespaces,
    previous_manifest,
    *,
    ignored,
    ignored_paths=None,
    touched_paths_hint=None,
    trust_untouched_manifest=False,
):
    normalized_touched_paths_hint = _normalize_touched_paths_hint(touched_paths_hint)
    if not normalized_touched_paths_hint or len(discovered or ()) != 1:
        return None
    item = discovered[0] if isinstance(discovered[0], dict) else None
    if not isinstance(item, dict):
        return None
    path_value = item.get("path")
    if path_value is None:
        return None
    root = Path(path_value)
    if not root.is_dir():
        return None
    root_resolved = root.resolve(strict=False)
    source_namespace = str(namespaces.get(id(item), ""))
    if not source_namespace:
        return None
    previous_rel_paths = []
    for manifest_entry_key in previous_manifest or {}:
        entry_key = str(manifest_entry_key or "")
        namespace, sep, rel_path = entry_key.partition("\0")
        if not sep or namespace != source_namespace:
            continue
        rel_path = PurePosixPath(str(rel_path or "")).as_posix()
        if rel_path in {"", "."}:
            continue
        previous_rel_paths.append(rel_path)
    if not previous_rel_paths:
        return None
    previous_rel_paths_set = set(previous_rel_paths)
    current = set(previous_rel_paths)
    touched_candidates = {}
    added = []
    for rel in normalized_touched_paths_hint:
        rel = PurePosixPath(str(rel or "")).as_posix()
        if rel in {"", "."}:
            continue
        candidate = root_resolved / rel
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or _should_skip(candidate, root_resolved, ignored, ignored_paths=ignored_paths)
            or not language_for_path(candidate)
        ):
            current.discard(rel)
            continue
        touched_candidates[rel] = candidate
        if rel not in current:
            added.append(rel)
        current.add(rel)
    files = []
    for rel in previous_rel_paths:
        if rel not in current:
            continue
        candidate = touched_candidates.get(rel)
        if candidate is not None:
            files.append((item, candidate, rel))
            continue
        candidate = root_resolved / rel
        if trust_untouched_manifest and rel in previous_rel_paths_set:
            files.append((item, candidate, rel))
            continue
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or _should_skip(candidate, root_resolved, ignored, ignored_paths=ignored_paths)
            or not language_for_path(candidate)
        ):
            continue
        files.append((item, candidate, rel))
    for rel in added:
        candidate = touched_candidates.get(rel)
        if candidate is not None:
            files.append((item, candidate, rel))
    return files or None


def _seed_discovery_manifests_from_clean_worktree_hint(
    discovered,
    ignored,
    *,
    cache_dir=None,
    ignored_paths=None,
    repository_state_fingerprint=None,
    clean_worktree_files_hint=None,
):
    normalized_hint = _normalize_touched_paths_hint(clean_worktree_files_hint)
    if not normalized_hint or len(discovered or ()) != 1 or cache_dir is None:
        return
    item = discovered[0] if isinstance(discovered[0], dict) else None
    if not isinstance(item, dict):
        return
    path_value = item.get("path")
    if path_value is None:
        return
    root = Path(path_value)
    if not root.is_dir():
        return
    git_top = _git_top(root)
    if git_top is None:
        return
    git_top = git_top.resolve(strict=False)
    root_resolved = root.resolve(strict=False)
    scope_rel = _git_scope_rel(root_resolved, git_top)
    if scope_rel is None:
        return
    root_key = str(root_resolved)
    filter_key = _discovery_filter_key(root_resolved, ignored, ignored_paths=ignored_paths)
    item_relative_files = []
    scope_prefix = "" if scope_rel in {"", "."} else f"{scope_rel.rstrip('/')}/"
    for rel in normalized_hint:
        if scope_prefix:
            if rel == scope_rel or not rel.startswith(scope_prefix):
                continue
            item_rel = rel[len(scope_prefix) :]
        else:
            item_rel = rel
        if item_rel in {"", "."}:
            continue
        if not language_for_path(Path(item_rel)):
            continue
        candidate = git_top / rel
        if _should_skip(candidate, root_resolved, ignored, ignored_paths=ignored_paths):
            continue
        item_relative_files.append(PurePosixPath(item_rel).as_posix())
    if not item_relative_files:
        return
    item_relative_files = list(dict.fromkeys(item_relative_files))
    discovery_record = {
        "git_top": str(git_top),
        "scope_rel": str(scope_rel or ""),
        "scope_oid": "",
        "repository_state_fingerprint": str(repository_state_fingerprint or ""),
        "filter_key": filter_key,
        "files": list(item_relative_files),
        "dir_signatures": {},
    }
    _write_preferred_discovery_working_manifest(
        _discovery_manifest_path(Path(cache_dir)),
        _shared_discovery_manifest_path(create=True),
        {root_key: discovery_record},
    )
    if repository_state_fingerprint:
        _write_preferred_discovery_working_manifest(
            _discovery_fingerprint_manifest_path(Path(cache_dir), str(repository_state_fingerprint)),
            _shared_discovery_fingerprint_manifest_path(str(repository_state_fingerprint), create=True),
            {root_key: discovery_record},
        )
    _store_discovery_record_process_cache(root_key, discovery_record)
    if repository_state_fingerprint:
        _store_discovery_exact_process_cache(
            root_resolved,
            str(repository_state_fingerprint),
            filter_key,
            discovery_record,
        )


def _normalize_discovery_record(record):
    if not isinstance(record, dict):
        return None
    return {
        "git_top": str(record.get("git_top") or ""),
        "scope_rel": str(record.get("scope_rel") or ""),
        "scope_oid": str(record.get("scope_oid") or ""),
        "repository_state_fingerprint": str(record.get("repository_state_fingerprint") or ""),
        "filter_key": str(record.get("filter_key") or ""),
        "files": [str(item) for item in (record.get("files") or ()) if isinstance(item, str)],
        "dir_signatures": {
            str(key): str(value)
            for key, value in dict(record.get("dir_signatures") or {}).items()
            if isinstance(key, str)
        },
    }


def _discovery_record_inventory_equal(left, right):
    left = _normalize_discovery_record(left)
    right = _normalize_discovery_record(right)
    if left is None or right is None:
        return False
    return (
        str(left.get("git_top") or "") == str(right.get("git_top") or "")
        and str(left.get("scope_rel") or "") == str(right.get("scope_rel") or "")
        and str(left.get("scope_oid") or "") == str(right.get("scope_oid") or "")
        and str(left.get("filter_key") or "") == str(right.get("filter_key") or "")
        and list(left.get("files") or ()) == list(right.get("files") or ())
        and dict(left.get("dir_signatures") or {}) == dict(right.get("dir_signatures") or {})
    )


def _prune_discovery_record_process_cache():
    while len(_DISCOVERY_RECORD_PROCESS_CACHE) > _DISCOVERY_RECORD_PROCESS_CACHE_LIMIT:
        _DISCOVERY_RECORD_PROCESS_CACHE.pop(next(iter(_DISCOVERY_RECORD_PROCESS_CACHE)))


def _load_discovery_record_process_cache(root_key: str):
    cached = _DISCOVERY_RECORD_PROCESS_CACHE.get(str(root_key))
    normalized = _normalize_discovery_record(cached)
    if normalized is None:
        return None
    _DISCOVERY_RECORD_PROCESS_CACHE.pop(str(root_key), None)
    _DISCOVERY_RECORD_PROCESS_CACHE[str(root_key)] = normalized
    return dict(normalized)


def _store_discovery_record_process_cache(root_key: str, record):
    normalized = _normalize_discovery_record(record)
    if normalized is None:
        return
    _DISCOVERY_RECORD_PROCESS_CACHE.pop(str(root_key), None)
    _DISCOVERY_RECORD_PROCESS_CACHE[str(root_key)] = normalized
    _prune_discovery_record_process_cache()


def _discovery_exact_process_cache_key(root: Path, repository_state_fingerprint: str, filter_key: str):
    return (
        str(Path(root).resolve(strict=False)),
        str(repository_state_fingerprint or ""),
        str(filter_key or ""),
    )


def _prune_discovery_exact_process_cache():
    while len(_DISCOVERY_EXACT_PROCESS_CACHE) > _DISCOVERY_EXACT_PROCESS_CACHE_LIMIT:
        _DISCOVERY_EXACT_PROCESS_CACHE.pop(next(iter(_DISCOVERY_EXACT_PROCESS_CACHE)))


def _load_discovery_exact_process_cache(root: Path, repository_state_fingerprint: str, filter_key: str):
    cache_key = _discovery_exact_process_cache_key(root, repository_state_fingerprint, filter_key)
    cached = _DISCOVERY_EXACT_PROCESS_CACHE.get(cache_key)
    normalized = _normalize_discovery_record(cached)
    if normalized is None:
        return None
    _DISCOVERY_EXACT_PROCESS_CACHE.pop(cache_key, None)
    _DISCOVERY_EXACT_PROCESS_CACHE[cache_key] = normalized
    return dict(normalized)


def _store_discovery_exact_process_cache(root: Path, repository_state_fingerprint: str, filter_key: str, record):
    normalized = _normalize_discovery_record(record)
    if normalized is None:
        return
    cache_key = _discovery_exact_process_cache_key(root, repository_state_fingerprint, filter_key)
    _DISCOVERY_EXACT_PROCESS_CACHE.pop(cache_key, None)
    _DISCOVERY_EXACT_PROCESS_CACHE[cache_key] = normalized
    _prune_discovery_exact_process_cache()


def _exact_discovery_record_matches(record, *, repository_state_fingerprint: str, filter_key: str):
    normalized = _normalize_discovery_record(record)
    if normalized is None:
        return False
    if str(normalized.get("repository_state_fingerprint") or "") != str(repository_state_fingerprint or ""):
        return False
    previous_filter_key = str(normalized.get("filter_key") or "")
    return bool(previous_filter_key) and previous_filter_key == str(filter_key or "")


def _cached_metadata_source_candidates(root: Path, ignored: set[str], ignored_paths=None, *, git_probe_cache=None):
    bucket = _git_probe_cache_bucket(git_probe_cache, "metadata_source_candidates")
    if bucket is None:
        return None
    cache_key = _metadata_source_candidates_cache_key(root, ignored, ignored_paths=ignored_paths)
    cached = bucket.get(cache_key)
    if not isinstance(cached, list):
        return None
    return [Path(candidate) for candidate in cached]


def _store_metadata_source_candidates(root: Path, ignored: set[str], candidates, ignored_paths=None, *, git_probe_cache=None):
    bucket = _git_probe_cache_bucket(git_probe_cache, "metadata_source_candidates")
    if bucket is None:
        return
    cache_key = _metadata_source_candidates_cache_key(root, ignored, ignored_paths=ignored_paths)
    bucket[cache_key] = [str(Path(candidate).resolve(strict=False)) for candidate in candidates]


def _normalize_scope_paths(top: Path, scope_paths):
    normalized = []
    for scope in scope_paths or ():
        try:
            rel = Path(scope).resolve(strict=False).relative_to(top).as_posix()
        except ValueError:
            continue
        normalized.append(rel or ".")
    return sorted(set(normalized))


def _scope_pathspec(normalized_scope_paths):
    if normalized_scope_paths and any(item not in {"", "."} for item in normalized_scope_paths):
        return ["--", *normalized_scope_paths]
    return []


def _scoped_git_head_identity(top: Path, normalized_scope_paths, *, git_probe_cache=None):
    cache_key = (str(top.resolve(strict=False)), tuple(normalized_scope_paths))
    bucket = _git_probe_cache_bucket(git_probe_cache, "scoped_head_identity")
    if bucket is not None and cache_key in bucket:
        return bucket[cache_key]
    if not normalized_scope_paths or all(item in {"", "."} for item in normalized_scope_paths):
        repo_head_bucket = _git_probe_cache_bucket(git_probe_cache, "repo_head")
        repo_head_key = str(top.resolve(strict=False))
        if repo_head_bucket is not None and repo_head_key in repo_head_bucket:
            identity = repo_head_bucket[repo_head_key]
        else:
            identity = _git_head_state_signature(top)
            if not identity:
                try:
                    identity = _run_git(top, "rev-parse", "HEAD^{tree}", text=True).strip()
                except subprocess.SubprocessError:
                    identity = "unborn"
            if repo_head_bucket is not None:
                repo_head_bucket[repo_head_key] = identity
        if bucket is not None:
            bucket[cache_key] = identity
        return identity
    object_ids = []
    head_oid_bucket = _git_probe_cache_bucket(git_probe_cache, "scope_head_oid")
    for rel in normalized_scope_paths:
        oid_key = (str(top.resolve(strict=False)), rel)
        if head_oid_bucket is not None and oid_key in head_oid_bucket:
            oid = head_oid_bucket[oid_key]
        else:
            oid = _git_scope_head_oid(top, rel)
            if head_oid_bucket is not None:
                head_oid_bucket[oid_key] = oid
        if oid is None:
            oid = "missing"
        object_ids.append((rel, oid))
    encoded = json.dumps(object_ids, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    identity = f"scope:{sha256_text(encoded)}"
    if bucket is not None:
        bucket[cache_key] = identity
    return identity


def _scoped_git_status_output(top: Path, normalized_scope_paths, *, untracked_files="all", git_probe_cache=None):
    cache_key = _git_status_scope_cache_key(top, normalized_scope_paths, untracked_files=untracked_files)
    bucket = _git_probe_cache_bucket(git_probe_cache, "scoped_status_output")
    if bucket is not None and cache_key in bucket:
        return bucket[cache_key]
    if bucket is not None:
        covering = _cached_covering_scoped_status_value(
            bucket,
            top,
            tuple(normalized_scope_paths),
            untracked_files=untracked_files,
        )
        if covering is not None:
            bucket[cache_key] = covering
            return covering
    args = [
        "status",
        "--porcelain=v1",
        "--no-renames",
        "-z",
        f"--untracked-files={str(untracked_files or 'all')}",
        "--ignored=no",
        *_scope_pathspec(normalized_scope_paths),
    ]
    output = _run_git_optional(top, *args)
    if bucket is not None:
        bucket[cache_key] = output
    return output


def _scoped_git_status_snapshot(top: Path, normalized_scope_paths, *, untracked_files="all", git_probe_cache=None):
    cache_key = _git_status_scope_cache_key(top, normalized_scope_paths, untracked_files=untracked_files)
    bucket = _git_probe_cache_bucket(git_probe_cache, "scoped_status_snapshot")
    if bucket is not None and cache_key in bucket:
        snapshot = bucket[cache_key]
        if isinstance(snapshot, dict):
            return snapshot
    if bucket is not None:
        covering = _cached_covering_scoped_status_value(
            bucket,
            top,
            tuple(normalized_scope_paths),
            untracked_files=untracked_files,
        )
        if isinstance(covering, dict):
            bucket[cache_key] = covering
            return covering
    output = _scoped_git_status_output(
        top,
        normalized_scope_paths,
        untracked_files=untracked_files,
        git_probe_cache=git_probe_cache,
    )
    if output is None:
        return None
    snapshot = {
        "output": output,
        "parsed_entries": _parse_git_status_entries(output),
        "touched_rel_paths": None,
        "subset_dirty": {},
        "subset_dirty_only": {},
        "filtered_details": {},
        "inventory_preserved": {},
    }
    if bucket is not None:
        bucket[cache_key] = snapshot
    return snapshot


def _scoped_git_inventory_delta_paths(top: Path, normalized_scope_paths, *, ignored=None, ignored_paths=None, git_probe_cache=None):
    cache_key = (str(top.resolve(strict=False)), tuple(normalized_scope_paths), _ignored_filter_cache_key(top, ignored or set(), ignored_paths=ignored_paths))
    bucket = _git_probe_cache_bucket(git_probe_cache, "scoped_inventory_delta_paths")
    if bucket is not None and cache_key in bucket:
        return set(bucket[cache_key])

    cached_snapshot = None
    bucket_snapshot = _git_probe_cache_bucket(git_probe_cache, "scoped_status_snapshot")
    if bucket_snapshot is not None:
        cached_snapshot = bucket_snapshot.get((str(top.resolve(strict=False)), tuple(normalized_scope_paths), "all"))
    if isinstance(cached_snapshot, dict):
        touched = set()
        normalized_scope_paths = [str(path) for path in (normalized_scope_paths or ()) if str(path)]
        scope_prefixes = tuple(f"{scope.rstrip('/')}/" for scope in normalized_scope_paths)
        for entry in cached_snapshot.get("parsed_entries") or ():
            for raw_path in entry.get("paths") or ():
                if not raw_path:
                    continue
                normalized = PurePosixPath(raw_path).as_posix()
                if normalized in {"", "."}:
                    continue
                if normalized_scope_paths:
                    in_scope = normalized in normalized_scope_paths or any(
                        normalized.startswith(prefix) for prefix in scope_prefixes
                    )
                    if not in_scope:
                        continue
                candidate = (top / normalized).resolve(strict=False)
                if _should_skip(candidate, top, ignored or set(), ignored_paths=ignored_paths):
                    continue
                touched.add(normalized)
        if bucket is not None:
            bucket[cache_key] = tuple(sorted(touched))
        return touched
    cached_dirty_paths = _cached_git_fastpath_dirty_paths(
        top,
        normalized_scope_paths,
        git_probe_cache=git_probe_cache,
    )
    if cached_dirty_paths is not None:
        touched = set()
        for normalized in cached_dirty_paths:
            candidate = (top / normalized).resolve(strict=False)
            if _should_skip(candidate, top, ignored or set(), ignored_paths=ignored_paths):
                continue
            touched.add(normalized)
        if bucket is not None:
            bucket[cache_key] = tuple(sorted(touched))
        return touched

    pathspec = _scope_pathspec(normalized_scope_paths)
    base = ["git", "-C", str(top), "--no-optional-locks"]

    try:
        tracked = subprocess.run(
            [*base, "diff", "--name-only", "--diff-filter=ADRTUXB", "-z", "HEAD", *pathspec],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
        if tracked.returncode != 0:
            return None
        untracked = subprocess.run(
            [*base, "ls-files", "--others", "--exclude-standard", "-z", *pathspec],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
        if untracked.returncode != 0:
            return None
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None

    touched = set()
    for output in (tracked.stdout, untracked.stdout):
        for raw_path in output.split(b"\0"):
            if not raw_path:
                continue
            normalized = PurePosixPath(raw_path.decode("utf-8", errors="surrogateescape")).as_posix()
            candidate = (top / normalized).resolve(strict=False)
            if _should_skip(candidate, top, ignored or set(), ignored_paths=ignored_paths):
                continue
            if normalized in {"", "."}:
                continue
            touched.add(normalized)
    if bucket is not None:
        bucket[cache_key] = tuple(sorted(touched))
    return touched


def _scoped_git_clean_probe(top: Path, normalized_scope_paths, *, ignored=None, ignored_paths=None, git_probe_cache=None):
    ignored_names = tuple(sorted(str(name) for name in (ignored or ())))
    relative_ignored_paths = []
    for path in (ignored_paths or ()):
        try:
            relative_ignored_paths.append(str(Path(path).resolve(strict=False).relative_to(top)))
        except ValueError:
            continue
    cache_key = (
        str(top.resolve(strict=False)),
        tuple(normalized_scope_paths),
        ignored_names,
        tuple(sorted(relative_ignored_paths)),
    )
    bucket = _git_probe_cache_bucket(git_probe_cache, "scoped_clean_probe")
    if bucket is not None and cache_key in bucket:
        return bucket[cache_key]
    def record(value):
        if bucket is not None:
            bucket[cache_key] = value
        return value
    status_snapshot = _scoped_git_status_snapshot(
        top,
        normalized_scope_paths,
        git_probe_cache=git_probe_cache,
    )
    if status_snapshot is None:
        return record(None)
    filter_key = _ignored_filter_cache_key(top, ignored or set(), ignored_paths=ignored_paths)
    status_digest, filtered_status, _relevant_entries, _tracked_index_paths = _filtered_git_status_details(
        status_snapshot.get("output"),
        top,
        ignored or set(),
        ignored_paths=ignored_paths,
        parsed_entries=status_snapshot.get("parsed_entries"),
    )
    status_snapshot.setdefault("filtered_details", {})[filter_key] = (
        status_digest,
        filtered_status,
        _relevant_entries,
        _tracked_index_paths,
    )
    return record(not filtered_status)


def _cached_repo_scope_status_output(top: Path, *, untracked_files="all", git_probe_cache=None):
    bucket = _git_probe_cache_bucket(git_probe_cache, "scoped_status_output")
    if bucket is None:
        return None
    top_key = str(top.resolve(strict=False))
    for scope_key in ((), (".",)):
        cache_key = (top_key, scope_key, str(untracked_files or "all"))
        if cache_key in bucket:
            return bucket[cache_key]
    return None


def _cached_repo_scope_status_snapshot(top: Path, *, untracked_files="all", git_probe_cache=None):
    bucket = _git_probe_cache_bucket(git_probe_cache, "scoped_status_snapshot")
    if bucket is None:
        return None
    top_key = str(top.resolve(strict=False))
    for scope_key in ((), (".",)):
        cache_key = (top_key, scope_key, str(untracked_files or "all"))
        snapshot = bucket.get(cache_key)
        if isinstance(snapshot, dict):
            return snapshot
    return None


def _cached_scoped_status_output(top: Path, scope_rel: str | None, *, untracked_files="all", git_probe_cache=None):
    bucket = _git_probe_cache_bucket(git_probe_cache, "scoped_status_output")
    if bucket is None:
        return None
    top_key = str(top.resolve(strict=False))
    normalized_scope = "." if scope_rel in {None, "", "."} else str(scope_rel)
    for scope_key in ((normalized_scope,), tuple() if normalized_scope == "." else None):
        if scope_key is None:
            continue
        cache_key = (top_key, scope_key, str(untracked_files or "all"))
        if cache_key in bucket:
            return bucket[cache_key]
    return _cached_covering_scoped_status_value(bucket, top, (normalized_scope,), untracked_files=untracked_files)


def _cached_scoped_status_snapshot(top: Path, scope_rel: str | None, *, untracked_files="all", git_probe_cache=None):
    bucket = _git_probe_cache_bucket(git_probe_cache, "scoped_status_snapshot")
    if bucket is None:
        return None
    top_key = str(top.resolve(strict=False))
    normalized_scope = "." if scope_rel in {None, "", "."} else str(scope_rel)
    for scope_key in ((normalized_scope,), tuple() if normalized_scope == "." else None):
        if scope_key is None:
            continue
        cache_key = (top_key, scope_key, str(untracked_files or "all"))
        snapshot = bucket.get(cache_key)
        if isinstance(snapshot, dict):
            return snapshot
    covering = _cached_covering_scoped_status_value(bucket, top, (normalized_scope,), untracked_files=untracked_files)
    return covering if isinstance(covering, dict) else None


def _cached_covering_scoped_status_value(bucket, top: Path, requested_scope_key, *, untracked_files="all"):
    if not isinstance(bucket, dict):
        return None
    top_key = str(top.resolve(strict=False))
    cache_untracked = str(untracked_files or "all")
    normalized_requested_scope_key = tuple(
        "." if item in {"", "."} else str(item)
        for item in (requested_scope_key or ())
        if str(item or "")
    )
    for cache_key, value in bucket.items():
        if not isinstance(cache_key, tuple) or len(cache_key) != 3:
            continue
        cached_top, cached_scope_key, cached_untracked = cache_key
        if cached_top != top_key or cached_untracked != cache_untracked:
            continue
        if _cached_scope_tuple_covers_request(cached_scope_key, normalized_requested_scope_key):
            return value
    return None


def _cached_scope_tuple_covers_request(cached_scope_key, requested_scope_key) -> bool:
    if not isinstance(cached_scope_key, tuple) or not isinstance(requested_scope_key, tuple):
        return False
    normalized_cached = tuple("." if item in {"", "."} else str(item) for item in cached_scope_key)
    if requested_scope_key in {tuple(), (".",)}:
        return cached_scope_key in {tuple(), (".",)}
    if normalized_cached in {tuple(), (".",)}:
        return True
    for requested_scope in requested_scope_key:
        requested_prefix = f"{requested_scope.rstrip('/')}/"
        matched = False
        for normalized_candidate in normalized_cached:
            candidate_prefix = f"{normalized_candidate.rstrip('/')}/"
            if normalized_candidate == ".":
                matched = True
                break
            if normalized_candidate == requested_scope:
                matched = True
                break
            if requested_scope.startswith(candidate_prefix):
                matched = True
                break
            if normalized_candidate.startswith(requested_prefix):
                matched = True
                break
        if not matched:
            return False
    return True


def _cached_scoped_status_preserves_inventory(scope_rel: str | None, snapshot, *, ignored=None, ignored_paths=None, top: Path):
    if not isinstance(snapshot, dict):
        return False
    filter_key = _ignored_filter_cache_key(top, ignored or set(), ignored_paths=ignored_paths)
    cached = dict(snapshot.get("inventory_preserved") or {}).get((str(scope_rel or "."), filter_key))
    if isinstance(cached, bool):
        return cached
    parsed_entries = snapshot.get("parsed_entries")
    if not isinstance(parsed_entries, list):
        return False
    normalized_scope = "." if scope_rel in {None, "", "."} else str(scope_rel)
    scope_prefix = "" if normalized_scope == "." else f"{normalized_scope.rstrip('/')}/"
    for entry in parsed_entries:
        if not isinstance(entry, dict):
            return False
        code = str(entry.get("code") or "")
        raw_paths = entry.get("paths") or ()
        relevant_paths = []
        for raw_path in raw_paths:
            normalized = PurePosixPath(str(raw_path or "")).as_posix()
            if not normalized or normalized == ".":
                continue
            if scope_prefix:
                if normalized != normalized_scope and not normalized.startswith(scope_prefix):
                    continue
            candidate = (top / normalized).resolve(strict=False)
            if _should_skip(candidate, top, ignored or set(), ignored_paths=ignored_paths):
                continue
            relevant_paths.append(normalized)
        if not relevant_paths:
            continue
        x_code = code[:1]
        y_code = code[1:2]
        if x_code == "?" or y_code == "?":
            snapshot.setdefault("inventory_preserved", {})[(str(scope_rel or "."), filter_key)] = False
            return False
        if x_code not in {"", " ", "M"}:
            snapshot.setdefault("inventory_preserved", {})[(str(scope_rel or "."), filter_key)] = False
            return False
        if y_code not in {"", " ", "M"}:
            snapshot.setdefault("inventory_preserved", {})[(str(scope_rel or "."), filter_key)] = False
            return False
    snapshot.setdefault("inventory_preserved", {})[(str(scope_rel or "."), filter_key)] = True
    return True


def _git_scope_status_paths(root: Path, top: Path, scope_rel: str, *, ignored=None, ignored_paths=None, git_probe_cache=None):
    normalized_scope_paths = [] if scope_rel in {"", "."} else [scope_rel]
    snapshot = _scoped_git_status_snapshot(top, normalized_scope_paths, git_probe_cache=git_probe_cache)
    if snapshot is None:
        return None
    cached_touched = snapshot.get("touched_rel_paths")
    if isinstance(cached_touched, tuple):
        return set(cached_touched)
    touched = set()
    scope_prefix = "" if scope_rel in {"", "."} else f"{scope_rel.rstrip('/')}/"
    for entry in snapshot.get("parsed_entries") or ():
        raw_paths = entry.get("paths") or ()
        for raw_path in raw_paths:
            if not raw_path:
                continue
            normalized = PurePosixPath(raw_path).as_posix()
            candidate = (top / normalized).resolve(strict=False)
            if _should_skip(candidate, top, ignored or set(), ignored_paths=ignored_paths):
                continue
            if scope_prefix:
                if normalized == scope_rel:
                    rel = "."
                elif normalized.startswith(scope_prefix):
                    rel = normalized[len(scope_prefix) :]
                else:
                    continue
            else:
                rel = normalized
            if rel in {"", "."}:
                continue
            touched.add(rel)
    snapshot["touched_rel_paths"] = tuple(sorted(touched))
    return touched


def _parse_git_status_entries(status_output: bytes):
    entries = status_output.decode("utf-8", errors="surrogateescape").split("\0")
    parsed = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry or len(entry) < 4:
            continue
        code = entry[:2]
        paths = [entry[3:]]
        if code and code[0] in {"R", "C"} and index < len(entries):
            paths.append(entries[index])
            index += 1
        parsed.append(
            {
                "code": code,
                "paths": [PurePosixPath(path).as_posix() for path in paths if path],
            }
        )
    return parsed


def _status_subset_digest_and_dirty(status_output: bytes, allowed_paths, *, parsed_entries=None):
    allowed = set(str(path) for path in (allowed_paths or ()))
    kept = []
    dirty = set()
    for entry in parsed_entries if parsed_entries is not None else _parse_git_status_entries(status_output):
        relevant_paths = [path for path in entry["paths"] if path in allowed]
        if not relevant_paths:
            continue
        kept.append((entry["code"], tuple(relevant_paths)))
        dirty.update(relevant_paths)
    encoded = json.dumps(kept, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_text(encoded), dirty


def _status_subset_dirty(allowed_paths, *, parsed_entries=None):
    allowed = set(str(path) for path in (allowed_paths or ()))
    dirty = set()
    for entry in parsed_entries or ():
        relevant_paths = [path for path in entry["paths"] if path in allowed]
        if relevant_paths:
            dirty.update(relevant_paths)
    return dirty


def _filtered_git_status_details(status_output: bytes, top: Path, ignored: set[str], ignored_paths=None, *, parsed_entries=None):
    kept = []
    relevant_entries = []
    tracked_index_paths = set()
    for entry in parsed_entries if parsed_entries is not None else _parse_git_status_entries(status_output):
        code = entry["code"]
        relevant_paths = []
        for raw_path in entry["paths"]:
            candidate = (top / raw_path).resolve(strict=False)
            if _should_skip(candidate, top, ignored, ignored_paths=ignored_paths):
                continue
            relevant_paths.append(raw_path)
        if relevant_paths:
            kept.append((code, tuple(relevant_paths)))
            relevant_entries.append({"code": code, "paths": relevant_paths})
            x_code = code[:1]
            y_code = code[1:2]
            if x_code not in {"", " ", "?"}:
                tracked_index_paths.add(relevant_paths[0])
            if x_code in {"R", "C"} and relevant_paths:
                tracked_index_paths.add(relevant_paths[-1])
    encoded = json.dumps(kept, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_text(encoded), kept, relevant_entries, tracked_index_paths


def _git_fingerprint_manifest_path(cache_dir: Path, top: Path, normalized_scope_paths):
    payload = json.dumps(
        {
            "git_top": str(top.resolve(strict=False)),
            "scope_paths": list(normalized_scope_paths),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    cache_root = _private_cache_dir(Path(cache_dir) / "git-fingerprint-cache")
    return cache_root / f"git-{sha256_text(payload)}.json"


def _shared_git_fingerprint_manifest_path(top: Path, normalized_scope_paths, *, create=False):
    cache_root = _shared_repo_inspection_cache_root(create=create)
    if cache_root is None:
        return None
    payload = json.dumps(
        {
            "git_top": str(top.resolve(strict=False)),
            "scope_paths": list(normalized_scope_paths),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    cache_root = cache_root / "git-fingerprint-cache"
    try:
        cache_root = _private_cache_dir(cache_root) if create else cache_root.expanduser().absolute()
    except OSError:
        return None
    return cache_root / f"git-{sha256_text(payload)}.json"


def _clone_git_fingerprint_manifest_record(record):
    return {
        "head": str(record.get("head") or ""),
        "status": str(record.get("status") or ""),
        "state": dict(record.get("state") or {}),
        "aux": dict(record.get("aux") or {}),
    }


def _load_git_fingerprint_manifest(path: Path):
    cache_key = _snapshot_metadata_cache_key(path)
    if cache_key is not None:
        cached = _GIT_FINGERPRINT_MANIFEST_CACHE.get(cache_key)
        if cached is not None:
            _GIT_FINGERPRINT_MANIFEST_CACHE.pop(cache_key, None)
            _GIT_FINGERPRINT_MANIFEST_CACHE[cache_key] = cached
            return _clone_git_fingerprint_manifest_record(cached)
    elif not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        path.unlink(missing_ok=True)
        return None
    if not isinstance(payload, dict) or payload.get("schema") != GIT_FINGERPRINT_MANIFEST_SCHEMA:
        return None
    state = payload.get("state")
    if not isinstance(state, dict):
        return None
    aux = payload.get("aux")
    if not isinstance(aux, dict):
        aux = {}
    normalized = {
        "head": str(payload.get("head") or ""),
        "status": str(payload.get("status") or ""),
        "state": state,
        "aux": aux,
    }
    if cache_key is not None:
        _GIT_FINGERPRINT_MANIFEST_CACHE.pop(cache_key, None)
        _GIT_FINGERPRINT_MANIFEST_CACHE[cache_key] = _clone_git_fingerprint_manifest_record(normalized)
        while len(_GIT_FINGERPRINT_MANIFEST_CACHE) > _GIT_FINGERPRINT_MANIFEST_CACHE_LIMIT:
            _GIT_FINGERPRINT_MANIFEST_CACHE.pop(next(iter(_GIT_FINGERPRINT_MANIFEST_CACHE)))
    return normalized


def _write_git_fingerprint_manifest(path: Path, *, head: str, status: str, state, aux=None):
    _atomic_private_json(
        path,
        {
            "schema": GIT_FINGERPRINT_MANIFEST_SCHEMA,
            "head": str(head),
            "status": str(status),
            "state": dict(state),
            "aux": dict(aux or {}),
        },
    )
    cache_key = _snapshot_metadata_cache_key(path)
    if cache_key is not None:
        _GIT_FINGERPRINT_MANIFEST_CACHE.pop(cache_key, None)
        _GIT_FINGERPRINT_MANIFEST_CACHE[cache_key] = {
            "head": str(head or ""),
            "status": str(status or ""),
            "state": dict(state or {}),
            "aux": dict(aux or {}),
        }
        while len(_GIT_FINGERPRINT_MANIFEST_CACHE) > _GIT_FINGERPRINT_MANIFEST_CACHE_LIMIT:
            _GIT_FINGERPRINT_MANIFEST_CACHE.pop(next(iter(_GIT_FINGERPRINT_MANIFEST_CACHE)))


def _git_fingerprint_manifest_equals(cached, *, head: str, status: str, state, aux=None) -> bool:
    if not isinstance(cached, dict):
        return False
    if str(cached.get("head") or "") != str(head):
        return False
    if str(cached.get("status") or "") != str(status):
        return False
    if dict(cached.get("state") or {}) != dict(state or {}):
        return False
    return dict(cached.get("aux") or {}) == dict(aux or {})


def _empty_git_status_digest():
    return sha256_text(json.dumps([], sort_keys=True, separators=(",", ":"), ensure_ascii=True))


def _git_state_is_clean(state) -> bool:
    if not isinstance(state, dict) or str(state.get("kind") or "") != "git":
        return False
    return (
        str(state.get("staged") or "") == f"sha256:{_empty_git_status_digest()}"
        and str(state.get("unstaged") or "") == f"sha256:{_empty_git_status_digest()}"
        and not list(state.get("untracked") or ())
    )


def _clean_git_fingerprint_manifest(cached, *, head: str):
    if not isinstance(cached, dict):
        return None
    if str(cached.get("head") or "") != str(head):
        return None
    if str(cached.get("status") or "") != _empty_git_status_digest():
        return None
    state = dict(cached.get("state") or {})
    if not _git_state_is_clean(state):
        return None
    return state


def _git_clean_fingerprint_aux(cached, *, head: str):
    if not isinstance(cached, dict):
        return None
    if str(cached.get("head") or "") != str(head):
        return None
    if str(cached.get("status") or "") != _empty_git_status_digest():
        return None
    state = dict(cached.get("state") or {})
    if not _git_state_is_clean(state):
        return None
    aux = cached.get("aux")
    if not isinstance(aux, dict):
        return None
    return dict(aux)


def _normalize_git_clean_fastpath_files(raw_files):
    if not isinstance(raw_files, dict):
        return None
    normalized = {}
    for raw_rel, raw_signature in raw_files.items():
        rel = str(raw_rel or "").strip()
        signature = str(raw_signature or "").strip()
        if not rel or not signature:
            return None
        rel = PurePosixPath(rel).as_posix()
        if rel in {"", "."}:
            return None
        normalized[rel] = signature
    return normalized


def _normalize_git_clean_fastpath_directory_signatures(raw_directories):
    if not isinstance(raw_directories, dict):
        return None
    normalized = {}
    for raw_rel, raw_signature in raw_directories.items():
        rel = str(raw_rel or "").strip()
        signature = str(raw_signature or "").strip()
        if not signature:
            return None
        rel = PurePosixPath(rel or ".").as_posix()
        if rel == "":
            rel = "."
        normalized[rel] = signature
    return normalized


def _git_clean_fastpath_aux_payload(clean_worktree_files, clean_directory_signatures=None):
    normalized_files = _normalize_git_clean_fastpath_files(clean_worktree_files)
    if not normalized_files:
        return {}
    payload = {"clean_worktree_files": normalized_files}
    normalized_directories = _normalize_git_clean_fastpath_directory_signatures(clean_directory_signatures)
    if normalized_directories:
        payload["clean_directory_signatures"] = normalized_directories
    return payload


def _git_index_state_signature(top: Path):
    git_dir = _git_dir_path(top)
    if git_dir is None:
        return ""
    return _file_state_signature(git_dir / "index")


def _git_dir_path(top: Path) -> Path | None:
    git_path = top / ".git"
    if git_path.is_dir():
        return git_path
    try:
        data = git_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    prefix = "gitdir:"
    if not data.lower().startswith(prefix):
        return None
    target = data[len(prefix) :].strip()
    if not target:
        return None
    git_dir = Path(target)
    if not git_dir.is_absolute():
        git_dir = (top / git_dir).resolve(strict=False)
    return git_dir


def _git_head_state_signature(top: Path) -> str:
    git_dir = _git_dir_path(top)
    if git_dir is None:
        return ""
    try:
        head_text = (git_dir / "HEAD").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if not head_text:
        return ""
    if head_text.startswith("ref: "):
        ref_path = head_text[len("ref: ") :].strip()
        try:
            ref_text = (git_dir / Path(ref_path)).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return f"headref:{head_text}:missing"
        return f"headref:{head_text}:{ref_text}"
    return f"head:{head_text}"


def _git_state_is_unstaged_only(state) -> bool:
    if not isinstance(state, dict) or str(state.get("kind") or "") != "git":
        return False
    if str(state.get("staged") or "") != f"sha256:{_empty_git_status_digest()}":
        return False
    if not str(state.get("unstaged") or "") or str(state.get("unstaged") or "") == f"sha256:{_empty_git_status_digest()}":
        return False
    return not list(state.get("untracked") or ())


def _git_unstaged_only_fingerprint_aux(cached, *, head: str):
    if not isinstance(cached, dict):
        return None
    if str(cached.get("head") or "") != str(head):
        return None
    state = dict(cached.get("state") or {})
    if not _git_state_is_unstaged_only(state):
        return None
    aux = cached.get("aux")
    if not isinstance(aux, dict):
        return None
    normalized_clean_files = _normalize_git_clean_fastpath_files(aux.get("clean_worktree_files"))
    normalized_clean_content = _normalize_git_clean_fastpath_files(aux.get("clean_tracked_content"))
    normalized_clean_blob_oids = _normalize_git_clean_fastpath_files(aux.get("clean_tracked_blob_oids"))
    if not normalized_clean_files or not normalized_clean_content or not normalized_clean_blob_oids:
        return None
    if set(normalized_clean_files) != set(normalized_clean_content) or set(normalized_clean_files) != set(normalized_clean_blob_oids):
        return None
    index_signature = str(aux.get("git_index_signature") or "")
    if not index_signature:
        return None
    return {
        "clean_worktree_files": normalized_clean_files,
        "clean_tracked_content": normalized_clean_content,
        "clean_tracked_blob_oids": normalized_clean_blob_oids,
        "git_index_signature": index_signature,
    }


def _git_staged_fingerprint_aux(cached, *, head: str):
    if not isinstance(cached, dict):
        return None
    if str(cached.get("head") or "") != str(head):
        return None
    state = dict(cached.get("state") or {})
    if str(state.get("kind") or "") != "git":
        return None
    if str(state.get("staged") or "") == f"sha256:{_empty_git_status_digest()}":
        return None
    aux = cached.get("aux")
    if not isinstance(aux, dict):
        return None
    normalized_clean_files = _normalize_git_clean_fastpath_files(aux.get("clean_worktree_files"))
    normalized_clean_content = _normalize_git_clean_fastpath_files(aux.get("clean_tracked_content"))
    index_signature = str(aux.get("git_index_signature") or "")
    staged_entries = aux.get("staged_entries")
    if not normalized_clean_files or not normalized_clean_content or not index_signature or not isinstance(staged_entries, list):
        return None
    if set(normalized_clean_files) != set(normalized_clean_content):
        return None
    normalized_staged_entries = []
    for entry in staged_entries:
        if not isinstance(entry, dict):
            return None
        rel = PurePosixPath(str(entry.get("path") or "")).as_posix()
        code = str(entry.get("code") or "")
        if not rel or not code:
            return None
        normalized_entry = {
            "code": code,
            "path": rel,
        }
        mode = str(entry.get("mode") or "")
        blob_oid = str(entry.get("blob_oid") or "")
        stage = str(entry.get("stage") or "")
        if mode:
            normalized_entry["mode"] = mode
        if blob_oid:
            normalized_entry["blob_oid"] = blob_oid
        if stage:
            normalized_entry["stage"] = stage
        normalized_staged_entries.append(normalized_entry)
    return {
        "clean_worktree_files": normalized_clean_files,
        "clean_tracked_content": normalized_clean_content,
        "git_index_signature": index_signature,
        "staged_entries": normalized_staged_entries,
    }


def _git_clean_fastpath_match_state(
    top: Path,
    normalized_scope_paths,
    *,
    ignored,
    ignored_paths=None,
    aux=None,
):
    if not isinstance(aux, dict):
        return False
    expected_files = _normalize_git_clean_fastpath_files(aux.get("clean_worktree_files"))
    if not expected_files:
        return False
    expected_directories = _normalize_git_clean_fastpath_directory_signatures(aux.get("clean_directory_signatures"))
    if expected_directories:
        for rel, expected_signature in expected_directories.items():
            directory = top if rel in {"", "."} else (top / rel)
            if _directory_state_signature(directory) != expected_signature:
                return False
        for rel, expected_signature in expected_files.items():
            candidate = top / rel
            if candidate.is_symlink() or _should_skip(candidate, top, ignored, ignored_paths=ignored_paths):
                return False
            if _file_state_signature(candidate) != expected_signature:
                return False
        return True
    current_files = _git_clean_fastpath_capture_files(
        top,
        normalized_scope_paths,
        ignored=ignored,
        ignored_paths=ignored_paths,
    )
    if current_files is None:
        return False
    return current_files == expected_files


def _git_clean_fastpath_capture_files(top: Path, normalized_scope_paths, *, ignored, ignored_paths=None):
    scope_roots = []
    if normalized_scope_paths:
        for rel in normalized_scope_paths:
            scope_root = (top / rel).resolve(strict=False)
            if not scope_root.exists():
                return None
            scope_roots.append(scope_root)
    else:
        scope_roots.append(top)
    deduped_roots = []
    for scope_root in sorted(scope_roots, key=lambda item: (len(item.parts), str(item))):
        if any(scope_root == existing or scope_root.is_relative_to(existing) for existing in deduped_roots):
            continue
        deduped_roots.append(scope_root)
    current_files = {}
    for scope_root in deduped_roots:
        if scope_root.is_symlink():
            return None
        if scope_root.is_file():
            if _should_skip(scope_root, top, ignored, ignored_paths=ignored_paths):
                continue
            rel = scope_root.relative_to(top).as_posix()
            current_files[rel] = _file_state_signature(scope_root)
            if len(current_files) > SMALL_GIT_FINGERPRINT_FASTPATH_FILE_THRESHOLD:
                return None
            continue
        if not scope_root.is_dir():
            return None
        for current_root, dirnames, filenames in os.walk(scope_root):
            current_root_path = Path(current_root)
            kept_dirs = []
            for dirname in dirnames:
                candidate = current_root_path / dirname
                if candidate.is_symlink() or _should_skip(candidate, top, ignored, ignored_paths=ignored_paths):
                    continue
                kept_dirs.append(dirname)
            dirnames[:] = kept_dirs
            for filename in filenames:
                candidate = current_root_path / filename
                if candidate.is_symlink() or _should_skip(candidate, top, ignored, ignored_paths=ignored_paths):
                    continue
                if not candidate.is_file():
                    continue
                rel = candidate.relative_to(top).as_posix()
                current_files[rel] = _file_state_signature(candidate)
                if len(current_files) > SMALL_GIT_FINGERPRINT_FASTPATH_FILE_THRESHOLD:
                    return None
    return current_files


def _git_clean_fastpath_capture_content_signatures(
    top: Path,
    normalized_scope_paths,
    *,
    ignored,
    ignored_paths=None,
    cached_worktree_signatures=None,
):
    files = _git_clean_fastpath_capture_files(
        top,
        normalized_scope_paths,
        ignored=ignored,
        ignored_paths=ignored_paths,
    )
    if files is None:
        return None
    content_signatures = {}
    blob_oids = {}
    cached_worktree_signatures = dict(cached_worktree_signatures or {})
    for rel, state_signature in files.items():
        candidate = top / rel
        cached_record = cached_worktree_signatures.get(rel)
        if (
            isinstance(cached_record, dict)
            and str(cached_record.get("state_signature") or "") == str(state_signature)
            and str(cached_record.get("content_signature") or "")
        ):
            content_signatures[rel] = str(cached_record.get("content_signature") or "")
            cached_git_blob_oid = str(cached_record.get("git_blob_oid") or "")
            if cached_git_blob_oid:
                blob_oids[rel] = cached_git_blob_oid
                continue
            if str(content_signatures[rel]).startswith("sha256:"):
                try:
                    content = candidate.read_bytes()
                except OSError:
                    blob_oids[rel] = "unreadable"
                else:
                    blob_oids[rel] = _git_blob_oid_for_bytes(content)
            continue
        content_signature, _state_signature, git_blob_oid = _worktree_signatures_with_cache(candidate, cached_record)
        content_signatures[rel] = content_signature
        blob_oids[rel] = git_blob_oid
    return files, content_signatures, blob_oids


def _git_unstaged_only_fastpath_state(
    top: Path,
    normalized_scope_paths,
    *,
    head: str,
    ignored,
    ignored_paths=None,
    aux=None,
    cached_state_for_reuse=None,
):
    if not isinstance(aux, dict):
        return None
    if str(aux.get("git_index_signature") or "") != _git_index_state_signature(top):
        return None
    baseline_files = _normalize_git_clean_fastpath_files(aux.get("clean_worktree_files"))
    baseline_content = _normalize_git_clean_fastpath_files(aux.get("clean_tracked_content"))
    if not baseline_files or not baseline_content:
        return None
    if set(baseline_files) != set(baseline_content):
        return None
    current_files = _git_clean_fastpath_capture_files(
        top,
        normalized_scope_paths,
        ignored=ignored,
        ignored_paths=ignored_paths,
    )
    if current_files is None:
        return None
    worktree_signature_cache = _cached_worktree_signature_map(cached_state_for_reuse)
    unstaged_entries = []
    untracked_entries = []
    dirty_paths = []
    next_worktree_signature_cache = {}
    baseline_rel_paths = set(baseline_files)
    current_rel_paths = set(current_files)
    for rel in sorted(baseline_rel_paths):
        current_state_signature = str(current_files.get(rel) or "")
        baseline_state_signature = str(baseline_files.get(rel) or "")
        if rel in current_rel_paths and current_state_signature == baseline_state_signature:
            continue
        candidate = top / rel
        content_signature, state_signature, git_blob_oid = _worktree_signatures_with_cache(
            candidate,
            worktree_signature_cache.get(rel),
        )
        if state_signature:
            cached_signature_record = {
                "state_signature": state_signature,
                "content_signature": content_signature,
            }
            if git_blob_oid and git_blob_oid not in {"missing", "nonfile", "symlink", "unreadable"}:
                cached_signature_record["git_blob_oid"] = git_blob_oid
            next_worktree_signature_cache[rel] = cached_signature_record
        if str(content_signature or "") == str(baseline_content.get(rel) or ""):
            continue
        dirty_paths.append(rel)
        unstaged_entries.append(
            {
                "code": "D" if str(content_signature or "") == "missing" else "M",
                "path": rel,
                "signature": content_signature,
            }
        )
    for rel in sorted(current_rel_paths - baseline_rel_paths):
        candidate = top / rel
        content_signature, state_signature, git_blob_oid = _worktree_signatures_with_cache(
            candidate,
            worktree_signature_cache.get(rel),
        )
        if state_signature:
            cached_signature_record = {
                "state_signature": state_signature,
                "content_signature": content_signature,
            }
            if git_blob_oid and git_blob_oid not in {"missing", "nonfile", "symlink", "unreadable"}:
                cached_signature_record["git_blob_oid"] = git_blob_oid
            next_worktree_signature_cache[rel] = cached_signature_record
        dirty_paths.append(rel)
        untracked_entries.append((rel, content_signature))
    empty_status = f"sha256:{_empty_git_status_digest()}"
    state = {
        "kind": "git",
        "head": head,
        "staged": empty_status,
        "unstaged": f"sha256:{sha256_text(json.dumps(unstaged_entries, sort_keys=True, separators=(',', ':'), ensure_ascii=True))}",
        "untracked": sorted(untracked_entries),
    }
    if dirty_paths:
        state["dirty_paths"] = list(dirty_paths)
    if normalized_scope_paths:
        state["scope_paths"] = list(normalized_scope_paths)
    if next_worktree_signature_cache:
        state["worktree_signatures"] = next_worktree_signature_cache
    state["fingerprint_digest"] = sha256_text(
        json.dumps(
            {
                "kind": "git",
                "head": state["head"],
                "staged": state["staged"],
                "unstaged": state["unstaged"],
                "untracked": state["untracked"],
                **({"scope_paths": state["scope_paths"]} if "scope_paths" in state else {}),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return state


def _parse_git_diff_cached_raw_entries(output: bytes):
    if output is None:
        return None
    entries = []
    parts = output.split(b"\0")
    index = 0
    while index < len(parts):
        raw_meta = parts[index]
        if not raw_meta:
            index += 1
            continue
        if index + 1 >= len(parts):
            return None
        raw_path = parts[index + 1]
        index += 2
        text = raw_meta.decode("utf-8", errors="surrogateescape")
        if not text.startswith(":"):
            return None
        try:
            old_mode_token, new_mode, old_oid, new_oid, status_token = text.split()
        except ValueError:
            return None
        old_mode = str(old_mode_token or "")[1:]
        status_code = str(status_token or "")[:1]
        rel_path = raw_path.decode("utf-8", errors="surrogateescape").strip()
        if not status_code or not rel_path:
            return None
        entries.append(
            {
                "code": status_code,
                "path": PurePosixPath(rel_path).as_posix(),
                "old_mode": str(old_mode or ""),
                "new_mode": str(new_mode or ""),
                "old_oid": str(old_oid or ""),
                "new_oid": str(new_oid or ""),
            }
        )
    return entries


def _git_staged_small_repo_fastpath_state(
    top: Path,
    normalized_scope_paths,
    *,
    head: str,
    ignored,
    ignored_paths=None,
    aux=None,
    cached_state_for_reuse=None,
):
    if not isinstance(aux, dict):
        return None
    baseline_files = _normalize_git_clean_fastpath_files(aux.get("clean_worktree_files"))
    baseline_content = _normalize_git_clean_fastpath_files(aux.get("clean_tracked_content"))
    baseline_index_signature = str(aux.get("git_index_signature") or "")
    current_index_signature = _git_index_state_signature(top)
    if not baseline_files or not baseline_content or not baseline_index_signature or not current_index_signature:
        return None
    if set(baseline_files) != set(baseline_content):
        return None
    if current_index_signature == baseline_index_signature:
        return None
    current_files = _git_clean_fastpath_capture_files(
        top,
        normalized_scope_paths,
        ignored=ignored,
        ignored_paths=ignored_paths,
    )
    if current_files is None:
        return None
    diff_output = _run_git_optional(
        top,
        "diff-index",
        "--cached",
        "--raw",
        "-z",
        "--no-renames",
        "HEAD",
        "--",
        *_scope_pathspec(normalized_scope_paths),
    )
    raw_entries = _parse_git_diff_cached_raw_entries(diff_output)
    if raw_entries is None:
        return None
    staged_entries = []
    staged_by_path = {}
    for entry in raw_entries:
        rel = str(entry.get("path") or "")
        code = str(entry.get("code") or "")
        if not rel or not code:
            return None
        staged_record = {
            "code": code,
            "path": rel,
        }
        new_mode = str(entry.get("new_mode") or "")
        new_oid = str(entry.get("new_oid") or "")
        if code != "D" and new_mode and new_mode != "000000":
            staged_record["mode"] = new_mode
        if code != "D" and new_oid and new_oid != "0" * len(new_oid):
            staged_record["blob_oid"] = new_oid
            staged_record["stage"] = "0"
        staged_entries.append(staged_record)
        staged_by_path[rel] = entry
    worktree_signature_cache = _cached_worktree_signature_map(cached_state_for_reuse)
    next_worktree_signature_cache = {}
    unstaged_entries = []
    untracked_entries = []
    dirty_paths = set()
    current_rel_paths = set(current_files)
    baseline_rel_paths = set(baseline_files)
    for rel in sorted(current_rel_paths | baseline_rel_paths):
        candidate = top / rel
        current_state_signature = str(current_files.get(rel) or "")
        baseline_state_signature = str(baseline_files.get(rel) or "")
        staged_entry = staged_by_path.get(rel)
        if staged_entry is not None:
            content_signature, state_signature, git_blob_oid = _worktree_signatures_with_cache(
                candidate,
                worktree_signature_cache.get(rel),
            )
            if state_signature:
                cached_signature_record = {
                    "state_signature": state_signature,
                    "content_signature": content_signature,
                }
                if git_blob_oid and git_blob_oid not in {"missing", "nonfile", "symlink", "unreadable"}:
                    cached_signature_record["git_blob_oid"] = git_blob_oid
                next_worktree_signature_cache[rel] = cached_signature_record
            dirty_paths.add(rel)
            code = str(staged_entry.get("code") or "")
            if code == "D":
                if str(content_signature or "") not in {"missing", str(baseline_content.get(rel) or "")}:
                    unstaged_entries.append({"code": "M", "path": rel, "signature": content_signature})
                continue
            new_oid = str(staged_entry.get("new_oid") or "")
            if new_oid and new_oid != "0" * len(new_oid):
                if str(git_blob_oid or "") not in {"missing", "nonfile", "symlink", "unreadable", new_oid}:
                    unstaged_entries.append({"code": "M", "path": rel, "signature": content_signature})
            continue
        if rel not in baseline_rel_paths and rel in current_rel_paths:
            content_signature, state_signature, git_blob_oid = _worktree_signatures_with_cache(
                candidate,
                worktree_signature_cache.get(rel),
            )
            if state_signature:
                cached_signature_record = {
                    "state_signature": state_signature,
                    "content_signature": content_signature,
                }
                if git_blob_oid and git_blob_oid not in {"missing", "nonfile", "symlink", "unreadable"}:
                    cached_signature_record["git_blob_oid"] = git_blob_oid
                next_worktree_signature_cache[rel] = cached_signature_record
            dirty_paths.add(rel)
            untracked_entries.append((rel, content_signature))
            continue
        if current_state_signature and baseline_state_signature and current_state_signature != baseline_state_signature:
            content_signature, state_signature, git_blob_oid = _worktree_signatures_with_cache(
                candidate,
                worktree_signature_cache.get(rel),
            )
            if state_signature:
                cached_signature_record = {
                    "state_signature": state_signature,
                    "content_signature": content_signature,
                }
                if git_blob_oid and git_blob_oid not in {"missing", "nonfile", "symlink", "unreadable"}:
                    cached_signature_record["git_blob_oid"] = git_blob_oid
                next_worktree_signature_cache[rel] = cached_signature_record
            if str(content_signature or "") != str(baseline_content.get(rel) or ""):
                dirty_paths.add(rel)
                unstaged_entries.append({"code": "M", "path": rel, "signature": content_signature})
    empty_status = f"sha256:{_empty_git_status_digest()}"
    state = {
        "kind": "git",
        "head": head,
        "staged": f"sha256:{sha256_text(json.dumps(sorted(staged_entries, key=lambda item: (item.get('path', ''), item.get('source_path', ''), item.get('code', ''))), sort_keys=True, separators=(',', ':'), ensure_ascii=True))}",
        "unstaged": f"sha256:{sha256_text(json.dumps(sorted(unstaged_entries, key=lambda item: (item.get('path', ''), item.get('code', ''))), sort_keys=True, separators=(',', ':'), ensure_ascii=True))}",
        "untracked": sorted(untracked_entries),
    }
    if dirty_paths:
        state["dirty_paths"] = sorted(dirty_paths)
    if normalized_scope_paths:
        state["scope_paths"] = list(normalized_scope_paths)
    if next_worktree_signature_cache:
        state["worktree_signatures"] = next_worktree_signature_cache
    if staged_entries:
        state["_staged_entries_detail"] = [dict(entry) for entry in staged_entries]
    state["fingerprint_digest"] = sha256_text(
        json.dumps(
            {
                "kind": "git",
                "head": state["head"],
                "staged": state["staged"],
                "unstaged": state["unstaged"],
                "untracked": state["untracked"],
                **({"scope_paths": state["scope_paths"]} if "scope_paths" in state else {}),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return state


def _git_staged_cached_fastpath_state(
    top: Path,
    normalized_scope_paths,
    *,
    head: str,
    ignored,
    ignored_paths=None,
    aux=None,
    cached_state_for_reuse=None,
):
    if not isinstance(aux, dict):
        return None
    baseline_files = _normalize_git_clean_fastpath_files(aux.get("clean_worktree_files"))
    baseline_content = _normalize_git_clean_fastpath_files(aux.get("clean_tracked_content"))
    cached_index_signature = str(aux.get("git_index_signature") or "")
    staged_entries = aux.get("staged_entries")
    current_index_signature = _git_index_state_signature(top)
    if (
        not baseline_files
        or not baseline_content
        or not cached_index_signature
        or not current_index_signature
        or current_index_signature != cached_index_signature
        or not isinstance(staged_entries, list)
    ):
        return None
    if set(baseline_files) != set(baseline_content):
        return None
    current_files = _git_clean_fastpath_capture_files(
        top,
        normalized_scope_paths,
        ignored=ignored,
        ignored_paths=ignored_paths,
    )
    if current_files is None:
        return None
    normalized_staged_entries = []
    staged_by_path = {}
    for entry in staged_entries:
        if not isinstance(entry, dict):
            return None
        rel = PurePosixPath(str(entry.get("path") or "")).as_posix()
        code = str(entry.get("code") or "")
        if not rel or not code:
            return None
        normalized_entry = {"code": code, "path": rel}
        mode = str(entry.get("mode") or "")
        blob_oid = str(entry.get("blob_oid") or "")
        stage = str(entry.get("stage") or "")
        if mode:
            normalized_entry["mode"] = mode
        if blob_oid:
            normalized_entry["blob_oid"] = blob_oid
        if stage:
            normalized_entry["stage"] = stage
        normalized_staged_entries.append(normalized_entry)
        staged_by_path[rel] = normalized_entry
    worktree_signature_cache = _cached_worktree_signature_map(cached_state_for_reuse)
    next_worktree_signature_cache = {}
    unstaged_entries = []
    untracked_entries = []
    dirty_paths = set()
    current_rel_paths = set(current_files)
    baseline_rel_paths = set(baseline_files)
    for rel in sorted(current_rel_paths | baseline_rel_paths):
        candidate = top / rel
        current_state_signature = str(current_files.get(rel) or "")
        baseline_state_signature = str(baseline_files.get(rel) or "")
        staged_entry = staged_by_path.get(rel)
        if staged_entry is not None:
            content_signature, state_signature, git_blob_oid = _worktree_signatures_with_cache(
                candidate,
                worktree_signature_cache.get(rel),
            )
            if state_signature:
                cached_signature_record = {
                    "state_signature": state_signature,
                    "content_signature": content_signature,
                }
                if git_blob_oid and git_blob_oid not in {"missing", "nonfile", "symlink", "unreadable"}:
                    cached_signature_record["git_blob_oid"] = git_blob_oid
                next_worktree_signature_cache[rel] = cached_signature_record
            dirty_paths.add(rel)
            code = str(staged_entry.get("code") or "")
            if code == "D":
                if str(content_signature or "") not in {"missing", str(baseline_content.get(rel) or "")}:
                    unstaged_entries.append({"code": "M", "path": rel, "signature": content_signature})
                continue
            staged_blob_oid = str(staged_entry.get("blob_oid") or "")
            if staged_blob_oid and str(git_blob_oid or "") not in {"missing", "nonfile", "symlink", "unreadable", staged_blob_oid}:
                unstaged_entries.append({"code": "M", "path": rel, "signature": content_signature})
            continue
        if rel in baseline_rel_paths and rel not in current_rel_paths:
            dirty_paths.add(rel)
            unstaged_entries.append({"code": "D", "path": rel, "signature": "missing"})
            continue
        if rel not in baseline_rel_paths and rel in current_rel_paths:
            content_signature, state_signature, git_blob_oid = _worktree_signatures_with_cache(
                candidate,
                worktree_signature_cache.get(rel),
            )
            if state_signature:
                cached_signature_record = {
                    "state_signature": state_signature,
                    "content_signature": content_signature,
                }
                if git_blob_oid and git_blob_oid not in {"missing", "nonfile", "symlink", "unreadable"}:
                    cached_signature_record["git_blob_oid"] = git_blob_oid
                next_worktree_signature_cache[rel] = cached_signature_record
            dirty_paths.add(rel)
            untracked_entries.append((rel, content_signature))
            continue
        if current_state_signature and baseline_state_signature and current_state_signature != baseline_state_signature:
            content_signature, state_signature, git_blob_oid = _worktree_signatures_with_cache(
                candidate,
                worktree_signature_cache.get(rel),
            )
            if state_signature:
                cached_signature_record = {
                    "state_signature": state_signature,
                    "content_signature": content_signature,
                }
                if git_blob_oid and git_blob_oid not in {"missing", "nonfile", "symlink", "unreadable"}:
                    cached_signature_record["git_blob_oid"] = git_blob_oid
                next_worktree_signature_cache[rel] = cached_signature_record
            if str(content_signature or "") != str(baseline_content.get(rel) or ""):
                dirty_paths.add(rel)
                unstaged_entries.append({"code": "M", "path": rel, "signature": content_signature})
    state = {
        "kind": "git",
        "head": head,
        "staged": f"sha256:{sha256_text(json.dumps(sorted(normalized_staged_entries, key=lambda item: (item.get('path', ''), item.get('source_path', ''), item.get('code', ''))), sort_keys=True, separators=(',', ':'), ensure_ascii=True))}",
        "unstaged": f"sha256:{sha256_text(json.dumps(sorted(unstaged_entries, key=lambda item: (item.get('path', ''), item.get('code', ''))), sort_keys=True, separators=(',', ':'), ensure_ascii=True))}",
        "untracked": sorted(untracked_entries),
    }
    if dirty_paths:
        state["dirty_paths"] = sorted(dirty_paths)
    if normalized_scope_paths:
        state["scope_paths"] = list(normalized_scope_paths)
    if next_worktree_signature_cache:
        state["worktree_signatures"] = next_worktree_signature_cache
    state["fingerprint_digest"] = sha256_text(
        json.dumps(
            {
                "kind": "git",
                "head": state["head"],
                "staged": state["staged"],
                "unstaged": state["unstaged"],
                "untracked": state["untracked"],
                **({"scope_paths": state["scope_paths"]} if "scope_paths" in state else {}),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return state


def _repository_state_fingerprint_from_states(states):
    normalized = []
    reusable = True
    for state in states:
        if not isinstance(state, dict):
            reusable = False
            normalized.append(state)
            continue
        if str(state.get("kind") or "") == "git":
            digest = str(state.get("fingerprint_digest") or "").strip()
            if digest:
                normalized.append({"kind": "git", "fingerprint_digest": digest})
                continue
        reusable = False
        normalized.append(state)
    encoded = json.dumps(normalized if reusable else states, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{sha256_text(encoded)}"


def _cached_git_source_files(
    root: Path,
    ignored: set[str],
    ignored_paths=None,
    previous_record=None,
    *,
    repository_state_fingerprint=None,
    git_probe_cache=None,
    prefer_inventory_delta=False,
    touched_paths_hint=None,
):
    if not previous_record:
        return None, None
    previous_files = [str(item) for item in previous_record.get("files") or () if str(item)]
    root_resolved = root.resolve(strict=False)
    filter_key = _discovery_filter_key(root, ignored, ignored_paths=ignored_paths)
    previous_filter_key = str(previous_record.get("filter_key") or "")
    filter_matches = not previous_filter_key or previous_filter_key == filter_key
    top = _git_top(root)
    if top is None:
        return None, None
    scope_rel = _git_scope_rel(root, top)
    if scope_rel is None:
        return None, None
    if (
        str(previous_record.get("git_top") or "") != str(top)
        or str(previous_record.get("scope_rel") or "") != scope_rel
    ):
        return None, None
    previous_repository_state_fingerprint = str(previous_record.get("repository_state_fingerprint") or "")
    normalized_touched_paths_hint = _normalize_touched_paths_hint(touched_paths_hint)
    if (
        repository_state_fingerprint
        and previous_repository_state_fingerprint
        and previous_repository_state_fingerprint == str(repository_state_fingerprint)
        and filter_matches
        and not prefer_inventory_delta
    ):
        current = [str(item) for item in previous_record.get("files") or () if str(item)]
        return [root_resolved / rel for rel in current], {
            "git_top": str(top),
            "scope_rel": scope_rel,
            "scope_oid": str(previous_record.get("scope_oid") or ""),
            "repository_state_fingerprint": previous_repository_state_fingerprint,
            "filter_key": filter_key,
            "files": current,
        }
    if normalized_touched_paths_hint and filter_matches:
        current = set(previous_files)
        added = []
        for rel in normalized_touched_paths_hint:
            candidate = root_resolved / rel
            if (
                candidate.is_symlink()
                or not candidate.is_file()
                or _should_skip(candidate, root, ignored, ignored_paths=ignored_paths)
                or not language_for_path(candidate)
            ):
                current.discard(rel)
            else:
                if rel not in current:
                    added.append(rel)
                current.add(rel)
        ordered_files = [rel for rel in previous_files if rel in current]
        ordered_files.extend(rel for rel in added if rel not in ordered_files)
        files = [root_resolved / rel for rel in ordered_files]
        return files, {
            "git_top": str(top),
            "scope_rel": scope_rel,
            "scope_oid": str(previous_record.get("scope_oid") or ""),
            "repository_state_fingerprint": str(repository_state_fingerprint or ""),
            "filter_key": filter_key,
            "files": ordered_files,
        }
    cached_status_snapshot = _cached_scoped_status_snapshot(top, scope_rel, git_probe_cache=git_probe_cache)
    if (
        prefer_inventory_delta
        and filter_matches
        and _cached_scoped_status_preserves_inventory(
            scope_rel,
            cached_status_snapshot,
            ignored=ignored,
            ignored_paths=ignored_paths,
            top=top,
        )
    ):
        current = [str(item) for item in previous_record.get("files") or () if str(item)]
        return [root_resolved / rel for rel in current], {
            "git_top": str(top),
            "scope_rel": scope_rel,
            "scope_oid": str(previous_record.get("scope_oid") or ""),
            "repository_state_fingerprint": previous_repository_state_fingerprint,
            "filter_key": filter_key,
            "files": current,
        }
    scope_oid = _git_scope_inventory_identity(top, scope_rel, git_probe_cache=git_probe_cache)
    if not scope_oid or scope_oid != str(previous_record.get("scope_oid") or ""):
        return None, None
    normalized_scope_paths = [] if scope_rel in {"", "."} else [scope_rel]
    if prefer_inventory_delta:
        normalized_scope_paths = [] if scope_rel in {"", "."} else [scope_rel]
        touched = _scoped_git_inventory_delta_paths(
            top,
            normalized_scope_paths,
            ignored=ignored,
            ignored_paths=ignored_paths,
            git_probe_cache=git_probe_cache,
        )
        if touched is None:
            return None, None
        scope_prefix = "" if scope_rel in {"", "."} else f"{scope_rel.rstrip('/')}/"
        scoped_touched = set()
        for normalized in touched:
            if scope_prefix:
                if normalized == scope_rel:
                    rel = "."
                elif normalized.startswith(scope_prefix):
                    rel = normalized[len(scope_prefix) :]
                else:
                    continue
            else:
                rel = normalized
            if rel in {"", "."}:
                continue
            scoped_touched.add(rel)
    else:
        touched = _git_scope_status_paths(
            root,
            top,
            scope_rel,
            ignored=ignored,
            ignored_paths=ignored_paths,
            git_probe_cache=git_probe_cache,
        )
        if touched is None:
            return None, None
        scoped_touched = set(touched)
    if filter_matches and not scoped_touched:
        current = [str(item) for item in previous_record.get("files") or () if str(item)]
        return [root_resolved / rel for rel in current], {
            "git_top": str(top),
            "scope_rel": scope_rel,
            "scope_oid": scope_oid,
            "repository_state_fingerprint": previous_repository_state_fingerprint,
            "filter_key": filter_key,
            "files": current,
        }
    current = set(previous_files)
    added = []
    for rel in sorted(scoped_touched):
        candidate = root_resolved / rel
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or _should_skip(candidate, root, ignored, ignored_paths=ignored_paths)
            or not language_for_path(candidate)
        ):
            current.discard(rel)
        else:
            if rel not in current:
                added.append(rel)
            current.add(rel)
    ordered_files = [rel for rel in previous_files if rel in current]
    ordered_files.extend(rel for rel in added if rel not in ordered_files)
    files = []
    for rel in ordered_files:
        candidate = root_resolved / rel
        if not filter_matches and (
            candidate.is_symlink()
            or not candidate.is_file()
            or _should_skip(candidate, root, ignored, ignored_paths=ignored_paths)
            or not language_for_path(candidate)
        ):
            continue
        files.append(candidate)
    ordered_files = [path.relative_to(root_resolved).as_posix() for path in files]
    return files, {
        "git_top": str(top),
        "scope_rel": scope_rel,
        "scope_oid": scope_oid,
        "repository_state_fingerprint": previous_repository_state_fingerprint,
        "filter_key": filter_key,
        "files": ordered_files,
    }


def discover_source_files(
    discovered,
    excluded_dir_names=None,
    excluded_paths=None,
    *,
    cache_dir=None,
    repository_state_fingerprint=None,
    git_probe_cache=None,
    prefer_inventory_delta=False,
    touched_paths_hint=None,
):
    """Return input-backed source files with stable repository-relative paths."""

    git_probe_cache = _effective_git_probe_cache(git_probe_cache)
    ignored = set(DEFAULT_IGNORE_DIRS)
    ignored.update(excluded_dir_names or ())
    ignored_paths = {Path(path).resolve(strict=False) for path in (excluded_paths or ())}
    discovery_manifest_path = _discovery_manifest_path(cache_dir) if cache_dir is not None else None
    shared_discovery_manifest_path = _shared_discovery_manifest_path(create=False) if cache_dir is not None else None
    previous_discovery_manifest = {}
    previous_discovery_manifest_loaded = False

    def ensure_previous_discovery_manifest():
        nonlocal previous_discovery_manifest, previous_discovery_manifest_loaded
        if previous_discovery_manifest_loaded:
            return previous_discovery_manifest
        previous_discovery_manifest_loaded = True
        previous_discovery_manifest = (
            _load_preferred_discovery_working_manifest(discovery_manifest_path, shared_discovery_manifest_path)
            if discovery_manifest_path is not None or shared_discovery_manifest_path is not None
            else None
        ) or {}
        return previous_discovery_manifest
    fingerprint_discovery_manifest = {}
    if cache_dir is not None and repository_state_fingerprint:
        fingerprint_discovery_manifest = (
            _load_preferred_discovery_working_manifest(
                _discovery_fingerprint_manifest_path(Path(cache_dir), str(repository_state_fingerprint)),
                _shared_discovery_fingerprint_manifest_path(str(repository_state_fingerprint), create=False),
            )
            or {}
        )
    next_discovery_manifest = {}
    next_fingerprint_discovery_manifest = {}
    files = []
    seen = set()
    normalized_touched_paths_hint = _normalize_touched_paths_hint(touched_paths_hint)
    if not normalized_touched_paths_hint and len(discovered or ()) == 1:
        input_path = discovered[0].get("path")
        if input_path is not None:
            input_path = Path(input_path)
            probe_root = input_path if input_path.is_dir() else input_path.parent
            git_top = _git_top(probe_root)
            if git_top is not None:
                scope_rel = _git_scope_rel(probe_root, git_top)
                normalized_scope = [] if scope_rel in {"", "."} else [scope_rel]
                cached_dirty_paths = _cached_git_fastpath_dirty_paths(
                    git_top,
                    normalized_scope,
                    git_probe_cache=git_probe_cache,
                )
                if cached_dirty_paths is not None:
                    if scope_rel in {"", "."}:
                        normalized_touched_paths_hint = tuple(cached_dirty_paths)
                    else:
                        prefix = f"{scope_rel.rstrip('/')}/"
                        normalized_touched_paths_hint = _normalize_touched_paths_hint(
                            "." if rel == scope_rel else rel[len(prefix):]
                            for rel in cached_dirty_paths
                            if rel == scope_rel or rel.startswith(prefix)
                        )
    include_tracked_signatures = not bool(repository_state_fingerprint)
    for item in discovered:
        path = item.get("path")
        if path is None:
            continue
        path = Path(path)
        if path.is_dir():
            root_key = str(path.resolve(strict=False))
            root_path = Path(root_key)
            filter_key = _discovery_filter_key(path, ignored, ignored_paths=ignored_paths)
            exact_process_record = None
            if repository_state_fingerprint:
                exact_process_record = _load_discovery_exact_process_cache(
                    path,
                    repository_state_fingerprint,
                    filter_key,
                )
            if exact_process_record is not None:
                recorded_files = [str(item) for item in (exact_process_record.get("files") or ()) if isinstance(item, str)]
                for rel in recorded_files:
                    candidate = root_path / rel
                    key = (str(candidate), rel)
                    if key in seen:
                        continue
                    seen.add(key)
                    files.append((item, candidate, rel))
                _store_discovery_record_process_cache(root_key, exact_process_record)
                next_discovery_manifest[root_key] = dict(exact_process_record)
                if repository_state_fingerprint:
                    next_fingerprint_discovery_manifest[root_key] = dict(exact_process_record)
                continue
            exact_manifest_record = None
            if repository_state_fingerprint and fingerprint_discovery_manifest:
                candidate_record = fingerprint_discovery_manifest.get(root_key)
                if _exact_discovery_record_matches(
                    candidate_record,
                    repository_state_fingerprint=repository_state_fingerprint,
                    filter_key=filter_key,
                ):
                    exact_manifest_record = dict(candidate_record)
            if exact_manifest_record is not None:
                recorded_files = [
                    str(item) for item in (exact_manifest_record.get("files") or ()) if isinstance(item, str)
                ]
                for rel in recorded_files:
                    candidate = root_path / rel
                    key = (str(candidate), rel)
                    if key in seen:
                        continue
                    seen.add(key)
                    files.append((item, candidate, rel))
                _store_discovery_record_process_cache(root_key, exact_manifest_record)
                _store_discovery_exact_process_cache(
                    path,
                    repository_state_fingerprint,
                    filter_key,
                    exact_manifest_record,
                )
                next_discovery_manifest[root_key] = dict(exact_manifest_record)
                next_fingerprint_discovery_manifest[root_key] = dict(exact_manifest_record)
                continue
            cached_candidates = None
            discovery_record = None
            candidates_prevalidated = False
            previous_record = ensure_previous_discovery_manifest().get(root_key)
            if previous_record is None and fingerprint_discovery_manifest:
                previous_record = fingerprint_discovery_manifest.get(root_key)
            if previous_record is None:
                previous_record = _load_discovery_record_process_cache(root_key)
            if cache_dir is not None:
                cached_candidates, discovery_record = _cached_git_source_files(
                    path,
                    ignored,
                    ignored_paths=ignored_paths,
                    previous_record=previous_record,
                    repository_state_fingerprint=repository_state_fingerprint,
                    git_probe_cache=git_probe_cache,
                    prefer_inventory_delta=prefer_inventory_delta,
                    touched_paths_hint=normalized_touched_paths_hint,
                )
                if cached_candidates is None:
                    cached_candidates, discovery_record = _cached_non_git_source_files(
                        path,
                        ignored,
                        ignored_paths=ignored_paths,
                        previous_record=previous_record,
                    )
            if cached_candidates is not None:
                candidates = cached_candidates
                candidates_prevalidated = True
            else:
                metadata_candidates = _cached_metadata_source_candidates(
                    path,
                    ignored,
                    ignored_paths=ignored_paths,
                    git_probe_cache=git_probe_cache,
                )
                if metadata_candidates is not None:
                    candidates = metadata_candidates
                    candidates_prevalidated = True
                else:
                    candidates = list(
                        _iter_source_candidates(
                            path,
                            ignored,
                            ignored_paths=ignored_paths,
                            git_probe_cache=git_probe_cache,
                            include_tracked_signatures=include_tracked_signatures,
                        )
                    )
                    _store_metadata_source_candidates(
                        path,
                        ignored,
                        candidates,
                        ignored_paths=ignored_paths,
                        git_probe_cache=git_probe_cache,
                    )
            discovered_rel_paths = []
            cached_rel_paths = None
            if cached_candidates is not None and isinstance(discovery_record, dict):
                recorded_files = [str(item) for item in (discovery_record.get("files") or ()) if isinstance(item, str)]
                if len(recorded_files) == len(candidates):
                    cached_rel_paths = recorded_files
            for index, candidate in enumerate(candidates):
                if not candidates_prevalidated and not language_for_path(candidate):
                    continue
                if cached_rel_paths is not None:
                    rel = cached_rel_paths[index]
                else:
                    rel = candidate.relative_to(path).as_posix()
                discovered_rel_paths.append(rel)
                candidate_key = str(candidate) if candidate.is_absolute() else str(candidate.resolve(strict=False))
                key = (candidate_key, rel)
                if key in seen:
                    continue
                seen.add(key)
                files.append((item, candidate, rel))
            if cache_dir is not None:
                if discovery_record is None:
                    top = _git_top(path)
                    scope_rel = _git_scope_rel(path, top) if top is not None else None
                    scope_oid = (
                        _git_scope_inventory_identity(top, scope_rel, git_probe_cache=git_probe_cache)
                        if top is not None and scope_rel is not None
                        else ""
                    )
                    discovery_record = {
                        "git_top": str(top) if top is not None else "",
                        "scope_rel": scope_rel or "",
                        "scope_oid": scope_oid or "",
                        "repository_state_fingerprint": str(repository_state_fingerprint or ""),
                        "filter_key": filter_key,
                        "files": list(discovered_rel_paths),
                        "dir_signatures": _directory_signatures_for_files(path, discovered_rel_paths)
                        if top is None
                        else {},
                    }
                else:
                    discovery_record = dict(discovery_record)
                    discovery_record["repository_state_fingerprint"] = str(repository_state_fingerprint or "")
                    discovery_record["filter_key"] = filter_key
                    discovery_record["files"] = list(discovered_rel_paths)
                    if not str(discovery_record.get("git_top") or "").strip():
                        discovery_record["dir_signatures"] = _directory_signatures_for_files(path, discovered_rel_paths)
                if repository_state_fingerprint:
                    next_fingerprint_discovery_manifest[root_key] = dict(discovery_record)
                working_manifest_record = dict(discovery_record)
                if (
                    str(working_manifest_record.get("git_top") or "").strip()
                    and previous_record is not None
                    and _discovery_record_inventory_equal(previous_record, working_manifest_record)
                ):
                    working_manifest_record["repository_state_fingerprint"] = str(
                        (previous_record or {}).get("repository_state_fingerprint") or ""
                    )
                next_discovery_manifest[root_key] = working_manifest_record
                _store_discovery_record_process_cache(root_key, discovery_record)
                if repository_state_fingerprint:
                    _store_discovery_exact_process_cache(
                        path,
                        repository_state_fingerprint,
                        filter_key,
                        discovery_record,
                    )
        elif (
            not path.is_symlink()
            and path.is_file()
            and language_for_path(path)
            and not _should_skip(path, path.parent, ignored, ignored_paths=ignored_paths)
        ):
            key = (str(path.resolve()), path.name)
            if key not in seen:
                seen.add(key)
                files.append((item, path, path.name))
    if discovery_manifest_path is not None or shared_discovery_manifest_path is not None:
        previous_discovery_manifest = ensure_previous_discovery_manifest()
        if next_discovery_manifest != previous_discovery_manifest:
            _write_preferred_discovery_working_manifest(
                discovery_manifest_path,
                _shared_discovery_manifest_path(create=True),
                next_discovery_manifest,
            )
    if cache_dir is not None and repository_state_fingerprint:
        if next_fingerprint_discovery_manifest != fingerprint_discovery_manifest:
            _write_preferred_discovery_working_manifest(
                _discovery_fingerprint_manifest_path(Path(cache_dir), str(repository_state_fingerprint)),
                _shared_discovery_fingerprint_manifest_path(str(repository_state_fingerprint), create=True),
                next_fingerprint_discovery_manifest,
            )
    return files


def _python_markers(text: str):
    top_level = []
    has_indented_defs = False
    newline_offsets = [match.start() for match in re.finditer(r"\n", text)]
    for match in PYTHON_TOP_LEVEL_PATTERN.finditer(text):
        indent = str(match.group("indent") or "")
        name = str(match.group("name") or "")
        if not name:
            continue
        line = bisect.bisect_right(newline_offsets, match.start()) + 1
        if indent:
            has_indented_defs = True
            continue
        top_level.append((max(1, line), name))
    if not has_indented_defs:
        return top_level
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return top_level
    markers = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            markers.append((max(1, node.lineno), node.name))
        elif isinstance(node, ast.ClassDef):
            markers.append((max(1, node.lineno), node.name))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    markers.append((max(1, child.lineno), f"{node.name}.{child.name}"))
    return markers


def _regex_markers(text: str, language: str):
    pattern = SYMBOL_PATTERNS.get(language)
    if pattern is None:
        return []
    newline_offsets = [match.start() for match in re.finditer(r"\n", text)]
    markers = []
    for match in pattern.finditer(text):
        name = next((group for group in match.groups() if group), "")
        if not name:
            continue
        line = bisect.bisect_right(newline_offsets, match.start()) + 1
        markers.append((line, name))
    return markers


def symbol_markers(text: str, language: str):
    markers = _python_markers(text) if language == "python" else _regex_markers(text, language)
    deduped = {}
    for line, symbol in markers:
        deduped.setdefault(int(line), str(symbol))
    return sorted(deduped.items())


def _chunk_regions(lines, markers, max_lines, overlap):
    starts_by_line = {1: ""}
    for line, symbol in markers:
        starts_by_line[line] = symbol
    starts = sorted(starts_by_line.items())
    for index, (region_start, symbol) in enumerate(starts):
        region_end = starts[index + 1][0] - 1 if index + 1 < len(starts) else len(lines)
        cursor = region_start
        while cursor <= region_end:
            end = min(region_end, cursor + max_lines - 1)
            content = "\n".join(lines[cursor - 1 : end])
            if content.strip():
                yield cursor, end, symbol, content
            if end >= region_end:
                break
            cursor = max(cursor + 1, end - overlap + 1)


def _split_utf8_chunks(content, max_bytes=MAX_CHUNK_BYTES):
    encoded = content.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return [content]
    chunks = []
    start = 0
    while start < len(encoded):
        end = min(len(encoded), start + max_bytes)
        if end < len(encoded):
            while end > start and encoded[end] & 0xC0 == 0x80:
                end -= 1
        if end <= start:
            end = min(len(encoded), start + max_bytes)
        chunks.append(encoded[start:end].decode("utf-8", errors="strict"))
        start = end
    return chunks


def _file_chunk_cache_key(
    *,
    source_namespace: str,
    rel_path: str,
    language: str,
    text_hash: str,
    input_id: str,
    input_type: str,
    classification: str,
    max_lines: int,
    overlap: int,
):
    encoded = json.dumps(
        {
            "schema": FILE_CHUNK_CACHE_SCHEMA,
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "source_namespace": source_namespace,
            "repository_path": rel_path,
            "language": language,
            "text_hash": text_hash,
            "input_id": input_id,
            "input_type": input_type,
            "classification": classification,
            "max_lines": int(max_lines),
            "overlap": int(overlap),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256_text(encoded)


def _file_chunk_cache_path(cache_dir: Path, cache_key: str):
    cache_root = _private_cache_dir(Path(cache_dir) / "file-chunk-cache")
    return cache_root / f"chunks-{cache_key}.json"


def _shared_repo_inspection_cache_root(create=False):
    configured = os.environ.get("BROKER_REPO_INSPECTION_SHARED_CACHE_DIR", "").strip()
    if not configured:
        return None
    root = Path(configured).expanduser()
    try:
        return _private_cache_dir(root) if create else root.expanduser().absolute()
    except OSError:
        return None


def _shared_file_chunk_cache_path(cache_key: str, *, create=False):
    cache_root = _shared_repo_inspection_cache_root(create=create)
    if cache_root is None:
        return None
    cache_root = cache_root / "file-chunk-cache"
    try:
        cache_root = _private_cache_dir(cache_root) if create else cache_root.expanduser().absolute()
    except OSError:
        return None
    return cache_root / f"chunks-{cache_key}.json"


def _shared_file_chunk_snapshot_path(repository_state_fingerprint: str, build_config_digest: str, *, create=False):
    cache_root = _shared_repo_inspection_cache_root(create=create)
    if cache_root is None:
        return None
    cache_root = cache_root / "file-chunk-snapshots"
    try:
        cache_root = _private_cache_dir(cache_root) if create else cache_root.expanduser().absolute()
    except OSError:
        return None
    safe_repository = str(repository_state_fingerprint or "").replace(":", "_")
    safe_config = str(build_config_digest or "").replace(":", "_")
    return cache_root / f"snapshot-{safe_repository}-{safe_config}.json"


def _shared_file_chunk_snapshot_metadata_path(repository_state_fingerprint: str, build_config_digest: str, *, create=False):
    cache_root = _shared_repo_inspection_cache_root(create=create)
    if cache_root is None:
        return None
    cache_root = cache_root / "file-chunk-snapshots"
    try:
        cache_root = _private_cache_dir(cache_root) if create else cache_root.expanduser().absolute()
    except OSError:
        return None
    safe_repository = str(repository_state_fingerprint or "").replace(":", "_")
    safe_config = str(build_config_digest or "").replace(":", "_")
    return cache_root / f"snapshot-{safe_repository}-{safe_config}-metadata.json"


def _symbol_marker_cache_key(language: str, text_hash: str):
    return sha256_text(
        json.dumps(
            {"language": str(language or ""), "text_hash": str(text_hash or "")},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )


def _symbol_marker_cache_path(cache_dir: Path, cache_key: str):
    cache_root = _private_cache_dir(Path(cache_dir) / "symbol-marker-cache")
    return cache_root / f"markers-{cache_key}.json"


def _shared_symbol_marker_cache_path(cache_key: str, *, create=False):
    cache_root = _shared_repo_inspection_cache_root(create=create)
    if cache_root is None:
        return None
    target_root = cache_root / "symbol-marker-cache"
    try:
        target_root = _private_cache_dir(target_root) if create else target_root.expanduser().absolute()
    except OSError:
        return None
    return target_root / f"markers-{cache_key}.json"


def _load_symbol_marker_payload(path: Path):
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        with contextlib.suppress(OSError):
            path.unlink()
        return None
    if not isinstance(payload, dict) or payload.get("schema") != SYMBOL_MARKER_CACHE_SCHEMA:
        return None
    markers = payload.get("markers")
    if not isinstance(markers, list):
        return None
    normalized = []
    for item in markers:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        normalized.append((int(item[0]), str(item[1])))
    return normalized


def _symbol_marker_memory_cache_key(language: str, text_hash: str):
    return (str(language or ""), str(text_hash or ""))


def _load_symbol_markers_from_memory(language: str, text_hash: str):
    cache_key = _symbol_marker_memory_cache_key(language, text_hash)
    markers = _SYMBOL_MARKER_MEMORY_CACHE.get(cache_key)
    if markers is None:
        return None
    _SYMBOL_MARKER_MEMORY_CACHE.pop(cache_key, None)
    _SYMBOL_MARKER_MEMORY_CACHE[cache_key] = markers
    return [(int(line), str(symbol)) for line, symbol in markers]


def _cache_symbol_markers_in_memory(language: str, text_hash: str, markers):
    cache_key = _symbol_marker_memory_cache_key(language, text_hash)
    _SYMBOL_MARKER_MEMORY_CACHE.pop(cache_key, None)
    _SYMBOL_MARKER_MEMORY_CACHE[cache_key] = [(int(line), str(symbol)) for line, symbol in (markers or ())]
    while len(_SYMBOL_MARKER_MEMORY_CACHE) > _SYMBOL_MARKER_MEMORY_CACHE_LIMIT:
        _SYMBOL_MARKER_MEMORY_CACHE.pop(next(iter(_SYMBOL_MARKER_MEMORY_CACHE)))


def _load_symbol_markers(cache_dir: Path, language: str, text_hash: str):
    markers = _load_symbol_markers_from_memory(language, text_hash)
    if markers is not None:
        return markers
    cache_key = _symbol_marker_cache_key(language, text_hash)
    path = _symbol_marker_cache_path(cache_dir, cache_key)
    markers = _load_symbol_marker_payload(path)
    if markers is not None:
        _cache_symbol_markers_in_memory(language, text_hash, markers)
        return markers
    shared_path = _shared_symbol_marker_cache_path(cache_key, create=False)
    markers = _load_symbol_marker_payload(shared_path)
    if markers is not None:
        with contextlib.suppress(OSError):
            _atomic_private_json(path, {"schema": SYMBOL_MARKER_CACHE_SCHEMA, "markers": markers})
        _cache_symbol_markers_in_memory(language, text_hash, markers)
    return markers


def _write_symbol_markers(
    cache_dir: Path,
    language: str,
    text_hash: str,
    markers,
    *,
    publish_shared=True,
    persist_local=True,
):
    _cache_symbol_markers_in_memory(language, text_hash, markers)
    if not persist_local and not publish_shared:
        return
    cache_key = _symbol_marker_cache_key(language, text_hash)
    payload = {"schema": SYMBOL_MARKER_CACHE_SCHEMA, "markers": [(int(line), str(symbol)) for line, symbol in markers]}
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    path = _symbol_marker_cache_path(cache_dir, cache_key) if persist_local else None
    shared_path = _shared_symbol_marker_cache_path(cache_key, create=True) if publish_shared else None
    if shared_path is not None and shared_path != path:
        with contextlib.suppress(OSError):
            _atomic_private_bytes(shared_path, payload_bytes)
    if path is not None and (shared_path is None or shared_path == path):
        _atomic_private_bytes(path, payload_bytes)


def _shared_file_chunk_state_manifest_path(*, create=False):
    cache_root = _shared_repo_inspection_cache_root(create=create)
    if cache_root is None:
        return None
    return cache_root / "file-chunk-state-manifest.json"


def _shared_file_chunk_state_entry_dir(*, create=False):
    cache_root = _shared_repo_inspection_cache_root(create=create)
    if cache_root is None:
        return None
    return cache_root / "file-chunk-state-entries"


def _shared_file_chunk_state_key(config_key: str, state_signature: str):
    return sha256_text(f"{str(config_key or '')}\0{str(state_signature or '')}")


def _sharded_shared_state_entry_path(entry_dir: Path | None, state_key: str, *, create=False):
    if entry_dir is None:
        return None
    safe_key = str(state_key or "")
    shard = safe_key[:2] or "00"
    shard_dir = Path(entry_dir) / shard
    if create:
        shard_dir = _private_cache_dir(shard_dir)
    return shard_dir / f"{safe_key}.json"


def _legacy_shared_state_entry_path(entry_dir: Path | None, state_key: str, *, create=False):
    if entry_dir is None:
        return None
    if create:
        entry_dir = _private_cache_dir(entry_dir)
    return Path(entry_dir) / f"{str(state_key or '')}.json"


def _shared_file_chunk_state_entry_path(state_key: str, *, create=False):
    entry_dir = _shared_file_chunk_state_entry_dir(create=create)
    return _sharded_shared_state_entry_path(entry_dir, state_key, create=create)


def _legacy_shared_file_chunk_state_entry_path(state_key: str, *, create=False):
    entry_dir = _shared_file_chunk_state_entry_dir(create=create)
    return _legacy_shared_state_entry_path(entry_dir, state_key, create=create)


def _load_shared_file_chunk_state_entry(state_key: str):
    path = _shared_file_chunk_state_entry_path(state_key, create=False)
    legacy_path = _legacy_shared_file_chunk_state_entry_path(state_key, create=False)

    def load_path(candidate: Path | None):
        if candidate is None:
            return None
        cache_key = _snapshot_metadata_cache_key(candidate)
        if cache_key is not None:
            cached = _SHARED_FILE_CHUNK_STATE_ENTRY_MEMORY_CACHE.get(cache_key)
            if cached is not None:
                _SHARED_FILE_CHUNK_STATE_ENTRY_MEMORY_CACHE.pop(cache_key, None)
                _SHARED_FILE_CHUNK_STATE_ENTRY_MEMORY_CACHE[cache_key] = dict(cached)
                return dict(cached)
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            with contextlib.suppress(OSError):
                candidate.unlink()
            return None
        if not isinstance(payload, dict) or payload.get("schema") != SHARED_FILE_CHUNK_STATE_MANIFEST_SCHEMA:
            return None
        normalized = {
            "cache_key": str(payload.get("cache_key") or ""),
            "empty": bool(payload.get("empty")),
        }
        if cache_key is not None:
            _SHARED_FILE_CHUNK_STATE_ENTRY_MEMORY_CACHE.pop(cache_key, None)
            _SHARED_FILE_CHUNK_STATE_ENTRY_MEMORY_CACHE[cache_key] = dict(normalized)
            while len(_SHARED_FILE_CHUNK_STATE_ENTRY_MEMORY_CACHE) > _SHARED_FILE_CHUNK_STATE_ENTRY_MEMORY_CACHE_LIMIT:
                _SHARED_FILE_CHUNK_STATE_ENTRY_MEMORY_CACHE.pop(next(iter(_SHARED_FILE_CHUNK_STATE_ENTRY_MEMORY_CACHE)))
        return normalized

    loaded = load_path(path)
    if loaded is not None:
        return loaded
    loaded = load_path(legacy_path)
    return loaded


def _load_shared_file_chunk_state_manifest():
    path = _shared_file_chunk_state_manifest_path(create=False)
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    if not isinstance(payload, dict) or payload.get("schema") != SHARED_FILE_CHUNK_STATE_MANIFEST_SCHEMA:
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return {}
    normalized = {}
    for key, record in entries.items():
        if not isinstance(record, dict):
            continue
        normalized[str(key)] = {
            "cache_key": str(record.get("cache_key") or ""),
            "empty": bool(record.get("empty")),
        }
    return normalized


def _update_shared_file_chunk_state_manifest(config_key: str, state_signature: str, *, cache_key: str, empty: bool):
    state_key = _shared_file_chunk_state_key(config_key, state_signature)
    path = _shared_file_chunk_state_entry_path(state_key, create=True)
    if path is None:
        return
    _atomic_private_json(
        path,
        {
            "schema": SHARED_FILE_CHUNK_STATE_MANIFEST_SCHEMA,
            "cache_key": str(cache_key or ""),
            "empty": bool(empty),
        },
    )


def _shared_file_chunk_state_manifest_cache(entries=None, *, legacy_available=False):
    return {
        "entries": dict(entries or {}),
        "misses": set(),
        "dirty_entries": {},
        "dirty": False,
        "legacy_available": bool(legacy_available),
        "legacy_loaded": False,
    }


def _seed_shared_state_manifest_cache_from_working_manifest(manifest_cache, files):
    if not isinstance(manifest_cache, dict) or not isinstance(files, dict):
        return
    entries = manifest_cache.setdefault("entries", {})
    for _file_key, record in files.items():
        if not isinstance(record, dict):
            continue
        config_key = str(record.get("config_key") or "")
        state_signature = str(record.get("signature") or "")
        if not config_key or not state_signature:
            continue
        state_key = _shared_file_chunk_state_key(config_key, state_signature)
        normalized = {
            "cache_key": str(record.get("cache_key") or ""),
            "empty": bool(record.get("empty")),
        }
        if state_key not in entries:
            entries[state_key] = normalized


def _load_file_chunk_cache_by_state_signature(cache_dir: Path, config_key: str, state_signature: str, *, manifest_cache=None):
    state_key = _shared_file_chunk_state_key(config_key, state_signature)
    cached_bundle = _FILE_CHUNK_STATE_BUNDLE_MEMORY_CACHE.get(state_key)
    if isinstance(cached_bundle, dict):
        _FILE_CHUNK_STATE_BUNDLE_MEMORY_CACHE.pop(state_key, None)
        _FILE_CHUNK_STATE_BUNDLE_MEMORY_CACHE[state_key] = {
            "cache_key": str(cached_bundle.get("cache_key") or ""),
            "bundle": _file_chunk_cache_bundle(cached_bundle.get("bundle")),
            "empty": bool(cached_bundle.get("empty")),
        }
        if bool(cached_bundle.get("empty")):
            return "", []
        bundle = _file_chunk_cache_bundle(cached_bundle.get("bundle"))
        cache_key = str(cached_bundle.get("cache_key") or "")
        if cache_key and bundle is not None:
            return cache_key, bundle["chunks"]
    entries = manifest_cache.setdefault("entries", {}) if manifest_cache is not None else {}
    misses = manifest_cache.setdefault("misses", set()) if manifest_cache is not None else set()
    if state_key in misses:
        return None
    entry = entries.get(state_key)
    if entry is None:
        entry = _load_shared_file_chunk_state_entry(state_key)
        if entry is None and manifest_cache is not None:
            if bool(manifest_cache.get("legacy_available")) and not bool(manifest_cache.get("legacy_loaded")):
                legacy_entries = _load_shared_file_chunk_state_manifest()
                manifest_cache.setdefault("entries", {}).update(legacy_entries)
                manifest_cache["legacy_loaded"] = True
                entry = legacy_entries.get(state_key)
        elif entry is None:
            entry = (_load_shared_file_chunk_state_manifest()).get(state_key)
        if manifest_cache is not None and entry is not None:
            manifest_cache.setdefault("entries", {})[state_key] = dict(entry)
        elif manifest_cache is not None:
            misses.add(state_key)
    entry = entry or {}
    if bool(entry.get("empty")):
        return "", []
    cache_key = str(entry.get("cache_key") or "")
    if not cache_key:
        return None
    chunks = _load_file_chunk_cache(cache_dir, cache_key)
    if chunks is None:
        return None
    return cache_key, chunks


def _update_shared_file_chunk_state_manifest_cached(
    config_key: str,
    state_signature: str,
    *,
    cache_key: str,
    empty: bool,
    bundle=None,
    manifest_cache=None,
):
    key = _shared_file_chunk_state_key(config_key, state_signature)
    bundle_payload = _file_chunk_cache_bundle(bundle)
    if empty or bundle_payload is not None:
        _FILE_CHUNK_STATE_BUNDLE_MEMORY_CACHE.pop(key, None)
        _FILE_CHUNK_STATE_BUNDLE_MEMORY_CACHE[key] = {
            "cache_key": str(cache_key or ""),
            "bundle": bundle_payload,
            "empty": bool(empty),
        }
        while len(_FILE_CHUNK_STATE_BUNDLE_MEMORY_CACHE) > _FILE_CHUNK_STATE_BUNDLE_MEMORY_CACHE_LIMIT:
            _FILE_CHUNK_STATE_BUNDLE_MEMORY_CACHE.pop(next(iter(_FILE_CHUNK_STATE_BUNDLE_MEMORY_CACHE)))
    if manifest_cache is None:
        _update_shared_file_chunk_state_manifest(config_key, state_signature, cache_key=cache_key, empty=empty)
        return
    entries = manifest_cache.setdefault("entries", {})
    record = {
        "cache_key": str(cache_key or ""),
        "empty": bool(empty),
    }
    if entries.get(key) != record:
        entries[key] = record
        manifest_cache.setdefault("dirty_entries", {})[key] = dict(record)
        manifest_cache["dirty"] = True


def _flush_shared_file_chunk_state_manifest_cache(manifest_cache):
    if not isinstance(manifest_cache, dict) or not bool(manifest_cache.get("dirty")):
        return
    dirty_entries = dict(manifest_cache.get("dirty_entries") or {})
    if not dirty_entries:
        return
    for state_key, record in dirty_entries.items():
        path = _shared_file_chunk_state_entry_path(state_key, create=True)
        if path is None:
            continue
        _atomic_private_json(
            path,
            {
                "schema": SHARED_FILE_CHUNK_STATE_MANIFEST_SCHEMA,
                "cache_key": str((record or {}).get("cache_key") or ""),
                "empty": bool((record or {}).get("empty")),
            },
        )
    manifest_cache["dirty_entries"] = {}
    manifest_cache["dirty"] = False
    manifest_cache["dirty"] = False


def _file_chunk_manifest_path(cache_dir: Path):
    return _private_cache_dir(Path(cache_dir)) / "file-chunk-working-manifest.json"


def _shared_file_chunk_working_manifest_path(repository_state_fingerprint: str, build_config_digest: str, *, create=False):
    cache_root = _shared_repo_inspection_cache_root(create=create)
    if cache_root is None:
        return None
    cache_root = cache_root / "file-chunk-manifests"
    try:
        cache_root = _private_cache_dir(cache_root) if create else cache_root.expanduser().absolute()
    except OSError:
        return None
    safe_repository = str(repository_state_fingerprint or "").replace(":", "_")
    safe_config = str(build_config_digest or "").replace(":", "_")
    return cache_root / f"manifest-{safe_repository}-{safe_config}.json"


def _shared_file_chunk_latest_manifest_path(build_config_digest: str, *, create=False):
    cache_root = _shared_repo_inspection_cache_root(create=create)
    if cache_root is None:
        return None
    cache_root = cache_root / "file-chunk-manifests"
    try:
        cache_root = _private_cache_dir(cache_root) if create else cache_root.expanduser().absolute()
    except OSError:
        return None
    safe_config = str(build_config_digest or "").replace(":", "_")
    return cache_root / f"latest-{safe_config}.json"


def _file_chunk_snapshot_path(cache_dir: Path):
    return _private_cache_dir(Path(cache_dir)) / "file-chunk-working-snapshot.json"


def _file_chunk_snapshot_metadata_path(cache_dir: Path):
    return _private_cache_dir(Path(cache_dir)) / "file-chunk-working-snapshot-metadata.json"


def _file_chunk_manifest_entry_key(*, source_namespace: str, rel_path: str):
    return f"{source_namespace}\0{rel_path}"


def _manifest_records_for_file_chunks(file_chunks):
    if not file_chunks:
        return None, None
    first = file_chunks[0]
    file_key = _lexical_chunk_file_key(first)
    lexical_record = {
        "file_key": file_key,
        "path": first["path"],
        "language": first.get("language", ""),
        "repository_path": first.get("repository_path", first["path"]),
        "source_namespace": first.get("source_namespace", ""),
        "chunks": [],
    }
    index_signature_payload = {
        "path": first["path"],
        "repository_path": first.get("repository_path", first["path"]),
        "source_namespace": first.get("source_namespace", ""),
        "chunks": [],
    }
    for chunk in sorted(file_chunks, key=lambda item: (item["line_start"], item["chunk_id"])):
        lexical_record["chunks"].append(
            {
                "chunk_id": chunk["chunk_id"],
                "symbol": chunk.get("symbol", ""),
                "line_start": int(chunk["line_start"]),
                "line_end": int(chunk["line_end"]),
                "content_hash": chunk["content_hash"],
                "token_estimate": int(chunk["token_estimate"]),
            }
        )
        index_signature_payload["chunks"].append(
            {
                "chunk_id": chunk["chunk_id"],
                "path": chunk["path"],
                "language": chunk.get("language", ""),
                "symbol": chunk.get("symbol", ""),
                "line_start": int(chunk["line_start"]),
                "line_end": int(chunk["line_end"]),
                "content_hash": chunk["content_hash"],
                "input_id": chunk.get("input_id", ""),
                "source_namespace": chunk.get("source_namespace", ""),
                "classification": chunk.get("classification", "unknown"),
            }
        )
    lexical_signature_payload = {
        "repository_path": lexical_record["repository_path"],
        "source_namespace": lexical_record["source_namespace"],
        "chunks": lexical_record["chunks"],
    }
    lexical_record["signature"] = f"sha256:{sha256_text(json.dumps(lexical_signature_payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True))}"
    index_signature = f"sha256:{sha256_text(json.dumps(index_signature_payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True))}"
    return lexical_record, index_signature


def _load_file_chunk_working_manifest(path: Path):
    cache_key = _snapshot_metadata_cache_key(path)
    if cache_key is not None:
        cached = _FILE_CHUNK_WORKING_MANIFEST_CACHE.get(cache_key)
        if cached is not None:
            _FILE_CHUNK_WORKING_MANIFEST_CACHE.pop(cache_key, None)
            _FILE_CHUNK_WORKING_MANIFEST_CACHE[cache_key] = cached
            return {
                "repository_state_fingerprint": str(cached.get("repository_state_fingerprint") or ""),
                "build_config_digest": str(cached.get("build_config_digest") or ""),
                "files": {
                    str(file_key): {
                        "signature": str(record.get("signature") or ""),
                        "cache_key": str(record.get("cache_key") or ""),
                        "config_key": str(record.get("config_key") or ""),
                        "empty": bool(record.get("empty")),
                    }
                    for file_key, record in dict(cached.get("files") or {}).items()
                },
                "_serialized_files_json": str(cached.get("_serialized_files_json") or ""),
            }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        path.unlink(missing_ok=True)
        return None
    if not isinstance(payload, dict):
        return None
    schema = str(payload.get("schema") or "")
    if schema not in {"repo-inspection-file-chunk-manifest-v1", FILE_CHUNK_WORKING_MANIFEST_SCHEMA}:
        return None
    files = payload.get("files")
    if not isinstance(files, dict):
        return None
    normalized = {}
    for file_key, record in files.items():
        if not isinstance(record, dict):
            return None
        normalized[str(file_key)] = {
            "signature": str(record.get("signature") or ""),
            "cache_key": str(record.get("cache_key") or ""),
            "config_key": str(record.get("config_key") or ""),
            "empty": bool(record.get("empty")),
        }
    normalized_payload = {
        "repository_state_fingerprint": str(payload.get("repository_state_fingerprint") or ""),
        "build_config_digest": str(payload.get("build_config_digest") or ""),
        "files": normalized,
        "_serialized_files_json": _serialize_file_chunk_manifest_files(normalized),
    }
    if cache_key is not None:
        _FILE_CHUNK_WORKING_MANIFEST_CACHE.pop(cache_key, None)
        _FILE_CHUNK_WORKING_MANIFEST_CACHE[cache_key] = {
            "repository_state_fingerprint": str(normalized_payload.get("repository_state_fingerprint") or ""),
            "build_config_digest": str(normalized_payload.get("build_config_digest") or ""),
            "files": {
                str(file_key): {
                    "signature": str(record.get("signature") or ""),
                    "cache_key": str(record.get("cache_key") or ""),
                    "config_key": str(record.get("config_key") or ""),
                    "empty": bool(record.get("empty")),
                }
                for file_key, record in normalized.items()
            },
            "_serialized_files_json": str(normalized_payload.get("_serialized_files_json") or ""),
        }
        while len(_FILE_CHUNK_WORKING_MANIFEST_CACHE) > _FILE_CHUNK_WORKING_MANIFEST_CACHE_LIMIT:
            _FILE_CHUNK_WORKING_MANIFEST_CACHE.pop(next(iter(_FILE_CHUNK_WORKING_MANIFEST_CACHE)))
    return normalized_payload


def _write_file_chunk_working_manifest(path: Path, files, *, repository_state_fingerprint="", build_config_digest=""):
    _atomic_private_json(
        path,
        {
            "schema": FILE_CHUNK_WORKING_MANIFEST_SCHEMA,
            "repository_state_fingerprint": str(repository_state_fingerprint),
            "build_config_digest": str(build_config_digest),
            "files": files,
        },
    )


def _serialize_file_chunk_manifest_files(files) -> str:
    return json.dumps(files, separators=(",", ":"))


def _file_chunk_working_manifest_payload_bytes(
    files,
    *,
    repository_state_fingerprint="",
    build_config_digest="",
    serialized_files_json=None,
) -> bytes:
    files_json = (
        str(serialized_files_json)
        if isinstance(serialized_files_json, str) and serialized_files_json
        else _serialize_file_chunk_manifest_files(files)
    )
    return (
        "{"
        + f"\"schema\":{json.dumps(FILE_CHUNK_WORKING_MANIFEST_SCHEMA)},"
        + f"\"repository_state_fingerprint\":{json.dumps(str(repository_state_fingerprint))},"
        + f"\"build_config_digest\":{json.dumps(str(build_config_digest))},"
        + f"\"files\":{files_json}"
        + "}"
    ).encode("utf-8")


def _cache_file_chunk_working_manifest_payload(
    path: Path | None,
    *,
    repository_state_fingerprint="",
    build_config_digest="",
    files=None,
    serialized_files_json=None,
):
    if path is None or files is None:
        return
    cache_key = _snapshot_metadata_cache_key(Path(path))
    if cache_key is None:
        return
    _FILE_CHUNK_WORKING_MANIFEST_CACHE.pop(cache_key, None)
    _FILE_CHUNK_WORKING_MANIFEST_CACHE[cache_key] = {
        "repository_state_fingerprint": str(repository_state_fingerprint or ""),
        "build_config_digest": str(build_config_digest or ""),
        "files": {
            str(file_key): {
                "signature": str(record.get("signature") or ""),
                "cache_key": str(record.get("cache_key") or ""),
                "config_key": str(record.get("config_key") or ""),
                "empty": bool(record.get("empty")),
            }
            for file_key, record in dict(files or {}).items()
        },
        "_serialized_files_json": (
            str(serialized_files_json)
            if isinstance(serialized_files_json, str) and serialized_files_json
            else _serialize_file_chunk_manifest_files(files)
        ),
    }
    while len(_FILE_CHUNK_WORKING_MANIFEST_CACHE) > _FILE_CHUNK_WORKING_MANIFEST_CACHE_LIMIT:
        _FILE_CHUNK_WORKING_MANIFEST_CACHE.pop(next(iter(_FILE_CHUNK_WORKING_MANIFEST_CACHE)))


def _write_preferred_file_chunk_working_manifest(
    local_path,
    shared_path,
    shared_latest_path,
    files,
    *,
    repository_state_fingerprint="",
    build_config_digest="",
    publish_shared=True,
    serialized_files_json=None,
    diagnostics=None,
):
    payload_bytes = _file_chunk_working_manifest_payload_bytes(
        files,
        repository_state_fingerprint=repository_state_fingerprint,
        build_config_digest=build_config_digest,
        serialized_files_json=serialized_files_json,
    )
    if local_path is not None:
        started = time.perf_counter()
        if not _path_bytes_equal(local_path, payload_bytes):
            _atomic_private_bytes(local_path, payload_bytes)
        if isinstance(diagnostics, dict):
            diagnostics["working_manifest_local_write_ms"] = diagnostics.get("working_manifest_local_write_ms", 0.0) + round(
                (time.perf_counter() - started) * 1000.0, 3
            )
        _cache_file_chunk_working_manifest_payload(
            local_path,
            repository_state_fingerprint=repository_state_fingerprint,
            build_config_digest=build_config_digest,
            files=files,
            serialized_files_json=serialized_files_json,
        )
    if publish_shared and shared_path is not None and shared_path != local_path:
        with contextlib.suppress(OSError):
            started = time.perf_counter()
            if local_path is not None and local_path.exists():
                _clone_private_cache_file(local_path, shared_path)
            elif not _path_bytes_equal(shared_path, payload_bytes):
                _atomic_private_bytes(shared_path, payload_bytes)
            if isinstance(diagnostics, dict):
                diagnostics["working_manifest_shared_publish_ms"] = diagnostics.get(
                    "working_manifest_shared_publish_ms", 0.0
                ) + round((time.perf_counter() - started) * 1000.0, 3)
            _cache_file_chunk_working_manifest_payload(
                shared_path,
                repository_state_fingerprint=repository_state_fingerprint,
                build_config_digest=build_config_digest,
                files=files,
                serialized_files_json=serialized_files_json,
            )
    if (
        publish_shared
        and shared_latest_path is not None
        and shared_latest_path != local_path
        and shared_latest_path != shared_path
    ):
        with contextlib.suppress(OSError):
            started = time.perf_counter()
            if local_path is not None and local_path.exists():
                _clone_private_cache_file(local_path, shared_latest_path)
            elif not _path_bytes_equal(shared_latest_path, payload_bytes):
                _atomic_private_bytes(shared_latest_path, payload_bytes)
            if isinstance(diagnostics, dict):
                diagnostics["working_manifest_shared_latest_publish_ms"] = diagnostics.get(
                    "working_manifest_shared_latest_publish_ms", 0.0
                ) + round((time.perf_counter() - started) * 1000.0, 3)
            _cache_file_chunk_working_manifest_payload(
                shared_latest_path,
                repository_state_fingerprint=repository_state_fingerprint,
                build_config_digest=build_config_digest,
                files=files,
                serialized_files_json=serialized_files_json,
            )


def _load_file_chunk_cache_payload(path: Path):
    if path is None:
        return None
    cache_key = _snapshot_metadata_cache_key(path)
    if cache_key is not None:
        cached = _FILE_CHUNK_CACHE_PAYLOAD_MEMORY_CACHE.get(cache_key)
        if cached is not None:
            _FILE_CHUNK_CACHE_PAYLOAD_MEMORY_CACHE.pop(cache_key, None)
            _FILE_CHUNK_CACHE_PAYLOAD_MEMORY_CACHE[cache_key] = cached
            return {
                "chunks": [dict(chunk) for chunk in cached["chunks"]],
                "lexical_record": dict(cached["lexical_record"]) if isinstance(cached.get("lexical_record"), dict) else None,
                "index_signature": str(cached.get("index_signature") or ""),
                "semantic_document_signatures": (
                    {str(k): str(v) for k, v in dict(cached.get("semantic_document_signatures") or {}).items()}
                    if isinstance(cached.get("semantic_document_signatures"), dict)
                    else None
                ),
            }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        with contextlib.suppress(OSError):
            path.unlink()
        return None
    if not isinstance(payload, dict):
        return None
    schema = str(payload.get("schema") or "")
    if schema not in {"repo-inspection-file-chunks-v1", FILE_CHUNK_CACHE_SCHEMA}:
        return None
    chunks = payload.get("chunks")
    if not isinstance(chunks, list):
        return None
    normalized = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            return None
        if not chunk.get("chunk_id") or not chunk.get("path") or not isinstance(chunk.get("content"), str):
            return None
        normalized.append(dict(chunk))
    lexical_record = payload.get("lexical_record")
    if lexical_record is not None and not isinstance(lexical_record, dict):
        return None
    index_signature = payload.get("index_signature")
    if index_signature is not None and not isinstance(index_signature, str):
        return None
    semantic_document_signatures = payload.get("semantic_document_signatures")
    if semantic_document_signatures is not None and not isinstance(semantic_document_signatures, dict):
        return None
    normalized_payload = {
        "chunks": normalized,
        "lexical_record": dict(lexical_record) if isinstance(lexical_record, dict) else None,
        "index_signature": str(index_signature or "") if index_signature is not None else "",
        "semantic_document_signatures": (
            {str(k): str(v) for k, v in semantic_document_signatures.items()}
            if isinstance(semantic_document_signatures, dict)
            else None
        ),
    }
    if cache_key is not None:
        _FILE_CHUNK_CACHE_PAYLOAD_MEMORY_CACHE.pop(cache_key, None)
        _FILE_CHUNK_CACHE_PAYLOAD_MEMORY_CACHE[cache_key] = {
            "chunks": [dict(chunk) for chunk in normalized_payload["chunks"]],
            "lexical_record": (
                dict(normalized_payload["lexical_record"])
                if isinstance(normalized_payload.get("lexical_record"), dict)
                else None
            ),
            "index_signature": str(normalized_payload.get("index_signature") or ""),
            "semantic_document_signatures": (
                {str(k): str(v) for k, v in dict(normalized_payload.get("semantic_document_signatures") or {}).items()}
                if isinstance(normalized_payload.get("semantic_document_signatures"), dict)
                else None
            ),
        }
        while len(_FILE_CHUNK_CACHE_PAYLOAD_MEMORY_CACHE) > _FILE_CHUNK_CACHE_PAYLOAD_MEMORY_CACHE_LIMIT:
            _FILE_CHUNK_CACHE_PAYLOAD_MEMORY_CACHE.pop(next(iter(_FILE_CHUNK_CACHE_PAYLOAD_MEMORY_CACHE)))
    return normalized_payload


def _cache_file_chunk_cache_payload_in_memory(path: Path, payload):
    if not isinstance(payload, dict):
        return
    cache_key = _snapshot_metadata_cache_key(path)
    if cache_key is None:
        return
    normalized = []
    chunks = payload.get("chunks")
    if not isinstance(chunks, list):
        return
    for chunk in chunks:
        if not isinstance(chunk, dict):
            return
        normalized.append(dict(chunk))
    lexical_record = payload.get("lexical_record")
    semantic_document_signatures = payload.get("semantic_document_signatures")
    _FILE_CHUNK_CACHE_PAYLOAD_MEMORY_CACHE.pop(cache_key, None)
    _FILE_CHUNK_CACHE_PAYLOAD_MEMORY_CACHE[cache_key] = {
        "chunks": normalized,
        "lexical_record": dict(lexical_record) if isinstance(lexical_record, dict) else None,
        "index_signature": str(payload.get("index_signature") or ""),
        "semantic_document_signatures": (
            {str(k): str(v) for k, v in dict(semantic_document_signatures or {}).items()}
            if isinstance(semantic_document_signatures, dict)
            else None
        ),
    }
    while len(_FILE_CHUNK_CACHE_PAYLOAD_MEMORY_CACHE) > _FILE_CHUNK_CACHE_PAYLOAD_MEMORY_CACHE_LIMIT:
        _FILE_CHUNK_CACHE_PAYLOAD_MEMORY_CACHE.pop(next(iter(_FILE_CHUNK_CACHE_PAYLOAD_MEMORY_CACHE)))


def _file_chunk_cache_bundle(cache_payload):
    if not isinstance(cache_payload, dict):
        return None
    chunks = cache_payload.get("chunks")
    if not isinstance(chunks, list):
        return None
    lexical_record = cache_payload.get("lexical_record")
    index_signature = str(cache_payload.get("index_signature") or "")
    semantic_document_signatures = cache_payload.get("semantic_document_signatures")
    if isinstance(lexical_record, dict) and index_signature and isinstance(semantic_document_signatures, dict):
        return {
            "chunks": chunks,
            "lexical_record": dict(lexical_record),
            "index_signature": index_signature,
            "semantic_document_signatures": {str(k): str(v) for k, v in semantic_document_signatures.items()},
        }
    derived_lexical_record, derived_index_signature = _manifest_records_for_file_chunks(chunks)
    return {
        "chunks": chunks,
        "lexical_record": derived_lexical_record,
        "index_signature": derived_index_signature,
        "semantic_document_signatures": {
            str(chunk.get("chunk_id") or ""): semantic_chunk_signature(chunk) for chunk in chunks
        },
    }


def _load_file_chunk_cache_bundle(cache_dir: Path, cache_key: str):
    path = _file_chunk_cache_path(cache_dir, cache_key)
    payload = _load_file_chunk_cache_payload(path)
    bundle = _file_chunk_cache_bundle(payload)
    if bundle is not None:
        return bundle
    shared_path = _shared_file_chunk_cache_path(cache_key, create=False)
    payload = _load_file_chunk_cache_payload(shared_path)
    return _file_chunk_cache_bundle(payload)


def _load_file_chunk_cache(cache_dir: Path, cache_key: str):
    bundle = _load_file_chunk_cache_bundle(cache_dir, cache_key)
    return None if bundle is None else bundle["chunks"]


def _write_file_chunk_cache(
    cache_dir: Path,
    cache_key: str,
    chunks,
    *,
    publish_shared=True,
    lexical_record=None,
    index_signature="",
    semantic_document_signatures=None,
):
    path = _file_chunk_cache_path(cache_dir, cache_key)
    if not isinstance(lexical_record, dict) or not str(index_signature or ""):
        lexical_record, index_signature = _manifest_records_for_file_chunks(chunks)
    normalized_semantic_signatures = (
        {str(k): str(v) for k, v in dict(semantic_document_signatures or {}).items()}
        if isinstance(semantic_document_signatures, dict)
        else None
    )
    if normalized_semantic_signatures is None:
        normalized_semantic_signatures = {
            str(chunk.get("chunk_id") or ""): semantic_chunk_signature(chunk) for chunk in chunks
        }
    payload = {
        "schema": FILE_CHUNK_CACHE_SCHEMA,
        "chunks": list(chunks),
        "lexical_record": lexical_record,
        "index_signature": index_signature,
        "semantic_document_signatures": normalized_semantic_signatures,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    shared_path = _shared_file_chunk_cache_path(cache_key, create=True) if publish_shared else None
    if shared_path is not None and shared_path != path:
        with contextlib.suppress(OSError):
            _atomic_private_bytes(shared_path, payload_bytes)
            _cache_file_chunk_cache_payload_in_memory(shared_path, payload)
    else:
        _atomic_private_bytes(path, payload_bytes)
        _cache_file_chunk_cache_payload_in_memory(path, payload)


def _normalize_snapshot_payload(payload):
    if not isinstance(payload, dict) or payload.get("schema") != FILE_CHUNK_SNAPSHOT_SCHEMA:
        return None
    chunks = payload.get("chunks")
    if not isinstance(chunks, list):
        return None
    normalized_chunks = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            return None
        if not chunk.get("chunk_id") or not chunk.get("path") or not isinstance(chunk.get("content"), str):
            return None
        normalized_chunks.append(dict(chunk))
    lexical_manifest = payload.get("lexical_manifest")
    index_manifest = payload.get("index_manifest")
    chunks_by_file = payload.get("chunks_by_file")
    semantic_document_signatures = payload.get("semantic_document_signatures")
    chunk_ids = payload.get("chunk_ids")
    file_key_by_chunk_id = payload.get("file_key_by_chunk_id")
    if not isinstance(lexical_manifest, dict) or not isinstance(index_manifest, dict):
        return None
    if not isinstance(semantic_document_signatures, dict):
        return None
    if chunk_ids is not None and not isinstance(chunk_ids, list):
        return None
    if file_key_by_chunk_id is not None and not isinstance(file_key_by_chunk_id, dict):
        return None
    normalized_file_key_by_chunk_id = {
        str(k): str(v) for k, v in dict(file_key_by_chunk_id or {}).items()
    }
    normalized_chunks_by_file = None
    if isinstance(chunks_by_file, dict):
        normalized_chunks_by_file = {
            str(file_key): [dict(chunk) for chunk in file_chunks]
            for file_key, file_chunks in chunks_by_file.items()
            if isinstance(file_chunks, list)
        }
    if normalized_chunks_by_file is None:
        normalized_chunks_by_file = defaultdict(list)
        for chunk in normalized_chunks:
            chunk_id = str(chunk.get("chunk_id") or "")
            file_key = normalized_file_key_by_chunk_id.get(chunk_id) or _lexical_chunk_file_key(chunk)
            normalized_chunks_by_file[str(file_key)].append(chunk)
        normalized_chunks_by_file = dict(normalized_chunks_by_file)
    return {
        "repository_state_fingerprint": str(payload.get("repository_state_fingerprint") or ""),
        "build_config_digest": str(payload.get("build_config_digest") or ""),
        "chunks": normalized_chunks,
        "lexical_manifest": lexical_manifest,
        "index_manifest": index_manifest,
        "chunks_by_file": normalized_chunks_by_file,
        "semantic_document_signatures": {str(k): str(v) for k, v in semantic_document_signatures.items()},
        "chunk_ids": [str(value) for value in (chunk_ids or [])],
        "file_key_by_chunk_id": normalized_file_key_by_chunk_id,
    }


def _clone_snapshot_payload(snapshot):
    return {
        "repository_state_fingerprint": str(snapshot.get("repository_state_fingerprint") or ""),
        "build_config_digest": str(snapshot.get("build_config_digest") or ""),
        "chunks": [dict(chunk) for chunk in (snapshot.get("chunks") or ()) if isinstance(chunk, dict)],
        "lexical_manifest": dict(snapshot.get("lexical_manifest") or {}),
        "index_manifest": dict(snapshot.get("index_manifest") or {}),
        "chunks_by_file": {
            str(file_key): [dict(chunk) for chunk in file_chunks]
            for file_key, file_chunks in dict(snapshot.get("chunks_by_file") or {}).items()
            if isinstance(file_chunks, list)
        },
        "semantic_document_signatures": {
            str(k): str(v) for k, v in dict(snapshot.get("semantic_document_signatures") or {}).items()
        },
        "chunk_ids": [str(value) for value in (snapshot.get("chunk_ids") or ())],
        "file_key_by_chunk_id": {
            str(k): str(v) for k, v in dict(snapshot.get("file_key_by_chunk_id") or {}).items()
        },
    }


def _cache_file_chunk_snapshot_payload(path: Path, snapshot):
    cache_key = _snapshot_metadata_cache_key(path)
    if cache_key is None or not isinstance(snapshot, dict):
        return
    _FILE_CHUNK_SNAPSHOT_MEMORY_CACHE.pop(cache_key, None)
    _FILE_CHUNK_SNAPSHOT_MEMORY_CACHE[cache_key] = _clone_snapshot_payload(snapshot)
    while len(_FILE_CHUNK_SNAPSHOT_MEMORY_CACHE) > _FILE_CHUNK_SNAPSHOT_MEMORY_CACHE_LIMIT:
        _FILE_CHUNK_SNAPSHOT_MEMORY_CACHE.pop(next(iter(_FILE_CHUNK_SNAPSHOT_MEMORY_CACHE)))


def _drop_file_chunk_snapshot_payload(path: Path):
    path = Path(path)
    for cache_key in [key for key in _FILE_CHUNK_SNAPSHOT_MEMORY_CACHE if isinstance(key, tuple) and key and key[0] == str(path)]:
        _FILE_CHUNK_SNAPSHOT_MEMORY_CACHE.pop(cache_key, None)
    path.unlink(missing_ok=True)


def _load_file_chunk_snapshot(cache_dir: Path):
    path = _file_chunk_snapshot_path(cache_dir)
    if not path.exists():
        return None
    cache_key = _snapshot_metadata_cache_key(path)
    if cache_key is not None:
        cached = _FILE_CHUNK_SNAPSHOT_MEMORY_CACHE.get(cache_key)
        if cached is not None:
            _FILE_CHUNK_SNAPSHOT_MEMORY_CACHE.pop(cache_key, None)
            _FILE_CHUNK_SNAPSHOT_MEMORY_CACHE[cache_key] = cached
            return _clone_snapshot_payload(cached)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        path.unlink(missing_ok=True)
        return None
    normalized = _normalize_snapshot_payload(payload)
    if normalized is None:
        path.unlink(missing_ok=True)
        return None
    _cache_file_chunk_snapshot_payload(path, normalized)
    return normalized


def _snapshot_metadata_cache_key(path: Path):
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path), int(stat.st_mtime_ns), int(stat.st_size))


def _clone_snapshot_metadata(metadata):
    return {
        "repository_state_fingerprint": str(metadata.get("repository_state_fingerprint") or ""),
        "build_config_digest": str(metadata.get("build_config_digest") or ""),
        "index_manifest": dict(metadata.get("index_manifest") or {}),
        "semantic_document_signatures": dict(metadata.get("semantic_document_signatures") or {}),
        "chunk_ids": [str(value) for value in (metadata.get("chunk_ids") or ())],
        "chunk_count": int(metadata.get("chunk_count") or 0),
        "total_files": int(metadata.get("total_files") or 0),
        "_serialized_index_manifest_json": str(metadata.get("_serialized_index_manifest_json") or ""),
        "_serialized_semantic_document_signatures_json": str(
            metadata.get("_serialized_semantic_document_signatures_json") or ""
        ),
        "_serialized_chunk_ids_json": str(metadata.get("_serialized_chunk_ids_json") or ""),
    }


def _snapshot_matches_metadata(snapshot, metadata):
    if not isinstance(snapshot, dict) or not isinstance(metadata, dict):
        return False
    snapshot_chunk_ids = [str(value) for value in (snapshot.get("chunk_ids") or ())]
    metadata_chunk_ids = [str(value) for value in (metadata.get("chunk_ids") or ())]
    if snapshot_chunk_ids != metadata_chunk_ids:
        return False
    if dict(snapshot.get("index_manifest") or {}) != dict(metadata.get("index_manifest") or {}):
        return False
    if dict(snapshot.get("semantic_document_signatures") or {}) != dict(metadata.get("semantic_document_signatures") or {}):
        return False
    if int(metadata.get("chunk_count") or 0) != len(snapshot.get("chunks") or ()):
        return False
    total_files = int(metadata.get("total_files") or 0)
    if total_files and total_files != len(dict(snapshot.get("chunks_by_file") or {})):
        return False
    return True


def _cache_snapshot_metadata(path: Path, metadata):
    cache_key = _snapshot_metadata_cache_key(path)
    if cache_key is None:
        return
    _SNAPSHOT_METADATA_CACHE.pop(cache_key, None)
    _SNAPSHOT_METADATA_CACHE[cache_key] = _clone_snapshot_metadata(metadata)
    while len(_SNAPSHOT_METADATA_CACHE) > _SNAPSHOT_METADATA_CACHE_LIMIT:
        _SNAPSHOT_METADATA_CACHE.pop(next(iter(_SNAPSHOT_METADATA_CACHE)))


def _load_file_chunk_snapshot_metadata(cache_dir: Path):
    path = _file_chunk_snapshot_metadata_path(cache_dir)
    if not path.exists():
        return None
    cache_key = _snapshot_metadata_cache_key(path)
    if cache_key is not None:
        cached = _SNAPSHOT_METADATA_CACHE.get(cache_key)
        if cached is not None:
            _SNAPSHOT_METADATA_CACHE.pop(cache_key, None)
            _SNAPSHOT_METADATA_CACHE[cache_key] = cached
            return _clone_snapshot_metadata(cached)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        path.unlink(missing_ok=True)
        return None
    if not isinstance(payload, dict) or payload.get("schema") != FILE_CHUNK_SNAPSHOT_METADATA_SCHEMA:
        path.unlink(missing_ok=True)
        return None
    if not isinstance(payload.get("index_manifest"), dict):
        path.unlink(missing_ok=True)
        return None
    if not isinstance(payload.get("semantic_document_signatures"), dict):
        path.unlink(missing_ok=True)
        return None
    if not isinstance(payload.get("chunk_ids"), list):
        path.unlink(missing_ok=True)
        return None
    metadata = {
        "repository_state_fingerprint": str(payload.get("repository_state_fingerprint") or ""),
        "build_config_digest": str(payload.get("build_config_digest") or ""),
        "index_manifest": {str(k): str(v) for k, v in payload["index_manifest"].items()},
        "semantic_document_signatures": {str(k): str(v) for k, v in payload["semantic_document_signatures"].items()},
        "chunk_ids": [str(value) for value in payload.get("chunk_ids")],
        "chunk_count": int(payload.get("chunk_count") or 0),
        "total_files": int(payload.get("total_files") or 0),
        "_serialized_index_manifest_json": json.dumps(payload["index_manifest"], separators=(",", ":")),
        "_serialized_semantic_document_signatures_json": json.dumps(
            payload["semantic_document_signatures"], separators=(",", ":")
        ),
        "_serialized_chunk_ids_json": json.dumps(payload.get("chunk_ids"), separators=(",", ":")),
    }
    _cache_snapshot_metadata(path, metadata)
    return _clone_snapshot_metadata(metadata)


def _normalize_file_chunk_snapshot_metadata_payload(payload):
    if not isinstance(payload, dict) or payload.get("schema") != FILE_CHUNK_SNAPSHOT_METADATA_SCHEMA:
        return None
    if not isinstance(payload.get("index_manifest"), dict):
        return None
    if not isinstance(payload.get("semantic_document_signatures"), dict):
        return None
    if not isinstance(payload.get("chunk_ids"), list):
        return None
    return {
        "repository_state_fingerprint": str(payload.get("repository_state_fingerprint") or ""),
        "build_config_digest": str(payload.get("build_config_digest") or ""),
        "index_manifest": {str(k): str(v) for k, v in payload["index_manifest"].items()},
        "semantic_document_signatures": {str(k): str(v) for k, v in payload["semantic_document_signatures"].items()},
        "chunk_ids": [str(value) for value in payload.get("chunk_ids")],
        "chunk_count": int(payload.get("chunk_count") or 0),
        "total_files": int(payload.get("total_files") or 0),
        "_serialized_index_manifest_json": json.dumps(payload["index_manifest"], separators=(",", ":")),
        "_serialized_semantic_document_signatures_json": json.dumps(
            payload["semantic_document_signatures"], separators=(",", ":")
        ),
        "_serialized_chunk_ids_json": json.dumps(payload.get("chunk_ids"), separators=(",", ":")),
    }


def _write_file_chunk_snapshot_metadata(
    path: Path,
    *,
    repository_state_fingerprint,
    build_config_digest,
    chunks,
    serialized_index_manifest_json=None,
    serialized_semantic_document_signatures_json=None,
    serialized_chunk_ids_json=None,
    diagnostics=None,
):
    index_manifest = getattr(chunks, "_index_manifest", {})
    semantic_document_signatures = getattr(chunks, "_semantic_document_signatures", {})
    chunk_ids = list(getattr(chunks, "_chunk_ids", ()))
    metadata_payload_bytes = (
        "{"
        + f"\"schema\":{json.dumps(FILE_CHUNK_SNAPSHOT_METADATA_SCHEMA)},"
        + f"\"repository_state_fingerprint\":{json.dumps(str(repository_state_fingerprint or ''))},"
        + f"\"build_config_digest\":{json.dumps(str(build_config_digest or ''))},"
        + "\"index_manifest\":"
        + (
            str(serialized_index_manifest_json)
            if isinstance(serialized_index_manifest_json, str) and serialized_index_manifest_json
            else json.dumps(index_manifest, separators=(",", ":"))
        )
        + ",\"semantic_document_signatures\":"
        + (
            str(serialized_semantic_document_signatures_json)
            if isinstance(serialized_semantic_document_signatures_json, str)
            and serialized_semantic_document_signatures_json
            else json.dumps(semantic_document_signatures, separators=(",", ":"))
        )
        + ",\"chunk_ids\":"
        + (
            str(serialized_chunk_ids_json)
            if isinstance(serialized_chunk_ids_json, str) and serialized_chunk_ids_json
            else json.dumps(chunk_ids, separators=(",", ":"))
        )
        + f",\"chunk_count\":{int(len(chunks))}"
        + f",\"total_files\":{int(len(getattr(chunks, '_chunks_by_file', {}) or {}))}"
        + "}"
    ).encode("utf-8")
    started = time.perf_counter()
    if not _path_bytes_equal(path, metadata_payload_bytes):
        _atomic_private_bytes(path, metadata_payload_bytes)
    if isinstance(diagnostics, dict):
        diagnostics["snapshot_metadata_local_write_ms"] = diagnostics.get("snapshot_metadata_local_write_ms", 0.0) + round(
            (time.perf_counter() - started) * 1000.0, 3
        )
    metadata = {
        "repository_state_fingerprint": str(repository_state_fingerprint or ""),
        "build_config_digest": str(build_config_digest or ""),
        "index_manifest": dict(index_manifest),
        "semantic_document_signatures": dict(semantic_document_signatures),
        "chunk_ids": list(chunk_ids),
        "chunk_count": len(chunks),
        "total_files": len(getattr(chunks, "_chunks_by_file", {}) or {}),
        "_serialized_index_manifest_json": (
            str(serialized_index_manifest_json)
            if isinstance(serialized_index_manifest_json, str) and serialized_index_manifest_json
            else json.dumps(index_manifest, separators=(",", ":"))
        ),
        "_serialized_semantic_document_signatures_json": (
            str(serialized_semantic_document_signatures_json)
            if isinstance(serialized_semantic_document_signatures_json, str)
            and serialized_semantic_document_signatures_json
            else json.dumps(semantic_document_signatures, separators=(",", ":"))
        ),
        "_serialized_chunk_ids_json": (
            str(serialized_chunk_ids_json)
            if isinstance(serialized_chunk_ids_json, str) and serialized_chunk_ids_json
            else json.dumps(chunk_ids, separators=(",", ":"))
        ),
    }
    _cache_snapshot_metadata(path, metadata)
    return metadata


def _write_file_chunk_snapshot(
    cache_dir: Path,
    *,
    repository_state_fingerprint,
    build_config_digest,
    chunks,
    publish_shared=True,
    diagnostics=None,
):
    payload = {
        "schema": FILE_CHUNK_SNAPSHOT_SCHEMA,
        "repository_state_fingerprint": str(repository_state_fingerprint or ""),
        "build_config_digest": str(build_config_digest or ""),
        "chunks": list(chunks),
        "lexical_manifest": getattr(chunks, "_lexical_manifest", {}),
        "index_manifest": getattr(chunks, "_index_manifest", {}),
        "semantic_document_signatures": getattr(chunks, "_semantic_document_signatures", {}),
        "chunk_ids": list(getattr(chunks, "_chunk_ids", ())),
        "file_key_by_chunk_id": getattr(chunks, "_file_key_by_chunk_id", {}),
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    local_snapshot_path = _file_chunk_snapshot_path(cache_dir)
    local_metadata_path = _file_chunk_snapshot_metadata_path(cache_dir)
    shared_snapshot_path = (
        _shared_file_chunk_snapshot_path(repository_state_fingerprint, build_config_digest, create=True)
        if publish_shared
        else None
    )
    shared_metadata_path = (
        _shared_file_chunk_snapshot_metadata_path(repository_state_fingerprint, build_config_digest, create=True)
        if publish_shared
        else None
    )
    normalized_payload = _normalize_snapshot_payload(payload)
    started = time.perf_counter()
    if not _path_bytes_equal(local_snapshot_path, payload_bytes):
        _atomic_private_bytes(local_snapshot_path, payload_bytes)
    if isinstance(diagnostics, dict):
        diagnostics["snapshot_payload_local_write_ms"] = diagnostics.get("snapshot_payload_local_write_ms", 0.0) + round(
            (time.perf_counter() - started) * 1000.0, 3
        )
    _cache_file_chunk_snapshot_payload(local_snapshot_path, normalized_payload)
    if shared_snapshot_path is not None and shared_snapshot_path != local_snapshot_path:
        with contextlib.suppress(OSError):
            started = time.perf_counter()
            _clone_private_cache_file(local_snapshot_path, shared_snapshot_path)
            if normalized_payload is not None:
                _cache_file_chunk_snapshot_payload(shared_snapshot_path, normalized_payload)
            if isinstance(diagnostics, dict):
                diagnostics["snapshot_payload_shared_publish_ms"] = diagnostics.get(
                    "snapshot_payload_shared_publish_ms", 0.0
                ) + round((time.perf_counter() - started) * 1000.0, 3)
    if shared_metadata_path is not None and shared_metadata_path != local_metadata_path:
        with contextlib.suppress(OSError):
            pass
    local_metadata = _write_file_chunk_snapshot_metadata(
        local_metadata_path,
        repository_state_fingerprint=repository_state_fingerprint,
        build_config_digest=build_config_digest,
        chunks=chunks,
        diagnostics=diagnostics,
    )
    if shared_metadata_path is not None and shared_metadata_path != local_metadata_path:
        with contextlib.suppress(OSError):
            started = time.perf_counter()
            _clone_private_cache_file(local_metadata_path, shared_metadata_path)
            if isinstance(local_metadata, dict):
                _cache_snapshot_metadata(shared_metadata_path, local_metadata)
            if isinstance(diagnostics, dict):
                diagnostics["snapshot_metadata_shared_publish_ms"] = diagnostics.get(
                    "snapshot_metadata_shared_publish_ms", 0.0
                ) + round((time.perf_counter() - started) * 1000.0, 3)


def _reuse_file_chunk_snapshot_payload(
    cache_dir: Path,
    *,
    previous_repository_state_fingerprint,
    repository_state_fingerprint,
    build_config_digest,
    chunks,
    publish_shared=True,
    diagnostics=None,
):
    previous_local_metadata = _load_file_chunk_snapshot_metadata(cache_dir) or {}
    serialized_index_manifest_json = str(previous_local_metadata.get("_serialized_index_manifest_json") or "")
    serialized_semantic_document_signatures_json = str(
        previous_local_metadata.get("_serialized_semantic_document_signatures_json") or ""
    )
    serialized_chunk_ids_json = str(previous_local_metadata.get("_serialized_chunk_ids_json") or "")
    local_metadata_path = _file_chunk_snapshot_metadata_path(cache_dir)
    local_metadata = _write_file_chunk_snapshot_metadata(
        local_metadata_path,
        repository_state_fingerprint=repository_state_fingerprint,
        build_config_digest=build_config_digest,
        chunks=chunks,
        serialized_index_manifest_json=serialized_index_manifest_json,
        serialized_semantic_document_signatures_json=serialized_semantic_document_signatures_json,
        serialized_chunk_ids_json=serialized_chunk_ids_json,
        diagnostics=diagnostics,
    )
    if not publish_shared:
        return
    shared_metadata_path = _shared_file_chunk_snapshot_metadata_path(
        repository_state_fingerprint,
        build_config_digest,
        create=True,
    )
    if shared_metadata_path is not None:
        started = time.perf_counter()
        _clone_private_cache_file(local_metadata_path, shared_metadata_path)
        if isinstance(local_metadata, dict):
            _cache_snapshot_metadata(shared_metadata_path, local_metadata)
        if isinstance(diagnostics, dict):
            diagnostics["snapshot_metadata_shared_publish_ms"] = diagnostics.get(
                "snapshot_metadata_shared_publish_ms", 0.0
            ) + round((time.perf_counter() - started) * 1000.0, 3)
    shared_snapshot_path = _shared_file_chunk_snapshot_path(
        repository_state_fingerprint,
        build_config_digest,
        create=True,
    )
    if shared_snapshot_path is None or shared_snapshot_path.exists():
        return
    source_path = _shared_file_chunk_snapshot_path(
        previous_repository_state_fingerprint,
        build_config_digest,
        create=False,
    )
    if source_path is None or not source_path.exists():
        local_snapshot_path = _file_chunk_snapshot_path(cache_dir)
        source_path = local_snapshot_path if local_snapshot_path.exists() else None
    started = time.perf_counter()
    _clone_private_cache_file(source_path, shared_snapshot_path)
    if isinstance(diagnostics, dict):
        diagnostics["snapshot_payload_shared_publish_ms"] = diagnostics.get(
            "snapshot_payload_shared_publish_ms", 0.0
        ) + round((time.perf_counter() - started) * 1000.0, 3)
    if source_path is not None and source_path.exists():
        source_cache_key = _snapshot_metadata_cache_key(source_path)
        if source_cache_key is not None:
            cached = _FILE_CHUNK_SNAPSHOT_MEMORY_CACHE.get(source_cache_key)
            if cached is not None:
                _cache_file_chunk_snapshot_payload(shared_snapshot_path, cached)


def _write_file_chunk_snapshot_metadata_only(
    cache_dir: Path,
    *,
    repository_state_fingerprint,
    build_config_digest,
    chunks,
    publish_shared=True,
    diagnostics=None,
):
    previous_local_metadata = _load_file_chunk_snapshot_metadata(cache_dir) or {}
    serialized_index_manifest_json = str(previous_local_metadata.get("_serialized_index_manifest_json") or "")
    serialized_semantic_document_signatures_json = str(
        previous_local_metadata.get("_serialized_semantic_document_signatures_json") or ""
    )
    serialized_chunk_ids_json = str(previous_local_metadata.get("_serialized_chunk_ids_json") or "")
    local_metadata_path = _file_chunk_snapshot_metadata_path(cache_dir)
    local_metadata = _write_file_chunk_snapshot_metadata(
        local_metadata_path,
        repository_state_fingerprint=repository_state_fingerprint,
        build_config_digest=build_config_digest,
        chunks=chunks,
        serialized_index_manifest_json=serialized_index_manifest_json,
        serialized_semantic_document_signatures_json=serialized_semantic_document_signatures_json,
        serialized_chunk_ids_json=serialized_chunk_ids_json,
        diagnostics=diagnostics,
    )
    if not publish_shared:
        return
    shared_metadata_path = _shared_file_chunk_snapshot_metadata_path(
        repository_state_fingerprint,
        build_config_digest,
        create=True,
    )
    if shared_metadata_path is not None and shared_metadata_path != local_metadata_path:
        with contextlib.suppress(OSError):
            started = time.perf_counter()
            _clone_private_cache_file(local_metadata_path, shared_metadata_path)
            if isinstance(local_metadata, dict):
                _cache_snapshot_metadata(shared_metadata_path, local_metadata)
            if isinstance(diagnostics, dict):
                diagnostics["snapshot_metadata_shared_publish_ms"] = diagnostics.get(
                    "snapshot_metadata_shared_publish_ms", 0.0
                ) + round((time.perf_counter() - started) * 1000.0, 3)


def _chunk_build_config_digest(
    discovered,
    namespaces,
    *,
    max_lines,
    overlap,
    excluded_dir_names,
    excluded_paths,
):
    discovered_roots = []
    for item in discovered:
        path = item.get("path")
        if path is None:
            continue
        try:
            discovered_roots.append((str(namespaces.get(id(item), "")), Path(path).resolve(strict=False)))
        except OSError:
            continue

    def stable_input_locator(item):
        uri = str(item.get("uri") or "").strip()
        if uri:
            return {"uri": uri}
        path = item.get("path")
        if path is not None:
            return {"path": str(Path(path).resolve(strict=False))}
        content_hash = str(item.get("content_hash") or "").strip()
        if content_hash:
            return {"content_hash": content_hash}
        return {"path": ""}

    def normalize_excluded_path(path):
        resolved = Path(path).resolve(strict=False)
        matches = []
        for namespace, root in discovered_roots:
            try:
                rel = resolved.relative_to(root).as_posix()
            except ValueError:
                continue
            matches.append({"source_namespace": namespace, "relative_path": rel or "."})
        if matches:
            return sorted(matches, key=lambda item: (item["source_namespace"], item["relative_path"]))
        return None

    normalized_inputs = []
    for item in discovered:
        normalized_inputs.append(
            {
                "id": str(item.get("id", "")),
                "type": str(item.get("type", "")),
                "classification": str(item.get("classification", "unknown")),
                "source_namespace": str(namespaces.get(id(item), "")),
                **stable_input_locator(item),
            }
        )
    normalized_excluded_paths = []
    for path in sorted((Path(path).resolve(strict=False) for path in (excluded_paths or ())), key=str):
        normalized = normalize_excluded_path(path)
        if normalized is None:
            continue
        normalized_excluded_paths.extend(normalized)
    payload = {
        "schema": FILE_CHUNK_WORKING_MANIFEST_SCHEMA,
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "max_lines": int(max_lines),
        "overlap": int(overlap),
        "excluded_dir_names": sorted(str(item) for item in (excluded_dir_names or ())),
        "excluded_paths": normalized_excluded_paths,
        "inputs": normalized_inputs,
    }
    return f"sha256:{sha256_text(json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True))}"


def _snapshot_build_context(
    discovered,
    excluded_dir_names=None,
    *,
    max_lines=120,
    overlap=12,
    cache_dir=None,
    excluded_paths=None,
    transient_excluded_paths=None,
):
    discovered = list(discovered)
    namespaces = {}
    used_namespaces = set()
    for index, item in enumerate(discovered):
        raw_namespace = str(item.get("id") or f"input_{index}")
        namespace = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_namespace).strip("._-") or f"input_{index}"
        if namespace in used_namespaces:
            source_key = f"{raw_namespace}\0{item.get('uri', '')}\0{item.get('path', '')}"
            namespace = f"{namespace}_{sha256_text(source_key)[:8]}"
        used_namespaces.add(namespace)
        namespaces[id(item)] = namespace
    configured_excluded_paths = set(Path(path).resolve(strict=False) for path in (excluded_paths or ()))
    resolved_cache_dir = Path(cache_dir).resolve(strict=False) if cache_dir is not None else None
    if resolved_cache_dir is not None:
        configured_excluded_paths.discard(resolved_cache_dir)
    for transient_path in (transient_excluded_paths or ()):
        configured_excluded_paths.discard(Path(transient_path).resolve(strict=False))
    effective_excluded_paths = set(configured_excluded_paths)
    if resolved_cache_dir is not None:
        effective_excluded_paths.add(resolved_cache_dir)
    for transient_path in (transient_excluded_paths or ()):
        effective_excluded_paths.add(Path(transient_path).resolve(strict=False))
    build_config_digest = _chunk_build_config_digest(
        discovered,
        namespaces,
        max_lines=max_lines,
        overlap=overlap,
        excluded_dir_names=excluded_dir_names,
        excluded_paths=configured_excluded_paths,
    )
    return discovered, namespaces, effective_excluded_paths, build_config_digest


def _load_cached_chunk_snapshot(cache_dir: Path, manifest_files):
    combined_snapshot = _load_file_chunk_snapshot(cache_dir)
    if combined_snapshot is not None:
        chunks = _snapshot_chunks(combined_snapshot)
        chunks._snapshot_restore_diagnostics = {
            "source": "local_manifest_snapshot",
            "local_snapshot_load_ms": 0.0,
            "shared_snapshot_load_ms": 0.0,
        }
        return chunks, {
            "total_files": len(manifest_files),
            "reused_files": len(manifest_files),
            "rebuilt_files": 0,
            "snapshot_cache_hit": True,
        }
    chunks = ChunkList()
    lexical_manifest = {}
    index_manifest = {}
    chunks_by_file = {}
    semantic_document_signatures = {}
    total_files = 0
    for _manifest_entry_key, record in manifest_files.items():
        total_files += 1
        if bool(record.get("empty")):
            continue
        cache_key = str(record.get("cache_key") or "")
        if not cache_key:
            return None
        cached_bundle = _load_file_chunk_cache_bundle(cache_dir, cache_key)
        if cached_bundle is None:
            return None
        cached_chunks = cached_bundle["chunks"]
        lexical_record = cached_bundle["lexical_record"]
        index_signature = cached_bundle["index_signature"]
        if lexical_record is not None:
            lexical_manifest[lexical_record["file_key"]] = lexical_record
            index_manifest[lexical_record["file_key"]] = index_signature
            chunks_by_file[lexical_record["file_key"]] = list(cached_chunks)
        semantic_document_signatures.update(dict(cached_bundle["semantic_document_signatures"] or {}))
        chunks.extend(cached_chunks)
    chunks._lexical_manifest = lexical_manifest
    chunks._index_manifest = index_manifest
    chunks._chunks_by_file = chunks_by_file
    chunks._semantic_document_signatures = semantic_document_signatures
    chunks._chunk_ids = tuple(str(chunk.get("chunk_id") or "") for chunk in chunks)
    file_key_by_chunk_id = {}
    for file_key, file_chunks in chunks_by_file.items():
        for chunk in file_chunks:
            file_key_by_chunk_id[str(chunk.get("chunk_id") or "")] = str(file_key)
    chunks._file_key_by_chunk_id = file_key_by_chunk_id
    chunks._snapshot_restore_diagnostics = {
        "source": "manifest_bundle_rebuild",
        "local_snapshot_load_ms": 0.0,
        "shared_snapshot_load_ms": 0.0,
    }
    return chunks, {
        "total_files": total_files,
        "reused_files": total_files,
        "rebuilt_files": 0,
        "snapshot_cache_hit": True,
    }


def _snapshot_state(snapshot):
    if isinstance(snapshot, ChunkList):
        lexical_manifest = dict(getattr(snapshot, "_lexical_manifest", None) or {})
        index_manifest = dict(getattr(snapshot, "_index_manifest", None) or {})
        chunks_by_file = {
            str(file_key): [dict(chunk) for chunk in chunks]
            for file_key, chunks in dict(getattr(snapshot, "_chunks_by_file", None) or {}).items()
            if isinstance(chunks, list)
        }
        semantic_document_signatures = {
            str(chunk_id): str(signature)
            for chunk_id, signature in dict(getattr(snapshot, "_semantic_document_signatures", None) or {}).items()
        }
        chunk_ids = tuple(str(value) for value in (getattr(snapshot, "_chunk_ids", None) or ()))
        file_key_by_chunk_id = {
            str(chunk_id): str(file_key)
            for chunk_id, file_key in dict(getattr(snapshot, "_file_key_by_chunk_id", None) or {}).items()
        }
        return lexical_manifest, index_manifest, chunks_by_file, semantic_document_signatures, chunk_ids, file_key_by_chunk_id
    lexical_manifest = dict((snapshot or {}).get("lexical_manifest") or {})
    index_manifest = dict((snapshot or {}).get("index_manifest") or {})
    chunks_by_file = {
        str(file_key): [dict(chunk) for chunk in chunks]
        for file_key, chunks in dict((snapshot or {}).get("chunks_by_file") or {}).items()
        if isinstance(chunks, list)
    }
    semantic_document_signatures = {
        str(chunk_id): str(signature)
        for chunk_id, signature in dict((snapshot or {}).get("semantic_document_signatures") or {}).items()
    }
    chunk_ids = tuple(str(value) for value in (snapshot or {}).get("chunk_ids") or ())
    file_key_by_chunk_id = {
        str(chunk_id): str(file_key)
        for chunk_id, file_key in dict((snapshot or {}).get("file_key_by_chunk_id") or {}).items()
    }
    return lexical_manifest, index_manifest, chunks_by_file, semantic_document_signatures, chunk_ids, file_key_by_chunk_id


def _snapshot_chunks(snapshot):
    chunks = ChunkList(dict(chunk) for chunk in (snapshot or {}).get("chunks") or ())
    chunks._lexical_manifest = dict((snapshot or {}).get("lexical_manifest") or {})
    chunks._index_manifest = dict((snapshot or {}).get("index_manifest") or {})
    chunks._chunks_by_file = {
        str(file_key): [dict(chunk) for chunk in file_chunks]
        for file_key, file_chunks in dict((snapshot or {}).get("chunks_by_file") or {}).items()
        if isinstance(file_chunks, list)
    }
    chunks._semantic_document_signatures = dict((snapshot or {}).get("semantic_document_signatures") or {})
    chunks._chunk_ids = tuple(str(value) for value in ((snapshot or {}).get("chunk_ids") or ()))
    chunks._file_key_by_chunk_id = dict((snapshot or {}).get("file_key_by_chunk_id") or {})
    return chunks


def load_cached_chunk_snapshot(cache_dir, *, repository_state_fingerprint=None, build_config_digest=None):
    cache_dir = Path(cache_dir)
    local_snapshot_path = _file_chunk_snapshot_path(cache_dir)
    local_metadata = _load_file_chunk_snapshot_metadata(cache_dir)
    started = time.perf_counter()
    local_load_ms = 0.0
    shared_load_ms = 0.0
    source = "none"
    snapshot = _load_file_chunk_snapshot(cache_dir)
    local_load_ms = (time.perf_counter() - started) * 1000.0
    if snapshot is None:
        if not repository_state_fingerprint or not str(build_config_digest or ""):
            return None
        shared_path = _shared_file_chunk_snapshot_path(
            str(repository_state_fingerprint or ""),
            str(build_config_digest or ""),
            create=False,
        )
        shared_metadata_path = _shared_file_chunk_snapshot_metadata_path(
            str(repository_state_fingerprint or ""),
            str(build_config_digest or ""),
            create=False,
        )
        if shared_path is None:
            return None
        shared_started = time.perf_counter()
        try:
            payload = json.loads(shared_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            with contextlib.suppress(OSError):
                shared_path.unlink()
            return None
        shared_load_ms = (time.perf_counter() - shared_started) * 1000.0
        snapshot = _normalize_snapshot_payload(payload)
        if snapshot is None:
            with contextlib.suppress(OSError):
                shared_path.unlink()
            return None
        source = "shared"
    else:
        source = "local"
    requested_repository_state_fingerprint = str(repository_state_fingerprint or "")
    requested_build_config_digest = str(build_config_digest or "")
    if source == "local" and local_metadata is not None:
        if (
            (not requested_repository_state_fingerprint or str(local_metadata.get("repository_state_fingerprint") or "") == requested_repository_state_fingerprint)
            and (not requested_build_config_digest or str(local_metadata.get("build_config_digest") or "") == requested_build_config_digest)
            and _snapshot_matches_metadata(snapshot, local_metadata)
        ):
            repository_state_fingerprint = None
            build_config_digest = None
    elif source == "shared" and shared_metadata_path is not None and shared_metadata_path.exists():
        try:
            shared_metadata_payload = json.loads(shared_metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            shared_metadata_payload = None
        shared_metadata = _normalize_file_chunk_snapshot_metadata_payload(shared_metadata_payload)
        if (
            shared_metadata is not None
            and (not requested_repository_state_fingerprint or str(shared_metadata.get("repository_state_fingerprint") or "") == requested_repository_state_fingerprint)
            and (not requested_build_config_digest or str(shared_metadata.get("build_config_digest") or "") == requested_build_config_digest)
            and _snapshot_matches_metadata(snapshot, shared_metadata)
        ):
            repository_state_fingerprint = None
            build_config_digest = None
    if repository_state_fingerprint is not None and (
        str(snapshot.get("repository_state_fingerprint") or "") != str(repository_state_fingerprint or "")
    ):
        return None
    if build_config_digest is not None and str(build_config_digest or "") and (
        str(snapshot.get("build_config_digest") or "") != str(build_config_digest or "")
    ):
        return None
    chunks = _snapshot_chunks(snapshot)
    chunks._snapshot_restore_diagnostics = {
        "source": source,
        "local_snapshot_load_ms": round(local_load_ms, 3),
        "shared_snapshot_load_ms": round(shared_load_ms, 3),
    }
    return chunks


def _drop_file_state(file_key, lexical_manifest, index_manifest, chunks_by_file, semantic_document_signatures):
    stale_chunks = chunks_by_file.pop(file_key, [])
    lexical_manifest.pop(file_key, None)
    index_manifest.pop(file_key, None)
    for chunk in stale_chunks:
        semantic_document_signatures.pop(str(chunk.get("chunk_id") or ""), None)
    return stale_chunks


def _drop_chunk_file_mapping(stale_chunks, file_key_by_chunk_id):
    for chunk in stale_chunks or ():
        file_key_by_chunk_id.pop(str(chunk.get("chunk_id") or ""), None)


def _install_reused_file_chunks(
    *,
    chunk_rows,
    lexical_record,
    index_signature,
    lexical_manifest,
    index_manifest,
    chunks_by_file,
    semantic_document_signatures,
    file_key_by_chunk_id,
    semantic_signatures=None,
):
    if lexical_record is not None:
        file_key = str(lexical_record["file_key"])
        lexical_manifest[file_key] = lexical_record
        index_manifest[file_key] = index_signature
        chunks_by_file[file_key] = list(chunk_rows)
        for chunk in chunk_rows:
            file_key_by_chunk_id[str(chunk.get("chunk_id") or "")] = file_key
    if isinstance(semantic_signatures, dict):
        semantic_document_signatures.update({str(k): str(v) for k, v in semantic_signatures.items()})
    else:
        semantic_document_signatures.update(
            {str(chunk.get("chunk_id") or ""): semantic_chunk_signature(chunk) for chunk in chunk_rows}
        )


def build_syntax_chunks(
    discovered,
    excluded_dir_names=None,
    max_lines=120,
    overlap=12,
    *,
    cache_dir=None,
    excluded_paths=None,
    repository_state_fingerprint=None,
    repository_fingerprint_state=None,
    git_probe_cache=None,
    return_diagnostics=False,
    transient_excluded_paths=None,
    touched_paths_hint=None,
    publish_shared_cache_publication=True,
):
    git_probe_cache = _effective_git_probe_cache(git_probe_cache)
    discovered, namespaces, effective_excluded_paths, build_config_digest = _snapshot_build_context(
        discovered,
        excluded_dir_names,
        max_lines=max_lines,
        overlap=overlap,
        cache_dir=cache_dir,
        excluded_paths=excluded_paths,
        transient_excluded_paths=transient_excluded_paths,
    )
    multiple_inputs = len(discovered) > 1

    diagnostics = {
        "total_files": 0,
        "reused_files": 0,
        "rebuilt_files": 0,
        "snapshot_cache_hit": False,
    }
    build_substage = {
        "manifest_load_ms": 0.0,
        "shared_manifest_load_ms": 0.0,
        "shared_state_manifest_load_ms": 0.0,
        "snapshot_probe_ms": 0.0,
        "discover_source_files_ms": 0.0,
        "discover_from_clean_hint_ms": 0.0,
        "discover_from_previous_manifest_ms": 0.0,
        "discover_generic_fallback_ms": 0.0,
        "discover_seed_manifest_ms": 0.0,
        "git_dirty_manifest_keys_ms": 0.0,
        "previous_snapshot_load_ms": 0.0,
        "git_file_signatures_ms": 0.0,
        "shared_state_lookup_ms": 0.0,
        "file_chunk_bundle_load_ms": 0.0,
        "file_chunk_bundle_write_ms": 0.0,
        "source_read_ms": 0.0,
        "chunk_build_ms": 0.0,
        "chunk_content_hash_ms": 0.0,
        "chunk_identity_hash_ms": 0.0,
        "chunk_token_estimate_ms": 0.0,
        "manifest_records_ms": 0.0,
        "semantic_signature_ms": 0.0,
        "symbol_marker_ms": 0.0,
        "symbol_marker_write_ms": 0.0,
        "working_manifest_write_ms": 0.0,
        "working_manifest_local_write_ms": 0.0,
        "working_manifest_shared_publish_ms": 0.0,
        "working_manifest_shared_latest_publish_ms": 0.0,
        "shared_state_manifest_flush_ms": 0.0,
        "snapshot_write_ms": 0.0,
        "snapshot_payload_local_write_ms": 0.0,
        "snapshot_payload_shared_publish_ms": 0.0,
        "snapshot_metadata_local_write_ms": 0.0,
        "snapshot_metadata_shared_publish_ms": 0.0,
    }
    manifest_path = _file_chunk_manifest_path(cache_dir) if cache_dir is not None else None
    shared_manifest_path = (
        _shared_file_chunk_working_manifest_path(
            str(repository_state_fingerprint or ""),
            build_config_digest,
            create=False,
        )
        if cache_dir is not None and repository_state_fingerprint
        else None
    )
    shared_latest_manifest_path = (
        _shared_file_chunk_latest_manifest_path(build_config_digest, create=False)
        if cache_dir is not None and build_config_digest
        else None
    )
    manifest_restore_ms = 0.0
    shared_manifest_load_ms = 0.0
    manifest_started = time.perf_counter()
    previous_manifest_payload = _load_file_chunk_working_manifest(manifest_path) if manifest_path is not None else None
    manifest_restore_ms = round((time.perf_counter() - manifest_started) * 1000.0, 3)
    build_substage["manifest_load_ms"] = manifest_restore_ms
    if previous_manifest_payload is None and shared_manifest_path is not None:
        shared_manifest_started = time.perf_counter()
        previous_manifest_payload = _load_file_chunk_working_manifest(shared_manifest_path)
        shared_manifest_load_ms = round((time.perf_counter() - shared_manifest_started) * 1000.0, 3)
        build_substage["shared_manifest_load_ms"] = shared_manifest_load_ms
    if previous_manifest_payload is None and shared_latest_manifest_path is not None:
        shared_manifest_started = time.perf_counter()
        previous_manifest_payload = _load_file_chunk_working_manifest(shared_latest_manifest_path)
        shared_manifest_load_ms += round((time.perf_counter() - shared_manifest_started) * 1000.0, 3)
        build_substage["shared_manifest_load_ms"] = shared_manifest_load_ms
    previous_manifest_payload = previous_manifest_payload or {}
    previous_manifest = previous_manifest_payload.get("files") or {}
    if (
        cache_dir is not None
        and repository_state_fingerprint
        and str(previous_manifest_payload.get("repository_state_fingerprint") or "") == str(repository_state_fingerprint)
        and str(previous_manifest_payload.get("build_config_digest") or "") == build_config_digest
    ):
        stage_started = time.perf_counter()
        snapshot = load_cached_chunk_snapshot(
            Path(cache_dir),
            repository_state_fingerprint=repository_state_fingerprint,
            build_config_digest=build_config_digest,
        )
        build_substage["snapshot_probe_ms"] = round((time.perf_counter() - stage_started) * 1000.0, 3)
        snapshot_diagnostics = None
        if snapshot is None and previous_manifest:
            snapshot = _load_cached_chunk_snapshot(Path(cache_dir), previous_manifest)
            if snapshot is not None:
                snapshot, snapshot_diagnostics = snapshot
        if snapshot is not None:
            snapshot_diag = dict(getattr(snapshot, "_snapshot_restore_diagnostics", None) or {})
            snapshot._chunk_restore_diagnostics = {
                "manifest_restore_ms": manifest_restore_ms,
                "shared_manifest_load_ms": shared_manifest_load_ms,
                "snapshot_local_load_ms": float(snapshot_diag.get("local_snapshot_load_ms") or 0.0),
                "snapshot_shared_load_ms": float(snapshot_diag.get("shared_snapshot_load_ms") or 0.0),
                "snapshot_restore_source": str(snapshot_diag.get("source") or ""),
            }
            snapshot._chunk_build_substage_timings = dict(build_substage)
            snapshot = (
                snapshot,
                snapshot_diagnostics
                if isinstance(snapshot_diagnostics, dict)
                else {
                    "total_files": len(previous_manifest),
                    "reused_files": len(previous_manifest),
                    "rebuilt_files": 0,
                    "snapshot_cache_hit": True,
                },
            )
        if snapshot is not None:
            chunks, snapshot_diagnostics = snapshot
            diagnostics.update(snapshot_diagnostics)
            if return_diagnostics:
                return chunks, diagnostics
            return chunks
    if cache_dir is not None and repository_state_fingerprint and not previous_manifest:
        stage_started = time.perf_counter()
        snapshot = load_cached_chunk_snapshot(
            Path(cache_dir),
            repository_state_fingerprint=repository_state_fingerprint,
            build_config_digest=build_config_digest,
        )
        build_substage["snapshot_probe_ms"] = round((time.perf_counter() - stage_started) * 1000.0, 3)
        if snapshot is not None:
            snapshot_metadata = load_cached_chunk_snapshot_metadata(
                discovered,
                excluded_dir_names,
                max_lines=max_lines,
                overlap=overlap,
                cache_dir=cache_dir,
                excluded_paths=excluded_paths,
                repository_state_fingerprint=repository_state_fingerprint,
                transient_excluded_paths=transient_excluded_paths,
            )
            total_files = int((snapshot_metadata or {}).get("total_files") or 0)
            if total_files <= 0:
                total_files = len(getattr(snapshot, "_chunks_by_file", None) or {})
            snapshot_diag = dict(getattr(snapshot, "_snapshot_restore_diagnostics", None) or {})
            snapshot._chunk_restore_diagnostics = {
                "manifest_restore_ms": manifest_restore_ms,
                "shared_manifest_load_ms": shared_manifest_load_ms,
                "snapshot_local_load_ms": float(snapshot_diag.get("local_snapshot_load_ms") or 0.0),
                "snapshot_shared_load_ms": float(snapshot_diag.get("shared_snapshot_load_ms") or 0.0),
                "snapshot_restore_source": str(snapshot_diag.get("source") or ""),
            }
            snapshot._chunk_build_substage_timings = dict(build_substage)
            diagnostics.update(
                {
                    "total_files": total_files,
                    "reused_files": total_files,
                    "rebuilt_files": 0,
                    "snapshot_cache_hit": True,
                }
            )
            if return_diagnostics:
                return snapshot, diagnostics
            return snapshot
    lexical_manifest = {}
    index_manifest = {}
    chunks_by_file = {}
    semantic_document_signatures = {}
    file_key_by_chunk_id = {}
    previous_manifest_repo_fp = str(previous_manifest_payload.get("repository_state_fingerprint") or "")
    warm_snapshot_loaded = False
    dirty_manifest_entry_keys = set()
    manifest_dirty_keys_computed = False
    publish_shared_cache_publication = bool(publish_shared_cache_publication)
    publish_shared_file_level_cache = bool(publish_shared_cache_publication)
    stage_started = time.perf_counter()
    legacy_shared_state_manifest_path = _shared_file_chunk_state_manifest_path(create=False)
    shared_state_manifest_cache = _shared_file_chunk_state_manifest_cache(
        legacy_available=bool(
            legacy_shared_state_manifest_path is not None and legacy_shared_state_manifest_path.exists()
        )
    )
    _seed_shared_state_manifest_cache_from_working_manifest(shared_state_manifest_cache, previous_manifest)
    build_substage["shared_state_manifest_load_ms"] = round((time.perf_counter() - stage_started) * 1000.0, 3)
    stage_started = time.perf_counter()
    prefer_inventory_delta = bool(
        previous_manifest
        and repository_state_fingerprint
        and previous_manifest_repo_fp
        and previous_manifest_repo_fp != str(repository_state_fingerprint)
    )
    ignored = set(DEFAULT_IGNORE_DIRS)
    ignored.update(excluded_dir_names or ())
    normalized_touched_paths_hint = _normalize_touched_paths_hint(touched_paths_hint)
    if not normalized_touched_paths_hint:
        normalized_touched_paths_hint = _touched_paths_hint_from_repository_fingerprint_state(
            repository_fingerprint_state
        )
    if not normalized_touched_paths_hint:
        cached_dirty_paths = []
        for item in discovered:
            input_path = item.get("path")
            if input_path is None:
                continue
            input_path = Path(input_path)
            probe_root = input_path if input_path.is_dir() else input_path.parent
            git_top = _git_top(probe_root)
            if git_top is None:
                continue
            scope_rel = _git_scope_rel(probe_root, git_top)
            normalized_scope = [] if scope_rel in {"", "."} else [scope_rel]
            derived_dirty_paths = _cached_git_fastpath_dirty_paths(
                git_top,
                normalized_scope,
                git_probe_cache=git_probe_cache,
            )
            if derived_dirty_paths is None:
                continue
            if scope_rel in {"", "."}:
                cached_dirty_paths.extend(derived_dirty_paths)
            else:
                prefix = f"{scope_rel.rstrip('/')}/"
                for rel in derived_dirty_paths:
                    if rel == scope_rel:
                        cached_dirty_paths.append(".")
                    elif rel.startswith(prefix):
                        cached_dirty_paths.append(rel[len(prefix):])
        normalized_touched_paths_hint = _normalize_touched_paths_hint(cached_dirty_paths)
    clean_worktree_files_hint = ()
    if not normalized_touched_paths_hint:
        clean_worktree_files_hint = _clean_worktree_files_hint_from_repository_fingerprint_state(
            repository_fingerprint_state
        )
    if prefer_inventory_delta and len(previous_manifest) <= SMALL_REPO_INVENTORY_DELTA_THRESHOLD:
        prefetched_status_roots = set()
        if not normalized_touched_paths_hint:
            for item in discovered:
                input_path = item.get("path")
                if input_path is None:
                    continue
                input_path = Path(input_path)
                probe_root = input_path if input_path.is_dir() else input_path.parent
                git_top = _git_top(probe_root)
                if git_top is None:
                    continue
                scope_rel = _git_scope_rel(probe_root, git_top)
                if scope_rel not in {"", "."}:
                    continue
                top_key = str(git_top.resolve(strict=False))
                if top_key in prefetched_status_roots:
                    continue
                if _cached_repo_scope_status_snapshot(git_top, git_probe_cache=git_probe_cache) is None:
                    _scoped_git_status_snapshot(git_top, [], git_probe_cache=git_probe_cache)
                prefetched_status_roots.add(top_key)
    discover_substage_started = time.perf_counter()
    discovered_files = _discovered_files_from_clean_worktree_hint(
        discovered,
        ignored,
        ignored_paths=effective_excluded_paths,
        git_probe_cache=git_probe_cache,
        clean_worktree_files_hint=clean_worktree_files_hint,
    )
    build_substage["discover_from_clean_hint_ms"] += round((time.perf_counter() - discover_substage_started) * 1000.0, 3)
    if (
        discovered_files is None
        and previous_manifest
        and normalized_touched_paths_hint
        and repository_state_fingerprint
        and previous_manifest_repo_fp
        and previous_manifest_repo_fp != str(repository_state_fingerprint)
    ):
        discover_substage_started = time.perf_counter()
        trust_untouched_manifest = (
            str(previous_manifest_payload.get("build_config_digest") or "") == str(build_config_digest or "")
        )
        discovered_files = _discovered_files_from_previous_manifest(
            discovered,
            namespaces,
            previous_manifest,
            ignored=ignored,
            ignored_paths=effective_excluded_paths,
            touched_paths_hint=normalized_touched_paths_hint,
            trust_untouched_manifest=trust_untouched_manifest,
        )
        build_substage["discover_from_previous_manifest_ms"] += round(
            (time.perf_counter() - discover_substage_started) * 1000.0, 3
        )
    if discovered_files is None:
        discover_substage_started = time.perf_counter()
        discovered_files = discover_source_files(
            discovered,
            excluded_dir_names,
            excluded_paths=effective_excluded_paths,
            cache_dir=cache_dir,
            repository_state_fingerprint=repository_state_fingerprint,
            git_probe_cache=git_probe_cache,
            prefer_inventory_delta=prefer_inventory_delta,
            touched_paths_hint=normalized_touched_paths_hint,
        )
        build_substage["discover_generic_fallback_ms"] += round(
            (time.perf_counter() - discover_substage_started) * 1000.0, 3
        )
    else:
        discover_substage_started = time.perf_counter()
        _seed_discovery_manifests_from_clean_worktree_hint(
            discovered,
            ignored,
            cache_dir=cache_dir,
            ignored_paths=effective_excluded_paths,
            repository_state_fingerprint=repository_state_fingerprint,
            clean_worktree_files_hint=clean_worktree_files_hint,
        )
        build_substage["discover_seed_manifest_ms"] += round(
            (time.perf_counter() - discover_substage_started) * 1000.0, 3
        )
    build_substage["discover_source_files_ms"] = round((time.perf_counter() - stage_started) * 1000.0, 3)
    if (
        cache_dir is not None
        and previous_manifest
        and str(previous_manifest_payload.get("build_config_digest") or "") == build_config_digest
        and str(previous_manifest_payload.get("repository_state_fingerprint") or "")
        and str(previous_manifest_payload.get("repository_state_fingerprint") or "") != str(repository_state_fingerprint or "")
    ):
        stage_started = time.perf_counter()
        if normalized_touched_paths_hint:
            touched_rel_paths = set(normalized_touched_paths_hint)
            dirty_manifest_entry_keys = {
                _file_chunk_manifest_entry_key(
                    source_namespace=namespaces[id(item)],
                    rel_path=rel,
                )
                for item, candidate, rel in discovered_files
                if rel in touched_rel_paths
            }
        else:
            dirty_manifest_entry_keys = _git_dirty_manifest_entry_keys(
                discovered_files,
                namespaces,
                git_probe_cache=git_probe_cache,
            )
        manifest_dirty_keys_computed = True
        build_substage["git_dirty_manifest_keys_ms"] = round((time.perf_counter() - stage_started) * 1000.0, 3)
        if dirty_manifest_entry_keys is not None:
            stage_started = time.perf_counter()
            previous_snapshot = load_cached_chunk_snapshot(
                Path(cache_dir),
                repository_state_fingerprint=str(previous_manifest_payload.get("repository_state_fingerprint") or ""),
                build_config_digest=build_config_digest,
            )
            build_substage["previous_snapshot_load_ms"] = round((time.perf_counter() - stage_started) * 1000.0, 3)
            if previous_snapshot is not None:
                (
                    lexical_manifest,
                    index_manifest,
                    chunks_by_file,
                    semantic_document_signatures,
                    _chunk_ids,
                    file_key_by_chunk_id,
                ) = _snapshot_state(previous_snapshot)
                warm_snapshot_loaded = True
                dirty_manifest_entry_keys = set(dirty_manifest_entry_keys)
                publish_shared_cache_publication = False
                build_substage["previous_snapshot_load_ms"] = round(
                    max(
                        build_substage["previous_snapshot_load_ms"],
                        float((getattr(previous_snapshot, "_snapshot_restore_diagnostics", {}) or {}).get("local_snapshot_load_ms") or 0.0)
                        + float((getattr(previous_snapshot, "_snapshot_restore_diagnostics", {}) or {}).get("shared_snapshot_load_ms") or 0.0),
                    ),
                    3,
                )
    next_manifest = {}
    seen_manifest_keys = set()
    file_order = []
    reusable_manifest_signatures = {}
    signature_probe_files = []
    same_repository_state = bool(
        previous_manifest
        and repository_state_fingerprint
        and previous_manifest_repo_fp
        and previous_manifest_repo_fp == str(repository_state_fingerprint)
    )
    have_partial_dirty_manifest = bool(
        previous_manifest
        and previous_manifest_repo_fp
        and previous_manifest_repo_fp != str(repository_state_fingerprint or "")
    )
    defer_cold_git_signatures = bool(
        repository_state_fingerprint
        and not previous_manifest
        and not warm_snapshot_loaded
        and not same_repository_state
        and not have_partial_dirty_manifest
    )
    skip_true_cold_local_bundle_probe = bool(
        repository_state_fingerprint
        and not previous_manifest
        and not warm_snapshot_loaded
        and not same_repository_state
        and not have_partial_dirty_manifest
    )
    if dirty_manifest_entry_keys is None:
        have_partial_dirty_manifest = False
    if (
        previous_manifest
        and not warm_snapshot_loaded
        and not same_repository_state
        and not have_partial_dirty_manifest
        and not manifest_dirty_keys_computed
    ):
        stage_started = time.perf_counter()
        computed_dirty_manifest_entry_keys = _git_dirty_manifest_entry_keys(
            discovered_files,
            namespaces,
            git_probe_cache=git_probe_cache,
        )
        build_substage["git_dirty_manifest_keys_ms"] += round((time.perf_counter() - stage_started) * 1000.0, 3)
        manifest_dirty_keys_computed = True
        if computed_dirty_manifest_entry_keys is not None:
            dirty_manifest_entry_keys = set(computed_dirty_manifest_entry_keys)
            have_partial_dirty_manifest = True
            publish_shared_cache_publication = False
    publish_shared_file_level_cache = bool(publish_shared_cache_publication)
    for item, path, rel_path in discovered_files:
        source_namespace = namespaces[id(item)]
        manifest_entry_key = _file_chunk_manifest_entry_key(source_namespace=source_namespace, rel_path=rel_path)
        previous = previous_manifest.get(manifest_entry_key) or {}
        previous_signature = str(previous.get("signature") or "")
        if previous_signature and (
            same_repository_state
            or (have_partial_dirty_manifest and manifest_entry_key not in dirty_manifest_entry_keys)
        ):
            reusable_manifest_signatures[_path_cache_lookup_key(path)] = previous_signature
            continue
        if have_partial_dirty_manifest and manifest_entry_key in dirty_manifest_entry_keys:
            continue
        signature_probe_files.append((item, path, rel_path))
    if warm_snapshot_loaded:
        git_signatures = {}
        build_substage["git_file_signatures_ms"] = 0.0
    elif defer_cold_git_signatures:
        git_signatures = dict(reusable_manifest_signatures)
        build_substage["git_file_signatures_ms"] = 0.0
    elif not signature_probe_files:
        git_signatures = dict(reusable_manifest_signatures)
        build_substage["git_file_signatures_ms"] = 0.0
    else:
        stage_started = time.perf_counter()
        git_signatures = _git_file_signatures(
            signature_probe_files,
            cache_dir=cache_dir,
            git_probe_cache=git_probe_cache,
        )
        git_signatures.update(reusable_manifest_signatures)
        build_substage["git_file_signatures_ms"] = round((time.perf_counter() - stage_started) * 1000.0, 3)
    for item, path, rel_path in discovered_files:
        diagnostics["total_files"] += 1
        source_namespace = namespaces[id(item)]
        display_path = f"{source_namespace}/{rel_path}" if multiple_inputs else rel_path
        language = language_for_path(path)
        config_key = sha256_text(
            json.dumps(
                {
                    "schema": FILE_CHUNK_CACHE_SCHEMA,
                    "index_schema_version": INDEX_SCHEMA_VERSION,
                    "source_namespace": source_namespace,
                    "repository_path": rel_path,
                    "language": language,
                    "input_id": str(item.get("id", "")),
                    "input_type": str(item.get("type", "repo")),
                    "classification": str(item.get("classification", "unknown")),
                    "max_lines": int(max_lines),
                    "overlap": int(overlap),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        )
        manifest_entry_key = _file_chunk_manifest_entry_key(source_namespace=source_namespace, rel_path=rel_path)
        seen_manifest_keys.add(manifest_entry_key)
        if warm_snapshot_loaded:
            previous = previous_manifest.get(manifest_entry_key) or {}
            if (
                previous.get("config_key") == config_key
                and manifest_entry_key not in dirty_manifest_entry_keys
                and previous.get("signature")
            ):
                if previous.get("empty"):
                    stale_chunks = _drop_file_state(manifest_entry_key, lexical_manifest, index_manifest, chunks_by_file, semantic_document_signatures)
                    _drop_chunk_file_mapping(stale_chunks, file_key_by_chunk_id)
                    next_manifest[manifest_entry_key] = {
                        "signature": str(previous.get("signature") or ""),
                        "cache_key": "",
                        "config_key": config_key,
                        "empty": True,
                    }
                    diagnostics["reused_files"] += 1
                    continue
                if manifest_entry_key in chunks_by_file:
                    next_manifest[manifest_entry_key] = {
                        "signature": str(previous.get("signature") or ""),
                        "cache_key": str(previous.get("cache_key") or ""),
                        "config_key": config_key,
                        "empty": False,
                    }
                    diagnostics["reused_files"] += 1
                    file_order.append(manifest_entry_key)
                    continue
        git_signature = git_signatures.get(_path_cache_lookup_key(path))
        dirty_partial_entry = bool(have_partial_dirty_manifest and manifest_entry_key in dirty_manifest_entry_keys and not git_signature)
        prefetched_text = None
        prefetched_lines = None
        prefetched_text_hash = None
        prefetched_cache_key = None
        prefetched_bundle_checked = False
        if dirty_partial_entry:
            state_signature = _worktree_content_signature(path)
        elif defer_cold_git_signatures:
            state_signature = ""
        else:
            state_signature = _file_state_signature(path, git_signature)
        if cache_dir is not None:
            previous = previous_manifest.get(manifest_entry_key) or {}
            if state_signature and previous.get("config_key") == config_key and previous.get("signature") == state_signature:
                cache_key = str(previous.get("cache_key") or "")
                if previous.get("empty"):
                    stale_chunks = _drop_file_state(manifest_entry_key, lexical_manifest, index_manifest, chunks_by_file, semantic_document_signatures)
                    _drop_chunk_file_mapping(stale_chunks, file_key_by_chunk_id)
                    next_manifest[manifest_entry_key] = {
                        "signature": state_signature,
                        "cache_key": "",
                        "config_key": config_key,
                        "empty": True,
                    }
                    diagnostics["reused_files"] += 1
                    continue
                if manifest_entry_key in chunks_by_file:
                    next_manifest[manifest_entry_key] = {
                        "signature": state_signature,
                        "cache_key": cache_key,
                        "config_key": config_key,
                        "empty": False,
                    }
                    diagnostics["reused_files"] += 1
                    file_order.append(manifest_entry_key)
                    continue
                if cache_key:
                    stage_started = time.perf_counter()
                    cached_bundle = _load_file_chunk_cache_bundle(Path(cache_dir), cache_key)
                    build_substage["file_chunk_bundle_load_ms"] += round((time.perf_counter() - stage_started) * 1000.0, 3)
                    if cached_bundle is not None:
                        cached_chunks = cached_bundle["chunks"]
                        stale_chunks = _drop_file_state(manifest_entry_key, lexical_manifest, index_manifest, chunks_by_file, semantic_document_signatures)
                        _drop_chunk_file_mapping(stale_chunks, file_key_by_chunk_id)
                        lexical_record = cached_bundle["lexical_record"]
                        index_signature = cached_bundle["index_signature"]
                        _install_reused_file_chunks(
                            chunk_rows=cached_chunks,
                            lexical_record=lexical_record,
                            index_signature=index_signature,
                            lexical_manifest=lexical_manifest,
                            index_manifest=index_manifest,
                            chunks_by_file=chunks_by_file,
                            semantic_document_signatures=semantic_document_signatures,
                            file_key_by_chunk_id=file_key_by_chunk_id,
                            semantic_signatures=cached_bundle["semantic_document_signatures"],
                        )
                        next_manifest[manifest_entry_key] = {
                            "signature": state_signature,
                            "cache_key": cache_key,
                            "config_key": config_key,
                            "empty": False,
                        }
                        diagnostics["reused_files"] += 1
                        file_order.append(manifest_entry_key)
                        continue
            if dirty_partial_entry:
                try:
                    stage_started = time.perf_counter()
                    prefetched_text = path.read_text(encoding="utf-8", errors="replace")
                    build_substage["source_read_ms"] += round((time.perf_counter() - stage_started) * 1000.0, 3)
                except OSError:
                    continue
                prefetched_lines = prefetched_text.splitlines()
                prefetched_text_hash = sha256_text(prefetched_text)
                prefetched_cache_key = _file_chunk_cache_key(
                    source_namespace=source_namespace,
                    rel_path=rel_path,
                    language=language,
                    text_hash=prefetched_text_hash,
                    input_id=str(item.get("id", "")),
                    input_type=str(item.get("type", "repo")),
                    classification=str(item.get("classification", "unknown")),
                    max_lines=max_lines,
                    overlap=overlap,
                )
                stage_started = time.perf_counter()
                prefetched_bundle_checked = True
                cached_bundle = _load_file_chunk_cache_bundle(Path(cache_dir), prefetched_cache_key)
                build_substage["file_chunk_bundle_load_ms"] += round((time.perf_counter() - stage_started) * 1000.0, 3)
                if cached_bundle is not None:
                    cached_chunks = cached_bundle["chunks"]
                    stale_chunks = _drop_file_state(manifest_entry_key, lexical_manifest, index_manifest, chunks_by_file, semantic_document_signatures)
                    _drop_chunk_file_mapping(stale_chunks, file_key_by_chunk_id)
                    lexical_record = cached_bundle["lexical_record"]
                    index_signature = cached_bundle["index_signature"]
                    _install_reused_file_chunks(
                        chunk_rows=cached_chunks,
                        lexical_record=lexical_record,
                        index_signature=index_signature,
                        lexical_manifest=lexical_manifest,
                        index_manifest=index_manifest,
                        chunks_by_file=chunks_by_file,
                        semantic_document_signatures=semantic_document_signatures,
                        file_key_by_chunk_id=file_key_by_chunk_id,
                        semantic_signatures=cached_bundle["semantic_document_signatures"],
                    )
                    next_manifest[manifest_entry_key] = {
                        "signature": state_signature,
                        "cache_key": prefetched_cache_key,
                        "config_key": config_key,
                        "empty": False,
                    }
                    diagnostics["reused_files"] += 1
                    file_order.append(manifest_entry_key)
                    continue
                if not prefetched_lines:
                    next_manifest[manifest_entry_key] = {
                        "signature": state_signature,
                        "cache_key": "",
                        "config_key": config_key,
                        "empty": True,
                    }
                    if publish_shared_file_level_cache:
                        _update_shared_file_chunk_state_manifest_cached(
                            config_key,
                            state_signature,
                            cache_key="",
                            empty=True,
                            bundle={"chunks": []},
                            manifest_cache=shared_state_manifest_cache,
                        )
                    diagnostics["rebuilt_files"] += 1
                    continue
            stage_started = time.perf_counter()
            if state_signature:
                shared_lookup = _load_file_chunk_cache_by_state_signature(
                    Path(cache_dir),
                    config_key,
                    state_signature,
                    manifest_cache=shared_state_manifest_cache,
                )
                build_substage["shared_state_lookup_ms"] += round((time.perf_counter() - stage_started) * 1000.0, 3)
                if shared_lookup == ("", []):
                    stale_chunks = _drop_file_state(manifest_entry_key, lexical_manifest, index_manifest, chunks_by_file, semantic_document_signatures)
                    _drop_chunk_file_mapping(stale_chunks, file_key_by_chunk_id)
                    next_manifest[manifest_entry_key] = {
                        "signature": state_signature,
                        "cache_key": "",
                        "config_key": config_key,
                        "empty": True,
                    }
                    diagnostics["reused_files"] += 1
                    continue
                if shared_lookup is not None:
                    shared_cache_key, shared_chunks = shared_lookup
                    stale_chunks = _drop_file_state(manifest_entry_key, lexical_manifest, index_manifest, chunks_by_file, semantic_document_signatures)
                    _drop_chunk_file_mapping(stale_chunks, file_key_by_chunk_id)
                    lexical_record, index_signature = _manifest_records_for_file_chunks(shared_chunks)
                    _install_reused_file_chunks(
                        chunk_rows=shared_chunks,
                        lexical_record=lexical_record,
                        index_signature=index_signature,
                        lexical_manifest=lexical_manifest,
                        index_manifest=index_manifest,
                        chunks_by_file=chunks_by_file,
                        semantic_document_signatures=semantic_document_signatures,
                        file_key_by_chunk_id=file_key_by_chunk_id,
                    )
                    next_manifest[manifest_entry_key] = {
                        "signature": state_signature,
                        "cache_key": shared_cache_key,
                        "config_key": config_key,
                        "empty": False,
                    }
                    diagnostics["reused_files"] += 1
                    file_order.append(manifest_entry_key)
                    continue
        stale_chunks = _drop_file_state(manifest_entry_key, lexical_manifest, index_manifest, chunks_by_file, semantic_document_signatures)
        _drop_chunk_file_mapping(stale_chunks, file_key_by_chunk_id)
        if prefetched_text is None:
            try:
                stage_started = time.perf_counter()
                text = path.read_text(encoding="utf-8", errors="replace")
                build_substage["source_read_ms"] += round((time.perf_counter() - stage_started) * 1000.0, 3)
            except OSError:
                continue
            lines = text.splitlines()
        else:
            text = prefetched_text
            lines = list(prefetched_lines or [])
        if not lines:
            if not state_signature:
                state_signature = f"sha256:{sha256_text(text)}"
            next_manifest[manifest_entry_key] = {
                "signature": state_signature,
                "cache_key": "",
                "config_key": config_key,
                "empty": True,
            }
            if publish_shared_file_level_cache:
                _update_shared_file_chunk_state_manifest_cached(
                    config_key,
                    state_signature,
                    cache_key="",
                    empty=True,
                    bundle={"chunks": []},
                    manifest_cache=shared_state_manifest_cache,
                )
            diagnostics["rebuilt_files"] += 1
            continue
        text_hash = prefetched_text_hash if prefetched_text_hash is not None else sha256_text(text)
        if not state_signature:
            state_signature = f"sha256:{text_hash}"
        cache_key = None
        if cache_dir is not None:
            cache_key = prefetched_cache_key
            if cache_key is None:
                cache_key = _file_chunk_cache_key(
                    source_namespace=source_namespace,
                    rel_path=rel_path,
                    language=language,
                    text_hash=text_hash,
                    input_id=str(item.get("id", "")),
                    input_type=str(item.get("type", "repo")),
                    classification=str(item.get("classification", "unknown")),
                    max_lines=max_lines,
                    overlap=overlap,
                )
            cached_bundle = None
            if not skip_true_cold_local_bundle_probe and not (prefetched_bundle_checked and cache_key == prefetched_cache_key):
                stage_started = time.perf_counter()
                cached_bundle = _load_file_chunk_cache_bundle(Path(cache_dir), cache_key)
                build_substage["file_chunk_bundle_load_ms"] += round((time.perf_counter() - stage_started) * 1000.0, 3)
            if cached_bundle is not None:
                cached_chunks = cached_bundle["chunks"]
                lexical_record = cached_bundle["lexical_record"]
                index_signature = cached_bundle["index_signature"]
                _install_reused_file_chunks(
                    chunk_rows=cached_chunks,
                    lexical_record=lexical_record,
                    index_signature=index_signature,
                    lexical_manifest=lexical_manifest,
                    index_manifest=index_manifest,
                    chunks_by_file=chunks_by_file,
                    semantic_document_signatures=semantic_document_signatures,
                    file_key_by_chunk_id=file_key_by_chunk_id,
                    semantic_signatures=cached_bundle["semantic_document_signatures"],
                )
                next_manifest[manifest_entry_key] = {
                    "signature": state_signature,
                    "cache_key": cache_key,
                    "config_key": config_key,
                    "empty": False,
                }
                diagnostics["reused_files"] += 1
                file_order.append(manifest_entry_key)
                continue
            stage_started = time.perf_counter()
            shared_lookup = _load_file_chunk_cache_by_state_signature(
                Path(cache_dir),
                config_key,
                state_signature,
                manifest_cache=shared_state_manifest_cache,
            )
            build_substage["shared_state_lookup_ms"] += round((time.perf_counter() - stage_started) * 1000.0, 3)
            if shared_lookup is not None:
                if shared_lookup == ("", []):
                    next_manifest[manifest_entry_key] = {
                        "signature": state_signature,
                        "cache_key": "",
                        "config_key": config_key,
                        "empty": True,
                    }
                    diagnostics["reused_files"] += 1
                    continue
                shared_cache_key, shared_chunks = shared_lookup
                lexical_record, index_signature = _manifest_records_for_file_chunks(shared_chunks)
                _install_reused_file_chunks(
                    chunk_rows=shared_chunks,
                    lexical_record=lexical_record,
                    index_signature=index_signature,
                    lexical_manifest=lexical_manifest,
                    index_manifest=index_manifest,
                    chunks_by_file=chunks_by_file,
                    semantic_document_signatures=semantic_document_signatures,
                    file_key_by_chunk_id=file_key_by_chunk_id,
                )
                next_manifest[manifest_entry_key] = {
                    "signature": state_signature,
                    "cache_key": shared_cache_key,
                    "config_key": config_key,
                    "empty": False,
                }
                diagnostics["reused_files"] += 1
                file_order.append(manifest_entry_key)
                continue
        markers = None
        if cache_dir is not None and cache_key is not None:
            stage_started = time.perf_counter()
            markers = _load_symbol_markers(Path(cache_dir), language, text_hash)
            build_substage["symbol_marker_ms"] += round((time.perf_counter() - stage_started) * 1000.0, 3)
        if markers is None:
            stage_started = time.perf_counter()
            markers = symbol_markers(text, language)
            if cache_dir is not None and cache_key is not None:
                write_started = time.perf_counter()
                _write_symbol_markers(
                    Path(cache_dir),
                    language,
                    text_hash,
                    markers,
                    publish_shared=publish_shared_file_level_cache,
                    persist_local=publish_shared_file_level_cache,
                )
                build_substage["symbol_marker_write_ms"] += round((time.perf_counter() - write_started) * 1000.0, 3)
            build_substage["symbol_marker_ms"] += round((time.perf_counter() - stage_started) * 1000.0, 3)
        file_chunks = []
        stage_started = time.perf_counter()
        for line_start, line_end, symbol, content in _chunk_regions(lines, markers, max_lines, overlap):
            current_line = line_start
            for segment_index, segment in enumerate(_split_utf8_chunks(content)):
                newline_count = segment.count("\n")
                segment_end = min(line_end, current_line + newline_count)
                if segment.endswith("\n") and newline_count:
                    segment_end = max(current_line, segment_end - 1)
                hash_started = time.perf_counter()
                content_digest = sha256_text(segment)
                build_substage["chunk_content_hash_ms"] += round((time.perf_counter() - hash_started) * 1000.0, 3)
                identity_started = time.perf_counter()
                identity = sha256_text(
                    f"{source_namespace}\0{rel_path}\0{current_line}\0{segment_end}\0{segment_index}\0{content_digest}"
                )
                build_substage["chunk_identity_hash_ms"] += round((time.perf_counter() - identity_started) * 1000.0, 3)
                token_started = time.perf_counter()
                token_estimate = estimate_tokens(segment)
                build_substage["chunk_token_estimate_ms"] += round((time.perf_counter() - token_started) * 1000.0, 3)
                file_chunks.append(
                    {
                    "chunk_id": f"chunk_{identity[:20]}",
                    "path": display_path,
                    "repository_path": rel_path,
                    "source_namespace": source_namespace,
                    "language": language,
                    "symbol": symbol,
                    "line_start": current_line,
                    "line_end": segment_end,
                    "content": segment,
                    "content_hash": f"sha256:{content_digest}",
                    "chunk_hash": f"sha256:{content_digest}",
                    "token_estimate": token_estimate,
                    "input_id": item.get("id", ""),
                    "input_type": item.get("type", "repo"),
                    "classification": item.get("classification", "unknown"),
                    }
                )
                current_line += newline_count
        build_substage["chunk_build_ms"] += round((time.perf_counter() - stage_started) * 1000.0, 3)
        diagnostics["rebuilt_files"] += 1
        stage_started = time.perf_counter()
        lexical_record, index_signature = _manifest_records_for_file_chunks(file_chunks)
        build_substage["manifest_records_ms"] += round((time.perf_counter() - stage_started) * 1000.0, 3)
        if lexical_record is not None:
            lexical_manifest[lexical_record["file_key"]] = lexical_record
            index_manifest[lexical_record["file_key"]] = index_signature
            chunks_by_file[lexical_record["file_key"]] = [dict(chunk) for chunk in file_chunks]
            for chunk in file_chunks:
                file_key_by_chunk_id[str(chunk.get("chunk_id") or "")] = lexical_record["file_key"]
        stage_started = time.perf_counter()
        for chunk in file_chunks:
            semantic_document_signatures[str(chunk.get("chunk_id") or "")] = semantic_chunk_signature(chunk)
        build_substage["semantic_signature_ms"] += round((time.perf_counter() - stage_started) * 1000.0, 3)
        file_order.append(manifest_entry_key)
        if cache_dir is not None and cache_key is not None:
            write_started = time.perf_counter()
            _write_file_chunk_cache(
                Path(cache_dir),
                cache_key,
                file_chunks,
                publish_shared=publish_shared_file_level_cache,
                lexical_record=lexical_record,
                index_signature=index_signature,
                semantic_document_signatures={
                    str(chunk.get("chunk_id") or ""): semantic_document_signatures[str(chunk.get("chunk_id") or "")]
                    for chunk in file_chunks
                },
            )
            build_substage["file_chunk_bundle_write_ms"] += round((time.perf_counter() - write_started) * 1000.0, 3)
            if publish_shared_file_level_cache:
                _update_shared_file_chunk_state_manifest_cached(
                    config_key,
                    state_signature,
                    cache_key=cache_key,
                    empty=False,
                    bundle={
                        "chunks": file_chunks,
                        "lexical_record": lexical_record,
                        "index_signature": index_signature,
                        "semantic_document_signatures": {
                            str(chunk.get("chunk_id") or ""): semantic_document_signatures[str(chunk.get("chunk_id") or "")]
                            for chunk in file_chunks
                        },
                    },
                    manifest_cache=shared_state_manifest_cache,
                )
            next_manifest[manifest_entry_key] = {
                "signature": state_signature,
                "cache_key": cache_key,
                "config_key": config_key,
                "empty": False,
            }
    for stale_key in sorted(set(previous_manifest) - seen_manifest_keys):
        stale_chunks = _drop_file_state(stale_key, lexical_manifest, index_manifest, chunks_by_file, semantic_document_signatures)
        _drop_chunk_file_mapping(stale_chunks, file_key_by_chunk_id)
    manifest_unchanged = (
        str(previous_manifest_payload.get("repository_state_fingerprint") or "") == str(repository_state_fingerprint or "")
        and str(previous_manifest_payload.get("build_config_digest") or "") == str(build_config_digest or "")
        and previous_manifest == next_manifest
    )
    chunk_state_unchanged = (
        str(previous_manifest_payload.get("build_config_digest") or "") == str(build_config_digest or "")
        and previous_manifest == next_manifest
    )
    if manifest_path is not None:
        stage_started = time.perf_counter()
        if not manifest_unchanged:
            serialized_next_manifest_files_json = ""
            if chunk_state_unchanged and previous_manifest:
                serialized_next_manifest_files_json = str(
                    previous_manifest_payload.get("_serialized_files_json") or ""
                )
            _write_preferred_file_chunk_working_manifest(
                manifest_path,
                _shared_file_chunk_working_manifest_path(
                    str(repository_state_fingerprint or ""),
                    build_config_digest,
                    create=True,
                ),
                _shared_file_chunk_latest_manifest_path(build_config_digest, create=True),
                next_manifest,
                repository_state_fingerprint=str(repository_state_fingerprint or ""),
                build_config_digest=build_config_digest,
                publish_shared=publish_shared_cache_publication,
                serialized_files_json=serialized_next_manifest_files_json,
                diagnostics=build_substage,
            )
        build_substage["working_manifest_write_ms"] += round((time.perf_counter() - stage_started) * 1000.0, 3)
    chunks = ChunkList()
    for file_key in file_order:
        for chunk in chunks_by_file.get(file_key, ()):
            chunks.append(chunk)
    chunks._lexical_manifest = lexical_manifest
    chunks._index_manifest = index_manifest
    chunks._chunks_by_file = chunks_by_file
    chunks._semantic_document_signatures = semantic_document_signatures
    chunks._chunk_ids = tuple(str(chunk.get("chunk_id") or "") for chunk in chunks)
    chunks._file_key_by_chunk_id = file_key_by_chunk_id
    chunks._chunk_build_substage_timings = build_substage
    chunks._skip_shared_cache_publication = not publish_shared_cache_publication
    stage_started = time.perf_counter()
    if publish_shared_file_level_cache:
        _flush_shared_file_chunk_state_manifest_cache(shared_state_manifest_cache)
    build_substage["shared_state_manifest_flush_ms"] += round((time.perf_counter() - stage_started) * 1000.0, 3)
    if cache_dir is not None:
        stage_started = time.perf_counter()
        if not manifest_unchanged or diagnostics["rebuilt_files"] > 0:
            if (
                previous_manifest
                and all(
                    bool((record or {}).get("empty")) or bool(str((record or {}).get("cache_key") or ""))
                    for record in next_manifest.values()
                )
                and (
                    (
                        diagnostics["rebuilt_files"] == 0
                        and not chunk_state_unchanged
                    )
                    or (
                        diagnostics["rebuilt_files"] > 0
                        and not publish_shared_cache_publication
                    )
                )
            ):
                _write_file_chunk_snapshot_metadata_only(
                    Path(cache_dir),
                    repository_state_fingerprint=repository_state_fingerprint,
                    build_config_digest=build_config_digest,
                    chunks=chunks,
                    publish_shared=publish_shared_cache_publication,
                    diagnostics=build_substage,
                )
                _drop_file_chunk_snapshot_payload(_file_chunk_snapshot_path(Path(cache_dir)))
            elif (
                diagnostics["rebuilt_files"] == 0
                and chunk_state_unchanged
                and str(previous_manifest_payload.get("repository_state_fingerprint") or "")
                and str(previous_manifest_payload.get("repository_state_fingerprint") or "") != str(repository_state_fingerprint or "")
            ):
                _reuse_file_chunk_snapshot_payload(
                    Path(cache_dir),
                    previous_repository_state_fingerprint=str(previous_manifest_payload.get("repository_state_fingerprint") or ""),
                    repository_state_fingerprint=repository_state_fingerprint,
                    build_config_digest=build_config_digest,
                    chunks=chunks,
                    publish_shared=publish_shared_cache_publication,
                    diagnostics=build_substage,
                )
            else:
                _write_file_chunk_snapshot(
                    Path(cache_dir),
                    repository_state_fingerprint=repository_state_fingerprint,
                    build_config_digest=build_config_digest,
                    chunks=chunks,
                    publish_shared=publish_shared_cache_publication,
                    diagnostics=build_substage,
                )
        build_substage["snapshot_write_ms"] += round((time.perf_counter() - stage_started) * 1000.0, 3)
    if return_diagnostics:
        return chunks, diagnostics
    return chunks


def load_cached_chunk_snapshot_metadata(
    discovered,
    excluded_dir_names=None,
    *,
    max_lines=120,
    overlap=12,
    cache_dir=None,
    excluded_paths=None,
    repository_state_fingerprint=None,
    transient_excluded_paths=None,
):
    if cache_dir is None or not repository_state_fingerprint:
        return None
    discovered, _namespaces, _effective_excluded_paths, build_config_digest = _snapshot_build_context(
        discovered,
        excluded_dir_names,
        max_lines=max_lines,
        overlap=overlap,
        cache_dir=cache_dir,
        excluded_paths=excluded_paths,
        transient_excluded_paths=transient_excluded_paths,
    )
    cache_dir = Path(cache_dir)
    metadata = _load_file_chunk_snapshot_metadata(cache_dir)
    if metadata is None:
        shared_metadata_path = _shared_file_chunk_snapshot_metadata_path(
            str(repository_state_fingerprint or ""),
            str(build_config_digest or ""),
            create=False,
        )
        if shared_metadata_path is not None:
            try:
                payload = json.loads(shared_metadata_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                payload = None
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                with contextlib.suppress(OSError):
                    shared_metadata_path.unlink()
                return None
            if payload is not None:
                metadata = _normalize_file_chunk_snapshot_metadata_payload(payload)
    if metadata is None:
        return None
    if (
        str(metadata.get("repository_state_fingerprint") or "") != str(repository_state_fingerprint)
        or str(metadata.get("build_config_digest") or "") != str(build_config_digest)
    ):
        return None
    if int(metadata.get("total_files") or 0) <= 0:
        manifest_path = _file_chunk_manifest_path(cache_dir)
        previous_manifest_payload = _load_file_chunk_working_manifest(manifest_path) or {}
        metadata["total_files"] = max(int(metadata.get("total_files") or 0), len(previous_manifest_payload.get("files") or {}))
    return metadata


def _run_git(root: Path, *args, text=False):
    return subprocess.run(
        ["git", "-C", str(root), "--no-optional-locks", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=20,
        text=text,
    ).stdout


def _git_top(root: Path):
    cache_key = str(Path(root).resolve(strict=False))
    cached = _GIT_TOP_CACHE.get(cache_key, ...)
    if cached is not ...:
        _GIT_TOP_CACHE.pop(cache_key, None)
        _GIT_TOP_CACHE[cache_key] = cached
        return None if cached == "" else Path(cached)
    start = Path(root).resolve(strict=False)
    current = start if start.is_dir() else start.parent
    visited = set()
    resolved = None
    while True:
        current_key = str(current)
        if current_key in visited:
            break
        visited.add(current_key)
        git_marker = current / ".git"
        try:
            if git_marker.exists():
                resolved = current
                break
        except OSError:
            pass
        parent = current.parent
        if parent == current:
            break
        current = parent
    if resolved is None:
        try:
            resolved = Path(_run_git(root, "rev-parse", "--show-toplevel", text=True).strip()).resolve()
        except (OSError, subprocess.SubprocessError, ValueError):
            resolved = None
    _GIT_TOP_CACHE[cache_key] = "" if resolved is None else str(resolved)
    while len(_GIT_TOP_CACHE) > _GIT_TOP_CACHE_LIMIT:
        oldest_key = next(iter(_GIT_TOP_CACHE))
        _GIT_TOP_CACHE.pop(oldest_key, None)
    return resolved


def _run_git_optional(root: Path, *args, text=False):
    try:
        return _run_git(root, *args, text=text)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _git_file_signature_manifest_path(cache_dir: Path, top: Path, relative_paths):
    cache_root = _private_cache_dir(Path(cache_dir) / "git-file-signature-cache")
    payload = json.dumps(
        {
            "git_top": str(top.resolve(strict=False)),
            "paths": list(sorted(relative_paths)),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return cache_root / f"git-sigs-{sha256_text(payload)}.json"


def _shared_git_file_signature_manifest_path(top: Path, relative_paths, *, create=False):
    cache_root = _shared_repo_inspection_cache_root(create=create)
    if cache_root is None:
        return None
    cache_root = cache_root / "git-file-signature-cache"
    try:
        cache_root = _private_cache_dir(cache_root) if create else cache_root.expanduser().absolute()
    except OSError:
        return None
    payload = json.dumps(
        {
            "git_top": str(top.resolve(strict=False)),
            "paths": list(sorted(relative_paths)),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return cache_root / f"git-sigs-{sha256_text(payload)}.json"


def _clone_git_file_signature_manifest_record(record):
    return {
        "head": str(record.get("head") or ""),
        "status": str(record.get("status") or ""),
        "signatures": {str(rel): str(sig) for rel, sig in dict(record.get("signatures") or {}).items()},
    }


def _load_git_file_signature_manifest(path: Path):
    if path is None:
        return None
    cache_key = _snapshot_metadata_cache_key(path)
    if cache_key is not None:
        cached = _GIT_FILE_SIGNATURE_MANIFEST_CACHE.get(cache_key)
        if cached is not None:
            _GIT_FILE_SIGNATURE_MANIFEST_CACHE.pop(cache_key, None)
            _GIT_FILE_SIGNATURE_MANIFEST_CACHE[cache_key] = cached
            return _clone_git_file_signature_manifest_record(cached)
    elif not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        path.unlink(missing_ok=True)
        return None
    if not isinstance(payload, dict) or payload.get("schema") != GIT_FILE_SIGNATURE_MANIFEST_SCHEMA:
        return None
    signatures = payload.get("signatures")
    if not isinstance(signatures, dict):
        return None
    normalized = {
        "head": str(payload.get("head") or ""),
        "status": str(payload.get("status") or ""),
        "signatures": {str(rel): str(sig) for rel, sig in signatures.items()},
    }
    if cache_key is not None:
        _GIT_FILE_SIGNATURE_MANIFEST_CACHE.pop(cache_key, None)
        _GIT_FILE_SIGNATURE_MANIFEST_CACHE[cache_key] = _clone_git_file_signature_manifest_record(normalized)
        while len(_GIT_FILE_SIGNATURE_MANIFEST_CACHE) > _GIT_FILE_SIGNATURE_MANIFEST_CACHE_LIMIT:
            _GIT_FILE_SIGNATURE_MANIFEST_CACHE.pop(next(iter(_GIT_FILE_SIGNATURE_MANIFEST_CACHE)))
    return normalized


def _load_preferred_git_file_signature_manifest(local_path: Path | None, shared_path: Path | None):
    local = _load_git_file_signature_manifest(local_path)
    if local is not None:
        return local
    return _load_git_file_signature_manifest(shared_path)


def _write_git_file_signature_manifest(path: Path, *, head: str, status: str, signatures):
    _atomic_private_json(
        path,
        {
            "schema": GIT_FILE_SIGNATURE_MANIFEST_SCHEMA,
            "head": str(head),
            "status": str(status),
            "signatures": dict(signatures),
        },
    )
    cache_key = _snapshot_metadata_cache_key(path)
    if cache_key is not None:
        _GIT_FILE_SIGNATURE_MANIFEST_CACHE.pop(cache_key, None)
        _GIT_FILE_SIGNATURE_MANIFEST_CACHE[cache_key] = {
            "head": str(head or ""),
            "status": str(status or ""),
            "signatures": {str(rel): str(sig) for rel, sig in dict(signatures or {}).items()},
        }
        while len(_GIT_FILE_SIGNATURE_MANIFEST_CACHE) > _GIT_FILE_SIGNATURE_MANIFEST_CACHE_LIMIT:
            _GIT_FILE_SIGNATURE_MANIFEST_CACHE.pop(next(iter(_GIT_FILE_SIGNATURE_MANIFEST_CACHE)))


def _write_preferred_git_file_signature_manifest(
    local_path: Path | None,
    shared_path: Path | None,
    *,
    head: str,
    status: str,
    signatures,
):
    if shared_path is not None and local_path is not None and shared_path != local_path:
        _write_git_file_signature_manifest(shared_path, head=head, status=status, signatures=signatures)
        return
    if local_path is not None:
        _write_git_file_signature_manifest(local_path, head=head, status=status, signatures=signatures)


def _git_file_signatures(discovered_files, *, cache_dir=None, git_probe_cache=None):
    files_by_root = defaultdict(list)
    git_top_by_input_root = {}
    for _item, path, _rel_path in discovered_files:
        input_path = _item.get("path")
        if input_path is None:
            probe_root = path.parent
        else:
            input_path = Path(input_path)
            probe_root = input_path if input_path.is_dir() else input_path.parent
        probe_key = str(probe_root.resolve(strict=False))
        if probe_key in git_top_by_input_root:
            git_top = git_top_by_input_root[probe_key]
        else:
            git_top = _git_top(probe_root)
            git_top_by_input_root[probe_key] = git_top
        if git_top is not None:
            files_by_root[str(git_top)].append(path.resolve(strict=False))
    signatures = {}
    for root_text, paths in files_by_root.items():
        root = Path(root_text)
        relative_paths = []
        path_lookup = {}
        for path in paths:
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                continue
            relative_paths.append(rel)
            path_lookup[rel] = path
        if not relative_paths:
            continue
        relative_paths = sorted(relative_paths)
        repo_scope_status_snapshot = _cached_repo_scope_status_snapshot(root, git_probe_cache=git_probe_cache)
        repo_scope_status = (
            repo_scope_status_snapshot.get("output")
            if isinstance(repo_scope_status_snapshot, dict)
            else _cached_repo_scope_status_output(root, git_probe_cache=git_probe_cache)
        )
        if repo_scope_status is not None:
            status_output = repo_scope_status
            subset_cache_key = tuple(relative_paths)
            cached_subset = (
                repo_scope_status_snapshot.get("subset_dirty", {}).get(subset_cache_key)
                if isinstance(repo_scope_status_snapshot, dict)
                else None
            )
            if isinstance(cached_subset, tuple) and len(cached_subset) == 2:
                status_digest, dirty_list = cached_subset
                dirty = set(dirty_list)
            else:
                status_digest, dirty = _status_subset_digest_and_dirty(
                    status_output,
                    relative_paths,
                    parsed_entries=(
                        repo_scope_status_snapshot.get("parsed_entries")
                        if isinstance(repo_scope_status_snapshot, dict)
                        else None
                    ),
                )
                if isinstance(repo_scope_status_snapshot, dict):
                    repo_scope_status_snapshot.setdefault("subset_dirty", {})[subset_cache_key] = (
                        status_digest,
                        tuple(sorted(dirty)),
                    )
        else:
            status_output = _scoped_git_status_output(root, relative_paths, git_probe_cache=git_probe_cache)
            status_digest, dirty = _status_subset_digest_and_dirty(status_output, relative_paths) if status_output is not None else ("", set())
        tracked = {}
        cached_signature_map = {}
        tracked_signature_bucket = _git_probe_cache_bucket(git_probe_cache, "tracked_blob_signatures")
        tracked_signature_map = {}
        if tracked_signature_bucket is not None:
            tracked_signature_map = dict(tracked_signature_bucket.get(str(root.resolve(strict=False))) or {})
        direct_cached_clean_paths = [rel for rel in relative_paths if rel not in dirty and rel in tracked_signature_map]
        if len(direct_cached_clean_paths) == len(relative_paths) - len(dirty):
            for rel in direct_cached_clean_paths:
                path = path_lookup.get(rel)
                if path is not None:
                    signatures[str(path)] = str(tracked_signature_map[rel])
            continue
        head_bucket = _git_probe_cache_bucket(git_probe_cache, "repo_head")
        head_key = str(root.resolve(strict=False))
        if head_bucket is not None and head_key in head_bucket:
            head = head_bucket[head_key]
        else:
            head = _run_git_optional(root, "rev-parse", "HEAD^{tree}", text=True)
            if head_bucket is not None:
                head_bucket[head_key] = head
        head = head.strip() if isinstance(head, str) and head.strip() else "unborn"
        if status_output is not None:
            manifest_path = _git_file_signature_manifest_path(Path(cache_dir), root, relative_paths) if cache_dir is not None else None
            shared_manifest_path = (
                _shared_git_file_signature_manifest_path(root, relative_paths, create=False) if cache_dir is not None else None
            )
            if manifest_path is not None or shared_manifest_path is not None:
                cached = _load_preferred_git_file_signature_manifest(manifest_path, shared_manifest_path)
                if cached and cached.get("head") == head and cached.get("status") == status_digest:
                    for rel, signature in cached["signatures"].items():
                        path = path_lookup.get(rel)
                        if path is not None:
                            signatures[str(path)] = signature
                    continue
                if cached and cached.get("head") == head:
                    cached_signature_map = {
                        rel: str(signature)
                        for rel, signature in (cached.get("signatures") or {}).items()
                        if rel in path_lookup
                    }
        unresolved_clean = [
            rel
            for rel in relative_paths
            if rel not in dirty and rel not in cached_signature_map and rel not in tracked_signature_map
        ]
        for rel, signature in cached_signature_map.items():
            if rel not in dirty:
                tracked[rel] = signature
        for rel, signature in tracked_signature_map.items():
            if rel in path_lookup and rel not in dirty:
                tracked[rel] = str(signature)
        if unresolved_clean:
            tracked_output = _run_git_optional(root, "ls-files", "-s", "-z", "--", *unresolved_clean)
            if tracked_output is not None:
                for raw_entry in tracked_output.split(b"\0"):
                    if not raw_entry:
                        continue
                    try:
                        meta, rel = raw_entry.split(b"\t", 1)
                        _mode, blob_oid, _stage = meta.decode("utf-8", errors="replace").split()
                    except ValueError:
                        continue
                    tracked[rel.decode("utf-8", errors="surrogateescape")] = f"git:{blob_oid}"
        for rel, path in path_lookup.items():
            signature = tracked.get(rel)
            if signature is not None and rel not in dirty:
                signatures[str(path)] = signature
        if status_output is not None and cache_dir is not None:
            cached_signatures = {
                rel: signatures[str(path_lookup[rel])]
                for rel in relative_paths
                if rel in path_lookup and str(path_lookup[rel]) in signatures
            }
            _write_preferred_git_file_signature_manifest(
                _git_file_signature_manifest_path(Path(cache_dir), root, relative_paths),
                _shared_git_file_signature_manifest_path(root, relative_paths, create=True),
                head=head,
                status=status_digest,
                signatures=cached_signatures,
            )
    return signatures


def _git_dirty_manifest_entry_keys(discovered_files, namespaces, *, git_probe_cache=None):
    files_by_root = defaultdict(list)
    git_top_by_input_root = {}
    for item, path, rel_path in discovered_files:
        input_path = item.get("path")
        if input_path is None:
            probe_root = path.parent
        else:
            input_path = Path(input_path)
            probe_root = input_path if input_path.is_dir() else input_path.parent
        probe_key = str(probe_root.resolve(strict=False))
        if probe_key in git_top_by_input_root:
            git_top = git_top_by_input_root[probe_key]
        else:
            git_top = _git_top(probe_root)
            git_top_by_input_root[probe_key] = git_top
        if git_top is None:
            return None
        files_by_root[str(git_top)].append((item, path.resolve(strict=False), rel_path))

    dirty_entry_keys = set()
    for root_text, entries in files_by_root.items():
        root = Path(root_text)
        relative_paths = []
        manifest_keys_by_rel = defaultdict(list)
        for item, path, rel_path in entries:
            try:
                repo_rel = path.relative_to(root).as_posix()
            except ValueError:
                continue
            relative_paths.append(repo_rel)
            manifest_keys_by_rel[repo_rel].append(
                _file_chunk_manifest_entry_key(
                    source_namespace=namespaces.get(id(item), ""),
                    rel_path=rel_path,
                )
            )
        if not relative_paths:
            continue
        status_snapshot = _cached_repo_scope_status_snapshot(root, git_probe_cache=git_probe_cache)
        status_output = status_snapshot.get("output") if isinstance(status_snapshot, dict) else None
        if status_output is None:
            status_snapshot = _scoped_git_status_snapshot(root, [], git_probe_cache=git_probe_cache)
            status_output = status_snapshot.get("output") if isinstance(status_snapshot, dict) else None
        if status_output is None:
            return None
        subset_paths = tuple(sorted(set(relative_paths)))
        cached_subset_dirty = (
            status_snapshot.get("subset_dirty_only", {}).get(subset_paths)
            if isinstance(status_snapshot, dict)
            else None
        )
        if isinstance(cached_subset_dirty, tuple):
            dirty = set(cached_subset_dirty)
        else:
            cached_subset = (
                status_snapshot.get("subset_dirty", {}).get(subset_paths)
                if isinstance(status_snapshot, dict)
                else None
            )
            if isinstance(cached_subset, tuple) and len(cached_subset) == 2:
                dirty = set(cached_subset[1] or ())
            else:
                dirty = _status_subset_dirty(
                    subset_paths,
                    parsed_entries=status_snapshot.get("parsed_entries") if isinstance(status_snapshot, dict) else None,
                )
            if isinstance(status_snapshot, dict):
                status_snapshot.setdefault("subset_dirty_only", {})[subset_paths] = tuple(sorted(dirty))
        if not dirty:
            continue
        for rel in dirty:
            dirty_entry_keys.update(manifest_keys_by_rel.get(rel, ()))
    return dirty_entry_keys


def _file_state_signature(path: Path, git_signature=None):
    if git_signature:
        return str(git_signature)
    try:
        stat = path.stat()
    except OSError:
        return "unreadable"
    return f"meta:{int(stat.st_size)}:{int(stat.st_mtime_ns)}:{int(getattr(stat, 'st_ctime_ns', 0))}"


def _git_index_blob_oids(top: Path, relative_paths):
    if not relative_paths:
        return {}
    output = _run_git_optional(top, "ls-files", "-s", "-z", "--", *sorted(relative_paths))
    if output is None:
        return None
    tracked = {}
    for raw_entry in output.split(b"\0"):
        if not raw_entry:
            continue
        try:
            meta, rel = raw_entry.split(b"\t", 1)
            mode, blob_oid, stage = meta.decode("utf-8", errors="replace").split()
        except ValueError:
            continue
        tracked[rel.decode("utf-8", errors="surrogateescape")] = {
            "mode": mode,
            "blob_oid": blob_oid,
            "stage": stage,
        }
    return tracked


def _worktree_content_signature(path: Path):
    if path.is_symlink():
        return "symlink"
    if not path.exists():
        return "missing"
    if not path.is_file():
        return "nonfile"
    try:
        return f"sha256:{sha256_bytes(path.read_bytes())}"
    except OSError:
        return "unreadable"


def _git_blob_oid_for_bytes(content: bytes):
    header = f"blob {len(content)}\0".encode("utf-8")
    return hashlib.sha1(header + content).hexdigest()


def _worktree_blob_oid(path: Path):
    if path.is_symlink():
        return "symlink"
    if not path.exists():
        return "missing"
    if not path.is_file():
        return "nonfile"
    try:
        return _git_blob_oid_for_bytes(path.read_bytes())
    except OSError:
        return "unreadable"


def _normalize_worktree_signature_cache(cache):
    if not isinstance(cache, dict):
        return {}
    normalized = {}
    for path, record in cache.items():
        if not isinstance(record, dict):
            continue
        state_signature = str(record.get("state_signature") or "")
        content_signature = str(record.get("content_signature") or "")
        if not state_signature or not content_signature:
            continue
        normalized_record = {
            "state_signature": state_signature,
            "content_signature": content_signature,
        }
        git_blob_oid = str(record.get("git_blob_oid") or "")
        if git_blob_oid:
            normalized_record["git_blob_oid"] = git_blob_oid
        normalized[str(path)] = normalized_record
    return normalized


def _cached_worktree_signature_map(*states):
    for state in states:
        if not isinstance(state, dict):
            continue
        cache = _normalize_worktree_signature_cache(state.get("worktree_signatures"))
        if cache:
            return cache
    return {}


def _worktree_signatures_with_cache(path: Path, cached_record=None):
    if path.is_symlink():
        return "symlink", None, "symlink"
    if not path.exists():
        return "missing", None, "missing"
    if not path.is_file():
        return "nonfile", None, "nonfile"
    state_signature = _file_state_signature(path)
    if isinstance(cached_record, dict):
        cached_state_signature = str(cached_record.get("state_signature") or "")
        cached_content_signature = str(cached_record.get("content_signature") or "")
        cached_git_blob_oid = str(cached_record.get("git_blob_oid") or "")
        if (
            cached_state_signature
            and cached_content_signature
            and state_signature == cached_state_signature
        ):
            if cached_git_blob_oid:
                return cached_content_signature, state_signature, cached_git_blob_oid
            if str(cached_content_signature).startswith("sha256:"):
                try:
                    content = path.read_bytes()
                except OSError:
                    return cached_content_signature, state_signature, "unreadable"
                return cached_content_signature, state_signature, _git_blob_oid_for_bytes(content)
            return cached_content_signature, state_signature, cached_content_signature
    try:
        content = path.read_bytes()
    except OSError:
        return "unreadable", state_signature, "unreadable"
    return f"sha256:{sha256_bytes(content)}", state_signature, _git_blob_oid_for_bytes(content)


def _worktree_signature_with_cache(path: Path, cached_record=None):
    content_signature, state_signature, _git_blob_oid = _worktree_signatures_with_cache(path, cached_record)
    return content_signature, state_signature


def _git_fingerprint(root: Path, ignored: set[str], ignored_paths=None, scope_paths=None, cache_dir=None, git_probe_cache=None):
    top = _git_top(root)
    if top is None:
        return None
    normalized_scope_paths = _normalize_scope_paths(top, scope_paths)
    manifest_path = _git_fingerprint_manifest_path(Path(cache_dir), top, normalized_scope_paths) if cache_dir is not None else None
    shared_manifest_path = _shared_git_fingerprint_manifest_path(top, normalized_scope_paths, create=False)
    cached = _load_git_fingerprint_manifest(manifest_path) if manifest_path is not None else None
    head = ""
    root_scope = not normalized_scope_paths or all(item in {"", "."} for item in normalized_scope_paths)
    if root_scope and isinstance(cached, dict):
        cached_aux = cached.get("aux")
        current_head_signature = _git_head_state_signature(top)
        if (
            isinstance(cached_aux, dict)
            and current_head_signature
            and str(cached_aux.get("git_head_signature") or "") == current_head_signature
        ):
            head = str(cached.get("head") or "")
    if not head:
        head = _scoped_git_head_identity(top, normalized_scope_paths, git_probe_cache=git_probe_cache)
    process_cached = _load_git_fastpath_state_cache(git_probe_cache, top, normalized_scope_paths)
    cached_clean_state = _clean_git_fingerprint_manifest(cached, head=head)
    cached_clean_aux = _git_clean_fingerprint_aux(cached, head=head)
    cached_unstaged_only_aux = _git_unstaged_only_fingerprint_aux(cached, head=head)
    cached_staged_aux = _git_staged_fingerprint_aux(cached, head=head)
    process_staged_aux = _git_staged_fingerprint_aux(process_cached, head=head)
    shared_cached = None
    shared_clean_state = None
    shared_clean_aux = None
    shared_unstaged_only_aux = None
    shared_staged_aux = None
    shared_loaded = False

    def ensure_shared_cached():
        nonlocal shared_cached, shared_clean_state, shared_clean_aux, shared_unstaged_only_aux, shared_staged_aux, shared_loaded
        if shared_loaded:
            return
        shared_loaded = True
        shared_cached = _load_git_fingerprint_manifest(shared_manifest_path) if shared_manifest_path is not None else None
        shared_clean_state = _clean_git_fingerprint_manifest(shared_cached, head=head)
        shared_clean_aux = _git_clean_fingerprint_aux(shared_cached, head=head)
        shared_unstaged_only_aux = _git_unstaged_only_fingerprint_aux(shared_cached, head=head)
        shared_staged_aux = _git_staged_fingerprint_aux(shared_cached, head=head)

    def cache_fastpath_state(state, aux=None):
        if not isinstance(state, dict):
            return state
        cached_state = dict(state)
        cached_state.pop("_staged_entries_detail", None)
        _store_git_fastpath_state_cache(
            git_probe_cache,
            top,
            normalized_scope_paths,
            head=head,
            state=cached_state,
            aux=aux,
        )
        return cached_state

    status_output = None
    status_snapshot = None
    if cached_clean_state is not None:
        if cached_clean_state is not None and _git_clean_fastpath_match_state(
            top,
            normalized_scope_paths,
            ignored=ignored,
            ignored_paths=ignored_paths,
            aux=cached_clean_aux,
        ):
            return dict(cached_clean_state)
        ensure_shared_cached()
        if shared_clean_state is not None and _git_clean_fastpath_match_state(
            top,
            normalized_scope_paths,
            ignored=ignored,
            ignored_paths=ignored_paths,
            aux=shared_clean_aux,
        ):
            return dict(shared_clean_state)
        if cached_clean_aux is not None:
            dirty_fastpath_state = _git_unstaged_only_fastpath_state(
                top,
                normalized_scope_paths,
                head=head,
                ignored=ignored,
                ignored_paths=ignored_paths,
                aux=cached_clean_aux,
                cached_state_for_reuse=(cached or {}).get("state"),
            )
            if dirty_fastpath_state is not None:
                return cache_fastpath_state(dirty_fastpath_state, aux=cached_clean_aux)
        ensure_shared_cached()
        if shared_clean_aux is not None:
            dirty_fastpath_state = _git_unstaged_only_fastpath_state(
                top,
                normalized_scope_paths,
                head=head,
                ignored=ignored,
                ignored_paths=ignored_paths,
                aux=shared_clean_aux,
                cached_state_for_reuse=(shared_cached or {}).get("state"),
            )
            if dirty_fastpath_state is not None:
                return cache_fastpath_state(dirty_fastpath_state, aux=shared_clean_aux)
        if process_staged_aux is not None:
            staged_fastpath_state = _git_staged_cached_fastpath_state(
                top,
                normalized_scope_paths,
                head=head,
                ignored=ignored,
                ignored_paths=ignored_paths,
                aux=process_staged_aux,
                cached_state_for_reuse=(process_cached or {}).get("state"),
            )
            if staged_fastpath_state is not None:
                return cache_fastpath_state(staged_fastpath_state, aux=process_staged_aux)
        if cached_clean_aux is not None:
            staged_fastpath_state = _git_staged_small_repo_fastpath_state(
                top,
                normalized_scope_paths,
                head=head,
                ignored=ignored,
                ignored_paths=ignored_paths,
                aux=cached_clean_aux,
                cached_state_for_reuse=(cached or {}).get("state"),
            )
            if staged_fastpath_state is not None:
                return cache_fastpath_state(
                    staged_fastpath_state,
                    aux={
                        "clean_worktree_files": dict(cached_clean_aux.get("clean_worktree_files") or {}),
                        "clean_tracked_content": dict(cached_clean_aux.get("clean_tracked_content") or {}),
                        "git_index_signature": str(_git_index_state_signature(top) or ""),
                        "staged_entries": list(staged_fastpath_state.get("_staged_entries_detail") or []),
                    },
                )
        ensure_shared_cached()
        if shared_clean_aux is not None:
            staged_fastpath_state = _git_staged_small_repo_fastpath_state(
                top,
                normalized_scope_paths,
                head=head,
                ignored=ignored,
                ignored_paths=ignored_paths,
                aux=shared_clean_aux,
                cached_state_for_reuse=(shared_cached or {}).get("state"),
            )
            if staged_fastpath_state is not None:
                return cache_fastpath_state(
                    staged_fastpath_state,
                    aux={
                        "clean_worktree_files": dict(shared_clean_aux.get("clean_worktree_files") or {}),
                        "clean_tracked_content": dict(shared_clean_aux.get("clean_tracked_content") or {}),
                        "git_index_signature": str(_git_index_state_signature(top) or ""),
                        "staged_entries": list(staged_fastpath_state.get("_staged_entries_detail") or []),
                    },
                )
        clean_probe = _scoped_git_clean_probe(
            top,
            normalized_scope_paths,
            ignored=ignored,
            ignored_paths=ignored_paths,
            git_probe_cache=git_probe_cache,
        )
        if clean_probe is True:
            if cached_clean_state is not None:
                return dict(cached_clean_state)
            ensure_shared_cached()
            return dict(shared_clean_state)
        if clean_probe is None:
            return None
        status_snapshot = _scoped_git_status_snapshot(top, normalized_scope_paths, git_probe_cache=git_probe_cache)
        if status_snapshot is None:
            return None
        status_output = status_snapshot["output"]
    else:
        ensure_shared_cached()
    if cached_clean_state is None and shared_clean_state is not None:
        if _git_clean_fastpath_match_state(
            top,
            normalized_scope_paths,
            ignored=ignored,
            ignored_paths=ignored_paths,
            aux=shared_clean_aux,
        ):
            return dict(shared_clean_state)
        if shared_clean_aux is not None:
            dirty_fastpath_state = _git_unstaged_only_fastpath_state(
                top,
                normalized_scope_paths,
                head=head,
                ignored=ignored,
                ignored_paths=ignored_paths,
                aux=shared_clean_aux,
                cached_state_for_reuse=(shared_cached or {}).get("state"),
            )
            if dirty_fastpath_state is not None:
                return cache_fastpath_state(dirty_fastpath_state, aux=shared_clean_aux)
        if process_staged_aux is not None:
            staged_fastpath_state = _git_staged_cached_fastpath_state(
                top,
                normalized_scope_paths,
                head=head,
                ignored=ignored,
                ignored_paths=ignored_paths,
                aux=process_staged_aux,
                cached_state_for_reuse=(process_cached or {}).get("state"),
            )
            if staged_fastpath_state is not None:
                return cache_fastpath_state(staged_fastpath_state, aux=process_staged_aux)
        if shared_clean_aux is not None:
            staged_fastpath_state = _git_staged_small_repo_fastpath_state(
                top,
                normalized_scope_paths,
                head=head,
                ignored=ignored,
                ignored_paths=ignored_paths,
                aux=shared_clean_aux,
                cached_state_for_reuse=(shared_cached or {}).get("state"),
            )
            if staged_fastpath_state is not None:
                return cache_fastpath_state(
                    staged_fastpath_state,
                    aux={
                        "clean_worktree_files": dict(shared_clean_aux.get("clean_worktree_files") or {}),
                        "clean_tracked_content": dict(shared_clean_aux.get("clean_tracked_content") or {}),
                        "git_index_signature": str(_git_index_state_signature(top) or ""),
                        "staged_entries": list(staged_fastpath_state.get("_staged_entries_detail") or []),
                    },
                )
        clean_probe = _scoped_git_clean_probe(
            top,
            normalized_scope_paths,
            ignored=ignored,
            ignored_paths=ignored_paths,
            git_probe_cache=git_probe_cache,
        )
        if clean_probe is True:
            return dict(shared_clean_state)
        if clean_probe is None:
            return None
        status_snapshot = _scoped_git_status_snapshot(top, normalized_scope_paths, git_probe_cache=git_probe_cache)
        if status_snapshot is None:
            return None
        status_output = status_snapshot["output"]
    elif (
        process_staged_aux is not None
        or cached_staged_aux is not None
        or shared_staged_aux is not None
        or cached_unstaged_only_aux is not None
        or shared_unstaged_only_aux is not None
    ):
        if process_staged_aux is not None:
            staged_fastpath_state = _git_staged_cached_fastpath_state(
                top,
                normalized_scope_paths,
                head=head,
                ignored=ignored,
                ignored_paths=ignored_paths,
                aux=process_staged_aux,
                cached_state_for_reuse=(process_cached or {}).get("state"),
            )
            if staged_fastpath_state is not None:
                return cache_fastpath_state(staged_fastpath_state, aux=process_staged_aux)
        if cached_staged_aux is not None:
            staged_fastpath_state = _git_staged_cached_fastpath_state(
                top,
                normalized_scope_paths,
                head=head,
                ignored=ignored,
                ignored_paths=ignored_paths,
                aux=cached_staged_aux,
                cached_state_for_reuse=(cached or {}).get("state"),
            )
            if staged_fastpath_state is not None:
                return cache_fastpath_state(staged_fastpath_state, aux=cached_staged_aux)
        if shared_staged_aux is not None:
            staged_fastpath_state = _git_staged_cached_fastpath_state(
                top,
                normalized_scope_paths,
                head=head,
                ignored=ignored,
                ignored_paths=ignored_paths,
                aux=shared_staged_aux,
                cached_state_for_reuse=(shared_cached or {}).get("state"),
            )
            if staged_fastpath_state is not None:
                return cache_fastpath_state(staged_fastpath_state, aux=shared_staged_aux)
        if cached_unstaged_only_aux is not None:
            dirty_fastpath_state = _git_unstaged_only_fastpath_state(
                top,
                normalized_scope_paths,
                head=head,
                ignored=ignored,
                ignored_paths=ignored_paths,
                aux=cached_unstaged_only_aux,
                cached_state_for_reuse=(cached or {}).get("state"),
            )
            if dirty_fastpath_state is not None:
                return cache_fastpath_state(dirty_fastpath_state, aux=cached_unstaged_only_aux)
        if shared_unstaged_only_aux is not None:
            dirty_fastpath_state = _git_unstaged_only_fastpath_state(
                top,
                normalized_scope_paths,
                head=head,
                ignored=ignored,
                ignored_paths=ignored_paths,
                aux=shared_unstaged_only_aux,
                cached_state_for_reuse=(shared_cached or {}).get("state"),
            )
            if dirty_fastpath_state is not None:
                return cache_fastpath_state(dirty_fastpath_state, aux=shared_unstaged_only_aux)
    pathspec = _scope_pathspec(normalized_scope_paths)
    if status_output is None:
        status_snapshot = _scoped_git_status_snapshot(top, normalized_scope_paths, git_probe_cache=git_probe_cache)
    if status_snapshot is None:
        return None
    status_output = status_snapshot["output"]
    if not status_output or (isinstance(status_output, (bytes, str)) and not status_output.strip()):
        empty_status_digest = _empty_git_status_digest()
        if cached and cached.get("head") == head and cached.get("status") == empty_status_digest:
            return dict(cached["state"])
        if shared_cached and shared_cached.get("head") == head and shared_cached.get("status") == empty_status_digest:
            return dict(shared_cached["state"])
    filter_key = _ignored_filter_cache_key(top, ignored, ignored_paths=ignored_paths)
    filtered_details_cache = status_snapshot.setdefault("filtered_details", {})
    cached_filtered_details = filtered_details_cache.get(filter_key)
    if isinstance(cached_filtered_details, tuple) and len(cached_filtered_details) == 4:
        status_digest, filtered_status, relevant_entries, tracked_index_paths = cached_filtered_details
    else:
        status_digest, filtered_status, relevant_entries, tracked_index_paths = _filtered_git_status_details(
            status_output,
            top,
            ignored,
            ignored_paths=ignored_paths,
            parsed_entries=status_snapshot.get("parsed_entries"),
        )
        filtered_details_cache[filter_key] = (
            status_digest,
            filtered_status,
            relevant_entries,
            tracked_index_paths,
        )
    if not filtered_status:
        if cached and cached.get("head") == head:
            return dict(cached["state"])
        if shared_cached and shared_cached.get("head") == head:
            return dict(shared_cached["state"])
    cached_state_for_reuse = None
    if cached and cached.get("head") == head and cached.get("status") == status_digest:
        cached_state_for_reuse = cached.get("state")
    elif shared_cached and shared_cached.get("head") == head and shared_cached.get("status") == status_digest:
        cached_state_for_reuse = shared_cached.get("state")
    worktree_signature_cache = _cached_worktree_signature_map(cached_state_for_reuse)
    tracked_index = {}
    if tracked_index_paths:
        tracked_index = _git_index_blob_oids(top, tracked_index_paths)
        if tracked_index is None:
            return None

    staged_entries = []
    unstaged_entries = []
    untracked = []
    dirty_paths = set()
    next_worktree_signature_cache = {}
    for entry in relevant_entries:
        code = entry["code"]
        x_code = code[:1]
        y_code = code[1:2]
        paths = list(entry["paths"])
        destination = paths[-1] if paths else ""
        source = paths[0] if len(paths) > 1 else ""
        if x_code == "?":
            candidate = top / destination
            if candidate.is_symlink() or _should_skip(candidate, top, ignored, ignored_paths=ignored_paths) or not candidate.is_file():
                continue
            content_signature, state_signature, git_blob_oid = _worktree_signatures_with_cache(
                candidate,
                worktree_signature_cache.get(destination),
            )
            if state_signature:
                cached_signature_record = {
                    "state_signature": state_signature,
                    "content_signature": content_signature,
                }
                if git_blob_oid and git_blob_oid not in {"missing", "nonfile", "symlink", "unreadable"}:
                    cached_signature_record["git_blob_oid"] = git_blob_oid
                next_worktree_signature_cache[destination] = cached_signature_record
            untracked.append((destination, content_signature))
            if destination:
                dirty_paths.add(destination)
            continue
        if x_code not in {"", " "}:
            staged_record = {
                "code": x_code,
                "path": destination,
            }
            if source and source != destination:
                staged_record["source_path"] = source
            index_record = tracked_index.get(destination)
            if index_record is not None:
                staged_record["mode"] = str(index_record.get("mode") or "")
                staged_record["blob_oid"] = str(index_record.get("blob_oid") or "")
                staged_record["stage"] = str(index_record.get("stage") or "")
            staged_entries.append(staged_record)
            if destination:
                dirty_paths.add(destination)
            if source and source != destination:
                dirty_paths.add(source)
        if y_code not in {"", " "}:
            candidate = top / destination
            content_signature, state_signature, git_blob_oid = _worktree_signatures_with_cache(
                candidate,
                worktree_signature_cache.get(destination),
            )
            if state_signature:
                cached_signature_record = {
                    "state_signature": state_signature,
                    "content_signature": content_signature,
                }
                if git_blob_oid and git_blob_oid not in {"missing", "nonfile", "symlink", "unreadable"}:
                    cached_signature_record["git_blob_oid"] = git_blob_oid
                next_worktree_signature_cache[destination] = cached_signature_record
            unstaged_entries.append(
                {
                    "code": y_code,
                    "path": destination,
                    "signature": content_signature,
                }
            )
            if destination:
                dirty_paths.add(destination)
    state = {
        "kind": "git",
        "head": head,
        "staged": f"sha256:{sha256_text(json.dumps(sorted(staged_entries, key=lambda item: (item.get('path', ''), item.get('source_path', ''), item.get('code', ''))), sort_keys=True, separators=(',', ':'), ensure_ascii=True))}",
        "unstaged": f"sha256:{sha256_text(json.dumps(sorted(unstaged_entries, key=lambda item: (item.get('path', ''), item.get('code', ''))), sort_keys=True, separators=(',', ':'), ensure_ascii=True))}",
        "untracked": sorted(untracked),
    }
    if dirty_paths:
        state["dirty_paths"] = sorted(dirty_paths)
    if pathspec:
        state["scope_paths"] = pathspec[1:]
    if next_worktree_signature_cache:
        state["worktree_signatures"] = next_worktree_signature_cache
    state["fingerprint_digest"] = sha256_text(
        json.dumps(
            {
                "kind": "git",
                "head": state["head"],
                "staged": state["staged"],
                "unstaged": state["unstaged"],
                "untracked": state["untracked"],
                **({"scope_paths": state["scope_paths"]} if "scope_paths" in state else {}),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    aux = {}
    if _git_state_is_clean(state):
        clean_capture = _git_clean_fastpath_capture_content_signatures(
            top,
            normalized_scope_paths,
            ignored=ignored,
            ignored_paths=ignored_paths,
            cached_worktree_signatures=state.get("worktree_signatures"),
        )
        clean_files = clean_capture[0] if clean_capture is not None else None
        clean_content = clean_capture[1] if clean_capture is not None else None
        clean_directory_signatures = None
        if clean_files:
            state["clean_worktree_files"] = sorted(str(rel) for rel in clean_files)
            clean_directory_signatures = _directory_signatures_for_files(top, clean_files.keys())
        aux = _git_clean_fastpath_aux_payload(clean_files, clean_directory_signatures)
        if clean_content:
            aux["clean_tracked_content"] = dict(clean_content)
        index_signature = _git_index_state_signature(top)
        if index_signature:
            aux["git_index_signature"] = index_signature
        head_signature = _git_head_state_signature(top)
        if head_signature:
            aux["git_head_signature"] = head_signature
    elif _git_state_is_unstaged_only(state):
        baseline_aux = cached_clean_aux or shared_clean_aux or {}
        baseline_files = _normalize_git_clean_fastpath_files((baseline_aux or {}).get("clean_worktree_files"))
        baseline_content = _normalize_git_clean_fastpath_files((baseline_aux or {}).get("clean_tracked_content"))
        index_signature = str((baseline_aux or {}).get("git_index_signature") or _git_index_state_signature(top) or "")
        if baseline_files and baseline_content and set(baseline_files) == set(baseline_content) and index_signature:
            aux = {
                "clean_worktree_files": baseline_files,
                "clean_tracked_content": baseline_content,
                "git_index_signature": index_signature,
            }
    if manifest_path is not None and not _git_fingerprint_manifest_equals(cached, head=head, status=status_digest, state=state, aux=aux):
        _write_git_fingerprint_manifest(manifest_path, head=head, status=status_digest, state=state, aux=aux)
    shared_manifest_write_path = _shared_git_fingerprint_manifest_path(top, normalized_scope_paths, create=True)
    if shared_manifest_write_path is not None and not _git_fingerprint_manifest_equals(
        shared_cached, head=head, status=status_digest, state=state, aux=aux
    ):
        _write_git_fingerprint_manifest(shared_manifest_write_path, head=head, status=status_digest, state=state, aux=aux)
    return cache_fastpath_state(state, aux=aux)


def _metadata_manifest_path(cache_dir: Path, root: Path):
    cache_root = _private_cache_dir(Path(cache_dir) / "metadata-fingerprint-cache")
    digest = sha256_text(str(root.resolve(strict=False)))
    return cache_root / f"metadata-{digest}.json"


def _shared_metadata_manifest_path(root: Path, *, create=False):
    cache_root = _shared_repo_inspection_cache_root(create=create)
    if cache_root is None:
        return None
    cache_root = cache_root / "metadata-fingerprint-cache"
    try:
        cache_root = _private_cache_dir(cache_root) if create else cache_root.expanduser().absolute()
    except OSError:
        return None
    digest = sha256_text(str(root.resolve(strict=False)))
    return cache_root / f"metadata-{digest}.json"


def _load_metadata_fingerprint_manifest(path: Path):
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        path.unlink(missing_ok=True)
        return None
    if not isinstance(payload, dict) or payload.get("schema") != METADATA_FINGERPRINT_MANIFEST_SCHEMA:
        return None
    files = payload.get("files")
    if not isinstance(files, dict):
        return None
    normalized = {}
    for rel, record in files.items():
        if not isinstance(record, dict):
            return None
        normalized[str(rel)] = {
            "signature": str(record.get("signature") or ""),
            "content_hash": str(record.get("content_hash") or ""),
        }
    return normalized


def _load_preferred_metadata_fingerprint_manifest(local_path: Path | None, shared_path: Path | None):
    local = _load_metadata_fingerprint_manifest(local_path)
    if local is not None:
        return local
    return _load_metadata_fingerprint_manifest(shared_path)


def _write_metadata_fingerprint_manifest(path: Path, files):
    _atomic_private_json(
        path,
        {
            "schema": METADATA_FINGERPRINT_MANIFEST_SCHEMA,
            "files": files,
        },
    )


def _write_preferred_metadata_fingerprint_manifest(local_path: Path | None, shared_path: Path | None, files):
    if shared_path is not None and local_path is not None and shared_path != local_path:
        _write_metadata_fingerprint_manifest(shared_path, files)
        return
    if local_path is not None:
        _write_metadata_fingerprint_manifest(local_path, files)


def _metadata_fingerprint_manifest_equals(previous_manifest, next_manifest) -> bool:
    return dict(previous_manifest or {}) == dict(next_manifest or {})


def _metadata_fingerprint(path: Path, ignored: set[str], ignored_paths=None, cache_dir=None, git_probe_cache=None):
    root = path if path.is_dir() else path.parent
    if path.is_file():
        candidates = [path]
    else:
        cached_candidates = _cached_metadata_source_candidates(
            root,
            ignored,
            ignored_paths=ignored_paths,
            git_probe_cache=git_probe_cache,
        )
        if cached_candidates is not None:
            candidates = cached_candidates
        else:
            candidates = sorted(_iter_source_candidates(root, ignored, ignored_paths=ignored_paths))
            _store_metadata_source_candidates(
                root,
                ignored,
                candidates,
                ignored_paths=ignored_paths,
                git_probe_cache=git_probe_cache,
            )
    manifest_path = _metadata_manifest_path(Path(cache_dir), root) if cache_dir is not None else None
    shared_manifest_path = _shared_metadata_manifest_path(root, create=False) if cache_dir is not None else None
    previous_manifest = (
        _load_preferred_metadata_fingerprint_manifest(manifest_path, shared_manifest_path)
        if manifest_path is not None or shared_manifest_path is not None
        else None
    )
    previous_manifest = previous_manifest or {}
    next_manifest = {}
    entries = []
    for candidate in candidates:
        if candidate.is_symlink() or not candidate.is_file() or _should_skip(candidate, root, ignored, ignored_paths=ignored_paths):
            continue
        if not language_for_path(candidate):
            continue
        try:
            stat = candidate.stat()
        except OSError:
            continue
        rel = candidate.relative_to(root).as_posix() if candidate != path or path.is_dir() else candidate.name
        signature = _file_state_signature(candidate)
        cached = previous_manifest.get(rel) or {}
        content_hash = str(cached.get("content_hash") or "")
        if cached.get("signature") != signature or not content_hash:
            try:
                content_hash = sha256_bytes(candidate.read_bytes())
            except OSError:
                continue
        next_manifest[rel] = {
            "signature": signature,
            "content_hash": content_hash,
        }
        entries.append((rel, stat.st_size, stat.st_mtime_ns, content_hash))
    if (
        (manifest_path is not None or shared_manifest_path is not None)
        and not _metadata_fingerprint_manifest_equals(previous_manifest, next_manifest)
    ):
        _write_preferred_metadata_fingerprint_manifest(
            manifest_path,
            _shared_metadata_manifest_path(root, create=True),
            next_manifest,
        )
    return {"kind": "metadata", "entries": entries}


def repository_fingerprint(discovered, excluded_dir_names=None, excluded_paths=None, cache_dir=None, git_probe_cache=None):
    """Fingerprint HEAD plus every staged, unstaged, and untracked content state."""

    git_probe_cache = _effective_git_probe_cache(git_probe_cache)
    _invalidate_git_probe_worktree_caches(git_probe_cache)
    ignored = set(DEFAULT_IGNORE_DIRS)
    ignored.update(excluded_dir_names or ())
    ignored_paths = {Path(path).resolve(strict=False) for path in (excluded_paths or ())}
    if cache_dir is not None:
        ignored_paths.add(Path(cache_dir).resolve(strict=False))
    states = []
    git_scopes = defaultdict(list)
    for item in discovered:
        path = item.get("path")
        if path is None:
            continue
        path = Path(path)
        git_top = _git_top(path if path.is_dir() else path.parent)
        if git_top is not None:
            git_scopes[str(git_top)].append(path)
        else:
            states.append(
                _metadata_fingerprint(
                    path,
                    ignored,
                    ignored_paths=ignored_paths,
                    cache_dir=cache_dir,
                    git_probe_cache=git_probe_cache,
                )
            )
    for git_root, scope_paths in sorted(git_scopes.items()):
        git_state = _git_fingerprint(
            Path(git_root),
            ignored,
            ignored_paths=ignored_paths,
            scope_paths=scope_paths,
            cache_dir=cache_dir,
            git_probe_cache=git_probe_cache,
        )
        if git_state is not None:
            states.append(git_state)
    return _repository_state_fingerprint_from_states(states), states


def inspection_index_fingerprint(repository_state_fingerprint, chunks, schema_version=INDEX_SCHEMA_VERSION):
    file_manifest = getattr(chunks, "_index_manifest", None)
    if isinstance(file_manifest, dict):
        encoded = json.dumps(
            {
                "schema_version": str(schema_version),
                "repository_state_fingerprint": str(repository_state_fingerprint or ""),
                "files": [
                    {"file_key": str(file_key), "signature": str(file_manifest[file_key])}
                    for file_key in sorted(file_manifest)
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return f"sha256:{sha256_text(encoded)}"
    manifest = [
        {
            "chunk_id": chunk["chunk_id"],
            "path": chunk["path"],
            "language": chunk.get("language", ""),
            "symbol": chunk.get("symbol", ""),
            "line_start": int(chunk["line_start"]),
            "line_end": int(chunk["line_end"]),
            "content_hash": chunk["content_hash"],
            "input_id": chunk.get("input_id", ""),
            "source_namespace": chunk.get("source_namespace", ""),
            "classification": chunk.get("classification", "unknown"),
        }
        for chunk in sorted(chunks, key=lambda item: (item["path"], item["line_start"], item["chunk_id"]))
    ]
    encoded = json.dumps(
        {
            "schema_version": str(schema_version),
            "repository_state_fingerprint": str(repository_state_fingerprint or ""),
            "chunks": manifest,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return f"sha256:{sha256_text(encoded)}"


def _working_index_path(cache_dir: Path):
    return cache_dir / "lexical-working.sqlite3"


def _working_manifest_path(cache_dir: Path):
    return cache_dir / "lexical-working-manifest.json"


def _shared_lexical_index_path(fingerprint: str, *, create=False):
    cache_root = _shared_repo_inspection_cache_root(create=create)
    if cache_root is None:
        return None
    target_root = cache_root / "lexical-working-cache"
    try:
        target_root = _private_cache_dir(target_root) if create else target_root.expanduser().absolute()
    except OSError:
        return None
    safe = str(fingerprint or "").replace(":", "_")
    return target_root / f"lexical-{safe}.sqlite3"


def _shared_lexical_manifest_path(fingerprint: str, *, create=False):
    cache_root = _shared_repo_inspection_cache_root(create=create)
    if cache_root is None:
        return None
    target_root = cache_root / "lexical-working-cache"
    try:
        target_root = _private_cache_dir(target_root) if create else target_root.expanduser().absolute()
    except OSError:
        return None
    safe = str(fingerprint or "").replace(":", "_")
    return target_root / f"lexical-{safe}-manifest.json"


def _shared_latest_lexical_index_path(build_config_digest: str, *, create=False):
    cache_root = _shared_repo_inspection_cache_root(create=create)
    if cache_root is None:
        return None
    target_root = cache_root / "lexical-working-cache"
    try:
        target_root = _private_cache_dir(target_root) if create else target_root.expanduser().absolute()
    except OSError:
        return None
    safe = str(build_config_digest or "").replace(":", "_")
    return target_root / f"lexical-latest-{safe}.sqlite3"


def _shared_latest_lexical_manifest_path(build_config_digest: str, *, create=False):
    cache_root = _shared_repo_inspection_cache_root(create=create)
    if cache_root is None:
        return None
    target_root = cache_root / "lexical-working-cache"
    try:
        target_root = _private_cache_dir(target_root) if create else target_root.expanduser().absolute()
    except OSError:
        return None
    safe = str(build_config_digest or "").replace(":", "_")
    return target_root / f"lexical-latest-{safe}-manifest.json"


def _lexical_index_path_is_current(index_path: Path, fingerprint, chunk_count):
    if index_path.is_symlink() or not index_path.exists():
        return False
    try:
        with sqlite3.connect(index_path) as conn:
            row = conn.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
            count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
            return bool(_valid_lexical_schema(conn) and row and row[0] == fingerprint and count == int(chunk_count))
    except sqlite3.Error:
        return False


def _lexical_index_path_has_valid_schema(index_path: Path):
    if index_path.is_symlink() or not index_path.exists():
        return False
    try:
        with sqlite3.connect(index_path) as conn:
            return bool(_valid_lexical_schema(conn))
    except sqlite3.Error:
        return False


def _copy_shared_lexical_index(cache_dir: Path, shared_index: Path, shared_manifest: Path):
    working_path = _working_index_path(cache_dir)
    working_manifest = _working_manifest_path(cache_dir)
    tmp_index = working_path.with_suffix(working_path.suffix + f".tmp-{os.getpid()}")
    tmp_manifest = working_manifest.with_suffix(working_manifest.suffix + f".tmp-{os.getpid()}")
    tmp_index.unlink(missing_ok=True)
    tmp_manifest.unlink(missing_ok=True)
    try:
        try:
            os.link(shared_index, tmp_index)
        except OSError:
            shutil.copy2(shared_index, tmp_index)
        try:
            os.link(shared_manifest, tmp_manifest)
        except OSError:
            shutil.copy2(shared_manifest, tmp_manifest)
        os.chmod(tmp_index, 0o600)
        os.chmod(tmp_manifest, 0o600)
        os.replace(tmp_index, working_path)
        os.replace(tmp_manifest, working_manifest)
        return True
    except OSError:
        tmp_index.unlink(missing_ok=True)
        tmp_manifest.unlink(missing_ok=True)
        return False


def _shared_lexical_manifest_matches(shared_manifest: Path | None, fingerprint, chunk_count):
    if shared_manifest is None or not shared_manifest.exists():
        return False
    manifest = _load_lexical_working_manifest(shared_manifest)
    meta = _lexical_manifest_meta(manifest)
    return (
        str(meta.get("fingerprint") or "") == str(fingerprint or "")
        and int(meta.get("chunk_count") or 0) == int(chunk_count or 0)
    )


def _restore_shared_lexical_index(cache_dir: Path, fingerprint, chunk_count, *, build_config_digest=""):
    cache_dir = _private_cache_dir(cache_dir)
    shared_index = _shared_lexical_index_path(fingerprint, create=False)
    shared_manifest = _shared_lexical_manifest_path(fingerprint, create=False)
    if (
        shared_index is not None
        and shared_manifest is not None
        and shared_manifest.exists()
        and _shared_lexical_manifest_matches(shared_manifest, fingerprint, chunk_count)
        and _copy_shared_lexical_index(cache_dir, shared_index, shared_manifest)
    ):
        return "exact_fingerprint"
    if not str(build_config_digest or ""):
        return ""
    shared_latest_index = _shared_latest_lexical_index_path(build_config_digest, create=False)
    shared_latest_manifest = _shared_latest_lexical_manifest_path(build_config_digest, create=False)
    if (
        shared_latest_index is None
        or shared_latest_manifest is None
        or not shared_latest_manifest.exists()
        or not _lexical_index_path_has_valid_schema(shared_latest_index)
    ):
        return ""
    latest_manifest = _load_lexical_working_manifest(shared_latest_manifest)
    latest_meta = _lexical_manifest_meta(latest_manifest)
    if str(latest_meta.get("build_config_digest") or "") != str(build_config_digest or ""):
        return ""
    if _copy_shared_lexical_index(cache_dir, shared_latest_index, shared_latest_manifest):
        return "latest_build_config"
    return ""


def _publish_shared_lexical_index(cache_dir: Path, fingerprint, *, build_config_digest="", publish_shared=True):
    if not publish_shared:
        return
    working_path = _working_index_path(cache_dir)
    working_manifest = _working_manifest_path(cache_dir)
    working_manifest_payload = _load_lexical_working_manifest(working_manifest)
    working_meta = _lexical_manifest_meta(working_manifest_payload)
    chunk_count = int(working_meta.get("chunk_count") or 0)
    shared_index = _shared_lexical_index_path(fingerprint, create=True)
    shared_manifest = _shared_lexical_manifest_path(fingerprint, create=True)
    if shared_index is None or shared_manifest is None or not working_path.exists() or not working_manifest.exists():
        return
    exact_ready = (
        _shared_lexical_manifest_matches(shared_manifest, fingerprint, chunk_count)
        and _lexical_index_path_is_current(shared_index, fingerprint, chunk_count)
    )
    if not exact_ready:
        try:
            shutil.copy2(working_path, shared_index)
            shutil.copy2(working_manifest, shared_manifest)
            os.chmod(shared_index, 0o600)
            os.chmod(shared_manifest, 0o600)
            exact_ready = True
        except OSError:
            exact_ready = False
    if not str(build_config_digest or ""):
        return
    shared_latest_index = _shared_latest_lexical_index_path(build_config_digest, create=True)
    shared_latest_manifest = _shared_latest_lexical_manifest_path(build_config_digest, create=True)
    if shared_latest_index is None or shared_latest_manifest is None:
        return
    latest_ready = (
        _shared_lexical_manifest_matches(shared_latest_manifest, fingerprint, chunk_count)
        and _lexical_index_path_is_current(shared_latest_index, fingerprint, chunk_count)
    )
    if latest_ready:
        return
    source_index = shared_index if exact_ready and shared_index.exists() else working_path
    source_manifest = shared_manifest if exact_ready and shared_manifest.exists() else working_manifest
    tmp_latest_index = shared_latest_index.with_suffix(shared_latest_index.suffix + f".tmp-{os.getpid()}")
    tmp_latest_manifest = shared_latest_manifest.with_suffix(shared_latest_manifest.suffix + f".tmp-{os.getpid()}")
    tmp_latest_index.unlink(missing_ok=True)
    tmp_latest_manifest.unlink(missing_ok=True)
    try:
        try:
            os.link(source_index, tmp_latest_index)
        except OSError:
            shutil.copy2(source_index, tmp_latest_index)
        try:
            os.link(source_manifest, tmp_latest_manifest)
        except OSError:
            shutil.copy2(source_manifest, tmp_latest_manifest)
        os.chmod(tmp_latest_index, 0o600)
        os.chmod(tmp_latest_manifest, 0o600)
        os.replace(tmp_latest_index, shared_latest_index)
        os.replace(tmp_latest_manifest, shared_latest_manifest)
    except OSError:
        tmp_latest_index.unlink(missing_ok=True)
        tmp_latest_manifest.unlink(missing_ok=True)
        return


def _restore_matching_shared_latest_lexical_index(cache_dir: Path, desired_manifest, *, build_config_digest=""):
    if not str(build_config_digest or ""):
        return ""
    shared_latest_index = _shared_latest_lexical_index_path(build_config_digest, create=False)
    shared_latest_manifest = _shared_latest_lexical_manifest_path(build_config_digest, create=False)
    if (
        shared_latest_index is None
        or shared_latest_manifest is None
        or not shared_latest_manifest.exists()
        or not _lexical_index_path_has_valid_schema(shared_latest_index)
    ):
        return ""
    latest_manifest = _load_lexical_working_manifest(shared_latest_manifest)
    latest_meta = _lexical_manifest_meta(latest_manifest)
    latest_files = _lexical_manifest_files(latest_manifest)
    if str(latest_meta.get("build_config_digest") or "") != str(build_config_digest or ""):
        return ""
    if latest_files != dict(desired_manifest or {}):
        return ""
    if _copy_shared_lexical_index(cache_dir, shared_latest_index, shared_latest_manifest):
        return "latest_build_config_manifest_match"
    return ""


def _private_cache_dir(path):
    path = Path(path).expanduser().absolute()
    cache_key = str(path)
    if cache_key in _PRIVATE_CACHE_DIR_READY and path.exists() and not path.is_symlink():
        return path
    for component in reversed((path, *path.parents)):
        if component.is_symlink():
            raise OSError(f"inspection cache path contains a symlink: {component}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise OSError(f"inspection cache path is a symlink: {path}")
    path.chmod(0o700)
    _PRIVATE_CACHE_DIR_READY.add(cache_key)
    return path


def _existing_private_cache_dir(path):
    path = Path(path).expanduser().absolute()
    for component in reversed((path, *path.parents)):
        if component.exists() and component.is_symlink():
            raise OSError(f"inspection cache path contains a symlink: {component}")
    if not path.exists():
        return path
    if path.is_symlink():
        raise OSError(f"inspection cache path is a symlink: {path}")
    return path


def _atomic_private_json(path, payload):
    path = Path(path)
    _private_cache_dir(path.parent)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.unlink(missing_ok=True)
    descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    path.chmod(0o600)
    return path


def _atomic_private_bytes(path, payload_bytes):
    path = Path(path)
    _private_cache_dir(path.parent)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.unlink(missing_ok=True)
    descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload_bytes)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    path.chmod(0o600)
    _cache_small_path_payload(path, payload_bytes)
    return path


def _cache_small_path_payload(path: Path, payload: bytes, *, stat=None):
    if path is None or not isinstance(payload, (bytes, bytearray)):
        return
    if len(payload) > _PATH_SMALL_PAYLOAD_CACHE_MAX_BYTES:
        return
    try:
        stat = stat or path.stat()
    except OSError:
        return
    cache_key = str(path)
    _PATH_SMALL_PAYLOAD_MEMORY_CACHE.pop(cache_key, None)
    _PATH_SMALL_PAYLOAD_MEMORY_CACHE[cache_key] = {
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
        "payload": bytes(payload),
    }
    while len(_PATH_SMALL_PAYLOAD_MEMORY_CACHE) > _PATH_SMALL_PAYLOAD_MEMORY_CACHE_LIMIT:
        _PATH_SMALL_PAYLOAD_MEMORY_CACHE.pop(next(iter(_PATH_SMALL_PAYLOAD_MEMORY_CACHE)))


def _load_small_path_payload(path: Path, stat) -> bytes | None:
    entry = _PATH_SMALL_PAYLOAD_MEMORY_CACHE.get(str(path))
    if not isinstance(entry, dict):
        return None
    if int(entry.get("mtime_ns") or -1) != int(stat.st_mtime_ns) or int(entry.get("size") or -1) != int(stat.st_size):
        return None
    payload = entry.get("payload")
    if not isinstance(payload, (bytes, bytearray)):
        return None
    _PATH_SMALL_PAYLOAD_MEMORY_CACHE.pop(str(path), None)
    _PATH_SMALL_PAYLOAD_MEMORY_CACHE[str(path)] = entry
    return bytes(payload)


def _path_bytes_equal(path: Path | None, payload: bytes):
    if path is None or not path.exists():
        return False
    try:
        stat = path.stat()
        if int(stat.st_size) != len(payload):
            return False
        cached = _load_small_path_payload(path, stat)
        if cached is not None:
            return cached == payload
        existing = path.read_bytes()
        _cache_small_path_payload(path, existing, stat=stat)
        return existing == payload
    except OSError:
        return False


def _clone_private_cache_file(source_path: Path | None, target_path: Path | None):
    if source_path is None or target_path is None or source_path == target_path or not source_path.exists():
        return False
    tmp = target_path.with_suffix(target_path.suffix + f".tmp-copy-{os.getpid()}")
    try:
        _private_cache_dir(target_path.parent)
        tmp.unlink(missing_ok=True)
        try:
            os.link(source_path, tmp)
        except OSError:
            shutil.copy2(source_path, tmp)
        os.chmod(tmp, 0o600)
        os.replace(tmp, target_path)
        target_path.chmod(0o600)
        return True
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        return False


def _promote_shared_cache_file(shared_path: Path | None, local_path: Path | None):
    if shared_path is None or local_path is None or shared_path == local_path or not shared_path.exists():
        return
    try:
        _private_cache_dir(local_path.parent)
        tmp = local_path.with_suffix(local_path.suffix + f".tmp-copy-{os.getpid()}")
        tmp.unlink(missing_ok=True)
        try:
            os.link(shared_path, tmp)
        except OSError:
            shutil.copy2(shared_path, tmp)
        os.chmod(tmp, 0o600)
        os.replace(tmp, local_path)
        local_path.chmod(0o600)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        return


def _lexical_chunk_file_key(chunk):
    return f"{chunk.get('source_namespace', '')}\0{chunk.get('repository_path', chunk['path'])}"


def _lexical_manifest_for_chunks(chunks):
    cached = getattr(chunks, "_lexical_manifest", None)
    if isinstance(cached, dict):
        return {
            str(file_key): {
                "file_key": str(record.get("file_key") or file_key),
                "path": str(record.get("path") or ""),
                "language": str(record.get("language") or ""),
                "repository_path": str(record.get("repository_path") or ""),
                "source_namespace": str(record.get("source_namespace") or ""),
                "signature": str(record.get("signature") or ""),
                "chunks": [
                    {
                        "chunk_id": str(chunk.get("chunk_id") or ""),
                        "symbol": str(chunk.get("symbol") or ""),
                        "line_start": int(chunk.get("line_start") or 0),
                        "line_end": int(chunk.get("line_end") or 0),
                        "content_hash": str(chunk.get("content_hash") or ""),
                        "token_estimate": int(chunk.get("token_estimate") or 0),
                    }
                    for chunk in (record.get("chunks") or ())
                    if isinstance(chunk, dict)
                ],
            }
            for file_key, record in cached.items()
            if isinstance(record, dict)
        }
    files = {}
    for chunk in chunks:
        file_key = _lexical_chunk_file_key(chunk)
        record = files.setdefault(
            file_key,
            {
                "file_key": file_key,
                "path": chunk["path"],
                "language": chunk.get("language", ""),
                "repository_path": chunk.get("repository_path", chunk["path"]),
                "source_namespace": chunk.get("source_namespace", ""),
                "chunks": [],
            },
        )
        record["chunks"].append(
            {
                "chunk_id": chunk["chunk_id"],
                "symbol": chunk.get("symbol", ""),
                "line_start": int(chunk["line_start"]),
                "line_end": int(chunk["line_end"]),
                "content_hash": chunk["content_hash"],
                "token_estimate": int(chunk["token_estimate"]),
            }
        )
    for record in files.values():
        record["chunks"].sort(key=lambda item: (item["line_start"], item["chunk_id"]))
        signature_payload = {
            "repository_path": record["repository_path"],
            "source_namespace": record["source_namespace"],
            "chunks": record["chunks"],
        }
        record["signature"] = f"sha256:{sha256_text(json.dumps(signature_payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True))}"
    return files


def _cached_lexical_manifest_shallow_view(cached_manifest):
    if not isinstance(cached_manifest, dict):
        return None
    return {
        str(file_key): record
        for file_key, record in cached_manifest.items()
        if isinstance(record, dict)
    }


def _load_lexical_working_manifest(path: Path):
    cache_key = _snapshot_metadata_cache_key(path)
    if cache_key is not None:
        cached = _LEXICAL_WORKING_MANIFEST_CACHE.get(cache_key)
        if cached is not None:
            _LEXICAL_WORKING_MANIFEST_CACHE.pop(cache_key, None)
            _LEXICAL_WORKING_MANIFEST_CACHE[cache_key] = cached
            return {
                "__meta__": dict(cached.get("__meta__") or {}),
                **{
                    str(file_key): {
                        "file_key": str(record.get("file_key") or file_key),
                        "path": str(record.get("path") or ""),
                        "language": str(record.get("language") or ""),
                        "repository_path": str(record.get("repository_path") or ""),
                        "source_namespace": str(record.get("source_namespace") or ""),
                        "signature": str(record.get("signature") or ""),
                        "chunks": [
                            {
                                "chunk_id": str(chunk.get("chunk_id") or ""),
                                "symbol": str(chunk.get("symbol") or ""),
                                "line_start": int(chunk.get("line_start") or 0),
                                "line_end": int(chunk.get("line_end") or 0),
                                "content_hash": str(chunk.get("content_hash") or ""),
                                "token_estimate": int(chunk.get("token_estimate") or 0),
                            }
                            for chunk in (record.get("chunks") or ())
                            if isinstance(chunk, dict)
                        ],
                    }
                    for file_key, record in cached.items()
                    if str(file_key) != "__meta__" and isinstance(record, dict)
                },
            }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        path.unlink(missing_ok=True)
        return None
    if not isinstance(payload, dict) or payload.get("schema") != LEXICAL_WORKING_MANIFEST_SCHEMA:
        return None
    files = payload.get("files")
    if not isinstance(files, dict):
        return None
    normalized = {
        "__meta__": {
            "fingerprint": str(payload.get("fingerprint") or ""),
            "chunk_count": int(payload.get("chunk_count") or 0),
            "build_config_digest": str(payload.get("build_config_digest") or ""),
        }
    }
    for file_key, record in files.items():
        if not isinstance(record, dict):
            return None
        chunks = record.get("chunks")
        if not isinstance(chunks, list):
            return None
        normalized[file_key] = {
            "file_key": str(record.get("file_key") or file_key),
            "path": str(record.get("path") or ""),
            "language": str(record.get("language") or ""),
            "repository_path": str(record.get("repository_path") or ""),
            "source_namespace": str(record.get("source_namespace") or ""),
            "signature": str(record.get("signature") or ""),
            "chunks": [
                {
                    "chunk_id": str(chunk.get("chunk_id") or ""),
                    "symbol": str(chunk.get("symbol") or ""),
                    "line_start": int(chunk.get("line_start") or 0),
                    "line_end": int(chunk.get("line_end") or 0),
                    "content_hash": str(chunk.get("content_hash") or ""),
                    "token_estimate": int(chunk.get("token_estimate") or 0),
                }
                for chunk in chunks
                if isinstance(chunk, dict)
            ],
        }
    if cache_key is not None:
        _LEXICAL_WORKING_MANIFEST_CACHE.pop(cache_key, None)
        _LEXICAL_WORKING_MANIFEST_CACHE[cache_key] = {
            "__meta__": dict(normalized.get("__meta__") or {}),
            **{
                str(file_key): {
                    "file_key": str(record.get("file_key") or file_key),
                    "path": str(record.get("path") or ""),
                    "language": str(record.get("language") or ""),
                    "repository_path": str(record.get("repository_path") or ""),
                    "source_namespace": str(record.get("source_namespace") or ""),
                    "signature": str(record.get("signature") or ""),
                    "chunks": [dict(chunk) for chunk in (record.get("chunks") or ()) if isinstance(chunk, dict)],
                }
                for file_key, record in normalized.items()
                if str(file_key) != "__meta__" and isinstance(record, dict)
            },
        }
        while len(_LEXICAL_WORKING_MANIFEST_CACHE) > _LEXICAL_WORKING_MANIFEST_CACHE_LIMIT:
            _LEXICAL_WORKING_MANIFEST_CACHE.pop(next(iter(_LEXICAL_WORKING_MANIFEST_CACHE)))
    return normalized


def _lexical_manifest_meta(manifest):
    if not isinstance(manifest, dict):
        return {}
    meta = manifest.get("__meta__")
    return dict(meta) if isinstance(meta, dict) else {}


def _lexical_manifest_files(manifest):
    if not isinstance(manifest, dict):
        return {}
    return {str(key): value for key, value in manifest.items() if str(key) != "__meta__" and isinstance(value, dict)}


def _write_lexical_working_manifest(path: Path, files, *, fingerprint="", chunk_count=0, build_config_digest=""):
    _atomic_private_json(
        path,
        {
            "schema": LEXICAL_WORKING_MANIFEST_SCHEMA,
            "fingerprint": str(fingerprint or ""),
            "chunk_count": int(chunk_count or 0),
            "build_config_digest": str(build_config_digest or ""),
            "files": _lexical_manifest_files(files),
        },
    )


def _initialize_lexical_schema(conn):
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, file_key TEXT NOT NULL, path TEXT NOT NULL, "
        "repository_path TEXT NOT NULL, source_namespace TEXT NOT NULL, language TEXT, symbol TEXT, "
        "line_start INTEGER, line_end INTEGER, content_hash TEXT, token_estimate INTEGER, content TEXT NOT NULL)"
    )
    conn.execute("CREATE INDEX chunks_file_key_idx ON chunks(file_key)")
    conn.execute(
        "CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, path, symbol, content, "
        "tokenize='unicode61 tokenchars ''_./:-''')"
    )
    conn.execute("INSERT INTO metadata(key,value) VALUES('schema',?)", (LEXICAL_INDEX_SCHEMA,))


def _configure_lexical_connection(conn):
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-32768")


def _configure_lexical_rebuild_connection(conn):
    _configure_lexical_connection(conn)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")


def _configure_lexical_write_connection(conn):
    _configure_lexical_connection(conn)
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA synchronous=OFF")


def _valid_lexical_schema(conn):
    try:
        row = conn.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()
    except sqlite3.Error:
        return False
    return bool(row and row[0] == LEXICAL_INDEX_SCHEMA)


def _insert_lexical_chunks(conn, chunks):
    row_values = []
    fts_values = []
    for chunk in chunks:
        row_values.append(
            (
                chunk["chunk_id"],
                _lexical_chunk_file_key(chunk),
                chunk["path"],
                chunk.get("repository_path", chunk["path"]),
                chunk.get("source_namespace", ""),
                chunk.get("language", ""),
                chunk.get("symbol", ""),
                int(chunk["line_start"]),
                int(chunk["line_end"]),
                chunk["content_hash"],
                int(chunk["token_estimate"]),
                chunk["content"],
            )
        )
        fts_values.append(
            (
                chunk["chunk_id"],
                chunk["path"],
                chunk.get("symbol", ""),
                chunk["content"],
            )
        )
    if row_values:
        conn.executemany("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", row_values)
        conn.executemany(
            "INSERT INTO chunks_fts(chunk_id,path,symbol,content) VALUES(?,?,?,?)",
            fts_values,
        )


def _delete_lexical_fts_chunk_ids(conn, chunk_ids, *, batch_size=128):
    values = [str(chunk_id) for chunk_id in (chunk_ids or ()) if str(chunk_id)]
    if not values:
        return
    for start in range(0, len(values), max(1, int(batch_size))):
        batch = values[start : start + max(1, int(batch_size))]
        placeholders = ",".join("?" for _ in batch)
        conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", tuple(batch))


def _rebuild_working_lexical_index(path: Path, chunks, fingerprint):
    tmp = path.with_suffix(f".tmp-{os.getpid()}")
    tmp.unlink(missing_ok=True)
    try:
        descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        with sqlite3.connect(tmp) as conn:
            _configure_lexical_rebuild_connection(conn)
            _initialize_lexical_schema(conn)
            conn.execute("INSERT INTO metadata(key,value) VALUES('fingerprint',?)", (fingerprint,))
            _insert_lexical_chunks(conn, chunks)
        os.replace(tmp, path)
        path.chmod(0o600)
    finally:
        tmp.unlink(missing_ok=True)


def load_chunks_from_lexical_index(index_path, expected_fingerprint, *, include_content=False):
    rows = []
    columns = (
        "chunk_id, path, repository_path, source_namespace, language, symbol, "
        "line_start, line_end, content_hash, token_estimate"
    )
    if include_content:
        columns += ", content"
    try:
        with sqlite3.connect(index_path) as conn:
            if not _valid_lexical_schema(conn):
                return None
            row = conn.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
            if not row or str(row[0]) != str(expected_fingerprint):
                manifest = _load_lexical_working_manifest(_working_manifest_path(Path(index_path).parent)) or {}
                meta = _lexical_manifest_meta(manifest)
                if str(meta.get("fingerprint") or "") != str(expected_fingerprint or ""):
                    return None
                count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
                if int(meta.get("chunk_count") or 0) != int(count or 0):
                    return None
            rows = conn.execute(f"SELECT {columns} FROM chunks ORDER BY rowid").fetchall()
    except sqlite3.Error:
        return None
    chunks = ChunkList()
    for row in rows:
        (
            chunk_id,
            path,
            repository_path,
            source_namespace,
            language,
            symbol,
            line_start,
            line_end,
            content_hash,
            token_estimate,
            *content,
        ) = row
        chunk = {
            "chunk_id": str(chunk_id),
            "path": str(path),
            "repository_path": str(repository_path or path),
            "source_namespace": str(source_namespace or ""),
            "language": str(language or ""),
            "symbol": str(symbol or ""),
            "line_start": int(line_start or 0),
            "line_end": int(line_end or 0),
            "content_hash": str(content_hash or ""),
            "chunk_hash": str(content_hash or ""),
            "token_estimate": int(token_estimate or 0),
        }
        if include_content:
            chunk["content"] = str(content[0] or "")
        chunks.append(chunk)
    chunks._chunk_ids = tuple(str(chunk.get("chunk_id") or "") for chunk in chunks)
    chunks._chunk_count = len(chunks)
    return chunks


def load_chunks_from_lexical_manifest(index_path, expected_fingerprint):
    manifest = _load_lexical_working_manifest(_working_manifest_path(Path(index_path).parent)) or {}
    meta = _lexical_manifest_meta(manifest)
    if str(meta.get("fingerprint") or "") != str(expected_fingerprint or ""):
        return None
    files = _lexical_manifest_files(manifest)
    chunks = ChunkList()
    for file_key in sorted(files):
        record = files.get(file_key) or {}
        path = str(record.get("path") or "")
        repository_path = str(record.get("repository_path") or path)
        source_namespace = str(record.get("source_namespace") or "")
        language = str(record.get("language") or "") or language_for_path(Path(repository_path or path))
        for chunk in (record.get("chunks") or ()):
            if not isinstance(chunk, dict):
                continue
            chunk_id = str(chunk.get("chunk_id") or "")
            if not chunk_id:
                continue
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "path": path,
                    "repository_path": repository_path,
                    "source_namespace": source_namespace,
                    "language": language,
                    "symbol": str(chunk.get("symbol") or ""),
                    "line_start": int(chunk.get("line_start") or 0),
                    "line_end": int(chunk.get("line_end") or 0),
                    "content_hash": str(chunk.get("content_hash") or ""),
                    "chunk_hash": str(chunk.get("content_hash") or ""),
                    "token_estimate": int(chunk.get("token_estimate") or 0),
                }
            )
    if int(meta.get("chunk_count") or 0) != len(chunks):
        return None
    chunks._chunk_ids = tuple(str(chunk.get("chunk_id") or "") for chunk in chunks)
    chunks._chunk_count = len(chunks)
    chunks._lexical_manifest = files
    return chunks


def lexical_working_index_is_current(cache_dir, fingerprint, chunk_count):
    cache_dir = _existing_private_cache_dir(cache_dir)
    working_path = _working_index_path(cache_dir)
    working_manifest = _load_lexical_working_manifest(_working_manifest_path(cache_dir)) or {}
    working_meta = _lexical_manifest_meta(working_manifest)
    if (
        working_meta.get("fingerprint") == str(fingerprint)
        and int(working_meta.get("chunk_count") or 0) == int(chunk_count)
        and working_path.exists()
        and not working_path.is_symlink()
    ):
        return True
    return _lexical_index_path_is_current(working_path, fingerprint, chunk_count)


def ensure_lexical_index(chunks, cache_dir, fingerprint, *, build_config_digest=""):
    cache_dir = _private_cache_dir(cache_dir)
    working_path = _working_index_path(cache_dir)
    working_manifest_path = _working_manifest_path(cache_dir)
    cached_manifest = getattr(chunks, "_lexical_manifest", None)
    cached_chunks_by_file = getattr(chunks, "_chunks_by_file", None)
    desired_manifest = (
        _lexical_manifest_for_chunks(chunks)
        if not isinstance(cached_manifest, dict)
        else (_cached_lexical_manifest_shallow_view(cached_manifest) or {})
    )
    stats = {
        "working_cache_hit": False,
        "updated_files": 0,
        "removed_files": 0,
        "inserted_chunks": 0,
        "working_manifest_load_ms": 0.0,
        "working_index_check_ms": 0.0,
        "shared_restore_ms": 0.0,
        "sqlite_update_ms": 0.0,
        "sqlite_rebuild_ms": 0.0,
    }
    publish_shared = not bool(getattr(chunks, "_skip_shared_cache_publication", False))
    if working_path.is_symlink():
        raise OSError(f"inspection lexical index is a symlink: {working_path}")
    if not working_path.exists():
        restore_started = time.perf_counter()
        restored_source = _restore_shared_lexical_index(
            cache_dir,
            fingerprint,
            len(chunks),
            build_config_digest=build_config_digest,
        )
        stats["shared_restore_ms"] = round((time.perf_counter() - restore_started) * 1000.0, 3)
        if restored_source:
            stats["shared_restore"] = True
            stats["shared_restore_source"] = restored_source
    if working_path.exists():
        working_path.chmod(0o600)
        manifest_started = time.perf_counter()
        previous_manifest = _load_lexical_working_manifest(working_manifest_path) or {}
        stats["working_manifest_load_ms"] = round((time.perf_counter() - manifest_started) * 1000.0, 3)
        previous_meta = _lexical_manifest_meta(previous_manifest)
        previous_files = _lexical_manifest_files(previous_manifest)
        if (
            previous_meta.get("fingerprint") == str(fingerprint)
            and int(previous_meta.get("chunk_count") or 0) == len(chunks)
        ):
            stats["working_cache_hit"] = True
            return working_path, True, stats
        if previous_files == desired_manifest and int(previous_meta.get("chunk_count") or 0) == len(chunks):
            stats["working_cache_hit"] = True
            manifest_with_meta = dict(desired_manifest)
            manifest_with_meta["__meta__"] = {
                "fingerprint": str(fingerprint or ""),
                "chunk_count": len(chunks),
                "build_config_digest": str(build_config_digest or ""),
            }
            _write_lexical_working_manifest(
                working_manifest_path,
                manifest_with_meta,
                fingerprint=fingerprint,
                chunk_count=len(chunks),
                build_config_digest=build_config_digest,
            )
            return working_path, True, stats
        existing_keys = set(previous_files)
        desired_keys = set(desired_manifest)
        removed_keys = existing_keys - desired_keys
        changed_keys = {
            key
            for key in desired_keys
            if previous_files.get(key, {}).get("signature") != desired_manifest[key]["signature"]
        }
        if not changed_keys and not removed_keys and existing_keys == desired_keys:
            stats["working_cache_hit"] = True
            stats["updated_files"] = 0
            stats["removed_files"] = 0
            manifest_with_meta = dict(desired_manifest)
            manifest_with_meta["__meta__"] = {
                "fingerprint": str(fingerprint or ""),
                "chunk_count": len(chunks),
                "build_config_digest": str(build_config_digest or ""),
            }
            _write_lexical_working_manifest(
                working_manifest_path,
                manifest_with_meta,
                fingerprint=fingerprint,
                chunk_count=len(chunks),
                build_config_digest=build_config_digest,
            )
            return working_path, True, stats
        if publish_shared or not working_path.exists():
            restore_started = time.perf_counter()
            restored_source = _restore_matching_shared_latest_lexical_index(
                cache_dir,
                desired_manifest,
                build_config_digest=build_config_digest,
            )
            if restored_source:
                stats["shared_restore_ms"] += round((time.perf_counter() - restore_started) * 1000.0, 3)
                stats["working_cache_hit"] = True
                stats["shared_restore"] = True
                stats["shared_restore_source"] = restored_source
                return working_path, True, stats
        if not changed_keys and not removed_keys and existing_keys == desired_keys:
            try:
                check_started = time.perf_counter()
                with sqlite3.connect(working_path) as conn:
                    _configure_lexical_write_connection(conn)
                    row = conn.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
                    count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
                    stats["working_index_check_ms"] = round((time.perf_counter() - check_started) * 1000.0, 3)
                    if _valid_lexical_schema(conn) and row and row[0] == fingerprint and count == len(chunks):
                        stats["working_cache_hit"] = True
                        _write_lexical_working_manifest(
                            working_manifest_path,
                            previous_manifest,
                            fingerprint=fingerprint,
                            chunk_count=len(chunks),
                            build_config_digest=build_config_digest,
                        )
                        return working_path, True, stats
            except sqlite3.Error:
                working_path.unlink(missing_ok=True)
                working_manifest_path.unlink(missing_ok=True)
    previous_manifest = previous_manifest if 'previous_manifest' in locals() else (_load_lexical_working_manifest(working_manifest_path) or {})
    previous_files = previous_files if 'previous_files' in locals() else _lexical_manifest_files(previous_manifest)
    existing_keys = existing_keys if 'existing_keys' in locals() else set(previous_files)
    desired_keys = desired_keys if 'desired_keys' in locals() else set(desired_manifest)
    removed_keys = removed_keys if 'removed_keys' in locals() else (existing_keys - desired_keys)
    changed_keys = changed_keys if 'changed_keys' in locals() else {
        key
        for key in desired_keys
        if previous_files.get(key, {}).get("signature") != desired_manifest[key]["signature"]
    }
    stats["updated_files"] = len(changed_keys)
    stats["removed_files"] = len(removed_keys)
    if (
        working_path.exists()
        and not working_path.is_symlink()
        and not changed_keys
        and not removed_keys
        and existing_keys == desired_keys
    ):
        stats["working_cache_hit"] = True
        manifest_with_meta = dict(desired_manifest)
        manifest_with_meta["__meta__"] = {
            "fingerprint": str(fingerprint or ""),
            "chunk_count": len(chunks),
            "build_config_digest": str(build_config_digest or ""),
        }
        _write_lexical_working_manifest(
            working_manifest_path,
            manifest_with_meta,
            fingerprint=fingerprint,
            chunk_count=len(chunks),
            build_config_digest=build_config_digest,
        )
        return working_path, True, stats
    if (
        working_path.exists()
        and not working_path.is_symlink()
        and not changed_keys
        and not removed_keys
        and existing_keys == desired_keys
    ):
        try:
            update_started = time.perf_counter()
            with sqlite3.connect(working_path) as conn:
                _configure_lexical_write_connection(conn)
                if _valid_lexical_schema(conn):
                    with conn:
                        conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('fingerprint',?)", (fingerprint,))
                    stats["sqlite_update_ms"] = round((time.perf_counter() - update_started) * 1000.0, 3)
                    stats["working_cache_hit"] = True
                    manifest_with_meta = dict(desired_manifest)
                    manifest_with_meta["__meta__"] = {
                        "fingerprint": str(fingerprint or ""),
                        "chunk_count": len(chunks),
                    }
                    _write_lexical_working_manifest(
                        working_manifest_path,
                        manifest_with_meta,
                        fingerprint=fingerprint,
                        chunk_count=len(chunks),
                        build_config_digest=build_config_digest,
                    )
                    _publish_shared_lexical_index(
                        cache_dir,
                        fingerprint,
                        build_config_digest=build_config_digest,
                        publish_shared=publish_shared,
                    )
                    return working_path, True, stats
        except sqlite3.Error:
            working_path.unlink(missing_ok=True)
            working_manifest_path.unlink(missing_ok=True)
    rebuild = True
    if working_path.exists():
        working_path.chmod(0o600)
        try:
            with sqlite3.connect(working_path) as conn:
                _configure_lexical_write_connection(conn)
                if _valid_lexical_schema(conn):
                    update_started = time.perf_counter()
                    with conn:
                        for file_key in sorted(removed_keys | changed_keys):
                            stale = previous_files.get(file_key, {})
                            stale_chunk_ids = [
                                str(chunk.get("chunk_id") or "")
                                for chunk in stale.get("chunks") or ()
                                if str(chunk.get("chunk_id") or "")
                            ]
                            if stale_chunk_ids:
                                _delete_lexical_fts_chunk_ids(conn, stale_chunk_ids)
                            conn.execute("DELETE FROM chunks WHERE file_key=?", (file_key,))
                        if changed_keys:
                            file_chunks = cached_chunks_by_file if isinstance(cached_chunks_by_file, dict) else None
                            if file_chunks is None or any(file_key not in file_chunks for file_key in changed_keys):
                                file_chunks = defaultdict(list)
                                for chunk in chunks:
                                    file_chunks[_lexical_chunk_file_key(chunk)].append(chunk)
                            for file_key in sorted(changed_keys):
                                file_chunk_rows = list(file_chunks.get(file_key) or ())
                                stats["inserted_chunks"] += len(file_chunk_rows)
                                _insert_lexical_chunks(conn, file_chunk_rows)
                        conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('fingerprint',?)", (fingerprint,))
                    stats["sqlite_update_ms"] = round((time.perf_counter() - update_started) * 1000.0, 3)
                    rebuild = False
                    stats["working_cache_hit"] = not changed_keys and not removed_keys
                else:
                    working_path.unlink(missing_ok=True)
                    if rebuild and working_manifest_path.exists():
                        working_manifest_path.unlink(missing_ok=True)
        except sqlite3.Error:
            working_path.unlink(missing_ok=True)
            working_manifest_path.unlink(missing_ok=True)
    if rebuild:
        manifest = desired_manifest
        stats["updated_files"] = len(manifest)
        stats["removed_files"] = 0
        stats["inserted_chunks"] = len(chunks)
        rebuild_started = time.perf_counter()
        _rebuild_working_lexical_index(working_path, chunks, fingerprint)
        stats["sqlite_rebuild_ms"] = round((time.perf_counter() - rebuild_started) * 1000.0, 3)
    manifest_with_meta = dict(desired_manifest)
    manifest_with_meta["__meta__"] = {
        "fingerprint": str(fingerprint or ""),
        "chunk_count": len(chunks),
    }
    _write_lexical_working_manifest(
        working_manifest_path,
        manifest_with_meta,
        fingerprint=fingerprint,
        chunk_count=len(chunks),
        build_config_digest=build_config_digest,
    )
    _publish_shared_lexical_index(
        cache_dir,
        fingerprint,
        build_config_digest=build_config_digest,
        publish_shared=publish_shared,
    )
    return working_path, False, stats


def identifier_pieces(value):
    """Split paths and snake/camel identifiers into searchable components."""

    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value))
    expanded = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", expanded)
    return [piece.lower() for piece in re.split(r"[^A-Za-z0-9]+", expanded) if len(piece) >= 2]


def query_features(query: str):
    cache_key = str(query)
    cached = _QUERY_FEATURE_CACHE.get(cache_key)
    if cached is not None:
        _QUERY_FEATURE_CACHE.pop(cache_key, None)
        _QUERY_FEATURE_CACHE[cache_key] = cached
        return {
            "quoted": list(cached["quoted"]),
            "paths": list(cached["paths"]),
            "identifiers": list(cached["identifiers"]),
            "terms": list(cached["terms"]),
            "direct_terms": list(cached["direct_terms"]),
        }
    quoted = [value.strip() for value in re.findall(r'["\']([^"\']+)["\']', query) if value.strip()]
    tokens = re.findall(r"[A-Za-z_][\w./:-]*|\d+", query)
    paths = [token for token in tokens if "/" in token or re.search(r"\.[A-Za-z0-9]{1,8}(?::\d+)?$", token)]
    identifiers = [token for token in tokens if "_" in token or re.search(r"[a-z][A-Z]|[A-Z]{2,}", token)]
    terms = []
    seen = set()
    for value in quoted + paths + identifiers + tokens:
        value = value.strip(".,;:()[]{}")
        if len(value) < 2 or value.lower() in seen:
            continue
        seen.add(value.lower())
        terms.append(value)
        for piece in re.split(r"[./:_-]+", value):
            if len(piece) < 2 or piece.lower() in seen:
                continue
            seen.add(piece.lower())
            terms.append(piece)
        for piece in identifier_pieces(value):
            if piece not in seen:
                seen.add(piece)
                terms.append(piece)
    direct_terms = list(terms)
    aliases = {
        "authentication": ("auth",),
        "authorization": ("auth", "authz", "access"),
        "authorized": ("authorization", "auth", "authz", "access"),
        "configuration": ("config",),
        "configured": ("config",),
        "diagnostics": ("diagnostic",),
        "parameters": ("parameter", "param", "params"),
        "parameter": ("param", "params"),
        "repositories": ("repository", "repo"),
        "repository": ("repo",),
        "reranking": ("rerank",),
        "embeddings": ("embedding", "embed"),
        "capabilities": ("capability",),
        "scheduler": ("slurm",),
        "startup": ("launch",),
        "unhealthy": ("health",),
        "healthy": ("health",),
        "stale": ("health", "heartbeat"),
        "release": ("artifact",),
        "released": ("release", "artifact"),
        "resolution": ("resolve",),
        "arguments": ("argument", "args"),
        "traces": ("trace",),
    }
    for value in list(terms):
        lowered = value.lower()
        variants = list(aliases.get(lowered, ()))
        if len(lowered) > 4 and lowered.endswith("s"):
            variants.append(lowered[:-1])
        if len(lowered) > 5 and lowered.endswith("ing"):
            variants.append(lowered[:-3])
        if len(lowered) > 4 and lowered.endswith("ed"):
            variants.append(lowered[:-2])
        if len(lowered) > 5 and lowered.endswith("ies"):
            variants.append(lowered[:-3] + "y")
        for variant in variants:
            if len(variant) >= 2 and variant not in seen:
                seen.add(variant)
                terms.append(variant)
    features = {
        "quoted": quoted,
        "paths": paths,
        "identifiers": identifiers,
        "terms": terms,
        "direct_terms": direct_terms,
    }
    _QUERY_FEATURE_CACHE[cache_key] = {
        "quoted": list(quoted),
        "paths": list(paths),
        "identifiers": list(identifiers),
        "terms": list(terms),
        "direct_terms": list(direct_terms),
    }
    while len(_QUERY_FEATURE_CACHE) > QUERY_FEATURE_CACHE_LIMIT:
        oldest_key = next(iter(_QUERY_FEATURE_CACHE))
        _QUERY_FEATURE_CACHE.pop(oldest_key, None)
    return features


def _fts_expression(features):
    values = features["quoted"] + features["paths"] + features["identifiers"] + features["terms"][:16]
    escaped = []
    seen = set()
    for value in values:
        value = value.replace('"', '""').strip()
        if not value or value.lower() in seen:
            continue
        seen.add(value.lower())
        escaped.append(f'"{value}"')
    return " OR ".join(escaped)


def _lexical_helper_cache_key(index_path, chunks):
    try:
        stat = Path(index_path).stat()
        stamp = int(stat.st_mtime_ns)
    except OSError:
        stamp = 0
    return (str(Path(index_path).resolve(strict=False)), stamp, len(chunks))


def _lexical_result_cache_key(index_path, chunks, query, limit):
    helper_key = _lexical_helper_cache_key(index_path, chunks)
    return (*helper_key, str(query), int(limit))


def _lexical_helper_path(index_path, *, create=False):
    base = Path(index_path).parent
    return base / "lexical-helper.pkl"


def _shared_lexical_helper_path(fingerprint: str, *, create=False):
    cache_root = _shared_repo_inspection_cache_root(create=create)
    if cache_root is None:
        return None
    target_root = cache_root / "lexical-working-cache"
    try:
        target_root = _private_cache_dir(target_root) if create else target_root.expanduser().absolute()
    except OSError:
        return None
    safe = str(fingerprint or "").replace(":", "_")
    return target_root / f"lexical-{safe}-helper.pkl"


def _lexical_helper_identity(index_path, chunk_count):
    manifest = _load_lexical_working_manifest(_working_manifest_path(Path(index_path).parent)) or {}
    meta = _lexical_manifest_meta(manifest)
    fingerprint = str(meta.get("fingerprint") or "")
    if not fingerprint:
        return None
    if int(meta.get("chunk_count") or 0) != int(chunk_count):
        return None
    return {"fingerprint": fingerprint, "chunk_count": int(chunk_count)}


def _hydrate_loaded_lexical_helper(helper):
    if not isinstance(helper, dict):
        return None
    if not isinstance(helper.get("path_directory_by_path"), dict):
        helper["path_directory_by_path"] = {
            str(path): _path_parent_string(path)
            for path in (helper.get("chunks_by_path") or {})
        }
    helper["term_chunk_ids_cache"] = {}
    helper["term_file_paths_cache"] = {}
    helper["path_term_paths_cache"] = {}
    helper["path_catalog"] = _lexical_path_catalog_from_helper(helper)
    return helper


def _load_persisted_lexical_helper(index_path, cache_key):
    path = _lexical_helper_path(index_path, create=False)
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError, TypeError):
        path.unlink(missing_ok=True)
        return None
    if not isinstance(payload, dict) or payload.get("schema") != "repo-inspection-lexical-helper-v1":
        path.unlink(missing_ok=True)
        return None
    if tuple(payload.get("cache_key") or ()) != tuple(cache_key):
        return None
    helper = payload.get("helper")
    helper = _hydrate_loaded_lexical_helper(helper)
    if helper is None:
        path.unlink(missing_ok=True)
        return None
    return helper


def _load_shared_lexical_helper(index_path, chunk_count):
    identity = _lexical_helper_identity(index_path, chunk_count)
    if identity is None:
        return None
    path = _shared_lexical_helper_path(identity["fingerprint"], create=False)
    if path is None or not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError, TypeError):
        path.unlink(missing_ok=True)
        return None
    if not isinstance(payload, dict) or payload.get("schema") != "repo-inspection-lexical-helper-shared-v1":
        path.unlink(missing_ok=True)
        return None
    if str(payload.get("fingerprint") or "") != identity["fingerprint"]:
        return None
    if int(payload.get("chunk_count") or 0) != identity["chunk_count"]:
        return None
    helper = _hydrate_loaded_lexical_helper(payload.get("helper"))
    if helper is None:
        path.unlink(missing_ok=True)
        return None
    return helper


def _write_persisted_lexical_helper(index_path, cache_key, helper, *, write_shared=True):
    path = _lexical_helper_path(index_path, create=True)
    payload = {
        "schema": "repo-inspection-lexical-helper-v1",
        "cache_key": list(cache_key),
        "helper": dict(helper),
    }
    payload["helper"]["term_chunk_ids_cache"] = {}
    payload["helper"]["term_file_paths_cache"] = {}
    payload["helper"]["path_term_paths_cache"] = {}
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        path.chmod(0o600)
    except OSError:
        tmp.unlink(missing_ok=True)

    if not write_shared:
        return
    identity = _lexical_helper_identity(index_path, len(helper.get("chunk_ids") or ()))
    if identity is None:
        return
    shared_path = _shared_lexical_helper_path(identity["fingerprint"], create=True)
    if shared_path is None:
        return
    shared_payload = {
        "schema": "repo-inspection-lexical-helper-shared-v1",
        "fingerprint": identity["fingerprint"],
        "chunk_count": identity["chunk_count"],
        "helper": dict(helper),
    }
    shared_payload["helper"]["term_chunk_ids_cache"] = {}
    shared_payload["helper"]["term_file_paths_cache"] = {}
    shared_payload["helper"]["path_term_paths_cache"] = {}
    shared_tmp = shared_path.with_name(shared_path.name + ".tmp")
    try:
        with shared_tmp.open("wb") as handle:
            pickle.dump(shared_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(shared_tmp, shared_path)
        shared_path.chmod(0o600)
    except OSError:
        shared_tmp.unlink(missing_ok=True)


def lexical_cache_key(index_path, chunks):
    return _lexical_helper_cache_key(index_path, chunks)


def lexical_helper(index_path, chunks, *, cache_key=None):
    return _get_lexical_helper(index_path, chunks, cache_key=cache_key)


def _lexical_path_catalog_from_helper(helper):
    cached = helper.get("path_catalog")
    if isinstance(cached, dict):
        return cached
    catalog = {
        "unique_paths": tuple(helper["chunks_by_path"].keys()),
        "path_lower_by_path": helper["path_lower_by_path"],
        "path_basename_lower_by_path": helper["path_basename_lower_by_path"],
    }
    helper["path_catalog"] = catalog
    return catalog


def lexical_path_catalog(index_path, chunks, *, cache_key=None, helper=None):
    helper = helper if isinstance(helper, dict) else _get_lexical_helper(index_path, chunks, cache_key=cache_key)
    return _lexical_path_catalog_from_helper(helper)


def _path_parent_string(path: str) -> str:
    value = str(path or "")
    if not value:
        return "."
    if "/" not in value:
        return "."
    parent = value.rsplit("/", 1)[0]
    return parent or "."


def _path_name_string(path: str) -> str:
    value = str(path or "")
    if not value:
        return ""
    return value.rsplit("/", 1)[-1]


def _path_cache_lookup_key(path: Path) -> str:
    if path.is_absolute():
        return str(path)
    return str(path.resolve(strict=False))


def _build_lexical_helper(chunks):
    chunks_by_path = defaultdict(list)
    chunk_searchable = {}
    chunk_path_by_id = {}
    symbol_lower_by_chunk = {}
    symbol_counts_by_file = defaultdict(int)
    symbol_files = defaultdict(set)
    chunk_ids_by_symbol_lower = defaultdict(set)
    language_by_path = {}
    shell_contents_by_path = defaultdict(list)
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        path = chunk["path"]
        symbol = str(chunk.get("symbol", ""))
        content = chunk["content"]
        searchable = f"{symbol}\n{content}".lower()
        symbol_lower = symbol.lower()
        chunks_by_path[path].append(chunk_id)
        chunk_searchable[chunk_id] = searchable
        chunk_path_by_id[chunk_id] = path
        symbol_lower_by_chunk[chunk_id] = symbol_lower
        language_by_path.setdefault(path, chunk.get("language"))
        if chunk.get("language") == "shell":
            shell_contents_by_path[path].append(content)
        normalized_symbol = symbol_lower.strip()
        if normalized_symbol:
            symbol_counts_by_file[(path, normalized_symbol)] += 1
            symbol_files[normalized_symbol].add(path)
            chunk_ids_by_symbol_lower[normalized_symbol].add(chunk_id)

    file_symbol_pieces = {}
    path_lower_by_path = {}
    path_parts_by_path = {}
    path_basename_lower_by_path = {}
    path_directory_by_path = {}
    files_by_directory = defaultdict(list)
    paths_by_basename = defaultdict(list)
    for path, chunk_ids in chunks_by_path.items():
        file_symbol_pieces[path] = {
            piece
            for chunk_id in chunk_ids
            for piece in identifier_pieces(symbol_lower_by_chunk[chunk_id])
        }
        path_lower_by_path[path] = path.lower()
        pieces = set(identifier_pieces(path))
        pieces.update(re.findall(r"[a-z0-9_]+", path_lower_by_path[path]))
        path_parts_by_path[path] = pieces
        basename = _path_name_string(path_lower_by_path[path]).lower()
        directory = _path_parent_string(path)
        path_basename_lower_by_path[path] = basename
        path_directory_by_path[path] = directory
        files_by_directory[directory].append(path)
        paths_by_basename[_path_name_string(path)].append(path)

    wrapper_reference_targets = {}
    static_flags_by_path = {}
    for path, chunk_ids in chunks_by_path.items():
        if not chunk_ids or language_by_path.get(path) != "shell":
            continue
        source_text = "\n".join(shell_contents_by_path.get(path, ()))
        if len(source_text.splitlines()) > 20:
            continue
        referenced_targets = set()
        for chunk_id in chunk_ids:
            chunk_content = chunk_searchable[chunk_id]
            for reference in re.findall(
                r"[A-Za-z0-9_./${}-]+\.(?:go|py|sh|slurm|json|toml|ya?ml)",
                chunk_content,
            ):
                basename = Path(reference).name
                targets = paths_by_basename.get(basename) or []
                if len(targets) == 1 and targets[0] != path:
                    referenced_targets.add(targets[0])
        if referenced_targets:
            wrapper_reference_targets[path] = referenced_targets

    for path, chunk_ids in chunks_by_path.items():
        if not chunk_ids:
            continue
        path_lower = path_lower_by_path[path]
        path_parts = path_parts_by_path.get(path, set())
        basename_lower = path_basename_lower_by_path[path]
        language = language_by_path.get(path)
        static_flags_by_path[path] = {
            "is_test_or_fixture": (
                path_lower.startswith("tests/")
                or "/tests/" in path_lower
                or path_lower.endswith("_test.go")
                or basename_lower.startswith("test_")
                or basename_lower.endswith("_test.py")
                or "fixture" in path_parts
                or "fixtures" in path_parts
                or "golden" in path_parts
            ),
            "is_documentation": (
                path_lower.startswith("docs/")
                or basename_lower.startswith("readme")
                or language == "markdown"
            ),
            "config_source": (
                path_lower.startswith("configs/")
                or language in {"config", "json", "toml", "yaml"}
            ),
            "tooling_script": (
                path_lower.startswith("scripts/")
                or basename_lower.startswith("install")
            ),
            "is_architecture_primary_path": path_lower.startswith(("broker/pkg/", "src/", "internal/")),
            "is_cmd_path": path_lower.startswith("cmd/"),
            "is_examples_path": path_lower.startswith("examples/"),
            "language_is_code_like": language not in {"json", "text", "yaml", "toml", "xml"},
        }

    helper = {
        "chunk_ids": tuple(chunk_searchable.keys()),
        "chunks_by_path": chunks_by_path,
        "chunk_searchable": chunk_searchable,
        "symbol_lower_by_chunk": symbol_lower_by_chunk,
        "chunk_path_by_id": chunk_path_by_id,
        "file_symbol_pieces": file_symbol_pieces,
        "path_lower_by_path": path_lower_by_path,
        "path_parts_by_path": path_parts_by_path,
        "path_basename_lower_by_path": path_basename_lower_by_path,
        "path_directory_by_path": path_directory_by_path,
        "files_by_directory": files_by_directory,
        "paths_by_basename": paths_by_basename,
        "wrapper_reference_targets": wrapper_reference_targets,
        "static_flags_by_path": static_flags_by_path,
        "symbol_counts_by_file": symbol_counts_by_file,
        "symbol_files": symbol_files,
        "chunk_ids_by_symbol_lower": {
            symbol: frozenset(chunk_ids)
            for symbol, chunk_ids in chunk_ids_by_symbol_lower.items()
        },
        "corpus_size": max(1, len(chunks)),
        "file_count": max(1, len(chunks_by_path)),
        "term_chunk_ids_cache": {},
        "term_file_paths_cache": {},
        "path_term_paths_cache": {},
    }
    return helper


def _build_lexical_helper_from_index(index_path):
    rows = []
    try:
        with sqlite3.connect(index_path) as conn:
            rows = conn.execute(
                "SELECT chunk_id, path, language, symbol, content FROM chunks ORDER BY rowid"
            ).fetchall()
    except sqlite3.Error:
        return None
    chunks = [
        {
            "chunk_id": str(chunk_id),
            "path": str(path),
            "language": str(language or ""),
            "symbol": str(symbol or ""),
            "content": str(content or ""),
        }
        for chunk_id, path, language, symbol, content in rows
    ]
    return _build_lexical_helper(chunks)


def _matching_chunk_ids_for_term(helper, term):
    term = str(term).lower()
    cached = helper["term_chunk_ids_cache"].get(term)
    if cached is not None:
        return cached
    matching = frozenset(
        chunk_id
        for chunk_id, searchable in helper["chunk_searchable"].items()
        if term in searchable
    )
    helper["term_chunk_ids_cache"][term] = matching
    return matching


def _matching_file_paths_for_term(helper, term):
    term = str(term).lower()
    cached = helper["term_file_paths_cache"].get(term)
    if cached is not None:
        return cached
    matching = frozenset(helper["chunk_path_by_id"].get(chunk_id, "") for chunk_id in _matching_chunk_ids_for_term(helper, term))
    matching = frozenset(path for path in matching if path)
    helper["term_file_paths_cache"][term] = matching
    return matching


def _matching_paths_for_path_term(helper, term):
    term = str(term).lower()
    cached = helper["path_term_paths_cache"].get(term)
    if cached is not None:
        return cached
    matching = frozenset(
        path
        for path, path_parts in helper["path_parts_by_path"].items()
        if any(term == part or part.startswith(term) or part.endswith(term) for part in path_parts)
    )
    helper["path_term_paths_cache"][term] = matching
    return matching


def _get_lexical_helper(index_path, chunks, *, cache_key=None):
    key = cache_key if cache_key is not None else _lexical_helper_cache_key(index_path, chunks)
    explicit_cache_key = cache_key is not None
    cached = _LEXICAL_HELPER_CACHE.get(key)
    if cached is not None:
        _LEXICAL_HELPER_CACHE.pop(key, None)
        _LEXICAL_HELPER_CACHE[key] = cached
        return cached
    persisted = _load_persisted_lexical_helper(index_path, key)
    if persisted is not None:
        _LEXICAL_HELPER_CACHE[key] = persisted
        while len(_LEXICAL_HELPER_CACHE) > LEXICAL_HELPER_CACHE_LIMIT:
            oldest_key = next(iter(_LEXICAL_HELPER_CACHE))
            _LEXICAL_HELPER_CACHE.pop(oldest_key, None)
        return persisted
    shared = None if explicit_cache_key else _load_shared_lexical_helper(index_path, len(chunks))
    if shared is not None:
        _LEXICAL_HELPER_CACHE[key] = shared
        while len(_LEXICAL_HELPER_CACHE) > LEXICAL_HELPER_CACHE_LIMIT:
            oldest_key = next(iter(_LEXICAL_HELPER_CACHE))
            _LEXICAL_HELPER_CACHE.pop(oldest_key, None)
        return shared
    needs_index_fallback = any("content" not in chunk for chunk in chunks)
    helper = _build_lexical_helper_from_index(index_path) if needs_index_fallback else _build_lexical_helper(chunks)
    if helper is None:
        helper = _build_lexical_helper(chunks)
    helper["path_catalog"] = _lexical_path_catalog_from_helper(helper)
    _LEXICAL_HELPER_CACHE[key] = helper
    _write_persisted_lexical_helper(index_path, key, helper, write_shared=not explicit_cache_key)
    while len(_LEXICAL_HELPER_CACHE) > LEXICAL_HELPER_CACHE_LIMIT:
        oldest_key = next(iter(_LEXICAL_HELPER_CACHE))
        _LEXICAL_HELPER_CACHE.pop(oldest_key, None)
    return helper


def lexical_search(index_path, query, chunks, limit=128, *, cache_key=None, features=None, helper=None):
    helper_cache_key = cache_key if cache_key is not None else _lexical_helper_cache_key(index_path, chunks)
    result_cache_key = (*helper_cache_key, str(query), int(limit))
    cached_result = _LEXICAL_RESULT_CACHE.get(result_cache_key)
    if cached_result is not None:
        _LEXICAL_RESULT_CACHE.pop(result_cache_key, None)
        _LEXICAL_RESULT_CACHE[result_cache_key] = cached_result
        return [dict(item) for item in cached_result]
    helper = helper if isinstance(helper, dict) else _get_lexical_helper(index_path, chunks, cache_key=cache_key)
    chunk_path_by_id = helper["chunk_path_by_id"]
    features = features or query_features(query)
    fts = _fts_expression(features)
    raw_scores = defaultdict(float)
    structural_terms = {
        "architecture",
        "call",
        "chain",
        "flow",
        "implementation",
        "implemented",
        "entrypoint",
        "route",
        "request",
        "service",
        "worker",
        "handler",
        "dispatch",
    }
    architecture_query = bool({term.lower() for term in features["terms"]} & structural_terms)
    lowered_query_terms = {term.lower() for term in features["terms"]}
    call_chain_query = "call" in lowered_query_terms and bool(
        lowered_query_terms
        & {"chain", "dispatch", "entrypoint", "execution", "flow", "handler", "route", "staging", "submission"}
    )
    stop_words = {
        "and", "are", "does", "find", "for", "from", "how", "into", "its", "not", "the",
        "then", "this", "through", "to", "what", "when", "where", "which", "why", "with",
    }
    lexical_terms = [
        term.lower()
        for term in features["terms"]
        if len(term) >= 3 and term.lower() not in stop_words
    ][:32]
    scored_query_terms = [term.lower() for term in features["terms"] if len(term) >= 3]
    lowered_quoted_phrases = [phrase.lower() for phrase in features["quoted"]]
    lowered_named_paths = [named_path.lower().split(":", 1)[0] for named_path in features["paths"]]
    term_chunk_ids = {
        term: _matching_chunk_ids_for_term(helper, term)
        for term in lexical_terms
    }
    term_document_frequency = {
        term: len(chunk_ids)
        for term, chunk_ids in term_chunk_ids.items()
    }
    term_inverse_frequency = {
        term: math.log((max(1, len(helper["chunk_ids"])) + 1) / (count + 1)) + 1.0
        for term, count in term_document_frequency.items()
    }
    chunk_matching_terms = defaultdict(list)
    chunk_term_occurrences = defaultdict(dict)
    for term, chunk_ids in term_chunk_ids.items():
        for chunk_id in chunk_ids:
            chunk_matching_terms[chunk_id].append(term)
            occurrences = min(3, helper["chunk_searchable"][chunk_id].count(term))
            if occurrences:
                chunk_term_occurrences[chunk_id][term] = occurrences
    chunks_by_path = helper["chunks_by_path"]
    corpus_size = helper["corpus_size"]

    # A repository question often names concepts implemented across several
    # functions in one module.  Score bounded, distinct term coverage at file
    # scope so a focused module is not lost because no single chunk contains
    # every query concept.
    term_file_paths = {
        term: _matching_file_paths_for_term(helper, term)
        for term in lexical_terms
    }
    file_term_frequency = {
        term: len(paths)
        for term, paths in term_file_paths.items()
    }
    file_term_inverse_frequency = {
        term: math.log((helper["file_count"] + 1) / (count + 1)) + 1.0
        for term, count in file_term_frequency.items()
    }
    file_matching_terms = defaultdict(list)
    for term, paths in term_file_paths.items():
        for path in paths:
            file_matching_terms[path].append(term)
    file_count = helper["file_count"]
    file_coverage_bonus = {}
    for path in chunks_by_path:
        matched_weights = []
        symbol_pieces = helper["file_symbol_pieces"][path]
        symbol_bonus = 0.0
        for term in file_matching_terms.get(path, ()):
            inverse_frequency = file_term_inverse_frequency[term]
            matched_weights.append(min(5.0, inverse_frequency))
            if term in symbol_pieces:
                symbol_bonus += min(2.0, inverse_frequency * 0.35)
        strongest = sorted(matched_weights, reverse=True)[:10]
        coverage_ratio = len(strongest) / max(1, min(10, len(lexical_terms)))
        length_normalizer = 1.0 + 0.08 * math.log2(max(1, len(chunks_by_path[path])))
        file_coverage_bonus[path] = min(
            20.0,
            ((0.55 * sum(strongest)) + (5.0 * coverage_ratio) + min(4.0, symbol_bonus))
            / length_normalizer,
        )
    files_by_directory = helper["files_by_directory"]
    directory_coverage_bonus = {}
    for directory, directory_paths in files_by_directory.items():
        directory_terms = set()
        for path in directory_paths:
            directory_terms.update(file_matching_terms.get(path, ()))
        matched_weights = [
            min(5.0, file_term_inverse_frequency[term])
            for term in directory_terms
        ]
        strongest = sorted(matched_weights, reverse=True)[:10]
        coverage_ratio = len(strongest) / max(1, min(10, len(lexical_terms)))
        sibling_normalizer = 1.0 + 0.12 * max(0, len(directory_paths) - 1)
        bonus = min(20.0, ((0.6 * sum(strongest)) + (5.0 * coverage_ratio)) / sibling_normalizer)
        for path in directory_paths:
            directory_coverage_bonus[path] = bonus

    path_parts_by_path = helper["path_parts_by_path"]
    path_term_frequency = {term.lower(): 0 for term in features["terms"] if len(term) >= 3}
    path_term_matches = {
        term: _matching_paths_for_path_term(helper, term)
        for term in path_term_frequency
    }
    for term, paths in path_term_matches.items():
        path_term_frequency[term] = len(paths)
    path_term_inverse_frequency = {
        term: math.log((file_count + 1) / (count + 1)) + 1.0
        for term, count in path_term_frequency.items()
    }
    direct_terms = {term.lower() for term in features.get("direct_terms") or []}
    lowered_identifiers = [identifier.lower() for identifier in features["identifiers"]]
    identifier_bonus_by_chunk = defaultdict(float)
    chunk_ids_by_symbol_lower = helper["chunk_ids_by_symbol_lower"]
    for lowered in lowered_identifiers:
        for chunk_id in chunk_ids_by_symbol_lower.get(lowered, ()):
            identifier_bonus_by_chunk[chunk_id] += 5.0
        for chunk_id in _matching_chunk_ids_for_term(helper, lowered):
            if chunk_id not in chunk_ids_by_symbol_lower.get(lowered, ()):
                identifier_bonus_by_chunk[chunk_id] += 2.0
    # Concept queries such as "find retry logic" often omit the exact public
    # symbol name.  Keep all matching symbols visible to the diversity pass so
    # a package-level chunk cannot displace the actual entry point.
    concept_symbol_bonus_by_chunk = defaultdict(float)
    for chunk_id, symbol_lower in helper["symbol_lower_by_chunk"].items():
        if not symbol_lower:
            continue
        matches = [term for term in lexical_terms if term in symbol_lower]
        if matches:
            concept_symbol_bonus_by_chunk[chunk_id] = min(30.0, 8.0 * len(matches))
    quoted_phrase_bonus_by_chunk = defaultdict(float)
    for phrase in lowered_quoted_phrases:
        for chunk_id in _matching_chunk_ids_for_term(helper, phrase):
            quoted_phrase_bonus_by_chunk[chunk_id] += 4.0
    redundancy_intent = bool(
        {term.lower() for term in features["terms"]}
        & {"dry", "duplicate", "duplicat", "duplication", "redundancy", "repeat", "repeated"}
    )
    explicit_test_intent = bool(
        lowered_query_terms & {"assert", "coverage", "fixture", "test", "tests", "verification", "verify"}
    )
    validation_intent = bool(
        lowered_query_terms & {"contract", "schema", "validate", "validation"}
    ) or bool(
        lowered_query_terms & {"require", "required"}
        and lowered_query_terms & {"config", "configuration", "deployment"}
    )
    configuration_intent = bool(
        direct_terms & {"config", "configuration", "deployment", "environment", "settings"}
    )
    examples_intent = bool(direct_terms & {"client", "example", "integration", "template"})
    tooling_intent = bool(direct_terms & {"build", "install", "script"})
    wrapper_intent = bool(
        direct_terms & {"cli", "command", "entrypoint", "launcher", "script", "shell", "wrapper"}
    )
    symbol_counts_by_file = helper["symbol_counts_by_file"]
    symbol_files = helper["symbol_files"]
    redundancy_bonus_by_chunk = defaultdict(float)
    if redundancy_intent:
        for chunk_id, path in helper["chunk_path_by_id"].items():
            symbol_lower = helper["symbol_lower_by_chunk"][chunk_id]
            if not symbol_lower:
                continue
            same_file_count = symbol_counts_by_file[(path, symbol_lower)]
            cross_file_count = len(symbol_files[symbol_lower])
            redundancy_bonus_by_chunk[chunk_id] = min(
                80.0,
                (10.0 * math.log2(1 + same_file_count)) + (3.0 * max(0, cross_file_count - 1)),
            )
    use_sqlite_fts = bool(fts) and int(len(helper.get("chunk_ids") or ())) > SMALL_CORPUS_LEXICAL_FTS_THRESHOLD
    if use_sqlite_fts:
        try:
            with sqlite3.connect(index_path) as conn:
                rows = conn.execute(
                    "SELECT chunk_id, bm25(chunks_fts, 8.0, 6.0, 1.0) AS score "
                    "FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                    (fts, max(limit * 2, 64)),
                ).fetchall()
            for rank, (chunk_id, bm25_score) in enumerate(rows, start=1):
                raw_scores[chunk_id] += 1.0 / rank
                raw_scores[chunk_id] += 1.0 / (1.0 + abs(float(bm25_score)))
        except sqlite3.Error:
            pass

    explicit_named_path_match_by_path = {
        path: any(named_path in path_lower for named_path in lowered_named_paths)
        for path, path_lower in helper["path_lower_by_path"].items()
    }
    common_path_terms = {
        "broker",
        "code",
        "gpu",
        "inspect",
        "inspection",
        "main",
        "model",
        "repo",
        "repository",
        "service",
        "worker",
    }
    path_query_bonus_by_path = {}
    for path in chunks_by_path:
        path_boost = 0.0
        for lowered_term in scored_query_terms:
            if path in path_term_matches.get(lowered_term, ()):
                if lowered_term in common_path_terms:
                    path_boost += 0.75
                else:
                    inverse_path_frequency = path_term_inverse_frequency.get(lowered_term, 1.0)
                    base_weight = 22.0 if lowered_term in direct_terms else 9.0
                    path_boost += min(60.0, base_weight * inverse_path_frequency)
        path_query_bonus_by_path[path] = path_boost
    path_policy_by_path = {}
    for path in chunks_by_path:
        path_lower = helper["path_lower_by_path"][path]
        static_flags = helper["static_flags_by_path"][path]
        explicit_path_match = explicit_named_path_match_by_path.get(path, False)
        pre_multiplier = 1.0
        post_multiplier = 1.0
        additive = 0.0
        if static_flags["is_test_or_fixture"] and not explicit_path_match:
            pre_multiplier *= 0.7 if explicit_test_intent else (0.7 if validation_intent else 0.08)
            if explicit_test_intent and path_lower.startswith("tests/unit/"):
                additive += 25.0
            if validation_intent:
                additive += 20.0
        elif static_flags["is_documentation"] and not explicit_path_match:
            pre_multiplier *= 0.2
        elif static_flags["language_is_code_like"]:
            additive += 2.0
            if architecture_query and static_flags["is_architecture_primary_path"]:
                additive += 60.0 if call_chain_query else 3.0
            elif architecture_query and static_flags["is_cmd_path"]:
                additive += 2.0
        if static_flags["config_source"] and not explicit_path_match and not configuration_intent:
            post_multiplier *= 0.3
        if static_flags["is_examples_path"] and not examples_intent:
            post_multiplier *= 0.3
        if static_flags["tooling_script"] and not tooling_intent:
            post_multiplier *= 0.35
        path_policy_by_path[path] = {
            "explicit_path_match": explicit_path_match,
            "pre_multiplier": pre_multiplier,
            "post_multiplier": post_multiplier,
            "additive": additive,
        }
    path_static_score_by_path = {
        path: (
            file_coverage_bonus.get(path, 0.0)
            + directory_coverage_bonus.get(path, 0.0)
            + path_query_bonus_by_path.get(path, 0.0)
            + (5.0 if path_policy_by_path[path]["explicit_path_match"] else 0.0)
        )
        for path in chunks_by_path
    }

    # FTS tokenization is complemented by exact path, phrase, and identifier matches.
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        path = chunk["path"]
        path_policy = path_policy_by_path[path]
        score = raw_scores[chunk_id] + path_static_score_by_path[path]
        score += redundancy_bonus_by_chunk.get(chunk_id, 0.0)
        for term in chunk_matching_terms.get(chunk_id, ()):
            occurrences = chunk_term_occurrences.get(chunk_id, {}).get(term, 0)
            if not occurrences:
                continue
            inverse_frequency = term_inverse_frequency[term]
            score += inverse_frequency * (1.0 + (occurrences - 1) * 0.25)
        score += quoted_phrase_bonus_by_chunk.get(chunk_id, 0.0)
        score += identifier_bonus_by_chunk.get(chunk_id, 0.0)
        score += concept_symbol_bonus_by_chunk.get(chunk_id, 0.0)
        score = ((score * path_policy["pre_multiplier"]) + path_policy["additive"]) * path_policy["post_multiplier"]
        raw_scores[chunk_id] = score

    # Propagate a bounded score through explicit file references.  Thin CLI
    # wrappers and launchers often contain little vocabulary beyond the source
    # file they delegate to, but that reference is strong structural evidence.
    best_by_path = {}
    best_chunk_by_path = {}
    for path, chunk_ids in chunks_by_path.items():
        if not chunk_ids:
            continue
        strongest_chunk_id = max(chunk_ids, key=lambda chunk_id: (raw_scores.get(chunk_id, 0.0), chunk_id))
        best_by_path[path] = raw_scores.get(strongest_chunk_id, 0.0)
        best_chunk_by_path[path] = strongest_chunk_id
    wrapper_source_multiplier_by_path = {}
    for source_path in helper["wrapper_reference_targets"]:
        source_lower = helper["path_lower_by_path"].get(source_path, source_path.lower())
        source_basename_lower = helper["path_basename_lower_by_path"].get(source_path, Path(source_path).name.lower())
        multiplier = 1.0
        if (
            source_lower.startswith("tests/")
            or source_lower.endswith("_test.go")
            or source_basename_lower.startswith("test_")
        ):
            multiplier = 0.7 if explicit_test_intent else (0.4 if validation_intent else 0.08)
        elif source_lower.startswith("docs/") or source_basename_lower.startswith("readme"):
            multiplier = 0.2
        wrapper_source_multiplier_by_path[source_path] = multiplier
    for source_path, referenced_targets in helper["wrapper_reference_targets"].items():
        if not wrapper_intent:
            continue
        if not referenced_targets or source_path not in best_chunk_by_path:
            continue
        reference_bonus = min(
            120.0,
            max(best_by_path.get(target, 0.0) * 0.9 for target in referenced_targets),
        )
        reference_bonus *= wrapper_source_multiplier_by_path.get(source_path, 1.0)
        raw_scores[best_chunk_by_path[source_path]] += reference_bonus

    # Small implementation packages frequently split an interface and its
    # concrete backend across files.  Share a bounded portion of the strongest
    # sibling's score without allowing broad utility packages to flood the
    # candidate list.
    paths_by_directory = defaultdict(list)
    for path in best_chunk_by_path:
        paths_by_directory[helper["path_directory_by_path"].get(path, _path_parent_string(path))].append(path)
    for directory_paths in paths_by_directory.values():
        if len(directory_paths) < 2 or len(directory_paths) > 6:
            continue
        strongest_path = max(directory_paths, key=lambda path: best_by_path.get(path, 0.0))
        sibling_bonus = min(25.0, best_by_path.get(strongest_path, 0.0) * 0.22)
        for path in directory_paths:
            if path == strongest_path or best_by_path.get(path, 0.0) <= 0:
                continue
            raw_scores[best_chunk_by_path[path]] += sibling_bonus

    ranked_all = sorted(raw_scores.items(), key=lambda item: (-item[1], item[0]))
    ranked = []
    selected_ids = set()
    seen_paths = set()
    for chunk_id, score in ranked_all:
        path = chunk_path_by_id.get(chunk_id, "")
        if path in seen_paths:
            continue
        seen_paths.add(path)
        selected_ids.add(chunk_id)
        ranked.append((chunk_id, score))
        if len(ranked) >= limit:
            break
    if len(ranked) < limit:
        for chunk_id, score in ranked_all:
            if chunk_id in selected_ids:
                continue
            ranked.append((chunk_id, score))
            if len(ranked) >= limit:
                break
    results = [
        {"chunk_id": chunk_id, "score": round(score, 8), "rank": rank, "source": "lexical"}
        for rank, (chunk_id, score) in enumerate(ranked, start=1)
    ]
    _LEXICAL_RESULT_CACHE[result_cache_key] = [dict(item) for item in results]
    while len(_LEXICAL_RESULT_CACHE) > LEXICAL_RESULT_CACHE_LIMIT:
        oldest_key = next(iter(_LEXICAL_RESULT_CACHE))
        _LEXICAL_RESULT_CACHE.pop(oldest_key, None)
    return results
