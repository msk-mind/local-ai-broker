"""Lightweight repo-state and cache helpers for warm inspect_repo queries.

This module intentionally excludes chunk building, lexical indexing, and other
heavy retrieval helpers so the persisted query-stage cache path can make its
hit/miss decision without importing the full inspection implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import contextlib
import fnmatch
import shutil
import time
from collections import defaultdict
from pathlib import Path, PurePosixPath

from inspection_contract import VALID_MODES, validate_request


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

INDEX_SCHEMA_VERSION = "repo-inspection-chunks-v2"
FILE_CHUNK_WORKING_MANIFEST_SCHEMA = "repo-inspection-file-chunk-manifest-v2"
FILE_CHUNK_SNAPSHOT_METADATA_SCHEMA = "repo-inspection-file-chunk-snapshot-metadata-v1"
GIT_FINGERPRINT_MANIFEST_SCHEMA = "repo-inspection-git-fingerprint-v1"
METADATA_FINGERPRINT_MANIFEST_SCHEMA = "repo-inspection-metadata-fingerprint-v1"
QUERY_STAGE_CACHE_SCHEMA = "repo-inspection-query-stage-cache-v1"
QUERY_STAGE_CACHE_ALIAS_SCHEMA = "repo-inspection-query-stage-cache-alias-v1"
REPOSITORY_FINGERPRINT_ALIAS_SCHEMA = "repo-inspection-repository-fingerprint-alias-v1"
_SNAPSHOT_METADATA_CACHE = {}
_SNAPSHOT_METADATA_CACHE_LIMIT = 64
_QUERY_STAGE_MEMORY_CACHE = {}
_QUERY_STAGE_ALIAS_MEMORY_CACHE = {}
_QUERY_STAGE_MEMORY_CACHE_LIMIT = 128
_REPOSITORY_FINGERPRINT_ALIAS_MEMORY_CACHE = {}
_REPOSITORY_FINGERPRINT_ALIAS_MEMORY_CACHE_LIMIT = 256
_SNAPSHOT_BUILD_CONTEXT_CACHE = {}
_SNAPSHOT_BUILD_CONTEXT_CACHE_LIMIT = 64
_PREFETCH_STATE_CACHE = {}
_PREFETCH_STATE_CACHE_LIMIT = 64
_PREFETCH_REPO_STATE_CACHE = {}
_PREFETCH_REPO_STATE_CACHE_LIMIT = 64
_GIT_TOP_CACHE = {}
_GIT_TOP_CACHE_LIMIT = 256
_DEFAULT_GIT_PROBE_CACHE = {}
_PRIVATE_CACHE_DIR_READY = set()
_GPU_QUERY_STAGE_SYMBOLS = None
_METADATA_REPOSITORY_STATE_CACHE = {}
_METADATA_REPOSITORY_STATE_CACHE_LIMIT = 64
_PROCESS_CACHE_CONTAINERS = (
    _QUERY_STAGE_MEMORY_CACHE, _QUERY_STAGE_ALIAS_MEMORY_CACHE,
    _REPOSITORY_FINGERPRINT_ALIAS_MEMORY_CACHE, _SNAPSHOT_BUILD_CONTEXT_CACHE,
    _PREFETCH_STATE_CACHE, _PREFETCH_REPO_STATE_CACHE, _GIT_TOP_CACHE,
    _DEFAULT_GIT_PROBE_CACHE, _PRIVATE_CACHE_DIR_READY,
    _METADATA_REPOSITORY_STATE_CACHE,
)


def reset_process_caches():
    """Clear process-local hot-path caches between isolated test runs."""
    global _GPU_QUERY_STAGE_SYMBOLS
    for value in _PROCESS_CACHE_CONTAINERS:
        value.clear()
    _GPU_QUERY_STAGE_SYMBOLS = None


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8", errors="replace"))


def estimate_tokens(value: str) -> int:
    return max(1, (len(value) + 2) // 3)


def _directory_state_signature(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError:
        return "missing"
    return f"dir:{int(stat.st_mtime_ns)}:{int(getattr(stat, 'st_ctime_ns', 0))}"


def _metadata_repository_state_cache_key(path: Path, ignored: set[str], ignored_paths=None):
    root = (path if path.is_dir() else path.parent).resolve(strict=False)
    relative_ignored_paths = sorted(
        str(Path(candidate).resolve(strict=False).relative_to(root))
        for candidate in (ignored_paths or ())
        if root in Path(candidate).resolve(strict=False).parents
    )
    return (
        str(root),
        tuple(sorted(str(name) for name in ignored)),
        tuple(relative_ignored_paths),
    )


def _prune_metadata_repository_state_cache():
    while len(_METADATA_REPOSITORY_STATE_CACHE) > _METADATA_REPOSITORY_STATE_CACHE_LIMIT:
        _METADATA_REPOSITORY_STATE_CACHE.pop(next(iter(_METADATA_REPOSITORY_STATE_CACHE)))


def _cached_metadata_repository_state(path: Path, ignored: set[str], ignored_paths=None):
    cache_key = _metadata_repository_state_cache_key(path, ignored, ignored_paths=ignored_paths)
    cached = _METADATA_REPOSITORY_STATE_CACHE.get(cache_key)
    if not isinstance(cached, dict):
        return None
    file_signatures = cached.get("file_signatures")
    directory_signatures = cached.get("directory_signatures")
    if not isinstance(file_signatures, dict) or not isinstance(directory_signatures, dict):
        return None
    for directory, signature in directory_signatures.items():
        if _directory_state_signature(Path(directory)) != str(signature or ""):
            return None
    for candidate, signature in file_signatures.items():
        if _file_state_signature(Path(candidate)) != str(signature or ""):
            return None
    _METADATA_REPOSITORY_STATE_CACHE.pop(cache_key, None)
    _METADATA_REPOSITORY_STATE_CACHE[cache_key] = cached
    state = cached.get("state")
    if not isinstance(state, dict):
        return None
    return dict(state)


def _store_metadata_repository_state(path: Path, ignored: set[str], ignored_paths=None, *, candidates, state):
    cache_key = _metadata_repository_state_cache_key(path, ignored, ignored_paths=ignored_paths)
    file_signatures = {}
    directory_paths = set()
    root = (path if path.is_dir() else path.parent).resolve(strict=False)
    directory_paths.add(str(root))
    for candidate in candidates or ():
        candidate = Path(candidate).resolve(strict=False)
        if not candidate.exists() or not candidate.is_file():
            continue
        file_signatures[str(candidate)] = _file_state_signature(candidate)
        parent = candidate.parent
        while True:
            directory_paths.add(str(parent))
            if parent == root:
                break
            next_parent = parent.parent
            if next_parent == parent:
                break
            parent = next_parent
    directory_signatures = {
        directory: _directory_state_signature(Path(directory))
        for directory in sorted(directory_paths)
    }
    _METADATA_REPOSITORY_STATE_CACHE[cache_key] = {
        "state": dict(state or {}),
        "file_signatures": file_signatures,
        "directory_signatures": directory_signatures,
    }
    _prune_metadata_repository_state_cache()

def _positive_int(value, default, name):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive token count")
    return parsed


def normalize_token_budgets(constraints):
    constraints = constraints or {}
    return {
        "retrieval_token_budget": _positive_int(
            constraints.get("retrieval_token_budget", constraints.get("retrieved_chunk_budget")),
            32_000,
            "retrieval_token_budget",
        ),
        "evidence_token_budget": _positive_int(
            constraints.get("evidence_token_budget", constraints.get("final_evidence_pack_budget")),
            12_000,
            "evidence_token_budget",
        ),
        "final_pack_token_budget": _positive_int(
            constraints.get("final_pack_token_budget", constraints.get("final_evidence_pack_budget")),
            8_000,
            "final_pack_token_budget",
        ),
        "synthesis_context_token_budget": _positive_int(
            constraints.get("synthesis_context_token_budget", constraints.get("remote_model_context_budget")),
            16_000,
            "synthesis_context_token_budget",
        ),
    }


def _gpu_query_stage_symbols():
    global _GPU_QUERY_STAGE_SYMBOLS
    if _GPU_QUERY_STAGE_SYMBOLS is None:
        from gpu_client import select_endpoint, services_from_execution_plan

        _GPU_QUERY_STAGE_SYMBOLS = {
            "select_endpoint": select_endpoint,
            "services_from_execution_plan": services_from_execution_plan,
        }
    return _GPU_QUERY_STAGE_SYMBOLS


def _gpu_query_stage(name):
    return _gpu_query_stage_symbols()[name]


def _gpu_registry_configured(execution_plan):
    execution_plan = execution_plan or {}
    registry_path = execution_plan.get("gpu_service_registry_path") or os.environ.get(
        "BROKER_GPU_SERVICE_REGISTRY_PATH"
    )
    return bool(str(registry_path or "").strip())


def _query_stage_retrieval_signatures_for_prefetch(execution_plan, task_params=None):
    lexical_fallback = {"tier": "cpu-lexical-fallback", "mode": "lexical_fallback"}
    execution_plan = execution_plan or {}
    if not _gpu_registry_configured(execution_plan):
        return [lexical_fallback], lexical_fallback
    health_interval_seconds = execution_plan.get("gpu_service_health_interval_seconds") or os.environ.get(
        "BROKER_GPU_SERVICE_HEALTH_INTERVAL_SECONDS"
    )
    if health_interval_seconds not in (None, ""):
        try:
            health_interval_seconds = float(health_interval_seconds)
        except (TypeError, ValueError):
            health_interval_seconds = None
    services = list(_gpu_query_stage("services_from_execution_plan")(execution_plan, task_params))
    if not services:
        return [lexical_fallback], lexical_fallback
    endpoint = _gpu_query_stage("select_endpoint")(
        services,
        "p40-retrieval",
        "search",
        expected_gpu_count=1,
        health_interval_seconds=health_interval_seconds,
    )
    rerank_endpoint = _gpu_query_stage("select_endpoint")(
        services,
        "p40-retrieval",
        "rerank",
        expected_gpu_count=1,
        health_interval_seconds=health_interval_seconds,
    )
    if endpoint is None or rerank_endpoint is None:
        return [lexical_fallback], lexical_fallback
    signature = {
        "tier": str(endpoint.get("tier") or ""),
        "model_profile": str(endpoint.get("model_profile") or ""),
        "model": str(endpoint.get("model") or ""),
        "mode": "gpu",
    }
    return [signature, lexical_fallback], signature


def cache_dir_for_execution(execution_plan, output_dir, task_params=None):
    execution_plan = execution_plan or {}
    task_params = task_params or {}
    configured = (
        execution_plan.get("repo_inspection_cache_path")
        or task_params.get("index_cache_dir")
        or os.environ.get("BROKER_REPO_INSPECTION_CACHE_DIR")
    )
    if bool(execution_plan.get("repo_inspection_use_node_local_cache")):
        for env_name in ("TMPDIR", "TMP", "TEMP"):
            scratch_root = os.environ.get(env_name, "").strip()
            if not scratch_root:
                continue
            cache_token = str(
                (execution_plan or {}).get("repo_inspection_node_local_cache_namespace")
                or (execution_plan or {}).get("job_id")
                or Path(output_dir or ".").name
                or "job"
            )
            return Path(scratch_root).expanduser().resolve(strict=False) / "local-ai-broker" / "inspect-repo" / cache_token
        import tempfile

        scratch_root = tempfile.gettempdir()
        if scratch_root:
            cache_token = str(
                (execution_plan or {}).get("repo_inspection_node_local_cache_namespace")
                or (execution_plan or {}).get("job_id")
                or Path(output_dir or ".").name
                or "job"
            )
            return Path(scratch_root).expanduser().resolve(strict=False) / "local-ai-broker" / "inspect-repo" / cache_token
    if configured:
        return Path(configured).expanduser().resolve(strict=False)
    return Path(output_dir or ".").expanduser().resolve(strict=False) / "repo-inspection-v2-cache"


def cache_runtime_diagnostics(execution_plan, output_dir, cache_dir):
    execution_plan = execution_plan or {}
    cache_path = Path(cache_dir).expanduser().resolve(strict=False)
    node_local_requested = bool(execution_plan.get("repo_inspection_use_node_local_cache"))
    selected_tmp_env = ""
    tmp_root = None
    for env_name in ("TMPDIR", "TMP", "TEMP"):
        candidate = os.environ.get(env_name, "").strip()
        if candidate:
            selected_tmp_env = env_name
            tmp_root = Path(candidate).expanduser().resolve(strict=False)
            break
    if node_local_requested and tmp_root is None:
        import tempfile

        fallback_tmp = tempfile.gettempdir()
        if fallback_tmp:
            selected_tmp_env = "python_tempdir"
            tmp_root = Path(fallback_tmp).expanduser().resolve(strict=False)
    origin = "output_dir_default"
    if node_local_requested and tmp_root is not None:
        origin = "node_local_tmpdir"
    elif execution_plan.get("repo_inspection_cache_path") or os.environ.get("BROKER_REPO_INSPECTION_CACHE_DIR"):
        origin = "configured"
    diagnostics = {
        "local_cache_path": str(cache_path),
        "local_cache_origin": origin,
        "node_local_cache_requested": node_local_requested,
        "node_local_cache_selected": bool(origin == "node_local_tmpdir"),
    }
    if selected_tmp_env:
        diagnostics["tmp_env"] = selected_tmp_env
    if tmp_root is not None:
        diagnostics["tmp_root"] = str(tmp_root)
    shared_cache = os.environ.get("BROKER_REPO_INSPECTION_SHARED_CACHE_DIR", "").strip()
    if shared_cache:
        diagnostics["shared_cache_path"] = str(Path(shared_cache).expanduser().resolve(strict=False))
    if output_dir is not None:
        diagnostics["output_dir"] = str(Path(output_dir).expanduser().resolve(strict=False))
    return diagnostics


def exclusion_paths_for_execution(execution_plan, output_dir, task_params=None):
    cache_dir = cache_dir_for_execution(execution_plan, output_dir, task_params=task_params)
    excluded_paths = {cache_dir}
    if output_dir is not None:
        excluded_paths.add(Path(output_dir).expanduser().resolve(strict=False))
    return excluded_paths


def transient_excluded_paths_for_execution(output_dir):
    if output_dir is None:
        return set()
    return {Path(output_dir).expanduser().resolve(strict=False)}


def has_custom_exclusions(task_params):
    task_params = task_params or {}
    for key in ("excluded_dir_names", "exclude_dirs", "excluded_paths"):
        value = task_params.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return True
            continue
        if isinstance(value, (list, tuple, set, frozenset, dict)):
            if len(value) > 0:
                return True
    return False


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
        resolved = path.resolve(strict=False)
        for ignored_path in ignored_paths:
            if ignored_path is None:
                continue
            if resolved == ignored_path or ignored_path in resolved.parents:
                return True
    return False


def _input_manifest_repository_state_fingerprint(discovered, task_params):
    if has_custom_exclusions(task_params):
        return "", ""
    if len(discovered or ()) != 1:
        return "", ""
    item = discovered[0] or {}
    input_type = str(item.get("type") or "").strip().lower()
    if input_type not in {"repo", "directory"}:
        return "", ""
    fingerprint = str(item.get("content_hash") or "").strip()
    if not fingerprint:
        return "", ""
    return fingerprint, "input_manifest"


def fingerprint_hint_state(fingerprint: str, source: str, *, kind: str = "broker_hint"):
    fingerprint = str(fingerprint or "").strip()
    source = str(source or "").strip()
    kind = str(kind or "broker_hint").strip() or "broker_hint"
    if not fingerprint:
        return []
    return [
        {
            "kind": kind,
            "source": source,
            "fingerprint": fingerprint,
        }
    ]


def _fingerprint_state_is_hint_only(fingerprint_state):
    if not isinstance(fingerprint_state, (list, tuple)) or not fingerprint_state:
        return False
    saw_hint = False
    for item in fingerprint_state:
        if not isinstance(item, dict):
            return False
        kind = str(item.get("kind") or "").strip()
        if kind in {"broker_hint", "input_manifest"}:
            saw_hint = True
            continue
        return False
    return saw_hint


def fingerprint_alias_state(fingerprint: str, source: str):
    fingerprint = str(fingerprint or "").strip()
    source = str(source or "").strip() or "alias"
    if not fingerprint:
        return []
    return [
        {
            "kind": "broker_hint_alias",
            "source": source,
            "fingerprint": fingerprint,
        }
    ]


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


def _run_git(root: Path, *args, text=False):
    import subprocess

    return subprocess.run(
        ["git", "-C", str(root), "--no-optional-locks", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=20,
        text=text,
    ).stdout


def _run_git_optional(root: Path, *args, text=False):
    try:
        return _run_git(root, *args, text=text)
    except Exception:
        return None


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
        except Exception:
            resolved = None
    _GIT_TOP_CACHE[cache_key] = "" if resolved is None else str(resolved)
    while len(_GIT_TOP_CACHE) > _GIT_TOP_CACHE_LIMIT:
        oldest_key = next(iter(_GIT_TOP_CACHE))
        _GIT_TOP_CACHE.pop(oldest_key, None)
    return resolved


def _git_scope_head_oid(top: Path, scope_rel: str):
    try:
        if scope_rel in {"", "."}:
            return _run_git(top, "rev-parse", "HEAD^{tree}", text=True).strip()
        return _run_git(top, "rev-parse", f"HEAD:{scope_rel}", text=True).strip()
    except Exception:
        return None


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


def _invalidate_git_probe_worktree_caches(git_probe_cache):
    if not isinstance(git_probe_cache, dict):
        return
    for key in (
        "scoped_status_output",
        "scoped_clean_probe",
    ):
        git_probe_cache.pop(key, None)


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
        try:
            identity = _run_git(top, "rev-parse", "HEAD^{tree}", text=True).strip()
        except Exception:
            identity = "unborn"
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


def _scoped_git_status_output(top: Path, normalized_scope_paths, *, git_probe_cache=None):
    cache_key = (str(top.resolve(strict=False)), tuple(normalized_scope_paths))
    bucket = _git_probe_cache_bucket(git_probe_cache, "scoped_status_output")
    if bucket is not None and cache_key in bucket:
        return bucket[cache_key]
    args = [
        "--no-optional-locks",
        "status",
        "--porcelain=v1",
        "--no-renames",
        "-z",
        "--untracked-files=all",
        "--ignored=no",
        *_scope_pathspec(normalized_scope_paths),
    ]
    output = _run_git_optional(top, *args)
    if bucket is not None:
        bucket[cache_key] = output
    return output


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
    status_output = _scoped_git_status_output(
        top,
        normalized_scope_paths,
        git_probe_cache=git_probe_cache,
    )
    if status_output is None:
        return record(None)
    _status_digest, filtered_status = _filtered_git_status_digest(
        status_output,
        top,
        ignored or set(),
        ignored_paths=ignored_paths,
    )
    return record(not filtered_status)


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


def _filtered_git_status_digest(status_output: bytes, top: Path, ignored: set[str], ignored_paths=None):
    kept = []
    for entry in _parse_git_status_entries(status_output):
        code = entry["code"]
        relevant_paths = []
        for raw_path in entry["paths"]:
            candidate = (top / raw_path).resolve(strict=False)
            if _should_skip(candidate, top, ignored, ignored_paths=ignored_paths):
                continue
            relevant_paths.append(raw_path)
        if relevant_paths:
            kept.append((code, tuple(relevant_paths)))
    encoded = json.dumps(kept, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_text(encoded), kept


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


def _load_git_fingerprint_manifest(path: Path):
    if not path.exists():
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
    return {
        "head": str(payload.get("head") or ""),
        "status": str(payload.get("status") or ""),
        "state": state,
    }


def _write_git_fingerprint_manifest(path: Path, *, head: str, status: str, state):
    _atomic_private_json(
        path,
        {
            "schema": GIT_FINGERPRINT_MANIFEST_SCHEMA,
            "head": str(head),
            "status": str(status),
            "state": state,
        },
    )


def _git_fingerprint_manifest_equals(cached, *, head: str, status: str, state) -> bool:
    if not isinstance(cached, dict):
        return False
    if str(cached.get("head") or "") != str(head):
        return False
    if str(cached.get("status") or "") != str(status):
        return False
    return dict(cached.get("state") or {}) == dict(state or {})


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
        normalized[str(path)] = {
            "state_signature": state_signature,
            "content_signature": content_signature,
        }
    return normalized


def _cached_worktree_signature_map(*states):
    for state in states:
        if not isinstance(state, dict):
            continue
        cache = _normalize_worktree_signature_cache(state.get("worktree_signatures"))
        if cache:
            return cache
    return {}


def _worktree_signature_with_cache(path: Path, cached_record=None):
    if path.is_symlink():
        return "symlink", None
    if not path.exists():
        return "missing", None
    if not path.is_file():
        return "nonfile", None
    state_signature = _file_state_signature(path)
    if isinstance(cached_record, dict):
        cached_state_signature = str(cached_record.get("state_signature") or "")
        cached_content_signature = str(cached_record.get("content_signature") or "")
        if (
            cached_state_signature
            and cached_content_signature
            and state_signature == cached_state_signature
        ):
            return cached_content_signature, state_signature
    return _worktree_content_signature(path), state_signature


def _file_state_signature(path: Path, git_signature=None):
    if git_signature:
        return str(git_signature)
    try:
        stat = path.stat()
    except OSError:
        return "unreadable"
    return f"meta:{int(stat.st_size)}:{int(stat.st_mtime_ns)}:{int(getattr(stat, 'st_ctime_ns', 0))}"


def _git_fingerprint(root: Path, ignored: set[str], ignored_paths=None, scope_paths=None, cache_dir=None, git_probe_cache=None):
    top = _git_top(root)
    if top is None:
        return None
    normalized_scope_paths = _normalize_scope_paths(top, scope_paths)
    head = _scoped_git_head_identity(top, normalized_scope_paths, git_probe_cache=git_probe_cache)
    manifest_path = _git_fingerprint_manifest_path(Path(cache_dir), top, normalized_scope_paths) if cache_dir is not None else None
    shared_manifest_path = _shared_git_fingerprint_manifest_path(top, normalized_scope_paths, create=False)
    cached = _load_git_fingerprint_manifest(manifest_path) if manifest_path is not None else None
    shared_cached = _load_git_fingerprint_manifest(shared_manifest_path) if shared_manifest_path is not None else None
    cached_clean_state = _clean_git_fingerprint_manifest(cached, head=head)
    shared_clean_state = _clean_git_fingerprint_manifest(shared_cached, head=head)
    status_output = None
    if cached_clean_state is not None or shared_clean_state is not None:
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
            return dict(shared_clean_state)
        if clean_probe is None:
            return None
        status_output = _scoped_git_status_output(top, normalized_scope_paths, git_probe_cache=git_probe_cache)
        if status_output is None:
            return None
    pathspec = _scope_pathspec(normalized_scope_paths)
    if status_output is None:
        status_output = _scoped_git_status_output(top, normalized_scope_paths, git_probe_cache=git_probe_cache)
    if status_output is None:
        return None
    if not status_output or (isinstance(status_output, (bytes, str)) and not status_output.strip()):
        empty_status_digest = _empty_git_status_digest()
        if cached and cached.get("head") == head and cached.get("status") == empty_status_digest:
            return dict(cached["state"])
        if shared_cached and shared_cached.get("head") == head and shared_cached.get("status") == empty_status_digest:
            return dict(shared_cached["state"])
    status_digest, filtered_status = _filtered_git_status_digest(status_output, top, ignored, ignored_paths=ignored_paths)
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
    parsed_status = _parse_git_status_entries(status_output)
    relevant_entries = []
    tracked_index_paths = set()
    for entry in parsed_status:
        relevant_paths = []
        for raw_path in entry["paths"]:
            candidate = (top / raw_path).resolve(strict=False)
            if _should_skip(candidate, top, ignored, ignored_paths=ignored_paths):
                continue
            relevant_paths.append(raw_path)
        if not relevant_paths:
            continue
        normalized = {"code": entry["code"], "paths": relevant_paths}
        relevant_entries.append(normalized)
        x_code = entry["code"][:1]
        y_code = entry["code"][1:2]
        if x_code not in {"", " ", "?"}:
            tracked_index_paths.add(relevant_paths[0])
        if x_code in {"R", "C"} and relevant_paths:
            tracked_index_paths.add(relevant_paths[-1])
        if y_code not in {"", " ", "?"} and relevant_paths:
            tracked_index_paths.add(relevant_paths[-1])

    tracked_index = _git_index_blob_oids(top, tracked_index_paths)
    if tracked_index is None:
        return None

    staged_entries = []
    unstaged_entries = []
    untracked = []
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
            content_signature, state_signature = _worktree_signature_with_cache(
                candidate,
                worktree_signature_cache.get(destination),
            )
            if state_signature:
                next_worktree_signature_cache[destination] = {
                    "state_signature": state_signature,
                    "content_signature": content_signature,
                }
            untracked.append((destination, content_signature))
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
        if y_code not in {"", " "}:
            candidate = top / destination
            content_signature, state_signature = _worktree_signature_with_cache(
                candidate,
                worktree_signature_cache.get(destination),
            )
            if state_signature:
                next_worktree_signature_cache[destination] = {
                    "state_signature": state_signature,
                    "content_signature": content_signature,
                }
            unstaged_entries.append(
                {
                    "code": y_code,
                    "path": destination,
                    "signature": content_signature,
                }
            )
    state = {
        "kind": "git",
        "head": head,
        "staged": f"sha256:{sha256_text(json.dumps(sorted(staged_entries, key=lambda item: (item.get('path', ''), item.get('source_path', ''), item.get('code', ''))), sort_keys=True, separators=(',', ':'), ensure_ascii=True))}",
        "unstaged": f"sha256:{sha256_text(json.dumps(sorted(unstaged_entries, key=lambda item: (item.get('path', ''), item.get('code', ''))), sort_keys=True, separators=(',', ':'), ensure_ascii=True))}",
        "untracked": sorted(untracked),
    }
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
    if manifest_path is not None and not _git_fingerprint_manifest_equals(cached, head=head, status=status_digest, state=state):
        _write_git_fingerprint_manifest(manifest_path, head=head, status=status_digest, state=state)
    shared_manifest_write_path = _shared_git_fingerprint_manifest_path(top, normalized_scope_paths, create=True)
    if shared_manifest_write_path is not None and not _git_fingerprint_manifest_equals(
        shared_cached, head=head, status=status_digest, state=state
    ):
        _write_git_fingerprint_manifest(shared_manifest_write_path, head=head, status=status_digest, state=state)
    return state


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


def _rg_file_list(root: Path, ignored: set[str], ignored_paths=None):
    import subprocess

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


def _git_file_list(root: Path, ignored: set[str], ignored_paths=None):
    import subprocess

    top = _git_top(root)
    if top is None:
        return None
    try:
        scope_rel = root.resolve(strict=False).relative_to(top).as_posix()
    except ValueError:
        return None
    args = ["git", "-C", str(top), "--no-optional-locks", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    if scope_rel not in {"", "."}:
        args.extend(["--", scope_rel])
    try:
        output = subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            text=False,
        ).stdout
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    candidates = []
    for raw_name in output.split(b"\0"):
        if not raw_name:
            continue
        rel = raw_name.decode("utf-8", errors="surrogateescape")
        candidate = (top / rel).resolve(strict=False)
        if candidate == root.resolve(strict=False) or root.resolve(strict=False) in candidate.parents:
            if not _should_skip(candidate, root, ignored, ignored_paths=ignored_paths):
                candidates.append(candidate)
    return candidates


def _iter_source_candidates(root: Path, ignored: set[str], ignored_paths=None):
    candidates = _git_file_list(root, ignored, ignored_paths=ignored_paths)
    if candidates is None:
        candidates = _rg_file_list(root, ignored, ignored_paths=ignored_paths)
    if candidates is None:
        candidates = sorted(root.rglob("*"))
    for candidate in candidates:
        if _should_skip(candidate, root, ignored, ignored_paths=ignored_paths) or candidate.is_symlink() or not candidate.is_file():
            continue
        yield candidate


def _metadata_fingerprint(path: Path, ignored: set[str], ignored_paths=None, cache_dir=None, git_probe_cache=None):
    root = path if path.is_dir() else path.parent
    cached_state = _cached_metadata_repository_state(root, ignored, ignored_paths=ignored_paths)
    if cached_state is not None:
        return cached_state
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
    state = {"kind": "metadata", "entries": entries}
    _store_metadata_repository_state(root, ignored, ignored_paths=ignored_paths, candidates=candidates, state=state)
    return state


def repository_fingerprint(discovered, excluded_dir_names=None, excluded_paths=None, cache_dir=None, git_probe_cache=None):
    import inspection_index

    git_probe_cache = _effective_git_probe_cache(git_probe_cache)
    return inspection_index.repository_fingerprint(
        discovered,
        excluded_dir_names=excluded_dir_names,
        excluded_paths=excluded_paths,
        cache_dir=cache_dir,
        git_probe_cache=git_probe_cache,
    )


def _file_chunk_manifest_path(cache_dir: Path):
    return _private_cache_dir(Path(cache_dir)) / "file-chunk-working-manifest.json"


def _file_chunk_snapshot_metadata_path(cache_dir: Path):
    return _private_cache_dir(Path(cache_dir)) / "file-chunk-working-snapshot-metadata.json"


def _shared_repo_inspection_cache_root(create=False):
    configured = os.environ.get("BROKER_REPO_INSPECTION_SHARED_CACHE_DIR", "").strip()
    if not configured:
        return None
    root = Path(configured).expanduser()
    try:
        return _private_cache_dir(root) if create else root.expanduser().absolute()
    except OSError:
        return None


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


def _load_file_chunk_working_manifest(path: Path):
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
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
    return {
        "repository_state_fingerprint": str(payload.get("repository_state_fingerprint") or ""),
        "build_config_digest": str(payload.get("build_config_digest") or ""),
        "files": normalized,
    }


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
    }


def _snapshot_metadata_cache_key(path: Path):
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path), int(stat.st_mtime_ns), int(stat.st_size))


def _clone_snapshot_metadata(metadata):
    if metadata is None:
        return None
    return {
        "repository_state_fingerprint": str(metadata.get("repository_state_fingerprint") or ""),
        "build_config_digest": str(metadata.get("build_config_digest") or ""),
        "index_manifest": dict(metadata.get("index_manifest") or {}),
        "semantic_document_signatures": dict(metadata.get("semantic_document_signatures") or {}),
        "chunk_ids": [str(value) for value in (metadata.get("chunk_ids") or ())],
        "chunk_count": int(metadata.get("chunk_count") or 0),
        "total_files": int(metadata.get("total_files") or 0),
    }


def _cache_snapshot_metadata(path: Path, metadata):
    cache_key = _snapshot_metadata_cache_key(path)
    if cache_key is None:
        return
    _SNAPSHOT_METADATA_CACHE.pop(cache_key, None)
    _SNAPSHOT_METADATA_CACHE[cache_key] = _clone_snapshot_metadata(metadata)
    while len(_SNAPSHOT_METADATA_CACHE) > _SNAPSHOT_METADATA_CACHE_LIMIT:
        _SNAPSHOT_METADATA_CACHE.pop(next(iter(_SNAPSHOT_METADATA_CACHE)))


def _chunk_build_config_digest(
    normalized_inputs,
    *,
    max_lines,
    overlap,
    excluded_dir_names,
    normalized_excluded_paths,
):
    payload = {
        "schema": FILE_CHUNK_WORKING_MANIFEST_SCHEMA,
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "max_lines": int(max_lines),
        "overlap": int(overlap),
        "excluded_dir_names": sorted(str(item) for item in (excluded_dir_names or ())),
        "excluded_paths": sorted(str(path) for path in (normalized_excluded_paths or ())),
        "inputs": list(normalized_inputs),
    }
    return f"sha256:{sha256_text(json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True))}"


def _stable_build_context_inputs(discovered, namespaces):
    discovered_roots = []
    for item in discovered:
        path = item.get("path")
        if path is None:
            continue
        try:
            discovered_roots.append((str(namespaces.get(id(item), "")), Path(path).expanduser().resolve(strict=False)))
        except OSError:
            continue

    normalized_inputs = []
    normalized_input_signatures = []
    for item in discovered:
        uri = str(item.get("uri") or "").strip()
        content_hash = str(item.get("content_hash") or "").strip()
        path = item.get("path")
        if uri:
            locator = {"uri": uri}
        elif path is not None:
            locator = {"path": str(Path(path).expanduser().resolve(strict=False))}
        elif content_hash:
            locator = {"content_hash": content_hash}
        else:
            locator = {"path": ""}
        normalized_input = {
            "id": str(item.get("id", "")),
            "type": str(item.get("type", "")),
            "classification": str(item.get("classification", "unknown")),
            "source_namespace": str(namespaces.get(id(item), "")),
            **locator,
        }
        normalized_inputs.append(normalized_input)
        normalized_input_signatures.append(
            json.dumps(normalized_input, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        )

    def normalize_excluded_paths(paths):
        normalized = []
        for path in sorted((Path(path).expanduser().resolve(strict=False) for path in (paths or ())), key=str):
            matches = []
            for namespace, root in discovered_roots:
                try:
                    rel = path.relative_to(root).as_posix()
                except ValueError:
                    continue
                matches.append({"source_namespace": namespace, "relative_path": rel or "."})
            if matches:
                normalized.extend(sorted(matches, key=lambda item: (item["source_namespace"], item["relative_path"])))
        return normalized

    return normalized_inputs, tuple(normalized_input_signatures), normalize_excluded_paths


def _prune_snapshot_build_context_cache():
    if len(_SNAPSHOT_BUILD_CONTEXT_CACHE) <= _SNAPSHOT_BUILD_CONTEXT_CACHE_LIMIT:
        return
    while len(_SNAPSHOT_BUILD_CONTEXT_CACHE) > _SNAPSHOT_BUILD_CONTEXT_CACHE_LIMIT:
        _SNAPSHOT_BUILD_CONTEXT_CACHE.pop(next(iter(_SNAPSHOT_BUILD_CONTEXT_CACHE)))


def _prune_prefetch_state_cache():
    if len(_PREFETCH_STATE_CACHE) <= _PREFETCH_STATE_CACHE_LIMIT:
        return
    while len(_PREFETCH_STATE_CACHE) > _PREFETCH_STATE_CACHE_LIMIT:
        _PREFETCH_STATE_CACHE.pop(next(iter(_PREFETCH_STATE_CACHE)))


def _prune_prefetch_repo_state_cache():
    if len(_PREFETCH_REPO_STATE_CACHE) <= _PREFETCH_REPO_STATE_CACHE_LIMIT:
        return
    while len(_PREFETCH_REPO_STATE_CACHE) > _PREFETCH_REPO_STATE_CACHE_LIMIT:
        _PREFETCH_REPO_STATE_CACHE.pop(next(iter(_PREFETCH_REPO_STATE_CACHE)))


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
    def _resolved_path(path):
        if path is None:
            return None
        if isinstance(path, Path):
            return path.expanduser().resolve(strict=False)
        return Path(path).expanduser().resolve(strict=False)

    discovered = list(discovered)
    namespaces = {}
    used_namespaces = set()
    for index, item in enumerate(discovered):
        raw_namespace = str(item.get("id") or f"input_{index}")
        namespace = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_namespace).strip("._-") or f"input_{index}"
        item_uri = str(item.get("uri", ""))
        item_path = str(item.get("path", ""))
        if namespace in used_namespaces:
            source_key = f"{raw_namespace}\0{item_uri}\0{item_path}"
            namespace = f"{namespace}_{sha256_text(source_key)[:8]}"
        used_namespaces.add(namespace)
        namespaces[id(item)] = namespace
    normalized_inputs, normalized_input_signatures, normalize_excluded_paths = _stable_build_context_inputs(
        discovered,
        namespaces,
    )
    configured_excluded_paths = {resolved for resolved in (_resolved_path(path) for path in (excluded_paths or ())) if resolved is not None}
    resolved_cache_dir = _resolved_path(cache_dir)
    if resolved_cache_dir is not None:
        configured_excluded_paths.discard(resolved_cache_dir)
    resolved_transient_paths = {resolved for resolved in (_resolved_path(path) for path in (transient_excluded_paths or ())) if resolved is not None}
    for transient_path in resolved_transient_paths:
        configured_excluded_paths.discard(transient_path)
    effective_excluded_paths = set(configured_excluded_paths)
    if resolved_cache_dir is not None:
        effective_excluded_paths.add(resolved_cache_dir)
    effective_excluded_paths.update(resolved_transient_paths)
    normalized_excluded_path_records = tuple(
        (item["source_namespace"], item["relative_path"])
        for item in normalize_excluded_paths(configured_excluded_paths)
    )
    cache_key = (
        int(max_lines),
        int(overlap),
        tuple(sorted(str(item) for item in (excluded_dir_names or ()))),
        normalized_excluded_path_records,
        normalized_input_signatures,
    )
    cached = _SNAPSHOT_BUILD_CONTEXT_CACHE.get(cache_key)
    if cached is not None:
        _SNAPSHOT_BUILD_CONTEXT_CACHE.pop(cache_key, None)
        _SNAPSHOT_BUILD_CONTEXT_CACHE[cache_key] = cached
        return discovered, dict(cached["namespaces"]), set(cached["effective_excluded_paths"]), str(cached["build_config_digest"])
    build_config_digest = _chunk_build_config_digest(
        normalized_inputs,
        max_lines=max_lines,
        overlap=overlap,
        excluded_dir_names=excluded_dir_names,
        normalized_excluded_paths=normalized_excluded_path_records,
    )
    _SNAPSHOT_BUILD_CONTEXT_CACHE[cache_key] = {
        "namespaces": dict(namespaces),
        "effective_excluded_paths": tuple(effective_excluded_paths),
        "build_config_digest": str(build_config_digest),
    }
    _prune_snapshot_build_context_cache()
    return discovered, namespaces, set(effective_excluded_paths), build_config_digest


def _prefetch_state_cache_key(
    *,
    query,
    mode,
    budgets,
    cache_dir,
    repository_state_fingerprint,
    build_config_digest,
    retrieval_signature,
):
    return (
        str(query),
        str(mode),
        str(cache_dir),
        str(repository_state_fingerprint or ""),
        str(build_config_digest or ""),
        json.dumps(dict(retrieval_signature or {}), sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        int(budgets.get("retrieval_token_budget") or 0),
        int(budgets.get("evidence_token_budget") or 0),
        int(budgets.get("final_pack_token_budget") or 0),
        int(budgets.get("synthesis_context_token_budget") or 0),
    )


def _early_prefetch_state_cache_key(
    *,
    query,
    mode,
    budgets,
    cache_dir,
    excluded,
    discovered,
    retrieval_signature,
    hint_fingerprint,
    hint_source,
):
    normalized_inputs = []
    for item in list(discovered or ()):
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        resolved_path = ""
        if path is not None:
            resolved_path = str(Path(path).expanduser().resolve(strict=False))
        normalized_inputs.append(
            {
                "id": str(item.get("id", "")),
                "type": str(item.get("type", "")),
                "uri": str(item.get("uri", "")),
                "classification": str(item.get("classification", "unknown")),
                "path": resolved_path,
                "content_hash": str(item.get("content_hash") or ""),
            }
        )
    return (
        "early_prefetch",
        str(query),
        str(mode),
        str(Path(cache_dir).expanduser().resolve(strict=False)),
        tuple(sorted(str(item) for item in excluded)),
        json.dumps(normalized_inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        json.dumps(dict(retrieval_signature or {}), sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        str(hint_fingerprint or ""),
        str(hint_source or ""),
        int(budgets.get("retrieval_token_budget") or 0),
        int(budgets.get("evidence_token_budget") or 0),
        int(budgets.get("final_pack_token_budget") or 0),
        int(budgets.get("synthesis_context_token_budget") or 0),
    )


def _prefetch_repo_state_cache_key(
    *,
    cache_dir,
    repository_state_fingerprint,
    build_config_digest,
    discovered,
):
    input_identity = []
    for item in list(discovered or ()):
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        input_identity.append(
            (
                str(item.get("id") or ""),
                str(item.get("type") or ""),
                str(Path(path).expanduser().resolve(strict=False)) if path is not None else str(item.get("uri") or ""),
            )
        )
    return (
        "prefetch_repo_state",
        str(Path(cache_dir).expanduser().resolve(strict=False)),
        str(repository_state_fingerprint or ""),
        str(build_config_digest or ""),
        tuple(input_identity),
    )


def _clone_prefetched_metadata_chunks(metadata_chunks):
    if metadata_chunks is None:
        return None
    cloned = type("MetadataChunkList", (list,), {})()
    cloned._index_manifest = dict(getattr(metadata_chunks, "_index_manifest", {}) or {})
    cloned._semantic_document_signatures = dict(getattr(metadata_chunks, "_semantic_document_signatures", {}) or {})
    cloned._chunk_ids = tuple(getattr(metadata_chunks, "_chunk_ids", ()) or ())
    cloned._chunk_count = int(getattr(metadata_chunks, "_chunk_count", len(cloned._chunk_ids)) or 0)
    return cloned


def _clone_prefetched_state_payload(payload):
    cached_lexical_fallback_run = payload.get("cached_lexical_fallback_run")
    return {
        "cached_chunk_snapshot_metadata": _clone_snapshot_metadata(payload.get("cached_chunk_snapshot_metadata")),
        "metadata_chunks": _clone_prefetched_metadata_chunks(payload.get("metadata_chunks")),
        "fingerprint": str(payload.get("fingerprint") or ""),
        "build_config_digest": str(payload.get("build_config_digest") or ""),
        "cached_query_stage": _clone_query_stage_cache_payload(payload.get("cached_query_stage")),
        "prefetched_query_stage_cache_key": str(payload.get("prefetched_query_stage_cache_key") or ""),
        "prefetched_query_stage_cache_probed": bool(payload.get("prefetched_query_stage_cache_probed")),
        "prefetched_query_stage_requires_verification": bool(payload.get("prefetched_query_stage_requires_verification")),
        "prefetched_retrieval_signature": dict(payload.get("prefetched_retrieval_signature") or {}),
        "prefetch_state_source": str(payload.get("prefetch_state_source") or ""),
        "prefetch_state_cache_hit": bool(payload.get("prefetch_state_cache_hit")),
        "prefetch_stage_timings_ms": dict(payload.get("prefetch_stage_timings_ms") or {}),
        "cached_lexical_fallback_run": {
            "payload": dict((cached_lexical_fallback_run or {}).get("payload") or {}),
            "artifact_payloads": dict((cached_lexical_fallback_run or {}).get("artifact_payloads") or {}),
        }
        if isinstance(cached_lexical_fallback_run, dict)
        else None,
    }


def _clone_prefetch_repo_state_payload(payload):
    return {
        "cached_chunk_snapshot_metadata": _clone_snapshot_metadata(payload.get("cached_chunk_snapshot_metadata")),
        "metadata_chunks": _clone_prefetched_metadata_chunks(payload.get("metadata_chunks")),
        "fingerprint": str(payload.get("fingerprint") or ""),
        "build_config_digest": str(payload.get("build_config_digest") or ""),
        "prefetched_retrieval_signature": dict(payload.get("prefetched_retrieval_signature") or {}),
    }


def _attach_cached_lexical_fallback_run(state):
    if not isinstance(state, dict):
        return state
    try:
        cached_run = cached_lexical_fallback_from_context(state)
    except (KeyError, TypeError, ValueError):
        cached_run = None
    if cached_run is not None:
        state["cached_lexical_fallback_run"] = cached_run
    else:
        state.pop("cached_lexical_fallback_run", None)
    return state


def _mark_prefetch_state(state, source, *, cache_hit=False):
    if not isinstance(state, dict):
        return state
    state["prefetch_state_source"] = str(source or "")
    state["prefetch_state_cache_hit"] = bool(cache_hit)
    return state


def _prefetch_elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000.0, 3)


def _quality(result, retrieval, reranking, synthesis):
    return {
        "result": result,
        "retrieval": retrieval,
        "reranking": reranking,
        "synthesis": synthesis,
        "answer_ready": result == "answer_ready",
    }


def released_pack_tokens(payload):
    pack = {
        key: payload[key]
        for key in ("mode", "query", "answer", "findings", "evidence", "quality", "warnings", "provenance")
        if key in payload
    }
    return estimate_tokens(json.dumps(pack, ensure_ascii=True, separators=(",", ":")))


def trim_evidence_for_final_pack(evidence, base_payload, final_pack_token_budget, synthesis_reserve=0):
    kept = [dict(item) for item in evidence]
    target = max(0, int(final_pack_token_budget) - max(0, int(synthesis_reserve)))

    def pack_size(items):
        candidate = dict(base_payload)
        candidate["evidence"] = items
        return released_pack_tokens(candidate)

    minimum_excerpt_chars = 96
    while kept and pack_size(kept) > target:
        reducible = [
            (len(str(item.get("excerpt") or "")), index)
            for index, item in enumerate(kept)
            if len(str(item.get("excerpt") or "")) > minimum_excerpt_chars
        ]
        if reducible:
            current_length, index = max(reducible)
            excess_tokens = pack_size(kept) - target
            reduction = max(32, min(current_length - minimum_excerpt_chars, excess_tokens * 4 + 16))
            item = dict(kept[index])
            item["excerpt"] = str(item.get("excerpt") or "")[: current_length - reduction].rstrip() + "..."
            kept[index] = item
            continue
        kept.pop()
    return kept, len(kept) != len(evidence) or kept != evidence


def _artifact_payloads(payload, fingerprint, cache_dir, full_trace=False):
    artifacts = {
        "evidence_pack": {"query": payload["query"], "evidence": payload["evidence"]},
        "retrieval_result": {
            "fingerprint": fingerprint,
            "lexical_index": str(Path(cache_dir) / "lexical-working.sqlite3"),
            "lexical_index_cache_hit": False,
            **(payload.get("retrieval") or {}),
            "selected": [
                {
                    "id": item["id"],
                    "path": item["path"],
                    "source_refs": item["source_refs"],
                }
                for item in payload["evidence"]
            ],
        },
        "runtime_diagnostics": payload.get("runtime") or {},
    }
    if full_trace:
        artifacts["chunk_manifest"] = {
            "fingerprint": fingerprint,
            "chunks": [],
            "selected_chunk_ids": [],
        }
    return artifacts


def _released_artifact_payloads(payload, cached_query_stage, fingerprint, cache_dir, full_trace=False):
    artifacts = _artifact_payloads(payload, fingerprint, cache_dir, full_trace=full_trace)
    persisted = cached_query_stage.get("released_artifact_payloads")
    if not isinstance(persisted, dict) or not persisted:
        return artifacts
    evidence_pack = persisted.get("evidence_pack")
    if isinstance(evidence_pack, dict):
        artifacts["evidence_pack"] = dict(evidence_pack)
    if full_trace:
        chunk_manifest = persisted.get("chunk_manifest")
        if isinstance(chunk_manifest, dict):
            artifacts["chunk_manifest"] = dict(chunk_manifest)
    return artifacts


def cached_lexical_fallback_from_context(context):
    cached_query_stage = context.get("cached_query_stage")
    if cached_query_stage is None:
        return None
    query = context["query"]
    mode = context["mode"]
    budgets = context["budgets"]
    task_params = context["task_params"]
    cache_dir = context["cache_dir"]
    execution_plan = context.get("execution_plan") or {}
    repository_state_fingerprint = context["repository_state_fingerprint"]
    fingerprint_state = context["fingerprint_state"]
    cached_chunk_snapshot_metadata = context["cached_chunk_snapshot_metadata"]
    metadata_chunks = context["metadata_chunks"]
    fingerprint = context["fingerprint"]
    retrieval_quality = str(cached_query_stage.get("retrieval_quality") or "gpu")
    rerank_quality = str(cached_query_stage.get("rerank_quality") or "gpu")
    if retrieval_quality == "gpu" and rerank_quality == "gpu":
        return None
    released_payload = cached_query_stage.get("released_payload")
    persisted_released_payload = dict(released_payload) if isinstance(released_payload, dict) and released_payload else None

    total_files = int((cached_chunk_snapshot_metadata or {}).get("total_files") or 0)
    if persisted_released_payload is not None:
        evidence = [dict(item) for item in (persisted_released_payload.get("evidence") or ()) if isinstance(item, dict)]
        warnings = [str(item) for item in (persisted_released_payload.get("warnings") or ()) if str(item)]
        provenance = {
            "repository_fingerprint": repository_state_fingerprint,
            "index_fingerprint": fingerprint,
            "retrieval_model_profile": "",
            "rerank_model_profile": "",
        }
        provenance.update(dict(persisted_released_payload.get("provenance") or {}))
    else:
        evidence = [dict(item) for item in cached_query_stage["evidence"]]
        warnings = []
        if bool(cached_query_stage.get("evidence_budget_trimmed")):
            warnings.append("EVIDENCE_TOKEN_BUDGET_TRIMMED")
        if int(getattr(metadata_chunks, "_chunk_count", 0) or 0) and retrieval_quality != "gpu":
            warnings.append("GPU_RETRIEVAL_UNAVAILABLE_LEXICAL_FALLBACK")
        if int(getattr(metadata_chunks, "_chunk_count", 0) or 0) and rerank_quality != "gpu":
            warnings.append("GPU_RERANK_UNAVAILABLE")
        provenance = {
            "repository_fingerprint": repository_state_fingerprint,
            "index_fingerprint": fingerprint,
            "retrieval_model_profile": "",
            "rerank_model_profile": "",
        }
        trim_base = {
            "mode": mode,
            "query": query,
            "findings": [],
            "quality": _quality("evidence_only", retrieval_quality, rerank_quality, "not_requested"),
            "warnings": list(dict.fromkeys(warnings)),
            "provenance": provenance,
        }
        evidence, final_pack_trimmed = trim_evidence_for_final_pack(
            evidence,
            trim_base,
            budgets["final_pack_token_budget"],
        )
        if final_pack_trimmed:
            warnings.append("FINAL_PACK_EVIDENCE_TRIMMED")
        if not evidence and "NO_REPOSITORY_EVIDENCE" not in warnings:
            warnings.append("FINAL_PACK_BUDGET_EXHAUSTED")
    evidence_tokens = estimate_tokens(json.dumps(evidence, ensure_ascii=True)) if evidence else 0
    diagnostics = {
        "retrieval": {
            "fingerprint": fingerprint,
            "fingerprint_sources": [state.get("kind", "") for state in fingerprint_state],
            "chunks_indexed": 0,
            "lexical_candidates": 0,
            "semantic_candidates": 0,
            "fused_candidates": len(cached_query_stage["ranked"]),
            "reranked_candidates": len(cached_query_stage["ranked"]) if rerank_quality == "gpu" else 0,
            "selected_evidence": len(evidence),
            "candidate_tokens": 0,
            "evidence_tokens": evidence_tokens,
            "chunk_cache_total_files": total_files,
            "chunk_cache_reused_files": total_files,
            "chunk_cache_rebuilt_files": 0,
            "lexical_index_cache_hit": False,
            "lexical_index_working_cache_hit": False,
            "lexical_index_updated_files": 0,
            "lexical_index_removed_files": 0,
            "lexical_index_inserted_chunks": 0,
            "query_stage_cache_hit": True,
            "semantic_index_cache_hit": False,
            "semantic_index_document_count": int(getattr(metadata_chunks, "_chunk_count", 0)),
            "semantic_index_embedded_documents": 0,
            "semantic_index_reused_documents": 0,
            "budgets": budgets,
        },
        "runtime": {
            "attempts": [
                {
                    "tier": "",
                    "operation": "semantic_retrieval",
                    "status": "succeeded",
                    "failure_category": "",
                    "escalation_reason": "query_stage_cache_hit",
                    "attempt_number": 1,
                    "detail": "",
                }
                if retrieval_quality == "gpu"
                else None,
                {
                    "tier": "",
                    "operation": "rerank",
                    "status": "succeeded",
                    "failure_category": "",
                    "escalation_reason": "query_stage_cache_hit",
                    "attempt_number": 1,
                    "detail": "",
                }
                if rerank_quality == "gpu"
                else None,
            ],
            **cache_runtime_diagnostics(execution_plan, context.get("output_dir"), cache_dir),
        },
    }
    diagnostics["runtime"]["attempts"] = [
        attempt for attempt in diagnostics["runtime"]["attempts"] if isinstance(attempt, dict)
    ]
    payload = {
        "mode": mode,
        "query": query,
        "findings": [],
        "evidence": evidence,
        "quality": _quality("evidence_only", retrieval_quality, rerank_quality, "not_requested"),
        "warnings": list(dict.fromkeys(warnings)),
        "provenance": provenance,
        "retrieval": diagnostics["retrieval"],
        "runtime": diagnostics["runtime"],
    }
    diagnostics["retrieval"]["final_pack_tokens"] = released_pack_tokens(payload)
    diagnostics["retrieval"]["selected_evidence"] = len(payload["evidence"])
    return {
        "payload": payload,
        "artifact_payloads": _released_artifact_payloads(
            payload,
            cached_query_stage,
            fingerprint,
            cache_dir,
            bool(task_params.get("include_full_trace")),
        ),
    }


def load_cached_chunk_snapshot_metadata(
    discovered,
    excluded_dir_names=None,
    *,
    max_lines=120,
    overlap=12,
    cache_dir=None,
    excluded_paths=None,
    repository_state_fingerprint=None,
    build_config_digest=None,
    transient_excluded_paths=None,
):
    if cache_dir is None or not repository_state_fingerprint:
        return None
    if not build_config_digest:
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
    if metadata is not None and (
        str(metadata.get("repository_state_fingerprint") or "") != str(repository_state_fingerprint)
        or str(metadata.get("build_config_digest") or "") != str(build_config_digest)
    ):
        metadata = None
    if metadata is not None:
        return metadata
    if metadata is None:
        shared_metadata_path = _shared_file_chunk_snapshot_metadata_path(
            str(repository_state_fingerprint or ""),
            str(build_config_digest or ""),
            create=False,
        )
        if shared_metadata_path is not None and shared_metadata_path.exists():
            cache_key = _snapshot_metadata_cache_key(shared_metadata_path)
            if cache_key is not None:
                cached = _SNAPSHOT_METADATA_CACHE.get(cache_key)
                if cached is not None:
                    _SNAPSHOT_METADATA_CACHE.pop(cache_key, None)
                    _SNAPSHOT_METADATA_CACHE[cache_key] = cached
                    metadata = _clone_snapshot_metadata(cached)
            if metadata is None:
                try:
                    payload = json.loads(shared_metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    with contextlib.suppress(OSError):
                        shared_metadata_path.unlink()
                    return None
                metadata = _normalize_file_chunk_snapshot_metadata_payload(payload)
                if metadata is not None:
                    _cache_snapshot_metadata(shared_metadata_path, metadata)
    if metadata is None:
        return None
    if (
        str(metadata.get("repository_state_fingerprint") or "") != str(repository_state_fingerprint)
        or str(metadata.get("build_config_digest") or "") != str(build_config_digest)
    ):
        return None
    return metadata


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
    manifest = []
    for chunk in chunks:
        manifest.append(
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


def query_stage_cache_key(query, fingerprint, retrieval_signature, budgets):
    encoded = json.dumps(
        {
            "schema": QUERY_STAGE_CACHE_SCHEMA,
            "query": str(query),
            "index_fingerprint": str(fingerprint),
            "retrieval_signature": dict(retrieval_signature or {}),
            "retrieval_token_budget": int(budgets.get("retrieval_token_budget") or 0),
            "evidence_token_budget": int(budgets.get("evidence_token_budget") or 0),
            "final_pack_token_budget": int(budgets.get("final_pack_token_budget") or 0),
            "synthesis_context_token_budget": int(budgets.get("synthesis_context_token_budget") or 0),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return f"sha256:{sha256_text(encoded)}"


def _query_stage_cache_dir(cache_dir):
    return _private_cache_dir(Path(cache_dir) / "query-stage-cache")


def _shared_query_stage_cache_dir():
    configured = os.environ.get("BROKER_REPO_INSPECTION_SHARED_CACHE_DIR", "").strip()
    if not configured:
        return None
    try:
        return _private_cache_dir(Path(configured).expanduser().resolve(strict=False) / "query-stage-cache")
    except OSError:
        return None


def _sharded_json_cache_path(cache_dir: Path | None, key: str):
    if cache_dir is None:
        return None
    safe_key = str(key).replace(":", "_")
    shard = safe_key[:2] or "00"
    return _private_cache_dir(Path(cache_dir) / shard) / f"{safe_key}.json"


def _legacy_json_cache_path(cache_dir: Path | None, key: str):
    if cache_dir is None:
        return None
    safe_key = str(key).replace(":", "_")
    return Path(cache_dir) / f"{safe_key}.json"


def _query_stage_cache_path(cache_dir, cache_key):
    return _sharded_json_cache_path(_query_stage_cache_dir(cache_dir), cache_key)


def _shared_query_stage_cache_path(cache_key):
    return _sharded_json_cache_path(_shared_query_stage_cache_dir(), cache_key)


def _legacy_query_stage_cache_path(cache_dir, cache_key):
    return _legacy_json_cache_path(_query_stage_cache_dir(cache_dir), cache_key)


def _legacy_shared_query_stage_cache_path(cache_key):
    return _legacy_json_cache_path(_shared_query_stage_cache_dir(), cache_key)


def query_stage_cache_alias_key(query, repository_state_fingerprint, build_config_digest, retrieval_signature, budgets):
    encoded = json.dumps(
        {
            "schema": QUERY_STAGE_CACHE_ALIAS_SCHEMA,
            "query": str(query),
            "repository_state_fingerprint": str(repository_state_fingerprint),
            "build_config_digest": str(build_config_digest),
            "retrieval_signature": dict(retrieval_signature or {}),
            "retrieval_token_budget": int(budgets.get("retrieval_token_budget") or 0),
            "evidence_token_budget": int(budgets.get("evidence_token_budget") or 0),
            "final_pack_token_budget": int(budgets.get("final_pack_token_budget") or 0),
            "synthesis_context_token_budget": int(budgets.get("synthesis_context_token_budget") or 0),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return f"sha256:{sha256_text(encoded)}"


def _query_stage_cache_alias_dir(cache_dir):
    return _private_cache_dir(Path(cache_dir) / "query-stage-cache-alias")


def _shared_query_stage_cache_alias_dir():
    configured = os.environ.get("BROKER_REPO_INSPECTION_SHARED_CACHE_DIR", "").strip()
    if not configured:
        return None
    try:
        return _private_cache_dir(Path(configured).expanduser().resolve(strict=False) / "query-stage-cache-alias")
    except OSError:
        return None


def _query_stage_cache_alias_path(cache_dir, alias_key):
    return _sharded_json_cache_path(_query_stage_cache_alias_dir(cache_dir), alias_key)


def _shared_query_stage_cache_alias_path(alias_key):
    return _sharded_json_cache_path(_shared_query_stage_cache_alias_dir(), alias_key)


def _legacy_query_stage_cache_alias_path(cache_dir, alias_key):
    return _legacy_json_cache_path(_query_stage_cache_alias_dir(cache_dir), alias_key)


def _legacy_shared_query_stage_cache_alias_path(alias_key):
    return _legacy_json_cache_path(_shared_query_stage_cache_alias_dir(), alias_key)


def load_query_stage_cache_alias(cache_dir, alias_key):
    local_path = _query_stage_cache_alias_path(cache_dir, alias_key)
    shared_path = _shared_query_stage_cache_alias_path(alias_key)
    legacy_local_path = _legacy_query_stage_cache_alias_path(cache_dir, alias_key)
    legacy_shared_path = _legacy_shared_query_stage_cache_alias_path(alias_key)

    def load_path(path):
        if path is None:
            return None
        memory_key = _query_stage_memory_cache_key(path)
        if memory_key is not None:
            cached = _QUERY_STAGE_ALIAS_MEMORY_CACHE.get(memory_key)
            if cached is not None:
                _QUERY_STAGE_ALIAS_MEMORY_CACHE.pop(memory_key, None)
                _QUERY_STAGE_ALIAS_MEMORY_CACHE[memory_key] = dict(cached)
                return dict(cached)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            path.unlink(missing_ok=True)
            return None
        if not isinstance(payload, dict) or payload.get("schema") != QUERY_STAGE_CACHE_ALIAS_SCHEMA:
            path.unlink(missing_ok=True)
            return None
        cache_key = str(payload.get("cache_key") or "")
        if not cache_key:
            path.unlink(missing_ok=True)
            return None
        released_lexical_fallback = payload.get("released_lexical_fallback")
        normalized = {
            "cache_key": cache_key,
            "index_fingerprint": str(payload.get("index_fingerprint") or ""),
            "total_files": int(payload.get("total_files") or 0),
            "chunk_count": int(payload.get("chunk_count") or 0),
        }
        if isinstance(released_lexical_fallback, dict) and released_lexical_fallback:
            normalized["released_lexical_fallback"] = {
                "retrieval_quality": str(released_lexical_fallback.get("retrieval_quality") or ""),
                "rerank_quality": str(released_lexical_fallback.get("rerank_quality") or ""),
                "ranked": [dict(item) for item in (released_lexical_fallback.get("ranked") or ()) if isinstance(item, dict)],
                "selected": [dict(item) for item in (released_lexical_fallback.get("selected") or ()) if isinstance(item, dict)],
                "evidence_budget_trimmed": bool(released_lexical_fallback.get("evidence_budget_trimmed")),
                "ranked_count": int(released_lexical_fallback.get("ranked_count") or 0),
                "released_payload": dict(released_lexical_fallback.get("released_payload") or {})
                if isinstance(released_lexical_fallback.get("released_payload"), dict)
                else {},
                "released_artifact_payloads": {
                    str(key): dict(value)
                    for key, value in dict(released_lexical_fallback.get("released_artifact_payloads") or {}).items()
                    if isinstance(value, dict)
                },
            }
        if memory_key is not None:
            _QUERY_STAGE_ALIAS_MEMORY_CACHE.pop(memory_key, None)
            _QUERY_STAGE_ALIAS_MEMORY_CACHE[memory_key] = dict(normalized)
            while len(_QUERY_STAGE_ALIAS_MEMORY_CACHE) > _QUERY_STAGE_MEMORY_CACHE_LIMIT:
                _QUERY_STAGE_ALIAS_MEMORY_CACHE.pop(next(iter(_QUERY_STAGE_ALIAS_MEMORY_CACHE)))
        return normalized

    payload = load_path(shared_path)
    if payload is not None:
        return payload
    payload = load_path(legacy_shared_path)
    if payload is not None:
        return payload
    payload = load_path(local_path)
    if payload is not None:
        return payload
    return load_path(legacy_local_path)


def _normalize_cached_ranked(items):
    normalized = []
    for item in items or ():
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "")
        if not chunk_id:
            continue
        normalized.append(
            {
                "chunk_id": chunk_id,
                "rrf_score": float(item.get("rrf_score") or 0.0),
                "rerank_score": float(item.get("rerank_score") or 0.0),
                "rank": int(item.get("rank") or (len(normalized) + 1)),
                "sources": [str(value) for value in (item.get("sources") or ()) if str(value)],
            }
        )
    return normalized


def _normalize_cached_selected(items):
    normalized = []
    for item in items or ():
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "")
        if not chunk_id:
            continue
        normalized.append(
            {
                "chunk_id": chunk_id,
                "rank": int(item.get("rank") or (len(normalized) + 1)),
            }
        )
    return normalized


def _normalize_cached_findings(items):
    normalized = []
    for item in items or ():
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary") or "").strip()
        evidence_refs = [str(value).strip() for value in (item.get("evidence_refs") or ()) if str(value).strip()]
        if not summary or not evidence_refs:
            continue
        normalized.append({"summary": summary, "evidence_refs": evidence_refs})
    return normalized


def _clone_query_stage_cache_payload(payload):
    if payload is None:
        return None
    return {
        "ranked": [dict(item) for item in (payload.get("ranked") or ()) if isinstance(item, dict)],
        "selected": [dict(item) for item in (payload.get("selected") or ()) if isinstance(item, dict)],
        "evidence": [dict(item) for item in (payload.get("evidence") or ()) if isinstance(item, dict)],
        "evidence_budget_trimmed": bool(payload.get("evidence_budget_trimmed")),
        "retrieval_signature": dict(payload.get("retrieval_signature") or {}),
        "retrieval_quality": str(payload.get("retrieval_quality") or "gpu"),
        "rerank_quality": str(payload.get("rerank_quality") or "gpu"),
        "answer": str(payload.get("answer") or ""),
        "findings": [dict(item) for item in (payload.get("findings") or ()) if isinstance(item, dict)],
        "warnings": [str(item) for item in (payload.get("warnings") or ()) if str(item)],
        "provenance": dict(payload.get("provenance") or {}),
        "runtime_attempts": [dict(item) for item in (payload.get("runtime_attempts") or ()) if isinstance(item, dict)],
        "synthesis_quality": str(payload.get("synthesis_quality") or "not_requested"),
    }


def _query_stage_memory_cache_key(path: Path):
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path), int(stat.st_mtime_ns), int(stat.st_size))


def _prune_query_stage_memory_cache():
    if len(_QUERY_STAGE_MEMORY_CACHE) <= _QUERY_STAGE_MEMORY_CACHE_LIMIT:
        return
    while len(_QUERY_STAGE_MEMORY_CACHE) > _QUERY_STAGE_MEMORY_CACHE_LIMIT:
        _QUERY_STAGE_MEMORY_CACHE.pop(next(iter(_QUERY_STAGE_MEMORY_CACHE)))


def _cache_query_stage_memory_payload(path: Path, payload):
    cache_key = _query_stage_memory_cache_key(path)
    if cache_key is None:
        return
    _QUERY_STAGE_MEMORY_CACHE.pop(cache_key, None)
    _QUERY_STAGE_MEMORY_CACHE[cache_key] = _clone_query_stage_cache_payload(payload)
    _prune_query_stage_memory_cache()


def _touch_query_stage_cache_path(path: Path):
    try:
        os.utime(path, None)
    except OSError:
        return


def repository_fingerprint_alias_key(hint_fingerprint: str, hint_source: str):
    encoded = json.dumps(
        {
            "schema": REPOSITORY_FINGERPRINT_ALIAS_SCHEMA,
            "hint_fingerprint": str(hint_fingerprint or ""),
            "hint_source": str(hint_source or ""),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return f"sha256:{sha256_text(encoded)}"


def _repository_fingerprint_alias_dir(cache_dir):
    return _private_cache_dir(Path(cache_dir) / "repository-fingerprint-alias")


def _shared_repository_fingerprint_alias_dir():
    configured = os.environ.get("BROKER_REPO_INSPECTION_SHARED_CACHE_DIR", "").strip()
    if not configured:
        return None
    try:
        return _private_cache_dir(Path(configured).expanduser().resolve(strict=False) / "repository-fingerprint-alias")
    except OSError:
        return None


def _repository_fingerprint_alias_path(cache_dir, alias_key):
    return _sharded_json_cache_path(_repository_fingerprint_alias_dir(cache_dir), alias_key)


def _shared_repository_fingerprint_alias_path(alias_key):
    return _sharded_json_cache_path(_shared_repository_fingerprint_alias_dir(), alias_key)


def _prune_repository_fingerprint_alias_memory_cache():
    if len(_REPOSITORY_FINGERPRINT_ALIAS_MEMORY_CACHE) <= _REPOSITORY_FINGERPRINT_ALIAS_MEMORY_CACHE_LIMIT:
        return
    while len(_REPOSITORY_FINGERPRINT_ALIAS_MEMORY_CACHE) > _REPOSITORY_FINGERPRINT_ALIAS_MEMORY_CACHE_LIMIT:
        _REPOSITORY_FINGERPRINT_ALIAS_MEMORY_CACHE.pop(next(iter(_REPOSITORY_FINGERPRINT_ALIAS_MEMORY_CACHE)))


def load_repository_fingerprint_alias(cache_dir, hint_fingerprint: str, hint_source: str):
    alias_key = repository_fingerprint_alias_key(hint_fingerprint, hint_source)
    local_path = _repository_fingerprint_alias_path(cache_dir, alias_key)
    shared_path = _shared_repository_fingerprint_alias_path(alias_key)

    def load_path(path):
        if path is None:
            return ""
        memory_key = _query_stage_memory_cache_key(path)
        if memory_key is not None:
            cached = _REPOSITORY_FINGERPRINT_ALIAS_MEMORY_CACHE.get(memory_key)
            if isinstance(cached, str) and cached.strip():
                _REPOSITORY_FINGERPRINT_ALIAS_MEMORY_CACHE.pop(memory_key, None)
                _REPOSITORY_FINGERPRINT_ALIAS_MEMORY_CACHE[memory_key] = cached
                return cached
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return ""
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            path.unlink(missing_ok=True)
            return ""
        if not isinstance(payload, dict) or payload.get("schema") != REPOSITORY_FINGERPRINT_ALIAS_SCHEMA:
            path.unlink(missing_ok=True)
            return ""
        actual = str(payload.get("repository_state_fingerprint") or "").strip()
        if not actual:
            path.unlink(missing_ok=True)
            return ""
        memory_key = _query_stage_memory_cache_key(path)
        if memory_key is not None:
            _REPOSITORY_FINGERPRINT_ALIAS_MEMORY_CACHE.pop(memory_key, None)
            _REPOSITORY_FINGERPRINT_ALIAS_MEMORY_CACHE[memory_key] = actual
            _prune_repository_fingerprint_alias_memory_cache()
        return actual

    actual = load_path(shared_path)
    if actual:
        return actual
    return load_path(local_path)


def write_repository_fingerprint_alias(cache_dir, hint_fingerprint: str, hint_source: str, repository_state_fingerprint: str):
    hint_fingerprint = str(hint_fingerprint or "").strip()
    hint_source = str(hint_source or "").strip()
    repository_state_fingerprint = str(repository_state_fingerprint or "").strip()
    if not hint_fingerprint or not repository_state_fingerprint:
        return
    alias_key = repository_fingerprint_alias_key(hint_fingerprint, hint_source)
    payload = {
        "schema": REPOSITORY_FINGERPRINT_ALIAS_SCHEMA,
        "hint_fingerprint": hint_fingerprint,
        "hint_source": hint_source,
        "repository_state_fingerprint": repository_state_fingerprint,
    }
    local_path = _repository_fingerprint_alias_path(cache_dir, alias_key)
    _atomic_private_json(local_path, payload)
    memory_key = _query_stage_memory_cache_key(local_path)
    if memory_key is not None:
        _REPOSITORY_FINGERPRINT_ALIAS_MEMORY_CACHE.pop(memory_key, None)
        _REPOSITORY_FINGERPRINT_ALIAS_MEMORY_CACHE[memory_key] = repository_state_fingerprint
        _prune_repository_fingerprint_alias_memory_cache()
    shared_path = _shared_repository_fingerprint_alias_path(alias_key)
    if shared_path is not None and shared_path != local_path:
        _atomic_private_json(shared_path, payload)
        shared_memory_key = _query_stage_memory_cache_key(shared_path)
        if shared_memory_key is not None:
            _REPOSITORY_FINGERPRINT_ALIAS_MEMORY_CACHE.pop(shared_memory_key, None)
            _REPOSITORY_FINGERPRINT_ALIAS_MEMORY_CACHE[shared_memory_key] = repository_state_fingerprint
            _prune_repository_fingerprint_alias_memory_cache()


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


def load_query_stage_cache(cache_dir, cache_key):
    local_path = _query_stage_cache_path(cache_dir, cache_key)
    shared_path = _shared_query_stage_cache_path(cache_key)
    legacy_local_path = _legacy_query_stage_cache_path(cache_dir, cache_key)
    legacy_shared_path = _legacy_shared_query_stage_cache_path(cache_key)

    def load_path(path):
        if path is None:
            return None
        memory_key = _query_stage_memory_cache_key(path)
        if memory_key is not None:
            cached = _QUERY_STAGE_MEMORY_CACHE.get(memory_key)
            if cached is not None:
                _QUERY_STAGE_MEMORY_CACHE.pop(memory_key, None)
                _cache_query_stage_memory_payload(path, cached)
                return _clone_query_stage_cache_payload(cached)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            path.unlink(missing_ok=True)
            return None
        if not isinstance(payload, dict) or payload.get("schema") != QUERY_STAGE_CACHE_SCHEMA:
            path.unlink(missing_ok=True)
            return None
        ranked = _normalize_cached_ranked(payload.get("ranked"))
        if not ranked:
            path.unlink(missing_ok=True)
            return None
        selected = _normalize_cached_selected(payload.get("selected"))
        if not selected:
            path.unlink(missing_ok=True)
            return None
        evidence = payload.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            path.unlink(missing_ok=True)
            return None
        normalized = {
            "ranked": ranked,
            "selected": selected,
            "evidence": [dict(item) for item in evidence if isinstance(item, dict)],
            "evidence_budget_trimmed": bool(payload.get("evidence_budget_trimmed")),
            "retrieval_signature": dict(payload.get("retrieval_signature") or {}),
            "retrieval_quality": str(payload.get("retrieval_quality") or "gpu"),
            "rerank_quality": str(payload.get("rerank_quality") or "gpu"),
            "answer": str(payload.get("answer") or ""),
            "findings": _normalize_cached_findings(payload.get("findings")),
            "warnings": [str(item) for item in (payload.get("warnings") or ()) if str(item)],
            "provenance": dict(payload.get("provenance") or {}),
            "runtime_attempts": [dict(item) for item in (payload.get("runtime_attempts") or ()) if isinstance(item, dict)],
            "synthesis_quality": str(payload.get("synthesis_quality") or "not_requested"),
        }
        _cache_query_stage_memory_payload(path, normalized)
        return _clone_query_stage_cache_payload(normalized)
    payload = load_path(shared_path)
    if payload is not None:
        return payload
    payload = load_path(legacy_shared_path)
    if payload is not None:
        return payload
    payload = load_path(local_path)
    if payload is not None:
        return payload
    return load_path(legacy_local_path)


def prepare_prefetched_state(
    discovered,
    query,
    *,
    mode="auto",
    constraints=None,
    task_params=None,
    execution_plan=None,
    output_dir=None,
):
    prefetch_stage_timings = {}
    git_probe_cache = _effective_git_probe_cache(None)
    query, mode = validate_request(query, mode)
    constraints = constraints or {}
    task_params = task_params or {}
    execution_plan = execution_plan or {}
    budgets = normalize_token_budgets(constraints)
    cache_dir = cache_dir_for_execution(execution_plan, output_dir, task_params=task_params)
    output_dir_path = Path(output_dir).expanduser().resolve(strict=False) if output_dir is not None else None
    excluded = set(task_params.get("excluded_dir_names") or task_params.get("exclude_dirs") or [])
    excluded_paths = {cache_dir}
    transient_excluded_paths = set()
    if output_dir_path is not None:
        excluded_paths.add(output_dir_path)
        transient_excluded_paths.add(output_dir_path)
    broker_repository_state_fingerprint = str(task_params.get("_broker_repository_state_fingerprint") or "").strip()
    broker_repository_state_fingerprint_source = str(
        task_params.get("_broker_repository_state_fingerprint_source") or "request_cache"
    ).strip() or "request_cache"
    prefetched_signatures, prefetched_retrieval_signature = _query_stage_retrieval_signatures_for_prefetch(
        execution_plan,
        task_params,
    )
    early_hint_fingerprint = broker_repository_state_fingerprint
    early_hint_source = broker_repository_state_fingerprint_source if broker_repository_state_fingerprint else ""
    if not early_hint_fingerprint:
        stage_started = time.perf_counter()
        early_hint_fingerprint, early_hint_source = _input_manifest_repository_state_fingerprint(discovered, task_params)
        prefetch_stage_timings["input_manifest_hint_ms"] = _prefetch_elapsed_ms(stage_started)
    else:
        prefetch_stage_timings["input_manifest_hint_ms"] = 0.0
    translated_hint_fingerprint = ""
    if early_hint_fingerprint:
        stage_started = time.perf_counter()
        translated_hint_fingerprint = load_repository_fingerprint_alias(
            cache_dir,
            early_hint_fingerprint,
            early_hint_source,
        )
        prefetch_stage_timings["hint_alias_lookup_ms"] = _prefetch_elapsed_ms(stage_started)
    else:
        prefetch_stage_timings["hint_alias_lookup_ms"] = 0.0
    if translated_hint_fingerprint:
        early_hint_fingerprint = translated_hint_fingerprint
    early_prefetched_cache_key = None
    if early_hint_fingerprint:
        early_prefetched_cache_key = _early_prefetch_state_cache_key(
            query=query,
            mode=mode,
            budgets=budgets,
            cache_dir=cache_dir,
            excluded=excluded,
            discovered=discovered,
            retrieval_signature=prefetched_retrieval_signature,
            hint_fingerprint=early_hint_fingerprint,
            hint_source=early_hint_source,
        )
        cached_prefetched = _PREFETCH_STATE_CACHE.get(early_prefetched_cache_key)
        if cached_prefetched is not None:
            _PREFETCH_STATE_CACHE.pop(early_prefetched_cache_key, None)
            _PREFETCH_STATE_CACHE[early_prefetched_cache_key] = cached_prefetched
            cloned_cached = _clone_prefetched_state_payload(cached_prefetched)
            state = {
                "query": query,
                "mode": mode,
                "constraints": constraints,
                "task_params": task_params,
                "execution_plan": execution_plan,
                "output_dir": output_dir,
                "budgets": budgets,
                "cache_dir": cache_dir,
                "excluded": excluded,
                "excluded_paths": excluded_paths,
                "build_config_digest": str(cloned_cached.get("build_config_digest") or ""),
                "git_probe_cache": git_probe_cache,
                "repository_state_fingerprint": str(early_hint_fingerprint),
                "fingerprint_state": fingerprint_hint_state(
                    early_hint_fingerprint,
                    early_hint_source,
                    kind=("input_manifest" if early_hint_source == "input_manifest" else "broker_hint"),
                ),
                "prefetch_stage_timings_ms": dict(prefetch_stage_timings),
                **cloned_cached,
            }
            _mark_prefetch_state(state, "early_process_cache", cache_hit=True)
            if isinstance(state.get("cached_lexical_fallback_run"), dict):
                return state
            return _attach_cached_lexical_fallback_run(state)
    stage_started = time.perf_counter()
    discovered, _namespaces, _effective_excluded_paths, build_config_digest = _snapshot_build_context(
        discovered,
        excluded,
        cache_dir=cache_dir,
        excluded_paths=excluded_paths,
        transient_excluded_paths=transient_excluded_paths,
    )
    prefetch_stage_timings["snapshot_build_context_ms"] = _prefetch_elapsed_ms(stage_started)
    manifest_repository_state_fingerprint = ""
    manifest_repository_state_fingerprint_source = ""
    if not broker_repository_state_fingerprint:
        stage_started = time.perf_counter()
        manifest_repository_state_fingerprint, manifest_repository_state_fingerprint_source = (
            _input_manifest_repository_state_fingerprint(discovered, task_params)
        )
        prefetch_stage_timings["manifest_repository_hint_ms"] = _prefetch_elapsed_ms(stage_started)
    else:
        prefetch_stage_timings["manifest_repository_hint_ms"] = 0.0
    if broker_repository_state_fingerprint:
        repository_state_fingerprint = translated_hint_fingerprint or broker_repository_state_fingerprint
        fingerprint_state = (
            fingerprint_alias_state(repository_state_fingerprint, broker_repository_state_fingerprint_source)
            if translated_hint_fingerprint
            else fingerprint_hint_state(
                broker_repository_state_fingerprint,
                broker_repository_state_fingerprint_source,
            )
        )
    elif manifest_repository_state_fingerprint:
        repository_state_fingerprint = translated_hint_fingerprint or manifest_repository_state_fingerprint
        fingerprint_state = (
            fingerprint_alias_state(repository_state_fingerprint, manifest_repository_state_fingerprint_source)
            if translated_hint_fingerprint
            else fingerprint_hint_state(
                manifest_repository_state_fingerprint,
                manifest_repository_state_fingerprint_source,
                kind=("input_manifest" if manifest_repository_state_fingerprint_source == "input_manifest" else "broker_hint"),
            )
        )
    else:
        stage_started = time.perf_counter()
        repository_state_fingerprint, fingerprint_state = repository_fingerprint(
            discovered,
            excluded,
            excluded_paths=excluded_paths,
            cache_dir=cache_dir,
            git_probe_cache=git_probe_cache,
        )
        prefetch_stage_timings["repository_fingerprint_ms"] = _prefetch_elapsed_ms(stage_started)
    if "repository_fingerprint_ms" not in prefetch_stage_timings:
        prefetch_stage_timings["repository_fingerprint_ms"] = 0.0
    prefetched_cache_key = _prefetch_state_cache_key(
        query=query,
        mode=mode,
        budgets=budgets,
        cache_dir=cache_dir,
        repository_state_fingerprint=repository_state_fingerprint,
        build_config_digest=build_config_digest,
        retrieval_signature=prefetched_retrieval_signature,
    )
    cached_prefetched = _PREFETCH_STATE_CACHE.get(prefetched_cache_key)
    if cached_prefetched is not None:
        _PREFETCH_STATE_CACHE.pop(prefetched_cache_key, None)
        _PREFETCH_STATE_CACHE[prefetched_cache_key] = cached_prefetched
        cloned_cached = _clone_prefetched_state_payload(cached_prefetched)
        state = {
            "query": query,
            "mode": mode,
            "constraints": constraints,
            "task_params": task_params,
            "execution_plan": execution_plan,
            "output_dir": output_dir,
            "budgets": budgets,
            "cache_dir": cache_dir,
            "excluded": excluded,
            "excluded_paths": excluded_paths,
            "build_config_digest": build_config_digest,
            "git_probe_cache": git_probe_cache,
            "repository_state_fingerprint": repository_state_fingerprint,
            "fingerprint_state": fingerprint_state,
            "prefetch_stage_timings_ms": dict(prefetch_stage_timings),
            **cloned_cached,
        }
        _mark_prefetch_state(state, "process_cache", cache_hit=True)
        if isinstance(state.get("cached_lexical_fallback_run"), dict):
            return state
        return _attach_cached_lexical_fallback_run(state)
    prefetched_repo_state_cache_key = None
    # Broker-supplied fingerprints identify shared repository state, but do
    # not prove that this worker process owns a compatible in-memory snapshot.
    # Resolve those requests through persisted snapshot metadata.  Direct
    # warm-daemon requests without a broker hint retain the faster process
    # cache path.
    if (
        repository_state_fingerprint
        and build_config_digest
        and not broker_repository_state_fingerprint
        and not bool(execution_plan.get("repo_inspection_use_node_local_cache"))
    ):
        prefetched_repo_state_cache_key = _prefetch_repo_state_cache_key(
            cache_dir=cache_dir,
            repository_state_fingerprint=repository_state_fingerprint,
            build_config_digest=build_config_digest,
            discovered=discovered,
        )
        cached_repo_state = _PREFETCH_REPO_STATE_CACHE.get(prefetched_repo_state_cache_key)
        if cached_repo_state is not None:
            _PREFETCH_REPO_STATE_CACHE.pop(prefetched_repo_state_cache_key, None)
            _PREFETCH_REPO_STATE_CACHE[prefetched_repo_state_cache_key] = cached_repo_state
        if cached_repo_state is not None:
            cloned_repo_state = _clone_prefetch_repo_state_payload(cached_repo_state)
            result_state = {
                "query": query,
                "mode": mode,
                "constraints": constraints,
                "task_params": task_params,
                "execution_plan": execution_plan,
                "output_dir": output_dir,
                "budgets": budgets,
                "cache_dir": cache_dir,
                "excluded": excluded,
                "excluded_paths": excluded_paths,
                "build_config_digest": str(cloned_repo_state.get("build_config_digest") or build_config_digest or ""),
                "git_probe_cache": git_probe_cache,
                "repository_state_fingerprint": repository_state_fingerprint,
                "fingerprint_state": fingerprint_state,
                "prefetch_stage_timings_ms": dict(prefetch_stage_timings),
                "cached_chunk_snapshot_metadata": cloned_repo_state.get("cached_chunk_snapshot_metadata"),
                "metadata_chunks": cloned_repo_state.get("metadata_chunks"),
                "fingerprint": str(cloned_repo_state.get("fingerprint") or ""),
                "cached_query_stage": None,
                "prefetched_query_stage_cache_key": "",
                "prefetched_query_stage_cache_probed": False,
                "prefetched_query_stage_requires_verification": False,
                "prefetched_retrieval_signature": dict(
                    cloned_repo_state.get("prefetched_retrieval_signature") or prefetched_retrieval_signature or {}
                ),
            }
            result_state = _attach_cached_lexical_fallback_run(result_state)
            _mark_prefetch_state(result_state, "repo_state_process_cache", cache_hit=True)
            return result_state
    prefetched_query_stage_cache_key = ""
    prefetched_query_stage_cache_probed = False
    cached_query_stage = None
    metadata_chunks = None
    fingerprint = ""
    cached_chunk_snapshot_metadata = None
    if repository_state_fingerprint and build_config_digest:
        alias_lookup_started = time.perf_counter()
        for signature in prefetched_signatures:
            prefetched_alias_key = query_stage_cache_alias_key(
                query,
                repository_state_fingerprint,
                build_config_digest,
                signature,
                budgets,
            )
            cached_alias = load_query_stage_cache_alias(cache_dir, prefetched_alias_key)
            if cached_alias is None:
                continue
            prefetched_query_stage_cache_key = str(cached_alias.get("cache_key") or "")
            if not prefetched_query_stage_cache_key:
                continue
            prefetched_retrieval_signature = dict(signature)
            released_lexical_fallback = dict(cached_alias.get("released_lexical_fallback") or {})
            fingerprint = str(cached_alias.get("index_fingerprint") or "")
            if released_lexical_fallback:
                released_payload = dict(released_lexical_fallback.get("released_payload") or {})
                released_evidence = [
                    dict(item)
                    for item in (released_payload.get("evidence") or ())
                    if isinstance(item, dict)
                ]
                cached_query_stage = {
                    "retrieval_quality": str(released_lexical_fallback.get("retrieval_quality") or "gpu"),
                    "rerank_quality": str(released_lexical_fallback.get("rerank_quality") or "gpu"),
                    "ranked": [dict(item) for item in (released_lexical_fallback.get("ranked") or ()) if isinstance(item, dict)],
                    "selected": [dict(item) for item in (released_lexical_fallback.get("selected") or ()) if isinstance(item, dict)],
                    "evidence": released_evidence,
                    "evidence_budget_trimmed": bool(released_lexical_fallback.get("evidence_budget_trimmed")),
                    "released_payload": released_payload,
                    "released_artifact_payloads": {
                        str(key): dict(value)
                        for key, value in dict(released_lexical_fallback.get("released_artifact_payloads") or {}).items()
                        if isinstance(value, dict)
                    },
                }
                if not cached_query_stage["ranked"]:
                    cached_query_stage["ranked"] = [None] * max(0, int(released_lexical_fallback.get("ranked_count") or 0))
                prefetched_query_stage_cache_probed = True
            else:
                cached_query_stage = load_query_stage_cache(cache_dir, prefetched_query_stage_cache_key)
                prefetched_query_stage_cache_probed = True
                if cached_query_stage is None:
                    continue
                if not fingerprint:
                    fingerprint = str(((cached_query_stage.get("provenance") or {}).get("index_fingerprint")) or "")
            metadata_chunks = type("MetadataChunkList", (list,), {})()
            metadata_chunks._index_manifest = {}
            metadata_chunks._semantic_document_signatures = {}
            metadata_chunks._chunk_ids = ()
            metadata_chunks._chunk_count = int(cached_alias.get("chunk_count") or 0)
            cached_chunk_snapshot_metadata = {
                "repository_state_fingerprint": str(repository_state_fingerprint or ""),
                "build_config_digest": str(build_config_digest or ""),
                "index_manifest": {},
                "semantic_document_signatures": {},
                "chunk_ids": [],
                "chunk_count": int(cached_alias.get("chunk_count") or 0),
                "total_files": int(cached_alias.get("total_files") or 0),
            }
            break
        prefetch_stage_timings["query_stage_alias_lookup_ms"] = _prefetch_elapsed_ms(alias_lookup_started)
    else:
        prefetch_stage_timings["query_stage_alias_lookup_ms"] = 0.0
    if cached_query_stage is not None:
        result_state = _attach_cached_lexical_fallback_run({
            "query": query,
            "mode": mode,
            "constraints": constraints,
            "task_params": task_params,
            "execution_plan": execution_plan,
            "output_dir": output_dir,
            "budgets": budgets,
            "cache_dir": cache_dir,
            "excluded": excluded,
            "excluded_paths": excluded_paths,
            "build_config_digest": build_config_digest,
            "git_probe_cache": git_probe_cache,
            "repository_state_fingerprint": repository_state_fingerprint,
            "fingerprint_state": fingerprint_state,
            "prefetch_stage_timings_ms": dict(prefetch_stage_timings),
            "cached_chunk_snapshot_metadata": cached_chunk_snapshot_metadata,
            "metadata_chunks": metadata_chunks,
            "fingerprint": fingerprint,
            "cached_query_stage": cached_query_stage,
            "prefetched_query_stage_cache_key": prefetched_query_stage_cache_key,
            "prefetched_query_stage_cache_probed": prefetched_query_stage_cache_probed,
            "prefetched_retrieval_signature": prefetched_retrieval_signature,
        })
        _mark_prefetch_state(result_state, "query_stage_alias", cache_hit=False)
        cache_payload = _clone_prefetched_state_payload(result_state)
        cache_payload["build_config_digest"] = str(build_config_digest or "")
        _PREFETCH_STATE_CACHE[prefetched_cache_key] = cache_payload
        if early_prefetched_cache_key is not None:
            _PREFETCH_STATE_CACHE[early_prefetched_cache_key] = _clone_prefetched_state_payload(cache_payload)
        _prune_prefetch_state_cache()
        if prefetched_repo_state_cache_key is not None:
            _PREFETCH_REPO_STATE_CACHE[prefetched_repo_state_cache_key] = _clone_prefetch_repo_state_payload(result_state)
            _prune_prefetch_repo_state_cache()
        return result_state
    stage_started = time.perf_counter()
    cached_chunk_snapshot_metadata = load_cached_chunk_snapshot_metadata(
        discovered,
        excluded,
        cache_dir=cache_dir,
        excluded_paths=excluded_paths,
        repository_state_fingerprint=repository_state_fingerprint,
        build_config_digest=build_config_digest,
        transient_excluded_paths=transient_excluded_paths,
    )
    prefetch_stage_timings["cached_snapshot_metadata_lookup_ms"] = _prefetch_elapsed_ms(stage_started)
    prefetch_stage_timings["hint_alias_verification_repository_fingerprint_ms"] = 0.0
    prefetch_stage_timings["post_alias_snapshot_metadata_lookup_ms"] = 0.0
    metadata_chunks = None
    fingerprint = ""
    if cached_chunk_snapshot_metadata is not None:
        metadata_chunks = type("MetadataChunkList", (list,), {})()
        metadata_chunks._index_manifest = dict(cached_chunk_snapshot_metadata["index_manifest"])
        metadata_chunks._semantic_document_signatures = dict(cached_chunk_snapshot_metadata["semantic_document_signatures"])
        metadata_chunks._chunk_ids = tuple(cached_chunk_snapshot_metadata["chunk_ids"])
        metadata_chunks._chunk_count = int(cached_chunk_snapshot_metadata.get("chunk_count") or len(metadata_chunks._chunk_ids))
        fingerprint = inspection_index_fingerprint(repository_state_fingerprint, metadata_chunks)
        prefetched_query_stage_cache_key = query_stage_cache_key(query, fingerprint, prefetched_retrieval_signature, budgets)
        stage_started = time.perf_counter()
        cached_query_stage = load_query_stage_cache(cache_dir, prefetched_query_stage_cache_key)
        prefetch_stage_timings["query_stage_cache_lookup_ms"] = _prefetch_elapsed_ms(stage_started)
        prefetched_query_stage_cache_probed = True
    else:
        prefetch_stage_timings["query_stage_cache_lookup_ms"] = 0.0
    prefetched_query_stage_requires_verification = bool(cached_query_stage is not None)
    result_state = {
        "query": query,
        "mode": mode,
        "constraints": constraints,
        "task_params": task_params,
        "execution_plan": execution_plan,
        "output_dir": output_dir,
        "budgets": budgets,
        "cache_dir": cache_dir,
        "excluded": excluded,
        "excluded_paths": excluded_paths,
        "build_config_digest": build_config_digest,
        "git_probe_cache": git_probe_cache,
        "repository_state_fingerprint": repository_state_fingerprint,
        "fingerprint_state": fingerprint_state,
        "prefetch_stage_timings_ms": dict(prefetch_stage_timings),
        "cached_chunk_snapshot_metadata": cached_chunk_snapshot_metadata,
        "metadata_chunks": metadata_chunks,
        "fingerprint": fingerprint,
        "cached_query_stage": cached_query_stage,
        "prefetched_query_stage_cache_key": prefetched_query_stage_cache_key,
        "prefetched_query_stage_cache_probed": prefetched_query_stage_cache_probed,
        "prefetched_query_stage_requires_verification": prefetched_query_stage_requires_verification,
        "prefetched_retrieval_signature": prefetched_retrieval_signature,
    }
    result_state = _attach_cached_lexical_fallback_run(result_state)
    if isinstance(result_state.get("cached_lexical_fallback_run"), dict):
        result_state["prefetched_query_stage_requires_verification"] = False
    if cached_chunk_snapshot_metadata is not None:
        _mark_prefetch_state(result_state, "snapshot_metadata", cache_hit=False)
    else:
        _mark_prefetch_state(result_state, "fresh", cache_hit=False)
    cache_payload = _clone_prefetched_state_payload(result_state)
    cache_payload["build_config_digest"] = str(build_config_digest or "")
    _PREFETCH_STATE_CACHE[prefetched_cache_key] = cache_payload
    if early_prefetched_cache_key is not None:
        _PREFETCH_STATE_CACHE[early_prefetched_cache_key] = _clone_prefetched_state_payload(cache_payload)
    _prune_prefetch_state_cache()
    if prefetched_repo_state_cache_key is not None:
        _PREFETCH_REPO_STATE_CACHE[prefetched_repo_state_cache_key] = _clone_prefetch_repo_state_payload(result_state)
        _prune_prefetch_repo_state_cache()
    return result_state
