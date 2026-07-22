#!/usr/bin/env python3
"""Launch and register one authenticated GPU model service.

The operator supplies the exact runtime executable and every runtime argument.
This wrapper only substitutes documented placeholders, starts the runtime on a
loopback port, exposes a bearer-authenticated proxy, and renews the shared
registry lease. It performs no model selection and has no model defaults.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gzip
import hashlib
import http.client
import io
import json
import math
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - runtime fallback when numpy is unavailable
    np = None

try:
    import faiss
except ImportError:  # pragma: no cover - optional acceleration only
    faiss = None


REGISTRY_SCHEMA = "gpu_service_registry_v1"
MAX_BODY_BYTES = 64 * 1024 * 1024
RUNTIME_PLACEHOLDER_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

RETRIEVAL_ENDPOINTS = frozenset(
    {
        "/v1/indexes/status",
        "/v1/indexes/upsert",
        "/v1/search",
        "/v1/rerank",
    }
)


def utc_now():
    return datetime.now(timezone.utc)


def format_time(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = json.dumps(payload, indent=2, sort_keys=False).encode("utf-8")
    fd, temp_path = tempfile.mkstemp(prefix=".gpu-registry-", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        os.chmod(path, 0o600)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_path)


def _write_bytes_if_changed(path, data, *, prefix, suffix):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        if path.exists() and path.read_bytes() == data:
            return
    except OSError:
        pass
    fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        os.chmod(path, 0o600)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_path)


class RegistryPublisher:
    def __init__(self, path, service_id, registration_token):
        self.path = Path(path)
        self.lock_path = Path(str(path) + ".lock")
        self.service_id = service_id
        self.registration_token_hash = hashlib.sha256(registration_token.encode("utf-8")).hexdigest()
        self.lease_duration = None

    @contextlib.contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.lock_path.open("a+b") as lock:
            os.chmod(self.lock_path, 0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if not self.path.exists():
                    raise RuntimeError("GPU service registry does not exist")
                registry = load_json(self.path)
                if registry.get("schema") != REGISTRY_SCHEMA:
                    raise RuntimeError("unsupported GPU service registry schema")
                yield registry
                registry["updated_at"] = format_time(utc_now())
                _atomic_write(self.path, registry)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _record(self, registry):
        for record in registry.get("records") or []:
            if record.get("id") == self.service_id:
                expected = str(record.get("registration_token_sha256") or "")
                if not secrets.compare_digest(expected, self.registration_token_hash):
                    raise RuntimeError("GPU service registration denied")
                if self.lease_duration is None:
                    created = parse_time(record["created_at"])
                    expires = parse_time(record["lease_expires_at"])
                    self.lease_duration = max(timedelta(seconds=1), expires - created)
                return record
        raise RuntimeError(f"GPU service reservation {self.service_id!r} was not found")

    def _bounded_lease_expiry(self, record, candidate):
        value = record.get("absolute_lease_expires_at")
        if not value:
            return candidate
        absolute = parse_time(value)
        # Go's encoding/json may emit the zero time despite `omitempty` on a
        # time.Time field. That sentinel means the warm P40 lease is unbounded.
        if absolute.year <= 1:
            return candidate
        return min(candidate, absolute)

    def publish(self, endpoint, bearer_token, slurm_job_id):
        with self._locked() as registry:
            record = self._record(registry)
            now = utc_now()
            if record.get("state") != "starting" or parse_time(record["startup_deadline"]) <= now:
                raise RuntimeError("GPU service reservation is no longer publishable")
            record.update(
                {
                    "state": "ready",
                    "endpoint": endpoint,
                    "endpoint_auth": {"type": "bearer", "bearer_token": bearer_token},
                    "slurm_job_id": str(slurm_job_id or record.get("slurm_job_id") or ""),
                    "heartbeat_at": format_time(now),
                    "lease_expires_at": format_time(
                        self._bounded_lease_expiry(record, now + self.lease_duration)
                    ),
                    "health_error": "",
                    "failure_category": "",
                }
            )

    def renew(self):
        with self._locked() as registry:
            record = self._record(registry)
            if record.get("state") != "ready":
                raise RuntimeError("GPU service reservation is no longer ready")
            now = utc_now()
            record["heartbeat_at"] = format_time(now)
            record["lease_expires_at"] = format_time(
                self._bounded_lease_expiry(record, now + self.lease_duration)
            )

    def mark_unhealthy(self, reason):
        with self._locked() as registry:
            record = self._record(registry)
            now = utc_now()
            record.update(
                {
                    "state": "unhealthy",
                    "health_error": str(reason)[:1000],
                    "failure_category": "service_failure",
                    "lease_expires_at": format_time(now),
                    "last_health_check_at": format_time(now),
                }
            )


def reserve_port(host):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def substitute_runtime_args(deployment, internal_port, endpoint_token, gpu_count):
    values = {
        "model": str(deployment["model"]),
        "model_path": str(deployment["model"]),
        "quantization": str(deployment["quantization"]),
        "context_limit_tokens": str(deployment["context_limit_tokens"]),
        "host": "127.0.0.1",
        "port": str(internal_port),
        "gpu_count": str(gpu_count),
        "endpoint_token": endpoint_token,
    }
    runtime_args = deployment.get("runtime_args")
    if not isinstance(runtime_args, list) or not runtime_args or not all(isinstance(item, str) for item in runtime_args):
        raise ValueError("runtime_args must be a non-empty array of strings")

    def replacement(match):
        name = match.group(1)
        if name not in values:
            raise ValueError(f"unsupported runtime argument placeholder {{{name}}}")
        return values[name]

    # Substitute against each original argument exactly once. In particular, a
    # model path containing placeholder-looking text must remain literal.
    result = [RUNTIME_PLACEHOLDER_PATTERN.sub(replacement, raw) for raw in runtime_args]
    return result


class AuthenticatedProxy(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, bearer_token, runtime_process, upstream_port, retrieval_adapter=None):
        self.bearer_token = bearer_token
        self.runtime_process = runtime_process
        self.upstream_port = upstream_port
        self.retrieval_adapter = retrieval_adapter
        super().__init__(address, ProxyHandler)


class UpstreamJSONClient:
    def __init__(self, port, bearer_token, timeout=300):
        self.port = int(port)
        self.bearer_token = str(bearer_token)
        self.timeout = float(timeout)

    def post(self, path, payload):
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
        try:
            connection.request(
                "POST",
                path,
                body=body,
                headers={
                    "Authorization": "Bearer " + self.bearer_token,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            response = connection.getresponse()
            raw = response.read(MAX_BODY_BYTES + 1)
            if len(raw) > MAX_BODY_BYTES:
                raise RuntimeError("upstream response too large")
            if not (200 <= response.status < 300):
                detail = raw.decode("utf-8", errors="replace")
                raise RuntimeError(f"upstream {path} failed ({response.status}): {detail[:500]}")
            return json.loads(raw.decode("utf-8", errors="replace"))
        finally:
            connection.close()


def _vector_norm(values):
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _normalize_vector(values):
    norm = _vector_norm(values)
    if norm <= 0:
        return [0.0 for _ in values]
    return [float(value) / norm for value in values]


def _cosine_similarity(left, right):
    if len(left) != len(right):
        raise ValueError("vector dimensions do not match")
    return sum(float(a) * float(b) for a, b in zip(left, right))


class RetrievalServiceAdapter:
    def __init__(self, upstream_port, bearer_token, *, cache_dir=None):
        self.upstream = UpstreamJSONClient(upstream_port, bearer_token)
        self.indexes = {}
        self._search_indexes = {}
        self._vector_cache = {}
        self._vector_cache_files = {}
        self._embedding_cache = {}
        self._rerank_supported = None
        self.max_embed_batch_tokens = self._env_int("BROKER_GPU_SERVICE_EMBED_BATCH_TOKENS", 2048, minimum=64)
        self.max_embed_batch_items = self._env_int("BROKER_GPU_SERVICE_EMBED_BATCH_ITEMS", 16, minimum=1)
        self.max_embed_segment_tokens = self._env_int("BROKER_GPU_SERVICE_EMBED_SEGMENT_TOKENS", 256, minimum=32)
        self.cache_dir = self._prepare_cache_dir(cache_dir)

    @staticmethod
    def _invalidate_search_state(record):
        if isinstance(record, dict):
            record.pop("_search_matrix", None)
            record.pop("_search_ids", None)
            record.pop("_search_index", None)

    @staticmethod
    def _prepare_search_state(record):
        if not isinstance(record, dict):
            return None, None, None
        cached_matrix = record.get("_search_matrix")
        cached_ids = record.get("_search_ids")
        cached_index = record.get("_search_index")
        if cached_ids is not None and (cached_matrix is not None or cached_index is not None):
            return cached_matrix, cached_ids, cached_index
        documents = record.get("documents") or ()
        identifiers = []
        vectors = []
        for document in documents:
            identifier = str(document.get("id") or "")
            vector = document.get("vector")
            if not identifier or not isinstance(vector, list) or not vector:
                continue
            identifiers.append(identifier)
            vectors.append(vector)
        if np is not None and vectors:
            matrix = np.asarray(vectors, dtype=np.float32)
            if matrix.ndim != 2:
                matrix = None
        else:
            matrix = None
        search_index = None
        if faiss is not None and matrix is not None and len(identifiers):
            try:
                search_index = faiss.IndexFlatIP(int(matrix.shape[1]))
                search_index.add(matrix)
            except Exception:  # pragma: no cover - defensive fallback
                search_index = None
        record["_search_matrix"] = matrix
        record["_search_ids"] = identifiers
        record["_search_index"] = search_index
        return matrix, identifiers, search_index

    @staticmethod
    def _index_key(payload):
        return (
            str(payload.get("model_profile") or ""),
            str(payload.get("index_fingerprint") or ""),
        )

    @staticmethod
    def _prepare_cache_dir(path):
        if not path:
            return None
        cache_dir = Path(path).expanduser().resolve(strict=False)
        cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(cache_dir, 0o700)
        return cache_dir

    @staticmethod
    def _env_int(name, fallback, *, minimum=1):
        raw = os.environ.get(name)
        if raw not in (None, ""):
            try:
                return max(minimum, int(raw))
            except (TypeError, ValueError):
                pass
        return max(minimum, int(fallback))

    @staticmethod
    def _cache_digest(key):
        return hashlib.sha256("\0".join(key).encode("utf-8", errors="replace")).hexdigest()

    def _cache_path(self, key):
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"semantic-index-{self._cache_digest(key)}.json.gz"

    def _cache_alias_path(self, key):
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"semantic-index-alias-{self._cache_digest(key)}.json"

    def _status_cache_path(self, key):
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"semantic-index-status-{self._cache_digest(key)}.json"

    def _faiss_cache_path(self, key):
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"semantic-index-{self._cache_digest(key)}.faiss"

    def _matrix_cache_path(self, key):
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"semantic-index-{self._cache_digest(key)}.npy"

    def _legacy_vector_cache_path(self, key):
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"semantic-vector-{self._cache_digest(key)}.json.gz"

    def _vector_cache_path(self, model):
        if self.cache_dir is None:
            return None
        model_digest = hashlib.sha256(str(model).encode("utf-8", errors="replace")).hexdigest()[:24]
        return self.cache_dir / f"semantic-vectors-{model_digest}.json.gz"

    @staticmethod
    def _vector_cache_shard(content_fingerprint):
        content_fingerprint = str(content_fingerprint or "").strip()
        if content_fingerprint.startswith("sha256:"):
            digest = content_fingerprint.removeprefix("sha256:")
            if len(digest) >= 2 and all(char in "0123456789abcdefABCDEF" for char in digest[:2]):
                return digest[:2].lower()
        return hashlib.sha256(content_fingerprint.encode("utf-8", errors="replace")).hexdigest()[:2]

    def _vector_cache_shard_path(self, model, shard):
        if self.cache_dir is None:
            return None
        model_digest = hashlib.sha256(str(model).encode("utf-8", errors="replace")).hexdigest()[:24]
        return self.cache_dir / f"semantic-vectors-{model_digest}-{str(shard).lower()}.json.gz"

    def _vector_cache_entry_path(self, key):
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"semantic-vector-entry-{self._cache_digest((str(key[0]), str(key[1])))}.json.gz"

    def _embedding_cache_path(self, key):
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"embedding-vector-{self._cache_digest(key)}.json.gz"

    def _write_cache(self, key, record):
        path = self._cache_path(key)
        if path is None:
            return
        alias_path = self._cache_alias_path(key)
        if alias_path is not None:
            alias_path.unlink(missing_ok=True)
        if self._can_reconstruct_index_from_status(record):
            path.unlink(missing_ok=True)
        else:
            payload = {
                "schema": "gpu_semantic_index_v3",
                "key": {"model_profile": key[0], "index_fingerprint": key[1]},
                "record": {
                    "ready": bool(record.get("ready")),
                    "model": str(record.get("model") or ""),
                    "documents": [
                        {
                            "id": str(document.get("id") or ""),
                            "vector": list(document.get("vector") or []),
                            "content_fingerprint": str(document.get("content_fingerprint") or ""),
                        }
                        for document in (record.get("documents") or [])
                    ],
                },
            }
            fd, temp_path = tempfile.mkstemp(prefix=".semantic-index-", suffix=".tmp", dir=path.parent)
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb", closefd=True) as handle:
                    with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as zipped:
                        zipped.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, path)
                os.chmod(path, 0o600)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temp_path)
        self._write_status_cache(key, record)
        self._write_faiss_cache(key, record)
        self._write_matrix_cache(key, record)

    def _clone_cache(self, source_key, target_key):
        source_path = self._cache_path(source_key)
        target_path = self._cache_path(target_key)
        alias_path = self._cache_alias_path(target_key)
        if source_path is None or target_path is None or alias_path is None:
            return False
        if source_key == target_key:
            return True
        source_status = self._load_status_cache(source_key)
        if source_status is None:
            source_record = self._get_index(source_key)
            if source_record is None:
                return False
            source_status_record = {
                "ready": bool(source_record.get("ready")),
                "model": str(source_record.get("model") or ""),
                "documents": [dict(document) for document in (source_record.get("documents") or ())],
            }
        else:
            source_status_record = {
                "ready": bool(source_status.get("ready")),
                "model": str(source_status.get("model") or ""),
                "documents": [
                    {
                        "id": str(item.get("id") or ""),
                        "content_fingerprint": str(item.get("content_fingerprint") or ""),
                    }
                    for item in (source_status.get("documents_meta") or ())
                    if str(item.get("id") or "") and str(item.get("content_fingerprint") or "")
                ],
            }
        target_path.unlink(missing_ok=True)
        payload = {
            "schema": "gpu_semantic_index_alias_v1",
            "source": {
                "model_profile": str(source_key[0]),
                "index_fingerprint": str(source_key[1]),
            },
        }
        fd, temp_path = tempfile.mkstemp(prefix=".semantic-index-alias-", suffix=".tmp", dir=alias_path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8", closefd=True) as handle:
                json.dump(payload, handle, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, alias_path)
            os.chmod(alias_path, 0o600)
            self._write_status_cache(target_key, source_status_record)
            return True
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_path)

    def _write_faiss_cache(self, key, record):
        path = self._faiss_cache_path(key)
        if path is None:
            return
        search_index = record.get("_search_index") if isinstance(record, dict) else None
        if faiss is None or search_index is None or not hasattr(faiss, "serialize_index"):
            path.unlink(missing_ok=True)
            return
        try:
            serialized = faiss.serialize_index(search_index)
        except Exception:  # pragma: no cover - optional acceleration only
            path.unlink(missing_ok=True)
            return
        if hasattr(serialized, "tobytes"):
            serialized = serialized.tobytes()
        else:
            serialized = bytes(serialized)
        _write_bytes_if_changed(path, serialized, prefix=".semantic-index-faiss-", suffix=".tmp")

    def _write_matrix_cache(self, key, record):
        path = self._matrix_cache_path(key)
        if path is None:
            return
        matrix = record.get("_search_matrix") if isinstance(record, dict) else None
        search_index = record.get("_search_index") if isinstance(record, dict) else None
        if np is None or matrix is None or search_index is not None:
            path.unlink(missing_ok=True)
            return
        buffer = io.BytesIO()
        np.save(buffer, matrix, allow_pickle=False)
        _write_bytes_if_changed(path, buffer.getvalue(), prefix=".semantic-index-matrix-", suffix=".tmp")

    def _load_faiss_cache(self, key):
        path = self._faiss_cache_path(key)
        if path is None or not path.exists() or faiss is None or not hasattr(faiss, "deserialize_index"):
            return None
        try:
            payload = path.read_bytes()
            return faiss.deserialize_index(payload)
        except Exception:  # pragma: no cover - optional acceleration only
            with contextlib.suppress(OSError):
                path.unlink()
            return None

    def _load_matrix_cache(self, key, *, _seen=None):
        _seen = set(_seen or ())
        if key in _seen:
            return None
        _seen.add(key)
        path = self._matrix_cache_path(key)
        alias_path = self._cache_alias_path(key)
        if path is None or np is None:
            return None
        if not path.exists():
            if alias_path is None or not alias_path.exists():
                return None
            try:
                payload = json.loads(alias_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                alias_path.unlink(missing_ok=True)
                return None
            if not isinstance(payload, dict) or payload.get("schema") != "gpu_semantic_index_alias_v1":
                alias_path.unlink(missing_ok=True)
                return None
            source = payload.get("source") or {}
            source_key = (
                str(source.get("model_profile") or ""),
                str(source.get("index_fingerprint") or ""),
            )
            if not source_key[1]:
                alias_path.unlink(missing_ok=True)
                return None
            return self._load_matrix_cache(source_key, _seen=_seen)
        try:
            with path.open("rb") as handle:
                matrix = np.load(handle, allow_pickle=False)
            if getattr(matrix, "ndim", None) != 2:
                raise ValueError("search matrix must be rank 2")
            return matrix.astype(np.float32, copy=False)
        except Exception:
            with contextlib.suppress(OSError):
                path.unlink()
            return None

    def _load_cache(self, key, *, _seen=None):
        _seen = set(_seen or ())
        if key in _seen:
            return None
        _seen.add(key)
        path = self._cache_path(key)
        alias_path = self._cache_alias_path(key)
        if path is None:
            return None
        if not path.exists():
            if alias_path is None or not alias_path.exists():
                return None
            try:
                payload = json.loads(alias_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                alias_path.unlink(missing_ok=True)
                return None
            if not isinstance(payload, dict) or payload.get("schema") != "gpu_semantic_index_alias_v1":
                alias_path.unlink(missing_ok=True)
                return None
            source = payload.get("source") or {}
            source_key = (
                str(source.get("model_profile") or ""),
                str(source.get("index_fingerprint") or ""),
            )
            if not source_key[1]:
                alias_path.unlink(missing_ok=True)
                return None
            loaded = self._load_cache(source_key, _seen=_seen)
            if loaded is None:
                alias_path.unlink(missing_ok=True)
                return None
            normalized = {
                "ready": bool(loaded.get("ready")),
                "model": str(loaded.get("model") or ""),
                "documents": [dict(document) for document in (loaded.get("documents") or ())],
            }
            self._invalidate_search_state(normalized)
            search_index = self._load_faiss_cache(source_key)
            if search_index is not None:
                normalized["_search_index"] = search_index
                normalized["_search_ids"] = [str(document.get("id") or "") for document in normalized["documents"]]
            return normalized
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            with contextlib.suppress(OSError):
                path.unlink()
            return None
        if not isinstance(payload, dict) or payload.get("schema") not in {"gpu_semantic_index_v1", "gpu_semantic_index_v2", "gpu_semantic_index_v3"}:
            return None
        record = payload.get("record")
        cache_key = payload.get("key") or {}
        schema = str(payload.get("schema") or "")
        if (
            not isinstance(record, dict)
            or str(cache_key.get("model_profile") or "") != key[0]
            or str(cache_key.get("index_fingerprint") or "") != key[1]
        ):
            return None
        documents = record.get("documents")
        if not isinstance(documents, list):
            return None
        normalized = {"ready": bool(record.get("ready")), "model": str(record.get("model") or ""), "documents": []}
        for document in documents:
            if not isinstance(document, dict):
                return None
            identifier = str(document.get("id") or "")
            vector = document.get("vector")
            if not identifier or not isinstance(vector, list) or not vector:
                return None
            normalized_document = {
                "id": identifier,
                "vector": [float(value) for value in vector],
            }
            content_fingerprint = str(document.get("content_fingerprint") or "")
            if content_fingerprint:
                normalized_document["content_fingerprint"] = content_fingerprint
            if schema == "gpu_semantic_index_v1":
                text = document.get("text")
                if not isinstance(text, str):
                    return None
                normalized_document["text"] = text
            normalized["documents"].append(normalized_document)
        self._invalidate_search_state(normalized)
        search_index = self._load_faiss_cache(key)
        if search_index is not None:
            normalized["_search_index"] = search_index
            normalized["_search_ids"] = [str(document.get("id") or "") for document in normalized["documents"]]
        return normalized

    def _write_status_cache(self, key, record):
        path = self._status_cache_path(key)
        if path is None:
            return
        payload = {
            "schema": "gpu_semantic_index_status_v3",
            "key": {"model_profile": key[0], "index_fingerprint": key[1]},
            "record": {
                "ready": bool(record.get("ready")),
                "document_count": len(record.get("documents") or []),
                "model": str(record.get("model") or ""),
                "identifiers": [
                    str(document.get("id") or "")
                    for document in (record.get("documents") or [])
                    if str(document.get("id") or "")
                ],
                "documents_meta": [
                    {
                        "id": str(document.get("id") or ""),
                        "content_fingerprint": str(document.get("content_fingerprint") or ""),
                    }
                    for document in (record.get("documents") or [])
                    if str(document.get("id") or "") and str(document.get("content_fingerprint") or "")
                ],
            },
        }
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        _write_bytes_if_changed(path, encoded, prefix=".semantic-index-status-", suffix=".tmp")

    def _load_status_cache(self, key, *, _seen=None):
        _seen = set(_seen or ())
        if key in _seen:
            return None
        _seen.add(key)
        path = self._status_cache_path(key)
        alias_path = self._cache_alias_path(key)
        if path is None:
            return None
        if not path.exists():
            if alias_path is None or not alias_path.exists():
                return None
            try:
                payload = json.loads(alias_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                alias_path.unlink(missing_ok=True)
                return None
            if not isinstance(payload, dict) or payload.get("schema") != "gpu_semantic_index_alias_v1":
                alias_path.unlink(missing_ok=True)
                return None
            source = payload.get("source") or {}
            source_key = (
                str(source.get("model_profile") or ""),
                str(source.get("index_fingerprint") or ""),
            )
            if not source_key[1]:
                alias_path.unlink(missing_ok=True)
                return None
            return self._load_status_cache(source_key, _seen=_seen)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            with contextlib.suppress(OSError):
                path.unlink()
            return None
        if not isinstance(payload, dict) or payload.get("schema") not in {"gpu_semantic_index_status_v1", "gpu_semantic_index_status_v2", "gpu_semantic_index_status_v3"}:
            return None
        cache_key = payload.get("key") or {}
        record = payload.get("record") or {}
        if (
            str(cache_key.get("model_profile") or "") != key[0]
            or str(cache_key.get("index_fingerprint") or "") != key[1]
            or not isinstance(record, dict)
        ):
            return None
        identifiers = record.get("identifiers")
        if not isinstance(identifiers, list):
            identifiers = []
        documents_meta = record.get("documents_meta")
        if not isinstance(documents_meta, list):
            documents_meta = []
        return {
            "ready": bool(record.get("ready")),
            "document_count": int(record.get("document_count") or 0),
            "model": str(record.get("model") or ""),
            "identifiers": [str(identifier) for identifier in identifiers if str(identifier)],
            "documents_meta": [
                {
                    "id": str(item.get("id") or ""),
                    "content_fingerprint": str(item.get("content_fingerprint") or ""),
                }
                for item in documents_meta
                if isinstance(item, dict) and str(item.get("id") or "") and str(item.get("content_fingerprint") or "")
            ],
        }

    def _write_vector_cache_file(self, model, records, *, shard=None):
        path = self._vector_cache_shard_path(model, shard) if shard is not None else self._vector_cache_path(model)
        if path is None:
            return
        payload = {
            "schema": "gpu_semantic_vector_v4" if shard is not None else "gpu_semantic_vector_v2",
            "model": str(model),
            "records": {
                str(content_fingerprint): {
                    "text_hash": str(record.get("text_hash") or ""),
                    "vector": list(record.get("vector") or []),
                }
                for content_fingerprint, record in records.items()
            },
        }
        if shard is not None:
            payload["shard"] = str(shard).lower()
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as zipped:
            zipped.write(encoded)
        _write_bytes_if_changed(path, buffer.getvalue(), prefix=".semantic-vector-", suffix=".tmp")

    def _write_vector_cache_entry(self, key, record):
        path = self._vector_cache_entry_path(key)
        if path is None:
            return
        payload = {
            "schema": "gpu_semantic_vector_v3",
            "key": {"model": str(key[0]), "content_fingerprint": str(key[1])},
            "record": {
                "text_hash": str(record.get("text_hash") or ""),
                "vector": list(record.get("vector") or []),
            },
        }
        fd, temp_path = tempfile.mkstemp(prefix=".semantic-vector-entry-", suffix=".tmp", dir=path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb", closefd=True) as handle:
                with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as zipped:
                    zipped.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            os.chmod(path, 0o600)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_path)

    def _load_legacy_vector_cache(self, key):
        path = self._legacy_vector_cache_path(key)
        if path is None or not path.exists():
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            with contextlib.suppress(OSError):
                path.unlink()
            return None
        if not isinstance(payload, dict) or payload.get("schema") != "gpu_semantic_vector_v1":
            return None
        cache_key = payload.get("key") or {}
        record = payload.get("record")
        if (
            not isinstance(record, dict)
            or str(cache_key.get("model") or "") != key[0]
            or str(cache_key.get("content_fingerprint") or "") != key[1]
        ):
            return None
        vector = record.get("vector")
        if not isinstance(vector, list) or not vector:
            return None
        try:
            normalized = {
                "text_hash": str(record.get("text_hash") or ""),
                "vector": [float(value) for value in vector],
            }
        except (TypeError, ValueError):
            return None
        self._vector_cache[key] = normalized
        return normalized

    def _load_vector_cache_entry(self, key):
        path = self._vector_cache_entry_path(key)
        if path is None or not path.exists():
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            with contextlib.suppress(OSError):
                path.unlink()
            return None
        if not isinstance(payload, dict) or payload.get("schema") != "gpu_semantic_vector_v3":
            return None
        cache_key = payload.get("key") or {}
        record = payload.get("record")
        if (
            not isinstance(record, dict)
            or str(cache_key.get("model") or "") != str(key[0])
            or str(cache_key.get("content_fingerprint") or "") != str(key[1])
        ):
            return None
        vector = record.get("vector")
        if not isinstance(vector, list) or not vector:
            return None
        try:
            normalized = {
                "text_hash": str(record.get("text_hash") or ""),
                "vector": [float(value) for value in vector],
            }
        except (TypeError, ValueError):
            return None
        self._vector_cache[key] = normalized
        return normalized

    def _load_vector_cache_file(self, model, *, shard=None):
        path = self._vector_cache_shard_path(model, shard) if shard is not None else self._vector_cache_path(model)
        if path is None or not path.exists():
            return {}
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            with contextlib.suppress(OSError):
                path.unlink()
            return {}
        expected_schemas = {"gpu_semantic_vector_v2"} if shard is None else {"gpu_semantic_vector_v4"}
        if (
            not isinstance(payload, dict)
            or str(payload.get("schema") or "") not in expected_schemas
            or str(payload.get("model") or "") != str(model)
        ):
            return {}
        if shard is not None and str(payload.get("shard") or "").lower() != str(shard).lower():
            return {}
        records = payload.get("records")
        if not isinstance(records, dict):
            return {}
        normalized = {}
        for content_fingerprint, record in records.items():
            if not isinstance(record, dict):
                return {}
            vector = record.get("vector")
            if not isinstance(vector, list) or not vector:
                return {}
            try:
                normalized[str(content_fingerprint)] = {
                    "text_hash": str(record.get("text_hash") or ""),
                    "vector": [float(value) for value in vector],
                }
            except (TypeError, ValueError):
                return {}
        cache_key = (str(model), str(shard).lower() if shard is not None else "")
        self._vector_cache_files[cache_key] = normalized
        for content_fingerprint, record in normalized.items():
            self._vector_cache[(str(model), content_fingerprint)] = record
        return normalized

    def _write_vector_cache_batch(self, entries):
        if not entries:
            return
        by_model_shard = {}
        for key, record in entries:
            model = str(key[0])
            content_fingerprint = str(key[1])
            shard = self._vector_cache_shard(content_fingerprint)
            shard_records = by_model_shard.setdefault((model, shard), {})
            shard_records[content_fingerprint] = {
                "text_hash": str(record.get("text_hash") or ""),
                "vector": list(record.get("vector") or []),
            }
        for (model, shard), records in by_model_shard.items():
            persisted = dict(self._vector_cache_files.get((model, shard)) or {})
            if not persisted:
                persisted = dict(self._load_vector_cache_file(model, shard=shard) or {})
            if not persisted:
                monolithic = self._vector_cache_files.get((model, ""))
                if monolithic is None:
                    monolithic = self._load_vector_cache_file(model)
                for content_fingerprint, prior in (monolithic or {}).items():
                    if self._vector_cache_shard(content_fingerprint) == shard:
                        persisted[content_fingerprint] = dict(prior)
            persisted.update(records)
            self._vector_cache_files[(model, shard)] = persisted
            self._write_vector_cache_file(model, persisted, shard=shard)
        for key, _record in entries:
            path = self._vector_cache_entry_path(key)
            if path is not None:
                path.unlink(missing_ok=True)

    def _write_embedding_cache(self, key, vector):
        path = self._embedding_cache_path(key)
        if path is None:
            return
        payload = {
            "schema": "gpu_embedding_vector_v1",
            "key": {"model": key[0], "text_hash": key[1]},
            "vector": list(vector),
        }
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as zipped:
            zipped.write(encoded)
        _write_bytes_if_changed(path, buffer.getvalue(), prefix=".embedding-vector-", suffix=".tmp")

    def _load_embedding_cache(self, key):
        path = self._embedding_cache_path(key)
        if path is None or not path.exists():
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            with contextlib.suppress(OSError):
                path.unlink()
            return None
        if not isinstance(payload, dict) or payload.get("schema") != "gpu_embedding_vector_v1":
            return None
        cache_key = payload.get("key") or {}
        vector = payload.get("vector")
        if (
            str(cache_key.get("model") or "") != key[0]
            or str(cache_key.get("text_hash") or "") != key[1]
            or not isinstance(vector, list)
            or not vector
        ):
            return None
        try:
            normalized = [float(value) for value in vector]
        except (TypeError, ValueError):
            return None
        self._embedding_cache[key] = normalized
        return normalized

    def _get_index(self, key):
        record = self.indexes.get(key)
        if record is not None:
            return record
        cached = self._load_cache(key)
        if cached is not None:
            self.indexes[key] = cached
            return cached
        status_record = self._load_status_cache(key)
        documents_meta = list((status_record or {}).get("documents_meta") or ())
        model = str((status_record or {}).get("model") or "")
        if status_record and status_record.get("ready") and model and documents_meta:
            documents = []
            for item in documents_meta:
                cache_key = (model, str(item.get("content_fingerprint") or ""))
                cached_vector = self._get_vector_cache(cache_key)
                if cached_vector is None:
                    return None
                documents.append(
                    {
                        "id": str(item.get("id") or ""),
                        "content_fingerprint": str(item.get("content_fingerprint") or ""),
                        "vector": list(cached_vector.get("vector") or []),
                    }
                )
            reconstructed = {"ready": True, "model": model, "documents": documents}
            self.indexes[key] = reconstructed
            return reconstructed
        return None

    @staticmethod
    def _can_reconstruct_index_from_status(record):
        if not isinstance(record, dict):
            return False
        documents = record.get("documents")
        if not isinstance(documents, list):
            return False
        return all(
            str(document.get("id") or "")
            and str(document.get("content_fingerprint") or "")
            and isinstance(document.get("vector"), list)
            and document.get("vector")
            for document in documents
        )

    def _get_index_status(self, key):
        record = self.indexes.get(key)
        if record is not None:
            return {
                "ready": bool(record.get("ready")),
                "document_count": len(record.get("documents") or []),
                "model": str(record.get("model") or ""),
                "identifiers": [
                    str(document.get("id") or "")
                    for document in (record.get("documents") or [])
                    if str(document.get("id") or "")
                ],
            }
        return self._load_status_cache(key)

    def _get_search_index(self, key):
        record = self.indexes.get(key)
        if record is not None:
            return record
        cached = self._search_indexes.get(key)
        if cached is not None:
            return cached
        status_record = self._load_status_cache(key)
        identifiers = list((status_record or {}).get("identifiers") or ())
        if status_record and status_record.get("ready") and identifiers:
            search_index = self._load_faiss_cache(key)
            matrix = self._load_matrix_cache(key)
            if search_index is not None or matrix is not None:
                cached = {
                    "ready": True,
                    "model": str(status_record.get("model") or ""),
                    "documents": [],
                    "_search_ids": identifiers,
                    "_search_index": search_index,
                    "_search_matrix": matrix,
                }
                self._search_indexes[key] = cached
                return cached
        return self._get_index(key)

    def _get_vector_cache(self, key):
        record = self._vector_cache.get(key)
        if record is not None:
            return record
        loaded = self._load_vector_cache_entry(key)
        if loaded is not None:
            return loaded
        model = str(key[0])
        shard = self._vector_cache_shard(key[1])
        cached_file = self._vector_cache_files.get((model, shard))
        if cached_file is None:
            cached_file = self._load_vector_cache_file(model, shard=shard)
        loaded = cached_file.get(str(key[1]))
        if loaded is not None:
            self._vector_cache[key] = loaded
            return loaded
        cached_file = self._vector_cache_files.get((model, ""))
        if cached_file is None:
            cached_file = self._load_vector_cache_file(model)
        loaded = cached_file.get(str(key[1]))
        if loaded is not None:
            self._vector_cache[key] = loaded
            return loaded
        return self._load_legacy_vector_cache(key)

    def _get_embedding_cache(self, key):
        vector = self._embedding_cache.get(key)
        if vector is not None:
            return list(vector)
        loaded = self._load_embedding_cache(key)
        if loaded is not None:
            return list(loaded)
        return None

    @staticmethod
    def _document_text(document):
        if not isinstance(document, dict):
            raise ValueError("documents must be JSON objects")
        text = document.get("text")
        identifier = str(document.get("id") or document.get("chunk_id") or "")
        if not isinstance(text, str) or not identifier:
            raise ValueError("documents require string id and text")
        return identifier, text

    @staticmethod
    def _document_content_fingerprint(document, text):
        metadata = document.get("metadata") if isinstance(document, dict) else None
        if isinstance(metadata, dict):
            content_hash = str(metadata.get("content_hash") or "").strip()
            if content_hash:
                return content_hash
        return "sha256:" + hashlib.sha256(str(text).encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _merge_vectors(vectors):
        if not vectors:
            raise RuntimeError("cannot merge zero embedding vectors")
        if len(vectors) == 1:
            return list(vectors[0])
        dimensions = len(vectors[0])
        merged = [0.0] * dimensions
        for vector in vectors:
            if len(vector) != dimensions:
                raise RuntimeError("embedding dimensions do not match")
            for index, value in enumerate(vector):
                merged[index] += float(value)
        return _normalize_vector(merged)

    def _split_text_for_embedding(self, text):
        text = str(text)
        max_chars = max(64, int(self.max_embed_segment_tokens) * 4)
        if len(text) <= max_chars:
            return [text]
        parts = []
        start = 0
        while start < len(text):
            limit = min(len(text), start + max_chars)
            stop = limit
            if limit < len(text):
                newline = text.rfind("\n", start, limit)
                whitespace = text.rfind(" ", start, limit)
                boundary = max(newline, whitespace)
                if boundary > start + (max_chars // 2):
                    stop = boundary + 1
            part = text[start:stop].strip()
            if part:
                parts.append(part)
            start = max(stop, start + 1)
        return parts or [text[:max_chars]]

    def _embed(self, texts, model, *, persist_segment_cache=True):
        segment_texts = []
        segment_counts = []
        segment_keys = []
        for text in texts:
            segments = self._split_text_for_embedding(text)
            segment_texts.extend(segments)
            segment_counts.append(len(segments))
            segment_keys.extend(
                (model, hashlib.sha256(segment.encode("utf-8", errors="replace")).hexdigest()) for segment in segments
            )
        segment_vectors = [None] * len(segment_texts)
        batch = []
        batch_positions = []
        batch_tokens = 0

        def flush():
            nonlocal batch, batch_positions, batch_tokens, segment_vectors
            if not batch:
                return
            response = self.upstream.post(
                "/v1/embeddings",
                {"model": model, "input": list(batch), "encoding_format": "float"},
            )
            data = response.get("data")
            if not isinstance(data, list):
                raise RuntimeError("embedding response is missing data")
            ordered = sorted(data, key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0)
            for item, position in zip(ordered, batch_positions):
                vector = item.get("embedding") if isinstance(item, dict) else None
                if not isinstance(vector, list) or not vector:
                    raise RuntimeError("embedding response contains an invalid vector")
                normalized = _normalize_vector([float(value) for value in vector])
                segment_vectors[position] = normalized
                cache_key = segment_keys[position]
                self._embedding_cache[cache_key] = list(normalized)
                if persist_segment_cache:
                    self._write_embedding_cache(cache_key, normalized)
            if not batch_positions:
                raise RuntimeError("embedding response is empty")
            batch = []
            batch_positions = []
            batch_tokens = 0

        for index, segment in enumerate(segment_texts):
            cache_key = segment_keys[index]
            cached = self._get_embedding_cache(cache_key)
            if cached is not None:
                segment_vectors[index] = cached
                continue
            estimated_tokens = max(1, len(segment) // 4)
            if batch and (
                len(batch) >= self.max_embed_batch_items
                or batch_tokens + estimated_tokens > self.max_embed_batch_tokens
            ):
                flush()
            batch.append(segment)
            batch_positions.append(index)
            batch_tokens += estimated_tokens
        flush()

        if any(vector is None for vector in segment_vectors):
            raise RuntimeError("embedding response count does not match request")

        vectors = []
        cursor = 0
        for count in segment_counts:
            vectors.append(self._merge_vectors(segment_vectors[cursor : cursor + count]))
            cursor += count
        if len(vectors) != len(texts):
            raise RuntimeError("embedding response count does not match request")
        return vectors

    def index_status(self, payload):
        key = self._index_key(payload)
        record = self._get_index_status(key)
        document_count = int(payload.get("document_count") or 0)
        cached_count = (
            len(record.get("documents") or ())
            if record and "documents" in record
            else int(record.get("document_count") or 0) if record else 0
        )
        ready = bool(record and record.get("ready") and cached_count == document_count)
        return 200, {"ready": ready, "document_count": cached_count}

    def index_upsert(self, payload):
        key = self._index_key(payload)
        model = str(payload.get("model") or "")
        documents = payload.get("documents")
        if not model:
            raise ValueError("index_upsert requires model")
        if not isinstance(documents, list):
            raise ValueError("index_upsert requires documents")
        removed_document_ids = {str(value) for value in (payload.get("removed_document_ids") or []) if str(value)}
        base_reused = False
        base_cache_key = None
        if payload.get("replace"):
            base_fingerprint = str(payload.get("base_index_fingerprint") or "")
            base_documents = []
            if base_fingerprint:
                base_cache_key = (str(payload.get("model_profile") or ""), base_fingerprint)
                base_record = self._get_index(base_cache_key)
                if base_record and base_record.get("ready"):
                    base_documents = [dict(document) for document in (base_record.get("documents") or [])]
                    base_reused = True
            self.indexes[key] = {"ready": False, "documents": base_documents, "model": model}
            self._search_indexes.pop(key, None)
            self._invalidate_search_state(self.indexes[key])
        record = self.indexes.setdefault(key, {"ready": False, "documents": [], "model": model})
        self._search_indexes.pop(key, None)
        if removed_document_ids:
            record["documents"] = [
                dict(document)
                for document in (record.get("documents") or [])
                if str(document.get("id") or "") not in removed_document_ids
            ]
        prepared = []
        missing_positions = []
        missing_texts = []
        for document in documents:
            identifier, text = self._document_text(document)
            content_fingerprint = self._document_content_fingerprint(document, text)
            text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
            cache_key = (model, content_fingerprint)
            cached = self._get_vector_cache(cache_key)
            vector = None
            if cached is not None and str(cached.get("text_hash") or "") == text_hash:
                vector = list(cached["vector"])
            prepared.append(
                {
                    "id": identifier,
                    "text": text,
                    "text_hash": text_hash,
                    "content_fingerprint": content_fingerprint,
                    "vector": vector,
                }
            )
            if vector is None:
                missing_positions.append(len(prepared) - 1)
                missing_texts.append(text)
        if missing_texts:
            # Full-document index builds persist the coarser content-fingerprint
            # vector cache below. Avoid also fsyncing every intermediate segment
            # embedding, which reduces cold-index write amplification without
            # affecting hot query/rerank reuse.
            vectors = self._embed(missing_texts, model, persist_segment_cache=False)
            pending_vector_cache_writes = []
            for position, vector in zip(missing_positions, vectors):
                prepared[position]["vector"] = vector
                vector_cache_key = (model, prepared[position]["content_fingerprint"])
                vector_cache_record = {
                    "text_hash": prepared[position]["text_hash"],
                    "vector": vector,
                }
                self._vector_cache[vector_cache_key] = vector_cache_record
                pending_vector_cache_writes.append((vector_cache_key, vector_cache_record))
            self._write_vector_cache_batch(pending_vector_cache_writes)
        by_id = {str(document.get("id") or ""): dict(document) for document in (record.get("documents") or [])}
        for item in prepared:
            by_id[item["id"]] = {
                "id": item["id"],
                "content_fingerprint": item["content_fingerprint"],
                "vector": item["vector"],
            }
        record["documents"] = list(by_id.values())
        self._invalidate_search_state(record)
        if payload.get("finalize"):
            record["ready"] = True
            self._prepare_search_state(record)
            cloned = False
            if (
                base_reused
                and base_cache_key is not None
                and not removed_document_ids
                and not documents
            ):
                cloned = self._clone_cache(base_cache_key, key)
            if not cloned:
                self._write_cache(key, record)
        reused_documents = len(prepared) - len(missing_positions)
        embedded_documents = len(missing_positions)
        return 200, {
            "accepted": True,
            "base_reused": base_reused,
            "reused_documents": reused_documents,
            "embedded_documents": embedded_documents,
            "total_documents": len(record.get("documents") or []),
        }

    def search(self, payload):
        key = self._index_key(payload)
        record = self._get_search_index(key)
        if not record or not record.get("ready"):
            return 200, {"results": []}
        model = str(payload.get("model") or record.get("model") or "")
        query = str(payload.get("query") or "")
        if not model or not query:
            raise ValueError("search requires model and query")
        query_vector = self._embed([query], model)[0]
        limit = max(1, int(payload.get("limit") or 10))
        matrix, identifiers, search_index = self._prepare_search_state(record)
        if faiss is not None and search_index is not None and identifiers and np is not None:
            query_array = np.asarray([query_vector], dtype=np.float32)
            candidate_count = min(limit, len(identifiers))
            scores, indices = search_index.search(query_array, candidate_count)
            ranked = []
            for score, index in zip(scores[0].tolist(), indices[0].tolist()):
                if index < 0 or index >= len(identifiers):
                    continue
                ranked.append({"id": identifiers[index], "score": float(score)})
            ranked.sort(key=lambda item: (-item["score"], item["id"]))
            return 200, {"results": ranked}
        if np is not None and matrix is not None and identifiers:
            query_array = np.asarray(query_vector, dtype=np.float32)
            scores = matrix @ query_array
            candidate_count = min(limit, len(identifiers))
            if candidate_count >= len(identifiers):
                selected = list(range(len(identifiers)))
            else:
                selected = np.argpartition(scores, -candidate_count)[-candidate_count:].tolist()
            selected.sort(key=lambda index: (-float(scores[index]), identifiers[index]))
            ranked = [
                {
                    "id": identifiers[index],
                    "score": float(scores[index]),
                }
                for index in selected
            ]
            return 200, {"results": ranked}
        ranked = []
        for document in record.get("documents") or ():
            ranked.append(
                {
                    "id": document["id"],
                    "score": _cosine_similarity(query_vector, document["vector"]),
                }
            )
        ranked.sort(key=lambda item: (-item["score"], item["id"]))
        return 200, {"results": ranked[:limit]}

    def _embedding_rerank(self, payload):
        model = str(payload.get("model") or "")
        query = str(payload.get("query") or "")
        documents = payload.get("documents")
        if not model or not query or not isinstance(documents, list):
            raise ValueError("rerank requires model, query, and documents")
        query_vector = self._embed([query], model)[0]
        doc_vectors = self._embed([str(document) for document in documents], model)
        results = []
        for index, vector in enumerate(doc_vectors):
            results.append({"index": index, "relevance_score": _cosine_similarity(query_vector, vector)})
        results.sort(key=lambda item: (-item["relevance_score"], item["index"]))
        top_n = int(payload.get("top_n") or len(results))
        return 200, {"results": results[:top_n]}

    def rerank(self, payload):
        if self._rerank_supported is not False:
            try:
                result = self.upstream.post("/v1/rerank", payload)
                self._rerank_supported = True
                return 200, result
            except RuntimeError as exc:
                message = str(exc).lower()
                if any(marker in message for marker in ("(404)", "(405)", "(501)", "not implemented")):
                    self._rerank_supported = False
                else:
                    raise
        return self._embedding_rerank(payload)

    def handle(self, path, payload):
        if path == "/v1/indexes/status":
            return self.index_status(payload)
        if path == "/v1/indexes/upsert":
            return self.index_upsert(payload)
        if path == "/v1/search":
            return self.search(payload)
        if path == "/v1/rerank":
            return self.rerank(payload)
        raise ValueError(f"unsupported retrieval endpoint: {path}")


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "local-ai-broker-gpu-proxy/1"

    def log_message(self, message, *args):
        sys.stderr.write("gpu-proxy: " + (message % args) + "\n")

    def _authorized(self):
        expected = "Bearer " + self.server.bearer_token
        provided = self.headers.get_all("Authorization") or []
        return len(provided) == 1 and secrets.compare_digest(provided[0], expected)

    def _send_json(self, status, payload):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _upstream_healthy(self):
        if self.server.runtime_process.poll() is not None:
            return False
        connection = http.client.HTTPConnection("127.0.0.1", self.server.upstream_port, timeout=5)
        try:
            connection.request(
                "GET",
                "/health",
                headers={"Authorization": "Bearer " + self.server.bearer_token},
            )
            response = connection.getresponse()
            response.read(4096)
            return 200 <= response.status < 300
        except (OSError, http.client.HTTPException):
            return False
        finally:
            connection.close()

    def _handle(self):
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return
        if not self.path.startswith("/") or self.path.startswith("//"):
            self._send_json(400, {"error": "invalid request target"})
            return
        if self.path.rstrip("/") == "/health":
            if self._upstream_healthy():
                self._send_json(200, {"status": "ok"})
            else:
                self._send_json(503, {"status": "failed"})
            return
        if self.headers.get_all("Transfer-Encoding"):
            self._send_json(400, {"error": "unsupported transfer encoding"})
            return
        content_lengths = self.headers.get_all("Content-Length") or []
        if len(content_lengths) > 1:
            self._send_json(400, {"error": "invalid content length"})
            return
        try:
            length = int(content_lengths[0]) if content_lengths else 0
        except (TypeError, ValueError):
            self._send_json(400, {"error": "invalid content length"})
            return
        if length < 0:
            self._send_json(400, {"error": "invalid content length"})
            return
        if length > MAX_BODY_BYTES:
            self._send_json(413, {"error": "request body too large"})
            return
        body = self.rfile.read(length) if length else None
        path_only = self.path.split("?", 1)[0]
        if self.server.retrieval_adapter is not None and path_only in RETRIEVAL_ENDPOINTS:
            try:
                payload = json.loads(body.decode("utf-8", errors="replace")) if body else {}
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                status, response = self.server.retrieval_adapter.handle(path_only, payload)
                self._send_json(status, response)
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": str(exc)})
            except RuntimeError as exc:
                self._send_json(502, {"error": str(exc)})
            return
        connection = http.client.HTTPConnection("127.0.0.1", self.server.upstream_port, timeout=300)
        connection_tokens = {
            token.strip().lower()
            for value in self.headers.get_all("Connection") or []
            for token in value.split(",")
            if token.strip()
        }
        blocked_request_headers = HOP_BY_HOP_HEADERS | connection_tokens | {
            "authorization",
            "content-length",
            "host",
        }
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in blocked_request_headers
        }
        # Runtimes that accept the configured {endpoint_token} can enforce the
        # same credential internally; runtimes that do not use auth ignore it.
        headers["Authorization"] = "Bearer " + self.server.bearer_token
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read(MAX_BODY_BYTES + 1)
            if len(payload) > MAX_BODY_BYTES:
                self._send_json(502, {"error": "upstream response too large"})
                return
            self.send_response(response.status)
            response_headers = response.getheaders()
            response_connection_tokens = {
                token.strip().lower()
                for key, value in response_headers
                if key.lower() == "connection"
                for token in value.split(",")
                if token.strip()
            }
            blocked_response_headers = HOP_BY_HOP_HEADERS | response_connection_tokens | {"content-length"}
            for key, value in response_headers:
                if key.lower() not in blocked_response_headers:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (OSError, http.client.HTTPException) as exc:
            self._send_json(502, {"error": f"upstream service unavailable: {exc}"})
        finally:
            connection.close()

    do_GET = _handle
    do_POST = _handle


def wait_for_runtime(process, port, endpoint_token, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"GPU runtime exited during startup with code {process.returncode}")
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
        try:
            connection.request(
                "GET",
                "/health",
                headers={"Authorization": f"Bearer {endpoint_token}"},
            )
            response = connection.getresponse()
            response.read(MAX_BODY_BYTES + 1)
            if 200 <= response.status < 300:
                return
        except (OSError, http.client.HTTPException):
            pass
        finally:
            connection.close()
        if time.monotonic() < deadline:
            time.sleep(0.5)
    raise RuntimeError("GPU runtime startup timed out")


def run(spec):
    deployment = spec.get("deployment") or {}
    placement = spec.get("placement") or {}
    gpu = placement.get("gpu") or {}
    required = {
        "service_id": spec.get("service_id"),
        "registry_path": spec.get("registry_path"),
        "registration_token": spec.get("registration_token"),
        "runtime": deployment.get("runtime"),
        "model": deployment.get("model"),
        "quantization": deployment.get("quantization"),
        "context_limit_tokens": deployment.get("context_limit_tokens"),
        "runtime_args": deployment.get("runtime_args"),
    }
    missing = [name for name, value in required.items() if value in (None, "", [])]
    if missing:
        raise ValueError("GPU service launch spec is missing: " + ", ".join(missing))

    internal_port = reserve_port("127.0.0.1")
    endpoint_token = secrets.token_urlsafe(48)
    runtime_args = substitute_runtime_args(
        deployment,
        internal_port,
        endpoint_token,
        int(gpu.get("count") or 0),
    )
    command = [str(deployment["runtime"]), *runtime_args]
    runtime_env = os.environ.copy()
    runtime_env.pop("BROKER_GPU_SERVICE_SPEC_PATH", None)
    process = subprocess.Popen(command, env=runtime_env)
    publisher = RegistryPublisher(spec["registry_path"], spec["service_id"], spec["registration_token"])
    proxy = None
    retrieval_adapter = None
    stop = threading.Event()

    def request_stop(_signum=None, _frame=None):
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        startup_timeout = max(1, int(os.environ.get("BROKER_GPU_SERVICE_RUNTIME_STARTUP_TIMEOUT_SECONDS", "600")))
        wait_for_runtime(process, internal_port, endpoint_token, startup_timeout)
        capabilities = {str(value).lower() for value in (spec.get("capabilities") or ())}
        if {
            "embeddings",
            "index_status",
            "index_upsert",
            "faiss_search",
            "rerank",
        }.issubset(capabilities):
            default_cache_root = Path(spec["registry_path"]).resolve(strict=False).parent / "semantic-index-cache"
            cache_dir = os.environ.get("BROKER_GPU_SERVICE_INDEX_CACHE_DIR") or str(default_cache_root / str(spec.get("tier") or "retrieval"))
            retrieval_adapter = RetrievalServiceAdapter(internal_port, endpoint_token, cache_dir=cache_dir)
        # Let the listening socket choose and retain the public port atomically;
        # reserving and reopening it leaves an avoidable bind race.
        proxy = AuthenticatedProxy(("0.0.0.0", 0), endpoint_token, process, internal_port, retrieval_adapter)
        external_port = int(proxy.server_address[1])
        proxy_thread = threading.Thread(target=proxy.serve_forever, name="gpu-auth-proxy", daemon=True)
        proxy_thread.start()
        endpoint_host = os.environ.get("BROKER_GPU_SERVICE_ENDPOINT_HOST") or socket.getfqdn()
        endpoint = f"http://{endpoint_host}:{external_port}"
        publisher.publish(endpoint, endpoint_token, os.environ.get("SLURM_JOB_ID", ""))

        heartbeat_interval = max(1, int(spec.get("heartbeat_interval_seconds") or 15))
        while not stop.wait(heartbeat_interval):
            if process.poll() is not None:
                raise RuntimeError(f"GPU runtime exited with code {process.returncode}")
            publisher.renew()
        return 0
    except BaseException as exc:
        with contextlib.suppress(Exception):
            publisher.mark_unhealthy(exc)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise
    finally:
        if proxy is not None:
            proxy.shutdown()
            proxy.server_close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description="Launch an authenticated broker GPU service")
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    spec_path = Path(args.spec)
    spec = load_json(spec_path)
    # The registration token is needed only in memory after startup.
    with contextlib.suppress(OSError):
        spec_path.unlink()
    try:
        return run(spec)
    finally:
        with contextlib.suppress(OSError):
            spec_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
