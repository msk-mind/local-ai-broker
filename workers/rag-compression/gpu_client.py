"""Authenticated clients and endpoint selection for broker GPU services."""

from __future__ import annotations

import json
import os
import importlib
from datetime import datetime, timezone


TRANSIENT_SINGLE_GPU_FAILURES = {
    "availability",
    "queue_delay",
    "timeout",
    "service_failure",
}
MULTIGPU_FAILURES = {
    "oom",
    "context_overflow",
    "invalid_synthesis",
    "model_limit",
}
MAX_INDEX_BATCH_BYTES = 8 * 1024 * 1024
DEFAULT_GPU_SERVICE_TIMEOUT_SECONDS = 60.0
DEFAULT_GPU_SERVICE_INDEX_TIMEOUT_SECONDS = 600.0
_REGISTRY_CACHE = {}


class _LazyModuleNamespace:
    def __init__(self, base_name, allowed):
        self._base_name = str(base_name)
        self._allowed = {str(value) for value in allowed}
        self._loaded = {}

    def __getattr__(self, name):
        if name not in self._allowed:
            raise AttributeError(name)
        module = self._loaded.get(name)
        if module is None:
            module = importlib.import_module(f"{self._base_name}.{name}")
            self._loaded[name] = module
        return module


urllib = _LazyModuleNamespace("urllib", {"request", "error", "parse"})


def sanitized_service_diagnostics(record):
    """Return only non-secret service identity fields safe for result traces."""

    if not isinstance(record, dict):
        return {}
    gpu = record.get("gpu") or {}
    try:
        gpu_count = int(gpu.get("count") or record.get("gpu_count") or 0)
    except (TypeError, ValueError):
        gpu_count = 0
    diagnostics = {
        "tier": str(record.get("tier") or ""),
        "slurm_job_id": str(record.get("slurm_job_id") or record.get("job_id") or ""),
        "gpu": {
            "type": str(gpu.get("type") or record.get("gpu_type") or ""),
            "count": gpu_count,
        },
        "model_profile": str(record.get("model_profile") or ""),
    }
    if not any(
        (
            diagnostics["tier"],
            diagnostics["slurm_job_id"],
            diagnostics["gpu"]["type"],
            diagnostics["gpu"]["count"],
            diagnostics["model_profile"],
        )
    ):
        return {}
    return diagnostics


class GPUServiceError(RuntimeError):
    def __init__(
        self,
        category,
        message,
        *,
        status_code=None,
        retryable=True,
        service_diagnostics=None,
    ):
        super().__init__(message)
        self.category = str(category or "service_failure")
        self.status_code = status_code
        self.retryable = bool(retryable)
        self.service_diagnostics = sanitized_service_diagnostics(service_diagnostics)


def classify_failure(message, status_code=None):
    text = str(message or "").lower()
    if status_code in {401, 403} or "unauthorized" in text or "forbidden" in text:
        return "authentication"
    if status_code == 429 or "queue" in text or "capacity" in text or "unavailable" in text:
        return "availability"
    if "timed out" in text or "timeout" in text or "deadline" in text:
        return "timeout"
    if "out of memory" in text or "cuda oom" in text or "oom" in text:
        return "oom"
    if "context" in text and any(term in text for term in ("length", "window", "overflow", "maximum")):
        return "context_overflow"
    if "model" in text and any(term in text for term in ("limit", "unsupported", "too large")):
        return "model_limit"
    if status_code in {404, 405}:
        return "service_failure"
    if status_code is not None and status_code >= 500:
        return "service_failure"
    return "service_failure"


def _parse_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _effective_lease_expiry(record):
    lease_expires = _parse_time(record.get("lease_expires_at") or record.get("lease_expires"))
    if lease_expires is None:
        return None
    absolute = _parse_time(record.get("absolute_lease_expires_at"))
    # An encoded Go zero time represents an omitted absolute bound for warm
    # P40 services, not an already-expired lease.
    if absolute is None or absolute.year <= 1:
        return lease_expires
    return min(lease_expires, absolute)


def _flatten_services(value):
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []
    services = []
    for tier, record in value.items():
        records = record if isinstance(record, list) else [record]
        for item in records:
            if isinstance(item, dict):
                normalized = dict(item)
                normalized.setdefault("tier", tier)
                services.append(normalized)
    return services


def services_from_execution_plan(execution_plan, task_params=None):
    """Read full service leases only from the broker-owned registry.

    ``task_params`` is intentionally ignored. It is retained in the signature
    for worker-call compatibility, but callers must not be able to choose an
    endpoint, registry file, or credential-bearing service record.
    """

    execution_plan = execution_plan or {}
    registry_path = execution_plan.get("gpu_service_registry_path") or os.environ.get(
        "BROKER_GPU_SERVICE_REGISTRY_PATH"
    )
    services = []
    if registry_path:
        try:
            registry_path = os.path.abspath(os.fspath(registry_path))
            stat = os.stat(registry_path)
            cache_key = (registry_path, int(stat.st_mtime_ns), int(stat.st_size))
            cached = _REGISTRY_CACHE.get(cache_key)
            if cached is None:
                with open(registry_path, "r", encoding="utf-8") as handle:
                    registry = json.load(handle)
                cached = tuple(_flatten_services(registry.get("records"))) if isinstance(registry, dict) and registry.get("schema") == "gpu_service_registry_v1" else ()
                _REGISTRY_CACHE.clear()
                _REGISTRY_CACHE[cache_key] = cached
            else:
                registry = None
            services.extend(dict(item) for item in cached if isinstance(item, dict))
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return services


def endpoint_is_healthy(record, *, now=None, health_interval_seconds=None):
    state = str(record.get("state") or "").lower()
    if state != "ready":
        return False
    if record.get("healthy") is False or record.get("health_error"):
        return False
    endpoint = str(record.get("endpoint") or record.get("base_url") or "")
    import urllib.parse

    parsed_endpoint = urllib.parse.urlparse(endpoint)
    if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.hostname:
        return False
    auth = record.get("endpoint_auth") or {}
    if str(auth.get("type") or "").lower() != "bearer" or not str(auth.get("bearer_token") or ""):
        return False
    now = now or datetime.now(timezone.utc)
    lease_expires = _effective_lease_expiry(record)
    if lease_expires is None or lease_expires <= now:
        return False
    heartbeat = _parse_time(record.get("heartbeat_at"))
    if heartbeat is None:
        return False
    max_age = max(1, float(health_interval_seconds or 15)) * 3
    if (now - heartbeat).total_seconds() > max_age:
        return False
    return True


def endpoint_capabilities(record):
    values = record.get("capabilities") or record.get("operations") or []
    if isinstance(values, str):
        values = [values]
    aliases = {
        "embeddings": "embed",
        "embedding": "embed",
        "reranking": "rerank",
        "faiss_search": "search",
        "vector_search": "search",
        "semantic_search": "search",
        "chat_completions": "chat",
        "chat_completion": "chat",
        "synthesis": "chat",
    }
    return {aliases.get(str(value).lower(), str(value).lower()) for value in values}


def select_endpoint(services, tier, capability, *, expected_gpu_count=None, health_interval_seconds=None):
    candidates = []
    for record in services:
        if str(record.get("tier") or "") != tier:
            continue
        if not endpoint_is_healthy(record, health_interval_seconds=health_interval_seconds):
            continue
        capabilities = endpoint_capabilities(record)
        if capabilities and capability not in capabilities:
            continue
        gpu = record.get("gpu") or {}
        count = int(gpu.get("count") or record.get("gpu_count") or 0)
        if expected_gpu_count is not None and count != expected_gpu_count:
            continue
        candidates.append(record)
    candidates.sort(key=lambda item: (str(item.get("lease_expires_at") or ""), str(item.get("id") or "")), reverse=True)
    return dict(candidates[0]) if candidates else None


def endpoint_diagnostics(record):
    if not record:
        return {}
    gpu = record.get("gpu") or {}
    return {
        "tier": str(record.get("tier") or ""),
        "job_id": str(record.get("slurm_job_id") or record.get("job_id") or ""),
        "gpu_count": int(gpu.get("count") or record.get("gpu_count") or 0),
        "gpu_type": str(gpu.get("type") or record.get("gpu_type") or ""),
        "model_profile": str(record.get("model_profile") or ""),
        "context_limit_tokens": int(record.get("context_limit_tokens") or 0),
    }


class GPUServiceClient:
    """Small OpenAI-compatible client that never logs endpoint credentials."""

    def __init__(self, record):
        self.record = dict(record or {})
        self.base_url = str(self.record.get("endpoint") or self.record.get("base_url") or "").rstrip("/")
        self.model_profile = str(self.record.get("model_profile") or "").strip()
        self.model = str(self.record.get("model") or "").strip()
        self.timeout_seconds = float(self.record.get("timeout_seconds") or DEFAULT_GPU_SERVICE_TIMEOUT_SECONDS)
        self.index_timeout_seconds = self._configured_timeout(
            "BROKER_GPU_SERVICE_INDEX_TIMEOUT_SECONDS",
            self.record.get("index_timeout_seconds"),
            DEFAULT_GPU_SERVICE_INDEX_TIMEOUT_SECONDS,
        )
        if not self.base_url:
            raise GPUServiceError("availability", "GPU service endpoint is unavailable")
        if not self.model_profile or not self.model:
            raise GPUServiceError("model_limit", "GPU service record must provide model_profile and model", retryable=False)

    def _configured_timeout(self, env_name, record_value, fallback):
        raw = os.environ.get(env_name)
        if raw not in (None, ""):
            try:
                return max(1.0, float(raw))
            except (TypeError, ValueError):
                pass
        if record_value not in (None, ""):
            try:
                return max(1.0, float(record_value))
            except (TypeError, ValueError):
                pass
        return max(1.0, float(fallback))

    def _token(self):
        auth = self.record.get("endpoint_auth") or {}
        if str(auth.get("type") or "").lower() != "bearer":
            raise GPUServiceError("authentication", "GPU service must use bearer authentication", retryable=False)
        token = auth.get("bearer_token")
        if not token:
            raise GPUServiceError("authentication", "GPU service bearer token is not configured", retryable=False)
        return str(token)

    def _route(self, operation, default):
        import urllib.parse

        routes = self.record.get("routes") or {}
        configured = routes.get(operation)
        if configured:
            configured = str(configured)
            if configured.startswith("http://") or configured.startswith("https://"):
                base = urllib.parse.urlparse(self.base_url)
                routed = urllib.parse.urlparse(configured)
                if (routed.scheme, routed.hostname, routed.port) != (base.scheme, base.hostname, base.port):
                    raise GPUServiceError(
                        "authentication",
                        "GPU service operation route must use the registered endpoint origin",
                        retryable=False,
                    )
                return configured
            return self.base_url + "/" + configured.lstrip("/")
        if self.base_url.endswith("/v1"):
            return self.base_url + "/" + default.removeprefix("v1/")
        return self.base_url + "/" + default.lstrip("/")

    def _post(self, operation, default_route, body):
        import socket
        import urllib.error
        import urllib.request

        token = self._token()
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            self._route(operation, default_route),
            data=json.dumps(body, ensure_ascii=True).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        timeout_seconds = self.index_timeout_seconds if operation in {"index_status", "index_upsert"} else self.timeout_seconds
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except OSError:
                detail = str(exc)
            category = classify_failure(detail or str(exc), exc.code)
            raise GPUServiceError(category, f"GPU service request failed ({exc.code}): {detail[:500]}", status_code=exc.code) from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            category = classify_failure(str(exc))
            raise GPUServiceError(category, f"GPU service request failed: {exc}") from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GPUServiceError("service_failure", "GPU service returned non-JSON output") from exc
        if not isinstance(payload, dict):
            raise GPUServiceError("service_failure", "GPU service returned an invalid JSON envelope")
        if payload.get("error"):
            detail = payload["error"]
            category = classify_failure(json.dumps(detail, ensure_ascii=True))
            raise GPUServiceError(category, f"GPU service error: {detail}")
        return payload

    def embed(self, texts):
        response = self._post(
            "embed",
            "v1/embeddings",
            {"model": self.model, "input": list(texts), "encoding_format": "float"},
        )
        data = response.get("data")
        if not isinstance(data, list):
            raise GPUServiceError("service_failure", "embedding response is missing data")
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0)
        vectors = []
        for item in ordered:
            vector = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vector, list) or not vector:
                raise GPUServiceError("service_failure", "embedding response contains an invalid vector")
            try:
                vectors.append([float(value) for value in vector])
            except (TypeError, ValueError) as exc:
                raise GPUServiceError("service_failure", "embedding response contains non-numeric values") from exc
        if len(vectors) != len(texts):
            raise GPUServiceError("service_failure", "embedding response count does not match request")
        return vectors

    def rerank(self, query, documents):
        response = self._post(
            "rerank",
            "v1/rerank",
            {"model": self.model, "query": query, "documents": list(documents), "top_n": len(documents)},
        )
        results = response.get("results") or response.get("data")
        if not isinstance(results, list):
            raise GPUServiceError("service_failure", "rerank response is missing results")
        scores = [None] * len(documents)
        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
                score = float(item.get("relevance_score", item.get("score")))
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(scores):
                scores[index] = score
        if any(score is None for score in scores):
            raise GPUServiceError("service_failure", "rerank response does not score every candidate")
        return [float(score) for score in scores]

    @staticmethod
    def _index_document(chunk):
        return {
            "id": chunk["chunk_id"],
            "text": chunk["content"],
            "metadata": {
                "path": chunk["path"],
                "language": chunk.get("language", ""),
                "symbol": chunk.get("symbol", ""),
                "line_start": chunk["line_start"],
                "line_end": chunk["line_end"],
                "content_hash": chunk["content_hash"],
            },
        }

    @staticmethod
    def _select_changed_chunks(chunks, changed_ids):
        changed_ids = {str(value) for value in changed_ids if str(value)}
        if not changed_ids:
            return []
        file_key_by_chunk_id = getattr(chunks, "_file_key_by_chunk_id", None)
        chunks_by_file = getattr(chunks, "_chunks_by_file", None)
        if isinstance(file_key_by_chunk_id, dict) and isinstance(chunks_by_file, dict):
            affected_files = {
                str(file_key_by_chunk_id.get(chunk_id) or "")
                for chunk_id in changed_ids
                if str(file_key_by_chunk_id.get(chunk_id) or "")
            }
            selected = []
            for file_key in affected_files:
                for chunk in chunks_by_file.get(file_key, ()):
                    if str(chunk.get("chunk_id") or "") in changed_ids:
                        selected.append(chunk)
            if selected:
                return selected
        return [chunk for chunk in chunks if str(chunk.get("chunk_id") or "") in changed_ids]

    def ensure_semantic_index(self, chunks, fingerprint, *, batch_size=32, sync_plan=None, content_loader=None):
        """Ensure the retrieval service owns a finalized fingerprinted index.

        Warm queries send only the fingerprint and document count. Repository
        content is uploaded in bounded batches only when the service reports a
        cache miss; the service is responsible for embeddings and persistent
        FAISS construction.
        """

        status_body = {
            "model": self.model,
            "model_profile": self.model_profile,
            "index_fingerprint": fingerprint,
            "document_count": len(chunks),
        }
        status = self._post("index_status", "v1/indexes/status", status_body)
        if status.get("ready") is True:
            return {
                "cache_hit": True,
                "document_count": int(status.get("document_count") or len(chunks)),
                "embedded_documents": 0,
                "reused_documents": 0,
            }
        if not chunks:
            raise GPUServiceError("service_failure", "semantic index has no repository chunks")
        if batch_size < 1:
            raise GPUServiceError("service_failure", "semantic index batch size must be positive")

        def upload(selected_chunks, *, base_fingerprint="", removed_ids=None):
            removed_ids = list(removed_ids or [])
            if selected_chunks and any("content" not in chunk for chunk in selected_chunks):
                if content_loader is None:
                    raise GPUServiceError("service_failure", "semantic index requires chunk content")
                content_loader(selected_chunks)
                if any("content" not in chunk for chunk in selected_chunks):
                    raise GPUServiceError("service_failure", "semantic index content loader did not hydrate chunk content")
            batches = []
            batch = []
            batch_bytes = 0
            for chunk in selected_chunks:
                document = self._index_document(chunk)
                document_bytes = len(json.dumps(document, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
                if document_bytes > MAX_INDEX_BATCH_BYTES:
                    raise GPUServiceError("service_failure", "one semantic index document exceeds the upload limit")
                if batch and (len(batch) >= batch_size or batch_bytes + document_bytes > MAX_INDEX_BATCH_BYTES):
                    batches.append(batch)
                    batch = []
                    batch_bytes = 0
                batch.append(document)
                batch_bytes += document_bytes
            if batch or removed_ids or base_fingerprint:
                batches.append(batch)

            embedded_documents = 0
            reused_documents = 0
            first_response = {}
            for batch_index, documents in enumerate(batches):
                final = batch_index == len(batches) - 1
                body = {
                    "model": self.model,
                    "model_profile": self.model_profile,
                    "index_fingerprint": fingerprint,
                    "document_count": len(chunks),
                    "replace": batch_index == 0,
                    "finalize": final,
                    "documents": documents,
                }
                if batch_index == 0 and base_fingerprint:
                    body["base_index_fingerprint"] = base_fingerprint
                if batch_index == 0 and removed_ids:
                    body["removed_document_ids"] = removed_ids
                response = self._post("index_upsert", "v1/indexes/upsert", body)
                if batch_index == 0:
                    first_response = response
                if response.get("accepted") is False:
                    raise GPUServiceError("service_failure", "semantic index rejected an update batch")
                embedded_documents += int(response.get("embedded_documents") or 0)
                reused_documents += int(response.get("reused_documents") or 0)
            return embedded_documents, reused_documents, first_response

        embedded_documents = 0
        reused_documents = 0
        base_fingerprint = str((sync_plan or {}).get("base_fingerprint") or "")
        changed_ids = {str(value) for value in ((sync_plan or {}).get("changed_ids") or []) if str(value)}
        removed_ids = [str(value) for value in ((sync_plan or {}).get("removed_ids") or []) if str(value)]
        if base_fingerprint:
            changed_chunks = self._select_changed_chunks(chunks, changed_ids)
            embedded_documents, reused_documents, first_response = upload(
                changed_chunks,
                base_fingerprint=base_fingerprint,
                removed_ids=removed_ids,
            )
            if first_response.get("base_reused") is not True:
                embedded_documents, reused_documents, _ = upload(chunks)
        else:
            embedded_documents, reused_documents, _ = upload(chunks)

        status = self._post("index_status", "v1/indexes/status", status_body)
        if status.get("ready") is not True:
            raise GPUServiceError("service_failure", "semantic index did not finalize the requested fingerprint")
        return {
            "cache_hit": False,
            "document_count": int(status.get("document_count") or len(chunks)),
            "embedded_documents": embedded_documents,
            "reused_documents": reused_documents,
        }

    def semantic_search(self, query, chunks_or_ids, fingerprint, limit):
        response = self._post(
            "search",
            "v1/search",
            {
                "model": self.model,
                "model_profile": self.model_profile,
                "query": query,
                "index_fingerprint": fingerprint,
                "limit": int(limit),
            },
        )
        results = response.get("results") or response.get("data")
        if not isinstance(results, list):
            raise GPUServiceError("service_failure", "semantic search response is missing results")
        requested_ids = set()
        for item in chunks_or_ids:
            if isinstance(item, dict):
                requested_ids.add(str(item.get("chunk_id") or ""))
            else:
                requested_ids.add(str(item or ""))
        seen_ids = set()
        ranked = []
        for index, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("id") or item.get("chunk_id") or "")
            if not chunk_id:
                continue
            if chunk_id not in requested_ids:
                raise GPUServiceError("service_failure", "semantic search returned an unknown chunk id")
            if chunk_id in seen_ids:
                raise GPUServiceError("service_failure", "semantic search returned a duplicate chunk id")
            seen_ids.add(chunk_id)
            try:
                score = float(item.get("score") or item.get("relevance_score") or 0.0)
            except (TypeError, ValueError) as exc:
                raise GPUServiceError("service_failure", "semantic search returned an invalid score") from exc
            ranked.append({
                "chunk_id": chunk_id,
                "score": score,
                "rank": index,
                "source": "semantic",
            })
        return ranked[:limit]

    def chat(self, messages, response_schema):
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "repo_inspection_answer",
                "strict": True,
                "schema": response_schema,
            },
        }
        response = self._post(
            "chat",
            "v1/chat/completions",
            {
                "model": self.model,
                "temperature": 0,
                "messages": list(messages),
                "response_format": response_format,
            },
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise GPUServiceError("invalid_synthesis", "chat response is missing choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            content = "".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            raise GPUServiceError("invalid_synthesis", "chat response is missing structured content")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise GPUServiceError("invalid_synthesis", "chat response is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise GPUServiceError("invalid_synthesis", "chat response is not a JSON object")
        return parsed
