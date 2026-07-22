"""GPU-first ``inspect_repo`` pipeline.

CPU work is limited to discovery/chunking, lexical retrieval, rank fusion, and
cache bookkeeping.  An answer is released only after successful GPU semantic
retrieval, GPU reranking, and validated GPU synthesis.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import shutil
import contextlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from inspection_contract import validate_request
from inspection_hotpath import (
    cache_runtime_diagnostics,
    exclusion_paths_for_execution,
    estimate_tokens,
    fingerprint_hint_state,
    inspection_index_fingerprint,
    load_cached_chunk_snapshot_metadata,
    repository_fingerprint,
    sha256_text,
    transient_excluded_paths_for_execution,
)

_INDEX_SYMBOLS = None


def _index_symbols():
    global _INDEX_SYMBOLS
    if _INDEX_SYMBOLS is None:
        from inspection_index import (
            build_syntax_chunks,
            ensure_lexical_index,
            lexical_working_index_is_current,
            lexical_cache_key,
            lexical_helper,
            lexical_path_catalog,
            lexical_search,
            load_cached_chunk_snapshot,
            load_chunks_from_lexical_manifest,
            load_chunks_from_lexical_index,
            query_features,
            semantic_chunk_signature,
        )

        _INDEX_SYMBOLS = {
            "build_syntax_chunks": build_syntax_chunks,
            "ensure_lexical_index": ensure_lexical_index,
            "lexical_working_index_is_current": lexical_working_index_is_current,
            "lexical_cache_key": lexical_cache_key,
            "lexical_helper": lexical_helper,
            "lexical_path_catalog": lexical_path_catalog,
            "lexical_search": lexical_search,
            "load_cached_chunk_snapshot": load_cached_chunk_snapshot,
            "load_chunks_from_lexical_manifest": load_chunks_from_lexical_manifest,
            "load_chunks_from_lexical_index": load_chunks_from_lexical_index,
            "query_features": query_features,
            "semantic_chunk_signature": semantic_chunk_signature,
        }
    return _INDEX_SYMBOLS


def _index(name):
    return _index_symbols()[name]


def build_syntax_chunks(*args, **kwargs):
    return _index("build_syntax_chunks")(*args, **kwargs)


def ensure_lexical_index(*args, **kwargs):
    return _index("ensure_lexical_index")(*args, **kwargs)


def lexical_cache_key(*args, **kwargs):
    return _index("lexical_cache_key")(*args, **kwargs)


def lexical_path_catalog(*args, **kwargs):
    return _index("lexical_path_catalog")(*args, **kwargs)


def lexical_helper(*args, **kwargs):
    return _index("lexical_helper")(*args, **kwargs)


def lexical_search(*args, **kwargs):
    return _index("lexical_search")(*args, **kwargs)


def lexical_working_index_is_current(*args, **kwargs):
    return _index("lexical_working_index_is_current")(*args, **kwargs)


def load_cached_chunk_snapshot(*args, **kwargs):
    return _index("load_cached_chunk_snapshot")(*args, **kwargs)


def load_chunks_from_lexical_index(*args, **kwargs):
    return _index("load_chunks_from_lexical_index")(*args, **kwargs)


def load_chunks_from_lexical_manifest(*args, **kwargs):
    return _index("load_chunks_from_lexical_manifest")(*args, **kwargs)


def query_features(*args, **kwargs):
    return _index("query_features")(*args, **kwargs)


def semantic_chunk_signature(*args, **kwargs):
    return _index("semantic_chunk_signature")(*args, **kwargs)


SEMANTIC_SYNC_MANIFEST_SCHEMA = "repo-inspection-semantic-sync-v1"
QUERY_STAGE_CACHE_SCHEMA = "repo-inspection-query-stage-cache-v1"
QUERY_STAGE_CACHE_ALIAS_SCHEMA = "repo-inspection-query-stage-cache-alias-v1"
NAMED_PATH_CACHE_LIMIT = 64
QUERY_STAGE_CACHE_LIMIT = 128
QUERY_STAGE_CACHE_PRUNE_INTERVAL = 16
_NAMED_PATH_CACHE = {}


def reset_process_caches():
    """Clear process-local pipeline caches between isolated test runs."""
    for value in _PROCESS_CACHE_CONTAINERS:
        value.clear()
    _QUERY_STAGE_MEMORY_CACHE.clear()
    _QUERY_STAGE_ALIAS_MEMORY_CACHE.clear()
_QUERY_STAGE_PRUNE_COUNTER = 0
_QUERY_STAGE_MEMORY_CACHE = {}
_QUERY_STAGE_ALIAS_MEMORY_CACHE = {}
_QUERY_STAGE_MEMORY_CACHE_LIMIT = 128
_PROCESS_CACHE_CONTAINERS = (_NAMED_PATH_CACHE, _QUERY_STAGE_MEMORY_CACHE, _QUERY_STAGE_ALIAS_MEMORY_CACHE)
_GPU_SYMBOLS = None
_SERVICE_CONTROL_SYMBOLS = None


@dataclass
class _InspectionContext:
    query: str
    mode: str
    constraints: dict
    task_params: dict
    execution_plan: dict
    budgets: dict
    client_factory: object


def _inspection_context(query, mode, constraints, task_params, execution_plan, client_factory):
    query, mode = validate_request(query, mode)
    constraints = constraints or {}
    task_params = task_params or {}
    execution_plan = execution_plan or {}
    return _InspectionContext(
        query=query, mode=mode, constraints=constraints, task_params=task_params,
        execution_plan=execution_plan, budgets=normalize_token_budgets(constraints),
        client_factory=client_factory or _gpu_client_factory(),
    )
SYNTHESIS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "findings"],
    "properties": {
        "answer": {"type": "string", "minLength": 1},
        "findings": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["summary", "evidence_refs"],
                "properties": {
                    "summary": {"type": "string", "minLength": 1},
                    "evidence_refs": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                },
            },
        },
    },
}


def _gpu_symbols():
    global _GPU_SYMBOLS
    if _GPU_SYMBOLS is None:
        from gpu_client import (
            GPUServiceClient,
            GPUServiceError,
            MULTIGPU_FAILURES,
            TRANSIENT_SINGLE_GPU_FAILURES,
            endpoint_diagnostics,
            select_endpoint,
            services_from_execution_plan,
        )

        _GPU_SYMBOLS = {
            "GPUServiceClient": GPUServiceClient,
            "GPUServiceError": GPUServiceError,
            "MULTIGPU_FAILURES": MULTIGPU_FAILURES,
            "TRANSIENT_SINGLE_GPU_FAILURES": TRANSIENT_SINGLE_GPU_FAILURES,
            "endpoint_diagnostics": endpoint_diagnostics,
            "select_endpoint": select_endpoint,
            "services_from_execution_plan": services_from_execution_plan,
        }
    return _GPU_SYMBOLS


def _service_control_symbols():
    global _SERVICE_CONTROL_SYMBOLS
    if _SERVICE_CONTROL_SYMBOLS is None:
        from service_control import failure_reporter_from_execution_plan, requester_from_execution_plan

        _SERVICE_CONTROL_SYMBOLS = {
            "failure_reporter_from_execution_plan": failure_reporter_from_execution_plan,
            "requester_from_execution_plan": requester_from_execution_plan,
        }
    return _SERVICE_CONTROL_SYMBOLS


def _gpu(name):
    return _gpu_symbols()[name]


def _service_control(name):
    return _service_control_symbols()[name]


def _gpu_client_factory():
    def factory(record):
        return _gpu("GPUServiceClient")(record)

    return factory


def _service_control_configured(execution_plan):
    execution_plan = execution_plan or {}
    request_path = execution_plan.get("gpu_service_request_path") or os.environ.get(
        "BROKER_GPU_SERVICE_CONTROL_REQUEST_DIR"
    )
    control_token = execution_plan.get("gpu_service_control_token") or os.environ.get(
        "BROKER_GPU_SERVICE_CONTROL_TOKEN"
    )
    return bool(request_path and control_token)


def _gpu_registry_configured(execution_plan):
    execution_plan = execution_plan or {}
    registry_path = execution_plan.get("gpu_service_registry_path") or os.environ.get(
        "BROKER_GPU_SERVICE_REGISTRY_PATH"
    )
    return bool(str(registry_path or "").strip())

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


def reciprocal_rank_fusion(rankings, *, k=60, limit=64):
    scores = defaultdict(float)
    sources = defaultdict(list)
    for ranking in rankings:
        seen = set()
        for default_rank, item in enumerate(ranking, start=1):
            chunk_id = str(item.get("chunk_id") or "")
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank = max(1, int(item.get("rank") or default_rank))
            scores[chunk_id] += 1.0 / (k + rank)
            source = str(item.get("source") or "")
            if source and source not in sources[chunk_id]:
                sources[chunk_id].append(source)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:limit]
    return [
        {
            "chunk_id": chunk_id,
            "rrf_score": round(scores[chunk_id], 10),
            "sources": sources[chunk_id],
            "rank": rank,
        }
        for rank, chunk_id in enumerate(ordered, start=1)
    ]


def _candidate_budget(fused, chunk_by_id, token_budget):
    selected = []
    used = 0
    for item in fused[:64]:
        chunk = chunk_by_id.get(item["chunk_id"])
        if not chunk:
            continue
        projected = used + int(chunk.get("token_estimate") or 0)
        if selected and projected > token_budget:
            continue
        selected.append(dict(item))
        used = projected
    return selected, used


def _elapsed_ms(started_at):
    return round((time.perf_counter() - started_at) * 1000.0, 3)


def _chunk_count(chunks):
    attached = getattr(chunks, "_chunk_count", None)
    if attached is not None:
        return int(attached)
    attached_ids = getattr(chunks, "_chunk_ids", None)
    if isinstance(attached_ids, (list, tuple)):
        return len(attached_ids)
    return len(chunks)


def explicitly_named_paths(query, chunks, *, features=None, cache_token=None, path_catalog=None):
    if cache_token is not None:
        cache_key = (cache_token, str(query))
        cached = _NAMED_PATH_CACHE.get(cache_key)
        if cached is not None:
            _NAMED_PATH_CACHE.pop(cache_key, None)
            _NAMED_PATH_CACHE[cache_key] = cached
            return set(cached)
    lower_query = query.lower()
    features = features or query_features(query)
    path_tokens = {value.lower().split(":", 1)[0] for value in features["paths"]}
    if not path_tokens:
        if cache_token is not None:
            _NAMED_PATH_CACHE[cache_key] = set()
            while len(_NAMED_PATH_CACHE) > NAMED_PATH_CACHE_LIMIT:
                oldest_key = next(iter(_NAMED_PATH_CACHE))
                _NAMED_PATH_CACHE.pop(oldest_key, None)
        return set()
    suffix_needles = tuple("/" + token for token in path_tokens if token)
    named = set()
    catalog_paths = tuple(path_catalog.get("unique_paths") or ()) if isinstance(path_catalog, dict) else ()
    lower_by_path = (path_catalog.get("path_lower_by_path") or {}) if isinstance(path_catalog, dict) else {}
    basename_by_path = (path_catalog.get("path_basename_lower_by_path") or {}) if isinstance(path_catalog, dict) else {}
    if not catalog_paths:
        seen_paths = set()
        unique_paths = []
        for chunk in chunks:
            path = str(chunk["path"])
            if path in seen_paths:
                continue
            seen_paths.add(path)
            unique_paths.append(path)
        catalog_paths = tuple(unique_paths)
    for path in catalog_paths:
        lower_path = lower_by_path.get(path, path.lower())
        basename = basename_by_path.get(path, Path(path).name.lower())
        suffix_match = any(lower_path.endswith(needle) for needle in suffix_needles)
        if lower_path in lower_query or lower_path in path_tokens or basename in path_tokens or suffix_match:
            named.add(path)
    if cache_token is not None:
        _NAMED_PATH_CACHE[cache_key] = set(named)
        while len(_NAMED_PATH_CACHE) > NAMED_PATH_CACHE_LIMIT:
            oldest_key = next(iter(_NAMED_PATH_CACHE))
            _NAMED_PATH_CACHE.pop(oldest_key, None)
    return named


def select_diverse_chunks(
    ranked,
    query,
    evidence_token_budget,
    *,
    limit=12,
    chunk_by_id,
    features=None,
    named_paths=None,
    cache_token=None,
    path_catalog=None,
):
    if named_paths is None:
        named_paths = explicitly_named_paths(
            query,
            chunk_by_id.values(),
            features=features,
            cache_token=cache_token,
            path_catalog=path_catalog,
        )
    path_counts = defaultdict(int)
    content_hashes = set()
    selected_ids = set()
    selected = []
    used = 0
    per_chunk_budget = max(64, evidence_token_budget // max(1, limit))
    # First cover distinct files (while allowing an explicitly named file to
    # contribute multiple chunks), then fill remaining slots up to the hard
    # two-per-file limit.  This avoids spending the evidence pack on adjacent
    # chunks from one large test or service module.
    for diversity_pass in (True, False):
        for ranked_item in ranked:
            chunk = chunk_by_id.get(ranked_item["chunk_id"])
            if (
                not chunk
                or chunk["chunk_id"] in selected_ids
                or (chunk.get("source_namespace", ""), chunk["content_hash"]) in content_hashes
            ):
                continue
            path = chunk["path"]
            if diversity_pass and path_counts[path] > 0 and path not in named_paths:
                continue
            if path not in named_paths and path_counts[path] >= 2:
                continue
            projected = used + min(int(chunk.get("token_estimate") or 0), per_chunk_budget)
            if selected and projected > evidence_token_budget:
                continue
            selected.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "rank": int(ranked_item.get("rank") or 0),
                }
            )
            selected_ids.add(chunk["chunk_id"])
            content_hashes.add((chunk.get("source_namespace", ""), chunk["content_hash"]))
            path_counts[path] += 1
            used = projected
            if len(selected) >= limit:
                return selected, used
    return selected, used


def build_evidence(selected, chunk_by_id):
    evidence = []
    for index, item in enumerate(selected, start=1):
        chunk = chunk_by_id[item["chunk_id"]]
        evidence.append(
            {
                "id": f"ev_{index:03d}",
                "path": chunk["path"],
                "language": chunk.get("language", ""),
                "symbol": chunk.get("symbol", ""),
                "source_refs": [
                    {
                        "path": chunk["path"],
                        "input_id": chunk.get("input_id", ""),
                        "source_namespace": chunk.get("source_namespace", ""),
                        "line_start": int(chunk["line_start"]),
                        "line_end": int(chunk["line_end"]),
                        "content_hash": chunk["content_hash"],
                    }
                ],
                "excerpt": chunk["content"],
                "rank": int(item.get("rank") or index),
            }
        )
    return evidence


def released_pack_tokens(payload):
    """Count the released content pack while excluding compact diagnostics."""

    pack = {
        key: payload[key]
        for key in ("mode", "query", "answer", "findings", "evidence", "quality", "warnings", "provenance")
        if key in payload
    }
    return estimate_tokens(json.dumps(pack, ensure_ascii=True, separators=(",", ":")))


def trim_evidence_for_final_pack(evidence, base_payload, final_pack_token_budget, synthesis_reserve=0):
    """Trim whole evidence items, then a final excerpt, to honor the pack cap."""

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


def _attempt(endpoint, operation, outcome, *, category="", reason="", number=1, detail=""):
    diagnostics = _gpu("endpoint_diagnostics")(endpoint)
    return {
        "operation": operation,
        "tier": diagnostics.get("tier") or str((endpoint or {}).get("tier") or ""),
        "slurm_job_id": diagnostics.get("job_id", ""),
        "gpu_count": diagnostics.get("gpu_count", 0),
        "gpu_type": diagnostics.get("gpu_type", ""),
        "model_profile": diagnostics.get("model_profile", ""),
        "attempt": int(number),
        "status": outcome,
        "failure_category": category,
        "escalation_reason": reason,
    }


def _missing_endpoint_record(tier, gpu_count, error=None):
    diagnostics = dict(getattr(error, "service_diagnostics", {}) or {})
    if diagnostics:
        return diagnostics
    return {"tier": tier, "gpu": {"count": gpu_count}}


def _report_warm_service_failure(reporter, endpoint, error, operation):
    if reporter is None or endpoint is None:
        return
    if error.category not in {"availability", "timeout", "service_failure", "authentication", "oom"}:
        return
    try:
        reporter(endpoint, error.category, f"{operation} request failed validation or execution")
    except Exception:
        # Failure reporting is best-effort and must never hide the original
        # inference failure or change the inspection result.
        return


def _unavailable_category(services, tier):
    allowed = _gpu("TRANSIENT_SINGLE_GPU_FAILURES") | _gpu("MULTIGPU_FAILURES")
    for record in services:
        if str(record.get("tier") or "") != tier:
            continue
        category = str(record.get("failure_category") or "")
        if category in allowed:
            return category
        state = str(record.get("state") or "").lower()
        if state == "starting":
            return "queue_delay"
    return "availability"


def _resolve_or_request_endpoint(
    services,
    tier,
    capability,
    gpu_count,
    health_interval_seconds,
    service_requester,
    failure_category,
    reason,
):
    endpoint = _gpu("select_endpoint")(
        services,
        tier,
        capability,
        expected_gpu_count=gpu_count,
        health_interval_seconds=health_interval_seconds,
    )
    if endpoint is not None or service_requester is None:
        return endpoint, None
    try:
        record = service_requester(tier, failure_category, reason)
        services.append(record)
    except _gpu("GPUServiceError") as exc:
        return None, exc
    endpoint = _gpu("select_endpoint")(
        services,
        tier,
        capability,
        expected_gpu_count=gpu_count,
        health_interval_seconds=health_interval_seconds,
    )
    if endpoint is None:
        return None, _gpu("GPUServiceError")("service_failure", "demanded GPU service lacks the required capability")
    return endpoint, None


def gpu_semantic_retrieval(
    services,
    chunks,
    query,
    fingerprint,
    build_config_digest,
    cache_dir,
    attempts,
    client_factory,
    health_interval_seconds=None,
    service_requester=None,
    failure_reporter=None,
    endpoint_override=None,
):
    GPUServiceError = _gpu("GPUServiceError")
    initial_category = _unavailable_category(services, "p40-retrieval")
    demand_error = None
    endpoint = endpoint_override
    if endpoint is None:
        endpoint, demand_error = _resolve_or_request_endpoint(
            services,
            "p40-retrieval",
            "search",
            1,
            health_interval_seconds,
            service_requester,
            initial_category,
            "GPU FAISS retrieval is required for inspect_repo",
        )
    if endpoint is None:
        category = demand_error.category if demand_error is not None else initial_category
        missing = _missing_endpoint_record("p40-retrieval", 1, demand_error)
        attempts.append(
            _attempt(
                missing,
                "semantic_retrieval",
                "failed",
                category=category,
                reason="primary_retrieval",
                detail=demand_error or "",
            )
        )
        return [], "lexical_degraded", {"cache_hit": False, "document_count": len(chunks), "embedded_documents": 0, "reused_documents": 0}, None
    try:
        client = client_factory(endpoint)
        def content_loader(selected_chunks):
            _hydrate_chunk_contents(
                Path(cache_dir) / "lexical-working.sqlite3",
                {str(chunk.get("chunk_id") or ""): chunk for chunk in chunks},
                [str(chunk.get("chunk_id") or "") for chunk in selected_chunks],
            )
        semantic_manifest_scope = "sha256:" + sha256_text(
            f"{build_config_digest}\0{endpoint.get('model_profile', '')}\0{endpoint.get('model', '')}"
        )
        service_fingerprint = "sha256:" + sha256_text(
            f"{fingerprint}\0{endpoint.get('model_profile', '')}\0{endpoint.get('model', '')}"
        )
        semantic_documents = _semantic_document_signatures(chunks)
        previous_semantic_manifest = _load_semantic_sync_manifest(cache_dir, semantic_manifest_scope)
        semantic_sync_plan = _semantic_sync_plan(previous_semantic_manifest, semantic_documents, service_fingerprint)
        try:
            index_stats = client.ensure_semantic_index(
                chunks,
                service_fingerprint,
                sync_plan=semantic_sync_plan,
                content_loader=content_loader,
            )
        except TypeError as exc:
            if "content_loader" not in str(exc):
                raise
            index_stats = client.ensure_semantic_index(
                chunks,
                service_fingerprint,
                sync_plan=semantic_sync_plan,
            )
        _write_semantic_sync_manifest(
            cache_dir,
            semantic_documents,
            service_fingerprint,
            semantic_manifest_scope,
            previous=previous_semantic_manifest,
        )
        if isinstance(index_stats, dict):
            cache_hit = bool(index_stats.get("cache_hit"))
        else:
            cache_hit = bool(index_stats)
            index_stats = {
                "cache_hit": cache_hit,
                "document_count": len(chunks),
                "embedded_documents": 0 if cache_hit else len(chunks),
                "reused_documents": 0,
            }
        if not isinstance(client, _gpu("GPUServiceClient")) and any("content" not in chunk for chunk in chunks):
            content_loader(chunks)
        ranked = client.semantic_search(
            query,
            _semantic_chunk_ids(chunks) if isinstance(client, _gpu("GPUServiceClient")) else chunks,
            service_fingerprint,
            128,
        )
        if not ranked:
            raise GPUServiceError("service_failure", "GPU semantic retrieval returned no candidates")
        attempts.append(_attempt(endpoint, "semantic_retrieval", "succeeded", reason="primary_retrieval"))
        return ranked, "gpu", index_stats, endpoint
    except GPUServiceError as exc:
        _report_warm_service_failure(failure_reporter, endpoint, exc, "semantic_retrieval")
        attempts.append(
            _attempt(endpoint, "semantic_retrieval", "failed", category=exc.category, reason="primary_retrieval", detail=exc)
        )
        return [], "lexical_degraded", {"cache_hit": False, "document_count": len(chunks), "embedded_documents": 0, "reused_documents": 0}, endpoint


def _semantic_sync_manifest_path(cache_dir, manifest_scope):
    safe_scope = str(manifest_scope or "default").replace(":", "_")
    return Path(cache_dir) / f"semantic-working-manifest-{safe_scope}.json"


def _shared_semantic_sync_manifest_path(manifest_scope):
    cache_dir = _shared_query_stage_cache_dir()
    if cache_dir is None:
        return None
    safe_scope = str(manifest_scope or "default").replace(":", "_")
    return cache_dir.parent / f"semantic-working-manifest-{safe_scope}.json"


def _semantic_chunk_signature(chunk):
    return semantic_chunk_signature(chunk)


def _semantic_document_signatures(chunks):
    attached = getattr(chunks, "_semantic_document_signatures", None)
    if isinstance(attached, dict):
        return {str(chunk_id): str(signature) for chunk_id, signature in attached.items()}
    return {str(chunk.get("chunk_id") or ""): _semantic_chunk_signature(chunk) for chunk in chunks}


def _semantic_chunk_ids(chunks):
    attached = getattr(chunks, "_chunk_ids", None)
    if isinstance(attached, (list, tuple)):
        return tuple(str(value) for value in attached if str(value))
    attached_signatures = getattr(chunks, "_semantic_document_signatures", None)
    if isinstance(attached_signatures, dict):
        return tuple(str(chunk_id) for chunk_id in attached_signatures if str(chunk_id))
    return tuple(str(chunk.get("chunk_id") or "") for chunk in chunks if str(chunk.get("chunk_id") or ""))


def _load_semantic_sync_manifest(cache_dir, manifest_scope):
    def load_path(path):
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            path.unlink(missing_ok=True)
            return None
        if not isinstance(payload, dict) or payload.get("schema") != SEMANTIC_SYNC_MANIFEST_SCHEMA:
            return None
        documents = payload.get("documents")
        if not isinstance(documents, dict):
            return None
        return {
            "fingerprint": str(payload.get("fingerprint") or ""),
            "documents": {str(chunk_id): str(signature) for chunk_id, signature in documents.items()},
        }

    payload = load_path(_semantic_sync_manifest_path(cache_dir, manifest_scope))
    if payload is not None:
        return payload
    return load_path(_shared_semantic_sync_manifest_path(manifest_scope))


def _write_semantic_sync_manifest(cache_dir, document_signatures, fingerprint, manifest_scope, *, previous=None):
    previous = previous or {}
    if (
        str(previous.get("fingerprint") or "") == str(fingerprint)
        and dict(previous.get("documents") or {}) == dict(document_signatures)
    ):
        return False
    path = _semantic_sync_manifest_path(cache_dir, manifest_scope)
    payload = {
        "schema": SEMANTIC_SYNC_MANIFEST_SCHEMA,
        "fingerprint": str(fingerprint),
        "documents": dict(document_signatures),
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    shared_path = _shared_semantic_sync_manifest_path(manifest_scope)
    if shared_path is not None and shared_path != path:
        _atomic_private_bytes(shared_path, payload_bytes)
    else:
        _atomic_private_bytes(path, payload_bytes)
    return True


def _semantic_sync_plan(previous, document_signatures, fingerprint):
    if not previous:
        return None
    previous_fingerprint = str(previous.get("fingerprint") or "")
    if not previous_fingerprint or previous_fingerprint == fingerprint:
        return None
    previous_documents = previous.get("documents") or {}
    changed_ids = sorted(
        chunk_id
        for chunk_id, signature in document_signatures.items()
        if previous_documents.get(chunk_id) != signature
    )
    removed_ids = sorted(chunk_id for chunk_id in previous_documents if chunk_id not in document_signatures)
    return {
        "base_fingerprint": previous_fingerprint,
        "changed_ids": changed_ids,
        "removed_ids": removed_ids,
    }


def _private_cache_dir(path: Path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass
    return path


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


def _query_stage_cache_key(query, fingerprint, retrieval_signature, budgets):
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


def _query_stage_cache_path(cache_dir, cache_key):
    return _sharded_json_cache_path(_query_stage_cache_dir(cache_dir), cache_key)


def _shared_query_stage_cache_path(cache_key):
    return _sharded_json_cache_path(_shared_query_stage_cache_dir(), cache_key)


def _legacy_query_stage_cache_path(cache_dir, cache_key):
    return _legacy_json_cache_path(_query_stage_cache_dir(cache_dir), cache_key)


def _legacy_shared_query_stage_cache_path(cache_key):
    return _legacy_json_cache_path(_shared_query_stage_cache_dir(), cache_key)


def _query_stage_cache_alias_key(query, repository_state_fingerprint, build_config_digest, retrieval_signature, budgets):
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
        normalized.append(
            {
                "summary": summary,
                "evidence_refs": evidence_refs,
            }
        )
    return normalized


def _clone_query_stage_cache_payload(payload):
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
        "released_payload": dict(payload.get("released_payload") or {}) if isinstance(payload.get("released_payload"), dict) else {},
        "released_artifact_payloads": (
            {str(key): dict(value) for key, value in dict(payload.get("released_artifact_payloads") or {}).items() if isinstance(value, dict)}
            if isinstance(payload.get("released_artifact_payloads"), dict)
            else {}
        ),
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


def _cache_query_stage_alias_memory_payload(path: Path, payload):
    cache_key = _query_stage_memory_cache_key(path)
    if cache_key is None:
        return
    _QUERY_STAGE_ALIAS_MEMORY_CACHE.pop(cache_key, None)
    _QUERY_STAGE_ALIAS_MEMORY_CACHE[cache_key] = dict(payload)
    while len(_QUERY_STAGE_ALIAS_MEMORY_CACHE) > _QUERY_STAGE_MEMORY_CACHE_LIMIT:
        _QUERY_STAGE_ALIAS_MEMORY_CACHE.pop(next(iter(_QUERY_STAGE_ALIAS_MEMORY_CACHE)))


def _query_stage_memory_payload_matches(path: Path | None, payload):
    if path is None:
        return False
    cache_key = _query_stage_memory_cache_key(path)
    if cache_key is None:
        return False
    cached = _QUERY_STAGE_MEMORY_CACHE.get(cache_key)
    if cached is None:
        return False
    _QUERY_STAGE_MEMORY_CACHE.pop(cache_key, None)
    _QUERY_STAGE_MEMORY_CACHE[cache_key] = cached
    return _clone_query_stage_cache_payload(cached) == _clone_query_stage_cache_payload(payload)


def _query_stage_alias_memory_payload_matches(path: Path | None, payload):
    if path is None:
        return False
    cache_key = _query_stage_memory_cache_key(path)
    if cache_key is None:
        return False
    cached = _QUERY_STAGE_ALIAS_MEMORY_CACHE.get(cache_key)
    if cached is None:
        return False
    _QUERY_STAGE_ALIAS_MEMORY_CACHE.pop(cache_key, None)
    _QUERY_STAGE_ALIAS_MEMORY_CACHE[cache_key] = cached
    return dict(cached) == dict(payload)


def _touch_query_stage_cache_path(path: Path):
    try:
        os.utime(path, None)
    except OSError:
        return


def _atomic_private_bytes(path, payload: bytes):
    path = Path(path)
    _private_cache_dir(path.parent)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.unlink(missing_ok=True)
    descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    path.chmod(0o600)
    return path


def _path_bytes_equal(path: Path | None, payload: bytes):
    if path is None or not path.exists():
        return False
    try:
        stat = path.stat()
        if int(stat.st_size) != len(payload):
            return False
        return path.read_bytes() == payload
    except OSError:
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


def _load_query_stage_cache(cache_dir, cache_key):
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
            "released_payload": dict(payload.get("released_payload") or {}) if isinstance(payload.get("released_payload"), dict) else {},
            "released_artifact_payloads": (
                {str(key): dict(value) for key, value in dict(payload.get("released_artifact_payloads") or {}).items() if isinstance(value, dict)}
                if isinstance(payload.get("released_artifact_payloads"), dict)
                else {}
            ),
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


def _prune_query_stage_cache(cache_dir, limit=QUERY_STAGE_CACHE_LIMIT):
    def prune_dir(directory):
        if directory is None:
            return
        entries = []
        for path in directory.rglob("*.json"):
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((stat.st_mtime, path))
        if len(entries) <= limit:
            return
        entries.sort(reverse=True)
        for _, path in entries[limit:]:
            path.unlink(missing_ok=True)
    prune_dir(_query_stage_cache_dir(cache_dir))
    prune_dir(_shared_query_stage_cache_dir())


def _maybe_prune_query_stage_cache(cache_dir, limit=QUERY_STAGE_CACHE_LIMIT):
    global _QUERY_STAGE_PRUNE_COUNTER
    _QUERY_STAGE_PRUNE_COUNTER += 1
    if _QUERY_STAGE_PRUNE_COUNTER % QUERY_STAGE_CACHE_PRUNE_INTERVAL != 0:
        return
    _prune_query_stage_cache(cache_dir, limit=limit)


def _write_query_stage_cache(
    cache_dir,
    query,
    cache_key,
    retrieval_signature,
    ranked,
    selected,
    evidence,
    evidence_budget_trimmed,
    *,
    retrieval_quality="gpu",
    rerank_quality="gpu",
    answer="",
    findings=None,
    warnings=None,
    provenance=None,
    runtime_attempts=None,
    synthesis_quality="not_requested",
    released_payload=None,
    released_artifact_payloads=None,
    repository_state_fingerprint="",
    build_config_digest="",
    index_fingerprint="",
    total_files=0,
    chunk_count=0,
    budgets=None,
):
    normalized_ranked = _normalize_cached_ranked(ranked)
    normalized_selected = _normalize_cached_selected(selected)
    normalized_evidence = [dict(item) for item in (evidence or ()) if isinstance(item, dict)]
    if not normalized_ranked or not normalized_selected or not normalized_evidence:
        return False
    filtered_released_artifact_payloads = (
        {
            str(key): dict(value)
            for key, value in dict(released_artifact_payloads or {}).items()
            if isinstance(value, dict) and str(key) in {"evidence_pack", "chunk_manifest"}
        }
        if isinstance(released_artifact_payloads, dict)
        else {}
    )
    payload = {
        "schema": QUERY_STAGE_CACHE_SCHEMA,
        "retrieval_signature": dict(retrieval_signature or {}),
        "retrieval_quality": str(retrieval_quality or "gpu"),
        "rerank_quality": str(rerank_quality or "gpu"),
        "ranked": normalized_ranked,
        "selected": normalized_selected,
        "evidence": normalized_evidence,
        "evidence_budget_trimmed": bool(evidence_budget_trimmed),
        "answer": str(answer or ""),
        "findings": _normalize_cached_findings(findings),
        "warnings": [str(item) for item in (warnings or ()) if str(item)],
        "provenance": dict(provenance or {}),
        "runtime_attempts": [dict(item) for item in (runtime_attempts or ()) if isinstance(item, dict)],
        "synthesis_quality": str(synthesis_quality or "not_requested"),
        "released_payload": dict(released_payload or {}) if isinstance(released_payload, dict) else {},
        "released_artifact_payloads": filtered_released_artifact_payloads,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    path = _query_stage_cache_path(cache_dir, cache_key)
    shared_path = _shared_query_stage_cache_path(cache_key)
    if shared_path is not None and shared_path != path:
        if not _query_stage_memory_payload_matches(shared_path, payload) and not _path_bytes_equal(shared_path, payload_bytes):
            _atomic_private_bytes(shared_path, payload_bytes)
            _cache_query_stage_memory_payload(shared_path, payload)
    elif not _query_stage_memory_payload_matches(path, payload) and not _path_bytes_equal(path, payload_bytes):
        _atomic_private_bytes(path, payload_bytes)
        _cache_query_stage_memory_payload(path, payload)
    if repository_state_fingerprint and build_config_digest and retrieval_signature and budgets:
        released_alias_payload = {}
        released_quality = dict(released_payload.get("quality") or {}) if isinstance(released_payload, dict) else {}
        if (
            isinstance(released_payload, dict)
            and bool(filtered_released_artifact_payloads)
            and str(released_quality.get("result") or "") == "evidence_only"
        ):
            released_alias_payload = {
                "retrieval_quality": str(retrieval_quality or "gpu"),
                "rerank_quality": str(rerank_quality or "gpu"),
                "ranked": [dict(item) for item in normalized_ranked],
                "selected": [dict(item) for item in normalized_selected],
                "evidence_budget_trimmed": bool(evidence_budget_trimmed),
                "ranked_count": len(normalized_ranked),
                "released_payload": dict(released_payload),
                "released_artifact_payloads": dict(filtered_released_artifact_payloads),
            }
        alias_payload = {
            "schema": QUERY_STAGE_CACHE_ALIAS_SCHEMA,
            "cache_key": str(cache_key),
            "index_fingerprint": str(index_fingerprint or ""),
            "total_files": int(total_files or 0),
            "chunk_count": int(chunk_count or 0),
        }
        if released_alias_payload:
            alias_payload["released_lexical_fallback"] = released_alias_payload
        alias_bytes = json.dumps(alias_payload, separators=(",", ":")).encode("utf-8")
        alias_key = _query_stage_cache_alias_key(
            query,
            repository_state_fingerprint,
            build_config_digest,
            retrieval_signature,
            budgets,
        )
        alias_path = _query_stage_cache_alias_path(cache_dir, alias_key)
        shared_alias_path = _shared_query_stage_cache_alias_path(alias_key)
        if shared_alias_path is not None and shared_alias_path != alias_path:
            if not _query_stage_alias_memory_payload_matches(shared_alias_path, alias_payload) and not _path_bytes_equal(shared_alias_path, alias_bytes):
                _atomic_private_bytes(shared_alias_path, alias_bytes)
                _cache_query_stage_alias_memory_payload(shared_alias_path, alias_payload)
        elif not _query_stage_alias_memory_payload_matches(alias_path, alias_payload) and not _path_bytes_equal(alias_path, alias_bytes):
            _atomic_private_bytes(alias_path, alias_bytes)
            _cache_query_stage_alias_memory_payload(alias_path, alias_payload)
    _maybe_prune_query_stage_cache(cache_dir)
    return True


def _query_stage_retrieval_signature(services, health_interval_seconds=None):
    if not services:
        return {"tier": "cpu-lexical-fallback", "mode": "lexical_fallback"}, None
    endpoint = _gpu("select_endpoint")(
        services,
        "p40-retrieval",
        "search",
        expected_gpu_count=1,
        health_interval_seconds=health_interval_seconds,
    )
    if endpoint is None:
        return {"tier": "cpu-lexical-fallback", "mode": "lexical_fallback"}, None
    rerank_endpoint = _gpu("select_endpoint")(
        services,
        "p40-retrieval",
        "rerank",
        expected_gpu_count=1,
        health_interval_seconds=health_interval_seconds,
    )
    if rerank_endpoint is None:
        return {"tier": "cpu-lexical-fallback", "mode": "lexical_fallback"}, None
    signature = {
        "tier": str(endpoint.get("tier") or ""),
        "model_profile": str(endpoint.get("model_profile") or ""),
        "model": str(endpoint.get("model") or ""),
        "mode": "gpu",
    }
    return signature, {"search": endpoint, "rerank": rerank_endpoint}


def gpu_rerank(
    services,
    query,
    candidates,
    chunks,
    chunk_by_id,
    attempts,
    client_factory,
    health_interval_seconds=None,
    service_requester=None,
    failure_reporter=None,
    endpoint_override=None,
):
    GPUServiceError = _gpu("GPUServiceError")
    initial_category = _unavailable_category(services, "p40-retrieval")
    demand_error = None
    endpoint = endpoint_override
    if endpoint is None:
        endpoint, demand_error = _resolve_or_request_endpoint(
            services,
            "p40-retrieval",
            "rerank",
            1,
            health_interval_seconds,
            service_requester,
            initial_category,
            "GPU reranking is required for inspect_repo",
        )
    if endpoint is None:
        category = demand_error.category if demand_error is not None else initial_category
        missing = _missing_endpoint_record("p40-retrieval", 1, demand_error)
        attempts.append(
            _attempt(
                missing,
                "rerank",
                "failed",
                category=category,
                reason="semantic_candidates_ready",
                detail=demand_error or "",
            )
        )
        return candidates, "unavailable", None
    documents = []
    valid_candidates = []
    for candidate in candidates[:64]:
        chunk = chunk_by_id.get(candidate["chunk_id"])
        if not chunk:
            continue
        documents.append(
            f"path: {chunk['path']}\nlanguage: {chunk.get('language', '')}\nsymbol: {chunk.get('symbol', '')}\n"
            f"lines: {chunk['line_start']}-{chunk['line_end']}\n{chunk['content']}"
        )
        valid_candidates.append(candidate)
    if not valid_candidates:
        attempts.append(_attempt(endpoint, "rerank", "failed", category="service_failure", reason="no_candidates"))
        return candidates, "failed", endpoint
    try:
        scores = client_factory(endpoint).rerank(query, documents)
        reranked = []
        for candidate, score in zip(valid_candidates, scores):
            reranked.append(dict(candidate, rerank_score=float(score)))
        reranked.sort(key=lambda item: (-item["rerank_score"], -item.get("rrf_score", 0.0), item["chunk_id"]))
        for index, item in enumerate(reranked, start=1):
            item["rank"] = index
        attempts.append(_attempt(endpoint, "rerank", "succeeded", reason="semantic_candidates_ready"))
        return reranked, "gpu", endpoint
    except GPUServiceError as exc:
        _report_warm_service_failure(failure_reporter, endpoint, exc, "rerank")
        attempts.append(
            _attempt(endpoint, "rerank", "failed", category=exc.category, reason="semantic_candidates_ready", detail=exc)
        )
        return candidates, "failed", endpoint


def _synthesis_messages(query, evidence, validation_feedback=""):
    system = (
        "Answer the repository question using only the released evidence. Return strict JSON matching the schema. "
        "Every finding must cite one or more evidence ids from the evidence array. Do not invent paths, symbols, "
        "line numbers, or evidence ids."
    )
    evidence_payload = [
        {
            "id": item["id"],
            "path": item["path"],
            "language": item["language"],
            "symbol": item["symbol"],
            "source_refs": item["source_refs"],
            "excerpt": item["excerpt"],
        }
        for item in evidence
    ]
    user = {"query": query, "evidence": evidence_payload}
    if validation_feedback:
        user["validation_feedback"] = validation_feedback
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=True)},
    ]


def validate_synthesis(response, evidence, synthesis_output_token_budget):
    GPUServiceError = _gpu("GPUServiceError")
    if not isinstance(response, dict):
        raise GPUServiceError("invalid_synthesis", "synthesis output must be a JSON object")
    answer = response.get("answer")
    findings = response.get("findings")
    if not isinstance(answer, str) or not answer.strip() or not isinstance(findings, list) or not findings:
        raise GPUServiceError("invalid_synthesis", "synthesis output requires a non-empty answer and findings")
    valid_refs = {item["id"] for item in evidence}
    normalized_findings = []
    for finding in findings:
        if not isinstance(finding, dict) or not isinstance(finding.get("summary"), str) or not finding["summary"].strip():
            raise GPUServiceError("invalid_synthesis", "every finding requires a non-empty summary")
        refs = finding.get("evidence_refs")
        if not isinstance(refs, list) or not refs:
            raise GPUServiceError("unsupported_claim", "every finding must cite released evidence")
        refs = [str(value) for value in refs]
        unknown = sorted(set(refs) - valid_refs)
        if unknown:
            raise GPUServiceError("unsupported_claim", f"finding cites unknown evidence ids: {unknown}")
        normalized_findings.append({"summary": finding["summary"].strip(), "evidence_refs": list(dict.fromkeys(refs))})
    mentioned_refs = set(re.findall(r"\bev_\d+\b", json.dumps(response, ensure_ascii=True)))
    unknown_mentions = sorted(mentioned_refs - valid_refs)
    if unknown_mentions:
        raise GPUServiceError("unsupported_claim", f"output mentions unknown evidence ids: {unknown_mentions}")
    normalized = {"answer": answer.strip(), "findings": normalized_findings}
    if estimate_tokens(json.dumps(normalized, ensure_ascii=True)) > synthesis_output_token_budget:
        raise GPUServiceError("model_limit", "synthesis output exceeds the remaining final_pack_token_budget")
    return normalized


def _try_synthesis_tier(
    services,
    tier,
    gpu_count,
    query,
    evidence,
    budgets,
    attempts,
    client_factory,
    escalation_reason,
    health_interval_seconds=None,
    service_requester=None,
    failure_reporter=None,
):
    GPUServiceError = _gpu("GPUServiceError")
    initial_category = _unavailable_category(services, tier)
    endpoint, demand_error = _resolve_or_request_endpoint(
        services,
        tier,
        "chat",
        gpu_count,
        health_interval_seconds,
        service_requester,
        initial_category,
        escalation_reason,
    )
    if endpoint is None:
        category = demand_error.category if demand_error is not None else initial_category
        missing = _missing_endpoint_record(tier, gpu_count, demand_error)
        attempts.append(
            _attempt(
                missing,
                "synthesis",
                "failed",
                category=category,
                reason=escalation_reason,
                number=1,
                detail=demand_error or "",
            )
        )
        return None, category, None

    feedback = ""
    for number in (1, 2):
        messages = _synthesis_messages(query, evidence, feedback)
        request_tokens = estimate_tokens(json.dumps(messages, ensure_ascii=True))
        configured_limit = budgets["synthesis_context_token_budget"]
        endpoint_limit = int(endpoint.get("context_limit_tokens") or 0)
        effective_limit = min(value for value in (configured_limit, endpoint_limit) if value > 0)
        if request_tokens > effective_limit:
            category = "context_overflow"
            detail = f"synthesis request uses {request_tokens} tokens, limit is {effective_limit}"
            attempts.append(
                _attempt(endpoint, "synthesis", "failed", category=category, reason=escalation_reason, number=number, detail=detail)
            )
            return None, category, endpoint
        try:
            response = client_factory(endpoint).chat(messages, SYNTHESIS_SCHEMA)
            output_budget = int(
                budgets.get("synthesis_output_token_budget") or budgets["final_pack_token_budget"]
            )
            validated = validate_synthesis(response, evidence, output_budget)
            attempts.append(_attempt(endpoint, "synthesis", "succeeded", reason=escalation_reason, number=number))
            return validated, "", endpoint
        except GPUServiceError as exc:
            _report_warm_service_failure(failure_reporter, endpoint, exc, "synthesis")
            attempts.append(
                _attempt(
                    endpoint,
                    "synthesis",
                    "failed",
                    category=exc.category,
                    reason=escalation_reason,
                    number=number,
                    detail=exc,
                )
            )
            if number == 1 and exc.category in {"unsupported_claim", "invalid_synthesis"}:
                feedback = (
                    f"The prior output was rejected ({exc.category}): {exc}. "
                    "Regenerate from the released evidence and cite only its ids."
                )
                continue
            return None, exc.category, endpoint
    return None, "invalid_synthesis", endpoint


def synthesize_with_escalation(
    services,
    query,
    evidence,
    budgets,
    attempts,
    client_factory,
    health_interval_seconds=None,
    service_requester=None,
    failure_reporter=None,
):
    # The first two tiers are fixed. A100 size is selected only after the V100
    # result, preserving the adaptive escalation policy and its reasons.
    plan = [("p40-synthesis", 1, "primary_synthesis")]
    last_category = ""
    for tier, gpu_count, reason in plan:
        result, last_category, endpoint = _try_synthesis_tier(
            services, tier, gpu_count, query, evidence, budgets, attempts,
            client_factory, reason, health_interval_seconds, service_requester,
            failure_reporter,
        )
        if result is not None:
            return result, endpoint

    v100_reason = f"p40_{last_category or 'service_failure'}"
    result, v100_category, endpoint = _try_synthesis_tier(
        services, "v100-reasoning", 4, query, evidence, budgets, attempts,
        client_factory, v100_reason, health_interval_seconds, service_requester,
        failure_reporter,
    )
    if result is not None:
        return result, endpoint

    a100_count = 4 if v100_category in _gpu("MULTIGPU_FAILURES") or v100_category == "unsupported_claim" else 1
    a100_tier = "a100-multigpu" if a100_count == 4 else "a100-single"
    result, _, endpoint = _try_synthesis_tier(
        services, a100_tier, a100_count, query, evidence, budgets, attempts,
        client_factory, f"v100_{v100_category or 'service_failure'}",
        health_interval_seconds, service_requester, failure_reporter,
    )
    return result, endpoint if result is not None else None


def _quality(result, retrieval, reranking, synthesis):
    ready = result == "answer_ready"
    return {
        "result": result,
        "retrieval": retrieval,
        "reranking": reranking,
        "synthesis": synthesis,
        "answer_ready": ready,
    }


def _artifact_payloads(payload, fingerprint, chunks, selected, index_path, cache_hit, full_trace=False):
    artifacts = {
        "evidence_pack": {"query": payload["query"], "evidence": payload["evidence"]},
        "retrieval_result": {
            "fingerprint": fingerprint,
            "lexical_index": str(index_path),
            "lexical_index_cache_hit": cache_hit,
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
            "chunks": [
                {key: value for key, value in chunk.items() if key != "content"}
                for chunk in chunks
            ],
            "selected_chunk_ids": [chunk["chunk_id"] for chunk in selected],
        }
    return artifacts


def _dehydrate_chunks(chunks):
    dehydrated = type("DehydratedChunkList", (list,), {})()
    for chunk in chunks:
        dehydrated.append({key: value for key, value in chunk.items() if key != "content"})
    for attr in (
        "_chunk_restore_diagnostics",
        "_chunk_build_substage_timings",
        "_lexical_manifest",
        "_index_manifest",
        "_chunks_by_file",
        "_semantic_document_signatures",
        "_chunk_ids",
        "_chunk_count",
        "_file_key_by_chunk_id",
    ):
        if hasattr(chunks, attr):
            setattr(dehydrated, attr, getattr(chunks, attr))
    return dehydrated


def _hydrate_chunk_contents(index_path, chunk_by_id, chunk_ids):
    import sqlite3

    missing_ids = [
        str(chunk_id)
        for chunk_id in chunk_ids
        if str(chunk_id)
        and chunk_by_id.get(str(chunk_id)) is not None
        and "content" not in chunk_by_id[str(chunk_id)]
    ]
    if not missing_ids:
        return
    unique_ids = list(dict.fromkeys(missing_ids))
    loaded = {}
    try:
        with sqlite3.connect(index_path) as conn:
            batch_size = 256
            for offset in range(0, len(unique_ids), batch_size):
                batch = unique_ids[offset : offset + batch_size]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"SELECT chunk_id, content FROM chunks WHERE chunk_id IN ({placeholders})",
                    batch,
                ).fetchall()
                for chunk_id, content in rows:
                    loaded[str(chunk_id)] = str(content)
    except sqlite3.Error:
        return
    for chunk_id, content in loaded.items():
        chunk = chunk_by_id.get(chunk_id)
        if chunk is not None:
            chunk["content"] = content


def _cache_dir(execution_plan, output_dir, task_params=None):
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
            job_token = str((execution_plan or {}).get("job_id") or Path(output_dir or ".").name or "job")
            return Path(scratch_root).expanduser().resolve(strict=False) / "local-ai-broker" / "inspect-repo" / job_token
        import tempfile

        scratch_root = tempfile.gettempdir()
        if scratch_root:
            job_token = str((execution_plan or {}).get("job_id") or Path(output_dir or ".").name or "job")
            return Path(scratch_root).expanduser().resolve(strict=False) / "local-ai-broker" / "inspect-repo" / job_token
    if configured:
        return Path(configured).expanduser().resolve(strict=False)
    return Path(output_dir or ".").expanduser().resolve(strict=False) / "repo-inspection-v2-cache"


def _has_custom_exclusions(task_params):
    if not isinstance(task_params, dict) or not task_params:
        return False
    for key in ("excluded_dir_names", "exclude_dirs", "excluded_paths"):
        value = task_params.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return True
            continue
        if isinstance(value, (list, tuple, set)):
            if any(str(item).strip() for item in value if item is not None):
                return True
    return False


def _repository_fingerprint_hint(discovered, task_params):
    task_params = task_params if isinstance(task_params, dict) else {}
    if _has_custom_exclusions(task_params):
        return "", []
    broker_hint = str(task_params.get("_broker_repository_state_fingerprint") or "").strip()
    if broker_hint:
        source = str(task_params.get("_broker_repository_state_fingerprint_source") or "request_cache").strip()
        source = source or "request_cache"
        return broker_hint, fingerprint_hint_state(broker_hint, source)
    if len(discovered or ()) != 1:
        return "", []
    item = discovered[0] if isinstance(discovered[0], dict) else {}
    input_type = str(item.get("type") or "").strip().lower()
    content_hash = str(item.get("content_hash") or "").strip()
    if content_hash and input_type in {"repo", "directory"}:
        return content_hash, [{"kind": "input_manifest", "source": "input_manifest", "fingerprint": content_hash}]
    return "", []


def _broker_touched_paths_hint(task_params):
    if not isinstance(task_params, dict):
        return ()
    raw_paths = task_params.get("_broker_touched_paths")
    if not isinstance(raw_paths, (list, tuple, set)):
        return ()
    normalized = []
    seen = set()
    for raw_path in raw_paths:
        path = str(raw_path or "").strip()
        if not path:
            continue
        if path not in seen:
            seen.add(path)
            normalized.append(path)
    return tuple(normalized)


def _broker_clean_worktree_files_hint(task_params):
    if not isinstance(task_params, dict):
        return ()
    raw_paths = task_params.get("_broker_clean_worktree_files")
    if not isinstance(raw_paths, (list, tuple, set)):
        return ()
    normalized = []
    seen = set()
    for raw_path in raw_paths:
        path = str(raw_path or "").strip()
        if not path:
            continue
        path = PurePosixPath(path).as_posix()
        if path in {"", "."} or path in seen:
            continue
        seen.add(path)
        normalized.append(path)
    return tuple(sorted(normalized))


def _fingerprint_state_touched_paths_hint(fingerprint_state):
    if not isinstance(fingerprint_state, (list, tuple)):
        return ()
    normalized = []
    seen = set()
    for state in fingerprint_state:
        if not isinstance(state, dict):
            continue
        raw_paths = state.get("dirty_paths")
        if not isinstance(raw_paths, (list, tuple, set)):
            continue
        for raw_path in raw_paths:
            path = str(raw_path or "").strip()
            if not path or path in seen:
                continue
            seen.add(path)
            normalized.append(path)
    return tuple(normalized)


def _fingerprint_state_has_clean_worktree_files(fingerprint_state):
    if not isinstance(fingerprint_state, (list, tuple)):
        return False
    for state in fingerprint_state:
        if not isinstance(state, dict):
            continue
        raw_files = state.get("clean_worktree_files")
        if isinstance(raw_files, (list, tuple, set)) and any(str(item or "").strip() for item in raw_files):
            return True
    return False


def run_inspection(
    discovered,
    query,
    *,
    mode="auto",
    constraints=None,
    task_params=None,
    execution_plan=None,
    output_dir=None,
    services=None,
    client_factory=None,
    prefetched_state=None,
):
    context = _inspection_context(query, mode, constraints, task_params, execution_plan, client_factory)
    query, mode = context.query, context.mode
    constraints, task_params, execution_plan = context.constraints, context.task_params, context.execution_plan
    budgets, client_factory = context.budgets, context.client_factory
    parsed_query_features = None
    setup_stage_timings = {}
    health_interval_seconds = (
        execution_plan.get("gpu_service_health_interval_seconds")
        or os.environ.get("BROKER_GPU_SERVICE_HEALTH_INTERVAL_SECONDS")
    )
    if health_interval_seconds is not None:
        health_interval_seconds = float(health_interval_seconds)
    stage_started = time.perf_counter()
    if services is not None:
        services = list(services)
    elif _gpu_registry_configured(execution_plan):
        services = list(_gpu("services_from_execution_plan")(execution_plan, task_params))
    else:
        services = []
    setup_stage_timings["load_services_ms"] = _elapsed_ms(stage_started)
    stage_started = time.perf_counter()
    if _service_control_configured(execution_plan):
        service_requester = _service_control("requester_from_execution_plan")(execution_plan)
        failure_reporter = _service_control("failure_reporter_from_execution_plan")(execution_plan)
    else:
        service_requester = None
        failure_reporter = None
    setup_stage_timings["service_control_hooks_ms"] = _elapsed_ms(stage_started)
    excluded = set(task_params.get("excluded_dir_names") or task_params.get("exclude_dirs") or [])
    cache_dir = _cache_dir(execution_plan, output_dir, task_params=task_params)
    excluded_paths = exclusion_paths_for_execution(execution_plan, output_dir, task_params=task_params)
    transient_excluded_paths = transient_excluded_paths_for_execution(output_dir)
    git_probe_cache = {}
    repository_state_fingerprint = ""
    fingerprint_state = []
    cached_chunk_snapshot_metadata = None
    metadata_chunks = None
    fingerprint = ""
    build_config_digest = ""
    touched_paths_hint = ()
    prefetched_query_stage_cache_key = ""
    prefetched_query_stage_cache_probed = False
    prefetched_query_stage_requires_verification = False
    prefetched_retrieval_signature = {}
    prefetched_cached_query_stage = None
    stage_started = time.perf_counter()
    if isinstance(prefetched_state, dict):
        repository_state_fingerprint = str(prefetched_state.get("repository_state_fingerprint") or "")
        fingerprint_state = list(prefetched_state.get("fingerprint_state") or [])
        cached_chunk_snapshot_metadata = prefetched_state.get("cached_chunk_snapshot_metadata")
        metadata_chunks = prefetched_state.get("metadata_chunks")
        fingerprint = str(prefetched_state.get("fingerprint") or "")
        build_config_digest = str(prefetched_state.get("build_config_digest") or "")
        prefetched_query_stage_cache_key = str(prefetched_state.get("prefetched_query_stage_cache_key") or "")
        prefetched_query_stage_cache_probed = bool(prefetched_state.get("prefetched_query_stage_cache_probed"))
        prefetched_query_stage_requires_verification = bool(
            prefetched_state.get("prefetched_query_stage_requires_verification")
        )
        prefetched_retrieval_signature = dict(prefetched_state.get("prefetched_retrieval_signature") or {})
        if not prefetched_query_stage_requires_verification:
            prefetched_cached_query_stage = prefetched_state.get("cached_query_stage")
        prefetched_cache_dir = prefetched_state.get("cache_dir")
        if prefetched_cache_dir:
            cache_dir = prefetched_cache_dir
            excluded_paths = {cache_dir}
        prefetched_excluded = prefetched_state.get("excluded")
        if prefetched_excluded is not None:
            excluded = set(prefetched_excluded)
        prefetched_probe_cache = prefetched_state.get("git_probe_cache")
        if isinstance(prefetched_probe_cache, dict):
            git_probe_cache = prefetched_probe_cache
    setup_stage_timings["apply_prefetched_state_ms"] = _elapsed_ms(stage_started)
    touched_paths_hint = _broker_touched_paths_hint(task_params)
    if not repository_state_fingerprint:
        repository_state_fingerprint, fingerprint_state = _repository_fingerprint_hint(discovered, task_params)
    if not repository_state_fingerprint:
        stage_started = time.perf_counter()
        repository_state_fingerprint, fingerprint_state = repository_fingerprint(
            discovered,
            excluded,
            excluded_paths=excluded_paths,
            cache_dir=cache_dir,
            git_probe_cache=git_probe_cache,
        )
        setup_stage_timings["repository_fingerprint_ms"] = _elapsed_ms(stage_started)
    else:
        setup_stage_timings["repository_fingerprint_ms"] = 0.0
    if not touched_paths_hint:
        touched_paths_hint = _fingerprint_state_touched_paths_hint(fingerprint_state)
    broker_clean_worktree_files_hint = _broker_clean_worktree_files_hint(task_params)
    if (
        repository_state_fingerprint
        and broker_clean_worktree_files_hint
        and not touched_paths_hint
        and not _fingerprint_state_has_clean_worktree_files(fingerprint_state)
    ):
        fingerprint_state = list(fingerprint_state or [])
        fingerprint_state.append(
            {
                "kind": "broker_hint",
                "source": "broker_clean_worktree_files",
                "fingerprint": repository_state_fingerprint,
                "clean_worktree_files": list(broker_clean_worktree_files_hint),
            }
        )
    if cached_chunk_snapshot_metadata is None:
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
        setup_stage_timings["load_cached_snapshot_metadata_ms"] = _elapsed_ms(stage_started)
    else:
        setup_stage_timings["load_cached_snapshot_metadata_ms"] = 0.0
    fingerprint_from_metadata = False
    if metadata_chunks is None and cached_chunk_snapshot_metadata is not None:
        stage_started = time.perf_counter()
        metadata_chunks = type("MetadataChunkList", (list,), {})()
        metadata_chunks._index_manifest = dict(cached_chunk_snapshot_metadata["index_manifest"])
        metadata_chunks._semantic_document_signatures = dict(cached_chunk_snapshot_metadata["semantic_document_signatures"])
        metadata_chunks._chunk_ids = tuple(cached_chunk_snapshot_metadata["chunk_ids"])
        metadata_chunks._chunk_count = int(cached_chunk_snapshot_metadata.get("chunk_count") or len(metadata_chunks._chunk_ids))
        setup_stage_timings["hydrate_metadata_chunks_ms"] = _elapsed_ms(stage_started)
    else:
        setup_stage_timings["hydrate_metadata_chunks_ms"] = 0.0
    if not fingerprint and metadata_chunks is not None:
        stage_started = time.perf_counter()
        fingerprint = inspection_index_fingerprint(repository_state_fingerprint, metadata_chunks)
        setup_stage_timings["inspection_index_fingerprint_ms"] = _elapsed_ms(stage_started)
        fingerprint_from_metadata = True
    else:
        setup_stage_timings["inspection_index_fingerprint_ms"] = 0.0
    warnings = []
    fingerprint_hint_only = any(
        str((item or {}).get("kind") or "") in {"input_manifest", "broker_hint"}
        for item in (fingerprint_state or [])
        if isinstance(item, dict)
    )
    stage_started = time.perf_counter()
    retrieval_signature, selected_retrieval_endpoints = _query_stage_retrieval_signature(
        services,
        health_interval_seconds=health_interval_seconds,
    )
    setup_stage_timings["query_stage_signature_ms"] = _elapsed_ms(stage_started)
    cached_retrieval_endpoint = None
    cached_rerank_endpoint = None
    if isinstance(selected_retrieval_endpoints, dict):
        cached_retrieval_endpoint = selected_retrieval_endpoints.get("search")
        cached_rerank_endpoint = selected_retrieval_endpoints.get("rerank")
    else:
        cached_retrieval_endpoint = selected_retrieval_endpoints

    query_stage_cache_key = None
    cached_query_stage = None
    if fingerprint and retrieval_signature is not None:
        stage_started = time.perf_counter()
        query_stage_cache_key = _query_stage_cache_key(query, fingerprint, retrieval_signature, budgets)
        if (
            not prefetched_query_stage_requires_verification
            and
            prefetched_query_stage_cache_probed
            and prefetched_query_stage_cache_key == query_stage_cache_key
            and dict(prefetched_retrieval_signature or {}) == dict(retrieval_signature or {})
        ):
            if cached_query_stage is None and isinstance(prefetched_cached_query_stage, dict):
                cached_query_stage = dict(prefetched_cached_query_stage)
            setup_stage_timings["query_stage_cache_probe_ms"] = _elapsed_ms(stage_started)
        else:
            cached_query_stage = _load_query_stage_cache(cache_dir, query_stage_cache_key)
            setup_stage_timings["query_stage_cache_probe_ms"] = _elapsed_ms(stage_started)
    else:
        setup_stage_timings["query_stage_cache_probe_ms"] = 0.0
    if cached_query_stage is not None and metadata_chunks is not None:
        total_files = int(cached_chunk_snapshot_metadata.get("total_files") or 0)
        chunks = metadata_chunks
        chunk_cache_stats = {
            "total_files": total_files,
            "reused_files": total_files,
            "rebuilt_files": 0,
            "snapshot_cache_hit": True,
        }
        pre_retrieval_stage_timings = {
            "build_syntax_chunks_ms": 0.0,
        }
    else:
        chunks = None
        chunk_cache_stats = None
        index_path = Path(cache_dir) / "lexical-working.sqlite3"
        if metadata_chunks is not None and fingerprint:
            stage_started = time.perf_counter()
            chunks = load_chunks_from_lexical_manifest(index_path, fingerprint)
            if chunks is None:
                chunks = load_chunks_from_lexical_index(index_path, fingerprint, include_content=False)
            pre_retrieval_stage_timings = {
                "build_syntax_chunks_ms": _elapsed_ms(stage_started),
            }
            if chunks is not None:
                total_files = int(cached_chunk_snapshot_metadata.get("total_files") or 0)
                chunk_cache_stats = {
                    "total_files": total_files,
                    "reused_files": total_files,
                    "rebuilt_files": 0,
                    "snapshot_cache_hit": True,
                        "full_snapshot_reload": True,
                    }
            else:
                stage_started = time.perf_counter()
                chunks = load_cached_chunk_snapshot(
                    cache_dir,
                    repository_state_fingerprint=repository_state_fingerprint,
                    build_config_digest=build_config_digest or str(cached_chunk_snapshot_metadata.get("build_config_digest") or ""),
                )
                pre_retrieval_stage_timings = {
                    "build_syntax_chunks_ms": _elapsed_ms(stage_started),
                }
                if chunks is not None:
                    total_files = int(cached_chunk_snapshot_metadata.get("total_files") or 0)
                    chunk_cache_stats = {
                        "total_files": total_files,
                        "reused_files": total_files,
                        "rebuilt_files": 0,
                        "snapshot_cache_hit": True,
                        "full_snapshot_reload": True,
                    }
                else:
                    stage_started = time.perf_counter()
                    chunks = load_chunks_from_lexical_index(index_path, fingerprint, include_content=True)
                    pre_retrieval_stage_timings = {
                        "build_syntax_chunks_ms": _elapsed_ms(stage_started),
                    }
                    if chunks is not None:
                        total_files = int(cached_chunk_snapshot_metadata.get("total_files") or 0)
                        chunk_cache_stats = {
                            "total_files": total_files,
                            "reused_files": total_files,
                            "rebuilt_files": 0,
                            "snapshot_cache_hit": True,
                            "lexical_index_chunk_reload": True,
                        }
        if chunks is None:
            stage_started = time.perf_counter()
            chunks, chunk_cache_stats = build_syntax_chunks(
                discovered,
                excluded,
                cache_dir=cache_dir,
                excluded_paths=excluded_paths,
                repository_state_fingerprint=repository_state_fingerprint,
                repository_fingerprint_state=fingerprint_state,
                git_probe_cache=git_probe_cache,
                return_diagnostics=True,
                transient_excluded_paths=transient_excluded_paths,
                touched_paths_hint=touched_paths_hint,
                publish_shared_cache_publication=not bool(task_params.get("_broker_skip_shared_chunk_publish")),
            )
            pre_retrieval_stage_timings = {
                "build_syntax_chunks_ms": _elapsed_ms(stage_started),
            }
        if bool(task_params.get("_broker_skip_shared_lexical_publish")):
            try:
                chunks._skip_shared_cache_publication = True
            except Exception:
                pass
        fingerprint_recomputed = False
        if not fingerprint or fingerprint_from_metadata:
            fingerprint = inspection_index_fingerprint(repository_state_fingerprint, chunks)
            fingerprint_from_metadata = False
            fingerprint_recomputed = True
        if (query_stage_cache_key is None or fingerprint_recomputed) and fingerprint and retrieval_signature is not None:
            query_stage_cache_key = _query_stage_cache_key(query, fingerprint, retrieval_signature, budgets)
        if cached_query_stage is None and query_stage_cache_key is not None:
            if not (
                not prefetched_query_stage_requires_verification
                and
                prefetched_query_stage_cache_probed
                and prefetched_query_stage_cache_key == query_stage_cache_key
                and dict(prefetched_retrieval_signature or {}) == dict(retrieval_signature or {})
            ):
                stage_started = time.perf_counter()
                cached_query_stage = _load_query_stage_cache(cache_dir, query_stage_cache_key)
                setup_stage_timings["query_stage_cache_probe_ms"] += _elapsed_ms(stage_started)
    try:
        if cached_query_stage is not None:
            index_path = Path(cache_dir) / "lexical-working.sqlite3"
            lexical_cache_hit = False
            lexical_index_stats = {
                "working_cache_hit": False,
                "updated_files": 0,
                "removed_files": 0,
                "inserted_chunks": 0,
                "skipped_by_query_stage_cache": True,
            }
            pre_retrieval_stage_timings["ensure_lexical_index_ms"] = 0.0
        elif chunk_cache_stats is not None and chunk_cache_stats.get("lexical_index_chunk_reload"):
            lexical_cache_hit = True
            lexical_index_stats = {
                "working_cache_hit": True,
                "updated_files": 0,
                "removed_files": 0,
                "inserted_chunks": 0,
                "reloaded_from_index": True,
            }
            pre_retrieval_stage_timings["ensure_lexical_index_ms"] = 0.0
        elif chunk_cache_stats is not None and chunk_cache_stats.get("full_snapshot_reload"):
            stage_started = time.perf_counter()
            lexical_ready = lexical_working_index_is_current(cache_dir, fingerprint, len(chunks))
            lexical_check_ms = _elapsed_ms(stage_started)
            if lexical_ready:
                lexical_cache_hit = True
                lexical_index_stats = {
                    "working_cache_hit": True,
                    "updated_files": 0,
                    "removed_files": 0,
                    "inserted_chunks": 0,
                    "reloaded_from_snapshot": True,
                }
                pre_retrieval_stage_timings["ensure_lexical_index_ms"] = lexical_check_ms
            else:
                stage_started = time.perf_counter()
                index_path, lexical_cache_hit, lexical_index_stats = ensure_lexical_index(
                    chunks,
                    cache_dir,
                    fingerprint,
                    build_config_digest=build_config_digest,
                )
                pre_retrieval_stage_timings["ensure_lexical_index_ms"] = _elapsed_ms(stage_started) + lexical_check_ms
        else:
            stage_started = time.perf_counter()
            index_path, lexical_cache_hit, lexical_index_stats = ensure_lexical_index(
                chunks,
                cache_dir,
                fingerprint,
                build_config_digest=build_config_digest,
            )
            pre_retrieval_stage_timings["ensure_lexical_index_ms"] = _elapsed_ms(stage_started)
    except (OSError, ValueError) as exc:
        fallback_cache = Path(output_dir or ".") / "repo-inspection-v2-cache"
        stage_started = time.perf_counter()
        index_path, lexical_cache_hit, lexical_index_stats = ensure_lexical_index(
            chunks,
            fallback_cache,
            fingerprint,
            build_config_digest=build_config_digest,
        )
        pre_retrieval_stage_timings["ensure_lexical_index_ms"] = _elapsed_ms(stage_started)
        warnings.append(f"LEXICAL_CACHE_FALLBACK:{type(exc).__name__}")
    cache_dir = index_path.parent

    attempts = []
    if cached_query_stage is not None:
        retrieval_quality = str(cached_query_stage.get("retrieval_quality") or "gpu")
        rerank_quality = str(cached_query_stage.get("rerank_quality") or "gpu")
        semantic_cache_hit = retrieval_quality == "gpu"
        semantic, semantic_index_stats, retrieval_endpoint = [], {
            "cache_hit": semantic_cache_hit,
            "document_count": _chunk_count(chunks),
            "embedded_documents": 0,
            "reused_documents": 0,
        }, cached_retrieval_endpoint if semantic_cache_hit else None
        pre_retrieval_stage_timings["gpu_semantic_retrieval_ms"] = 0.0
    elif chunks and services:
        stage_started = time.perf_counter()
        semantic, retrieval_quality, semantic_index_stats, retrieval_endpoint = gpu_semantic_retrieval(
            services,
            chunks,
            query,
            fingerprint,
            build_config_digest,
            cache_dir,
            attempts,
            client_factory,
            health_interval_seconds,
            service_requester,
            failure_reporter,
            cached_retrieval_endpoint,
        )
        pre_retrieval_stage_timings["gpu_semantic_retrieval_ms"] = _elapsed_ms(stage_started)
    elif chunks:
        semantic, retrieval_quality, semantic_index_stats, retrieval_endpoint = [], "lexical_degraded", {
            "cache_hit": False,
            "document_count": _chunk_count(chunks),
            "embedded_documents": 0,
            "reused_documents": 0,
        }, None
        pre_retrieval_stage_timings["gpu_semantic_retrieval_ms"] = 0.0
    else:
        semantic, retrieval_quality, semantic_index_stats, retrieval_endpoint = [], "failed", {
            "cache_hit": False,
            "document_count": 0,
            "embedded_documents": 0,
            "reused_documents": 0,
        }, None
        warnings.append("NO_SUPPORTED_REPOSITORY_SOURCES")
        pre_retrieval_stage_timings["gpu_semantic_retrieval_ms"] = 0.0
    stage_started = time.perf_counter()
    chunks = _dehydrate_chunks(chunks)
    tail_stage_timings = {
        "dehydrate_chunks_ms": _elapsed_ms(stage_started),
    }
    lexical = []
    lexical_key = None
    path_catalog = None
    named_paths = set()
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    fused = []
    query_stage_cache_hit = False
    retrieval_stage_timings = {}
    retrieval_stage_timings.update(pre_retrieval_stage_timings)
    if cached_query_stage is not None:
        ranked = cached_query_stage["ranked"]
        selected = cached_query_stage["selected"]
        evidence = [dict(item) for item in cached_query_stage["evidence"]]
        rerank_endpoint = cached_rerank_endpoint if rerank_quality == "gpu" else None
        retrieval_endpoint = cached_retrieval_endpoint if retrieval_quality == "gpu" else None
        query_stage_cache_hit = True
        if retrieval_endpoint is not None:
            attempts.append(_attempt(retrieval_endpoint, "semantic_retrieval", "succeeded", reason="query_stage_cache_hit"))
        if rerank_endpoint is not None:
            attempts.append(_attempt(rerank_endpoint, "rerank", "succeeded", reason="query_stage_cache_hit"))
        candidate_tokens = sum(int((chunk_by_id.get(item["chunk_id"]) or {}).get("token_estimate") or 0) for item in ranked[:64])
        fused = [dict(item) for item in ranked[:64]]
        evidence_tokens = estimate_tokens(json.dumps(evidence, ensure_ascii=True)) if evidence else 0
        evidence_budget_trimmed = bool(cached_query_stage.get("evidence_budget_trimmed"))
        retrieval_stage_timings["query_stage_cache_ms"] = 0.0
    else:
        stage_started = time.perf_counter()
        parsed_query_features = query_features(query)
        retrieval_stage_timings["query_features_ms"] = _elapsed_ms(stage_started)

        lexical_key = lexical_cache_key(index_path, chunks)
        needs_path_catalog = bool((parsed_query_features.get("paths") or ()))
        lexical_helper_state = None
        if needs_path_catalog:
            stage_started = time.perf_counter()
            lexical_helper_state = lexical_helper(index_path, chunks, cache_key=lexical_key)
            path_catalog = lexical_path_catalog(index_path, chunks, cache_key=lexical_key, helper=lexical_helper_state)
            retrieval_stage_timings["path_catalog_ms"] = _elapsed_ms(stage_started)
        else:
            retrieval_stage_timings["path_catalog_ms"] = 0.0

        stage_started = time.perf_counter()
        lexical = lexical_search(
            index_path,
            query,
            chunks,
            limit=128,
            cache_key=lexical_key,
            features=parsed_query_features,
            helper=lexical_helper_state,
        )
        retrieval_stage_timings["lexical_search_ms"] = _elapsed_ms(stage_started)

        if needs_path_catalog:
            stage_started = time.perf_counter()
            named_paths = explicitly_named_paths(
                query,
                chunks,
                features=parsed_query_features,
                cache_token=lexical_key,
                path_catalog=path_catalog,
            )
            retrieval_stage_timings["named_paths_ms"] = _elapsed_ms(stage_started)
        else:
            named_paths = set()
            retrieval_stage_timings["named_paths_ms"] = 0.0

        stage_started = time.perf_counter()
        fused = reciprocal_rank_fusion([lexical, semantic] if semantic else [lexical], limit=64)
        candidates, candidate_tokens = _candidate_budget(fused, chunk_by_id, budgets["retrieval_token_budget"])
        retrieval_stage_timings["fuse_and_budget_ms"] = _elapsed_ms(stage_started)

        if retrieval_quality == "gpu":
            stage_started = time.perf_counter()
            _hydrate_chunk_contents(index_path, chunk_by_id, [item["chunk_id"] for item in candidates[:64]])
            retrieval_stage_timings["hydrate_rerank_candidates_ms"] = _elapsed_ms(stage_started)

            stage_started = time.perf_counter()
            ranked, rerank_quality, rerank_endpoint = gpu_rerank(
                services,
                query,
                candidates,
                chunks,
                chunk_by_id,
                attempts,
                client_factory,
                health_interval_seconds,
                service_requester,
                failure_reporter,
                cached_rerank_endpoint,
            )
            retrieval_stage_timings["gpu_rerank_ms"] = _elapsed_ms(stage_started)
        else:
            ranked, rerank_quality, rerank_endpoint = candidates, "unavailable", None
            retrieval_stage_timings["hydrate_rerank_candidates_ms"] = 0.0
            retrieval_stage_timings["gpu_rerank_ms"] = 0.0

        stage_started = time.perf_counter()
        selected, evidence_tokens = select_diverse_chunks(
            ranked,
            query,
            budgets["evidence_token_budget"],
            limit=12,
            chunk_by_id=chunk_by_id,
            features=parsed_query_features,
            named_paths=named_paths,
            cache_token=lexical_key,
            path_catalog=path_catalog,
        )
        retrieval_stage_timings["select_diverse_chunks_ms"] = _elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        _hydrate_chunk_contents(index_path, chunk_by_id, [item["chunk_id"] for item in selected])
        evidence = build_evidence(selected, chunk_by_id)
        evidence, evidence_budget_trimmed = trim_evidence_for_final_pack(
            evidence,
            {},
            budgets["evidence_token_budget"],
        )
        retrieval_stage_timings["build_evidence_ms"] = _elapsed_ms(stage_started)
        if (
            retrieval_quality != "failed"
            and query_stage_cache_key is not None
            and retrieval_signature is not None
            and ranked
            and selected
            and evidence
            and mode != "evidence"
        ):
            stage_started = time.perf_counter()
            _write_query_stage_cache(
                cache_dir,
                query,
                query_stage_cache_key,
                retrieval_signature,
                ranked,
                selected,
                evidence,
                evidence_budget_trimmed,
                retrieval_quality=retrieval_quality,
                rerank_quality=rerank_quality,
                repository_state_fingerprint=repository_state_fingerprint,
                build_config_digest=build_config_digest,
                index_fingerprint=fingerprint,
                total_files=int(chunk_cache_stats.get("total_files") or 0),
                chunk_count=int(getattr(chunks, "_chunk_count", 0) or len(getattr(chunks, "_chunk_ids", ()) or ()) or len(chunks or [])),
                budgets=budgets,
            )
            retrieval_stage_timings["write_query_stage_cache_ms"] = _elapsed_ms(stage_started)
        else:
            retrieval_stage_timings["write_query_stage_cache_ms"] = 0.0
    if evidence_budget_trimmed:
        warnings.append("EVIDENCE_TOKEN_BUDGET_TRIMMED")
    if chunks and retrieval_quality != "gpu":
        warnings.append("GPU_RETRIEVAL_UNAVAILABLE_LEXICAL_FALLBACK")
    if chunks and rerank_quality != "gpu":
        warnings.append("GPU_RERANK_UNAVAILABLE")
    if not evidence:
        retrieval_quality = "failed"
        warnings.append("NO_REPOSITORY_EVIDENCE")

    provenance = {
        "repository_fingerprint": repository_state_fingerprint,
        "index_fingerprint": fingerprint,
        "retrieval_model_profile": str((retrieval_endpoint or {}).get("model_profile") or ""),
        "rerank_model_profile": str((rerank_endpoint or {}).get("model_profile") or ""),
    }
    synthesis_reserve = 0
    if mode != "evidence" and retrieval_quality == "gpu" and rerank_quality == "gpu":
        synthesis_reserve = max(128, min(1_024, budgets["final_pack_token_budget"] // 4))
    trim_base = {
        "mode": mode,
        "query": query,
        "findings": [],
        "quality": _quality("evidence_only", retrieval_quality, rerank_quality, "not_requested"),
        "warnings": list(dict.fromkeys(warnings)),
        "provenance": provenance,
    }
    stage_started = time.perf_counter()
    evidence, final_pack_trimmed = trim_evidence_for_final_pack(
        evidence,
        trim_base,
        budgets["final_pack_token_budget"],
        synthesis_reserve,
    )
    tail_stage_timings["final_pack_trim_ms"] = _elapsed_ms(stage_started)
    if final_pack_trimmed:
        warnings.append("FINAL_PACK_EVIDENCE_TRIMMED")
    stage_started = time.perf_counter()
    evidence_tokens = estimate_tokens(json.dumps(evidence, ensure_ascii=True)) if evidence else 0
    tail_stage_timings["estimate_evidence_tokens_ms"] = _elapsed_ms(stage_started)
    if not evidence and "NO_REPOSITORY_EVIDENCE" not in warnings:
        warnings.append("FINAL_PACK_BUDGET_EXHAUSTED")

    diagnostics = {
        "retrieval": {
            "fingerprint": fingerprint,
            "fingerprint_sources": [state.get("kind", "") for state in fingerprint_state],
            "chunks_indexed": _chunk_count(chunks),
            "lexical_candidates": len(lexical),
            "semantic_candidates": len(semantic),
            "fused_candidates": len(fused),
            "reranked_candidates": len(ranked) if rerank_quality == "gpu" else 0,
            "selected_evidence": len(evidence),
            "candidate_tokens": candidate_tokens,
            "evidence_tokens": evidence_tokens,
            "chunk_cache_total_files": int(chunk_cache_stats.get("total_files") or 0),
            "chunk_cache_reused_files": int(chunk_cache_stats.get("reused_files") or 0),
            "chunk_cache_rebuilt_files": int(chunk_cache_stats.get("rebuilt_files") or 0),
            "chunk_manifest_restore_ms": float(
                (getattr(chunks, "_chunk_restore_diagnostics", {}) or {}).get("manifest_restore_ms") or 0.0
            ),
            "chunk_shared_manifest_load_ms": float(
                (getattr(chunks, "_chunk_restore_diagnostics", {}) or {}).get("shared_manifest_load_ms") or 0.0
            ),
            "chunk_snapshot_local_load_ms": float(
                (getattr(chunks, "_chunk_restore_diagnostics", {}) or {}).get("snapshot_local_load_ms") or 0.0
            ),
            "chunk_snapshot_shared_load_ms": float(
                (getattr(chunks, "_chunk_restore_diagnostics", {}) or {}).get("snapshot_shared_load_ms") or 0.0
            ),
            "chunk_snapshot_restore_source": str(
                (getattr(chunks, "_chunk_restore_diagnostics", {}) or {}).get("snapshot_restore_source") or ""
            ),
            "chunk_build_substage_timings_ms": dict(getattr(chunks, "_chunk_build_substage_timings", {}) or {}),
            "lexical_index_cache_hit": lexical_cache_hit,
            "lexical_index_working_cache_hit": bool(lexical_index_stats.get("working_cache_hit")),
            "lexical_index_updated_files": int(lexical_index_stats.get("updated_files") or 0),
            "lexical_index_removed_files": int(lexical_index_stats.get("removed_files") or 0),
            "lexical_index_inserted_chunks": int(lexical_index_stats.get("inserted_chunks") or 0),
            "lexical_index_working_manifest_load_ms": float(lexical_index_stats.get("working_manifest_load_ms") or 0.0),
            "lexical_index_working_check_ms": float(lexical_index_stats.get("working_index_check_ms") or 0.0),
            "lexical_index_shared_restore_ms": float(lexical_index_stats.get("shared_restore_ms") or 0.0),
            "lexical_index_sqlite_update_ms": float(lexical_index_stats.get("sqlite_update_ms") or 0.0),
            "lexical_index_sqlite_rebuild_ms": float(lexical_index_stats.get("sqlite_rebuild_ms") or 0.0),
            "query_stage_cache_hit": query_stage_cache_hit,
            "semantic_index_cache_hit": bool(semantic_index_stats.get("cache_hit")),
            "semantic_index_document_count": int(semantic_index_stats.get("document_count") or 0),
            "semantic_index_embedded_documents": int(semantic_index_stats.get("embedded_documents") or 0),
            "semantic_index_reused_documents": int(semantic_index_stats.get("reused_documents") or 0),
            "setup_timings_ms": setup_stage_timings,
            "stage_timings_ms": retrieval_stage_timings,
            "tail_timings_ms": tail_stage_timings,
            "budgets": budgets,
        },
        "runtime": {
            "attempts": attempts,
            **cache_runtime_diagnostics(execution_plan, output_dir, cache_dir),
        },
    }
    cached_answer_ready = (
        cached_query_stage is not None
        and mode != "evidence"
        and retrieval_quality == "gpu"
        and rerank_quality == "gpu"
        and str(cached_query_stage.get("synthesis_quality") or "") == "gpu"
        and str(cached_query_stage.get("answer") or "").strip()
        and bool(cached_query_stage.get("findings"))
        and bool(cached_query_stage.get("runtime_attempts"))
    )
    if cached_answer_ready:
        cached_provenance = dict(cached_query_stage.get("provenance") or {})
        provenance.update({key: value for key, value in cached_provenance.items() if value not in ("", None)})
        payload = {
            "mode": mode,
            "query": query,
            "answer": str(cached_query_stage.get("answer") or ""),
            "findings": [dict(item) for item in cached_query_stage.get("findings") or ()],
            "evidence": evidence,
            "quality": _quality("answer_ready", "gpu", "gpu", "gpu"),
            "warnings": list(dict.fromkeys([*cached_query_stage.get("warnings", []), *warnings])),
            "provenance": provenance,
            "retrieval": diagnostics["retrieval"],
            "runtime": {
                "attempts": [dict(item) for item in cached_query_stage.get("runtime_attempts") or ()],
            },
        }
        stage_started = time.perf_counter()
        diagnostics["retrieval"]["final_pack_tokens"] = released_pack_tokens(payload)
        tail_stage_timings["cached_answer_final_pack_tokens_ms"] = _elapsed_ms(stage_started)
        stage_started = time.perf_counter()
        artifacts = _artifact_payloads(
            payload, fingerprint, chunks, selected, index_path, lexical_cache_hit, bool(task_params.get("include_full_trace"))
        )
        tail_stage_timings["artifact_payloads_ms"] = _elapsed_ms(stage_started)
        return {"payload": payload, "artifact_payloads": artifacts}

    def terminal_payload(result_state, synthesis_state):
        terminal_started = time.perf_counter()
        result = {
            "mode": mode,
            "query": query,
            "findings": [],
            "evidence": evidence,
            "quality": _quality(result_state, retrieval_quality, rerank_quality, synthesis_state),
            "warnings": list(dict.fromkeys(warnings)),
            "provenance": provenance,
            "retrieval": diagnostics["retrieval"],
            "runtime": diagnostics["runtime"],
        }
        tail_stage_timings["terminal_payload_build_ms"] = _elapsed_ms(terminal_started)
        if released_pack_tokens(result) > budgets["final_pack_token_budget"]:
            stage_started = time.perf_counter()
            base = dict(result)
            base.pop("evidence", None)
            compacted, _ = trim_evidence_for_final_pack(
                result["evidence"],
                base,
                max(1, budgets["final_pack_token_budget"] - 16),
            )
            result["evidence"] = compacted
            if "FINAL_PACK_EVIDENCE_TRIMMED" not in result["warnings"]:
                result["warnings"].append("FINAL_PACK_EVIDENCE_TRIMMED")
            tail_stage_timings["terminal_payload_compact_trim_ms"] = _elapsed_ms(stage_started)
        else:
            tail_stage_timings["terminal_payload_compact_trim_ms"] = 0.0
        stage_started = time.perf_counter()
        if released_pack_tokens(result) > budgets["final_pack_token_budget"]:
            raise ValueError("final_pack_token_budget is too small for the required v2 contract fields")
        diagnostics["retrieval"]["final_pack_tokens"] = released_pack_tokens(result)
        tail_stage_timings["terminal_payload_final_pack_tokens_ms"] = _elapsed_ms(stage_started)
        diagnostics["retrieval"]["selected_evidence"] = len(result["evidence"])
        return result

    def finalize_payload(payload):
        stage_started = time.perf_counter()
        artifacts = _artifact_payloads(
            payload,
            fingerprint,
            chunks,
            selected,
            index_path,
            lexical_cache_hit,
            bool(task_params.get("include_full_trace")),
        )
        tail_stage_timings["artifact_payloads_ms"] = _elapsed_ms(stage_started)
        return {"payload": payload, "artifact_payloads": artifacts}

    if mode == "evidence":
        payload = terminal_payload("evidence_only", "not_requested")
        finalized = finalize_payload(payload)
        artifacts = finalized["artifact_payloads"]
        if query_stage_cache_key is not None:
            _write_query_stage_cache(
                cache_dir,
                query,
                query_stage_cache_key,
                retrieval_signature,
                ranked,
                selected,
                evidence,
                evidence_budget_trimmed,
                retrieval_quality=retrieval_quality,
                rerank_quality=rerank_quality,
                warnings=payload["warnings"],
                provenance=payload["provenance"],
                released_payload=payload,
                released_artifact_payloads=artifacts,
                repository_state_fingerprint=repository_state_fingerprint,
                build_config_digest=build_config_digest,
                index_fingerprint=fingerprint,
                total_files=int(chunk_cache_stats.get("total_files") or 0),
                chunk_count=int(getattr(chunks, "_chunk_count", 0) or len(getattr(chunks, "_chunk_ids", ()) or ()) or len(chunks or [])),
                budgets=budgets,
            )
        return finalized

    if retrieval_quality != "gpu" or rerank_quality != "gpu" or not evidence:
        warnings.append("ANSWER_REQUIRES_GPU_RETRIEVAL_AND_RERANK")
        # An answer request can still produce useful, cited lexical evidence.
        # Keep that result non-authoritative, but do not turn a recoverable
        # retrieval fallback into a failed broker job. An empty repository is
        # still a genuine failure because there is nothing to return.
        has_fallback_evidence = bool(evidence)
        result_state = "evidence_only" if has_fallback_evidence else "failed"
        synthesis_state = "not_requested" if has_fallback_evidence else "failed"
        payload = terminal_payload(result_state, synthesis_state)
        finalized = finalize_payload(payload)
        artifacts = finalized["artifact_payloads"]
        if query_stage_cache_key is not None and has_fallback_evidence:
            _write_query_stage_cache(
                cache_dir,
                query,
                query_stage_cache_key,
                retrieval_signature,
                ranked,
                selected,
                evidence,
                evidence_budget_trimmed,
                retrieval_quality=retrieval_quality,
                rerank_quality=rerank_quality,
                warnings=payload["warnings"],
                provenance=payload["provenance"],
                released_payload=payload,
                released_artifact_payloads=artifacts,
                repository_state_fingerprint=repository_state_fingerprint,
                build_config_digest=build_config_digest,
                index_fingerprint=fingerprint,
                total_files=int(chunk_cache_stats.get("total_files") or 0),
                chunk_count=int(getattr(chunks, "_chunk_count", 0) or len(getattr(chunks, "_chunk_ids", ()) or ()) or len(chunks or [])),
                budgets=budgets,
            )
        return finalized

    synthesis_budgets = dict(budgets)
    synthesis_base = {
        "mode": mode,
        "query": query,
        "findings": [],
        "evidence": evidence,
        "quality": _quality("answer_ready", "gpu", "gpu", "gpu"),
        "warnings": list(dict.fromkeys(warnings)),
        "provenance": provenance,
    }
    synthesis_budgets["synthesis_output_token_budget"] = max(
        1,
        budgets["final_pack_token_budget"] - released_pack_tokens(synthesis_base) - 128,
    )
    synthesis, synthesis_endpoint = synthesize_with_escalation(
        services,
        query,
        evidence,
        synthesis_budgets,
        attempts,
        client_factory,
        health_interval_seconds,
        service_requester,
        failure_reporter,
    )
    endpoint_context_limit = int((synthesis_endpoint or {}).get("context_limit_tokens") or 0)
    configured_context_limit = int(budgets.get("synthesis_context_token_budget") or 0)
    effective_context_limit = min(
        value for value in (configured_context_limit, endpoint_context_limit) if value > 0
    ) if configured_context_limit > 0 or endpoint_context_limit > 0 else 0
    diagnostics["runtime"]["configured_synthesis_context_token_budget"] = configured_context_limit
    diagnostics["runtime"]["endpoint_synthesis_context_limit_tokens"] = endpoint_context_limit
    diagnostics["runtime"]["effective_synthesis_context_token_budget"] = effective_context_limit
    if synthesis is None:
        warnings.append("GPU_SYNTHESIS_EXHAUSTED")
        payload = terminal_payload("failed" if mode == "answer" else "evidence_only", "failed")
        return finalize_payload(payload)

    provenance.update(
        {
            "synthesis_tier": str(synthesis_endpoint.get("tier") or ""),
            "synthesis_model_profile": str(synthesis_endpoint.get("model_profile") or ""),
        }
    )
    payload = {
        "mode": mode,
        "query": query,
        "answer": synthesis["answer"],
        "findings": synthesis["findings"],
        "evidence": evidence,
        "quality": _quality("answer_ready", "gpu", "gpu", "gpu"),
        "warnings": list(dict.fromkeys(warnings)),
        "provenance": provenance,
        "retrieval": diagnostics["retrieval"],
        "runtime": diagnostics["runtime"],
    }
    stage_started = time.perf_counter()
    diagnostics["retrieval"]["final_pack_tokens"] = released_pack_tokens(payload)
    tail_stage_timings["answer_final_pack_tokens_ms"] = _elapsed_ms(stage_started)
    if query_stage_cache_key is not None:
        _write_query_stage_cache(
            cache_dir,
            query,
            query_stage_cache_key,
            retrieval_signature,
            ranked,
            selected,
            evidence,
            evidence_budget_trimmed,
            retrieval_quality=retrieval_quality,
            rerank_quality=rerank_quality,
            answer=payload["answer"],
            findings=payload["findings"],
            warnings=payload["warnings"],
            provenance=payload["provenance"],
            runtime_attempts=payload["runtime"]["attempts"],
            synthesis_quality="gpu",
            repository_state_fingerprint=repository_state_fingerprint,
            build_config_digest=build_config_digest,
            index_fingerprint=fingerprint,
            total_files=int(chunk_cache_stats.get("total_files") or 0),
            chunk_count=int(getattr(chunks, "_chunk_count", 0) or len(getattr(chunks, "_chunk_ids", ()) or ()) or len(chunks or [])),
            budgets=budgets,
        )
    return finalize_payload(payload)
