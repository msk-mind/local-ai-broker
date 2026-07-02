# MCP Tools And Broker API

## Purpose

The broker exposes a small MCP surface over a standard asynchronous job model. Clients submit local work, poll status, fetch compact results, and cancel when needed.

## Core Principle

The northbound contract is not “start a model server.” It is:

1. submit a local computation task
2. track job state
3. fetch a compact structured result

## HTTP API

Implemented endpoints:

- `GET /healthz`
- `POST /v1/jobs`
- `GET /v1/jobs`
- `GET /v1/jobs/{id}`
- `GET /v1/roots/{root_job_id}`
- `GET /v1/jobs/{id}/result`
- `GET /v1/jobs/{id}/logs`
- `POST /v1/jobs/{id}:cancel`
- `GET /v1/system/audit-health`

RAG-oriented convenience endpoints may exist later, but the standard job endpoints remain the core lifecycle.

## MCP Session Identity

Each MCP session should bind one broker principal.

Broker expectations:

- identity is established at session start
- subsequent tool calls run as that principal
- no implicit broad fallback identity should be assumed

## Core MCP Tools

Primary tools:

- `submit_local_job`
- `submit_parallel_jobs`
- `get_job_status`
- `get_root_job_status`
- `fetch_result`
- `fetch_job_logs`
- `cancel_job`
- `list_local_capabilities`

RAG-focused tools:

- `rag_compress`
- `debug_with_local_context`
- `summarize_logs`
- `inspect_repo`
- `propose_patch`
- `retry_failed_root_shards`
- `release_deferred_root_chunks`

All of these map to normal broker jobs and the same lifecycle endpoints.

## Request Model

The MCP and HTTP surfaces share the same logical request shape:

```json
{
  "task_type": "rag_compress",
  "input_refs": [],
  "task_params": {},
  "constraints": {},
  "execution_profile": {},
  "output_schema": {
    "name": "rag_evidence_pack_v1"
  }
}
```

Important request concepts:

- `task_type`
- `input_refs`
- `task_params`
- `constraints`
- `execution_profile`
- `output_schema`

## Tool Summary

### `submit_local_job`

Generic async submission path for a single broker job.

Use when:

- the client already knows the exact `task_type`
- a task-specific convenience tool is unnecessary

### `submit_parallel_jobs`

Submits related child jobs under one logical root investigation.

Use when:

- work can fan out across shards or subproblems
- the client wants broker-visible root tracking

### `get_job_status`

Returns current job lifecycle state and progress.

Important fields:

- `state`
- `progress`
- `backend_state`
- `execution_quality`
- `retry_recommended`

### `get_root_job_status`

Returns aggregate status for a root investigation with child jobs and reducers.

### `fetch_result`

Returns the broker-filtered release view of the final result.

Important fields:

- `result`
- `degraded_local_execution`
- `retry_recommended`
- `artifacts` when releasable

### `fetch_job_logs`

Returns redacted and bounded worker log content.

This is for debugging, not bulk artifact export.

### `cancel_job`

Requests cancellation of a running or queued job.

### `list_local_capabilities`

Advertises the currently enabled broker features and backend capabilities.

### `rag_compress`

Submits a RAG compression job that returns an evidence-backed compact result.

### `debug_with_local_context`

Submits a local debugging pass over authorized code, logs, or build outputs.

### `summarize_logs`

Submits a log-focused compression pass.

### `inspect_repo`

Submits a repository inspection task for local analysis and evidence extraction.

### `propose_patch`

Submits a patch-oriented local reasoning task that returns candidate changes plus evidence.

## Status And Result Quality Fields

Both status and result responses may expose:

- `runtime_diagnostics`
- `execution_quality`
- `degraded_local_execution`
- `retry_recommended`

Typical `execution_quality` values:

- `real_local`
- `degraded_local`
- `no_real_backend`

## Result Contract

Results are schema-first JSON envelopes:

```json
{
  "schema_name": "rag_evidence_pack_v1",
  "schema_version": "1.0.0",
  "payload": {},
  "evidence_refs": [],
  "quality_signals": {},
  "provenance": {}
}
```

Clients should rely on structured fields rather than expecting large inline raw data.

## Error Contract

Errors should be machine-readable and small.

Useful categories:

- invalid request
- unauthorized
- policy denied
- backend unavailable
- result not ready
- job not found
- release blocked

## Design Rules

- job lifecycle stays uniform across task types
- task-specific tools are convenience wrappers over the same broker model
- inputs should be passed by reference, not inlined, whenever possible
- remote callers should receive compact evidence-backed outputs by default
