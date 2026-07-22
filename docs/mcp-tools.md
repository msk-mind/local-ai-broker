# MCP Tools And Broker API

## Purpose

The broker exposes a small MCP surface over a standard asynchronous job model. Clients submit local work, poll status, fetch compact results, and cancel when needed.

## Core Principle

The northbound contract is not “start a model server.” It is:

1. submit a local computation task
2. track job state
3. fetch a compact structured result

The broker may maintain scheduler-managed GPU services behind this contract. MCP clients select a task and mode, not an endpoint, model path, or bearer token.

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

`inspect_repo` is stricter than the generic shape:

- `query` is required and must be non-empty
- `mode` is optional and defaults to `auto`; allowed values are `auto`, `evidence`, and `answer`
- the output schema is `repo_inspection_v2`
- token constraints are `retrieval_token_budget`, `evidence_token_budget`, `final_pack_token_budget`, and `synthesis_context_token_budget`; an explicit inspection final-pack budget has a 2,048-token minimum for evidence mode and 4,096-token minimum for answer mode
- `max_runtime_seconds` is an optional worker runtime limit capped at 24 hours; timed-out jobs report `timed_out` and a `runtime_limit` diagnostic

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

For GPU services it includes a sanitized per-tier snapshot with configured limits, active/starting replicas, queue state, endpoint health, profile, context limit, GPU type/count, and supported operations. Endpoint bearer credentials are never part of the MCP response.

### `rag_compress`

Submits a RAG compression job that returns an evidence-backed compact result.

### `debug_with_local_context`

Submits a local debugging pass over authorized code, logs, or build outputs.

### `summarize_logs`

Submits a log-focused compression pass.

### `inspect_repo`

Submits a repository inspection task for local analysis and evidence extraction. Normal semantic retrieval, reranking, and synthesis use warm P40 services. Four-GPU V100 and adaptive one-/four-GPU A100 services are ordered fallbacks. CPU-only execution can return evidence, never an answer-ready result.

### `propose_patch`

Submits a patch-oriented local reasoning task that returns candidate changes plus evidence.

## Status And Result Quality Fields

Generic task status and result responses may expose:

- `runtime_diagnostics`
- `execution_quality`
- `degraded_local_execution`
- `retry_recommended`

Typical `execution_quality` values:

- `real_local`
- `degraded_local`
- `no_real_backend`

Do not use these generic flags to infer `inspect_repo` answer quality. `repo_inspection_v2.payload.quality` reports retrieval, reranking, and synthesis independently, and only the all-GPU combination is answer-ready.

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

For Codex-oriented use, legacy RAG-style payloads may include an agent-ready layer:

- `answer_brief`: short broker-produced findings that can be lifted into a final answer
- `findings`: cited observations or hypotheses with `evidence_refs` and `source_refs`
- `recommended_next_action`: the broker's suggested next move for the agent
- `confidence`: compact trust signal such as `high`, `medium`, or `low`
- `must_cite_evidence`: whether the agent should cite broker evidence when using the result
- `usage_guidance`: whether the result is ready to cite directly or should only be used as a lead

### `repo_inspection_v2` Result

`inspect_repo` returns schema version `2.0.0`:

```json
{
  "schema_name": "repo_inspection_v2",
  "schema_version": "2.0.0",
  "payload": {
    "mode": "auto",
    "query": "Trace an MCP inspect_repo call into the service layer",
    "answer": "The MCP handler normalizes the request before service submission.",
    "findings": [
      {
        "summary": "The task-specific MCP handler dispatches through the shared service.",
        "evidence_refs": ["ev_001"]
      }
    ],
    "evidence": [
      {
        "id": "ev_001",
        "source_refs": [{"path": "broker/pkg/mcp/server.go", "line_start": 1, "line_end": 40}]
      }
    ],
    "quality": {
      "result": "answer_ready",
      "retrieval": "gpu",
      "reranking": "gpu",
      "synthesis": "gpu",
      "answer_ready": true
    },
    "warnings": [],
    "provenance": {},
    "retrieval": {},
    "runtime": {"attempts": []}
  }
}
```

Every evidence item has one or more `source_refs`, and every finding in an answer-ready payload must cite released evidence. Unknown IDs, missing references, malformed synthesis, and budget violations are rejected. The default release includes compact top-level `retrieval` and `runtime` diagnostics; full traces stay local unless explicitly authorized.

An evidence-only payload omits `answer`, contains `findings: []`, and reports:

```json
{
  "result": "evidence_only",
  "retrieval": "lexical_degraded",
  "reranking": "unavailable",
  "synthesis": "not_requested",
  "answer_ready": false
}
```

If GPU retrieval or reranking is unavailable, every mode returns the evidence-only shape. If retrieval and reranking succeed but every synthesis tier fails, `mode=auto` returns evidence-only while `mode=answer` returns `quality.result=failed`, no answer/findings, and the complete P40/V100/A100 attempt history so the caller can distinguish availability, queue/timeout, service, OOM, context, validation, and model-limit failures.

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

## Blocking Tool Calls

`submit_local_job` and the task-specific tools such as `inspect_repo`, `rag_compress`, `debug_with_local_context`, `summarize_logs`, and `propose_patch` can now block until a result is ready.

Optional arguments:

- `wait_for_result`: when `true`, the MCP tool polls broker job state until terminal and then returns the same release envelope as `fetch_result`
- `max_wait_seconds`: optional timeout for the blocking wait; default is 900 seconds
- `poll_interval_ms`: optional polling interval; default is 500 milliseconds

Example:

```json
{
  "name": "inspect_repo",
  "arguments": {
    "input_refs": [{"type":"repo","uri":"file:///workspace/repo"}],
    "query": "find the default model and whether it is tested",
    "mode": "auto",
    "wait_for_result": true,
    "max_wait_seconds": 600
  }
}
```

Use canonical local file URIs for local inputs:

- preferred: `file:///workspace/repo`
- accepted broker-side aliases for local paths may be normalized, but client examples should emit canonical `file://` URIs

For Codex, treat `wait_for_result: true` as the default for non-trivial broker work so the tool returns the released result envelope directly.

## Design Rules

- job lifecycle stays uniform across task types
- task-specific tools are convenience wrappers over the same broker model
- inputs should be passed by reference, not inlined, whenever possible
- remote callers should receive compact evidence-backed outputs by default
- MCP clients never receive or supply GPU service bearer credentials
- A100 attempts are invalid unless the result history records prior P40 and four-GPU V100 attempts
