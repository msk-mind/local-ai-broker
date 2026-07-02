# Data Model

## Purpose

The broker data model separates user intent, execution state, artifacts, and cacheable intermediates.

## Core Entities

### Job

Externally visible lifecycle record.

Important fields:

- `id`
- `task_type`
- `state`
- `submitted_by`
- `created_at`
- `updated_at`
- `submitted_at`
- `started_at`
- `completed_at`
- `parent_job_id`
- `root_job_id`
- `cache_status`

### JobSpec

Immutable normalized request used for execution and caching.

Important fields:

- `job_id`
- `spec_hash`
- `task_type`
- `input_refs`
- `task_params`
- `constraints`
- `execution_profile`
- `output_schema`

### InputRef

Reference to a local input.

Important fields:

- `type`
- `uri`
- `content_hash`
- `classification`
- `metadata`

### ExecutionPlan

Broker-selected execution details for a run.

Important fields:

- `backend`
- `tier`
- `runtime`
- `model_profile`
- `resource_hints`
- `runtime_connection`
- `policy_notes`

### BackendRun

Backend-specific execution record linked to a broker job.

Important fields:

- `job_id`
- `backend_kind`
- `backend_run_id`
- `backend_state`
- `exit_code`
- `submitted_at`
- `completed_at`

### Result

Structured schema-validated worker output.

Important fields:

- `schema_name`
- `schema_version`
- `payload`
- `evidence_refs`
- `quality_signals`
- `provenance`

### Artifact

Typed output or intermediate object.

Important fields:

- `artifact_id`
- `artifact_type`
- `path`
- `content_hash`
- `classification`

### PolicyDecision

Execution-time or release-time authorization result.

Important fields:

- `scope`
- `decision`
- `reason`
- `redactions`
- `actor`

### AuditEvent

Tamper-evident operational record.

Important fields:

- `timestamp`
- `actor`
- `role`
- `action`
- `outcome`
- `job_id`
- `event_hash`
- `prev_hash`

## RAG-Specific Entities

RAG compression introduces additional typed intermediates.

### InputManifest

Resolved list of authorized local inputs for a run.

### Chunk

Addressable local unit of code, log, or document context with source refs.

Typical source refs:

- path
- line range
- timestamp range
- page range
- commit hash

### LocalIndex

Local retrieval structure such as lexical index, semantic index, or symbol map.

### RetrievalRun

Recorded retrieval, rerank, and deduplication pass for a query.

### EvidencePack

Compressed remote-safe output containing:

- summary
- evidence items
- retrieval metadata
- warnings

## State Model

Broker-visible job states:

- `queued`
- `running`
- `succeeded`
- `failed`
- `cancelled`

Backend-native states are preserved internally but mapped to these public states.

## Cache Model

Cache keys should account for:

- input content hashes
- task type
- relevant task params
- execution-relevant planner choices
- output schema
- policy-sensitive release dimensions

Useful cache layers:

- final job result cache
- chunk and index cache
- retrieval cache
- evidence-pack cache
- local model output cache

## Design Rules

- the job model is backend-agnostic
- results are schema-first
- artifacts are typed and referenceable
- RAG intermediates are first-class metadata even if stored as artifacts
- policy and audit data stay separate from user-facing payloads
