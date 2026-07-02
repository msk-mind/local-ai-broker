# Task And Result Schemas

## Purpose

The broker exposes a small task surface with schema-first inputs and outputs. Workers return compact JSON for agent consumption, not free-form prose.

## Common Request Shape

Every task request uses the same outer structure:

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

Core fields:

- `task_type`: logical broker task
- `input_refs`: local inputs by reference
- `task_params`: task-specific parameters
- `constraints`: token, runtime, confidentiality, and priority limits
- `execution_profile`: backend, tier, runtime, and model hints
- `output_schema`: requested result schema

## Common Result Shape

Every task returns a versioned result envelope:

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

Core fields:

- `schema_name`
- `schema_version`
- `payload`
- `evidence_refs`
- `quality_signals`
- `provenance`

## Shared Conventions

Result payloads should be:

- compact
- evidence-backed
- safe for remote consumption by default
- tolerant of additive fields

Typical shared fields:

- `summary`
- `top_findings`
- `warnings`
- `suggested_next_steps`
- `confidence`

## Current Task Set

The implemented or broker-recognized task set is:

- `document_summary`
- `log_analysis`
- `repo_summary`
- `rag_compress`
- `debug_with_local_context`
- `summarize_logs`
- `inspect_repo`
- `propose_patch`

Additional task names may exist in examples or future extension points, but the repository is centered on the set above.

## Key Result Schemas

### `document_summary_v1`

Expected payload shape:

- `summary`
- `key_points`
- `open_questions`
- `sections`
- `source_metadata`

### `log_analysis_v1`

Expected payload shape:

- `summary`
- `top_findings`
- `suspected_failure_points`
- `warnings`
- `suggested_next_steps`

### `repo_summary_v1`

Expected payload shape:

- `summary`
- `key_components`
- `notable_paths`
- `top_findings`
- `warnings`

### `rag_evidence_pack_v1`

Expected payload shape:

- `query`
- `summary`
- `evidence`
- `retrieval`
- `warnings`
- `provenance`

Each evidence item should preserve local references such as:

- file path
- line range
- timestamp range
- commit hash
- artifact ID
- content hash

### `debug_evidence_pack_v1`

Expected payload shape:

- `problem_statement`
- `summary`
- `likely_causes`
- `evidence`
- `suggested_next_steps`

### `log_evidence_pack_v1`

Expected payload shape:

- `summary`
- `timeline`
- `key_events`
- `evidence`
- `warnings`

### `repo_inspection_pack_v1`

Expected payload shape:

- `summary`
- `areas_of_interest`
- `key_files`
- `evidence`
- `warnings`

### `patch_proposal_pack_v1`

Expected payload shape:

- `summary`
- `proposed_changes`
- `candidate_patch`
- `risks`
- `evidence`

## Artifact Conventions

Artifacts are referenced by ID and type rather than inlined in full.

Common artifact types:

- `evidence_pack`
- `retrieval_plan`
- `retrieval_trace`
- `chunk_manifest`
- `validation_report`
- `runtime_diagnostics`
- `excerpt`

## Redaction Rules

The broker may redact or omit fields before returning results to remote callers.

Default posture:

- raw chunk text stays local
- evidence references are preferred over long inline excerpts
- sensitive logs and artifacts are withheld unless explicitly allowed

## Versioning

Schema changes should be:

- additive when possible
- versioned when breaking
- tracked in worker and broker provenance

## Design Rule

A new task should be added only when:

- it has a stable request contract
- it has a stable result schema
- it can enforce local-first release behavior
- it fits the broker job lifecycle without special-casing transport
