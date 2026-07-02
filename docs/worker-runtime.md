# Worker Runtime

## Purpose

The worker runtime is the execution-side contract between the broker and local compute. It consumes a staged run directory and emits structured outputs and artifacts.

## Worker Input Contract

Each run is driven by a small explicit bundle:

- `job_spec.json`
- `execution_plan.json`
- `input_manifest.json`
- output directory

These files are the source of truth for worker behavior.

### `job_spec.json`

Carries:

- broker job ID
- task type
- task params
- requested output schema
- constraints

### `execution_plan.json`

Carries:

- backend choice
- tier
- selected runtime backend
- selected model profile
- runtime connection metadata

### `input_manifest.json`

Carries:

- resolved input refs
- content hashes
- classifications
- staged local metadata needed by the worker

## Worker Responsibilities

Workers should:

- load the staged bundle
- execute task logic
- emit progress
- write structured result JSON
- write artifact metadata
- record terminal metadata

Workers should not:

- discover arbitrary extra inputs
- bypass the broker release model
- invent their own untracked output contract

## Runtime Lifecycle

Typical flow:

1. validate staged inputs
2. emit initial heartbeat
3. run task logic
4. write result and artifacts
5. write terminal metadata

## Heartbeats

Workers should emit a compact heartbeat file with:

- `state`
- `phase`
- `percent`
- `message`
- `timestamp`
- optional metrics

This allows the broker to surface progress without parsing raw logs.

## Execution Patterns

Common patterns:

- tool-first workers
- model-first workers
- hybrid workers

The broker currently leans toward hybrid workers, where deterministic local tooling narrows inputs before model inference is used.

## Output Contract

Workers should emit:

- `result.json`
- `artifacts.json`
- `run-metadata.json`
- `heartbeat.json`
- `stdout.log`
- `stderr.log`

### `result.json`

Schema-validated result envelope returned through broker APIs.

### `artifacts.json`

Typed manifest of generated artifacts such as:

- evidence packs
- retrieval traces
- chunk manifests
- validation reports
- excerpts
- runtime diagnostics

### `run-metadata.json`

Execution-side metadata such as:

- backend job ID
- node
- start time
- completion time
- terminal status

## Validation

Workers should not declare success until:

- result schema is valid
- required artifacts are present
- terminal metadata is written

## Local Runtime Integration

Workers may use:

- deterministic local tooling
- `llama.cpp`
- `vLLM`
- `SGLang`

Runtime selection should come from the execution plan, not ambient assumptions.

## Isolation Expectations

Workers should run with:

- least-privilege input access
- explicit output directories
- minimal dependence on mutable host state

## Design Rules

- workers are backend-agnostic
- broker owns authorization and release policy
- worker outputs must stay schema-first and typed
- evidence-preserving compression is preferred over broad raw output
