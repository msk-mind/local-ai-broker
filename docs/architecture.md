# Local AI Broker Architecture

## Purpose

`local-ai-broker` lets an MCP-capable agent delegate token-heavy local work to on-prem compute while keeping the remote model as the orchestrator.

Default flow:

1. Remote agent submits a broker task.
2. Broker validates request, policy, and budgets.
3. Broker routes work to a backend such as Slurm or local command mode.
4. Worker performs retrieval, compression, or analysis locally.
5. Broker returns compact structured output plus evidence references.

Raw repositories, logs, documents, and other sensitive inputs stay local unless explicitly released.

## System Context

```text
Developer / MCP-capable agent
  |
  v
Broker MCP server or HTTP API
  |
  +--> auth and policy
  +--> job planning
  +--> cache and artifact indexing
  +--> backend adapter
            |
            +--> Slurm
            +--> local command mode
            +--> future backends
                      |
                      v
                 local worker runtime
                      |
                      +--> retrieval tools
                      +--> local model runtime
                      +--> schema-validated result
```

## Core Components

### Broker API Surface

Implemented now:

- HTTP API for job submission, status, result fetch, log fetch, and cancel
- stdio MCP server exposing broker tools

Responsibilities:

- normalize requests
- bind caller identity
- enforce policy and token budgets
- expose stable job lifecycle semantics

### Planner And Policy Layer

Responsibilities:

- map `task_type` to worker behavior
- select backend, tier, and runtime
- keep remote release bounded to compact evidence-backed outputs
- reject or redact disallowed outputs

### Backend Adapter

Implemented now:

- Slurm backend
- local command backend

Responsibilities:

- submit runs
- poll state
- cancel runs
- fetch backend logs and metadata

### Worker Runtime

Workers execute against a staged run directory containing:

- `job_spec.json`
- `execution_plan.json`
- `input_manifest.json`

Workers emit:

- `result.json`
- `artifacts.json`
- `heartbeat.json`
- run metadata and worker logs

### Cache And Artifact Layer

Responsibilities:

- reuse exact job results where safe
- reuse intermediate retrieval and compression artifacts
- persist evidence packs and validation artifacts
- keep cache keys content- and policy-aware

## Current Task Model

The broker currently centers on structured local tasks rather than raw model serving:

- `document_summary`
- `log_analysis`
- `repo_summary`
- `rag_compress`
- `debug_with_local_context`
- `summarize_logs`
- `inspect_repo`
- `propose_patch`

All tasks use the same job lifecycle and return schema-first JSON.

## RAG Compression Position

RAG compression is the main token-reduction path.

Local workers should:

1. discover inputs
2. chunk and index locally
3. retrieve and rerank
4. deduplicate
5. compress into evidence packs with source references
6. validate before release

The remote model should receive evidence packs, not raw corpora.

## Execution Profiles

Current practical tiers:

- `cpu-rag-indexing`
- `p40-rag-compression`
- `a100-reasoning`

Expected behavior:

- CPU does discovery, chunking, hashing, and lexical retrieval
- P40 is the default low-contention local compression tier
- A100 is reserved for harder reasoning or larger local jobs

GPUs are not reserved persistently. Work runs as ordinary scheduled jobs.

## Current Boundary

The current repository is scoped to:

- broker control plane
- worker contract
- MCP and HTTP interfaces
- Slurm and local execution backends
- evidence-preserving local compression

Future backends or runtimes can extend this boundary without changing the northbound broker model.
