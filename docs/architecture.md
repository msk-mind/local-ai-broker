# Local AI Broker Architecture

## Purpose

`local-ai-broker` lets an MCP-capable agent delegate token-heavy local work to on-prem compute while keeping the remote model as the orchestrator.

Default flow:

1. Remote agent submits a broker task.
2. Broker validates request, policy, and budgets.
3. Broker routes ordinary tasks to a backend such as Slurm or local command mode.
4. For `inspect_repo`, the request worker uses healthy scheduler-managed GPU services from the shared registry; it does not start a model.
5. Worker performs retrieval, compression, or analysis locally.
6. Broker returns compact structured output plus evidence references.

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

`inspect_repo` adds a service control plane beside the ordinary job path:

```text
GPU service reconciler ----> Slurm service jobs
        |                       |
        |                       +--> p40-retrieval (warm)
        |                       +--> p40-synthesis (warm)
        |                       +--> v100-reasoning (scale to zero, 4 GPUs)
        |                       +--> a100-single (scale to zero, 1 GPU)
        |                       +--> a100-multigpu (scale to zero, 4 GPUs)
        v
shared authenticated endpoint registry
        |
        +--> healthy leases only --> inspect_repo request worker
```

The registry is cluster-visible state. A record binds a service ID and role to its endpoint, bearer credential, model profile, supported operations, context limit, GPU type and count, Slurm job ID, heartbeat, lease expiry, and health/failure metadata. An authenticated control-request directory carries coalesced scale-from-zero demand to the reconciler. Control, registration, and endpoint credentials remain internal and are redacted from released diagnostics.

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

The `inspect_repo` worker receives only broker-owned registry/control paths in its protected execution plan. It routes to fresh authenticated leases read from the private registry; caller task parameters cannot supply endpoints or registry paths. Missing endpoints degrade retrieval to lexical evidence and never authorize CPU synthesis.

### GPU Service Reconciler

The reconciler is independent of inspection requests. It:

- maintains a minimum of one `p40-retrieval` and one `p40-synthesis` replica
- renews four-hour leases while heartbeats remain healthy
- enforces configurable per-role replica limits (normally one minimum and two maximum for each P40 role)
- replaces services that miss startup, heartbeat, health, or lease deadlines
- recovers valid registry leases after broker restart without duplicating replicas
- leaves V100 and A100 profiles at zero minimum and one maximum replica

An inspection request may select or wait for a service, but model startup remains a scheduler/reconciler responsibility.

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

`inspect_repo` uses the breaking `repo_inspection_v2` contract. A non-empty query is required and `mode` is `auto`, `evidence`, or `answer`. The other RAG-oriented tasks retain their existing schemas until each has an independent golden evaluation suite.

## `inspect_repo` Data Plane

CPU work is deliberately bounded to authorized file discovery, syntax-aware chunking, hashing, SQLite FTS5 exact search, lexical fallback, and cache bookkeeping. Normal semantic work is GPU-backed:

1. Build or refresh content-addressed chunks and the repository fingerprint.
2. Check the `p40-retrieval` fingerprinted index; upload bounded chunk batches only on an index miss, then request FAISS candidates by fingerprint.
3. Query SQLite FTS5 independently for identifiers, phrases, and paths.
4. Fuse semantic and lexical ranks with reciprocal-rank fusion.
5. Rerank the top 64 candidates on `p40-retrieval`.
6. Release at most 12 evidence chunks, normally no more than two per file.
7. Ask `p40-synthesis` for strict structured output grounded in those evidence IDs.
8. Validate citations, shape, and token budgets before release.

If GPU retrieval is unavailable, the worker returns degraded lexical evidence while the reconciler replaces the service. CPU-only processing can produce `evidence_only`; it cannot produce an answer-ready result.

Synthesis escalates in strict order: warm P40, a four-GPU V100 service, then adaptive A100. Availability, queue delay, timeout, or service failure selects `a100-single`; OOM, context overflow, repeated invalid synthesis, or a configured model limit selects `a100-multigpu`. An ordinary unsupported claim first gets one same-tier retry with validator feedback.

## GPU Service Profiles

`inspect_repo` recognizes five first-class service tiers:

| Tier | GPUs | Lifecycle | Supported role |
| --- | ---: | --- | --- |
| `p40-retrieval` | 1 P40 | warm, renewable lease | embeddings, FAISS search, cross-encoder reranking |
| `p40-synthesis` | 1 P40 | warm, renewable lease | structured chat completion |
| `v100-reasoning` | 4 V100 on one node | scale from zero | stronger tensor-parallel synthesis fallback |
| `a100-single` | 1 A100 | scale from zero | availability/service fallback |
| `a100-multigpu` | 4 A100 on one node | scale from zero | capacity/context/validation fallback |

Every enabled profile requires operator-provided model path, quantization, context limit, runtime, and runtime arguments. The broker does not hardcode model artifacts. `cpu-rag-indexing`, `p40-rag-compression`, and `a100-reasoning` remain ordinary-job compatibility tiers for tasks not yet migrated; they are not evidence of GPU-backed `inspect_repo` quality.

## Current Boundary

The current repository is scoped to:

- broker control plane
- worker contract
- MCP and HTTP interfaces
- Slurm and local execution backends
- evidence-preserving local compression
- scheduler-managed GPU inference services and their lease registry

Authorization, classification, audit logging, artifact release restrictions, and the common job lifecycle still apply on both paths. Full traces remain local/optional; the default release is the v2 result, evidence pack, and compact credential-free GPU diagnostics. Hidden trace artifacts cannot be reintroduced through `artifact://`, and an allowed source artifact promotes its classification monotonically into the consuming job before cache lookup and staging.
