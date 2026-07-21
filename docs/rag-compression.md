# RAG Compression

## Purpose

RAG compression is the broker’s main local token-reduction path. Workers should not return raw repositories, logs, or documents to the remote model. They should return compact evidence packs backed by local references.

`inspect_repo` is the first GPU-service-backed specialization. It uses `repo_inspection_v2` and the quality gates below. `rag_compress`, debugging, log summarization, and patch proposal keep their current behavior until each has its own golden evaluation suite; a successful `inspect_repo` migration must not be inferred for those tools.

## Pipeline

The `inspect_repo` pipeline is:

1. authorized CPU file discovery and syntax-aware chunking
2. repository fingerprinting and incremental index refresh
3. P40 embedding plus FAISS semantic retrieval
4. independent SQLite FTS5 lexical retrieval
5. reciprocal-rank fusion
6. P40 cross-encoder reranking of up to 64 candidates
7. diverse evidence selection (at most 12 chunks)
8. P40 structured synthesis
9. citation, shape, and token-budget validation
10. ordered GPU escalation when necessary

The remote orchestrator receives the released result and evidence pack, not the raw corpus or full retrieval trace.

## Input Discovery

Workers operate only on broker-authorized inputs from `input_manifest.json`.

Discovery output should capture:

- normalized path or URI
- content hash
- classification
- language or MIME hint
- commit or artifact identity where available

For Git repositories, index identity incorporates `HEAD` plus staged, unstaged, and untracked-content hashes. This prevents a clean-HEAD cache hit from hiding working-tree changes. Non-Git inputs use metadata/content fingerprints. CPU performs this bookkeeping; it does not synthesize an answer.

## Chunking

Chunking is source-aware.

Typical strategies:

- tree-sitter-aware code chunks
- stack-trace- and timestamp-aware log chunks
- heading- or page-aware document chunks
- diff- and history-aware git chunks

Every chunk should retain local source references.

Repository chunks additionally carry path, language, symbol, line range, and content hash. Chunk hashes make vector reuse and citation validation independent of mutable line offsets.

## Retrieval

The broker supports multiple local retrieval styles:

- ripgrep and BM25 for exact text, logs, and code
- tree-sitter-aware symbol retrieval
- embeddings for semantic document retrieval
- stack-trace and path-aware retrieval
- git diff and history-aware retrieval

For `inspect_repo`, the P40 retrieval service owns embeddings, FAISS search, and cross-encoder reranking. SQLite FTS5 remains complementary: it is preferred for exact identifiers, quoted phrases, and paths and is the emergency CPU fallback.

The worker first checks the P40 index by repository fingerprint and document count. A warm hit sends only the query and fingerprint. On a miss, it uploads bounded document batches, asks the service to finalize the persistent FAISS index, and then searches by fingerprint. Repository content is therefore not reposted on every query, and the retrieval service—not the CPU request worker—owns answer-ready embedding and vector search.

Semantic and lexical candidates are retrieved independently and fused with reciprocal-rank fusion. The top 64 fused candidates are reranked on the P40. Selection releases at most 12 chunks and normally limits each file to two chunks; an explicitly named file may exceed that diversity limit. If the retrieval service is unavailable, the result is lexical `evidence_only` with a degradation warning while the reconciler replaces the service.

## Evidence And Synthesis Contract

Compression should preserve evidence, not just produce a summary.

Each evidence item should include enough reference material for local verification, such as:

- path or artifact URI
- line range
- timestamp range
- commit hash
- content hash
- short derived claim

Long raw excerpts should not be the default.

Synthesis receives the original query and only the released ranked evidence. It must return strict structured output. Every finding references one or more released evidence IDs; unknown IDs, missing references, malformed output, and token-budget violations fail validation.

An ordinary unsupported or malformed claim gets one retry on the same tier with validation feedback. It does not immediately consume a stronger GPU.

## `repo_inspection_v2`

The schema envelope is `repo_inspection_v2`, version `2.0.0`. Its payload contains:

- `mode`: requested `auto`, `evidence`, or `answer`
- `query`
- optional `answer`
- `findings`
- `evidence`
- `quality`
- `warnings`
- `provenance`
- compact top-level `retrieval` and `runtime` diagnostics (`runtime.attempts` records tier history)

Quality is stage-specific:

- `quality.result`: `answer_ready`, `evidence_only`, or `failed`
- `quality.retrieval`: `gpu`, `lexical_degraded`, or `failed`
- `quality.reranking`: `gpu`, `unavailable`, or `failed`
- `quality.synthesis`: `gpu`, `not_requested`, or `failed`
- `quality.answer_ready`: boolean

An answer-ready result requires GPU retrieval, GPU reranking, and GPU synthesis. Evidence-only output omits `answer`, has `findings: []`, and sets `answer_ready: false`. CPU-only execution therefore cannot return an answer-ready result.

When GPU retrieval or reranking is unavailable, `mode=auto`, `mode=evidence`, and `mode=answer` return non-authoritative lexical `evidence_only` output with no synthesized answer or findings. A GPU-backed retrieval, rerank, and synthesis path is still required for `answer_ready`.

## Token Budgets

`inspect_repo` uses token-explicit budgets:

- `retrieval_token_budget`
- `evidence_token_budget`
- `final_pack_token_budget`
- `synthesis_context_token_budget`

When explicitly set, `final_pack_token_budget` must be at least 2,048 tokens, and an inspection query is limited to 2,048 UTF-8 bytes. These submission bounds ensure the required structured contract and retry diagnostics always fit. If a stage overruns budget, the worker trims deterministically or fails validation before release.

## Scheduling

The service tiers are:

- warm `p40-retrieval` for embeddings, FAISS, and reranking
- warm `p40-synthesis` for normal answers
- scale-to-zero `v100-reasoning` requesting four V100s on one node
- scale-to-zero `a100-single` requesting one A100
- scale-to-zero `a100-multigpu` requesting four A100s on one node

P40 services use renewable four-hour leases. The normal replica range is one to two per role. V100 and A100 profiles have zero minimum and one maximum replica. The service reconciler starts models and publishes authenticated endpoints; inspection request workers only consume healthy leases.

Synthesis always attempts P40 and then four-GPU V100. V100 availability, queue delay, timeout, or service failure selects one A100. OOM, context overflow, repeated invalid synthesis, or configured model limits select four A100s. Every attempt records tier, Slurm job ID, GPU count, profile, failure category, and escalation reason.

## Cache Layers

`inspect_repo` keeps a private, fingerprinted SQLite lexical index under the broker-owned run cache. Normal answer-ready vector indexes live in the P40 retrieval service and are addressed by the repository/chunk-manifest fingerprint plus the configured model profile. Dirty Git content, chunk metadata, or an index-schema upgrade changes that fingerprint. Retrieval-result, rerank-result, evidence-pack, and model-output caches are not currently part of the v2 answer path.

## MCP Surface

RAG-aware broker tools include:

- `rag_compress`
- `debug_with_local_context`
- `summarize_logs`
- `inspect_repo`
- `propose_patch`
- `get_job_status`
- `fetch_result`
- `cancel_job`

These still map to ordinary broker jobs.

The convenience tool still maps to the common broker lifecycle, but its worker calls scheduler-managed services. `list_local_capabilities` exposes per-tier replica, queue, endpoint-health, profile, context-limit, GPU, and supported-operation summaries without releasing bearer credentials.

## Failure Model

Useful failure categories:

- input discovery failure
- unsupported input type
- indexing failure
- retrieval failure
- local runtime unavailable
- schema validation failure
- policy release denial
- service unavailable or startup timeout
- queue delay or lease expiry
- OOM or context overflow
- invalid structured synthesis
- unknown or missing evidence citation

Degraded lexical outcomes may still succeed with warnings as evidence-only. They are non-authoritative leads and cannot be promoted through a fixed confidence value or a deterministic `real_local` flag.

## Golden Evaluation

The checked-in 30-query suite at `tests/acceptance/inspect_repo` runs without GPUs against fixtures or the staged worker CLI. Release gates are:

- Recall@10 at least `0.90`
- MRR at least `0.75`
- citation precision at least `0.95` when answers exist
- zero findings with missing or unknown evidence references
- MCP and service code above the generic RAG worker for the MCP call-chain regression

The fixture run validates the evaluator itself. A staged worker run is the system-under-test measurement and may legitimately fail until retrieval quality reaches the gate.

## Performance Fast Paths

`inspect_repo` now relies on a few explicit fast paths to keep repeated broker
requests cheap:

- identical repeated queries may reuse the query-stage cache and skip chunk
  loading, lexical search, semantic retrieval, and rerank
- fresh local cache roots may reuse shared chunk snapshots and the shared
  lexical working index instead of rebuilding all source chunks
- repeated dirty-repository runs may reuse cached per-path worktree signatures
  for unchanged dirty files while still changing the repository fingerprint
  when unstaged or untracked content changes under the same status shape
- eligible broker-shaped requests inject a repository fingerprint hint so the
  worker can skip local repository fingerprint recomputation entirely

The broker-hint path is part of the acceptance contract, not just an
optimization detail. In command-mode broker proofs it must report
`fingerprint_sources=["broker_hint"]` and
`setup_timings_ms.repository_fingerprint_ms == 0.0`.

## Design Rules

- raw local corpora stay local by default
- remote outputs should be evidence-backed and compact
- source references must survive compression
- policy checks apply before execution and before release
- degraded local runs should be visible in structured result metadata
- no model artifact or quantization setting is hardcoded in worker code
- A100 is never selected before recorded P40 and V100 attempts
- service credentials never appear in released evidence or diagnostics
