# RAG Compression

## Purpose

RAG compression is the broker’s main local token-reduction path. Workers should not return raw repositories, logs, or documents to the remote model. They should return compact evidence packs backed by local references.

## Pipeline

The current intended pipeline is:

1. input discovery
2. chunking
3. local indexing
4. retrieval
5. reranking
6. deduplication
7. evidence-preserving compression
8. JSON validation
9. remote synthesis

The remote model receives the final evidence pack, not the raw corpus.

## Input Discovery

Workers operate only on broker-authorized inputs from `input_manifest.json`.

Discovery output should capture:

- normalized path or URI
- content hash
- classification
- language or MIME hint
- commit or artifact identity where available

## Chunking

Chunking is source-aware.

Typical strategies:

- tree-sitter-aware code chunks
- stack-trace- and timestamp-aware log chunks
- heading- or page-aware document chunks
- diff- and history-aware git chunks

Every chunk should retain local source references.

## Retrieval

The broker supports multiple local retrieval styles:

- ripgrep and BM25 for exact text, logs, and code
- tree-sitter-aware symbol retrieval
- embeddings for semantic document retrieval
- stack-trace and path-aware retrieval
- git diff and history-aware retrieval

Workers may combine deterministic and model-assisted retrieval.

## Compression Contract

Compression should preserve evidence, not just produce a summary.

Each evidence item should include enough reference material for local verification, such as:

- path or artifact URI
- line range
- timestamp range
- commit hash
- content hash
- short derived claim

Long raw excerpts should not be the default.

## Evidence Pack Shape

At a minimum, a RAG evidence pack should contain:

- `query`
- `summary`
- `evidence`
- `retrieval`
- `warnings`
- `provenance`

Useful retrieval metadata includes:

- strategies used
- compression backend mode
- budget decisions
- degraded execution flags

## Token Budgets

The broker should enforce budgets at every stage:

- retrieved chunk budget
- per-chunk compression budget
- final evidence-pack budget
- remote model context budget

If a stage overruns budget, the worker should trim deterministically or fail validation before release.

## Scheduling

Current practical tiering:

- CPU for indexing and deterministic retrieval
- P40 for routine local compression
- A100 for harder reasoning, larger context windows, or patch generation

Jobs run as ordinary scheduled jobs. GPUs are not held permanently.

## Cache Layers

RAG compression benefits from caching:

- file hashes
- chunk manifests
- local indexes
- embeddings
- retrieval results
- rerank results
- evidence packs
- local model outputs

Cache keys should be content-aware and policy-aware.

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

## Failure Model

Useful failure categories:

- input discovery failure
- unsupported input type
- indexing failure
- retrieval failure
- local runtime unavailable
- schema validation failure
- policy release denial

Some degraded outcomes may still succeed with warnings when the worker can produce a useful evidence pack without violating policy.

## Design Rules

- raw local corpora stay local by default
- remote outputs should be evidence-backed and compact
- source references must survive compression
- policy checks apply before execution and before release
- degraded local runs should be visible in structured result metadata
