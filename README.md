# Local AI Broker

`local-ai-broker` lets a remote LLM or MCP-capable agent keep the high-level reasoning loop, while offloading token-heavy local work to your own machine or cluster.

Instead of sending a full repo, log bundle, or document set to a remote model, the broker runs local workers that retrieve, compress, and validate evidence first. The remote model gets a compact structured result rather than the raw corpus.

## What This Repo Does

This repository contains the working broker implementation, local workers, and client integration examples for:

- running a broker as an HTTP service or stdio MCP server
- dispatching jobs to either local command mode or Slurm
- executing local analysis tasks such as RAG compression, repo inspection, log analysis, and summarization
- returning evidence-backed JSON results with audit logging, policy filtering, and cache reuse

In short: this is not a generic model-serving repo. It is a control plane for "do the expensive local context work near the data, then send back a smaller answer."

## Why Use It

Use this when you want a remote coding agent to work with local or sensitive context without exporting everything upstream.

Common cases:

- compress a repository into a small evidence pack before remote reasoning
- inspect local logs and only return the relevant failure evidence
- keep large or sensitive corpora on-prem while still using a stronger remote orchestrator
- route heavier local jobs to Slurm instead of running them inline on a laptop

## How It Works

Default flow:

1. A client submits a broker job over MCP or HTTP.
2. The broker validates the request, policies, and budgets.
3. The broker chooses a backend such as local execution or Slurm.
4. A worker retrieves and compresses local context into evidence-backed JSON.
5. The broker returns the compact result to the caller.

The main token-reduction path is `rag_compress`: retrieve locally, compress locally, release a bounded evidence pack.

### Runtime orchestration

Inspection requests are handled as explicit internal stages: setup, repository and cache preparation, retrieval and reranking, synthesis, and result/artifact finalization. Stage boundaries keep cache reuse and diagnostics visible without changing the public result contract.

The standalone document, log-analysis, and repository-summary workers share the common JSON and heartbeat runtime in `workers/worker_runtime.py`. The RAG worker keeps its atomic output writer because its cache and artifact files have stronger write guarantees.

When GPU services are enabled, retrieval and reranking use the P40 retrieval tier. Synthesis can escalate through P40, V100, and either single-GPU or multi-GPU A100 capacity when the response requires more context or recovery from a failed attempt. If GPU retrieval is unavailable, the broker can return explicitly marked lexical evidence rather than presenting degraded results as authoritative answers.

## Quick Start

Fastest local path:

```bash
./install.sh --with-codex
export PATH="$HOME/.local/bin:$PATH"
local-ai-broker demo --config configs/broker/generated.local.json
local-ai-broker up --config configs/broker/generated.local.json
```

What those commands do:

- install the broker binaries
- generate a known-good local config
- run a demo submission to validate the control plane
- start the broker server

Check the installed command surface:

```bash
local-ai-broker version
local-ai-broker doctor
```

Top-level CLI commands:

```text
local-ai-broker demo [--config PATH]
local-ai-broker init [--local|--slurm] [--output PATH]
local-ai-broker doctor [--local|--slurm] [--config PATH]
local-ai-broker install codex [--local|--slurm|--all]
local-ai-broker install binaries [--bin-dir PATH]
local-ai-broker up [--local|--slurm] [--listen-addr ADDR] [--config PATH] [--env-file PATH]
local-ai-broker version
```

## Typical Usage Modes

### 1. Local validation on one machine

Use the generated local config and start the broker directly:

```bash
local-ai-broker up --config configs/broker/generated.local.json
curl -sf http://127.0.0.1:8081/healthz
```

### 2. MCP integration for an agent

Install the example Codex MCP profiles:

```bash
local-ai-broker install codex --all
codex -p local-broker
codex -p slurm-broker
```

You do not need to start `local-ai-broker up` first for these profiles.
`codex -p local-broker` and `codex -p slurm-broker` launch their own stdio MCP broker process through `examples/mcp-clients/run_broker_mcp.sh`.

Use `local-ai-broker up ...` only when you want the HTTP server directly for `curl`, `broker-cli`, or manual API inspection.

See [examples/mcp-clients/README.md](examples/mcp-clients/README.md) for client examples and templates.

### 3. Slurm-backed execution

Initialize or copy a Slurm config, then run the broker against that profile:

```bash
./install.sh --slurm --config-output /tmp/local-ai-broker.json
local-ai-broker up --config /tmp/local-ai-broker.json
```

See [docs/quickstart.md](docs/quickstart.md) and [configs/broker/README.md](configs/broker/README.md) for cluster-specific configuration details.

## What Results Look Like

The northbound contract is job-oriented and schema-first, not "stream me the whole local dataset."

Typical tasks:

- `rag_compress`
- `document_summary`
- `log_analysis`
- `repo_summary`
- `debug_with_local_context`
- `summarize_logs`
- `inspect_repo`
- `propose_patch`

Typical result shape:

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

See [docs/mcp-tools.md](docs/mcp-tools.md) for the MCP and HTTP job lifecycle.

## Development Checks

Run focused worker and service tests during local changes:

```bash
python -m pytest -q tests/unit/test_workers.py tests/unit/test_service_control.py
CGO_ENABLED=0 go test ./broker/pkg/gpuservice ./broker/pkg/mcp ./broker/pkg/service
python -m py_compile workers/worker_runtime.py workers/rag-compression/main.py
git diff --check
```

The full reliability suite is available with `bash scripts/test_reliability.sh`. Inspection cache/index tests exercise repository-state reuse separately because they depend on isolated temporary cache directories and filesystem state.

## Approximate Token Savings

This repo does not currently publish a benchmark suite with one canonical "saves X tokens" number. The savings depend on the corpus, retrieval quality, and the budgets you set.

What the current implementation does support is explicit token budgeting:

- retrieved local context budget
- per-chunk compression budget
- final evidence-pack budget
- remote model context budget

Two concrete numbers already used in this repo:

- the live RAG smoke tests use `retrieved_chunk_budget=16000` and `final_evidence_pack_budget=1200`, which implies a maximum reduction from retrieved context to released pack of about `92.5%` or about `13x` smaller
- service tests use `retrieved_chunk_budget=64000` and `final_evidence_pack_budget=4000`, which implies a maximum reduction of about `93.75%` or about `16x` smaller

Important caveat: those numbers are only from retrieved chunks to final released evidence pack. Compared with the original raw repo or log corpus, the end-to-end reduction can be much larger because the worker never sends most local data upstream at all.

## Repository Layout

- `cmd/local-ai-broker/`: unified bootstrap CLI
- `broker/`: Go broker server, MCP server, CLI, and control-plane packages
- `workers/`: local worker implementations
- `configs/`: example broker and policy configs
- `deploy/`: local and Slurm execution entrypoints
- `examples/`: MCP client examples and profile templates
- `docs/`: architecture, quickstart, contracts, and operations
- `tests/`: smoke tests and e2e coverage

## Start Reading Here

- [docs/quickstart.md](docs/quickstart.md)
- [docs/mcp-tools.md](docs/mcp-tools.md)
- [docs/rag-compression.md](docs/rag-compression.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/README.md](docs/README.md)

## Current Scope

This repo is currently scoped to the broker control plane and worker runtime surface:

- broker HTTP API
- stdio MCP server
- local and Slurm backends
- schema-validated worker results
- cache reuse and audit logging
- evidence-preserving release of compact local results

It is not trying to be a general-purpose hosted inference platform.
