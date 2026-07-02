# Local AI Broker

`local-ai-broker` is a broker-first project for delegating token-intensive local AI work to on-prem compute while keeping a remote frontier model as the orchestrator.

Primary flow:

- remote LLM or MCP-capable agent
- broker MCP server or HTTP API
- execution backend such as Slurm or local command mode
- local workers for RAG compression, repository inspection, log analysis, summarization, and patch-oriented tasks
- compact evidence-backed JSON returned to the remote model

Current implementation includes:

- broker HTTP server
- stdio MCP server
- Slurm backend adapter
- local command backend
- RAG compression worker pipeline
- structured results, cache reuse, and audit logging
- Codex profile templates and MCP client examples

Start here:

- [docs/README.md](docs/README.md)
- [docs/quickstart.md](docs/quickstart.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/rag-compression.md](docs/rag-compression.md)
- [docs/mcp-tools.md](docs/mcp-tools.md)

Key directories:

- `broker/` Go control plane and MCP surface
- `workers/` local worker runtimes
- `deploy/` backend execution entrypoints
- `configs/` broker and policy examples
- `examples/` MCP client integration examples
- `tests/` smoke tests and e2e coverage

This repo intentionally excludes the older direct llama.cpp and Ollama launcher workflow that remains in `ollama-slurm`.
