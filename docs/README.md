# Documentation

## Scope

This repository contains a working broker implementation under `broker/` plus the reference docs needed to run it, extend it, and understand its contracts.

The implemented baseline currently includes:

- broker HTTP server
- stdio MCP server
- Slurm backend adapter
- schema-validated worker results
- cache lookup and reuse
- worker progress heartbeats
- sensitive result and log release filtering
- tamper-evident audit logging with verification and maintenance tooling

## Core Docs

Read these first:

- [Broker Quickstart](./quickstart.md)
- [MCP Tools And Broker API](./mcp-tools.md)
- [RAG Compression](./rag-compression.md)
- [Architecture](./architecture.md)

## Reference Docs

- [Data Model](./data-model.md)
- [Task And Result Schemas](./task-schemas.md)
- [Backend Interface](./backend-interface.md)
- [Parallel Execution](./parallel-execution.md)
- [Worker Runtime](./worker-runtime.md)
- [Cache Strategy](./cache-strategy.md)
- [Policy Rules](./policy-rules.md)
- [Security Model](./security-model.md)
- [Threat Model](./threat-model.md)
- [Operations](./operations.md)

## Reading Order

For a quick orientation:

1. read [Broker Quickstart](./quickstart.md)
2. read [MCP Tools And Broker API](./mcp-tools.md)
3. read [RAG Compression](./rag-compression.md)
4. read [Architecture](./architecture.md)
5. read [Operations](./operations.md)
