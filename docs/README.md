# Documentation

## Current State

This repository contains:

- a working broker implementation under `broker/`
- design and planning documents for the broader product direction

The implemented baseline currently includes:

- broker HTTP server
- stdio MCP server
- Slurm backend adapter
- schema-validated worker results
- cache lookup and reuse
- worker progress heartbeats
- sensitive result and log release filtering
- tamper-evident audit logging with verification and maintenance tooling

## Design Docs

These documents describe the target architecture for evolving the current implementation into a general local AI compute broker for MCP-capable agents.

- [Broker Quickstart](./quickstart.md)
- [Architecture](./architecture.md)
- [RAG Compression](./rag-compression.md)
- [MCP Tools And Broker API](./mcp-tools.md)
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
- [Roadmap](./roadmap.md)

## Positioning

The design docs treat the current implementation as the starting point for a broader broker architecture:

- Slurm is the first backend
- local model runtimes are one worker/runtime mechanism
- the primary product becomes the broker control plane and its job/result contracts

## Reading Order

For a quick orientation:

1. read [Broker Quickstart](./quickstart.md)
2. read [Architecture](./architecture.md)
3. read [MCP Tools And Broker API](./mcp-tools.md)
4. read [Operations](./operations.md)
5. read [Roadmap](./roadmap.md)
