# Documentation

This directory is the reference set for operating and extending `local-ai-broker`.

If you are new to the repo, start with the docs below in this order:

1. [Broker Quickstart](./quickstart.md) for the fastest local validation path
2. [MCP Tools And Broker API](./mcp-tools.md) for the northbound contract
3. [RAG Compression](./rag-compression.md) for the main token-reduction workflow
4. [Architecture](./architecture.md) for the control-plane and worker model
5. [Operations](./operations.md) for runtime and maintenance concerns

## Choose By Goal

Read this if you want to run the broker:

- [Broker Quickstart](./quickstart.md)
- [Operations](./operations.md)
- [Security Model](./security-model.md)

Read this if you want to integrate a client or agent:

- [MCP Tools And Broker API](./mcp-tools.md)
- [Task And Result Schemas](./task-schemas.md)
- [Data Model](./data-model.md)

Read this if you want to understand the main value path:

- [RAG Compression](./rag-compression.md)
- [Architecture](./architecture.md)
- [Cache Strategy](./cache-strategy.md)

Read this if you want to extend the implementation:

- [Backend Interface](./backend-interface.md)
- [Worker Runtime](./worker-runtime.md)
- [Parallel Execution](./parallel-execution.md)
- [Policy Rules](./policy-rules.md)

Read this if you are reviewing trust boundaries:

- [Security Model](./security-model.md)
- [Threat Model](./threat-model.md)
- [Policy Rules](./policy-rules.md)

## Current Implemented Baseline

The current repo includes:

- broker HTTP server
- stdio MCP server
- Slurm backend adapter
- local command backend
- schema-validated worker results
- cache lookup and reuse
- worker heartbeat ingestion
- sensitive result and log release filtering
- tamper-evident audit logging and verification utilities
