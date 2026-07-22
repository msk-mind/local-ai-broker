# MCP Integration

## Purpose

These examples show how to connect an MCP-capable client to the broker over stdio without assuming a single client-specific config format.

The broker MCP entrypoint is:

```bash
examples/mcp-clients/run_broker_mcp.sh
```

That wrapper:

- sets sane default broker paths under `.broker/`
- preserves override support through environment variables
- launches `broker/cmd/broker-mcp`
- works around the local `GOROOT` mismatch by unsetting it before `go run`

Important distinction:

- `codex -p local-broker` and `codex -p slurm-broker` use this stdio MCP path directly
- they do not require a separate `local-ai-broker up ...` process to be running
- `local-ai-broker up ...` is only for the broker HTTP server

## Generic Stdio Definition

If your MCP client accepts a stdio server definition with `command`, `args`, and `env`, use the pattern in [generic-stdio-config.json](./generic-stdio-config.json).

Agent-oriented starter templates are also available:

- [Copilot CLI template](./copilot-cli.example.json)
- [Claude Code template](./claude-code.example.json)
- [Codex CLI template](./codex-cli.example.json)
- [Template notes](./client-config-templates.md)
- [Codex profile installer](./install_codex_profiles.sh)

Core values:

- command: `./examples/mcp-clients/run_broker_mcp.sh`
- args: `[]`
- env:
  - `BROKER_SLURM_MODE=stub` for local development
  - `BROKER_SLURM_MODE=command` when you want real Slurm submission

## Exposed Tools

The broker currently exposes these MCP tools:

- `rag_compress`
- `debug_with_local_context`
- `summarize_logs`
- `inspect_repo`
- `propose_patch`
- `submit_local_job`
- `submit_parallel_jobs`
- `get_job_status`
- `get_root_job_status`
- `retry_failed_root_shards`
- `release_deferred_root_chunks`
- `fetch_result`
- `get_retry_recommendation`
- `retry_with_recommended_profile`
- `fetch_job_logs`
- `cancel_job`
- `list_local_capabilities`

For agents that should wait on broker-backed Slurm work instead of submitting and continuing, the submit-style tools also accept:

- `wait_for_result: true`
- optional `max_wait_seconds`
- optional `poll_interval_ms`

With `wait_for_result: true`, the MCP server polls broker status internally and returns the released result envelope directly instead of only the initial `job_id` submission response.

For Codex, this should be the default for non-trivial broker work because the released envelope contains answer-oriented summary fields alongside the underlying evidence.

When the released payload includes `answer_brief`, `findings`, `recommended_next_action`, or `usage_guidance`, the agent should use those fields as the primary synthesis layer and still cite the referenced files when making claims.

## Development Mode

For local or demo use:

```bash
BROKER_SLURM_MODE=stub examples/mcp-clients/run_broker_mcp.sh
```

That is the safest option when you want tool wiring without cluster interaction.

## Slurm Command Mode

For real scheduler-backed use:

```bash
BROKER_SLURM_MODE=command \
BROKER_SLURM_SCRIPT_PATH="$PWD/deploy/slurm/broker_worker.slurm" \
examples/mcp-clients/run_broker_mcp.sh
```

You will also need working `sbatch`, `sacct`, and `scancel` binaries in `PATH`.

## Notes

- This repo intentionally avoids claiming an exact config file format for a specific external MCP client unless that format is verified separately.
- The generic stdio shape here is meant to be adapted into the target client's MCP settings format.
- For direct HTTP demos instead of MCP, use `broker/cmd/broker-cli`.

Current observed client status:

- Codex CLI is verified against the current stdio MCP server, including `list_local_capabilities` and real `rag_compress` submission to Slurm.
- GitHub Copilot CLI still times out after `initialize` in this environment, so the Copilot template should be treated as a starting point rather than a verified integration.

## Codex Profiles

To keep the broker disabled by default and enable it only for selected Codex sessions, install the provided profiles:

```bash
examples/mcp-clients/install_codex_profiles.sh
```

That writes:

- `~/.codex/slurm-broker.config.toml`
- `~/.codex/local-broker.config.toml`

Use them explicitly:

```bash
codex -p slurm-broker
codex -p local-broker
```

The `slurm-broker` profile targets the Slurm-backed P40 tier.

When `configs/broker/cdsi-live.env` is present, the launcher loads it for
Slurm profiles so inspect-repo jobs receive the GPU-service registry and
control-plane settings. If a service lease is stale or its endpoint dies, the
worker can request a replacement service instead of permanently falling back
to lexical retrieval. Restart the MCP process after changing this file or the
profile.
The `local-broker` profile targets the local command backend on the current machine.

When one of these profiles is active, Codex starts its own stdio MCP broker process.
Do not start `local-ai-broker up --slurm` or `local-ai-broker up --local` just to use the Codex profile.
Start `up` only if you also want the HTTP API available separately.

## Broker-First Scope

This repo is focused on broker-mediated MCP and HTTP flows.
Direct Codex-to-`llama.cpp` compatibility helpers are intentionally left in the original mixed-purpose repository.
