# Operations

## Purpose

This document describes the practical operating model for the current broker implementation.

## Current Deployment Shape

The smallest useful deployment is:

- broker service
- local filesystem job store and run root
- audit log
- one backend adapter
- worker runtime on the same machine or reachable cluster nodes

Productionizing from there usually means replacing local files with stronger storage, not changing the broker contract.

## Required Runtime Pieces

At minimum, operators need:

- broker server
- configured backend
- worker entrypoint
- writable run root
- audit log path
- caller authentication mode

For Slurm-backed deployments, operators also need working:

- `sbatch`
- `sacct`
- `scancel`
- cluster-visible worker runtime environment

## Recommended Modes

### Local Validation

Use:

- `BROKER_BACKEND=local`
- `deploy/local/broker_worker.sh`
- `tests/e2e/smoke_command_mode.sh`

This is the shortest path for testing MCP, HTTP, and worker behavior without cluster dependencies.

### Slurm Execution

Use:

- `BROKER_BACKEND=slurm`
- `deploy/slurm/broker_worker.slurm`
- tier-specific partition and QoS settings

Expected placement:

- CPU for indexing and retrieval
- P40 for routine local compression
- A100 only for harder local jobs

## Configuration Priorities

The most important settings are:

- `BROKER_LISTEN_ADDR`
- `BROKER_JOB_STORE_PATH`
- `BROKER_RUN_ROOT_PATH`
- `BROKER_REPO_ROOT_PATH`
- `BROKER_AUDIT_LOG_PATH`
- backend selector and backend commands
- worker script path

For cluster routing, also set:

- partition per tier
- optional nodelist and constraint per tier
- default model profile per tier
- local runtime endpoint metadata when using live local inference

## Observability

Operators should watch:

- job submission success rate
- queue delay
- run completion rate
- worker heartbeat freshness
- cache hit and miss behavior
- audit chain health

Current observable surfaces include:

- `/healthz`
- `/v1/system/audit-health`
- worker `heartbeat.json`
- staged `stdout.log` and `stderr.log`

## Recovery Model

Broker restarts should preserve enough state to continue serving job status and result fetch from the job store and run directories.

Operational recovery focus:

- reconcile broker state with backend state
- inspect staged run metadata
- determine whether output ingestion completed
- retry only when policy and idempotency rules allow it

## Security Baseline

Minimum operating rules:

- keep raw inputs local by default
- do not expose artifact paths directly to remote clients
- use authenticated callers for non-demo deployments
- treat header-based identity as trusted-gateway-only
- keep audit logging enabled

## Practical Checks

Before calling a deployment healthy, verify:

1. `go test ./...` passes
2. `bash tests/e2e/run_smoke_suite.sh` passes
3. broker health endpoint is reachable
4. audit health endpoint is valid
5. a real job can be submitted, completed, and fetched
6. sensitive outputs are redacted or withheld as expected
