# Backend Interface

## Purpose

The broker treats execution backends as adapters behind a stable job model. Slurm is implemented now, but the broker contract should not depend on Slurm-specific behavior.

## Broker Contract

The broker hands the backend:

- a normalized job spec
- an execution plan
- staged inputs
- an output directory

The backend returns:

- a durable backend run ID
- backend-native state
- terminal metadata such as exit status and log locations

## Required Backend Operations

Every backend must support:

- `SubmitRun`
  - accept a staged run
  - return backend run ID and accepted metadata
- `GetRun`
  - return current state and useful scheduler metadata
- `CancelRun`
  - request termination
- `FetchRunLogs`
  - expose stdout and stderr sources for broker ingestion or relay
- optional `ListRuns`
  - useful for reconciliation and tests

## Public State Mapping

Backends may expose many native states, but the broker reduces them to stable public states:

- `queued`
- `running`
- `succeeded`
- `failed`
- `cancelled`

Backend-native values stay internal metadata. MCP and HTTP clients should not need scheduler-specific state handling.

## Staging Model

Backends should not assume a shared filesystem beyond what the broker stages explicitly.

The run directory is the execution contract:

- broker writes inputs and plan artifacts there
- backend launches worker against that directory
- worker writes outputs back there
- broker ingests outputs from there

This is why the same worker contract can run under Slurm or local command mode.

## Slurm Implementation

The current Slurm backend is responsible for:

- translating execution profile to `sbatch` arguments
- preserving scheduler IDs and exit metadata
- mapping queue and terminal states back to broker states
- handling cancellation through `scancel`
- supporting one-job-per-run and bounded array fanout where applicable

Important Slurm-facing inputs include:

- partition
- QoS
- nodelist
- constraint
- runtime script path

## Local Backend

The local backend exists for workstation execution and smoke validation.

It preserves the same broker semantics:

- staged run directory
- worker contract
- status polling
- result ingestion

It is not a different product path, only a different adapter.

## Failure Semantics

Backends should classify failures into a small set of broker-meaningful outcomes:

- submission failure
- queue timeout or starvation
- preemption or external cancellation
- worker failure
- result-ingestion failure
- unknown backend drift

Retry policy belongs to the broker, but backend classification should make retry decisions possible.

## Design Rule

If a behavior would force MCP clients or worker code to care about scheduler-specific details, it probably belongs inside the backend adapter instead.
