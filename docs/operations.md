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

GPU-backed `inspect_repo` additionally requires:

- a cluster-visible GPU service registry
- the GPU service reconciler
- a Slurm service-job entrypoint
- one warm P40 retrieval service and one warm P40 synthesis service
- configured scale-to-zero V100 and A100 fallback profiles

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
- a cluster-visible registry path writable only by the broker and service jobs
- scheduler placement that can allocate one P40, four same-node V100s, and one or four same-node A100s

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

Expected `inspect_repo` placement:

- CPU for discovery, chunking, hashing, FTS5, lexical fallback, and cache bookkeeping only
- one warm P40 for embeddings, FAISS search, and reranking
- one warm P40 for normal synthesis
- four V100s on one node for the first synthesis fallback
- one or four A100s only after recorded P40 and V100 attempts

The request worker never starts a model. The reconciler submits and replaces service jobs; request workers route only to healthy registry leases.

Warm retrieval checks an index fingerprint before search. Repository text is uploaded to the P40 in bounded batches only when that fingerprint is absent; subsequent queries send the fingerprint and query, not the entire corpus.

## GPU Service Tradeoffs

The GPU service model keeps model servers warm and exposes them through leases in the broker registry. It is most useful when requests are frequent enough to amortize service startup and when several workers should share the same model endpoint.

Advantages:

- Lower repeated-request latency: warm P40 services avoid starting a model for every job.
- Better GPU utilization: multiple requests can share an embedding, retrieval, reranking, or synthesis service.
- More predictable routing: the broker sends work only to healthy, unexpired leases and records tier, GPU count, job ID, and failure category.
- Smaller request payloads after cache warmup: retrieval sends an index fingerprint and query instead of re-uploading the full repository.
- Controlled escalation: normal work stays on P40 capacity, while V100/A100 resources are reserved for recorded failures or stronger synthesis requirements.
- Safer separation of concerns: request workers handle orchestration and evidence preparation; model services handle inference and can be restarted independently.

Costs and limitations:

- Operational complexity: the deployment needs Slurm, a reconciler, registry and control-spool permissions, service health checks, and compatible runtime endpoints.
- Startup and queue delays: scale-from-zero V100/A100 services can take substantially longer than a warm P40, especially when the cluster is busy.
- Resource reservation: minimum P40 replicas consume GPU capacity while idle; scale-to-zero tiers trade idle cost for cold-start latency.
- More failure modes: stale leases, expired heartbeats, scheduler priority, endpoint incompatibility, and service startup failures all need monitoring and recovery.
- Shared-state management: registry, index-cache, run-root, and audit-log storage must be durable and visible to the processes that use them.
- Model/runtime coupling: each configured profile must match its runtime API, context limit, quantization, GPU count, and tensor-parallel settings.
- Degraded fallback is not equivalent to inference: when GPU retrieval is unavailable, the broker can return lexical evidence-only output, but it should not be treated as a model-backed answer.

Use the service model when warm latency, shared GPU utilization, and tiered escalation matter. Use local command mode when validating behavior, running infrequent jobs, or minimizing deployment dependencies; use direct worker execution for focused diagnostics and benchmarks. The GPU service does not remove the need to monitor queue delay, lease health, cache hit rate, and degraded-result rates.

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
- operator-supplied model profile per tier
- local runtime endpoint metadata when using live local inference

## GPU Service Configuration

Enable the service control plane and choose its shared state:

- `BROKER_GPU_SERVICE_ENABLED`
- `BROKER_GPU_SERVICE_REGISTRY_PATH`
- `BROKER_GPU_SERVICE_CONTROL_TOKEN` (required secret; no default)
- `BROKER_GPU_SERVICE_CONTROL_REQUEST_DIR` (defaults to `<registry path>.requests`)
- `BROKER_GPU_SERVICE_SCRIPT_PATH`
- `BROKER_GPU_SERVICE_LEASE_DURATION_SECONDS` (default `14400`, or four hours)
- `BROKER_GPU_SERVICE_HEALTH_INTERVAL_SECONDS` (default `15`)
- `BROKER_GPU_SERVICE_HEARTBEAT_TIMEOUT_SECONDS` (default `45`)
- `BROKER_GPU_SERVICE_STARTUP_TIMEOUT_SECONDS` (default `600`)

Relative registry, control-spool, and GPU launcher paths are normalized once against `BROKER_REPO_ROOT_PATH` at broker startup, so the API service, reconciler, and Slurm backend share the same files even when the process starts from another working directory.

Each tier prefix is one of:

- `BROKER_GPU_SERVICE_P40_RETRIEVAL`
- `BROKER_GPU_SERVICE_P40_SYNTHESIS`
- `BROKER_GPU_SERVICE_V100_REASONING`
- `BROKER_GPU_SERVICE_A100_SINGLE`
- `BROKER_GPU_SERVICE_A100_MULTIGPU`

Append each of these required/configurable suffixes to a tier prefix:

- `_PROFILE`
- `_MODEL_PATH`
- `_QUANTIZATION`
- `_CONTEXT_LIMIT_TOKENS`
- `_RUNTIME`
- `_RUNTIME_ARGS_JSON`
- `_MIN_REPLICAS`
- `_MAX_REPLICAS`

When the control plane is enabled, all five profiles are required and have no model-artifact defaults. Startup validation rejects a missing model path, quantization, context limit, runtime, or runtime argument set rather than silently choosing an artifact. Replica bounds are `1..2` for both P40 roles and exactly `0..1` for V100 and A100 profiles.

Existing P40 and A100 partition, GPU type, node list, and constraint settings remain applicable. V100 placement adds:

- `BROKER_SLURM_PARTITION_V100`
- `BROKER_SLURM_GPU_TYPE_V100`
- `BROKER_SLURM_NODELIST_V100`
- `BROKER_SLURM_CONSTRAINT_V100`

Configure the V100 profile for four GPUs on one node and tensor parallelism across all four. Startup validation requires `{gpu_count}` in both four-GPU runtime argument sets, and requires the V100 profile name and model artifact to differ from `p40-synthesis`. Operators remain responsible for validating that the selected V100 model is materially stronger and compatible with the runtime. `a100-single` requests one A100; `a100-multigpu` requests four A100s on one node.

The configured retrieval runtime must expose authenticated `/health`, `/v1/embeddings`, and preferably `/v1/rerank`. The broker GPU launcher provides the retrieval-tier `/v1/indexes/status`, `/v1/indexes/upsert`, and `/v1/search` endpoints itself, and falls back to embedding-similarity reranking when the upstream runtime does not implement `/v1/rerank`. Synthesis runtimes must expose `/health` and the OpenAI-compatible `/v1/chat/completions` operation. The launcher publishes a lease only after upstream health succeeds; declaring a capability for an incompatible runtime causes the reconciler to retire and replace that lease.

Legacy `BROKER_MODEL_PROFILE_P40` and `BROKER_MODEL_PROFILE_A100` have empty defaults. They do not supply model artifacts for these service profiles.

### Registry Records

Each service publishes:

- service ID, tier, role, and lifecycle state
- endpoint and bearer authentication for internal clients
- profile, model, capabilities, and context limit
- GPU type/count and Slurm job ID
- creation time, startup deadline, heartbeat, and lease expiry
- health and failure metadata

The control token authenticates broker/reconciler scale-from-zero demand in the control-request directory; registration tokens authenticate service publication and renewal; endpoint bearer tokens authenticate inference and health calls. Treat all three as secrets.

Do not expose the raw registry or control-request directory through MCP, result artifacts, logs, or metrics. Released diagnostics may contain tier, profile, GPU count, job ID, failure category, and timing, but must redact every credential.

## Observability

Operators should watch:

- job submission success rate
- queue delay
- run completion rate
- worker heartbeat freshness
- cache hit and miss behavior
- audit chain health
- active and starting replicas by service tier
- P40 endpoint health and heartbeat age
- service startup deadline and lease expiry
- V100/A100 queue state and scale-from-zero latency
- synthesis attempts grouped by failure category and escalation reason

Current observable surfaces include:

- `/healthz`
- `/v1/system/audit-health`
- worker `heartbeat.json`
- staged `stdout.log` and `stderr.log`
- `list_local_capabilities` service-tier snapshots
- the operator-only shared GPU service registry

The capability snapshot should report configured minimum/maximum replicas, active/starting replicas, queue state, endpoint health, profile, context limit, GPU type/count, and supported operations for every tier. An endpoint is routable only while its lease and heartbeat are healthy.

## Recovery Model

Broker restarts should preserve enough state to continue serving job status and result fetch from the job store and run directories.

Operational recovery focus:

- reconcile broker state with backend state
- inspect staged run metadata
- determine whether output ingestion completed
- retry only when policy and idempotency rules allow it
- reload registry state, retain healthy unexpired leases, and avoid duplicate warm replicas
- replace expired, stale-heartbeat, failed-startup, or unhealthy P40 services

The reconciler should run on its health interval, not in the request path. V100 and A100 services return to zero after their lease/work completes. P40 services renew four-hour leases while healthy.

## `inspect_repo` Failure And Escalation

Normal synthesis order is fixed:

1. `p40-synthesis`
2. `v100-reasoning` with four GPUs
3. `a100-single` or `a100-multigpu`

V100 availability, queue delay, timeout, or service failure selects one A100. V100 OOM, context overflow, repeated invalid synthesis, or configured model-limit failure selects four A100s. A malformed/unsupported synthesis is retried once on the current tier with validator feedback before escalation.

Every attempt must retain tier, service/Slurm job ID, GPU count, model profile, failure category, and escalation reason. A100 must never appear before recorded P40 and V100 attempts.

If GPU retrieval is unavailable, all modes return lexical `evidence_only` output while the broker replaces the P40 retrieval service. If all synthesis tiers fail after successful GPU retrieval and reranking, `auto` returns evidence-only; `answer` returns `quality.result=failed`, no answer/findings, and the complete attempt history.

## Security Baseline

Minimum operating rules:

- keep raw inputs local by default
- do not expose artifact paths directly to remote clients
- use authenticated callers for non-demo deployments
- treat header-based identity as trusted-gateway-only
- keep audit logging enabled
- restrict registry file permissions and never release service bearer tokens
- restrict the control-request directory and inject the control token from a secret store
- require TLS or a protected cluster network when registry endpoints are not loopback-local

## Practical Checks

Before calling a deployment healthy, verify:

1. `go test ./...` passes
2. `python3 -m unittest discover -s tests/acceptance/inspect_repo -p 'test_*.py'` passes
3. `python3 tests/acceptance/inspect_repo/evaluate.py` passes the 30-query fixture baseline
4. `bash tests/e2e/smoke_inspect_repo_perf_proof.sh` passes and confirms `broker_hint` requests keep `repository_fingerprint_ms == 0`
5. `python3 tests/acceptance/inspect_repo/perf_proof.py --git-init` passes, including the partial-dirty repository fingerprint budget gate
6. `bash tests/e2e/run_smoke_suite.sh` passes
7. broker health endpoint is reachable
8. audit health endpoint is valid
9. both minimum P40 services have healthy endpoints and renewable leases
10. a four-GPU V100 service can start independently of an inspection request
11. simulated availability failures choose one A100, while capacity/context failures choose four
12. `bash tests/e2e/run_inspect_repo_gpu_perf_proof.sh --base-url http://127.0.0.1:8081` passes when GPU services are expected to be live, with GPU retrieval/reranking on cold, warm, and partial-dirty runs
13. a real job can be submitted, completed, and fetched
14. sensitive outputs and registry credentials are redacted or withheld as expected
