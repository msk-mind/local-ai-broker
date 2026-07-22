# Broker Config

This directory is for broker service configuration such as:

- database connection settings
- artifact storage settings
- backend enablement
- observability configuration
- authentication mode defaults
- audit log and maintenance settings
- MCP service identity fallbacks

Representative examples for this area would include:

- local development config
- single-node file-backed config
- first production-like Slurm deployment config

Current examples:

- `cdsi-cluster.example.json`: CDSI cluster profile using a shared GPU partition and typed GPU requests, without unnecessary node pinning
- `cdsi-cluster.env.example`: environment-variable equivalent of the CDSI cluster profile
- `local.example.json`: local command-mode development config
- `slurm-p40-a100.example.json`: GPU-service config template covering warm P40 retrieval/synthesis, four-GPU V100 reasoning, and adaptive A100 fallback
- `slurm-p40-a100.env.example`: environment-variable equivalent for shells or existing automation

The GPU-first mapping is:

- `cpu-rag-indexing` maps to `BROKER_SLURM_PARTITION_CPU`
- all GPU service tiers may share `BROKER_SLURM_PARTITION_GPU`
- `p40-retrieval` and `p40-synthesis` map to `BROKER_SLURM_GPU_TYPE_P40`
- `v100-reasoning` maps to `BROKER_SLURM_GPU_TYPE_V100` with four GPUs
- `a100-single` and `a100-multigpu` map to `BROKER_SLURM_GPU_TYPE_A100`

Every enabled deployment requires an exact profile name, model path,
quantization, context limit, runtime executable, and runtime-argument array.
There are no model defaults. Four-GPU V100 and A100 argument arrays must use
the `{gpu_count}` placeholder; the V100 profile and model artifact must be
distinct from the normal P40 synthesis deployment.

The CDSI profile intentionally leaves `BROKER_SLURM_PARTITION_CPU` unset because that cluster does not expose a separate CPU partition.

Optional tier-locality controls are also supported:

- `BROKER_SLURM_GPU_REQUEST_MODE`
- `BROKER_SLURM_NODELIST_CPU`, `BROKER_SLURM_NODELIST_P40`, `BROKER_SLURM_NODELIST_V100`, `BROKER_SLURM_NODELIST_A100`
- `BROKER_SLURM_CONSTRAINT_CPU`, `BROKER_SLURM_CONSTRAINT_P40`, `BROKER_SLURM_CONSTRAINT_V100`, `BROKER_SLURM_CONSTRAINT_A100`
- `BROKER_MODEL_PROFILE_CPU`, `BROKER_MODEL_PROFILE_P40`, `BROKER_MODEL_PROFILE_A100`
- `BROKER_RUNTIME_LLAMACPP_BASE_URL`, `BROKER_RUNTIME_LLAMACPP_TIMEOUT_SECONDS`
- `BROKER_RUNTIME_VLLM_BASE_URL`, `BROKER_RUNTIME_VLLM_TIMEOUT_SECONDS`
- `BROKER_RUNTIME_SGLANG_BASE_URL`, `BROKER_RUNTIME_SGLANG_TIMEOUT_SECONDS`
- `BROKER_GPU_SERVICE_INDEX_CACHE_DIR`
- `BROKER_GPU_SERVICE_INDEX_TIMEOUT_SECONDS`
- `BROKER_GPU_SERVICE_EMBED_BATCH_ITEMS`
- `BROKER_GPU_SERVICE_EMBED_BATCH_TOKENS`
- `BROKER_GPU_SERVICE_EMBED_SEGMENT_TOKENS`

The retrieval tier now keeps fingerprinted semantic indexes on disk and reloads
them on warm-service restart. Set `BROKER_GPU_SERVICE_INDEX_CACHE_DIR` to a
stable shared path if you want semantic cache reuse across broker restarts. If
unset, the GPU launcher stores semantic indexes next to the configured GPU
service registry.

`BROKER_GPU_SERVICE_INDEX_TIMEOUT_SECONDS` applies only to semantic
`index_status` and `index_upsert` requests. Keep it materially higher than the
ordinary runtime request timeout so large cold repository builds do not fail
while normal search, rerank, and synthesis calls still use tighter budgets.

For cold semantic builds, the retrieval service also supports larger
GPU-embedding batches via `BROKER_GPU_SERVICE_EMBED_BATCH_ITEMS`,
`BROKER_GPU_SERVICE_EMBED_BATCH_TOKENS`, and
`BROKER_GPU_SERVICE_EMBED_SEGMENT_TOKENS`. Increase those only within the
actual prompt and memory limits of your retrieval model runtime.

If a job request includes `execution_profile.nodelist` or `execution_profile.constraint`, that explicit override wins over the tier default.
Legacy non-inspection tasks may still use task-local model settings, but
`inspect_repo` model paths come only from the five required GPU service
deployments.
If a runtime endpoint is configured here, the broker stages it into `execution_plan.json` so workers do not need to discover it from ambient environment variables.
