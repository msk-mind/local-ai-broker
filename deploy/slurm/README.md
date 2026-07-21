# Slurm Assets

Slurm-specific broker execution assets live here.

Current contents:

- `broker_worker.slurm`: batch template for broker-managed worker execution
- `gpu_service.slurm`: dedicated long-lived GPU model-service launcher; it
  consumes a protected launch spec and runs `workers/gpu-service/main.py`

This directory is intended for scheduler-facing broker artifacts such as:

- worker batch templates
- backend adapter support files
- future queue-class-specific wrappers

GPU service tier mapping:

- `cpu-rag-indexing` should target your CPU-oriented partition
- `p40-retrieval` and `p40-synthesis` request one P40 each
- `v100-reasoning` requests four V100 GPUs on one node
- `a100-single` requests one A100
- `a100-multigpu` requests four A100 GPUs on one node

The broker env example at `configs/broker/slurm-p40-a100.env.example` shows one concrete mapping for that layout. Tier defaults can inject `--gres` or `--gpus`, plus optional `--nodelist` and `--constraint`, which is useful when the P40 lane is a known host set such as `pllimsksparky[1-4]`.

Separation of concerns:

- `broker/pkg/backends/slurm/` owns the Go adapter logic
- `deploy/slurm/` owns scheduler-facing execution templates
- inspection request jobs remain CPU-only; only the service reconciler submits
  `gpu_service.slurm`
- the launcher substitutes `{model_path}`, `{host}`, `{port}`, `{gpu_count}`,
  `{context_limit_tokens}`, `{quantization}`, and `{endpoint_token}` in the
  explicitly configured runtime arguments
