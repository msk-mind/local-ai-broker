# Slurm Assets

Slurm-specific broker execution assets live here.

Current contents:

- `broker_worker.slurm`: batch template for broker-managed worker execution

This directory is intended for scheduler-facing broker artifacts such as:

- worker batch templates
- backend adapter support files
- future queue-class-specific wrappers

Broker tier mapping:

- `cpu-rag-indexing` should target your CPU-oriented partition
- `p40-rag-compression` should usually target your shared GPU partition with a typed P40 GPU request
- `a100-reasoning` should usually target that same GPU partition with a typed A100 GPU request

The broker env example at `configs/broker/slurm-p40-a100.env.example` shows one concrete mapping for that layout. Tier defaults can inject `--gres` or `--gpus`, plus optional `--nodelist` and `--constraint`, which is useful when the P40 lane is a known host set such as `pllimsksparky[1-4]`.

Separation of concerns:

- `broker/pkg/backends/slurm/` owns the Go adapter logic
- `deploy/slurm/` owns scheduler-facing execution templates
