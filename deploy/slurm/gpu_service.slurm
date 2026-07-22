#!/bin/bash
#SBATCH --job-name=broker-gpu-service

set -euo pipefail
umask 077

if [ -z "${BROKER_GPU_SERVICE_SPEC_PATH:-}" ]; then
  echo "BROKER_GPU_SERVICE_SPEC_PATH is required" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
if [ -z "${REPO_ROOT}" ]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
SPEC_PATH="${BROKER_GPU_SERVICE_SPEC_PATH}"
unset BROKER_GPU_SERVICE_SPEC_PATH

exec python3 "${REPO_ROOT}/workers/gpu-service/main.py" \
  --spec "${SPEC_PATH}"
