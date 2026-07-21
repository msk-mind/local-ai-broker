#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${1:-${REPO_ROOT}/configs/broker/cdsi-live.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "env file not found: ${ENV_FILE}" >&2
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

required_vars=(
  BROKER_GPU_SERVICE_CONTROL_TOKEN
  BROKER_GPU_SERVICE_P40_RETRIEVAL_PROFILE
  BROKER_GPU_SERVICE_P40_RETRIEVAL_MODEL_PATH
  BROKER_GPU_SERVICE_P40_RETRIEVAL_QUANTIZATION
  BROKER_GPU_SERVICE_P40_RETRIEVAL_RUNTIME
  BROKER_GPU_SERVICE_P40_SYNTHESIS_PROFILE
  BROKER_GPU_SERVICE_P40_SYNTHESIS_MODEL_PATH
  BROKER_GPU_SERVICE_P40_SYNTHESIS_QUANTIZATION
  BROKER_GPU_SERVICE_P40_SYNTHESIS_RUNTIME
  BROKER_GPU_SERVICE_V100_REASONING_PROFILE
  BROKER_GPU_SERVICE_V100_REASONING_MODEL_PATH
  BROKER_GPU_SERVICE_V100_REASONING_QUANTIZATION
  BROKER_GPU_SERVICE_V100_REASONING_RUNTIME
  BROKER_GPU_SERVICE_A100_SINGLE_PROFILE
  BROKER_GPU_SERVICE_A100_SINGLE_MODEL_PATH
  BROKER_GPU_SERVICE_A100_SINGLE_QUANTIZATION
  BROKER_GPU_SERVICE_A100_SINGLE_RUNTIME
  BROKER_GPU_SERVICE_A100_MULTIGPU_PROFILE
  BROKER_GPU_SERVICE_A100_MULTIGPU_MODEL_PATH
  BROKER_GPU_SERVICE_A100_MULTIGPU_QUANTIZATION
  BROKER_GPU_SERVICE_A100_MULTIGPU_RUNTIME
)

missing=()
for name in "${required_vars[@]}"; do
  value="${!name:-}"
  if [[ -z "${value}" ]]; then
    missing+=("${name}")
  fi
done

if (( ${#missing[@]} > 0 )); then
  printf 'missing required live Slurm settings in %s:\n' "${ENV_FILE}" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 1
fi

mkdir -p \
  "${BROKER_RUN_ROOT_PATH}" \
  "$(dirname "${BROKER_JOB_STORE_PATH}")" \
  "$(dirname "${BROKER_AUDIT_LOG_PATH}")" \
  "$(dirname "${BROKER_GPU_SERVICE_REGISTRY_PATH}")" \
  "${BROKER_GPU_SERVICE_CONTROL_REQUEST_DIR}" \
  "${BROKER_GPU_SERVICE_INDEX_CACHE_DIR}"

GO_BIN="${GO_BIN:-$(command -v go || true)}"
if [[ -z "${GO_BIN}" ]]; then
  echo "go binary not found in PATH; set GO_BIN to an explicit path" >&2
  exit 127
fi

cd "${REPO_ROOT}"

exec env -u GOROOT GOENV=off \
  CGO_ENABLED="${CGO_ENABLED:-0}" \
  GOCACHE="${GOCACHE:-/tmp/local-ai-broker-gocache}" \
  GOPATH="${GOPATH:-/tmp/local-ai-broker-gopath}" \
  "${GO_BIN}" run ./broker/cmd/broker-server
