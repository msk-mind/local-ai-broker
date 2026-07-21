#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# A deployment env file may provide GPU-service registry/control-plane
# settings in addition to the ordinary broker settings. Load it before
# applying defaults so profile-specific values remain authoritative. The
# checked-in CDSI live profile is the default for Slurm when present; a
# deployment can override it with BROKER_ENV_FILE.
if [[ -z "${BROKER_ENV_FILE:-}" && "${BROKER_BACKEND:-local}" == "slurm" && -f "${REPO_ROOT}/configs/broker/cdsi-live.env" ]]; then
  BROKER_ENV_FILE="${REPO_ROOT}/configs/broker/cdsi-live.env"
fi
if [[ -n "${BROKER_ENV_FILE:-}" ]]; then
  if [[ ! -f "${BROKER_ENV_FILE}" ]]; then
    echo "broker env file not found: ${BROKER_ENV_FILE}" >&2
    exit 1
  fi
  set -a
  source "${BROKER_ENV_FILE}"
  set +a
fi

export BROKER_JOB_STORE_PATH="${BROKER_JOB_STORE_PATH:-${REPO_ROOT}/.broker/jobs.json}"
export BROKER_RUN_ROOT_PATH="${BROKER_RUN_ROOT_PATH:-${REPO_ROOT}/.broker/runs}"
export BROKER_REPO_ROOT_PATH="${BROKER_REPO_ROOT_PATH:-${REPO_ROOT}}"
export BROKER_BACKEND="${BROKER_BACKEND:-local}"
export BROKER_LOCAL_MODE="${BROKER_LOCAL_MODE:-command}"
export BROKER_LOCAL_SCRIPT_PATH="${BROKER_LOCAL_SCRIPT_PATH:-${REPO_ROOT}/deploy/local/broker_worker.sh}"
export BROKER_SLURM_MODE="${BROKER_SLURM_MODE:-stub}"
export BROKER_SLURM_SUBMIT_CMD="${BROKER_SLURM_SUBMIT_CMD:-sbatch}"
export BROKER_SLURM_STATUS_CMD="${BROKER_SLURM_STATUS_CMD:-sacct}"
export BROKER_SLURM_CANCEL_CMD="${BROKER_SLURM_CANCEL_CMD:-scancel}"
export BROKER_SLURM_SCRIPT_PATH="${BROKER_SLURM_SCRIPT_PATH:-${REPO_ROOT}/deploy/slurm/broker_worker.slurm}"
export BROKER_MCP_ACTOR="${BROKER_MCP_ACTOR:-copilot-cli}"
export BROKER_MCP_ROLE="${BROKER_MCP_ROLE:-user}"

cd "${REPO_ROOT}"

GO_BIN="${GO_BIN:-$(command -v go || true)}"
if [[ -z "${GO_BIN}" ]]; then
  echo "go binary not found in PATH; set GO_BIN to an explicit path" >&2
  exit 127
fi

exec env -u GOROOT GOENV=off \
  CGO_ENABLED="${CGO_ENABLED:-0}" \
  GOCACHE="${GOCACHE:-/tmp/local-ai-broker-gocache}" \
  GOPATH="${GOPATH:-/tmp/local-ai-broker-gopath}" \
  "${GO_BIN}" run ./broker/cmd/broker-mcp
