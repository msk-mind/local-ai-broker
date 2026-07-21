#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_DIR="$(mktemp -d /tmp/local-ai-broker-broker-cleanup.XXXXXX)"
source "${SCRIPT_DIR}/smoke_lib.sh"

cleanup() {
  kill_pid_if_running "${BROKER_PID:-}"
  rm -rf "${BASE_DIR}"
}
trap cleanup EXIT

export BROKER_LISTEN_ADDR="$(pick_free_loopback_addr)"
export BROKER_JOB_STORE_PATH="${BASE_DIR}/jobs.json"
export BROKER_RUN_ROOT_PATH="${BASE_DIR}/runs"
export BROKER_REPO_ROOT_PATH="${REPO_ROOT}"
export BROKER_BACKEND="local"
export BROKER_LOCAL_MODE="command"
export BROKER_LOCAL_SCRIPT_PATH="${REPO_ROOT}/deploy/local/broker_worker.sh"
export BROKER_AUDIT_LOG_PATH="${BASE_DIR}/audit.jsonl"
export BROKER_AUDIT_VERIFY_MODE="warn"
export CGO_ENABLED=0

start_broker_server "${REPO_ROOT}"

BROKER_PGID="$(ps -o pgid= -p "${BROKER_PID}" | tr -d '[:space:]')"
if [ -z "${BROKER_PGID}" ]; then
  echo "failed to resolve broker process group" >&2
  exit 1
fi

GROUP_PIDS_BEFORE="$(
  ps -eo pid=,pgid= |
    awk -v pgid="${BROKER_PGID}" '$2 == pgid { print $1 }' |
    tr '\n' ' '
)"
if [ -z "${GROUP_PIDS_BEFORE// }" ]; then
  echo "expected broker process group members before cleanup" >&2
  exit 1
fi

kill_pid_if_running "${BROKER_PID}"

if ps -eo pid=,pgid= | awk -v pgid="${BROKER_PGID}" '$2 == pgid { found=1 } END { exit(found ? 0 : 1) }'
then
  echo "broker process group ${BROKER_PGID} still has live members after cleanup" >&2
  ps -eo pid=,pgid=,args= | awk -v pgid="${BROKER_PGID}" '$2 == pgid { print }' >&2
  exit 1
fi

printf '{"broker_pgid":"%s","group_pids_before":"%s","cleanup_ok":true}\n' "${BROKER_PGID}" "${GROUP_PIDS_BEFORE}"
