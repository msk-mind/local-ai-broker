#!/usr/bin/env bash

wait_for_http_ok() {
  local url="$1"
  local attempts="${2:-50}"
  local sleep_seconds="${3:-0.1}"

  for _ in $(seq 1 "${attempts}"); do
    if curl -sf "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${sleep_seconds}"
  done

  echo "timed out waiting for ${url}" >&2
  return 1
}

pick_free_loopback_addr() {
  python3 - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(f"127.0.0.1:{sock.getsockname()[1]}")
PY
}

kill_pid_if_running() {
  local pid="${1:-}"
  if [ -z "${pid}" ]; then
    return 0
  fi

  local target_pgid=""
  local self_pgid=""
  target_pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]' || true)"
  self_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]' || true)"

  if [ -n "${target_pgid}" ] && [ "${target_pgid}" != "${self_pgid}" ]; then
    kill -- "-${target_pgid}" 2>/dev/null || true
    sleep 0.2
    kill -9 -- "-${target_pgid}" 2>/dev/null || true
  fi

  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    sleep 0.2
    kill -9 "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

start_broker_server() {
  local repo_root="$1"
  local go_bin="${GO_BIN:-$(command -v go)}"
  local cgo_enabled="${CGO_ENABLED:-0}"
  local broker_log_path="${BROKER_LOG_PATH:-}"
  if [ -z "${go_bin}" ]; then
    echo "go not found on PATH" >&2
    return 1
  fi
  if [ -z "${broker_log_path}" ]; then
    broker_log_path="$(mktemp /tmp/local-ai-broker-broker-server.XXXXXX.log)"
    export BROKER_LOG_PATH="${broker_log_path}"
  fi

  setsid env -u GOROOT CGO_ENABLED="${cgo_enabled}" GOCACHE=/tmp/local-ai-broker-gocache GOPATH=/tmp/local-ai-broker-gopath \
    "${go_bin}" run "${repo_root}/broker/cmd/broker-server" >"${broker_log_path}" 2>&1 &
  BROKER_PID=$!

  if ! wait_for_http_ok "http://${BROKER_LISTEN_ADDR}/healthz" 400 0.1; then
    echo "broker startup failed; log follows from ${broker_log_path}" >&2
    tail -n 200 "${broker_log_path}" >&2 || true
    return 1
  fi
}

extract_job_id() {
  python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])'
}

extract_job_state() {
  python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])'
}

wait_for_job_state() {
  local broker_addr="$1"
  local job_id="$2"
  local attempts="${3:-50}"
  local sleep_seconds="${4:-0.1}"

  local job_json=""
  local state=""
  for _ in $(seq 1 "${attempts}"); do
    job_json="$(curl -sf "http://${broker_addr}/v1/jobs/${job_id}")"
    state="$(printf '%s' "${job_json}" | extract_job_state)"
    if [ "${state}" = "succeeded" ]; then
      printf '%s\n' "${job_json}"
      return 0
    fi
    if [ "${state}" = "failed" ]; then
      printf '%s\n' "${job_json}" >&2
      return 1
    fi
    sleep "${sleep_seconds}"
  done

  echo "timed out waiting for job ${job_id}; last_state=${state:-unknown}" >&2
  if [ -n "${job_json}" ]; then
    printf '%s\n' "${job_json}" >&2
  fi
  return 1
}
