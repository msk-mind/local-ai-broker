#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
MODE="local"
BIN_DIR="${HOME}/.local/bin"
CONFIG_OUTPUT=""
WITH_CODEX=0
SKIP_DOCTOR=0

usage() {
  cat <<'EOF'
Usage:
  ./install.sh [--local|--slurm] [--bin-dir PATH] [--config-output PATH] [--with-codex] [--skip-doctor]

Examples:
  ./install.sh
  ./install.sh --with-codex
  ./install.sh --slurm --config-output /tmp/local-ai-broker.json --with-codex
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      MODE="local"
      shift
      ;;
    --slurm)
      MODE="slurm"
      shift
      ;;
    --bin-dir)
      BIN_DIR="$2"
      shift 2
      ;;
    --config-output)
      CONFIG_OUTPUT="$2"
      shift 2
      ;;
    --with-codex)
      WITH_CODEX=1
      shift
      ;;
    --skip-doctor)
      SKIP_DOCTOR=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${CONFIG_OUTPUT}" ]]; then
  if [[ "${MODE}" == "slurm" ]]; then
    CONFIG_OUTPUT="${REPO_ROOT}/configs/broker/generated.cdsi-slurm.json"
  else
    CONFIG_OUTPUT="${REPO_ROOT}/configs/broker/generated.local.json"
  fi
fi

GO_BIN="$(command -v go || true)"
if [[ -z "${GO_BIN}" ]]; then
  echo "missing required executable: go" >&2
  exit 1
fi

pick_free_loopback_addr() {
  python3 - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(f"127.0.0.1:{sock.getsockname()[1]}")
PY
}

export GOCACHE="${GOCACHE:-/tmp/local-ai-broker-gocache}"
export GOPATH="${GOPATH:-/tmp/local-ai-broker-gopath}"
LISTEN_ADDR="$(pick_free_loopback_addr)"

run_cli() {
  (
    cd "${REPO_ROOT}"
    env -u GOROOT GOENV=off \
      CGO_ENABLED=0 \
      GOCACHE="${GOCACHE}" \
      GOPATH="${GOPATH}" \
      "${GO_BIN}" run ./cmd/local-ai-broker "$@"
  )
}

echo "==> installing binaries into ${BIN_DIR}"
run_cli install binaries --bin-dir "${BIN_DIR}"

if [[ "${MODE}" == "slurm" ]]; then
  echo "==> writing CDSI Slurm config at ${CONFIG_OUTPUT}"
  mkdir -p "$(dirname "${CONFIG_OUTPUT}")"
  cp "${REPO_ROOT}/configs/broker/cdsi-cluster.example.json" "${CONFIG_OUTPUT}"
  python3 - "${CONFIG_OUTPUT}" "${LISTEN_ADDR}" <<'PY'
import json
import sys

path, listen_addr = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as fh:
    payload = json.load(fh)
payload["listen_addr"] = listen_addr
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
    fh.write("\n")
PY
else
  echo "==> generating ${MODE} config at ${CONFIG_OUTPUT}"
  run_cli init "--${MODE}" --listen-addr "${LISTEN_ADDR}" --output "${CONFIG_OUTPUT}"
fi

if [[ "${SKIP_DOCTOR}" -eq 0 ]]; then
  echo "==> running doctor"
  run_cli doctor --config "${CONFIG_OUTPUT}"
fi

if [[ "${WITH_CODEX}" -eq 1 ]]; then
  echo "==> installing Codex profiles"
  "${BIN_DIR}/local-ai-broker" install codex --all
fi

cat <<EOF

Install complete.

Add to PATH if needed:
  export PATH="${BIN_DIR}:\$PATH"

Next:
  ${BIN_DIR}/local-ai-broker up --config ${CONFIG_OUTPUT}
  ${BIN_DIR}/local-ai-broker demo --config ${CONFIG_OUTPUT}

Configured listen address:
  ${LISTEN_ADDR}
EOF
