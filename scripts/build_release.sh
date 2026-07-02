#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VERSION="${VERSION:-$(git -C "${REPO_ROOT}" describe --tags --always --dirty 2>/dev/null || echo dev)}"
COMMIT="${COMMIT:-$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)}"
DATE="${DATE:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/dist}"
PLATFORM="${PLATFORM:-linux-amd64}"
GO_BIN="${GO_BIN:-}"

if [ -z "${GO_BIN}" ]; then
  if [ -x /usr/bin/go ]; then
    GO_BIN=/usr/bin/go
  else
    GO_BIN="$(command -v go)"
  fi
fi

mkdir -p "${OUT_DIR}"

BUILD_ROOT="$(mktemp -d "${OUT_DIR}/local-ai-broker-release.XXXXXX")"
trap 'rm -rf "${BUILD_ROOT}"' EXIT

BUNDLE_ROOT="${BUILD_ROOT}/local-ai-broker_${VERSION}_${PLATFORM}"
BIN_DIR="${BUNDLE_ROOT}/bin"
mkdir -p "${BIN_DIR}"

LDFLAGS="-X main.version=${VERSION} -X main.commit=${COMMIT} -X main.date=${DATE}"

build_bin() {
  local output_name="$1"
  local pkg_path="$2"
  env -u GOROOT \
    CGO_ENABLED=0 GOENV=off \
    GOCACHE="${GOCACHE:-/tmp/local-ai-broker-gocache}" \
    GOPATH="${GOPATH:-/tmp/local-ai-broker-gopath}" \
    "${GO_BIN}" build -ldflags "${LDFLAGS}" -o "${BIN_DIR}/${output_name}" "${pkg_path}"
}

cd "${REPO_ROOT}"

build_bin "local-ai-broker" ./cmd/local-ai-broker
build_bin "broker-server" ./broker/cmd/broker-server
build_bin "broker-mcp" ./broker/cmd/broker-mcp
build_bin "broker-cli" ./broker/cmd/broker-cli

cp install.sh "${BUNDLE_ROOT}/"
mkdir -p "${BUNDLE_ROOT}/configs/broker" "${BUNDLE_ROOT}/examples/mcp-clients"
cp configs/broker/local.example.json "${BUNDLE_ROOT}/configs/broker/"
cp configs/broker/slurm-p40-a100.example.json "${BUNDLE_ROOT}/configs/broker/"
cp examples/mcp-clients/install_codex_profiles.sh "${BUNDLE_ROOT}/examples/mcp-clients/"
cp -R examples/mcp-clients/codex-profiles "${BUNDLE_ROOT}/examples/mcp-clients/"
cp README.md "${BUNDLE_ROOT}/"
cp docs/quickstart.md "${BUNDLE_ROOT}/"

TARBALL="${OUT_DIR}/local-ai-broker_${VERSION}_${PLATFORM}.tar.gz"
tar -C "${BUILD_ROOT}" -czf "${TARBALL}" "$(basename "${BUNDLE_ROOT}")"

echo "Built release bundle:"
echo "  ${TARBALL}"
