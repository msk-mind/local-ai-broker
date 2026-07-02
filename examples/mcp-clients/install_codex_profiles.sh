#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
exec env -u GOROOT GOENV=off \
  GOCACHE="${GOCACHE:-/tmp/local-ai-broker-gocache}" \
  GOPATH="${GOPATH:-/tmp/local-ai-broker-gopath}" \
  /usr/bin/go run "${REPO_ROOT}/cmd/local-ai-broker" install codex --all "$@"
