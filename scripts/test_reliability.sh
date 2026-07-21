#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_parent="${BROKER_TEST_TMP_ROOT:-/var/tmp}"
if [[ ! -d "$test_parent" || ! -w "$test_parent" ]]; then
    echo "reliability test root must be a writable directory: $test_parent" >&2
    exit 2
fi
test_parent="$(realpath -e "$test_parent")"
if git -C "$test_parent" rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "reliability test root must not be inside a Git worktree: $test_parent" >&2
    exit 2
fi
test_tmp="$(mktemp -d "$test_parent/local-ai-broker-tests.XXXXXX")"
trap 'rm -rf -- "$test_tmp"' EXIT

export TMPDIR="$test_tmp/tmp"
unset BROKER_REPO_INSPECTION_SHARED_CACHE_DIR
mkdir -p "$TMPDIR"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required" >&2
    exit 2
fi
if ! command -v go >/dev/null 2>&1; then
    echo "go is required" >&2
    exit 2
fi
if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
    echo "a C compiler is required for CGO-backed Go tests" >&2
    exit 2
fi

cd "$repo_root"

python3 -m pytest -q tests/unit
python3 -m pytest -q tests/unit

if command -v gcc >/dev/null 2>&1; then
    export CC="${CC:-gcc}"
fi
CGO_ENABLED=1 go test ./broker/pkg/... ./broker/cmd/... ./cmd/...
