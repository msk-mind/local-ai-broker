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
# Keep Go's t.TempDir paths short enough for Unix-domain daemon sockets.  The
# validated parent remains the cleanup/isolation root for broker-owned files.
test_tmp="$(mktemp -d "$test_parent/lab.XXXXXX")"
first_test_tmp="$test_tmp"
trap 'rm -rf -- "$first_test_tmp" "$test_tmp"' EXIT

export TMPDIR="/tmp"
unset BROKER_REPO_INSPECTION_SHARED_CACHE_DIR

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
# Keep package execution serial: the local warm-daemon tests intentionally
# launch short-lived subprocesses and are sensitive to concurrent process
# startup/teardown on shared CI nodes.
CGO_ENABLED=1 go test -p=1 ./broker/pkg/... ./broker/cmd/... ./cmd/...
