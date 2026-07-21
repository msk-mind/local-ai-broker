# Tests

Test code should be split by scope:

- `unit/`
- `acceptance/`
- `integration/`
- `e2e/`

The first end-to-end path should cover:

- MCP submission
- Slurm-backed execution
- result retrieval
- cache hit on repeated request
- safe-summary release behavior

Current smoke scripts:

For deterministic local reliability validation, run:

```bash
bash scripts/test_reliability.sh
```

The harness uses a canonical temporary root outside the repository and runs
the Python unit suite twice before the CGO-enabled Go packages. Set
`BROKER_TEST_TMP_ROOT` to a writable non-Git directory to override the temp
root. A failed local worker now preserves its output directory and stdout/
stderr log paths in job runtime diagnostics.

- `tests/e2e/smoke_command_mode.sh`: fake-Slurm document-summary control-plane smoke
- `tests/e2e/smoke_inspect_repo_perf_proof.sh`: local-broker inspect_repo performance-proof smoke through the HTTP adapter, covering warm query-stage reuse and partial-dirty delta updates on the command path
- `tests/e2e/run_inspect_repo_perf_proof.sh`: saves a broker-path inspect_repo performance proof bundle under `.broker-live-tests/`, including `summary.json` and `metadata.json` with the expected broker-hint/zero-fingerprint contract
- `tests/e2e/smoke_broker_server_cleanup.sh`: verifies broker test cleanup tears down the full `go run`/`broker-server` process group so repeated perf runs do not accumulate orphan broker processes
- `tests/e2e/run_inspect_repo_gpu_perf_proof.sh`: saves a broker-path inspect_repo performance proof bundle that additionally requires real GPU retrieval/reranking on cold, warm, and partial-dirty runs and fails on lexical fallback, while still emitting `summary.json`, `validation.json`, and `metadata.json` for post-mortem inspection
- `tests/e2e/run_inspect_repo_partial_dirty_benchmark.sh`: measures repeated partial-dirty inspect_repo command-path latency and reports per-iteration `repository_fingerprint_ms` after mutating one tracked file between runs
- `tests/e2e/run_inspect_repo_broker_path_breakdown.sh`: records local broker-path wall time together with `runtime.broker_phase_timings_ms` and worker total time for cold, warm, and repeated-query `inspect_repo` runs so broker overhead can be separated from worker and client costs
- `tests/e2e/run_inspect_repo_worker_microbenchmark.sh`: measures the direct worker hot path for cold, warm-clean, and warm partial-dirty runs without the broker in the loop
- `tests/e2e/run_inspect_repo_worker_staged_microbenchmark.sh`: measures the direct worker staged-dirty hot path so staged fingerprint regressions can be tracked separately from unstaged partial-dirty runs
- `tests/e2e/smoke_inspect_repo_worker_staged_perf.sh`: enforces a conservative staged worker fingerprint/build budget on the direct worker staged microbenchmark output
- `tests/e2e/smoke_inspect_repo_warm_snapshot_hint.sh`: proves a second warm-daemon request against the same clean repo but a different query reuses snapshot metadata instead of rediscovering files
- `tests/e2e/smoke_inspect_repo_warm_daemon_lexical_perf.sh`: enforces a warm-daemon new-query budget for `ensure_lexical_index_ms` and worker total time after the first request has already primed the repo cache
- `tests/acceptance/inspect_repo/shared_chunk_cache_proof.py`: proves shared file-chunk cache reuse across fresh local cache roots
- `tests/e2e/smoke_rag_llamacpp_runtime.sh`: local-backend RAG smoke with a fake OpenAI-compatible `llama.cpp` endpoint
- `tests/e2e/smoke_rag_llamacpp_unavailable.sh`: local-backend RAG smoke with an unreachable configured `llama.cpp` endpoint
- `tests/e2e/smoke_rag_no_real_backend.sh`: local-backend RAG smoke with no configured live local runtime, asserting `execution_quality=no_real_backend`
- `tests/e2e/run_smoke_suite.sh`: runs the default smoke set and optionally the loopback-binding RAG runtime smoke via `--with-loopback-bind`

Suggested usage:

```bash
python3 -m unittest discover -s tests/unit -p 'test_*.py'
python3 -m unittest discover -s tests/acceptance/inspect_repo -p 'test_*.py'
python3 tests/acceptance/inspect_repo/evaluate.py
bash tests/e2e/run_smoke_suite.sh
bash tests/e2e/run_smoke_suite.sh --with-loopback-bind
/usr/bin/env OLLAMA_SLURM_E2E_LOOPBACK=1 /usr/bin/go test ./tests/e2e -run TestLocalBackendRAGLlamaCPPRuntimeSmoke -count=1
bash tests/e2e/run_inspect_repo_worker_microbenchmark.sh
bash tests/e2e/run_inspect_repo_broker_path_breakdown.sh
bash tests/e2e/run_inspect_repo_worker_staged_microbenchmark.sh
bash tests/e2e/smoke_inspect_repo_worker_staged_perf.sh
bash tests/e2e/smoke_inspect_repo_warm_snapshot_hint.sh
bash tests/e2e/smoke_inspect_repo_warm_daemon_lexical_perf.sh
bash tests/e2e/smoke_broker_server_cleanup.sh
```

## `inspect_repo` Acceptance Gate

`tests/acceptance/inspect_repo` contains the 30-query `repo_inspection_v2` golden suite and a dependency-free evaluator. The default fixture run is CPU-only and enforces Recall@10 `>= 0.90`, MRR `>= 0.75`, citation precision `>= 0.95`, zero missing evidence references, and the MCP/service-above-RAG-worker ordering regression.

To measure the actual worker without GPU access, use its staged CLI adapter. `auto` requests may return lexical evidence-only results; they may not become answer-ready:

```bash
python3 tests/acceptance/inspect_repo/evaluate.py \
  --worker-cli workers/rag-compression/main.py \
  --repo . \
  --timeout-seconds 180
```

See `tests/acceptance/inspect_repo/README.md` for saved-result, generic command-adapter, and single-query runs. Passing the checked-in fixture proves the evaluator is functioning; the worker CLI run is the retrieval quality measurement.

For the real broker command path, the main fast-path acceptance check is:

```bash
bash tests/e2e/smoke_inspect_repo_perf_proof.sh
```

That smoke boots a local broker-server in command mode and asserts the
broker-shaped `inspect_repo` requests use the repository fingerprint hint
instead of forcing worker-side repository fingerprint recomputation.

The CPU in-process proof also enforces a local timing budget for the
partial-dirty repository fingerprint stage, so regressions in the dirty-state
hot path fail before they show up as broader wall-time regressions.

For direct worker-only performance debugging, use:

```bash
bash tests/e2e/run_inspect_repo_worker_microbenchmark.sh
bash tests/e2e/run_inspect_repo_worker_staged_microbenchmark.sh
bash tests/e2e/smoke_inspect_repo_worker_staged_perf.sh
bash tests/e2e/smoke_inspect_repo_warm_snapshot_hint.sh
```

The first script tracks cold, warm-clean, and warm partial-dirty behavior. The
staged microbenchmark isolates staged fingerprint/rebuild costs so staged-path
changes can be measured independently from the unstaged partial-dirty fast
path. The staged perf smoke wraps that benchmark with a conservative budget so
staged regressions can fail automatically in local validation.

Current unit coverage should also include:

- broker HTTP handler behavior
- MCP tool validation and dispatch
- backend adapter behavior
- cache and audit primitives
