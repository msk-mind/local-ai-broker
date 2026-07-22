# `inspect_repo` Golden Evaluation

This suite fixes the Phase 1 and Phase 4 quality gates for `repo_inspection_v2` in one CPU-runnable evaluator. The 30 queries cover repository retrieval, the reproduced false-`real_local` root cause, the MCP-to-service call chain, GPU service control-plane behavior, escalation, validation, authorization, and recovery.

The evaluator enforces:

- macro Recall@10 of at least `0.90`
- mean reciprocal rank (MRR) of at least `0.75`
- citation precision of at least `0.95` when answers exist
- zero findings with missing or unknown released-evidence references
- MCP and service candidates above the generic RAG worker for the MCP call-chain regression
- `answer_ready` only when retrieval, reranking, and synthesis each report `gpu`
- evidence-only results with no `answer` and no synthesized findings

## CPU Fixture Run

The checked-in compact result snapshots exercise ranking, contract, citation, and regression checks without contacting a scheduler, model endpoint, or GPU:

```bash
python3 tests/acceptance/inspect_repo/evaluate.py
python3 -m unittest discover -s tests/acceptance/inspect_repo -p 'test_*.py'
```

The fixture includes several synthetic answer-ready shapes so citation validation runs on CPU. The remaining records model lexical evidence fallback. The compact adapter expands both to the exact `repo_inspection_v2` contract before scoring.

## CPU Performance Proof

Use the dedicated proof harness to validate the warm and repeated-query fast
paths without a scheduler or GPU:

```bash
python3 tests/acceptance/inspect_repo/perf_proof.py
```

To exercise the real broker path instead of the in-process worker path, pass a
command template that accepts `{repo}`, `{query}`, and `{mode}`:

```bash
python3 tests/acceptance/inspect_repo/perf_proof.py \
  --command 'python3 /path/to/adapter.py --repo {repo} --query {query} --mode {mode}'
```

For broker-server specifically, use the bundled HTTP adapter:

```bash
python3 tests/acceptance/inspect_repo/perf_proof.py \
  --git-init \
  --expect-fingerprint-source broker_hint \
  --command 'python3 tests/acceptance/inspect_repo/broker_perf_adapter.py --base-url http://127.0.0.1:8081 --repo {repo} --query {query} --mode {mode}'
```

Or use the bundled broker smoke wrapper, which boots a local broker-server in
command mode and runs the same proof end-to-end:

```bash
bash tests/e2e/smoke_inspect_repo_perf_proof.sh
```

To save a broker-path proof bundle for later inspection, use:

```bash
bash tests/e2e/run_inspect_repo_perf_proof.sh --base-url http://127.0.0.1:8081
```

That bundle includes `summary.json` plus `metadata.json`, which records the
expected fingerprint source (`broker_hint`) and whether the saved run satisfied
the zero-`repository_fingerprint_ms` contract.

The same values are also provided as `INSPECT_REPO_PERF_*` environment
variables. This mode is intended for broker-server, MCP, or cluster wrappers
that preserve cache state across repeated calls.

To require a real GPU-backed command path instead of lexical fallback, use:

```bash
bash tests/e2e/run_inspect_repo_gpu_perf_proof.sh --base-url http://127.0.0.1:8081
```

That runner fails unless cold and partial-dirty requests produce semantic and
reranked candidates, and unless cold, warm, and partial-dirty runs all report
GPU retrieval/reranking quality together with the broker-hint zero-fingerprint
contract. It still writes `summary.json`, `validation.json`, and
`metadata.json` on failure so a live-cluster miss is diagnosable after the
fact.

To measure repeated partial-dirty command-path performance directly, use:

```bash
bash tests/e2e/run_inspect_repo_partial_dirty_benchmark.sh
```

That runner establishes a clean baseline, then mutates one tracked file between
repeated requests and reports end-to-end latency together with
`setup_timings_ms.repository_fingerprint_ms` for each partial-dirty iteration.

To isolate discovery-path cost directly across cold and partial-dirty
command-path runs, use:

```bash
bash tests/e2e/run_inspect_repo_discovery_benchmark.sh
```

That runner reports `discover_source_files_ms`, `git_dirty_manifest_keys_ms`,
`git_file_signatures_ms`, and `build_syntax_chunks_ms` as min/mean/median/p90
across repeated cold repos and repeated partial-dirty updates.
It also reports `worker_phase_total_ms` from the warm daemon worker and a
derived `broker_control_overhead_ms = wall_ms - worker_phase_total_ms`, which
helps separate worker execution time from broker submission, polling, and
release overhead on command paths.
It now also reports `enqueue_to_claim_ms`, `claim_to_result_write_ms`, and
`result_write_to_client_visible_ms` so command-path latency can be split across
broker enqueue, warm-daemon execution, and broker/client-visible release.

The proof asserts:

- first query is a cache miss
- second identical query is a query-stage cache hit
- second identical query skips chunk loading, lexical index setup, lexical
  search, semantic retrieval, and rerank
- a fresh local cache root can reuse shared chunk snapshots and the shared
  lexical working index without rebuilding source chunks
- a partial-dirty repository update invalidates the query cache while still
  reusing unchanged file chunks
- in CPU in-process proof mode, partial-dirty repository fingerprinting stays
  within the local timing budget instead of regressing into a near-cold full
  fingerprint pass
- when `--git-init --expect-fingerprint-source broker_hint` is used with the
  broker adapter, the broker request path injects the repository fingerprint
  hint instead of forcing the worker to rediscover the repository state locally
- when broker hints are expected, the worker reports
  `setup_timings_ms.repository_fingerprint_ms == 0.0` for cold, warm, and
  partial-dirty command-path runs

On broker command paths that do not expose warm GPU retrieval+rereank, the
same persisted query-stage cache can still be reused in lexical-fallback mode.
That improves repeated-call latency without promoting those results to
`answer_ready`.

This harness is for performance-path invariants, not retrieval-quality scoring.

## Saved Broker Or Worker Results

Pass a JSON file, JSONL file, or directory of JSON files:

```bash
python3 tests/acceptance/inspect_repo/evaluate.py \
  --results /path/to/results.jsonl \
  --verbose
```

Each record must have a golden query `id` and either a `result` containing a v2 schema envelope or a compact `ranked_paths` fixture. Full broker release envelopes and MCP text-content envelopes are unwrapped automatically.

## Staged Worker CLI

The evaluator can invoke the repository worker once per query using the production staged-file interface:

```bash
python3 tests/acceptance/inspect_repo/evaluate.py \
  --worker-cli workers/rag-compression/main.py \
  --repo . \
  --timeout-seconds 180
```

For each query it creates `job_spec.json`, `execution_plan.json`, and `input_manifest.json`, then reads `output-dir/result.json`. It sets `BROKER_GPU_SERVICE_ENABLED=0`, so this mode is safe on a CPU-only host, and excludes the `tests/acceptance` directory so retrieval cannot match the literal golden query. `auto` queries must fall back to evidence-only; they must never be promoted to answer-ready. This command is a real quality gate and can fail Recall@10 or MRR even when the fixture self-test passes.

For another CLI or broker wrapper, use a command template:

```bash
python3 tests/acceptance/inspect_repo/evaluate.py \
  --command 'python3 /path/to/adapter.py --repo {repo} --query {query} --mode {mode}'
```

The placeholders `{id}`, `{query}`, `{mode}`, and `{repo}` are substituted as individual argv tokens. The same values are also provided as `INSPECT_REPO_EVAL_*` environment variables.

Use `--query-id ID` repeatedly to isolate regressions. Updating a golden expected path or threshold should be reviewed as an acceptance-contract change, not treated as ordinary fixture churn.
