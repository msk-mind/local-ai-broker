# Cache Strategy

## Purpose

Caching reduces repeated local computation and is a core part of the broker’s value. The broker should reuse both final answers and expensive intermediates without weakening policy boundaries.

## Main Cache Layers

Useful layers:

- final job result cache
- intermediate analysis cache
- retrieval and index cache
- evidence-pack cache
- local model output cache

## What Should Be Cached

High-value cacheable assets include:

- file hashes
- chunk manifests
- symbol or lexical indexes
- embeddings
- retrieval results
- rerank results
- evidence packs
- final schema-validated job results

## Cache Key Inputs

Cache keys should account for:

- input content hashes
- task type
- task params that affect output
- execution-relevant planner choices
- output schema
- model or runtime identity when relevant
- policy-sensitive release dimensions

## Reuse Modes

Two main reuse modes matter:

### Exact Hit

The broker can return a previously validated final result for an equivalent request.

### Partial Reuse

The broker can reuse intermediates such as hashes, chunks, indexes, or retrieval outputs while still running later stages again.

Partial reuse is often more valuable than only caching final answers.

## Invalidation

Cache reuse should stop when meaningful inputs change, including:

- content hashes
- task semantics
- schema version
- planner or runtime behavior that changes output materially
- policy-relevant release differences

## Isolation

Caching must not widen access across users, tenants, or projects.

Practical rules:

- restricted results stay namespace-scoped
- policy checks still run before release
- broader-access results should not be silently reused for narrower viewers

## Observability

The broker should surface:

- cache hit vs miss
- exact hit vs partial reuse
- reused layer or artifact class

This helps operators understand whether local work is actually being avoided.

## Design Rules

- content-addressed inputs are the base of cache correctness
- policy boundaries apply to cached data as well as fresh data
- intermediate caches are first-class, not an afterthought
- cache correctness matters more than maximal reuse
