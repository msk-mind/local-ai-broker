# Policy Rules

## Purpose

This document defines the broker’s practical allow, deny, redact, and approval behavior.

## Policy Model

Every job is evaluated twice:

1. before execution
2. before result release

This matters because a task may be safe to run locally but unsafe to expose remotely in raw form.

## Policy Outcomes

The broker should support four basic decisions:

- `allow`
- `deny`
- `allow_with_redaction`
- `allow_with_approval`

## Main Inputs

Policy decisions should consider:

- actor identity
- task type
- input classification
- input location
- requested output schema
- requested backend or runtime
- explicit override flags

## Default Posture

Safe defaults:

- local execution is allowed for approved task types
- raw remote export is denied by default for non-public inputs
- remote clients receive compact evidence packs by default
- chunk text, indexes, embeddings, and broad raw artifacts stay local unless explicitly approved

## Pre-Execution Checks

Before a job runs, the broker should answer:

- may this actor analyze this input?
- is this task type allowed?
- is this backend or runtime allowed for this classification?
- is the requested execution profile acceptable?

## Pre-Release Checks

Before a result is returned remotely, the broker should answer:

- is the result schema releasable?
- should fields be redacted or omitted?
- should artifacts remain local-only?
- is explicit approval required?

## Classification Baseline

Useful minimum classes:

- `public`
- `internal`
- `restricted`
- `phi`
- `secret_adjacent`

These classes should affect release rules, cache visibility, and retention.

## Task-Level Defaults

High-level defaults:

- summaries and evidence packs are usually releasable after policy filtering
- raw search hits and long excerpts are not
- patch proposals may be releasable, but should still pass redaction rules
- embedding and index artifacts are local-only by default

## Redaction Rules

The broker should prefer:

- artifact references over raw inline content
- short screened excerpts over large copied text
- omission over risky release

Redaction should be deterministic enough that clients can reason about missing fields.

## Approval Model

Explicit approval should be required for:

- raw content export
- sensitive artifact release
- broader-than-default disclosure paths

Approved overrides should be auditable.

## Cache Policy

Cached results must still respect policy.

Rules:

- cache reuse must not widen access
- policy checks still run on cache hit
- policy-sensitive differences should affect cache keys when output would differ

## Audit Expectations

Policy-relevant events should be recorded for:

- denied execution
- denied release
- redacted release
- approval-required release
- override use

## Design Rules

- local execution and remote release are separate decisions
- compact evidence-backed release is the default
- raw local corpora should remain local unless explicitly approved
- policy applies equally to fresh and cached results
