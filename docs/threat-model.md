# Threat Model

## Purpose

This document captures the main assets, adversaries, risks, and mitigations that matter for the broker.

## Primary Assets

The system must protect:

- source repositories
- proprietary documents
- build and runtime logs
- PHI or regulated data
- embeddings, indexes, and evidence packs
- candidate patches
- credentials and audit integrity

## Main Actors

Trusted or semi-trusted participants:

- authorized developer
- broker service
- approved worker runtime
- backend scheduler
- remote orchestrator model

The remote orchestrator is useful, but not trusted to receive all local raw data by default.

## Key Trust Boundaries

Important boundaries:

- remote agent to broker
- broker to backend
- backend to worker runtime
- worker to artifact storage
- broker to result consumer

## Main Threats

The highest-value threats are:

- raw source or document leakage to remote systems
- sensitive log leakage in summaries or debug output
- cross-tenant or cross-project cache leakage
- overprivileged worker access
- artifact store overexposure
- forged or unauthorized job access
- policy bypass through worker output
- scheduler drift and orphaned jobs
- secret exposure in logs

## Core Mitigations

The broker should rely on:

- local-first release defaults
- policy checks before execution and release
- typed structured outputs
- constrained worker inputs
- cache scoping and policy-aware cache keys
- authenticated job access
- audit logging
- backend reconciliation

## Assumptions

This model assumes:

- the broker is a trusted control-plane component
- compute nodes can execute approved worker code
- remote frontier models are orchestration helpers, not default raw-data sinks

## Security Properties To Preserve

The system should preserve:

- raw local data does not leave by default
- result release is policy-filtered
- cached data does not bypass access controls
- job ownership and audit trails remain attributable

## Review Triggers

This threat model should be revisited when:

- a new backend is added
- a new worker class can emit richer artifacts
- multi-tenant sharing is expanded
- raw export behavior is broadened
- storage or auth architecture changes materially
