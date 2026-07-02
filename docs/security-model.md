# Security Model

## Objective

The broker should allow remote agents to orchestrate local computation without silently leaking raw sensitive data to remote systems.

## Core Principles

- local-first confidentiality
- least privilege
- explicit export for risky data release
- auditability
- defense in depth

## Trust Boundaries

### Agent Boundary

The remote agent is useful but should not automatically receive all local raw data.

### Broker Boundary

The broker is the main control point for:

- policy
- auth
- release filtering
- audit

### Execution Boundary

Workers and compute nodes execute approved local tasks, but should receive only scoped staged inputs.

### Storage Boundary

Artifacts and metadata may contain sensitive data and require controlled access and retention.

## Data Classification

Useful baseline classes:

- `public`
- `internal`
- `restricted`
- `phi`
- `secret_adjacent`

## Security Controls

The broker should enforce:

- policy before execution
- policy before remote release
- typed structured result filtering
- constrained worker input scope
- authenticated caller identity
- audit logging for sensitive actions

## Output Security

Safe default behavior:

- compact results are preferred over raw output
- evidence references are preferred over long excerpts
- sensitive artifacts remain local-only unless explicitly approved

## Execution Isolation

Workers should run with:

- explicit staged inputs
- explicit output directory
- minimal dependence on ambient host state

## Auth And Secrets

Minimum expectations:

- non-demo deployments authenticate callers
- header-derived identity is only acceptable behind a trusted gateway
- secrets should not be copied into ordinary outputs or logs

## Cache Risks

Cached data must not bypass classification or authorization rules.

Important rule:

- cached reuse cannot widen visibility across users, projects, or policy contexts

## Operational Defaults

Recommended baseline:

- keep raw corpora local
- keep audit logging enabled
- release only compact schema-first outputs by default
- prefer redaction or omission over risky disclosure

## Non-Goals

This project does not try to solve every enterprise security problem itself. It provides the control points needed to keep local AI delegation bounded and auditable.
