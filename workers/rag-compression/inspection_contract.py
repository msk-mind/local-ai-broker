"""Lightweight request-contract helpers for inspect_repo."""

from __future__ import annotations


VALID_MODES = {"auto", "evidence", "answer"}
MIN_FINAL_PACK_TOKENS = 2048
MIN_ANSWER_FINAL_PACK_TOKENS = 4096
MAX_RUNTIME_SECONDS = 24 * 60 * 60


def validate_request(query, mode):
    query = str(query or "").strip()
    mode = str(mode or "auto").strip().lower()
    if not query:
        raise ValueError("inspect_repo requires a non-empty query")
    if mode not in VALID_MODES:
        raise ValueError("inspect_repo mode must be one of: auto, evidence, answer")
    return query, mode


def validate_constraints(constraints, mode):
    constraints = constraints or {}
    final_pack = constraints.get("final_pack_token_budget")
    if final_pack is None:
        final_pack = constraints.get("final_evidence_pack_budget")
    if final_pack is not None:
        final_pack = int(final_pack)
        minimum = MIN_ANSWER_FINAL_PACK_TOKENS if mode == "answer" else MIN_FINAL_PACK_TOKENS
        if final_pack < minimum:
            raise ValueError(
                f"inspect_repo {mode} mode final_pack_token_budget must be at least {minimum} tokens when set"
            )
    runtime = constraints.get("max_runtime_seconds")
    if runtime is not None:
        runtime = int(runtime)
        if runtime < 0 or runtime > MAX_RUNTIME_SECONDS:
            raise ValueError(f"max_runtime_seconds must be between 0 and {MAX_RUNTIME_SECONDS} seconds")
