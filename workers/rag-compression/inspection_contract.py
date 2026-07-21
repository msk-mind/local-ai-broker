"""Lightweight request-contract helpers for inspect_repo."""

from __future__ import annotations


VALID_MODES = {"auto", "evidence", "answer"}


def validate_request(query, mode):
    query = str(query or "").strip()
    mode = str(mode or "auto").strip().lower()
    if not query:
        raise ValueError("inspect_repo requires a non-empty query")
    if mode not in VALID_MODES:
        raise ValueError("inspect_repo mode must be one of: auto, evidence, answer")
    return query, mode
