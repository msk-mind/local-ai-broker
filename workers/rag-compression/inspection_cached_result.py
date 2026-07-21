from __future__ import annotations

from inspection_hotpath import (
    cached_lexical_fallback_from_context,
    prepare_prefetched_state,
    released_pack_tokens,
    trim_evidence_for_final_pack,
    _artifact_payloads,
)


def try_cached_lexical_fallback_run(
    discovered,
    query,
    *,
    mode="auto",
    constraints=None,
    task_params=None,
    execution_plan=None,
    output_dir=None,
):
    context = prepare_cached_lexical_fallback_context(
        discovered,
        query,
        mode=mode,
        constraints=constraints,
        task_params=task_params,
        execution_plan=execution_plan,
        output_dir=output_dir,
    )
    return cached_lexical_fallback_from_context(context)


def prepare_cached_lexical_fallback_context(
    discovered,
    query,
    *,
    mode="auto",
    constraints=None,
    task_params=None,
    execution_plan=None,
    output_dir=None,
):
    return prepare_prefetched_state(
        discovered,
        query,
        mode=mode,
        constraints=constraints,
        task_params=task_params,
        execution_plan=execution_plan,
        output_dir=output_dir,
    )

