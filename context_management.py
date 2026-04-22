"""Build and store round-level context for LLM reasoning."""

from copy import deepcopy
import re
from typing import Any

from config import CONTEXT_EVIDENCE_TOP_K, NESTED_MAX_UNKNOWN_RETRIES_PER_VARIABLE


def _build_result_lookup(search_results: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for idx, row in enumerate(search_results):
        lookup[idx] = row
    return lookup


def _build_query_lookup(search_plan: list[dict[str, Any]]) -> dict[str, list[str]]:
    query_lookup: dict[str, list[str]] = {}
    for item in search_plan:
        sub_claim = str(item.get("sub_claim", "")).strip()
        queries = item.get("queries", [])
        if not sub_claim or not isinstance(queries, list):
            continue
        query_lookup[sub_claim] = [str(q).strip() for q in queries if str(q).strip()]
    return query_lookup


def _get_chunk_text_by_index(row: dict[str, Any], sentence_index: int) -> str:
    chunks = row.get("sentence_chunks", [])
    if not isinstance(chunks, list):
        return ""
    for chunk in chunks:
        idx = int(chunk.get("sentence_index", -1))
        if idx == sentence_index:
            return str(chunk.get("text", "")).strip()
    return ""


def _build_chunk_text(prev_sentence: str, center_sentence: str, next_sentence: str) -> str:
    parts = [prev_sentence.strip(), center_sentence.strip(), next_sentence.strip()]
    return " ".join([p for p in parts if p]).strip()


def _build_subclaim_context(
    sub_claim: str,
    ranked_sentences: list[dict[str, Any]],
    result_lookup: dict[int, dict[str, Any]],
    query_lookup: dict[str, list[str]],
) -> dict[str, Any]:
    top_ranked = ranked_sentences[:CONTEXT_EVIDENCE_TOP_K]
    evidence_chunks: list[dict[str, Any]] = []
    planned_queries = query_lookup.get(sub_claim, [])

    for item in top_ranked:
        result_index = int(item.get("result_index", -1))
        sentence_index = int(item.get("sentence_index", -1))
        row = result_lookup.get(result_index, {})

        prev_sentence = _get_chunk_text_by_index(row, sentence_index - 1)
        next_sentence = _get_chunk_text_by_index(row, sentence_index + 1)
        center_sentence = str(item.get("sentence_text", "")).strip()
        chunk_text = _build_chunk_text(prev_sentence, center_sentence, next_sentence)

        evidence_chunks.append(
            {
                "rank": int(item.get("global_rank", 0)),
                "chunk_text": chunk_text,
                "url": str(item.get("url", "")).strip(),
                "title": str(item.get("title", "")).strip(),
                "query": str(item.get("query", "")).strip(),
            }
        )

    return {
        "sub_claim": sub_claim,
        "planned_queries": planned_queries,
        "evidence_chunks": evidence_chunks,
    }


def _build_subclaim_contexts_from_retrieval(
    search_plan: list[dict[str, Any]],
    search_results: list[dict[str, Any]],
    rerank_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result_lookup = _build_result_lookup(search_results)
    query_lookup = _build_query_lookup(search_plan)

    rerank_lookup: dict[str, list[dict[str, Any]]] = {}
    for item in rerank_results:
        sub_claim = str(item.get("sub_claim", "")).strip()
        ranked_sentences = item.get("ranked_sentences", [])
        if not sub_claim or not isinstance(ranked_sentences, list):
            continue
        rerank_lookup[sub_claim] = ranked_sentences

    ordered_sub_claims: list[str] = []
    for item in search_plan:
        sub_claim = str(item.get("sub_claim", "")).strip()
        if sub_claim and sub_claim not in ordered_sub_claims:
            ordered_sub_claims.append(sub_claim)
    for sub_claim in rerank_lookup:
        if sub_claim not in ordered_sub_claims:
            ordered_sub_claims.append(sub_claim)

    subclaim_contexts: list[dict[str, Any]] = []
    for sub_claim in ordered_sub_claims:
        subclaim_contexts.append(
            _build_subclaim_context(
                sub_claim=sub_claim,
                ranked_sentences=rerank_lookup.get(sub_claim, []),
                result_lookup=result_lookup,
                query_lookup=query_lookup,
            )
        )
    return subclaim_contexts


def build_context_management(
    search_plan: list[dict[str, Any]],
    search_results: list[dict[str, Any]],
    rerank_results: list[dict[str, Any]],
    previous_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    subclaim_contexts = _build_subclaim_contexts_from_retrieval(
        search_plan=search_plan,
        search_results=search_results,
        rerank_results=rerank_results,
    )

    previous_rounds: list[dict[str, Any]] = []
    if isinstance(previous_context, dict):
        rounds = previous_context.get("rounds", [])
        if isinstance(rounds, list):
            previous_rounds = [r for r in rounds if isinstance(r, dict)]

    current_round_index = len(previous_rounds) + 1
    current_round = {
        "round": current_round_index,
        "search_plan": search_plan,
        "search_results": search_results,
        "rerank_results": rerank_results,
        "subclaim_contexts": subclaim_contexts,
        "llm_feedback": [],
        "reasoning_output": {},
        "next_action": "WAIT_REASONING",
    }

    all_rounds = [*previous_rounds, current_round]
    return {
        "rounds": all_rounds,
        "latest_round": current_round_index,
        "overall_next_action": "WAIT_REASONING",
        "reasoning_ready": False,
        "subclaim_contexts": subclaim_contexts,
    }


def _normalize_resolved_variable_values(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, value in raw.items():
        variable_id = str(key).strip()
        variable_value = str(value).strip()
        if not variable_id or not variable_value:
            continue
        normalized[variable_id] = variable_value
    return normalized


def _normalize_nested_attempt_counts(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, value in raw.items():
        variable_id = str(key).strip()
        if not variable_id:
            continue
        try:
            count = int(value)
        except Exception:
            count = 0
        normalized[variable_id] = max(0, count)
    return normalized


def _normalize_failed_nested_variables(raw: Any) -> set[str]:
    if not isinstance(raw, list):
        return set()
    return {str(item).strip() for item in raw if str(item).strip()}


def _render_query_with_resolved_values(query_template: str, resolved_values: dict[str, str]) -> str:
    if not query_template:
        return ""

    rendered = query_template

    def _replace_bracket(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        return resolved_values.get(key, match.group(0))

    def _replace_brace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        return resolved_values.get(key, match.group(0))

    rendered = re.sub(r"\[([A-Za-z0-9_]+)\]", _replace_bracket, rendered)
    rendered = re.sub(r"\{([A-Za-z0-9_]+)\}", _replace_brace, rendered)
    return re.sub(r"\s+", " ", rendered).strip()


def _render_nested_plan_queries(
    nested_plan: dict[str, Any],
    resolved_values: dict[str, str],
) -> dict[str, Any]:
    if not isinstance(nested_plan, dict):
        return {}
    rendered_plan = deepcopy(nested_plan)
    steps = rendered_plan.get("steps", [])
    if not isinstance(steps, list):
        return rendered_plan

    for step in steps:
        if not isinstance(step, dict):
            continue
        template = str(step.get("query_hint_template", step.get("query_hint", ""))).strip()
        step["query_hint_template"] = template
        step["query_hint"] = _render_query_with_resolved_values(template, resolved_values)
    return rendered_plan


def _all_nested_steps_resolved(
    nested_plan: dict[str, Any],
    resolved_values: dict[str, str],
) -> bool:
    steps = nested_plan.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return False

    step_ids: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        variable_id = str(step.get("variable_id", "")).strip()
        if variable_id:
            step_ids.append(variable_id)
    if not step_ids:
        return False
    return all(variable_id in resolved_values for variable_id in step_ids)


def _infer_nested_blocked_reason(
    nested_plan: dict[str, Any],
    resolved_values: dict[str, str],
    failed_variables: set[str],
) -> str:
    steps = nested_plan.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return "no_nested_steps"

    unresolved_step_ids: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        variable_id = str(step.get("variable_id", "")).strip()
        if not variable_id:
            continue
        if variable_id in resolved_values:
            continue
        unresolved_step_ids.append(variable_id)

    if not unresolved_step_ids:
        return "all_steps_resolved"

    if all(variable_id in failed_variables for variable_id in unresolved_step_ids):
        return "unknown_retry_exceeded"

    return "dependency_not_resolved"


def _select_next_nested_variable(
    nested_plan: dict[str, Any],
    resolved_values: dict[str, str],
    failed_variables: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(nested_plan, dict):
        return {}
    steps = nested_plan.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return {}

    failed = failed_variables if isinstance(failed_variables, set) else set()
    step_descriptions: dict[str, str] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        sid = str(step.get("variable_id", "")).strip()
        sdesc = str(step.get("description", "")).strip()
        if sid and sdesc:
            step_descriptions[sid] = sdesc

    # Pass 1: strict dependency execution.
    for step in steps:
        if not isinstance(step, dict):
            continue
        variable_id = str(step.get("variable_id", "")).strip()
        description = str(step.get("description", "")).strip()
        depends_on_raw = step.get("depends_on", [])
        depends_on = (
            [str(v).strip() for v in depends_on_raw if str(v).strip()]
            if isinstance(depends_on_raw, list)
            else []
        )
        query_hint_template = str(
            step.get("query_hint_template", step.get("query_hint", ""))
        ).strip()
        resolved_query_hint = _render_query_with_resolved_values(
            query_hint_template, resolved_values
        )

        if not variable_id or not description:
            continue
        if variable_id in resolved_values:
            continue
        if variable_id in failed:
            continue
        if any(dep not in resolved_values for dep in depends_on):
            continue

        return {
            "variable_id": variable_id,
            "variable_description": description,
            "depends_on": depends_on,
            "query_hint_template": query_hint_template,
            "query_hint": resolved_query_hint,
            "recovery_mode": False,
            "unresolved_depends_on": [],
            "unresolved_dependency_descriptions": [],
        }

    # Pass 2: recovery execution. Allow downstream variable when unresolved dependencies
    # are already marked as failed.
    for step in steps:
        if not isinstance(step, dict):
            continue
        variable_id = str(step.get("variable_id", "")).strip()
        description = str(step.get("description", "")).strip()
        depends_on_raw = step.get("depends_on", [])
        depends_on = (
            [str(v).strip() for v in depends_on_raw if str(v).strip()]
            if isinstance(depends_on_raw, list)
            else []
        )
        query_hint_template = str(
            step.get("query_hint_template", step.get("query_hint", ""))
        ).strip()
        resolved_query_hint = _render_query_with_resolved_values(
            query_hint_template, resolved_values
        )
        unresolved_depends_on = [dep for dep in depends_on if dep not in resolved_values]

        if not variable_id or not description:
            continue
        if variable_id in resolved_values:
            continue
        if variable_id in failed:
            continue
        if not unresolved_depends_on:
            continue
        if any(dep not in failed for dep in unresolved_depends_on):
            continue

        unresolved_dependency_descriptions = [
            step_descriptions.get(dep, dep) for dep in unresolved_depends_on
        ]
        return {
            "variable_id": variable_id,
            "variable_description": description,
            "depends_on": depends_on,
            "query_hint_template": query_hint_template,
            "query_hint": resolved_query_hint,
            "recovery_mode": True,
            "unresolved_depends_on": unresolved_depends_on,
            "unresolved_dependency_descriptions": unresolved_dependency_descriptions,
        }

    return {}


def build_nested_context_management(
    nested_plan: dict[str, Any],
    previous_context: dict[str, Any] | None = None,
    search_plan: list[dict[str, Any]] | None = None,
    search_results: list[dict[str, Any]] | None = None,
    rerank_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    previous_rounds: list[dict[str, Any]] = []
    if isinstance(previous_context, dict):
        rounds = previous_context.get("rounds", [])
        if isinstance(rounds, list):
            previous_rounds = [r for r in rounds if isinstance(r, dict)]

    nested_plan_data: dict[str, Any] = nested_plan if isinstance(nested_plan, dict) else {}
    if not nested_plan_data and isinstance(previous_context, dict):
        nested_plan_prev = previous_context.get("nested_plan", {})
        if isinstance(nested_plan_prev, dict):
            nested_plan_data = nested_plan_prev

    resolved_values: dict[str, str] = {}
    if isinstance(previous_context, dict):
        resolved_values = _normalize_resolved_variable_values(
            previous_context.get("resolved_variable_values", {})
        )
    nested_attempt_counts: dict[str, int] = {}
    failed_nested_variables: set[str] = set()
    if isinstance(previous_context, dict):
        nested_attempt_counts = _normalize_nested_attempt_counts(
            previous_context.get("nested_attempt_counts", {})
        )
        failed_nested_variables = _normalize_failed_nested_variables(
            previous_context.get("failed_nested_variables", [])
        )

    nested_plan_data = _render_nested_plan_queries(nested_plan_data, resolved_values)
    current_nested_variable = _select_next_nested_variable(
        nested_plan_data,
        resolved_values,
        failed_variables=failed_nested_variables,
    )
    all_steps_resolved = _all_nested_steps_resolved(nested_plan_data, resolved_values)
    resolved_pairs = [
        f"variable_id: {k}, resolved_value: {v}" for k, v in resolved_values.items()
    ]
    blocked_reason = ""

    plan_data = search_plan if isinstance(search_plan, list) else []
    result_data = search_results if isinstance(search_results, list) else []
    rerank_data = rerank_results if isinstance(rerank_results, list) else []
    retrieval_phase = bool(plan_data or result_data or rerank_data)
    current_round_index = len(previous_rounds) + 1
    if retrieval_phase:
        recovery_mode = bool(current_nested_variable.get("recovery_mode", False))
        subclaim_contexts = _build_subclaim_contexts_from_retrieval(
            search_plan=plan_data,
            search_results=result_data,
            rerank_results=rerank_data,
        )
        current_round = {
            "round": current_round_index,
            "mode": "NESTED_RECOVERY_RETRIEVAL" if recovery_mode else "NESTED_RETRIEVAL",
            "nested_plan": nested_plan_data,
            "current_nested_variable": current_nested_variable,
            "resolved_variable_values": resolved_values,
            "resolved_nested_variables": resolved_pairs,
            "nested_attempt_counts": nested_attempt_counts,
            "failed_nested_variables": sorted(failed_nested_variables),
            "blocked_reason": "",
            "search_plan": plan_data,
            "search_results": result_data,
            "rerank_results": rerank_data,
            "subclaim_contexts": subclaim_contexts,
            "llm_feedback": [],
            "reasoning_output": {},
            "nested_decision_output": {},
            "next_action": "NESTED_RECOVERY_SEARCH_DONE" if recovery_mode else "NESTED_SEARCH_DONE",
        }
        overall_next_action = "NESTED_RECOVERY_SEARCH_DONE" if recovery_mode else "NESTED_SEARCH_DONE"
        reasoning_ready = True
    else:
        if all_steps_resolved:
            next_action = "NESTED_COMPLETE"
            overall_next_action = "NESTED_COMPLETE"
            reasoning_ready = True
        elif current_nested_variable:
            if bool(current_nested_variable.get("recovery_mode", False)):
                next_action = "READY_FOR_NESTED_RECOVERY_SEARCH"
                overall_next_action = "READY_FOR_NESTED_RECOVERY_SEARCH"
            else:
                next_action = "READY_FOR_NESTED_SEARCH"
                overall_next_action = "READY_FOR_NESTED_SEARCH"
            reasoning_ready = False
        else:
            next_action = "NESTED_BLOCKED"
            overall_next_action = "NESTED_BLOCKED"
            reasoning_ready = True
            blocked_reason = _infer_nested_blocked_reason(
                nested_plan_data,
                resolved_values,
                failed_nested_variables,
            )

        current_round = {
            "round": current_round_index,
            "mode": "NESTED_PLAN",
            "nested_plan": nested_plan_data,
            "current_nested_variable": current_nested_variable,
            "resolved_variable_values": resolved_values,
            "resolved_nested_variables": resolved_pairs,
            "nested_attempt_counts": nested_attempt_counts,
            "failed_nested_variables": sorted(failed_nested_variables),
            "blocked_reason": blocked_reason,
            "search_plan": [],
            "search_results": [],
            "rerank_results": [],
            "subclaim_contexts": [],
            "llm_feedback": [],
            "reasoning_output": {},
            "nested_decision_output": {},
            "next_action": next_action,
        }

    all_rounds = [*previous_rounds, current_round]
    return {
        "rounds": all_rounds,
        "latest_round": current_round_index,
        "overall_next_action": overall_next_action,
        "reasoning_ready": reasoning_ready,
        "subclaim_contexts": current_round.get("subclaim_contexts", []),
        "nested_plan": nested_plan_data,
        "current_nested_variable": current_nested_variable,
        "resolved_variable_values": resolved_values,
        "resolved_nested_variables": resolved_pairs,
        "nested_attempt_counts": nested_attempt_counts,
        "failed_nested_variables": sorted(failed_nested_variables),
        "blocked_reason": blocked_reason,
    }


def apply_reasoning_feedback(
    context_management: dict[str, Any],
    reasoning_output: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(context_management, dict):
        return context_management

    updated = deepcopy(context_management)
    rounds = updated.get("rounds", [])
    if not isinstance(rounds, list) or not rounds:
        return updated

    latest_round = rounds[-1]
    feedback_list = latest_round.get("llm_feedback", [])
    if not isinstance(feedback_list, list):
        feedback_list = []
    feedback_list.append(reasoning_output)
    latest_round["llm_feedback"] = feedback_list
    latest_round["reasoning_output"] = reasoning_output

    search_needed = bool(
        reasoning_output.get("search_needed", reasoning_output.get("need_more_search", False))
    )
    next_action = "RETRIEVE_MORE" if search_needed else "READY_FOR_REASONING"
    latest_round["next_action"] = next_action

    updated["rounds"] = rounds
    updated["latest_round"] = len(rounds)
    updated["overall_next_action"] = next_action
    updated["reasoning_ready"] = not search_needed
    updated["subclaim_contexts"] = latest_round.get("subclaim_contexts", [])
    return updated


def apply_quick_reasoning_feedback(
    context_management: dict[str, Any],
    quick_reasoning_output: dict[str, Any],
    variable_id: str = "",
) -> dict[str, Any]:
    if not isinstance(context_management, dict):
        return context_management

    updated = deepcopy(context_management)
    rounds = updated.get("rounds", [])
    if not isinstance(rounds, list) or not rounds:
        return updated

    latest_round = rounds[-1]
    latest_round["quick_reasoning_output"] = quick_reasoning_output
    variable_value = str(quick_reasoning_output.get("variable", "")).strip()
    resolved_values = _normalize_resolved_variable_values(updated.get("resolved_variable_values", {}))
    attempt_counts = _normalize_nested_attempt_counts(updated.get("nested_attempt_counts", {}))
    failed_nested_variables = _normalize_failed_nested_variables(
        updated.get("failed_nested_variables", [])
    )
    variable_key = variable_id.strip()

    resolved = bool(
        variable_key
        and variable_value
        and variable_value.upper() != "UNKNOWN"
    )
    if resolved:
        resolved_values[variable_key] = variable_value
        failed_nested_variables.discard(variable_key)
        attempt_counts.pop(variable_key, None)
    elif variable_key:
        new_count = attempt_counts.get(variable_key, 0) + 1
        attempt_counts[variable_key] = new_count
        if new_count >= NESTED_MAX_UNKNOWN_RETRIES_PER_VARIABLE:
            failed_nested_variables.add(variable_key)

    nested_plan = _render_nested_plan_queries(
        nested_plan=updated.get("nested_plan", {}),
        resolved_values=resolved_values,
    )
    next_variable = _select_next_nested_variable(
        nested_plan,
        resolved_values,
        failed_variables=failed_nested_variables,
    )
    all_steps_resolved = _all_nested_steps_resolved(nested_plan, resolved_values)
    blocked_reason = ""

    if all_steps_resolved:
        next_action = "NESTED_COMPLETE"
        reasoning_ready = True
    elif next_variable:
        if bool(next_variable.get("recovery_mode", False)):
            next_action = "READY_FOR_NESTED_RECOVERY_SEARCH"
        else:
            next_action = "READY_FOR_NESTED_SEARCH"
        reasoning_ready = False
    else:
        next_action = "NESTED_BLOCKED"
        reasoning_ready = True
        blocked_reason = _infer_nested_blocked_reason(
            nested_plan,
            resolved_values,
            failed_nested_variables,
        )

    latest_round["next_action"] = next_action
    latest_round["resolved_variable_update"] = (
        {variable_key: variable_value} if resolved else {}
    )
    latest_round["nested_attempt_counts"] = attempt_counts
    latest_round["failed_nested_variables"] = sorted(failed_nested_variables)
    latest_round["blocked_reason"] = blocked_reason

    updated["rounds"] = rounds
    updated["latest_round"] = len(rounds)
    updated["overall_next_action"] = next_action
    updated["reasoning_ready"] = reasoning_ready
    updated["nested_plan"] = nested_plan
    updated["current_nested_variable"] = next_variable
    updated["resolved_variable_values"] = resolved_values
    updated["resolved_nested_variables"] = [
        f"variable_id: {k}, resolved_value: {v}" for k, v in resolved_values.items()
    ]
    updated["nested_attempt_counts"] = attempt_counts
    updated["failed_nested_variables"] = sorted(failed_nested_variables)
    updated["quick_reasoning_output"] = quick_reasoning_output
    updated["blocked_reason"] = blocked_reason
    return updated


def apply_nested_decision_feedback(
    context_management: dict[str, Any],
    nested_decision_output: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(context_management, dict):
        return context_management

    updated = deepcopy(context_management)
    rounds = updated.get("rounds", [])
    if not isinstance(rounds, list) or not rounds:
        return updated

    latest_round = rounds[-1]
    latest_round["nested_decision_output"] = nested_decision_output
    latest_round["next_action"] = "NESTED_DECISION_DONE"

    updated["rounds"] = rounds
    updated["latest_round"] = len(rounds)
    updated["overall_next_action"] = "NESTED_DECISION_DONE"
    updated["nested_decision_output"] = nested_decision_output
    return updated
