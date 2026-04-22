"""Pipeline with branch:
- NESTED: claim_analysis -> nested_planner -> context_management -> search_planner -> search -> text_processing -> reranker -> context_management -> quick_reasoning -> nested_decision -> END
- others: claim_analysis -> search_planner -> search -> text_processing -> reranker -> context_management -> reasoning

If nested_planner returns empty steps, the pipeline falls back to ATOMIC path.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
from threading import Lock
from typing import Any, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from claim_analysis import ClaimAnalyzer
from config import (
    REASONING_MAX_ROUNDS,
    SUBCLAIM_SEARCH_MAX_WORKERS,
    create_nested_decision_llm,
    create_nested_planner_llm,
    create_quick_reasoning_llm,
    create_reasoning_llm,
    create_search_planner_llm,
    create_small_llm,
)
from context_management import (
    apply_nested_decision_feedback,
    apply_quick_reasoning_feedback,
    apply_reasoning_feedback,
    build_context_management,
    build_nested_context_management,
)
from nested_decision import NestedDecisionEngine
from nested_planner import NestedPlanner
from quick_reasoning import QuickReasoningEngine
from reasoning import ReasoningEngine
from reranker import rerank_by_subclaim
from search import search_subclaim_queries
from search_planner import SearchPlanner
from text_processing import process_search_results

CLAIM_ANALYSIS_MODULE = "claim_analysis"
_SUPPORTED_RELATIONSHIP_TYPES = {"NESTED", "ATOMIC", "CAUSAL"}
_PROGRESS_LOG_PATH: Path | None = None
_PROGRESS_LOG_LOCK = Lock()


def _normalize_tags(raw_tags: Any) -> list[str]:
    if not isinstance(raw_tags, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_tags:
        tag = str(item).strip()
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(tag)
    return normalized


class PipelineState(TypedDict, total=False):
    original_claim: str
    relationship_type: str
    sub_claims: list[str]
    classification_basis: str
    nested_plan: dict[str, Any]
    search_plan: list[dict[str, Any]]
    search_results: list[dict[str, Any]]
    rerank_results: list[dict[str, Any]]
    context_management: dict[str, Any]
    reasoning_output: dict[str, Any]
    quick_reasoning_output: dict[str, Any]
    nested_decision_output: dict[str, Any]


def set_progress_log_path(path: str | Path | None) -> None:
    global _PROGRESS_LOG_PATH
    if path is None:
        _PROGRESS_LOG_PATH = None
        return
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("", encoding="utf-8")
    _PROGRESS_LOG_PATH = progress_path


def _progress(enabled: bool, message: str) -> None:
    if not enabled:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    print(line, flush=True)
    if _PROGRESS_LOG_PATH is not None:
        with _PROGRESS_LOG_LOCK:
            with _PROGRESS_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


def _progress_data(enabled: bool, label: str, data: Any) -> None:
    if not enabled:
        return
    _progress(enabled, f"dataflow {label} BEGIN")
    try:
        body = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except Exception:
        body = str(data)
    for line in body.splitlines():
        _progress(enabled, line)
    _progress(enabled, f"dataflow {label} END")


def _get_relationship_type(state: PipelineState) -> str:
    relationship_type = str(state.get("relationship_type", "")).strip().upper()
    if relationship_type not in _SUPPORTED_RELATIONSHIP_TYPES:
        raise ValueError(
            f"Unsupported relationship_type: {relationship_type!r}. "
            f"Expected one of {_SUPPORTED_RELATIONSHIP_TYPES}."
        )
    return relationship_type


def _claim_analysis_node(analyzer: ClaimAnalyzer, show_progress: bool = False):
    def _node(state: PipelineState) -> dict[str, Any]:
        _progress(show_progress, "step: claim_analysis")
        claim = state.get("original_claim", "").strip()
        _progress_data(show_progress, "claim_analysis.input", {"original_claim": claim})
        if not claim:
            raise ValueError("Pipeline input requires a non-empty 'original_claim'.")

        result = analyzer.analyze(claim)
        _progress(
            show_progress,
            f"claim_analysis result: relationship_type={result.relationship_type}, sub_claims={len(result.sub_claims)}",
        )
        output = {
            "relationship_type": result.relationship_type,
            "sub_claims": result.sub_claims,
            "classification_basis": result.classification_basis,
        }
        _progress_data(show_progress, "claim_analysis.output", output)
        return output

    return _node


def _nested_planner_node(planner: NestedPlanner, show_progress: bool = False):
    def _node(state: PipelineState) -> dict[str, Any]:
        _progress(show_progress, "step: nested_planner")
        relationship_type = str(state.get("relationship_type", "")).strip().upper()
        _progress_data(
            show_progress,
            "nested_planner.input",
            {
                "relationship_type": relationship_type,
                "sub_claims": state.get("sub_claims", []),
                "classification_basis": state.get("classification_basis", ""),
            },
        )
        if relationship_type != "NESTED":
            return {}

        sub_claims = state.get("sub_claims", [])
        if not isinstance(sub_claims, list) or not sub_claims:
            return {"nested_plan": {}}

        result = planner.plan(
            relationship_type=relationship_type,
            sub_claims=sub_claims,
            classification_basis=str(state.get("classification_basis", "")),
        )
        steps = result.steps if isinstance(result.steps, list) else []
        _progress(show_progress, f"nested_planner result: steps={len(steps)}")

        if not steps:
            fallback_claim = str(sub_claims[0]).strip() if sub_claims else ""
            if not fallback_claim:
                fallback_claim = str(state.get("original_claim", "")).strip()

            # Empty nested plan means no dependency chain is required.
            output = {
                "relationship_type": "ATOMIC",
                "sub_claims": [fallback_claim] if fallback_claim else [],
                "nested_plan": {},
            }
            _progress(
                show_progress,
                "nested_planner fallback: empty plan -> switch relationship_type to ATOMIC",
            )
            _progress_data(show_progress, "nested_planner.output", output)
            return output

        output = {"nested_plan": result.model_dump()}
        _progress_data(show_progress, "nested_planner.output", output)
        return output

    return _node


def _route_after_claim_analysis(state: PipelineState) -> str:
    relationship_type = _get_relationship_type(state)
    if relationship_type == "NESTED":
        return "nested_planner"
    return "search_planner"


def _route_after_nested_planner(state: PipelineState) -> str:
    relationship_type = _get_relationship_type(state)
    if relationship_type == "NESTED":
        return "context_management"
    return "search_planner"


def _route_after_context_management(state: PipelineState) -> str:
    relationship_type = _get_relationship_type(state)
    if relationship_type == "NESTED":
        context = state.get("context_management", {})
        context_data = context if isinstance(context, dict) else {}
        action = str(context_data.get("overall_next_action", "")).strip().upper()
        if action == "NESTED_COMPLETE":
            return "nested_decision"
        if action == "NESTED_BLOCKED":
            return "end"
        if "rerank_results" in state:
            return "quick_reasoning"
        return "search_planner"
    return "reasoning"


def _route_after_quick_reasoning(state: PipelineState) -> str:
    relationship_type = _get_relationship_type(state)
    if relationship_type != "NESTED":
        return "end"

    context = state.get("context_management", {})
    context_data = context if isinstance(context, dict) else {}
    action = str(context_data.get("overall_next_action", "")).strip().upper()
    if action == "NESTED_COMPLETE":
        return "nested_decision"
    if action == "NESTED_BLOCKED":
        return "end"
    return "search_planner"


def _build_followup_search_plan_from_context(
    previous_context: dict[str, Any],
    relationship_type: str,
) -> list[dict[str, Any]]:
    rounds = previous_context.get("rounds", [])
    if not isinstance(rounds, list) or not rounds:
        return []

    latest_round = rounds[-1]
    if not isinstance(latest_round, dict):
        return []

    reasoning_output = latest_round.get("reasoning_output", {})
    if not isinstance(reasoning_output, dict):
        return []

    search_needed = bool(reasoning_output.get("search_needed", False))
    if not search_needed:
        return []

    missing_info = reasoning_output.get("missing_information", [])
    if not isinstance(missing_info, list):
        return []

    followup_plan: list[dict[str, Any]] = []
    for item in missing_info:
        if not isinstance(item, dict):
            continue

        question = str(item.get("question", "")).strip()
        raw_queries = item.get("queries", [])
        if not question or not isinstance(raw_queries, list):
            continue

        queries = [str(q).strip() for q in raw_queries if str(q).strip()]

        followup_plan.append(
            {
                "sub_claim": question,
                "relationship_type": relationship_type,
                "is_followup": True,
                "query_source": "reasoning_missing_information",
                "queries": queries,
            }
        )

    return followup_plan


def _dedupe_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for query in queries:
        q = str(query).strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(q)
    return deduped


def _build_nested_recovery_queries(
    variable_description: str,
    unresolved_dependency_descriptions: list[str],
    original_claim: str,
) -> list[str]:
    variable_desc = variable_description.strip()
    unresolved = [str(item).strip() for item in unresolved_dependency_descriptions if str(item).strip()]
    claim = original_claim.strip()
    if not variable_desc:
        return []

    unresolved_text = " ".join(unresolved[:2]).strip()
    candidates: list[str] = []
    if unresolved_text:
        candidates.extend(
            [
                f"{variable_desc} related to {unresolved_text}",
                f"{unresolved_text} {variable_desc}",
            ]
        )
    if claim:
        candidates.append(f"{variable_desc} {claim}")
    candidates.append(f"{variable_desc} fact check evidence")
    return _dedupe_queries(candidates)[:3]


def _search_planner_node(planner: SearchPlanner, show_progress: bool = False):
    def _node(state: PipelineState) -> dict[str, Any]:
        _progress(show_progress, "step: search_planner")
        relationship_type = _get_relationship_type(state)
        _progress_data(
            show_progress,
            "search_planner.input",
            {
                "relationship_type": relationship_type,
                "sub_claims": state.get("sub_claims", []),
                "context_management": state.get("context_management", {}),
            },
        )

        # NESTED: read current target variable from context_management and generate queries.
        if relationship_type == "NESTED":
            context = state.get("context_management", {})
            context_data = context if isinstance(context, dict) else {}
            current_var = context_data.get("current_nested_variable", {})
            current_var_data = current_var if isinstance(current_var, dict) else {}

            variable_id = str(current_var_data.get("variable_id", "")).strip()
            variable_description = str(current_var_data.get("variable_description", "")).strip()
            if not variable_id or not variable_description:
                output = {"search_plan": []}
                _progress_data(show_progress, "search_planner.output", output)
                return output

            depends_on = current_var_data.get("depends_on", [])
            depends_on_list = depends_on if isinstance(depends_on, list) else []
            query_hint = str(current_var_data.get("query_hint", "")).strip()
            recovery_mode = bool(current_var_data.get("recovery_mode", False))
            unresolved_descriptions_raw = current_var_data.get(
                "unresolved_dependency_descriptions", []
            )
            unresolved_dependency_descriptions = (
                [str(item).strip() for item in unresolved_descriptions_raw if str(item).strip()]
                if isinstance(unresolved_descriptions_raw, list)
                else []
            )
            resolved_value_map_raw = context_data.get("resolved_variable_values", {})
            resolved_value_map = (
                {
                    str(k).strip(): str(v).strip()
                    for k, v in resolved_value_map_raw.items()
                    if str(k).strip() and str(v).strip()
                }
                if isinstance(resolved_value_map_raw, dict)
                else {}
            )
            resolved_variables = [
                f"variable_id: {k}, resolved_value: {v}"
                for k, v in resolved_value_map.items()
            ]

            result = planner.plan_nested_variable(
                variable_id=variable_id,
                variable_description=variable_description,
                query_hint=query_hint,
                resolved_variables=resolved_variables,
            )
            planned_queries = result.to_query_list()
            query_source = "nested_variable_query_generator"
            if recovery_mode:
                recovery_queries = _build_nested_recovery_queries(
                    variable_description=variable_description,
                    unresolved_dependency_descriptions=unresolved_dependency_descriptions,
                    original_claim=str(state.get("original_claim", "")),
                )
                planned_queries = _dedupe_queries([*planned_queries, *recovery_queries])[:6]
                query_source = "nested_variable_query_generator_with_recovery"

            _progress(
                show_progress,
                f"nested target variable: {variable_id} | hint: {query_hint} | recovery={recovery_mode}",
            )

            output = {
                "search_plan": [
                    {
                        "sub_claim": variable_description,
                        "relationship_type": relationship_type,
                        "nested_variable_id": variable_id,
                        "nested_depends_on": depends_on_list,
                        "resolved_variables": resolved_variables,
                        "recovery_mode": recovery_mode,
                        "unresolved_dependency_descriptions": unresolved_dependency_descriptions,
                        "query_hint": query_hint,
                        "query_plan": [item.model_dump() for item in result.query_plan],
                        "queries": planned_queries,
                        "query_source": query_source,
                    }
                ]
            }
            _progress_data(show_progress, "search_planner.output", output)
            return output

        # ATOMIC/CAUSAL only: follow-up queries from reasoning output.
        previous_context = state.get("context_management", {})
        if isinstance(previous_context, dict):
            followup_plan = _build_followup_search_plan_from_context(
                previous_context=previous_context,
                relationship_type=relationship_type,
            )
            if followup_plan:
                _progress(
                    show_progress,
                    f"search_planner followup plan: items={len(followup_plan)}",
                )
                output = {"search_plan": followup_plan}
                _progress_data(show_progress, "search_planner.output", output)
                return output

        sub_claims = [s for s in state.get("sub_claims", []) if s and s.strip()]

        if not sub_claims:
            output = {"search_plan": []}
            _progress_data(show_progress, "search_planner.output", output)
            return output

        is_causal = relationship_type == "CAUSAL"
        causal_original_idx = len(sub_claims) - 1 if is_causal else -1

        def _plan_one(item: tuple[int, str]) -> dict[str, Any]:
            idx, sub_claim = item
            if idx == causal_original_idx:
                result = planner.plan_causal_original(sub_claim=sub_claim)
                queries = result.to_causal_query_list()
            else:
                result = planner.plan(relationship_type=relationship_type, sub_claim=sub_claim)
                queries = result.to_query_list()
            return {
                "sub_claim": sub_claim,
                "relationship_type": relationship_type,
                "subclaim_analysis": result.subclaim_analysis.model_dump(),
                "minimal_sufficient_information_set": [
                    item.model_dump() for item in result.minimal_sufficient_information_set
                ],
                "query_plan": [item.model_dump() for item in result.query_plan],
                "queries": queries,
            }

        # Parallel planner calls for each sub-claim.
        max_workers = min(4, len(sub_claims))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            search_plan = list(executor.map(_plan_one, list(enumerate(sub_claims))))
        _progress(show_progress, f"search_planner result: items={len(search_plan)}")
        output = {"search_plan": search_plan}
        _progress_data(show_progress, "search_planner.output", output)
        return output

    return _node


def _search_node(show_progress: bool = False):
    def _node(state: PipelineState) -> dict[str, Any]:
        search_plan = state.get("search_plan", [])
        _progress(show_progress, f"step: search (plans={len(search_plan) if isinstance(search_plan, list) else 0})")
        _progress_data(show_progress, "search.input", {"search_plan": search_plan})
        if not search_plan:
            output = {"search_results": []}
            _progress_data(show_progress, "search.output", output)
            return output

        def _search_one_subclaim(item: dict[str, Any]) -> list[dict[str, Any]]:
            sub_claim = str(item.get("sub_claim", "")).strip()
            queries = item.get("queries", [])
            if not sub_claim or not isinstance(queries, list):
                return []
            return search_subclaim_queries(sub_claim=sub_claim, queries=queries)

        max_workers = min(SUBCLAIM_SEARCH_MAX_WORKERS, len(search_plan))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            nested_rows = list(executor.map(_search_one_subclaim, search_plan))

        rows: list[dict[str, Any]] = []
        for batch in nested_rows:
            rows.extend(batch)
        _progress(show_progress, f"search result rows={len(rows)}")
        output = {"search_results": rows}
        _progress_data(show_progress, "search.output", output)
        return output

    return _node


def _text_processing_node(show_progress: bool = False):
    def _node(state: PipelineState) -> dict[str, Any]:
        raw_rows = state.get("search_results", [])
        _progress(show_progress, f"step: text_processing (rows={len(raw_rows) if isinstance(raw_rows, list) else 0})")
        _progress_data(show_progress, "text_processing.input", {"search_results": raw_rows})
        processed = process_search_results(raw_rows)
        _progress(show_progress, f"text_processing done: rows={len(processed)}")
        output = {"search_results": processed}
        _progress_data(show_progress, "text_processing.output", output)
        return output

    return _node


def _reranker_node(show_progress: bool = False):
    def _node(state: PipelineState) -> dict[str, Any]:
        relationship_type = _get_relationship_type(state)
        search_results = state.get("search_results", [])
        _progress(
            show_progress,
            f"step: reranker (mode={'query' if relationship_type == 'NESTED' else 'sub_claim'}, rows={len(search_results) if isinstance(search_results, list) else 0})",
        )
        _progress_data(
            show_progress,
            "reranker.input",
            {
                "relationship_type": relationship_type,
                "use_query_target": relationship_type == "NESTED",
                "search_results": search_results,
            },
        )
        use_query_target = relationship_type == "NESTED"
        rerank_results = rerank_by_subclaim(
            search_results,
            use_query_target=use_query_target,
        )
        _progress(show_progress, f"reranker done: groups={len(rerank_results)}")
        output = {"rerank_results": rerank_results}
        _progress_data(show_progress, "reranker.output", output)
        return output

    return _node


def _context_management_node(show_progress: bool = False):
    def _node(state: PipelineState) -> dict[str, Any]:
        _progress(show_progress, "step: context_management")
        relationship_type = _get_relationship_type(state)
        previous_context = state.get("context_management", {})
        previous_context_data = previous_context if isinstance(previous_context, dict) else {}
        _progress_data(
            show_progress,
            "context_management.input",
            {
                "relationship_type": relationship_type,
                "previous_context": previous_context_data,
                "nested_plan": state.get("nested_plan", {}),
                "search_plan": state.get("search_plan", []),
                "search_results": state.get("search_results", []),
                "rerank_results": state.get("rerank_results", []),
            },
        )

        if relationship_type == "NESTED":
            nested_plan = state.get("nested_plan", {})
            context = build_nested_context_management(
                nested_plan=nested_plan if isinstance(nested_plan, dict) else {},
                previous_context=previous_context_data,
                search_plan=state.get("search_plan", []),
                search_results=state.get("search_results", []),
                rerank_results=state.get("rerank_results", []),
            )
            current_var = context.get("current_nested_variable", {})
            current_var_id = (
                str(current_var.get("variable_id", "")).strip()
                if isinstance(current_var, dict)
                else ""
            )
            action = str(context.get("overall_next_action", "")).strip()
            _progress(
                show_progress,
                f"context nested action: {action} | next variable: {current_var_id or 'none'}",
            )
            output = {"context_management": context}
            _progress_data(show_progress, "context_management.output", output)
            return output

        search_plan = state.get("search_plan", [])
        search_results = state.get("search_results", [])
        rerank_results = state.get("rerank_results", [])
        context = build_context_management(
            search_plan=search_plan,
            search_results=search_results,
            rerank_results=rerank_results,
            previous_context=previous_context_data,
        )
        output = {"context_management": context}
        _progress_data(show_progress, "context_management.output", output)
        return output

    return _node


def _reasoning_node(reasoner: ReasoningEngine, show_progress: bool = False):
    def _node(state: PipelineState) -> dict[str, Any]:
        _progress(show_progress, "step: reasoning")
        context = state.get("context_management", {})
        context_data = context if isinstance(context, dict) else {}
        subclaim_contexts = context_data.get("subclaim_contexts", [])
        if not isinstance(subclaim_contexts, list):
            subclaim_contexts = []
        reasoning_evidence_contexts: list[dict[str, Any]] = []
        for item in subclaim_contexts:
            if not isinstance(item, dict):
                continue
            reasoning_evidence_contexts.append(
                {
                    "planned_queries": item.get("planned_queries", []),
                    "evidence_chunks": item.get("evidence_chunks", []),
                }
            )

        previous_rounds_knowledge: list[dict[str, Any]] = []
        rounds = context_data.get("rounds", [])
        if isinstance(rounds, list) and len(rounds) >= 2:
            for round_item in rounds[:-1]:
                if not isinstance(round_item, dict):
                    continue
                round_reasoning = round_item.get("reasoning_output", {})
                if not isinstance(round_reasoning, dict):
                    continue

                known_information = round_reasoning.get("known_information", [])
                missing_information = round_reasoning.get("missing_information", [])
                if not isinstance(known_information, list):
                    known_information = []
                if not isinstance(missing_information, list):
                    missing_information = []

                previous_rounds_knowledge.append(
                    {
                        "round": int(round_item.get("round", len(previous_rounds_knowledge) + 1)),
                        "known_information": known_information,
                        "missing_information": missing_information,
                        "search_needed": bool(round_reasoning.get("search_needed", False)),
                        "reasoning_note": str(round_reasoning.get("reasoning_note", "")).strip(),
                    }
                )
        _progress_data(
            show_progress,
            "reasoning.input",
            {
                "original_claim": str(state.get("original_claim", "")),
                "relationship_type": str(state.get("relationship_type", "")),
                "subclaim_contexts": reasoning_evidence_contexts,
                "previous_rounds_knowledge": previous_rounds_knowledge,
            },
        )

        reasoning = reasoner.reason(
            original_claim=str(state.get("original_claim", "")),
            relationship_type=str(state.get("relationship_type", "")),
            subclaim_contexts=reasoning_evidence_contexts,
            previous_rounds_knowledge=previous_rounds_knowledge,
        ).model_dump()
        _progress(
            show_progress,
            f"reasoning result: search_needed={bool(reasoning.get('search_needed', False))}, missing_items={len(reasoning.get('missing_information', [])) if isinstance(reasoning.get('missing_information', []), list) else 0}",
        )

        updated_context = apply_reasoning_feedback(
            context_management=context_data,
            reasoning_output=reasoning,
        )
        output = {
            "reasoning_output": reasoning,
            "context_management": updated_context,
        }
        _progress_data(show_progress, "reasoning.output", output)
        return output

    return _node


def _quick_reasoning_node(quick_reasoner: QuickReasoningEngine, show_progress: bool = False):
    def _node(state: PipelineState) -> dict[str, Any]:
        _progress(show_progress, "step: quick_reasoning")
        relationship_type = _get_relationship_type(state)
        if relationship_type != "NESTED":
            return {}

        context = state.get("context_management", {})
        context_data = context if isinstance(context, dict) else {}
        current_var = context_data.get("current_nested_variable", {})
        current_var_data = current_var if isinstance(current_var, dict) else {}

        variable_id = str(current_var_data.get("variable_id", "")).strip()
        variable_description = str(current_var_data.get("variable_description", "")).strip()
        if not variable_id or not variable_description:
            quick_output = {"biref explain": "No variable information was provided.", "variable": "UNKNOWN"}
            updated_context = apply_quick_reasoning_feedback(context_data, quick_output)
            _progress(show_progress, "quick_reasoning result: UNKNOWN (missing variable input)")
            output = {
                "quick_reasoning_output": quick_output,
                "context_management": updated_context,
            }
            _progress_data(show_progress, "quick_reasoning.output", output)
            return output

        subclaim_contexts = context_data.get("subclaim_contexts", [])
        evidence_chunks: list[dict[str, Any]] = []
        if isinstance(subclaim_contexts, list):
            for item in subclaim_contexts:
                if not isinstance(item, dict):
                    continue
                sub_claim = str(item.get("sub_claim", "")).strip()
                if sub_claim != variable_description:
                    continue
                chunks = item.get("evidence_chunks", [])
                if isinstance(chunks, list):
                    evidence_chunks = chunks
                break
        _progress_data(
            show_progress,
            "quick_reasoning.input",
            {
                "variable_id": variable_id,
                "variable_description": variable_description,
                "top_k_evidence_chunks": evidence_chunks,
                "context_management": context_data,
            },
        )

        result = quick_reasoner.infer_variable(
            variable_id=variable_id,
            variable_description=variable_description,
            top_k_evidence_chunks=evidence_chunks,
        )
        quick_output = result.model_dump(by_alias=True)
        _progress(
            show_progress,
            f"quick_reasoning result: {variable_id} -> {str(quick_output.get('variable', '')).strip()}",
        )
        updated_context = apply_quick_reasoning_feedback(
            context_data,
            quick_output,
            variable_id=variable_id,
        )
        attempts = 0
        attempt_map = updated_context.get("nested_attempt_counts", {})
        if isinstance(attempt_map, dict):
            try:
                attempts = int(attempt_map.get(variable_id, 0))
            except Exception:
                attempts = 0
        action = str(updated_context.get("overall_next_action", "")).strip()
        _progress(
            show_progress,
            f"quick_reasoning status: action={action}, attempts_for_{variable_id}={attempts}",
        )
        output = {
            "quick_reasoning_output": quick_output,
            "context_management": updated_context,
        }
        _progress_data(show_progress, "quick_reasoning.output", output)
        return output

    return _node


def _nested_decision_node(
    decider: NestedDecisionEngine,
    show_progress: bool = False,
):
    def _node(state: PipelineState) -> dict[str, Any]:
        _progress(show_progress, "step: nested_decision")
        relationship_type = _get_relationship_type(state)
        if relationship_type != "NESTED":
            return {}

        context = state.get("context_management", {})
        context_data = context if isinstance(context, dict) else {}
        action = str(context_data.get("overall_next_action", "")).strip().upper()
        if action != "NESTED_COMPLETE":
            _progress(show_progress, f"nested_decision skipped: action={action or 'UNKNOWN'}")
            return {}

        nested_plan = context_data.get("nested_plan", {})
        nested_plan_data = nested_plan if isinstance(nested_plan, dict) else {}
        resolved_values_raw = context_data.get("resolved_variable_values", {})
        resolved_values: dict[str, str] = {}
        if isinstance(resolved_values_raw, dict):
            resolved_values = {
                str(k).strip(): str(v).strip()
                for k, v in resolved_values_raw.items()
                if str(k).strip() and str(v).strip()
            }

        _progress_data(
            show_progress,
            "nested_decision.input",
            {
                "original_claim": str(state.get("original_claim", "")),
                "nested_plan": nested_plan_data,
                "resolved_variable_values": resolved_values,
            },
        )

        result = decider.decide(
            original_claim=str(state.get("original_claim", "")),
            nested_plan=nested_plan_data,
            resolved_variable_values=resolved_values,
        )
        decision_output = result.model_dump()
        _progress(
            show_progress,
            f"nested_decision result: label={decision_output.get('label', '')}",
        )

        updated_context = apply_nested_decision_feedback(
            context_management=context_data,
            nested_decision_output=decision_output,
        )
        output = {
            "nested_decision_output": decision_output,
            "context_management": updated_context,
        }
        _progress_data(show_progress, "nested_decision.output", output)
        return output

    return _node


def build_pipeline(
    claim_analysis_llm: BaseChatModel | None = None,
    nested_planner_llm: BaseChatModel | None = None,
    nested_decision_llm: BaseChatModel | None = None,
    search_planner_llm: BaseChatModel | None = None,
    reasoning_llm: BaseChatModel | None = None,
    quick_reasoning_llm: BaseChatModel | None = None,
    show_progress: bool = False,
):
    """Build pipeline with NESTED context-management early-stop branch."""

    analysis_llm = claim_analysis_llm or create_small_llm()
    nested_llm = nested_planner_llm or create_nested_planner_llm()
    nested_decision_model = nested_decision_llm or create_nested_decision_llm()
    planner_llm = search_planner_llm or create_search_planner_llm()
    reasoner_llm = reasoning_llm or create_reasoning_llm()
    quick_llm = quick_reasoning_llm or create_quick_reasoning_llm()

    analyzer = ClaimAnalyzer(analysis_llm)
    nested_planner = NestedPlanner(nested_llm)
    planner = SearchPlanner(planner_llm)
    reasoner = ReasoningEngine(reasoner_llm)
    quick_reasoner = QuickReasoningEngine(quick_llm)
    nested_decider = NestedDecisionEngine(nested_decision_model)
    graph = StateGraph(PipelineState)

    graph.add_node("claim_analysis", _claim_analysis_node(analyzer, show_progress=show_progress))
    graph.add_node("nested_planner", _nested_planner_node(nested_planner, show_progress=show_progress))
    graph.add_node("search_planner", _search_planner_node(planner, show_progress=show_progress))
    graph.add_node("search", _search_node(show_progress=show_progress))
    graph.add_node("text_processing", _text_processing_node(show_progress=show_progress))
    graph.add_node("reranker", _reranker_node(show_progress=show_progress))
    graph.add_node("context_management", _context_management_node(show_progress=show_progress))
    graph.add_node("reasoning", _reasoning_node(reasoner, show_progress=show_progress))
    graph.add_node("quick_reasoning", _quick_reasoning_node(quick_reasoner, show_progress=show_progress))
    graph.add_node("nested_decision", _nested_decision_node(nested_decider, show_progress=show_progress))
    graph.set_entry_point("claim_analysis")
    graph.add_conditional_edges(
        "claim_analysis",
        _route_after_claim_analysis,
        {
            "nested_planner": "nested_planner",
            "search_planner": "search_planner",
        },
    )
    graph.add_conditional_edges(
        "nested_planner",
        _route_after_nested_planner,
        {
            "context_management": "context_management",
            "search_planner": "search_planner",
        },
    )
    graph.add_edge("search_planner", "search")
    graph.add_edge("search", "text_processing")
    graph.add_edge("text_processing", "reranker")
    graph.add_edge("reranker", "context_management")
    graph.add_conditional_edges(
        "context_management",
        _route_after_context_management,
        {
            "search_planner": "search_planner",
            "reasoning": "reasoning",
            "quick_reasoning": "quick_reasoning",
            "nested_decision": "nested_decision",
            "end": END,
        },
    )
    graph.add_edge("reasoning", END)
    graph.add_edge("nested_decision", END)
    graph.add_conditional_edges(
        "quick_reasoning",
        _route_after_quick_reasoning,
        {
            "search_planner": "search_planner",
            "nested_decision": "nested_decision",
            "end": END,
        },
    )

    return graph.compile()


def run_pipeline(
    original_claim: str,
    claim_analysis_llm: BaseChatModel | None = None,
    nested_planner_llm: BaseChatModel | None = None,
    nested_decision_llm: BaseChatModel | None = None,
    search_planner_llm: BaseChatModel | None = None,
    reasoning_llm: BaseChatModel | None = None,
    quick_reasoning_llm: BaseChatModel | None = None,
    previous_context: dict[str, Any] | None = None,
    max_rounds: int = REASONING_MAX_ROUNDS,
    show_progress: bool = True,
    invoke_config: dict[str, Any] | None = None,
) -> PipelineState:
    """Run the pipeline synchronously and return its final state."""

    claim = original_claim.strip()
    if not claim:
        raise ValueError("original_claim must not be empty.")

    app = build_pipeline(
        claim_analysis_llm=claim_analysis_llm,
        nested_planner_llm=nested_planner_llm,
        nested_decision_llm=nested_decision_llm,
        search_planner_llm=search_planner_llm,
        reasoning_llm=reasoning_llm,
        quick_reasoning_llm=quick_reasoning_llm,
        show_progress=show_progress,
    )
    rounds_limit = max(1, int(max_rounds))
    current_context = previous_context if isinstance(previous_context, dict) else {}
    final_result: dict[str, Any] | None = None
    base_invoke_config = dict(invoke_config) if isinstance(invoke_config, dict) else {}

    for round_idx in range(rounds_limit):
        _progress(show_progress, f"round {round_idx + 1}/{rounds_limit}: start")
        input_state: dict[str, Any] = {"original_claim": claim}
        if current_context:
            input_state["context_management"] = current_context
        _progress_data(show_progress, f"round_{round_idx + 1}.invoke_input", input_state)

        round_invoke_config = dict(base_invoke_config)
        base_run_name = str(base_invoke_config.get("run_name", "")).strip() or "fact_check_pipeline"
        round_invoke_config["run_name"] = f"{base_run_name}.round_{round_idx + 1}"

        round_tags = _normalize_tags(round_invoke_config.get("tags"))
        round_tag = f"round:{round_idx + 1}"
        if round_tag.lower() not in {t.lower() for t in round_tags}:
            round_tags.append(round_tag)
        round_invoke_config["tags"] = round_tags

        metadata = round_invoke_config.get("metadata", {})
        metadata_map = dict(metadata) if isinstance(metadata, dict) else {}
        metadata_map["round"] = round_idx + 1
        round_invoke_config["metadata"] = metadata_map

        result = app.invoke(input_state, config=round_invoke_config or None)
        if not isinstance(result, dict):
            raise TypeError("Pipeline returned non-dict result.")
        _progress_data(show_progress, f"round_{round_idx + 1}.invoke_output", result)

        final_result = result
        current_context = result.get("context_management", {})
        relationship_type = str(result.get("relationship_type", "")).strip().upper()

        if relationship_type == "NESTED":
            context_data = current_context if isinstance(current_context, dict) else {}
            nested_action = str(context_data.get("overall_next_action", "")).strip().upper()
            _progress(show_progress, f"round {round_idx + 1}: nested action = {nested_action}")
            if nested_action in {"NESTED_COMPLETE", "NESTED_BLOCKED", "NESTED_DECISION_DONE"}:
                _progress(show_progress, f"round {round_idx + 1}: stop")
                break
            continue

        reasoning_output = result.get("reasoning_output", {})
        needs_more_search = False
        if isinstance(reasoning_output, dict):
            needs_more_search = bool(reasoning_output.get("search_needed", False))
        _progress(show_progress, f"round {round_idx + 1}: search_needed = {needs_more_search}")

        if not needs_more_search:
            _progress(show_progress, f"round {round_idx + 1}: stop")
            break

    if final_result is None:
        raise RuntimeError("Pipeline did not produce a result.")
    _progress_data(show_progress, "pipeline.final_result", final_result)
    _progress(show_progress, "pipeline: done")
    return cast(PipelineState, final_result)
