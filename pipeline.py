"""Pipeline: claim_analysis -> search_planner -> search -> text_processing -> reranker -> context_management -> reasoning."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from claim_analysis import ClaimAnalyzer
from config import (
    REASONING_MAX_ROUNDS,
    SUBCLAIM_SEARCH_MAX_WORKERS,
    create_reasoning_llm,
    create_search_planner_llm,
    create_small_llm,
)
from context_management import apply_reasoning_feedback, build_context_management
from reasoning import ReasoningEngine
from reranker import rerank_by_subclaim
from search import search_subclaim_queries
from search_planner import SearchPlanner
from text_processing import process_search_results

CLAIM_ANALYSIS_MODULE = "claim_analysis"


class PipelineState(TypedDict, total=False):
    original_claim: str
    relationship_type: str
    sub_claims: list[str]
    classification_basis: str
    search_plan: list[dict[str, Any]]
    search_results: list[dict[str, Any]]
    rerank_results: list[dict[str, Any]]
    context_management: dict[str, Any]
    reasoning_output: dict[str, Any]


def _claim_analysis_node(analyzer: ClaimAnalyzer):
    def _node(state: PipelineState) -> dict[str, Any]:
        claim = state.get("original_claim", "").strip()
        if not claim:
            raise ValueError("Pipeline input requires a non-empty 'original_claim'.")

        result = analyzer.analyze(claim)
        return {
            "relationship_type": result.relationship_type,
            "sub_claims": result.sub_claims,
            "classification_basis": result.classification_basis,
        }

    return _node


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


def _search_planner_node(planner: SearchPlanner):
    def _node(state: PipelineState) -> dict[str, Any]:
        relationship_type = state.get("relationship_type", "")
        previous_context = state.get("context_management", {})
        if isinstance(previous_context, dict):
            followup_plan = _build_followup_search_plan_from_context(
                previous_context=previous_context,
                relationship_type=relationship_type,
            )
            if followup_plan:
                return {"search_plan": followup_plan}

        sub_claims = [s for s in state.get("sub_claims", []) if s and s.strip()]

        if not sub_claims:
            return {"search_plan": []}

        is_causal = relationship_type.strip().upper() == "CAUSAL"
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
        return {"search_plan": search_plan}

    return _node


def _search_node():
    def _node(state: PipelineState) -> dict[str, Any]:
        search_plan = state.get("search_plan", [])
        if not search_plan:
            return {"search_results": []}

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
        return {"search_results": rows}

    return _node


def _text_processing_node():
    def _node(state: PipelineState) -> dict[str, Any]:
        raw_rows = state.get("search_results", [])
        return {"search_results": process_search_results(raw_rows)}

    return _node


def _reranker_node():
    def _node(state: PipelineState) -> dict[str, Any]:
        search_results = state.get("search_results", [])
        return {"rerank_results": rerank_by_subclaim(search_results)}

    return _node


def _context_management_node():
    def _node(state: PipelineState) -> dict[str, Any]:
        search_plan = state.get("search_plan", [])
        search_results = state.get("search_results", [])
        rerank_results = state.get("rerank_results", [])
        previous_context = state.get("context_management", {})
        context = build_context_management(
            search_plan=search_plan,
            search_results=search_results,
            rerank_results=rerank_results,
            previous_context=previous_context if isinstance(previous_context, dict) else {},
        )
        return {"context_management": context}

    return _node


def _reasoning_node(reasoner: ReasoningEngine):
    def _node(state: PipelineState) -> dict[str, Any]:
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

        reasoning = reasoner.reason(
            original_claim=str(state.get("original_claim", "")),
            relationship_type=str(state.get("relationship_type", "")),
            subclaim_contexts=reasoning_evidence_contexts,
            previous_rounds_knowledge=previous_rounds_knowledge,
        ).model_dump()

        updated_context = apply_reasoning_feedback(
            context_management=context_data,
            reasoning_output=reasoning,
        )
        return {
            "reasoning_output": reasoning,
            "context_management": updated_context,
        }

    return _node


def build_pipeline(
    claim_analysis_llm: BaseChatModel | None = None,
    search_planner_llm: BaseChatModel | None = None,
    reasoning_llm: BaseChatModel | None = None,
):
    """Build pipeline: claim_analysis -> search_planner -> search -> text_processing -> reranker -> context_management -> reasoning."""

    analysis_llm = claim_analysis_llm or create_small_llm()
    planner_llm = search_planner_llm or create_search_planner_llm()
    reasoner_llm = reasoning_llm or create_reasoning_llm()

    analyzer = ClaimAnalyzer(analysis_llm)
    planner = SearchPlanner(planner_llm)
    reasoner = ReasoningEngine(reasoner_llm)
    graph = StateGraph(PipelineState)

    graph.add_node("claim_analysis", _claim_analysis_node(analyzer))
    graph.add_node("search_planner", _search_planner_node(planner))
    graph.add_node("search", _search_node())
    graph.add_node("text_processing", _text_processing_node())
    graph.add_node("reranker", _reranker_node())
    graph.add_node("context_management", _context_management_node())
    graph.add_node("reasoning", _reasoning_node(reasoner))
    graph.set_entry_point("claim_analysis")
    graph.add_edge("claim_analysis", "search_planner")
    graph.add_edge("search_planner", "search")
    graph.add_edge("search", "text_processing")
    graph.add_edge("text_processing", "reranker")
    graph.add_edge("reranker", "context_management")
    graph.add_edge("context_management", "reasoning")
    graph.add_edge("reasoning", END)

    return graph.compile()


def run_pipeline(
    original_claim: str,
    claim_analysis_llm: BaseChatModel | None = None,
    search_planner_llm: BaseChatModel | None = None,
    reasoning_llm: BaseChatModel | None = None,
    previous_context: dict[str, Any] | None = None,
    max_rounds: int = REASONING_MAX_ROUNDS,
) -> PipelineState:
    """Run the pipeline synchronously and return its final state."""

    claim = original_claim.strip()
    if not claim:
        raise ValueError("original_claim must not be empty.")

    app = build_pipeline(
        claim_analysis_llm=claim_analysis_llm,
        search_planner_llm=search_planner_llm,
        reasoning_llm=reasoning_llm,
    )
    rounds_limit = max(1, int(max_rounds))
    current_context = previous_context if isinstance(previous_context, dict) else {}
    final_result: dict[str, Any] | None = None

    for _ in range(rounds_limit):
        input_state: dict[str, Any] = {"original_claim": claim}
        if current_context:
            input_state["context_management"] = current_context

        result = app.invoke(input_state)
        if not isinstance(result, dict):
            raise TypeError("Pipeline returned non-dict result.")

        final_result = result
        current_context = result.get("context_management", {})
        reasoning_output = result.get("reasoning_output", {})
        needs_more_search = False
        if isinstance(reasoning_output, dict):
            needs_more_search = bool(reasoning_output.get("search_needed", False))

        if not needs_more_search:
            break

    if final_result is None:
        raise RuntimeError("Pipeline did not produce a result.")
    return cast(PipelineState, final_result)
