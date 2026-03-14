"""Pipeline: claim_analysis -> search_planner -> search -> text_processing."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from claim_analysis import ClaimAnalyzer
from config import SUBCLAIM_SEARCH_MAX_WORKERS, create_search_planner_llm, create_small_llm
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
    search_results: list[dict[str, str]]


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


def _search_planner_node(planner: SearchPlanner):
    def _node(state: PipelineState) -> dict[str, Any]:
        relationship_type = state.get("relationship_type", "")
        sub_claims = [s for s in state.get("sub_claims", []) if s and s.strip()]

        if not sub_claims:
            return {"search_plan": []}

        def _plan_one(sub_claim: str) -> dict[str, Any]:
            result = planner.plan(relationship_type=relationship_type, sub_claim=sub_claim)
            return {
                "sub_claim": sub_claim,
                "relationship_type": relationship_type,
                "queries": result.to_query_list(),
            }

        # Parallel planner calls for each sub-claim.
        max_workers = min(4, len(sub_claims))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            search_plan = list(executor.map(_plan_one, sub_claims))
        return {"search_plan": search_plan}

    return _node


def _search_node():
    def _node(state: PipelineState) -> dict[str, Any]:
        search_plan = state.get("search_plan", [])
        if not search_plan:
            return {"search_results": []}

        def _search_one_subclaim(item: dict[str, Any]) -> list[dict[str, str]]:
            sub_claim = str(item.get("sub_claim", "")).strip()
            queries = item.get("queries", [])
            if not sub_claim or not isinstance(queries, list):
                return []
            return search_subclaim_queries(sub_claim=sub_claim, queries=queries)

        max_workers = min(SUBCLAIM_SEARCH_MAX_WORKERS, len(search_plan))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            nested_rows = list(executor.map(_search_one_subclaim, search_plan))

        rows: list[dict[str, str]] = []
        for batch in nested_rows:
            rows.extend(batch)
        return {"search_results": rows}

    return _node


def _text_processing_node():
    def _node(state: PipelineState) -> dict[str, Any]:
        raw_rows = state.get("search_results", [])
        return {"search_results": process_search_results(raw_rows)}

    return _node


def build_pipeline(
    claim_analysis_llm: BaseChatModel | None = None,
    search_planner_llm: BaseChatModel | None = None,
):
    """Build pipeline: claim_analysis -> search_planner -> search -> text_processing."""

    analysis_llm = claim_analysis_llm or create_small_llm()
    planner_llm = search_planner_llm or create_search_planner_llm()

    analyzer = ClaimAnalyzer(analysis_llm)
    planner = SearchPlanner(planner_llm)
    graph = StateGraph(PipelineState)

    graph.add_node("claim_analysis", _claim_analysis_node(analyzer))
    graph.add_node("search_planner", _search_planner_node(planner))
    graph.add_node("search", _search_node())
    graph.add_node("text_processing", _text_processing_node())
    graph.set_entry_point("claim_analysis")
    graph.add_edge("claim_analysis", "search_planner")
    graph.add_edge("search_planner", "search")
    graph.add_edge("search", "text_processing")
    graph.add_edge("text_processing", END)

    return graph.compile()


def run_pipeline(
    original_claim: str,
    claim_analysis_llm: BaseChatModel | None = None,
    search_planner_llm: BaseChatModel | None = None,
) -> PipelineState:
    """Run the pipeline synchronously and return its final state."""

    claim = original_claim.strip()
    if not claim:
        raise ValueError("original_claim must not be empty.")

    app = build_pipeline(
        claim_analysis_llm=claim_analysis_llm,
        search_planner_llm=search_planner_llm,
    )
    result = app.invoke({"original_claim": claim})
    if not isinstance(result, dict):
        raise TypeError("Pipeline returned non-dict result.")
    return cast(PipelineState, result)
