"""Single-node pipeline: claim_analysis."""

from typing import Any, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from claim_analysis import ClaimAnalyzer
from config import create_claim_analysis_llm

CLAIM_ANALYSIS_MODULE = "claim_analysis"


class PipelineState(TypedDict, total=False):
    original_claim: str
    relationship_type: str
    sub_claims: list[str]
    classification_basis: str


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


def build_pipeline(
    claim_analysis_llm: BaseChatModel | None = None,
):
    """Build one-node graph for claim analysis."""

    llm = claim_analysis_llm or create_claim_analysis_llm()
    analyzer = ClaimAnalyzer(llm)
    graph = StateGraph(PipelineState)

    graph.add_node("claim_analysis", _claim_analysis_node(analyzer))
    graph.set_entry_point("claim_analysis")
    graph.add_edge("claim_analysis", END)

    return graph.compile()


def run_pipeline(
    original_claim: str,
    claim_analysis_llm: BaseChatModel | None = None,
) -> PipelineState:
    """Run the pipeline synchronously and return its final state."""

    claim = original_claim.strip()
    if not claim:
        raise ValueError("original_claim must not be empty.")

    app = build_pipeline(
        claim_analysis_llm=claim_analysis_llm,
    )
    result = app.invoke({"original_claim": claim})
    if not isinstance(result, dict):
        raise TypeError("Pipeline returned non-dict result.")
    return cast(PipelineState, result)
