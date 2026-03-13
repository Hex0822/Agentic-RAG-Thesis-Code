"""Claim relationship classification for downstream agentic RAG."""

from typing import Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """
You are an expert fact-checker and logic analyst.
Your task is to classify the relationship type of the original claim and output sub-claims accordingly.

RELATIONSHIP TYPES:
1) NESTED: The claim contains an embedded dependency (e.g., "X's spouse's birthplace ...").
   - Output relationship_type = "NESTED"
   - Do NOT split the claim. sub_claims must be exactly [original_claim].
2) CAUSAL: The claim expresses a causal/attribution relationship (e.g., "A caused B", "because of A, B changed").
   - Output relationship_type = "CAUSAL"
   - Split into independent sub-claims for each factual component, but do NOT include the causal statement itself.
3) ATOMIC: The claim contains multiple parallel, independent facts.
   - Output relationship_type = "ATOMIC"
   - Split into atomic sub-claims as usual.

STRICT RULES:
1. Resolve pronouns. If a clause says "he did X", rewrite as "Elon Musk did X".
2. If a specific year/date is mentioned, ensure it attaches to all relevant sub-claims if grammatically implied.
3. If relationship_type is NESTED, always return sub_claims = [original_claim] even if you can split it.
4. Provide classification_basis as 1–2 short sentences explaining why the relationship_type was chosen.

EXAMPLES:
Input: "Microsoft CEO's wife's birthplace is Yosemite."
Output: {{"relationship_type": "NESTED", "sub_claims": ["Microsoft CEO's wife's birthplace is Yosemite."], "classification_basis": "Nested dependency via possessive relationship (CEO's wife), so keep the original claim."}}

Input: "A happened in 2020, which caused B to change policy in 2021."
Output: {{"relationship_type": "CAUSAL", "sub_claims": ["A happened in 2020.", "B changed policy in 2021."], "classification_basis": "Explicit causal trigger ('caused') indicates a causal relation, so split into factual components only."}}

Input: "Elon Musk founded SpaceX in 2002 and later acquired Twitter."
Output: {{"relationship_type": "ATOMIC", "sub_claims": ["Elon Musk founded SpaceX in 2002.", "Elon Musk acquired Twitter."], "classification_basis": "Parallel independent facts joined by 'and' indicate an atomic split."}}

{format_instructions}
"""

HUMAN_PROMPT = """Original claim:
{original_claim}
"""


class ClaimAnalysisOutput(BaseModel):
    """Structured output for claim relationship classification."""

    relationship_type: Literal["NESTED", "CAUSAL", "ATOMIC"] = Field(
        description="Top-level relationship type for the original claim."
    )
    sub_claims: list[str] = Field(
        min_length=1,
        description="Output sub-claims after decomposition under the selected relationship type.",
    )
    classification_basis: str = Field(
        min_length=1,
        description="1-2 short sentences explaining the chosen relationship_type.",
    )


class ClaimAnalyzer:
    """LangChain wrapper that classifies claim relationship and emits sub-claims."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._parser = JsonOutputParser(pydantic_object=ClaimAnalysisOutput)
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
        ).partial(format_instructions=self._parser.get_format_instructions())
        self._chain = self._prompt | llm | self._parser

    def analyze(self, original_claim: str) -> ClaimAnalysisOutput:
        claim = original_claim.strip()
        if not claim:
            raise ValueError("original_claim must not be empty.")
        raw = self._chain.invoke({"original_claim": claim})
        return ClaimAnalysisOutput.model_validate(raw)

    async def aanalyze(self, original_claim: str) -> ClaimAnalysisOutput:
        claim = original_claim.strip()
        if not claim:
            raise ValueError("original_claim must not be empty.")
        raw = await self._chain.ainvoke({"original_claim": claim})
        return ClaimAnalysisOutput.model_validate(raw)


class ClaimAnalysisState(TypedDict, total=False):
    original_claim: str
    relationship_type: str
    sub_claims: list[str]
    classification_basis: str


def build_claim_analysis_graph(llm: BaseChatModel):
    """Build a single-node LangGraph for claim analysis."""

    analyzer = ClaimAnalyzer(llm)
    graph = StateGraph(ClaimAnalysisState)

    def _claim_analysis_node(state: ClaimAnalysisState) -> dict[str, Any]:
        result = analyzer.analyze(state.get("original_claim", ""))
        return {
            "relationship_type": result.relationship_type,
            "sub_claims": result.sub_claims,
            "classification_basis": result.classification_basis,
        }

    graph.add_node("claim_analysis", _claim_analysis_node)
    graph.set_entry_point("claim_analysis")
    graph.add_edge("claim_analysis", END)
    return graph.compile()


def analyze_claim(original_claim: str, llm: BaseChatModel) -> ClaimAnalysisOutput:
    """Convenience function for non-graph usage."""

    return ClaimAnalyzer(llm).analyze(original_claim)
