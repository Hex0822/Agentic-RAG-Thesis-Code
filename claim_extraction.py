import json
import logging
from typing import Any, List, Literal, TypedDict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
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


class ClaimDecomposition(BaseModel):
    relationship_type: Literal["NESTED", "CAUSAL", "ATOMIC"] = Field(
        description="The relationship type of the original claim."
    )
    classification_basis: str = Field(
        default="",
        description="1-2 short sentences explaining why the relationship type was chosen.",
    )
    sub_claims: List[str] = Field(
        description="A list of sub-claims produced based on the relationship type."
    )


class ClaimExtractionResult(TypedDict):
    relationship_type: str
    classification_basis: str
    sub_claims: List[str]


class ClaimExtractor:
    """Step 1: split an input claim into atomic sub-claims."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self._logger = logging.getLogger(__name__)
        self._parser = JsonOutputParser(pydantic_object=ClaimDecomposition)
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("user", "Claim: {input_claim}")]
        )
        self._format_instructions = self._parser.get_format_instructions()
        self._chain = prompt | llm | self._parser

    def extract(self, claim: str) -> ClaimExtractionResult:
        response = self._chain.invoke(
            {
                "input_claim": claim,
                "format_instructions": self._format_instructions,
            }
        )
        self._logger.debug(
            "Claim extraction LLM response (claim=%s): %s",
            claim,
            _serialize_response(response),
        )
        return _extract_result(response, claim)


def _extract_result(response: Any, original_claim: str) -> ClaimExtractionResult:
    if isinstance(response, ClaimDecomposition):
        data = response.model_dump()
    elif isinstance(response, dict):
        data = response
    else:
        return {
            "relationship_type": "ATOMIC",
            "classification_basis": "Fallback: unparsed response.",
            "sub_claims": [original_claim],
        }

    raw_type = str(data.get("relationship_type", "ATOMIC")).upper().strip()
    if raw_type not in {"NESTED", "CAUSAL", "ATOMIC"}:
        raw_type = "ATOMIC"

    basis = str(data.get("classification_basis", "")).strip()
    sub_claims = data.get("sub_claims")
    if isinstance(sub_claims, list):
        normalized = [str(item).strip() for item in sub_claims if str(item).strip()]
    else:
        normalized = []

    if raw_type == "NESTED":
        normalized = [original_claim]
        if not basis:
            basis = "Nested dependency detected; returning original claim without splitting."
    if not normalized:
        normalized = [original_claim]
        if not basis:
            basis = "No sub-claims produced; falling back to original claim."
    return {
        "relationship_type": raw_type,
        "classification_basis": basis,
        "sub_claims": normalized,
    }


def _serialize_response(response: Any) -> str:
    if isinstance(response, BaseModel):
        if hasattr(response, "model_dump"):
            data = response.model_dump()  # type: ignore[attr-defined]
        else:
            data = response.dict()  # type: ignore[attr-defined]
    elif isinstance(response, dict):
        data = response
    else:
        return str(response)
    try:
        return json.dumps(data, ensure_ascii=False)
    except TypeError:
        return str(data)
