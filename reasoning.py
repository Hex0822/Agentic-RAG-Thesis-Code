"""LLM reasoning over top-k evidence chunks."""

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """
You are a fact-checking reasoning controller.

Input:
- Original claim
- Relationship type
- Sub-claims
- Previous round summary (if available)
- Previous round missing information (if available)
- Top-k evidence chunks for each sub-claim

Task:
1) Summarize what this round has established.
2) Decide whether another retrieval round is needed.
3) If another round is needed, summarize what information is still missing.
4) If another round is needed, provide short follow-up search queries.

Rules:
- Base your judgment only on provided evidence chunks.
- If previous-round summary/missing information is provided, use it as compressed memory.
- Do not request already-resolved information unless new evidence creates conflict.
- Keep summaries concise and factual.
- If evidence is conflicting, mention the conflict.
- `suggested_queries` must be 0 to 6 short queries.
- If `need_more_search` is false, `suggested_queries` should be an empty list.

{format_instructions}
"""

HUMAN_PROMPT = """Original claim:
{original_claim}

Relationship type:
{relationship_type}

Sub-claims:
{sub_claims_json}

Previous round summary:
{previous_round_summary}

Previous round missing information:
{previous_round_missing_information}

Evidence contexts:
{subclaim_contexts_json}
"""


class ReasoningOutput(BaseModel):
    need_more_search: bool = Field(
        description="Whether another retrieval round is needed before final reasoning."
    )
    round_summary: str = Field(
        min_length=1,
        description="What this retrieval round has established so far.",
    )
    missing_information: str = Field(
        min_length=1,
        description="What evidence is still missing or uncertain.",
    )
    suggested_queries: list[str] = Field(
        default_factory=list,
        description="Follow-up search queries (0 to 6).",
    )


class ReasoningEngine:
    def __init__(self, llm: BaseChatModel) -> None:
        self._parser = JsonOutputParser(pydantic_object=ReasoningOutput)
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
        ).partial(format_instructions=self._parser.get_format_instructions())
        self._chain = self._prompt | llm | self._parser

    def reason(
        self,
        original_claim: str,
        relationship_type: str,
        sub_claims: list[str],
        subclaim_contexts: list[dict[str, Any]],
        previous_round_summary: str = "",
        previous_round_missing_information: str = "",
    ) -> ReasoningOutput:
        raw = self._chain.invoke(
            {
                "original_claim": original_claim.strip(),
                "relationship_type": relationship_type.strip(),
                "sub_claims_json": json.dumps(sub_claims, ensure_ascii=False, indent=2),
                "previous_round_summary": previous_round_summary.strip() or "N/A",
                "previous_round_missing_information": (
                    previous_round_missing_information.strip() or "N/A"
                ),
                "subclaim_contexts_json": json.dumps(
                    subclaim_contexts, ensure_ascii=False, indent=2
                ),
            }
        )
        result = ReasoningOutput.model_validate(raw)
        result.suggested_queries = [q.strip() for q in result.suggested_queries if q and q.strip()][
            :6
        ]
        if not result.need_more_search:
            result.suggested_queries = []
        return result
