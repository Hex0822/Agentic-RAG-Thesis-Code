"""LLM reasoning over top-k evidence chunks."""

import json
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """
You are a reasoning agent in a multi-step fact-checking retrieval system.

Your task is to analyze the current evidence and determine:
1) What information has already been established.
2) What key information is still missing.
3) What search queries should be generated to retrieve the missing information.

The system performs iterative retrieval. Your output will guide the next search round.

Input includes grouped evidence.
Each evidence group contains:
- the query used to retrieve it
- top-k evidence chunks (title + snippet)

Reasoning Instructions:
1. Carefully read the original claim.
2. Review the previous round summary and missing information (if available) to understand the current reasoning state.
3. Examine the grouped evidence and determine what new facts can be reliably established.
4. Produce a concise summary of what is currently known, combining:
   - previously established information
   - new evidence from this round.
5. Identify what important information is still missing that is necessary to verify or refute the claim.
6. If the current evidence is already sufficient to logically determine the claim (supported or refuted), mark that no further search is needed.
7. Otherwise, generate targeted search queries to retrieve evidence for the missing information.

Query Generation Rules:
- Generate queries only for information that is still missing.
- Queries must be concise and retrieval-friendly.
- Prefer factual or comparison-oriented queries.
- Avoid vague or conversational queries.
- Each missing information item should have 1–3 search queries.
- Total queries should not exceed 6.

Consistency Rules:
- `search_needed = false` -> `missing_information` must be an empty list `[]`.
- `search_needed = true` -> `missing_information` must contain at least 1 item.
- If `search_needed = true`, each `missing_information` item must include at least 1 query.
- The `reasoning_note` must be consistent with `search_needed` and `missing_information`.

Rules:
- Base your judgment only on provided evidence chunks and previous round context.

{format_instructions}
"""

HUMAN_PROMPT = """Original claim:
{original_claim}

Relationship type:
{relationship_type}

Previous round summary:
{previous_round_summary}

Previous round missing information:
{previous_round_missing_information}

Evidence contexts:
{subclaim_contexts_json}
"""


class ReasoningOutput(BaseModel):
    class MissingInformationItem(BaseModel):
        question: str = Field(min_length=1)
        importance: Literal["critical", "helpful"] = Field(default="critical")
        queries: list[str] = Field(default_factory=list)

    known_information: list[str] = Field(
        default_factory=list,
        description="Information already established from previous and current rounds.",
    )
    missing_information: list[MissingInformationItem] = Field(
        default_factory=list,
        description="Missing information items with question, importance, and queries.",
    )
    search_needed: bool = Field(
        description="Whether another retrieval round is needed."
    )
    reasoning_note: str = Field(
        min_length=1,
        description="Brief explanation of whether more evidence is needed and why.",
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
        subclaim_contexts: list[dict[str, Any]],
        previous_round_summary: str = "",
        previous_round_missing_information: str = "",
    ) -> ReasoningOutput:
        raw = self._chain.invoke(
            {
                "original_claim": original_claim.strip(),
                "relationship_type": relationship_type.strip(),
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
        result.known_information = [k.strip() for k in result.known_information if k and k.strip()]

        cleaned_missing: list[ReasoningOutput.MissingInformationItem] = []
        total_queries = 0
        for item in result.missing_information:
            question = item.question.strip()
            if not question:
                continue

            queries = [q.strip() for q in item.queries if q and q.strip()][:3]
            remaining = max(0, 6 - total_queries)
            if remaining == 0:
                break
            queries = queries[:remaining]
            if not queries:
                continue
            total_queries += len(queries)

            cleaned_missing.append(
                ReasoningOutput.MissingInformationItem(
                    question=question,
                    importance=item.importance,
                    queries=queries,
                )
            )

        # Enforce strict consistency from structured fields only.
        # search_needed is derived from whether valid missing items remain after cleaning.
        result.missing_information = cleaned_missing
        result.search_needed = len(cleaned_missing) > 0

        note = result.reasoning_note.strip()
        if result.search_needed:
            suffix = "Final decision: additional retrieval is needed."
            result.reasoning_note = f"{note} {suffix}".strip() if note else suffix
        else:
            suffix = "Final decision: no additional retrieval is needed."
            result.reasoning_note = f"{note} {suffix}".strip() if note else suffix

        return result
