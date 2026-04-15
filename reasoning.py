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
4) A final decision label for the claim status.

The system performs iterative retrieval. Your output will guide the next search round.

Input includes grouped evidence.
Each evidence group contains:
- the query used to retrieve it
- top-k evidence chunks (title + snippet)

Reasoning Instructions:
1. Carefully read the original claim.
2. Review previous rounds' known_information and missing_information (if available) to understand the current reasoning state.
3. Compare previous rounds' missing_information with current-round evidence:
   - identify which missing items are now resolved,
   - which remain unresolved,
   - and whether retrieval is repeating similar unresolved targets.
4. Examine the grouped evidence and determine what new facts can be reliably established.
5. Produce a concise summary of what is currently known, combining:
   - previously established information
   - new evidence from this round.
6. Identify what important information is still missing that is necessary to verify or refute the claim.
7. If the current evidence is already sufficient to logically determine the claim (supported or refuted), mark that no further search is needed.
8. Otherwise, generate targeted search queries to retrieve evidence for the missing information.

Search Value Assessment:
- Explicitly assess whether another retrieval round is likely to add meaningful new evidence.
- Do NOT continue searching just to consume max rounds.
- If high-quality queries have been tried across multiple rounds and still no direct support appears, treat this as a strong unsupported signal.
- If retrieved evidence repeatedly corrects or conflicts with the claim wording/value, treat this as a claim-correction signal rather than an endless-search signal.
- If missing questions do not converge (repeatedly asking almost the same unknown), stop looping and summarize what can already be concluded.
- If retrieval strategy still appears weak (poor query wording, weak source scope, low coverage), you may request another round with improved targeted queries.

Query Generation Rules:
- Generate queries only for information that is still missing.
- Queries must target the same missing variable represented by `question`.
- For each missing information item, generate at least 2 and at most 4 queries.
- The first two queries must be:
  1) Factoid query (WH-style factual question)
  2) Relation query (short entity-relation phrase)
- You may optionally add:
  3) Direct Statement query
  4) Verification query
- Queries must be concise and retrieval-friendly.
- Prefer factual or comparison-oriented queries.
- Avoid vague or conversational queries.
- Do not generate queries that are only superficial rephrasings of each other.
- Prefer queries likely to match web page titles, knowledge pages, factual sentences, or snippets.
- If multiple missing items exist, prioritize critical items and keep total queries across all items <= 6.

Missing Information Format (for reranker target):
- Each missing item must include `question` as a short, neutral factual question.
- `question` should be directly comparable to evidence sentences for reranking.
- Keep `question` concise (preferably 6-18 words), specific, and variable-focused.
- Avoid vague prefixes such as "Can you", "Please explain", or "How to".
- Keep one missing variable per `question`.

Consistency Rules:
- `search_needed = false` -> `missing_information` must be an empty list `[]`.
- `search_needed = true` -> `missing_information` must contain at least 1 item.
- If `search_needed = true`, each `missing_information` item must include at least 2 queries.
- The `reasoning_note` must be consistent with `search_needed` and `missing_information`.
- `label` must be one of: "Supported", "Refuted", "Not Enough Evidence".
- If `search_needed = true`, then `label` must be "Not Enough Evidence".
- In `reasoning_note`, explicitly explain whether you are stopping due to:
  - enough evidence, or
  - strong unsupported signal, or
  - claim-correction signal, or
  - non-convergent missing information.

Rules:
- Base your judgment only on provided evidence chunks and previous round context.

{format_instructions}
"""

HUMAN_PROMPT = """Original claim:
{original_claim}

Relationship type:
{relationship_type}

Previous rounds context (known_information + missing_information):
{previous_rounds_knowledge_json}

Evidence contexts:
{subclaim_contexts_json}
"""


class ReasoningOutput(BaseModel):
    class MissingInformationItem(BaseModel):
        question: str = Field(
            min_length=1,
            description="Short neutral factual question for reranker comparison.",
        )
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
    label: Literal["Supported", "Refuted", "Not Enough Evidence"] = Field(
        description="Final decision label for current reasoning status."
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
        self._chain = (self._prompt | llm | self._parser).with_config(
            {"run_name": "reasoning_chain", "tags": ["stage:reasoning"]}
        )

    def reason(
        self,
        original_claim: str,
        relationship_type: str,
        subclaim_contexts: list[dict[str, Any]],
        previous_rounds_knowledge: list[dict[str, Any]] | None = None,
    ) -> ReasoningOutput:
        raw = self._chain.invoke(
            {
                "original_claim": original_claim.strip(),
                "relationship_type": relationship_type.strip(),
                "previous_rounds_knowledge_json": json.dumps(
                    previous_rounds_knowledge or [], ensure_ascii=False, indent=2
                ),
                "subclaim_contexts_json": json.dumps(
                    subclaim_contexts, ensure_ascii=False, indent=2
                ),
            }
        )
        return ReasoningOutput.model_validate(raw)
