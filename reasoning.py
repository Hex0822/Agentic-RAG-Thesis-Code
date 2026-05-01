"""LLM reasoning over top-k evidence chunks."""

import json
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
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

Rules:
- Base your judgment only on provided evidence chunks and previous round context.

Reasoning Workflow:
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

Decision and Search Value Assessment:
- Explicitly assess whether another retrieval round is likely to add meaningful new evidence.
- Do NOT continue searching just to consume max rounds.
- If retrieval strategy still appears weak (poor query wording, weak source scope, low coverage), you may request another round with improved targeted queries.
- If missing questions do not converge (repeatedly asking almost the same unknown), stop looping and summarize what can already be concluded.

Unsupported and Refuted Signals:
- If high-quality queries have been tried across multiple rounds and still no direct support appears, treat this as a strong unsupported signal.
- However, absence of support alone usually leads to Not Enough Evidence, not Refuted.
- You may treat repeated absence of support as evidence toward Refuted only when the claim describes a high-salience, externally observable event that would normally leave reliable records if true, such as mass deaths, official actions, court rulings, public statements, major policy changes, or widely reported incidents.
- For low-salience, private, vague, or hard-to-observe claims, absence of support should remain Not Enough Evidence unless explicit contradictory evidence, official denial, or direct fact-check correction is available.
- A claim-correction signal supports Refuted only if the correction changes the core truth of the claim.
- Minor rounding differences, approximate wording, paraphrases, rhetorical degree modifiers, or non-central details should not by themselves trigger Refuted.
- Before labeling Refuted, ask whether the correction materially changes the claim's main public meaning.

Numeric Matching Rules:
- Public claims often use rounded or approximate numbers.
- Do not refute a claim solely because of small rounding differences.
- Treat values as compatible when the difference is small and does not change the main conclusion.
- For approximate public-statements, differences under about 5-10% are usually not decisive.
- But if the number is central and the difference is large enough to change the comparison or meaning, treat it as a serious mismatch.
- Always normalize units, scale, currency, and time period before judging numeric conflict.

Semantic Equivalence Rules:
- Evidence does not need to use the same wording as the claim.
- If official or reliable evidence supports the practical meaning, direct implication, or natural paraphrase of the claim, treat it as support.
- Do not downgrade to Not Enough Evidence solely because the evidence is more specific than the claim or uses a different level of abstraction.
- For purpose, objective, or reason-for-launch claims, evidence may support the claim by describing concrete goals, services, mechanisms, or intended outcomes that directly serve the broader stated purpose.
- A broad purpose claim can be Supported when reliable evidence establishes a more specific objective that naturally entails that purpose.
- Do not require the exact phrase “was launched to...” or identical purpose wording when the evidence clearly describes what the program, policy, or action was designed to achieve.
- Do not treat paraphrase differences, rhetorical wording, or ordinary-language summaries as contradictions.
- Be careful with idiomatic or rhetorical expressions; interpret them in ordinary context unless the claim clearly requires a legal, technical, or statistical meaning.
- Only treat wording differences as important if they change the core meaning, scope, causal relation, population, time period, or measurable outcome of the claim.

Examples:
- Evidence that a job-training program provides vocational courses and employment placement can support a claim that it was launched to improve employability.
- Evidence that a vaccination campaign provides free vaccines to children can support a claim that it was launched to protect children’s health.
- Evidence that a road-safety policy reduces speed limits and improves pedestrian crossings can support a claim that it was introduced to make roads safer.
- Evidence that a scholarship scheme funds tuition for low-income students can support a claim that it was created to expand access to education.

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

Missing Information Format:
- Each missing item must include `question` as a short, neutral factual question.
- `question` should be directly comparable to evidence sentences for reranking.
- Keep `question` concise (preferably 6-18 words), specific, and variable-focused.
- Avoid vague prefixes such as "Can you", "Please explain", or "How to".
- Keep one missing variable per `question`.

Output Consistency Rules:
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
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
        )
        structured_llm = llm.with_structured_output(
            ReasoningOutput,
            method="json_schema",
        )
        self._chain = (self._prompt | structured_llm).with_config(
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
