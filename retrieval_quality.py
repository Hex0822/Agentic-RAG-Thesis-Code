import json
import logging
from typing import Any, Dict, List, Literal, TypedDict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .evidence_packaging import LLMEvidence

SYSTEM_PROMPT = """
You are the Retrieval Controller and Lead Investigative Agent in an advanced OSINT RAG pipeline.

You receive:
1) the original (unsplit) claim,
2) one atomic subclaim,
3) current search keywords,
4) previous keywords used in earlier rounds,
5) current round number and max rounds,
6) top-k evidence snippets (title, url, evidence text).

Your objective is to make an autonomous, logically sound decision for THIS subclaim based on the provided evidence, or dynamically strategize the next phase of investigation.

### EVALUATION & INVESTIGATIVE FRAMEWORK:
### EVALUATION & INVESTIGATIVE FRAMEWORK:

1. Semantic Matching (Conceptual Truth):
Evaluate the core meaning, not exact literal matches (e.g., "comprehensive regulations" substantively supports "strict regulations"). Do NOT declare INCONCLUSIVE solely due to missing specific adjectives if the substantive truth is verified.

2. Information Void & Strategic Pivoting:
If direct evidence is missing, DO NOT rely on internal knowledge. Formulate hypotheses STRICTLY from peripheral clues in the retrieved snippets. Use REFINE_KEYWORDS to actively target suspected factual/temporal conflicts rather than blindly repeating the original claim.

3. Absence of Expected Evidence & Smart Early Stopping:
If a claim alleges a massive public/corporate event but initial searches yield zero confirmation of the CORE ACTION (even if entities are mentioned):
- Broaden ONCE: You may use REFINE_KEYWORDS for exactly ONE round with broader terms.
- Kill Switch: If the information void persists, DO NOT waste remaining rounds. Immediately choose FINALIZE_SUBCLAIM with REFUTED. In your rationale, state that the absolute lack of coverage for such a major event is definitive proof of fabrication.

### OUTPUT LOGIC & RULES:
- Reason STRICTLY from provided evidence only.
- Do NOT output numeric scores.
- provisional_conclusion MUST be one of:
  - SUPPORTED
  - REFUTED
  - INCONCLUSIVE
- next_action MUST be one of:
  - FINALIZE_SUBCLAIM: Evidence (direct or logically mutually exclusive) is sufficient to declare SUPPORTED or REFUTED.
  - REFINE_KEYWORDS: Evidence is insufficient/noisy. You formed a new hypothesis and need to test it. (Provisional conclusion should typically be INCONCLUSIVE here).
  - STOP_SUBCLAIM: The premise is entirely obscure yielding pure noise, no logical hypothesis can be formed, OR you have reached the maximum search rounds.
- Provide a thorough explanation in evidence_assessment and/or rationale for why you chose the conclusion and next_action. Be explicit about the evidence gaps, conflicts, or relevance issues you observed.

### CRITICAL CONSTRAINTS:
- Loop Prevention: If `current round` >= `max rounds`, your next_action MUST be FINALIZE_SUBCLAIM or STOP_SUBCLAIM. You CANNOT choose REFINE_KEYWORDS.
- Keyword Novelty: If next_action is REFINE_KEYWORDS, provide a concise keyword_update_plan (3-5 new queries). These queries MUST target your new hypothesis and MUST be strictly different from the `previous keywords`.
- Retrieval Intent: If next_action is REFINE_KEYWORDS, also provide retrieval_intent as a single sentence describing the investigation intent for reranking. Otherwise set it to an empty string.

Return strict JSON only.
{format_instructions}
{format_instructions}
"""


class KeywordUpdatePlanModel(BaseModel):
    keep_keywords: List[str] = Field(default_factory=list)
    drop_keywords: List[str] = Field(default_factory=list)
    new_keywords: List[str] = Field(default_factory=list)


class SubclaimAgentOutputModel(BaseModel):
    evidence_assessment: str
    conclusion: Literal["SUPPORTED", "REFUTED", "INCONCLUSIVE"]
    next_action: Literal["FINALIZE_SUBCLAIM", "REFINE_KEYWORDS", "STOP_SUBCLAIM"]
    keyword_update_plan: KeywordUpdatePlanModel = Field(default_factory=KeywordUpdatePlanModel)
    retrieval_intent: str = ""
    rationale: str = ""


class KeywordUpdatePlan(TypedDict):
    keep_keywords: List[str]
    drop_keywords: List[str]
    new_keywords: List[str]


class SubclaimAgentDecision(TypedDict):
    subclaim: str
    evidence_assessment: str
    conclusion: str
    next_action: str
    keyword_update_plan: KeywordUpdatePlan
    retrieval_intent: str
    rationale: str


class RetrievalQualityAgent:
    """Agentic controller that decides retrieval next-step for each subclaim."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self._logger = logging.getLogger(__name__)
        self._parser = JsonOutputParser(pydantic_object=SubclaimAgentOutputModel)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "user",
                    "Original claim: {original_claim}\n"
                    "Subclaim: {subclaim}\n"
                    "Round: {round_idx}/{max_rounds}\n"
                    "Current keywords: {keywords_json}\n"
                    "Previous keywords: {keyword_history_json}\n"
                    "Top-k evidence: {evidence_json}\n",
                ),
            ]
        )
        self._format_instructions = self._parser.get_format_instructions()
        self._chain = prompt | llm | self._parser

    def evaluate(
        self,
        original_claim: str,
        subclaim: str,
        current_keywords: List[str],
        evidence_items: List[LLMEvidence],
        round_idx: int,
        max_rounds: int,
        keyword_history: List[List[str]],
    ) -> SubclaimAgentDecision:
        evidence_payload = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "evidence": item.get("evidence", ""),
            }
            for item in evidence_items
        ]
        try:
            response = self._chain.invoke(
                {
                    "original_claim": original_claim,
                    "subclaim": subclaim,
                    "round_idx": round_idx,
                    "max_rounds": max_rounds,
                    "keywords_json": json.dumps(current_keywords, ensure_ascii=True),
                    "keyword_history_json": json.dumps(keyword_history, ensure_ascii=True),
                    "evidence_json": json.dumps(evidence_payload, ensure_ascii=True),
                    "format_instructions": self._format_instructions,
                }
            )
            self._logger.debug(
                "Retrieval quality LLM response (subclaim=%s, round=%s/%s): %s",
                subclaim,
                round_idx,
                max_rounds,
                _serialize_response(response),
            )
            return _normalize_decision(response, subclaim, current_keywords)
        except Exception as exc:
            self._logger.warning(
                "Retrieval agent failed for subclaim '%s'. Using fallback decision. Error: %s",
                subclaim,
                exc,
            )
            return _fallback_decision(subclaim, current_keywords)


def _normalize_decision(
    response: Any,
    subclaim: str,
    current_keywords: List[str],
) -> SubclaimAgentDecision:
    data: Dict[str, Any]
    if isinstance(response, BaseModel):
        if hasattr(response, "model_dump"):
            data = response.model_dump()  # type: ignore[attr-defined]
        else:
            data = response.dict()  # type: ignore[attr-defined]
    elif isinstance(response, dict):
        data = response
    else:
        return _fallback_decision(subclaim, current_keywords)

    raw_conclusion = str(data.get("conclusion", "INCONCLUSIVE")).upper().strip()
    if raw_conclusion not in {"SUPPORTED", "REFUTED", "INCONCLUSIVE"}:
        raw_conclusion = "INCONCLUSIVE"

    raw_next_action = str(data.get("next_action", "REFINE_KEYWORDS")).upper().strip()
    if raw_next_action not in {"FINALIZE_SUBCLAIM", "REFINE_KEYWORDS", "STOP_SUBCLAIM"}:
        raw_next_action = "REFINE_KEYWORDS"

    plan_data = data.get("keyword_update_plan", {})
    keep = _normalize_string_list(plan_data.get("keep_keywords", []))
    drop = _normalize_string_list(plan_data.get("drop_keywords", []))
    new = _normalize_string_list(plan_data.get("new_keywords", []))

    if raw_next_action == "REFINE_KEYWORDS" and not keep and not new:
        keep = current_keywords[:2]
        new = [subclaim]

    retrieval_intent = str(data.get("retrieval_intent", "")).strip()
    if raw_next_action == "REFINE_KEYWORDS" and not retrieval_intent:
        retrieval_intent = subclaim

    if raw_next_action != "REFINE_KEYWORDS":
        keep = []
        drop = []
        new = []
        retrieval_intent = ""

    return {
        "subclaim": subclaim,
        "evidence_assessment": str(data.get("evidence_assessment", "")).strip(),
        "conclusion": raw_conclusion,
        "next_action": raw_next_action,
        "keyword_update_plan": {
            "keep_keywords": keep,
            "drop_keywords": drop,
            "new_keywords": new,
        },
        "retrieval_intent": retrieval_intent,
        "rationale": str(data.get("rationale", "")).strip(),
    }


def _fallback_decision(subclaim: str, current_keywords: List[str]) -> SubclaimAgentDecision:
    return {
        "subclaim": subclaim,
        "evidence_assessment": "Agent fallback: failed to evaluate retrieval quality.",
        "conclusion": "INCONCLUSIVE",
        "next_action": "REFINE_KEYWORDS",
        "keyword_update_plan": {
            "keep_keywords": current_keywords[:2],
            "drop_keywords": [],
            "new_keywords": [subclaim],
        },
        "retrieval_intent": subclaim,
        "rationale": "Fallback decision due to agent invoke/parse error.",
    }


def _normalize_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    normalized: List[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


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
