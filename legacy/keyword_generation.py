import json
import logging
from typing import Any, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """
You are an expert OSINT Search Query Generator. 
Your ONLY task is to translate the provided [Atomic Claim] into highly effective, keyword-based search queries. Do NOT analyze, correct, or debunk the claim yourself. Your job is strictly to retrieve diverse context so the downstream reasoning agent has material to analyze.

CRITICAL RULES:
1. **Strict Entity Extraction (Zero Knowledge Leakage)**: You are a blind text extractor. You MUST extract the EXACT entities precisely as they appear. NEVER auto-correct, substitute, or infer entities based on your internal knowledge, even if the claim contains obvious factual errors (e.g., mismatched pairs).
2. **Precision Quoting**: Wrap specific Named Entities (People, Organizations, Specific Products) in double quotes (e.g., "Meta", "Libra"). DO NOT wrap verbs, dates, descriptive phrases, or long sentences in quotes. Allow the search engine to semantic-match the actions.
3. **Dynamic Dimensional Decomposition**: To maximize recall, decompose the claim into 2 to 4 distinct search dimensions. Do not use rigid templates. Adapt the dimensions organically based on the nature of the claim (e.g., relationship check, specific action, temporal status, statistical baseline, or industry background).

EXAMPLES OF DIMENSIONAL DECOMPOSITION (Do NOT copy these formats blindly, learn the underlying logic):

Claim: "Donald Trump officially bought Twitter in 2026."
Queries:
- "Donald Trump" "Twitter" acquisition OR buyout
- "Donald Trump" "Twitter" relationship OR interactions
- "Twitter" ownership status 2026
- "Donald Trump" social media business ventures

Claim: "The World Health Organization declared that drinking coffee cures lung cancer."
Queries:
- "World Health Organization" coffee "lung cancer" cure OR treatment
- "World Health Organization" official statement coffee health benefits
- coffee "lung cancer" medical consensus OR study

Claim: "Apple announced a new car named the iDrive in Berlin yesterday."
Queries:
- "Apple" "iDrive" car announcement "Berlin"
- "Apple" automotive project vehicle development
- "Apple" press release "Berlin"

{format_instructions}
"""


class KeywordsOutput(BaseModel):
    keywords: List[str] = Field(
        description="A list of short search keywords or phrases."
    )


class KeywordsGenerator:
    """Step 2: generate search keywords from a single claim."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self._logger = logging.getLogger(__name__)
        self._parser = JsonOutputParser(pydantic_object=KeywordsOutput)
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("user", "Claim: {input_claim}")]
        )
        self._format_instructions = self._parser.get_format_instructions()
        self._chain = prompt | llm | self._parser

    def generate(self, claim: str) -> List[str]:
        response = self._chain.invoke(
            {
                "input_claim": claim,
                "format_instructions": self._format_instructions,
            }
        )
        self._logger.debug(
            "Keyword generation LLM response (claim=%s): %s",
            claim,
            _serialize_response(response),
        )
        return _extract_keywords(response)


def _extract_keywords(response: Any) -> List[str]:
    if isinstance(response, KeywordsOutput):
        return response.keywords
    if isinstance(response, dict):
        keywords = response.get("keywords")
        if isinstance(keywords, list):
            return keywords
    raise ValueError("Parser response missing 'keywords'")


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
