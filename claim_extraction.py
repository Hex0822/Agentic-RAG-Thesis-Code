import json
import logging
from typing import Any, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """
You are an expert fact-checker and logic analyst.
Your task is to break down complex claims into separate, atomic, self-contained sentences.

STRICT RULES:
1. If a sentence has multiple parts (e.g., connected by "and", "but", "while"), SPLIT IT.
2. Resolve pronouns. If the second part says "he did X", change it to "Elon Musk did X".
3. NEVER return the original complex sentence as a single item.
4. If the sentence contains one time but there's two claims, make sure every subclaim contains time.
4. If a specific year/date is mentioned, ensure it attaches to ALL relevant sub-claims if grammatically implied.

EXAMPLES:
Input: "Elon Musk founded SpaceX in 2002 and later acquired Twitter."
Output: ["Elon Musk founded SpaceX in 2002.", "Elon Musk acquired Twitter."]

Input: "Trump brought Twitter company when he was president in 2022."
Output: "Trump brought Twitter company in 2022.", "Trump was president in 2022."

Input: "Apple released the iPhone 15, which features a Titanium frame."
Output: ["Apple released the iPhone 15.", "The iPhone 15 features a Titanium frame."]

Input: "The sky is blue."
Output: ["The sky is blue."]

{format_instructions}
"""


class ClaimDecomposition(BaseModel):
    sub_claims: List[str] = Field(
        description="A list of atomic, independent sub-claims extracted from the text."
    )


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

    def extract(self, claim: str) -> List[str]:
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
        return _extract_sub_claims(response)


def _extract_sub_claims(response: Any) -> List[str]:
    if isinstance(response, ClaimDecomposition):
        return response.sub_claims
    if isinstance(response, dict):
        sub_claims = response.get("sub_claims")
        if isinstance(sub_claims, list):
            return sub_claims
    raise ValueError("Parser response missing 'sub_claims'")


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
