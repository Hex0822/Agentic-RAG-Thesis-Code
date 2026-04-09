"""Quick reasoning for resolving a nested variable from top-k evidence."""

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field

SYSTEM_PROMPT = """
You are a quick reasoning agent for nested-variable resolution in a fact-checking system.

Goal:
- Infer the best value for ONE target variable from top-k evidence chunks.
- Do not verify the whole claim.
- Do not decompose into new variables.

Input:
- variable_id
- variable_description
- top_k_evidence_chunks

Rules:
1) Base your answer only on the provided evidence chunks.
2) Keep explanation concise and concrete (1-2 short sentences).
3) If evidence is insufficient or conflicting, set variable to "UNKNOWN".
4) Output JSON only.

{format_instructions}
"""

HUMAN_PROMPT = """variable_id: {variable_id}
variable_description: {variable_description}
top_k_evidence_chunks_json:
{top_k_evidence_chunks_json}
"""


class QuickReasoningOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    brief_explain: str = Field(
        alias="biref explain",
        min_length=1,
        description="Short reason for why this variable value is inferred.",
    )
    variable: str = Field(
        min_length=1,
        description="Resolved variable value, or UNKNOWN if unresolved.",
    )


class QuickReasoningEngine:
    def __init__(self, llm: BaseChatModel) -> None:
        self._parser = JsonOutputParser(pydantic_object=QuickReasoningOutput)
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
        ).partial(format_instructions=self._parser.get_format_instructions())
        self._chain = self._prompt | llm | self._parser

    def infer_variable(
        self,
        variable_id: str,
        variable_description: str,
        top_k_evidence_chunks: list[dict[str, Any]],
    ) -> QuickReasoningOutput:
        raw = self._chain.invoke(
            {
                "variable_id": variable_id.strip(),
                "variable_description": variable_description.strip(),
                "top_k_evidence_chunks_json": json.dumps(
                    top_k_evidence_chunks, ensure_ascii=False, indent=2
                ),
            }
        )
        return QuickReasoningOutput.model_validate(raw)
