"""Final label decision for fully resolved nested claims."""

import json
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """
You are a final decision agent for NESTED claims in a fact-checking pipeline.

Task:
- Use the original claim and resolved nested variables to determine a final label.
- Do not request additional search.

Label choices:
- "Supported"
- "Refuted"
- "Not Enough Evidence"

Rules:
1) Base the decision only on provided inputs.
2) If resolved variables are sufficient and consistent with the claim -> Supported.
3) If resolved variables clearly contradict the claim -> Refuted.
4) If key variable values are missing/unknown/ambiguous -> Not Enough Evidence.
5) Keep the explanation concise (1-2 short sentences).

{format_instructions}
"""

HUMAN_PROMPT = """original_claim: {original_claim}
nested_plan_json:
{nested_plan_json}
resolved_variable_values_json:
{resolved_variable_values_json}
"""


class NestedDecisionOutput(BaseModel):
    label: Literal["Supported", "Refuted", "Not Enough Evidence"] = Field(
        description="Final label for the nested claim."
    )
    reasoning_note: str = Field(
        min_length=1,
        description="Brief explanation for the final label.",
    )


class NestedDecisionEngine:
    def __init__(self, llm: BaseChatModel) -> None:
        self._parser = JsonOutputParser(pydantic_object=NestedDecisionOutput)
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
        ).partial(format_instructions=self._parser.get_format_instructions())
        self._chain = (self._prompt | llm | self._parser).with_config(
            {"run_name": "nested_decision_chain", "tags": ["stage:nested_decision"]}
        )

    def decide(
        self,
        original_claim: str,
        nested_plan: dict[str, Any],
        resolved_variable_values: dict[str, str],
    ) -> NestedDecisionOutput:
        raw = self._chain.invoke(
            {
                "original_claim": original_claim.strip(),
                "nested_plan_json": json.dumps(nested_plan, ensure_ascii=False, indent=2),
                "resolved_variable_values_json": json.dumps(
                    resolved_variable_values, ensure_ascii=False, indent=2
                ),
            }
        )
        return NestedDecisionOutput.model_validate(raw)
