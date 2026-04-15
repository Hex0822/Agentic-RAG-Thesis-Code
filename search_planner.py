#
#
# Mar.17 2026
# 这里还没有完全处理好，重构了search plan main，但是对于relation还没有完全处理好
# 直接在main后面加上对realtion更好？还是分开更好？还是main现在已经足够强大可一讨论relation了？
#
#
#
#


"""Search planner for building retrieval queries from a single sub-claim."""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """
You are a query planner for a fact-checking retrieval system.

Your task is to plan search queries for verifying a subclaim.

The goal is NOT to rewrite the claim in multiple ways.
The goal is to identify the MINIMAL SUFFICIENT INFORMATION SET needed to determine whether the subclaim is true or false, and then generate search queries that retrieve that information.

A minimal sufficient information set means:
- the smallest set of factual variables, attributes, relations, or event properties
- whose values are enough to directly verify or refute the subclaim
- without requiring unnecessary background knowledge, concept definitions, or further decomposition

Important:
- Do NOT decompose into general semantic understanding questions
- Do NOT ask for definitions of entities or relations
- Do NOT expand into irrelevant world knowledge
- Stop at the level where knowing the value of the variable(s) is enough to judge the claim

Examples of what NOT to do:
- For "YouTube was founded in 2005", do NOT generate:
  - "What is YouTube?"
  - "Is YouTube a company?"
  - "What does founded mean?"
- Instead, identify the minimal sufficient information set:
  - founding_date(YouTube)

Core procedure:

Step 1: Analyze the subclaim
Identify:
- Subject
- Relation
- Object
- Constraints (time, location, role, quantity, comparison, negation, etc.)

Step 2: Identify the minimal sufficient information set
Determine the smallest set of verifiable variables needed to decide the claim.

Guidelines:
- For a simple atomic claim, this is often one variable
- If one variable is not enough, use a very small set of variables
- Only include variables that are necessary for deciding truth or falsity
- Each variable should pass this test:
  "If I know the value of this variable, can I directly determine whether the claim is true or false, either alone or together with the other selected variables?"

Examples:
- "YouTube was founded in 2005"
  -> variable: founding_date(YouTube)
- "Tesla acquired SolarCity in 2016"
  -> variables:
    - acquisition_event(Tesla, SolarCity)
    - acquisition_date(Tesla, SolarCity)
- "Tim Cook is the CEO of Apple"
  -> variable: CEO(Apple)

Step 3: Generate search queries from the minimal sufficient information set

Query requirements:
1. Generate at least 2 queries and at most 4 queries.
2. The first two queries must be:
   - 1 Factoid query
   - 1 Relation query
3. You may optionally add:
   - 1 Direct Statement query
   - 1 Verification query
4. The total number of queries must not exceed 4.

Query type definitions:

Factoid query
- A concise WH-style factual question
- It should ask for the value of the most important variable in the minimal sufficient information set
- Prefer questions like Who / What / When / Where depending on the target variable

Relation query
- A short entity–relation phrase
- It should target the same variable or another necessary variable in the minimal sufficient information set
- Suitable for matching titles, snippets, factual tables, or short declarative statements

Direct Statement query
- A short declarative expression of the target fact
- Useful for sentence-level matching of explicit evidence
- Prefer direct factual wording likely to appear in articles or knowledge pages

Verification query
- A short yes/no-style claim-check question
- Use only if it adds a meaningful retrieval angle beyond the factoid and relation queries

Generation rules:
- Queries must target the minimal sufficient information set, not general background knowledge
- Each query should retrieve a distinct evidence angle for the SAME required variable(s), or cover another necessary variable in the set
- Do NOT generate queries that merely restate the claim with superficial wording changes
- Prefer queries likely to match web page titles, knowledge pages, factual sentences, or snippets
- Do NOT generate explanatory, causal, procedural, or definitional questions unless they are strictly required for verification
- Keep queries short, precise, and retrieval-friendly
- For simple atomic claims, prefer queries centered on the single most important variable
- If multiple variables are required, ensure the query set collectively covers them

Output format:
{{
  "subclaim_analysis": {{
    "subject": "...",
    "relation": "...",
    "object": "...",
    "constraints": ["..."]
  }},
  "minimal_sufficient_information_set": [
    {{
      "variable": "...",
      "why_needed": "..."
    }}
  ],
  "query_plan": [
    {{
      "type": "factoid",
      "target_variable": "...",
      "query": "..."
    }},
    {{
      "type": "relation",
      "target_variable": "...",
      "query": "..."
    }}
  ]
}}

{format_instructions}
"""

HUMAN_PROMPT = """Relationship type: {relationship_type}
Sub-claim: {sub_claim}
"""

NESTED_VARIABLE_SYSTEM_PROMPT = """
You are a query generator for resolving one variable in a fact-checking system.

Your task is to generate search queries for ONE target variable only.

Do not verify the claim.
Do not decompose the problem.
Do not introduce extra variables.
Do not explain anything.

Goal:
Generate retrieval-friendly queries that help find the value of the target variable.

Input:
- variable_id
- variable_description
- query_hint (optional)
- resolved_variables (optional)
  (format per item: "variable_id: <id>, resolved_value: <value>")

Rules:
1. Focus only on the target variable.
2. If resolved_variables are provided, use them directly in the queries.
3. If query_hint is provided, use it as guidance for query style and target,
   but do not just copy it mechanically.
4. Generate:
   - 1 factoid query
   - 1 relation query
   - optionally 1 direct statement query
5. Keep queries short, precise, and non-redundant.
6. Do not generate definitions, explanations, or background questions.

{format_instructions}
"""

NESTED_VARIABLE_HUMAN_PROMPT = """variable_id: {variable_id}
variable_description: {variable_description}
query_hint: {query_hint}
resolved_variables: {resolved_variables}
"""

CAUSAL_ORIGINAL_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """

Additional guidance for CAUSAL relation subclaims:
- Ensure the minimal sufficient information set explicitly covers:
  1) cause-side event/property
  2) effect-side event/property
  3) the causal link between them
- Keep query count compact while still covering all three where necessary.
"""
)


class SubclaimAnalysis(BaseModel):
    subject: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    object: str = Field(min_length=1)
    constraints: list[str] = Field(default_factory=list)


class MinimalSufficientInformationItem(BaseModel):
    variable: str = Field(min_length=1)
    why_needed: str = Field(min_length=1)


class QueryPlanItem(BaseModel):
    type: Literal["factoid", "relation", "direct_statement", "verification"]
    target_variable: str = Field(min_length=1)
    query: str = Field(min_length=1)


class SearchPlannerOutput(BaseModel):
    subclaim_analysis: SubclaimAnalysis
    minimal_sufficient_information_set: list[MinimalSufficientInformationItem] = Field(min_length=1)
    query_plan: list[QueryPlanItem] = Field(min_length=2, max_length=4)

    def _deduped_query_plan(self) -> list[QueryPlanItem]:
        seen: set[str] = set()
        deduped: list[QueryPlanItem] = []
        for item in self.query_plan:
            query = item.query.strip()
            if not query:
                continue
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(
                QueryPlanItem(
                    type=item.type,
                    target_variable=item.target_variable.strip(),
                    query=query,
                )
            )
        return deduped

    def _ordered_queries(self, type_priority: dict[str, int]) -> list[str]:
        deduped = self._deduped_query_plan()
        indexed_items = list(enumerate(deduped))
        indexed_items.sort(
            key=lambda pair: (
                type_priority.get(pair[1].type, 99),
                pair[0],
            )
        )
        queries = [item.query for _, item in indexed_items if item.query]
        return queries[:4]

    def to_query_list(self) -> list[str]:
        return self._ordered_queries(
            {
                "factoid": 0,
                "relation": 1,
                "direct_statement": 2,
                "verification": 3,
            }
        )

    def to_causal_query_list(self) -> list[str]:
        return self._ordered_queries(
            {
                "relation": 0,
                "factoid": 1,
                "direct_statement": 2,
                "verification": 3,
            }
        )


class NestedVariableQueryPlanItem(BaseModel):
    type: Literal["factoid", "relation", "direct_statement"]
    query: str = Field(min_length=1)


class NestedVariableQueryOutput(BaseModel):
    query_plan: list[NestedVariableQueryPlanItem] = Field(min_length=2, max_length=3)

    def to_query_list(self) -> list[str]:
        type_priority = {"factoid": 0, "relation": 1, "direct_statement": 2}
        seen: set[str] = set()
        ordered: list[tuple[int, str]] = []
        for item in self.query_plan:
            query = item.query.strip()
            if not query:
                continue
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append((type_priority.get(item.type, 99), query))
        ordered.sort(key=lambda x: x[0])
        return [q for _, q in ordered][:3]


class SearchPlanner:
    def __init__(self, llm: BaseChatModel) -> None:
        self._parser = JsonOutputParser(pydantic_object=SearchPlannerOutput)
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
        ).partial(format_instructions=self._parser.get_format_instructions())
        self._chain = (self._prompt | llm | self._parser).with_config(
            {"run_name": "search_planner_chain", "tags": ["stage:search_planner"]}
        )

        self._causal_original_prompt = ChatPromptTemplate.from_messages(
            [("system", CAUSAL_ORIGINAL_SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
        ).partial(format_instructions=self._parser.get_format_instructions())
        self._causal_original_chain = (self._causal_original_prompt | llm | self._parser).with_config(
            {"run_name": "search_planner_causal_chain", "tags": ["stage:search_planner"]}
        )

        self._nested_variable_parser = JsonOutputParser(pydantic_object=NestedVariableQueryOutput)
        self._nested_variable_prompt = ChatPromptTemplate.from_messages(
            [("system", NESTED_VARIABLE_SYSTEM_PROMPT), ("human", NESTED_VARIABLE_HUMAN_PROMPT)]
        ).partial(format_instructions=self._nested_variable_parser.get_format_instructions())
        self._nested_variable_chain = (
            self._nested_variable_prompt | llm | self._nested_variable_parser
        ).with_config({"run_name": "search_planner_nested_variable_chain", "tags": ["stage:search_planner"]})

    def plan(self, relationship_type: str, sub_claim: str) -> SearchPlannerOutput:
        if not sub_claim.strip():
            raise ValueError("sub_claim must not be empty.")
        raw = self._chain.invoke(
            {
                "relationship_type": relationship_type.strip(),
                "sub_claim": sub_claim.strip(),
            }
        )
        return SearchPlannerOutput.model_validate(raw)

    def plan_causal_original(self, sub_claim: str) -> SearchPlannerOutput:
        if not sub_claim.strip():
            raise ValueError("sub_claim must not be empty.")
        raw = self._causal_original_chain.invoke(
            {
                "relationship_type": "CAUSAL",
                "sub_claim": sub_claim.strip(),
            }
        )
        return SearchPlannerOutput.model_validate(raw)

    def plan_nested_variable(
        self,
        variable_id: str,
        variable_description: str,
        query_hint: str | None = None,
        resolved_variables: list[str] | None = None,
    ) -> NestedVariableQueryOutput:
        vid = variable_id.strip()
        vdesc = variable_description.strip()
        if not vid:
            raise ValueError("variable_id must not be empty.")
        if not vdesc:
            raise ValueError("variable_description must not be empty.")

        resolved = resolved_variables if isinstance(resolved_variables, list) else []
        clean_resolved = [str(v).strip() for v in resolved if str(v).strip()]
        raw = self._nested_variable_chain.invoke(
            {
                "variable_id": vid,
                "variable_description": vdesc,
                "query_hint": (query_hint or "").strip(),
                "resolved_variables": clean_resolved,
            }
        )
        return NestedVariableQueryOutput.model_validate(raw)
