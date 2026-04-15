"""Nested planner for decomposing nested claims into dependency steps."""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """
You are a nested-claim planner in a fact-checking retrieval pipeline.

Input:
- relationship_type (should be NESTED)
- sub_claims (for NESTED, it is usually a single original claim)
- classification_basis

Goal:
Do NOT verify the claim.
Transform the nested claim into a minimal, executable dependency chain of variables,
ONLY IF such a chain is truly required for verification.

If no real dependency chain exists, return an EMPTY plan.

---

Core principle:
The goal is NOT to fully normalize or decompose every phrase.
The goal is to produce the SMALLEST set of variables that is SUFFICIENT to verify the claim.

---

CRITICAL DISTINCTION (VERY IMPORTANT):

Distinguish between:
1) Surface linguistic structure (e.g., possessives, modifiers)
2) Verification dependency structure (what must be resolved first to verify the claim)

A claim should be treated as NESTED ONLY if:
→ resolving an intermediate variable is NECESSARY before verifying the main predicate.

Do NOT confuse syntactic nesting with reasoning dependency.

---

Step 1: Identify candidate nested structures (WEAK SIGNAL)

Detect:
- possessive chains (A's B's C)
- relative clauses ("the X that did Y")
- implicit references ("the inventor of X")

IMPORTANT:
These are ONLY candidates.
They do NOT automatically justify nested decomposition.

---

Step 2: Check if nested decomposition is actually needed (CRITICAL GATE)

Ask:

"Without resolving this intermediate variable, can the main claim still be verified directly through retrieval?"

If YES:
→ DO NOT create a nested plan
→ Return empty steps

If NO:
→ Proceed with nested decomposition

Examples where nested is NOT needed:
- "X's official account posted Y"
- "the company's CEO said Z" (if CEO identity is not the bottleneck)
- cases where retrieval already directly links X to Y

Examples where nested IS needed:
- "the wife of the CEO of Microsoft was born in France"
- "the inventor of the device used in X won a Nobel Prize"

---

Step 3: Distinguish constants vs variables (CRITICAL)

- Constants:
  Explicit entities, names, places, events
  → treat as known
  → DO NOT create variables

- Variables:
  Only introduce variables if they are:
  ✔ unknown
  ✔ necessary for verification
  ✔ cannot be bypassed by direct retrieval

---

Step 4: Predicate-first check (NEW, VERY IMPORTANT)

Before finalizing variables, analyze the MAIN predicate.

Ask:
→ Is the difficulty of this claim mainly due to:
   (A) resolving hidden entities?  → NESTED
   (B) interpreting a strong/ambiguous predicate? → NOT NESTED

If (B):
→ DO NOT decompose into nested variables
→ Return empty steps

Examples of predicate-driven difficulty:
- "caused", "proved", "guaranteed"
- "promoted", "led to", "resulted in"
- strong causal or stance claims

These should NOT trigger nested planning.

---

Step 5: Extract intermediate variables (ONLY IF STILL VALID)

Each variable must satisfy ALL:

1) Necessary:
   Removing it makes verification impossible or incorrect

2) Independent:
   Can be queried

3) Minimal:
   No redundant decomposition

4) Irreplaceable:
   Cannot be skipped via direct retrieval of the full relation

---

Step 6: Build dependencies

- DAG structure
- Prefer linear chain
- Each step depends only on previous ones

---

Step 7: Define execution order

- Inner dependency → outer dependency
- Must be executable sequentially

---

Important minimality rules:

- DO NOT create variables for:
  - "official account"
  - "spokesperson"
  - "website"
  - generic role descriptors

UNLESS:
→ the identity of that role is the core unknown required for verification

---

Final sanity check (MANDATORY):

Before outputting, verify:

"Is this plan truly necessary, or am I just decomposing syntax?"

If the plan is not strictly necessary:
→ return empty steps

---

Output requirements:
- Keep variable_id snake_case
- Keep query_hint concise
- depends_on valid
- steps must be executable
- DO NOT include unnecessary variables

{format_instructions}
"""

HUMAN_PROMPT = """relationship_type: {relationship_type}
sub_claims: {sub_claims}
classification_basis: {classification_basis}
"""


class NestedStep(BaseModel):
    variable_id: str = Field(min_length=1, description="snake_case variable identifier")
    description: str = Field(min_length=1, description="what this variable represents")
    query_hint: str = Field(
        min_length=1,
        description="query hint for generating retrieval queries for this variable",
    )
    depends_on: list[str] = Field(
        default_factory=list, description="list of variable_id dependencies"
    )


class NestedPlannerOutput(BaseModel):
    relationship_type: Literal["NESTED"] = "NESTED"
    nested_claim: str = Field(min_length=1)
    nested_structure: str = Field(min_length=1)
    planning_basis: str = Field(min_length=1)
    steps: list[NestedStep] = Field(min_length=1)
    execution_order: list[str] = Field(min_length=1)


class NestedPlannerEnvelope(BaseModel):
    nested_plan: NestedPlannerOutput


class NestedPlanner:
    def __init__(self, llm: BaseChatModel) -> None:
        self._parser = JsonOutputParser(pydantic_object=NestedPlannerEnvelope)
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
        ).partial(format_instructions=self._parser.get_format_instructions())
        self._chain = (self._prompt | llm | self._parser).with_config(
            {"run_name": "nested_planner_chain", "tags": ["stage:nested_planner"]}
        )

    def plan(
        self,
        relationship_type: str,
        sub_claims: list[str],
        classification_basis: str,
    ) -> NestedPlannerOutput:
        rel = relationship_type.strip().upper()
        if rel != "NESTED":
            raise ValueError("NestedPlanner only accepts relationship_type='NESTED'.")
        if not sub_claims:
            raise ValueError("sub_claims must not be empty for nested planning.")

        raw = self._chain.invoke(
            {
                "relationship_type": rel,
                "sub_claims": sub_claims,
                "classification_basis": classification_basis.strip(),
            }
        )
        envelope = NestedPlannerEnvelope.model_validate(raw)
        return envelope.nested_plan
