"""Search planner for building retrieval queries from a single sub-claim."""

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """
You are a query planner for a fact-checking retrieval system.

Task:
Given a subclaim, generate search queries to retrieve factual evidence that can support or refute the claim.

Query requirements:

1. Generate at least 2 queries and at most 4 queries.

2. The first two queries must be:
   - 1 Factoid query
   - 1 Relation query

3. You may optionally add (It's ok to not to include these if factoid and relation queries are sufficient):
   - 1 Bag-of-Words query
   - 1 Verification query

4. The total number of queries must not exceed 4.

Query type definitions:

Factoid query
A concise WH-style factual question (Who / What / When / Where).
Prefer questions that identify key entities, ownership, founding, acquisition, or dates.

Relation query
A short entity–relation phrase suitable for matching titles or factual statements.

Bag-of-Words query
A short lexical query containing 3–6 important content words (not a full sentence).

Verification query
A short yes/no-style claim-check question directly testing the subclaim.

Generation rules:

- Queries must be independent and suitable for parallel search.
- Each query should provide a different retrieval perspective (e.g., entity fact, entity relation, or claim verification).
- Avoid generating queries that only differ in word order or minor phrasing changes.
- Prefer queries likely to appear in web page titles, factual sentences, or snippets.
- Do not generate explanatory or procedural questions (avoid "why" or "how to").
- Keep queries short, precise, and retrieval-friendly.
- When possible, prefer WH-style questions over yes/no questions for verification.
- Prefer WH-style factoid questions that directly ask for verifiable facts (e.g., who founded X, when did Y happen, who owns Z).

{format_instructions}
"""

HUMAN_PROMPT = """Relationship type: {relationship_type}
Sub-claim: {sub_claim}
"""

CAUSAL_ORIGINAL_SYSTEM_PROMPT = """
You are a query planner for a fact-checking retrieval system.

Task:
Given a CAUSAL relation subclaim, generate search queries to retrieve evidence about the causal relationship between the proposed cause and the proposed effect.

Query requirements:
- Generate 2 to 4 queries.
- The first query must be a relation phrase query.
- The second query must directly test the causal link in the subclaim.

Optional queries:
- 1 Bag-of-Words query
- 1 Verification query

Query types:

Relation query  
A short entity–relation phrase designed to match titles or factual statements.  
Prefer phrases describing impact, effect, or connection between the cause and effect.

Examples:
- "COVID impact airline ticket prices"
- "pandemic effect on airline fares"
- "COVID airline ticket price increase"

Causal query  
A simple question that directly tests whether the cause produced the effect.

Examples:
- "Did COVID cause airline ticket prices to increase?"
- "Did the pandemic lead to higher airline fares?"

Bag-of-Words query  
3–6 important keywords capturing the main entities and events.

Examples:
- "COVID airline ticket price increase"
- "pandemic airfare surge 2021"

Verification query  
A short yes/no claim-check question.

Rules:
- Queries must resemble how information appears in web titles or factual statements.
- Avoid literal paraphrases of the subclaim.
- Prefer simple wording commonly used in news or research titles.
- Keep queries concise (phrases ≈3–8 words, questions <12 words).
- The relation query must include both the cause and the effect.
- The causal query must explicitly mention both the proposed cause and the proposed effect.
- If the subclaim includes a specific year/date, keep it in at least one of the first two queries.
- Avoid generating multiple queries that only differ in word order.

Output mapping:
- Put the relation query in `relation_query`.
- Put the causal-link test question in `factoid_query`.
- Put optional queries in `bag_of_words_query` and `verification_query`.

{format_instructions}
"""


class SearchPlannerOutput(BaseModel):
    factoid_query: str = Field(min_length=1)
    relation_query: str = Field(min_length=1)
    bag_of_words_query: str | None = None
    verification_query: str | None = None

    def to_query_list(self) -> list[str]:
        queries = [self.factoid_query, self.relation_query]
        if self.bag_of_words_query:
            queries.append(self.bag_of_words_query)
        if self.verification_query:
            queries.append(self.verification_query)
        return [q.strip() for q in queries if q and q.strip()]

    def to_causal_query_list(self) -> list[str]:
        queries = [self.relation_query, self.factoid_query]
        if self.bag_of_words_query:
            queries.append(self.bag_of_words_query)
        if self.verification_query:
            queries.append(self.verification_query)
        return [q.strip() for q in queries if q and q.strip()]


class SearchPlanner:
    def __init__(self, llm: BaseChatModel) -> None:
        self._parser = JsonOutputParser(pydantic_object=SearchPlannerOutput)
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
        ).partial(format_instructions=self._parser.get_format_instructions())
        self._chain = self._prompt | llm | self._parser
        self._causal_original_prompt = ChatPromptTemplate.from_messages(
            [("system", CAUSAL_ORIGINAL_SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
        ).partial(format_instructions=self._parser.get_format_instructions())
        self._causal_original_chain = self._causal_original_prompt | llm | self._parser

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
