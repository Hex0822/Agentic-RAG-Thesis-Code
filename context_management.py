"""Build and store round-level context for LLM reasoning."""

from copy import deepcopy
from typing import Any

from config import CONTEXT_EVIDENCE_TOP_K


def _build_result_lookup(search_results: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for idx, row in enumerate(search_results):
        lookup[idx] = row
    return lookup


def _build_query_lookup(search_plan: list[dict[str, Any]]) -> dict[str, list[str]]:
    query_lookup: dict[str, list[str]] = {}
    for item in search_plan:
        sub_claim = str(item.get("sub_claim", "")).strip()
        queries = item.get("queries", [])
        if not sub_claim or not isinstance(queries, list):
            continue
        query_lookup[sub_claim] = [str(q).strip() for q in queries if str(q).strip()]
    return query_lookup


def _get_chunk_text_by_index(row: dict[str, Any], sentence_index: int) -> str:
    chunks = row.get("sentence_chunks", [])
    if not isinstance(chunks, list):
        return ""
    for chunk in chunks:
        idx = int(chunk.get("sentence_index", -1))
        if idx == sentence_index:
            return str(chunk.get("text", "")).strip()
    return ""


def _build_chunk_text(prev_sentence: str, center_sentence: str, next_sentence: str) -> str:
    parts = [prev_sentence.strip(), center_sentence.strip(), next_sentence.strip()]
    return " ".join([p for p in parts if p]).strip()


def _build_subclaim_context(
    sub_claim: str,
    ranked_sentences: list[dict[str, Any]],
    result_lookup: dict[int, dict[str, Any]],
    query_lookup: dict[str, list[str]],
) -> dict[str, Any]:
    top_ranked = ranked_sentences[:CONTEXT_EVIDENCE_TOP_K]
    evidence_chunks: list[dict[str, Any]] = []
    planned_queries = query_lookup.get(sub_claim, [])

    for item in top_ranked:
        result_index = int(item.get("result_index", -1))
        sentence_index = int(item.get("sentence_index", -1))
        row = result_lookup.get(result_index, {})

        prev_sentence = _get_chunk_text_by_index(row, sentence_index - 1)
        next_sentence = _get_chunk_text_by_index(row, sentence_index + 1)
        center_sentence = str(item.get("sentence_text", "")).strip()
        chunk_text = _build_chunk_text(prev_sentence, center_sentence, next_sentence)

        evidence_chunks.append(
            {
                "rank": int(item.get("global_rank", 0)),
                "chunk_text": chunk_text,
                "url": str(item.get("url", "")).strip(),
                "title": str(item.get("title", "")).strip(),
                "query": str(item.get("query", "")).strip(),
            }
        )

    return {
        "sub_claim": sub_claim,
        "planned_queries": planned_queries,
        "evidence_chunks": evidence_chunks,
    }


def build_context_management(
    search_plan: list[dict[str, Any]],
    search_results: list[dict[str, Any]],
    rerank_results: list[dict[str, Any]],
    previous_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result_lookup = _build_result_lookup(search_results)
    query_lookup = _build_query_lookup(search_plan)

    rerank_lookup: dict[str, list[dict[str, Any]]] = {}
    for item in rerank_results:
        sub_claim = str(item.get("sub_claim", "")).strip()
        ranked_sentences = item.get("ranked_sentences", [])
        if not sub_claim or not isinstance(ranked_sentences, list):
            continue
        rerank_lookup[sub_claim] = ranked_sentences

    ordered_sub_claims: list[str] = []
    for item in search_plan:
        sub_claim = str(item.get("sub_claim", "")).strip()
        if sub_claim and sub_claim not in ordered_sub_claims:
            ordered_sub_claims.append(sub_claim)
    for sub_claim in rerank_lookup:
        if sub_claim not in ordered_sub_claims:
            ordered_sub_claims.append(sub_claim)

    subclaim_contexts: list[dict[str, Any]] = []
    for sub_claim in ordered_sub_claims:
        subclaim_contexts.append(
            _build_subclaim_context(
                sub_claim=sub_claim,
                ranked_sentences=rerank_lookup.get(sub_claim, []),
                result_lookup=result_lookup,
                query_lookup=query_lookup,
            )
        )

    previous_rounds: list[dict[str, Any]] = []
    if isinstance(previous_context, dict):
        rounds = previous_context.get("rounds", [])
        if isinstance(rounds, list):
            previous_rounds = [r for r in rounds if isinstance(r, dict)]

    current_round_index = len(previous_rounds) + 1
    current_round = {
        "round": current_round_index,
        "search_plan": search_plan,
        "search_results": search_results,
        "rerank_results": rerank_results,
        "subclaim_contexts": subclaim_contexts,
        "llm_feedback": [],
        "reasoning_output": {},
        "next_action": "WAIT_REASONING",
    }

    all_rounds = [*previous_rounds, current_round]
    return {
        "rounds": all_rounds,
        "latest_round": current_round_index,
        "overall_next_action": "WAIT_REASONING",
        "reasoning_ready": False,
        "subclaim_contexts": subclaim_contexts,
    }


def apply_reasoning_feedback(
    context_management: dict[str, Any],
    reasoning_output: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(context_management, dict):
        return context_management

    updated = deepcopy(context_management)
    rounds = updated.get("rounds", [])
    if not isinstance(rounds, list) or not rounds:
        return updated

    latest_round = rounds[-1]
    feedback_list = latest_round.get("llm_feedback", [])
    if not isinstance(feedback_list, list):
        feedback_list = []
    feedback_list.append(reasoning_output)
    latest_round["llm_feedback"] = feedback_list
    latest_round["reasoning_output"] = reasoning_output

    search_needed = bool(
        reasoning_output.get("search_needed", reasoning_output.get("need_more_search", False))
    )
    next_action = "RETRIEVE_MORE" if search_needed else "READY_FOR_REASONING"
    latest_round["next_action"] = next_action

    updated["rounds"] = rounds
    updated["latest_round"] = len(rounds)
    updated["overall_next_action"] = next_action
    updated["reasoning_ready"] = not search_needed
    updated["subclaim_contexts"] = latest_round.get("subclaim_contexts", [])
    return updated
