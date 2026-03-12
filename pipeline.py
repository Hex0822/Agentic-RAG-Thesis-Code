import json
import logging
from typing import Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from .claim_extraction import ClaimExtractor
from .config import (
    CLAIM_EXTRACTION_MODEL,
    KEYWORDS_MODEL,
    LLM_EVIDENCE_TOP_K,
    QUALITY_MAX_ROUNDS,
    RELEVANCE_BATCH_SIZE,
    RELEVANCE_MODEL_NAME,
    SEARCH_TOP_K,
    create_llm,
    ensure_openai_env,
    ensure_tavily_env,
)
from .evidence_packaging import LLMEvidence, prepare_llm_evidence
from .keyword_generation import KeywordsGenerator
from .relevance_ranking import MiniLMRelevanceRanker, RankedEvidence
from .retrieval_quality import (
    RetrievalQualityAgent,
    SubclaimAgentDecision,
)
from .search import TavilySearcher
from .text_processing import CandidateSentence, ProcessedArticleRecord, process_articles

DEMO_CLAIM = "Hunter Biden had no experience in Ukraine or in the energy sector when he joined the board of Burisma."
LOGGER = logging.getLogger(__name__)


def _short_text(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


def _build_overall_conclusion(decisions: Dict[str, SubclaimAgentDecision]) -> str:
    if not decisions:
        return "INCONCLUSIVE"

    conclusions = {item["conclusion"] for item in decisions.values()}
    if "INCONCLUSIVE" in conclusions:
        return "INCONCLUSIVE"
    if conclusions == {"SUPPORTED"}:
        return "SUPPORTED"
    if conclusions == {"REFUTED"}:
        return "REFUTED"
    if "REFUTED" in conclusions:
        return "REFUTED"
    return "MIXED"


def _build_overall_reason(
    overall_conclusion: str,
    decisions: Dict[str, SubclaimAgentDecision],
) -> str:
    if not decisions:
        return "No subclaim decisions were produced."
    if overall_conclusion == "SUPPORTED":
        return "All subclaims are marked SUPPORTED."
    if overall_conclusion == "REFUTED":
        refuted_claims = [
            claim for claim, item in decisions.items() if item["conclusion"] == "REFUTED"
        ]
        if refuted_claims:
            return f"At least one subclaim is REFUTED: {refuted_claims[0]}"
        return "At least one subclaim is REFUTED."
    if overall_conclusion == "INCONCLUSIVE":
        inconclusive_claims = [
            claim for claim, item in decisions.items() if item["conclusion"] == "INCONCLUSIVE"
        ]
        if inconclusive_claims:
            return f"At least one subclaim remains INCONCLUSIVE: {inconclusive_claims[0]}"
        return "Evidence is not sufficient to conclude all subclaims."
    return "Subclaims have mixed outcomes."


def _log_ranked_evidence_pretty(ranked_evidence: Dict[str, List[RankedEvidence]]) -> None:
    for claim, items in ranked_evidence.items():
        LOGGER.debug("Ranked evidence group\n  claim: %s\n  total_items: %s", claim, len(items))
        for item in items:
            raw_score_value = (
                "nan" if not isinstance(item["raw_score"], (int, float)) or item["raw_score"] != item["raw_score"] else f"{item['raw_score']:.6f}"
            )
            LOGGER.debug(
                "Ranked evidence item\n"
                "  claim: %s\n"
                "  rank: %s\n"
                "  raw_score: %s\n"
                "  score: %.6f\n"
                "  score_source: %s\n"
                "  keyword: %s\n"
                "  article_id: %s\n"
                "  title: %s\n"
                "  url: %s\n"
                "  sentence_index: %s/%s\n"
                "  sentence: %s",
                item["claim"],
                item["rank"],
                raw_score_value,
                item["score"],
                item["score_source"],
                item["keyword"],
                item["article_id"],
                item["article_title"],
                item["article_url"],
                item["sentence_index"] + 1,
                item["total_sentences"],
                item["sentence"],
            )


def _log_llm_evidence_pretty(llm_evidence: Dict[str, List[LLMEvidence]]) -> None:
    for claim, items in llm_evidence.items():
        LOGGER.debug("LLM evidence group\n  subclaim: %s\n  total_items: %s", claim, len(items))
        for item in items:
            LOGGER.debug(
                "LLM evidence item\n"
                "  subclaim: %s\n"
                "  title: %s\n"
                "  url: %s\n"
                "  evidence: %s",
                item["subclaim"],
                item["title"],
                item["url"],
                item["evidence"],
            )


def _log_quality_decisions_pretty(decisions: Dict[str, SubclaimAgentDecision]) -> None:
    for claim, decision in decisions.items():
        reasoning_text = decision["rationale"] or decision["evidence_assessment"]
        LOGGER.debug(
            "Quality reasoning\n"
            "  subclaim: %s\n"
            "  reasoning: %s",
            claim,
            reasoning_text,
        )
        LOGGER.debug(
            "Quality decision\n"
            "  subclaim: %s\n"
            "  next_action: %s\n"
            "  conclusion: %s\n"
            "  evidence_assessment: %s\n"
            "  keep_keywords: %s\n"
            "  drop_keywords: %s\n"
            "  new_keywords: %s\n"
            "  retrieval_intent: %s\n"
            "  rationale: %s",
            claim,
            decision["next_action"],
            decision["conclusion"],
            decision["evidence_assessment"],
            decision["keyword_update_plan"]["keep_keywords"],
            decision["keyword_update_plan"]["drop_keywords"],
            decision["keyword_update_plan"]["new_keywords"],
            decision["retrieval_intent"],
            decision["rationale"],
        )


def _log_fact_check_report(
    original_claim: str,
    decisions: Dict[str, SubclaimAgentDecision],
) -> None:
    overall_conclusion = _build_overall_conclusion(decisions)
    overall_reason = _build_overall_reason(overall_conclusion, decisions)
    LOGGER.info(
        "Fact-check report\n"
        "  original_claim: %s\n"
        "  overall_result: %s\n"
        "  overall_reason: %s",
        original_claim,
        overall_conclusion,
        overall_reason,
    )

    if not decisions:
        LOGGER.info(
            "Fact-check item\n"
            "  subclaim: (none)\n"
            "  result: INCONCLUSIVE\n"
            "  reason: No subclaim decisions were produced."
        )
        return

    for claim, decision in decisions.items():
        reason = decision["rationale"] or decision["evidence_assessment"] or "No reason provided."
        LOGGER.info(
            "Fact-check item\n"
            "  subclaim: %s\n"
            "  result: %s\n"
            "  reason: %s",
            claim,
            decision["conclusion"],
            reason,
        )


def _format_json(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True, sort_keys=True)


def _build_keywords_from_plan(
    current_keywords: List[str],
    decision: SubclaimAgentDecision,
) -> List[str]:
    plan = decision["keyword_update_plan"]
    keep_set = {item.lower() for item in plan["keep_keywords"]}
    drop_set = {item.lower() for item in plan["drop_keywords"]}

    merged: List[str] = []
    for keyword in current_keywords:
        cleaned = keyword.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if drop_set and lowered in drop_set:
            continue
        if keep_set and lowered not in keep_set:
            continue
        merged.append(cleaned)

    for keyword in plan["new_keywords"]:
        cleaned = keyword.strip()
        if cleaned:
            merged.append(cleaned)

    deduped: List[str] = []
    seen: set[str] = set()
    for keyword in merged:
        lowered = keyword.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(keyword)
    return deduped[:6]


def _group_candidates_by_claim(candidates: List[CandidateSentence]) -> Dict[str, List[CandidateSentence]]:
    grouped: Dict[str, List[CandidateSentence]] = {}
    for item in candidates:
        grouped.setdefault(item["claim"], []).append(item)
    return grouped


def _flatten_candidates(grouped: Dict[str, List[CandidateSentence]], ordered_claims: List[str]) -> List[CandidateSentence]:
    flattened: List[CandidateSentence] = []
    for claim in ordered_claims:
        flattened.extend(grouped.get(claim, []))
    for claim, items in grouped.items():
        if claim not in ordered_claims:
            flattened.extend(items)
    return flattened


class PipelineState(TypedDict):
    original_claim: str
    sub_claims: List[str]
    keywords: Dict[str, List[str]]
    articles: Dict[str, Dict[str, List[ProcessedArticleRecord]]]
    candidates: List[CandidateSentence]
    ranked_evidence: Dict[str, List[RankedEvidence]]
    llm_evidence: Dict[str, List[LLMEvidence]]
    quality_decisions: Dict[str, SubclaimAgentDecision]
    stats: Dict[str, object]


def build_pipeline():
    ensure_openai_env()
    ensure_tavily_env()
    claim_llm = create_llm(CLAIM_EXTRACTION_MODEL, temperature=0.2)
    keyword_llm = create_llm(KEYWORDS_MODEL, temperature=0.2)
    extractor = ClaimExtractor(claim_llm)
    generator = KeywordsGenerator(keyword_llm)
    searcher = TavilySearcher(top_k=SEARCH_TOP_K)
    ranker = MiniLMRelevanceRanker(
        model_name=RELEVANCE_MODEL_NAME,
        batch_size=RELEVANCE_BATCH_SIZE,
    )
    quality_agent = RetrievalQualityAgent(keyword_llm)

    def claim_extraction_node(state: PipelineState) -> PipelineState:
        LOGGER.info("Step 1/7: claim_extraction")
        sub_claims = extractor.extract(state["original_claim"])
        stats = dict(state.get("stats", {}))
        stats["sub_claims"] = len(sub_claims)
        return {"sub_claims": sub_claims, "stats": stats}  # type: ignore

    def keyword_generation_node(state: PipelineState) -> PipelineState:
        LOGGER.info("Step 2/7: keyword_generation")
        keywords = {claim: generator.generate(claim) for claim in state["sub_claims"]}
        stats = dict(state.get("stats", {}))
        stats["keywords"] = sum(len(items) for items in keywords.values())
        return {"keywords": keywords, "stats": stats}  # type: ignore

    def keyword_search_node(state: PipelineState) -> PipelineState:
        LOGGER.info("Step 3/7: keyword_search")
        articles, search_stats = searcher.search(state["keywords"])
        stats = dict(state.get("stats", {}))
        stats["search"] = search_stats
        return {"articles": articles, "stats": stats}  # type: ignore

    def text_processing_node(state: PipelineState) -> PipelineState:
        LOGGER.info("Step 4/7: text_processing")
        processed_articles, process_stats, candidates = process_articles(state["articles"])
        stats = dict(state.get("stats", {}))
        stats["text_processing"] = process_stats
        return {
            "articles": processed_articles,
            "candidates": candidates,
            "stats": stats,
        }  # type: ignore

    def relevance_ranking_node(state: PipelineState) -> PipelineState:
        LOGGER.info("Step 5/7: relevance_ranking")
        ranked_evidence, ranking_stats = ranker.rank(state["candidates"])
        stats = dict(state.get("stats", {}))
        stats["relevance_ranking"] = ranking_stats
        return {"ranked_evidence": ranked_evidence, "stats": stats}  # type: ignore

    def evidence_packaging_node(state: PipelineState) -> PipelineState:
        LOGGER.info("Step 6/7: evidence_packaging")
        llm_evidence, packaging_stats = prepare_llm_evidence(
            ranked_evidence=state["ranked_evidence"],
            articles=state["articles"],
            top_k=LLM_EVIDENCE_TOP_K,
        )
        stats = dict(state.get("stats", {}))
        stats["evidence_packaging"] = packaging_stats
        return {"llm_evidence": llm_evidence, "stats": stats}  # type: ignore

    def retrieval_quality_node(state: PipelineState) -> PipelineState:
        LOGGER.info("Step 7/7: retrieval_quality_agent")

        current_keywords = {claim: list(values) for claim, values in state["keywords"].items()}
        keyword_history = {claim: [list(values)] for claim, values in state["keywords"].items()}
        ranking_queries = {claim: claim for claim in state["sub_claims"]}
        current_articles = {
            claim: dict(keyword_map) for claim, keyword_map in state["articles"].items()
        }
        current_candidates_grouped = _group_candidates_by_claim(state["candidates"])
        current_ranked = {claim: list(values) for claim, values in state["ranked_evidence"].items()}
        current_llm_evidence = {
            claim: list(values) for claim, values in state["llm_evidence"].items()
        }

        decisions: Dict[str, SubclaimAgentDecision] = {}
        rounds_report: List[Dict[str, object]] = []
        pending_claims = list(state["sub_claims"])

        for round_idx in range(1, QUALITY_MAX_ROUNDS + 1):
            if not pending_claims:
                break

            LOGGER.info(
                "Quality round %s/%s: evaluating %s subclaim(s).",
                round_idx,
                QUALITY_MAX_ROUNDS,
                len(pending_claims),
            )
            LOGGER.debug(
                "Ranked evidence snapshot (round %s/%s).",
                round_idx,
                QUALITY_MAX_ROUNDS,
            )
            round_ranked = {claim: current_ranked.get(claim, []) for claim in pending_claims}
            _log_ranked_evidence_pretty(round_ranked)
            round_record: Dict[str, object] = {
                "round": round_idx,
                "evaluated_claims": list(pending_claims),
            }
            claims_to_refine: List[str] = []

            for claim in pending_claims:
                decision = quality_agent.evaluate(
                    original_claim=state["original_claim"],
                    subclaim=claim,
                    current_keywords=current_keywords.get(claim, []),
                    evidence_items=current_llm_evidence.get(claim, []),
                    round_idx=round_idx,
                    max_rounds=QUALITY_MAX_ROUNDS,
                    keyword_history=keyword_history.get(claim, []),
                )
                decisions[claim] = decision
                if decision["next_action"] == "REFINE_KEYWORDS":
                    claims_to_refine.append(claim)
                    intent = decision.get("retrieval_intent", "").strip()
                    if intent:
                        ranking_queries[claim] = intent

            round_record["refine_claims"] = list(claims_to_refine)
            rounds_report.append(round_record)

            if not claims_to_refine:
                LOGGER.info("Quality round %s: no further refine action requested.", round_idx)
                pending_claims = []
                break

            if round_idx == QUALITY_MAX_ROUNDS:
                LOGGER.info("Reached max quality rounds (%s).", QUALITY_MAX_ROUNDS)
                pending_claims = claims_to_refine
                break

            refine_keywords: Dict[str, List[str]] = {}
            for claim in claims_to_refine:
                new_keywords = _build_keywords_from_plan(
                    current_keywords=current_keywords.get(claim, []),
                    decision=decisions[claim],
                )
                if not new_keywords:
                    new_keywords = generator.generate(claim)
                current_keywords[claim] = new_keywords
                refine_keywords[claim] = new_keywords
                if new_keywords:
                    history = keyword_history.setdefault(claim, [])
                    if not history:
                        history.append(new_keywords)
                    else:
                        last = [item.lower() for item in history[-1]]
                        current = [item.lower() for item in new_keywords]
                        if last != current:
                            history.append(new_keywords)

            LOGGER.info(
                "Quality round %s: re-running retrieval for %s subclaim(s).",
                round_idx,
                len(refine_keywords),
            )
            refined_articles_raw, refined_search_stats = searcher.search(refine_keywords)
            refined_articles, refined_process_stats, refined_candidates = process_articles(refined_articles_raw)
            refined_ranked, refined_rank_stats = ranker.rank(
                refined_candidates,
                claim_queries=ranking_queries,
            )
            refined_llm_evidence, refined_packaging_stats = prepare_llm_evidence(
                ranked_evidence=refined_ranked,
                articles=refined_articles,
                top_k=LLM_EVIDENCE_TOP_K,
            )

            round_record["refine_keywords"] = refine_keywords
            round_record["search_stats"] = refined_search_stats
            round_record["text_processing_stats"] = refined_process_stats
            round_record["ranking_stats"] = refined_rank_stats
            round_record["packaging_stats"] = refined_packaging_stats

            for claim in claims_to_refine:
                current_articles[claim] = refined_articles.get(claim, {})
                current_ranked[claim] = refined_ranked.get(claim, [])
                current_llm_evidence[claim] = refined_llm_evidence.get(claim, [])

            refined_candidates_grouped = _group_candidates_by_claim(refined_candidates)
            for claim in claims_to_refine:
                current_candidates_grouped[claim] = refined_candidates_grouped.get(claim, [])

            pending_claims = claims_to_refine

        updated_candidates = _flatten_candidates(current_candidates_grouped, state["sub_claims"])
        stats = dict(state.get("stats", {}))
        stats["retrieval_quality_agent"] = {
            "rounds": rounds_report,
            "final_action_counts": {
                "finalize_subclaim": sum(
                    1 for item in decisions.values() if item["next_action"] == "FINALIZE_SUBCLAIM"
                ),
                "refine_keywords": sum(
                    1 for item in decisions.values() if item["next_action"] == "REFINE_KEYWORDS"
                ),
                "stop_subclaim": sum(
                    1 for item in decisions.values() if item["next_action"] == "STOP_SUBCLAIM"
                ),
            },
            "max_rounds": QUALITY_MAX_ROUNDS,
        }

        return {
            "keywords": current_keywords,
            "articles": current_articles,
            "candidates": updated_candidates,
            "ranked_evidence": current_ranked,
            "llm_evidence": current_llm_evidence,
            "quality_decisions": decisions,
            "stats": stats,
        }  # type: ignore

    workflow = StateGraph(PipelineState)
    workflow.add_node("claim_extraction", claim_extraction_node)
    workflow.add_node("keyword_generation", keyword_generation_node)
    workflow.add_node("keyword_search", keyword_search_node)
    workflow.add_node("text_processing", text_processing_node)
    workflow.add_node("relevance_ranking", relevance_ranking_node)
    workflow.add_node("evidence_packaging", evidence_packaging_node)
    workflow.add_node("retrieval_quality", retrieval_quality_node)
    workflow.set_entry_point("claim_extraction")
    workflow.add_edge("claim_extraction", "keyword_generation")
    workflow.add_edge("keyword_generation", "keyword_search")
    workflow.add_edge("keyword_search", "text_processing")
    workflow.add_edge("text_processing", "relevance_ranking")
    workflow.add_edge("relevance_ranking", "evidence_packaging")
    workflow.add_edge("evidence_packaging", "retrieval_quality")
    workflow.add_edge("retrieval_quality", END)
    return workflow.compile()


def run_demo() -> None:
    app = build_pipeline()
    result = app.invoke({"original_claim": DEMO_CLAIM})  # type: ignore
    print("\nSub-claims:", result["sub_claims"])
    print("\nKeywords:", result["keywords"])
    article_counts = {
        claim: {keyword: len(items) for keyword, items in keyword_map.items()}
        for claim, keyword_map in result["articles"].items()
    }
    ranking_counts = {claim: len(items) for claim, items in result["ranked_evidence"].items()}
    llm_evidence_counts = {claim: len(items) for claim, items in result["llm_evidence"].items()}
    quality_decision_counts = {
        "finalize_subclaim": sum(
            1
            for item in result.get("quality_decisions", {}).values()
            if item["next_action"] == "FINALIZE_SUBCLAIM"
        ),
        "refine_keywords": sum(
            1
            for item in result.get("quality_decisions", {}).values()
            if item["next_action"] == "REFINE_KEYWORDS"
        ),
        "stop_subclaim": sum(
            1
            for item in result.get("quality_decisions", {}).values()
            if item["next_action"] == "STOP_SUBCLAIM"
        ),
    }
    print("\nArticles (counts only):", article_counts)
    print("\nCandidates:", len(result["candidates"]))
    print("\nRanked evidence (counts only):", ranking_counts)
    print("\nLLM evidence (top-k counts):", llm_evidence_counts)
    print("\nQuality decisions:", quality_decision_counts)
    print("\nStats:")
    print(_format_json(result["stats"]))
    LOGGER.debug("Full sub-claims: %s", result["sub_claims"])
    LOGGER.debug("Full keywords: %s", result["keywords"])
    LOGGER.debug("Full articles: %s", result["articles"])
    _log_ranked_evidence_pretty(result["ranked_evidence"])
    _log_llm_evidence_pretty(result["llm_evidence"])
    _log_quality_decisions_pretty(result.get("quality_decisions", {}))
    LOGGER.debug("Full stats: %s", result["stats"])
    _log_fact_check_report(
        original_claim=result.get("original_claim", DEMO_CLAIM),
        decisions=result.get("quality_decisions", {}),
    )


if __name__ == "__main__":
    run_demo()
