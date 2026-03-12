from typing import Dict, List, Optional, TypedDict

from .relevance_ranking import RankedEvidence
from .text_processing import ProcessedArticleRecord


class LLMEvidence(TypedDict):
    subclaim: str
    title: str
    url: str
    evidence: str


class EvidencePackagingStats(TypedDict):
    claims_in: int
    claims_out: int
    selected_total: int
    missing_article_context: int
    skipped_same_article: int
    skipped_duplicate_evidence: int


def prepare_llm_evidence(
    ranked_evidence: Dict[str, List[RankedEvidence]],
    articles: Dict[str, Dict[str, List[ProcessedArticleRecord]]],
    top_k: int,
) -> tuple[Dict[str, List[LLMEvidence]], EvidencePackagingStats]:
    capped_top_k = max(top_k, 0)
    packaged: Dict[str, List[LLMEvidence]] = {}
    selected_total = 0
    missing_article_context = 0
    skipped_same_article = 0
    skipped_duplicate_evidence = 0

    for claim, ranked_items in ranked_evidence.items():
        sentence_lookup = _build_sentence_lookup(articles.get(claim, {}))
        claim_packaged: List[LLMEvidence] = []
        seen_article_ids: set[str] = set()
        seen_evidence_keys: set[str] = set()
        backlog_same_article: List[tuple[RankedEvidence, str, bool, str]] = []

        for item in ranked_items:
            if len(claim_packaged) >= capped_top_k:
                break
            evidence, used_context = _build_evidence_text(item, sentence_lookup)
            evidence_key = _normalize_text_key(evidence)
            if evidence_key in seen_evidence_keys:
                skipped_duplicate_evidence += 1
                continue
            article_id = item["article_id"]
            if article_id in seen_article_ids:
                skipped_same_article += 1
                backlog_same_article.append((item, evidence, used_context, evidence_key))
                continue
            seen_article_ids.add(article_id)
            seen_evidence_keys.add(evidence_key)
            if not used_context:
                missing_article_context += 1
            claim_packaged.append(
                {
                    "subclaim": claim,
                    "title": item["article_title"],
                    "url": item["article_url"],
                    "evidence": evidence,
                }
            )
            selected_total += 1

        if len(claim_packaged) < capped_top_k:
            for item, evidence, used_context, evidence_key in backlog_same_article:
                if len(claim_packaged) >= capped_top_k:
                    break
                if evidence_key in seen_evidence_keys:
                    skipped_duplicate_evidence += 1
                    continue
                seen_evidence_keys.add(evidence_key)
                if not used_context:
                    missing_article_context += 1
                claim_packaged.append(
                    {
                        "subclaim": claim,
                        "title": item["article_title"],
                        "url": item["article_url"],
                        "evidence": evidence,
                    }
                )
                selected_total += 1

        packaged[claim] = claim_packaged

    stats: EvidencePackagingStats = {
        "claims_in": len(ranked_evidence),
        "claims_out": len(packaged),
        "selected_total": selected_total,
        "missing_article_context": missing_article_context,
        "skipped_same_article": skipped_same_article,
        "skipped_duplicate_evidence": skipped_duplicate_evidence,
    }
    return packaged, stats


def _build_sentence_lookup(
    keyword_map: Dict[str, List[ProcessedArticleRecord]],
) -> Dict[str, List[str]]:
    lookup: Dict[str, List[str]] = {}
    for records in keyword_map.values():
        for record in records:
            article_id = str(record.get("article_id", "")).strip()
            if not article_id or article_id in lookup:
                continue

            raw_sentences = record.get("sentences", [])
            if not isinstance(raw_sentences, list):
                continue

            sentences = [str(sentence).strip() for sentence in raw_sentences if str(sentence).strip()]
            if sentences:
                lookup[article_id] = sentences
    return lookup


def _build_evidence_text(
    item: RankedEvidence,
    sentence_lookup: Dict[str, List[str]],
) -> tuple[str, bool]:
    article_id = item["article_id"]
    sentences = sentence_lookup.get(article_id)
    if not sentences:
        return item["sentence"], False

    target_index = _resolve_sentence_index(sentences, item["sentence_index"], item["sentence"])
    if target_index is None:
        return item["sentence"], False

    # Build a local context: prev + current + next. Edge sentences only keep existing neighbors.
    start = max(0, target_index - 1)
    end = min(len(sentences), target_index + 2)
    evidence = " ".join(sentences[start:end]).strip()
    if not evidence:
        return item["sentence"], False
    return evidence, True


def _resolve_sentence_index(sentences: List[str], index: int, target_sentence: str) -> Optional[int]:
    if 0 <= index < len(sentences):
        return index
    target = target_sentence.strip()
    for idx, sentence in enumerate(sentences):
        if sentence.strip() == target:
            return idx
    return None


def _normalize_text_key(text: str) -> str:
    compact = " ".join(text.lower().split())
    return compact
