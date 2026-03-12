import logging
import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, TypedDict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .text_processing import CandidateSentence


class RankedEvidence(CandidateSentence):
    raw_score: float
    score: float
    score_source: str
    rank: int


class RankingStats(TypedDict):
    claims_in: int
    claims_ranked: int
    evidence_in: int
    evidence_ranked: int
    evidence_model_scored: int
    evidence_fallback_scored: int


class MiniLMRelevanceRanker:
    """Cross-encoder relevance scorer for (claim, evidence sentence) pairs."""

    def __init__(self, model_name: str, batch_size: int = 32, max_length: int = 256) -> None:
        self._logger = logging.getLogger(__name__)
        self._model_name = model_name
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model.to(self._device)
            self._model.eval()
            self._logger.info(
                "Loaded cross-encoder: %s (device=%s)",
                model_name,
                self._device,
            )
        except Exception as exc:
            self._logger.warning(
                "Failed to load cross-encoder '%s'. Falling back to lexical score. Error: %s",
                model_name,
                exc,
            )

    def rank(
        self,
        candidates: List[CandidateSentence],
        claim_queries: Optional[Dict[str, str]] = None,
    ) -> tuple[Dict[str, List[RankedEvidence]], RankingStats]:
        grouped: Dict[str, List[CandidateSentence]] = defaultdict(list)
        for item in candidates:
            grouped[item["claim"]].append(item)

        ranked: Dict[str, List[RankedEvidence]] = {}
        model_scored = 0
        fallback_scored = 0

        for claim, items in grouped.items():
            query = claim
            if claim_queries:
                override = claim_queries.get(claim, "").strip()
                if override:
                    query = override
            if self._model is not None and self._tokenizer is not None:
                pair_scores = self._score_pairs(query, items)
            else:
                pair_scores = [
                    self._fallback_result(query, item["sentence"], "fallback_model_unavailable")
                    for item in items
                ]

            scored_items: List[RankedEvidence] = []
            for item, (raw_score, score, source) in zip(items, pair_scores):
                scored_items.append(
                    {
                        **item,
                        "raw_score": raw_score,
                        "score": score,
                        "score_source": source,
                        "rank": 0,
                    }
                )
                if source == "cross_encoder":
                    model_scored += 1
                else:
                    fallback_scored += 1

            scored_items.sort(key=lambda x: x["score"], reverse=True)
            for rank, record in enumerate(scored_items, start=1):
                record["rank"] = rank

            ranked[claim] = scored_items
            self._logger.debug(
                "Ranked evidence for claim '%s' (query='%s'): total=%s model=%s fallback=%s",
                claim,
                query,
                len(scored_items),
                sum(1 for x in scored_items if x["score_source"] == "cross_encoder"),
                sum(1 for x in scored_items if x["score_source"] != "cross_encoder"),
            )

        stats: RankingStats = {
            "claims_in": len(grouped),
            "claims_ranked": len(ranked),
            "evidence_in": len(candidates),
            "evidence_ranked": sum(len(items) for items in ranked.values()),
            "evidence_model_scored": model_scored,
            "evidence_fallback_scored": fallback_scored,
        }
        return ranked, stats

    def _score_pairs(
        self, query: str, items: List[CandidateSentence]
    ) -> List[Tuple[float, float, str]]:
        if self._model is None or self._tokenizer is None:
            return [
                self._fallback_result(query, item["sentence"], "fallback_model_unavailable")
                for item in items
            ]

        results: List[Tuple[float, float, str]] = []
        for item in items:
            evidence = item["sentence"]
            raw_score = self._score_single_pair(query, evidence)
            if math.isfinite(raw_score):
                results.append((raw_score, raw_score, "cross_encoder"))
                continue

            self._logger.warning(
                "Cross-encoder returned non-finite score for query '%s'. Using fallback score.",
                query,
            )
            results.append(self._fallback_result(query, evidence, "fallback_non_finite"))
        return results

    def _score_single_pair(self, query: str, evidence: str) -> float:
        if self._model is None or self._tokenizer is None:
            return float("nan")

        try:
            inputs = self._tokenizer(
                query,
                evidence,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_length,
                padding=True,
            ).to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            logits = outputs.logits
            if logits.ndim == 0:
                return float(logits.item())
            if logits.ndim == 1:
                return float(logits[0].item())
            if logits.ndim == 2 and logits.size(-1) == 1:
                return float(logits[0, 0].item())
            if logits.ndim == 2:
                return float(logits[0, -1].item())
            return float("nan")
        except Exception as exc:
            self._logger.warning("Cross-encoder scoring failed: %s", exc)
            return float("nan")

    def _fallback_result(self, query: str, sentence: str, source: str) -> Tuple[float, float, str]:
        return float("nan"), self._fallback_score(query, sentence), source

    @staticmethod
    def _fallback_score(query: str, sentence: str) -> float:
        claim_tokens = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        sentence_tokens = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
        if not claim_tokens or not sentence_tokens:
            return 0.0
        overlap = len(claim_tokens & sentence_tokens)
        return overlap / max(len(claim_tokens), 1)
