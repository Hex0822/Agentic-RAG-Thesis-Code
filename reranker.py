"""Sentence reranking using a cross-encoder."""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from langsmith import traceable
except Exception:
    def traceable(*_args: Any, **_kwargs: Any):
        def _decorator(func):
            return func

        return _decorator

from config import RERANKER_BATCH_SIZE, RERANKER_MAX_WORKERS, RERANKER_MODEL_NAME

_MODEL_LOCK = threading.Lock()
_TOKENIZER: AutoTokenizer | None = None
_MODEL: AutoModelForSequenceClassification | None = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_reranker():
    global _TOKENIZER, _MODEL
    if _TOKENIZER is not None and _MODEL is not None:
        return _TOKENIZER, _MODEL

    with _MODEL_LOCK:
        if _TOKENIZER is None:
            _TOKENIZER = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        if _MODEL is None:
            _MODEL = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
            _MODEL.to(_DEVICE)
            _MODEL.eval()
    return _TOKENIZER, _MODEL


def _score_pairs(pairs: list[list[str]]) -> list[float]:
    tokenizer, model = _get_reranker()
    scores: list[float] = []

    for i in range(0, len(pairs), RERANKER_BATCH_SIZE):
        batch = pairs[i : i + RERANKER_BATCH_SIZE]
        texts_a = [p[0] for p in batch]
        texts_b = [p[1] for p in batch]

        inputs = tokenizer(
            texts_a,
            texts_b,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(-1).detach().cpu().tolist()

        if isinstance(logits, float):
            scores.append(float(logits))
        else:
            scores.extend(float(x) for x in logits)

    return scores


def _build_subclaim_candidates(
    search_results: list[dict[str, Any]],
    use_query_target: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}

    for result_idx, row in enumerate(search_results):
        sub_claim = str(row.get("sub_claim", "")).strip()
        if not sub_claim:
            continue
        query = str(row.get("query", "")).strip()
        target_text = query if use_query_target and query else sub_claim

        sentence_chunks = row.get("sentence_chunks", [])
        if not isinstance(sentence_chunks, list):
            continue

        for chunk in sentence_chunks:
            sentence_text = str(chunk.get("text", "")).strip()
            if not sentence_text:
                continue

            if sub_claim not in grouped:
                grouped[sub_claim] = []

            grouped[sub_claim].append(
                {
                    "target_text": target_text,
                    "result_index": result_idx,
                    "sentence_index": int(chunk.get("sentence_index", -1)),
                    "sentence_text": sentence_text,
                    "url": str(row.get("url", "")),
                    "title": str(row.get("title", "")),
                    "query": str(row.get("query", "")),
                }
            )

    return grouped


def _rerank_one_subclaim(item: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
    sub_claim, candidates = item
    if not candidates:
        return {"sub_claim": sub_claim, "ranked_sentences": []}

    pairs = []
    for c in candidates:
        target_text = str(c.get("target_text", "")).strip() or sub_claim
        pairs.append([target_text, c["sentence_text"]])
    scores = _score_pairs(pairs)

    scored: list[dict[str, Any]] = []
    for candidate, score in zip(candidates, scores):
        row = dict(candidate)
        row["score"] = float(score)
        scored.append(row)

    scored.sort(key=lambda x: x["score"], reverse=True)

    ranked: list[dict[str, Any]] = []
    for rank, row in enumerate(scored, start=1):
        ranked.append(
            {
                "global_rank": rank,
                "score": row["score"],
                "result_index": row["result_index"],
                "sentence_index": row["sentence_index"],
                "sentence_text": row["sentence_text"],
                "url": row["url"],
                "title": row["title"],
                "query": row["query"],
            }
        )

    return {"sub_claim": sub_claim, "ranked_sentences": ranked}


@traceable(name="rerank_by_subclaim", run_type="tool")
def rerank_by_subclaim(
    search_results: list[dict[str, Any]],
    use_query_target: bool = False,
) -> list[dict[str, Any]]:
    grouped = _build_subclaim_candidates(
        search_results=search_results,
        use_query_target=use_query_target,
    )
    if not grouped:
        return []

    items = list(grouped.items())
    max_workers = min(RERANKER_MAX_WORKERS, len(items))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        reranked = list(executor.map(_rerank_one_subclaim, items))

    return reranked
