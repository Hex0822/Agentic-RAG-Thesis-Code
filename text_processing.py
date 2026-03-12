import hashlib
import logging
import re
from typing import Dict, List, Optional, TypedDict

from .search import ArticleRecord


class CandidateSentence(TypedDict):
    claim: str
    keyword: str
    article_id: str
    article_title: str
    article_url: str
    article_source: str
    article_date: str
    sentence: str
    sentence_index: int
    total_sentences: int


class ProcessedArticleRecord(ArticleRecord):
    article_id: str
    sentences: List[str]


class TextProcessStats(TypedDict):
    articles_in: int
    articles_kept: int
    sentences_in: int
    sentences_kept: int
    dropped_empty: int
    dropped_too_short: int
    dropped_too_long: int
    dropped_nonalpha: int
    dropped_allcaps: int
    dropped_boilerplate: int
    dropped_duplicate: int


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_BOILERPLATE_PATTERNS = [
    re.compile(r"\bElection Results\b", re.IGNORECASE),
    re.compile(r"\bGround Game Early voting tracker\b", re.IGNORECASE),
    re.compile(r"\bWhite House Congress Supreme Court\b", re.IGNORECASE),
    re.compile(r"\bTest Your News I\.Q\.\b", re.IGNORECASE),
    re.compile(r"\bRedistricting Tracker\b", re.IGNORECASE),
    re.compile(r"\bEmail me a log in link\b", re.IGNORECASE),
]


def process_articles(
    articles: Dict[str, Dict[str, List[ArticleRecord]]]
) -> tuple[Dict[str, Dict[str, List[ProcessedArticleRecord]]], TextProcessStats, List[CandidateSentence]]:
    logger = logging.getLogger(__name__)
    coref_model = _load_coref_model(logger)

    stats: TextProcessStats = {
        "articles_in": 0,
        "articles_kept": 0,
        "sentences_in": 0,
        "sentences_kept": 0,
        "dropped_empty": 0,
        "dropped_too_short": 0,
        "dropped_too_long": 0,
        "dropped_nonalpha": 0,
        "dropped_allcaps": 0,
        "dropped_boilerplate": 0,
        "dropped_duplicate": 0,
    }
    processed: Dict[str, Dict[str, List[ProcessedArticleRecord]]] = {}
    candidates: List[CandidateSentence] = []

    for claim, keyword_map in articles.items():
        claim_seen_candidate_keys: set[str] = set()
        processed_keywords: Dict[str, List[ProcessedArticleRecord]] = {}
        for keyword, records in keyword_map.items():
            cleaned_records: List[ProcessedArticleRecord] = []
            for record in records:
                stats["articles_in"] += 1
                cleaned_text = _clean_text(record.get("content", ""))
                if not cleaned_text:
                    stats["dropped_empty"] += 1
                    continue

                resolved_text = _resolve_coref(cleaned_text, coref_model, logger)
                raw_sentences = _sentence_split(resolved_text, logger)
                stats["sentences_in"] += len(raw_sentences)

                article_id = _article_id(record)
                kept_sentences: List[str] = []
                seen_sentence_keys: set[str] = set()

                for sentence in raw_sentences:
                    sentence_text = sentence.strip()
                    if not sentence_text:
                        continue
                    if _is_boilerplate_sentence(sentence_text):
                        stats["dropped_boilerplate"] += 1
                        continue
                    if not _passes_rules(sentence_text, stats):
                        continue
                    sentence_key = _normalize_sentence_key(sentence_text)
                    if sentence_key in seen_sentence_keys:
                        stats["dropped_duplicate"] += 1
                        continue
                    seen_sentence_keys.add(sentence_key)
                    kept_sentences.append(sentence_text)

                if not kept_sentences:
                    stats["dropped_empty"] += 1
                    continue

                for index, sentence in enumerate(kept_sentences):
                    candidate_key = f"{article_id}|{_normalize_sentence_key(sentence)}"
                    if candidate_key in claim_seen_candidate_keys:
                        stats["dropped_duplicate"] += 1
                        continue
                    claim_seen_candidate_keys.add(candidate_key)
                    stats["sentences_kept"] += 1
                    candidate = _build_candidate(
                        claim=claim,
                        keyword=keyword,
                        record=record,
                        sentence=sentence,
                        sentence_index=index,
                        total_sentences=len(kept_sentences),
                    )
                    candidates.append(candidate)
                    _log_candidate(logger, candidate)

                updated = dict(record)
                updated["content"] = " ".join(kept_sentences)
                updated["article_id"] = article_id
                updated["sentences"] = kept_sentences
                cleaned_records.append(updated)  # type: ignore
                stats["articles_kept"] += 1
            processed_keywords[keyword] = cleaned_records
        processed[claim] = processed_keywords

    return processed, stats, candidates


def _clean_text(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = _CONTROL_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _resolve_coref(text: str, model: Optional[object], logger: logging.Logger) -> str:
    if model is None:
        return text
    try:
        result = model.predict([text])
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                return str(first.get("resolved_text") or first.get("resolved") or text)
            if hasattr(first, "get_resolved_text"):
                return str(first.get_resolved_text())
            if hasattr(first, "resolved_text"):
                return str(first.resolved_text)
        return text
    except Exception as exc:
        logger.warning("Coreference resolution failed; using cleaned text: %s", exc)
        return text


def _sentence_split(text: str, logger: logging.Logger) -> List[str]:
    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            logger.info("NLTK punkt not found; falling back to regex sentence split.")
            return _regex_split(text)
        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except Exception as exc:
        logger.info("NLTK not available; falling back to regex sentence split: %s", exc)
        return _regex_split(text)


def _regex_split(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _passes_rules(sentence: str, stats: TextProcessStats) -> bool:
    words = sentence.split()
    word_count = len(words)
    if word_count < 5:
        stats["dropped_too_short"] += 1
        return False
    if word_count > 80:
        stats["dropped_too_long"] += 1
        return False

    non_space = [ch for ch in sentence if not ch.isspace()]
    letters = [ch for ch in non_space if ch.isalpha()]
    if non_space:
        alpha_ratio = len(letters) / len(non_space)
        if alpha_ratio < 0.5:
            stats["dropped_nonalpha"] += 1
            return False
    if letters and all(ch.isupper() for ch in letters):
        stats["dropped_allcaps"] += 1
        return False
    return True


def _is_boilerplate_sentence(sentence: str) -> bool:
    compact = _WHITESPACE_RE.sub(" ", sentence).strip()
    if not compact:
        return True
    for pattern in _BOILERPLATE_PATTERNS:
        if pattern.search(compact):
            return True
    return False


def _normalize_sentence_key(sentence: str) -> str:
    lowered = sentence.lower()
    normalized = _NON_ALNUM_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def _build_candidate(
    claim: str,
    keyword: str,
    record: ArticleRecord,
    sentence: str,
    sentence_index: int,
    total_sentences: int,
) -> CandidateSentence:
    return {
        "claim": claim,
        "keyword": keyword,
        "article_id": _article_id(record),
        "article_title": record.get("title", ""),
        "article_url": record.get("url", ""),
        "article_source": record.get("source", ""),
        "article_date": record.get("date", ""),
        "sentence": sentence,
        "sentence_index": sentence_index,
        "total_sentences": total_sentences,
    }


def _article_id(record: ArticleRecord) -> str:
    base = f"{record.get('url','')}|{record.get('title','')}|{record.get('date','')}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _load_coref_model(logger: logging.Logger) -> Optional[object]:
    try:
        from fastcoref import FCoref
    except Exception as exc:
        logger.info("FastCoref not available; skipping coreference: %s", exc)
        return None
    try:
        return FCoref(device="cpu")
    except Exception as exc:
        logger.warning("FastCoref initialization failed; skipping coreference: %s", exc)
        return None


def _log_candidate(logger: logging.Logger, candidate: CandidateSentence) -> None:
    logger.debug(
        "Candidate sentence\n"
        "  claim: %s\n"
        "  keyword: %s\n"
        "  article_id: %s\n"
        "  title: %s\n"
        "  url: %s\n"
        "  source: %s\n"
        "  date: %s\n"
        "  sentence_index: %s/%s\n"
        "  sentence: %s",
        candidate["claim"],
        candidate["keyword"],
        candidate["article_id"],
        candidate["article_title"],
        candidate["article_url"],
        candidate["article_source"],
        candidate["article_date"],
        candidate["sentence_index"] + 1,
        candidate["total_sentences"],
        candidate["sentence"],
    )
