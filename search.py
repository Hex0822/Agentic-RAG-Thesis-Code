import logging
from typing import Dict, List, TypedDict

from langchain_tavily import TavilySearch

from .config import (
    SEARCH_DEPTH,
    SEARCH_INCLUDE_ANSWER,
    SEARCH_INCLUDE_IMAGES,
    SEARCH_INCLUDE_RAW_CONTENT,
    SEARCH_TOP_K,
    SEARCH_TOPIC,
)


class ArticleRecord(TypedDict):
    source: str
    date: str
    content: str
    url: str
    title: str
    keyword: str


class KeywordResults(TypedDict):
    keyword: str
    articles: List[ArticleRecord]


class SearchStats(TypedDict):
    raw_count: int
    duplicate_count: int
    deduped_count: int


class TavilySearcher:
    """Step 3: search keywords and return top articles with metadata."""

    def __init__(self, top_k: int = SEARCH_TOP_K) -> None:
        self._top_k = top_k
        self._tool = TavilySearch(
            max_results=top_k,
            topic=SEARCH_TOPIC,
            search_depth=SEARCH_DEPTH,
            include_answer=SEARCH_INCLUDE_ANSWER,
            include_raw_content=SEARCH_INCLUDE_RAW_CONTENT,
            include_images=SEARCH_INCLUDE_IMAGES,
        )
        self._logger = logging.getLogger(__name__)

    def search(
        self, keywords: Dict[str, List[str]]
    ) -> tuple[Dict[str, Dict[str, List[ArticleRecord]]], Dict[str, object]]:
        results: Dict[str, Dict[str, List[ArticleRecord]]] = {}
        per_claim_stats: Dict[str, SearchStats] = {}
        per_claim_keyword_stats: Dict[str, Dict[str, SearchStats]] = {}
        total_raw = 0
        total_duplicates = 0
        total_deduped = 0
        for claim, keyword_list in keywords.items():
            claim_seen: set[str] = set()
            raw_count = 0
            duplicate_count = 0
            deduped_count = 0
            keyword_results: Dict[str, List[ArticleRecord]] = {}
            keyword_stats: Dict[str, SearchStats] = {}
            for keyword in keyword_list:
                collected: List[ArticleRecord] = []
                keyword_seen: set[str] = set()
                keyword_raw = 0
                keyword_dup = 0
                keyword_deduped = 0
                self._logger.debug("Searching keyword: %s", keyword)
                response = self._tool.invoke({"query": keyword})
                for item in _extract_items(response):
                    keyword_raw += 1
                    raw_count += 1
                    record = _normalize_article(item, keyword)
                    dedupe_key = _dedupe_key(record)
                    if dedupe_key in keyword_seen or dedupe_key in claim_seen:
                        keyword_dup += 1
                        duplicate_count += 1
                        continue
                    keyword_seen.add(dedupe_key)
                    claim_seen.add(dedupe_key)
                    collected.append(record)
                    keyword_deduped += 1
                    deduped_count += 1
                keyword_results[keyword] = collected
                keyword_stats[keyword] = {
                    "raw_count": keyword_raw,
                    "duplicate_count": keyword_dup,
                    "deduped_count": keyword_deduped,
                }
                _log_keyword_results(self._logger, claim, keyword, collected)
                self._logger.debug(
                    "Keyword search stats - raw: %s, dup: %s, deduped: %s, keyword: %s",
                    keyword_raw,
                    keyword_dup,
                    keyword_deduped,
                    keyword,
                )
            results[claim] = keyword_results
            self._logger.info(
                "Claim search stats - raw: %s, dup: %s, deduped: %s",
                raw_count,
                duplicate_count,
                deduped_count,
            )
            per_claim_stats[claim] = {
                "raw_count": raw_count,
                "duplicate_count": duplicate_count,
                "deduped_count": deduped_count,
            }
            per_claim_keyword_stats[claim] = keyword_stats
            total_raw += raw_count
            total_duplicates += duplicate_count
            total_deduped += deduped_count
        stats = {
            "per_claim": per_claim_stats,
            "per_claim_keyword": per_claim_keyword_stats,
            "total": {
                "raw_count": total_raw,
                "duplicate_count": total_duplicates,
                "deduped_count": total_deduped,
            },
        }
        return results, stats


def _extract_items(response: object) -> List[dict]:
    if isinstance(response, dict):
        items = response.get("results", [])
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def _normalize_article(item: dict, keyword: str) -> ArticleRecord:
    url = str(item.get("url") or item.get("source") or "")
    title = str(item.get("title") or "")
    source = str(item.get("source") or url)
    date = str(item.get("published_date") or item.get("date") or "")
    content = str(item.get("content") or "")
    return {
        "source": source,
        "date": date,
        "content": content,
        "url": url,
        "title": title,
        "keyword": keyword,
    }


def _dedupe_key(record: ArticleRecord) -> str:
    if record["url"]:
        return record["url"].strip().lower()
    return f"{record['source']}|{record['title']}".strip().lower()


def _log_keyword_results(
    logger: logging.Logger,
    claim: str,
    keyword: str,
    records: List[ArticleRecord],
) -> None:
    logger.debug("Claim: %s", claim)
    logger.debug("Keyword: %s (results: %s)", keyword, len(records))
    for index, record in enumerate(records, start=1):
        logger.debug(
            "  [%s] title: %s\n  source: %s\n  date: %s\n  url: %s\n  content:\n%s",
            index,
            record["title"],
            record["source"],
            record["date"],
            record["url"],
            record["content"],
        )
