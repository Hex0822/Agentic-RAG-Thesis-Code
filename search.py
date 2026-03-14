"""Tavily search utilities."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

from config import (
    QUERY_SEARCH_MAX_WORKERS,
    TAVILY_API_URL,
    TAVILY_MAX_RESULTS,
    TAVILY_REQUEST_TIMEOUT,
    TAVILY_SEARCH_DEPTH,
    TAVILY_TOPIC,
    get_tavily_api_key,
)


def _dedupe_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for query in queries:
        key = query.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(query.strip())
    return deduped


def _dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for row in rows:
        url = row.get("url", "").strip()
        title = row.get("title", "").strip()
        content = row.get("content", "").strip()
        # Prefer URL-based dedupe; fallback to title+content when URL is missing.
        key = url.lower() if url else f"{title.lower()}|{content.lower()}"
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _search_one_query(sub_claim: str, query: str) -> list[dict[str, str]]:
    payload = {
        "api_key": get_tavily_api_key(),
        "query": query,
        "topic": TAVILY_TOPIC,
        "search_depth": TAVILY_SEARCH_DEPTH,
        "max_results": TAVILY_MAX_RESULTS,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }

    response = requests.post(TAVILY_API_URL, json=payload, timeout=TAVILY_REQUEST_TIMEOUT)
    response.raise_for_status()
    data: dict[str, Any] = response.json()

    rows: list[dict[str, str]] = []
    for item in data.get("results", []):
        rows.append(
            {
                "sub_claim": sub_claim,
                "query": query,
                "title": str(item.get("title", "")),
                "content": str(item.get("content", "")),
                "url": str(item.get("url", "")),
            }
        )
    return rows


def search_subclaim_queries(sub_claim: str, queries: list[str]) -> list[dict[str, str]]:
    clean_queries = _dedupe_queries([q.strip() for q in queries if q and q.strip()])
    if not clean_queries:
        return []

    max_workers = min(QUERY_SEARCH_MAX_WORKERS, len(clean_queries))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        nested_rows = list(executor.map(lambda q: _search_one_query(sub_claim, q), clean_queries))

    rows: list[dict[str, str]] = []
    for batch in nested_rows:
        rows.extend(batch)
    return _dedupe_rows(rows)
