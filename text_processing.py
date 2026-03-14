"""Basic text cleaning utilities for retrieved web content."""

import html
import re

WEB_NOISE_PATTERNS = [
    r"\bread more\b[:\s-]*",
    r"\bcontinue reading\b[:\s-]*",
    r"\bsign up\b[:\s-]*",
    r"\bsubscribe\b[:\s-]*",
    r"\bnewsletter\b[:\s-]*",
    r"\bsign in\b[:\s-]*",
    r"\blog in\b[:\s-]*",
    r"\badvertisement\b[:\s-]*",
]


def clean_text(text: str) -> str:
    if not text:
        return ""

    cleaned = html.unescape(text)
    cleaned = cleaned.replace("\u00a0", " ").replace("\ufeff", " ").replace("\ufffd", " ")

    # Remove common HTML residues.
    cleaned = re.sub(
        r"<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"<style\b[^<]*(?:(?!</style>)<[^<]*)*</style>",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    # Remove common web noise phrases.
    for pattern in WEB_NOISE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    # Remove control chars and collapse whitespace.
    cleaned = "".join(ch if ch.isprintable() else " " for ch in cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_cleaned_text(title: str, content: str) -> str:
    title = title.strip()
    content = content.strip()
    if title and content:
        return f"{title} {content}"
    return title or content


def process_search_results(search_results: list[dict[str, str]]) -> list[dict[str, str]]:
    processed: list[dict[str, str]] = []
    for item in search_results:
        title = clean_text(str(item.get("title", "")))
        content = clean_text(str(item.get("content", "")))

        row = dict(item)
        row["title"] = title
        row["content"] = content
        row["cleaned_text"] = build_cleaned_text(title, content)
        processed.append(row)
    return processed
