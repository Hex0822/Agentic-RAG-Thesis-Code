"""Basic text cleaning utilities for retrieved web content."""

import html
import re
from typing import Any

import spacy

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

_NLP = spacy.blank("xx")
_NLP.add_pipe("sentencizer", config={"punct_chars": [".", "!", "?", "。", "！", "？"]})


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


def split_into_sentences(text: str) -> list[str]:
    if not text:
        return []
    doc = _NLP(text)
    return [sent.text.strip() for sent in doc.sents if sent.text and sent.text.strip()]


def build_sentence_chunks(text: str) -> list[dict[str, Any]]:
    sentences = split_into_sentences(text)
    return [{"sentence_index": idx, "text": sentence} for idx, sentence in enumerate(sentences)]


def process_search_results(search_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    processed: list[dict[str, Any]] = []
    for item in search_results:
        title = clean_text(str(item.get("title", "")))
        content = clean_text(str(item.get("content", "")))
        cleaned_text = build_cleaned_text(title, content)

        row = dict(item)
        row["title"] = title
        row["content"] = content
        row["cleaned_text"] = cleaned_text
        row["sentence_chunks"] = build_sentence_chunks(cleaned_text)
        processed.append(row)
    return processed
