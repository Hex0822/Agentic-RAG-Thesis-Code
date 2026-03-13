import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# 用法
# python process/test_interface.py claim_extraction --claim "你的claim"
# python process/test_interface.py keyword_generation --subclaim "单条subclaim"
# python process/test_interface.py search --claim "你的claim"
# python process/test_interface.py text_processing --claim "你的claim"
# python process/test_interface.py relevance_ranking --claim "你的claim"
# python process/test_interface.py evidence_packaging --claim "你的claim"
# python process/test_interface.py retrieval_quality \
#   --original-claim "..." \
#   --subclaim "..." \
#   --keywords-json '["k1","k2"]' \
#   --evidence-json '[{"title":"...","url":"...","evidence":"..."}]'


def _ensure_project_root_on_path() -> None:
    project_root = os.path.dirname(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _ensure_logs_dir() -> str:
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def _configure_logging(level: int, log_path: Optional[str]) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_path:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def _dump_output(payload: Any, output_path: Optional[str]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.write("\n")
        return
    print(serialized)


def _load_json_value(value: str) -> Any:
    if os.path.isfile(value):
        with open(value, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(value)


def _get_claims_from_input(claim: Optional[str], subclaim: Optional[str]) -> List[str]:
    if subclaim:
        return [subclaim]
    if not claim:
        raise ValueError("Missing --claim or --subclaim.")
    from process.claim_extraction import ClaimExtractor
    from process.config import CLAIM_EXTRACTION_MODEL, create_llm, ensure_openai_env

    ensure_openai_env()
    extractor = ClaimExtractor(create_llm(CLAIM_EXTRACTION_MODEL, temperature=0.2))
    extraction = extractor.extract(claim)
    return extraction["sub_claims"]


def _build_keywords_for_claims(claims: List[str]) -> Dict[str, List[str]]:
    from process.config import KEYWORDS_MODEL, create_llm, ensure_openai_env
    from process.keyword_generation import KeywordsGenerator

    ensure_openai_env()
    generator = KeywordsGenerator(create_llm(KEYWORDS_MODEL, temperature=0.2))
    return {claim: generator.generate(claim) for claim in claims}


def _load_keywords(
    keywords_json: Optional[str],
    keywords_file: Optional[str],
    claims: Optional[List[str]],
) -> Dict[str, List[str]]:
    if keywords_json:
        data = _load_json_value(keywords_json)
    elif keywords_file:
        data = _load_json_value(keywords_file)
    elif claims is not None:
        return _build_keywords_for_claims(claims)
    else:
        raise ValueError("Missing keywords input.")

    if not isinstance(data, dict):
        raise ValueError("Keywords must be a JSON object of {claim: [keywords]}.")
    return {str(k): [str(item) for item in v] for k, v in data.items() if isinstance(v, list)}


def _load_articles(
    articles_json: Optional[str],
    articles_file: Optional[str],
    keywords: Optional[Dict[str, List[str]]],
) -> Dict[str, Dict[str, List[dict]]]:
    if articles_json:
        data = _load_json_value(articles_json)
    elif articles_file:
        data = _load_json_value(articles_file)
    elif keywords is not None:
        from process.config import ensure_tavily_env
        from process.search import TavilySearcher

        ensure_tavily_env()
        searcher = TavilySearcher()
        data, _ = searcher.search(keywords)
    else:
        raise ValueError("Missing articles input.")

    if not isinstance(data, dict):
        raise ValueError("Articles must be a JSON object of {claim: {keyword: [records]}}.")
    return data  # type: ignore[return-value]


def _build_single_article_payload(
    article: dict,
    claim: str,
    keyword: str,
) -> Dict[str, Dict[str, List[dict]]]:
    if not article.get("content"):
        raise ValueError("Article content is required for text_processing.")
    record = {
        "source": str(article.get("source") or article.get("url") or ""),
        "date": str(article.get("published_date") or article.get("date") or ""),
        "content": str(article.get("content") or ""),
        "url": str(article.get("url") or ""),
        "title": str(article.get("title") or ""),
        "keyword": keyword,
    }
    return {claim: {keyword: [record]}}


def run_claim_extraction(args: argparse.Namespace) -> None:
    if args.subclaim:
        _dump_output(
            {
                "relationship_type": "ATOMIC",
                "classification_basis": "Subclaim provided directly; no classification needed.",
                "sub_claims": [args.subclaim],
            },
            args.output,
        )
        return
    if not args.claim:
        raise ValueError("Missing --claim.")
    from process.claim_extraction import ClaimExtractor
    from process.config import CLAIM_EXTRACTION_MODEL, create_llm, ensure_openai_env

    ensure_openai_env()
    extractor = ClaimExtractor(create_llm(CLAIM_EXTRACTION_MODEL, temperature=0.2))
    extraction = extractor.extract(args.claim)
    _dump_output(extraction, args.output)


def run_keyword_generation(args: argparse.Namespace) -> None:
    claims = _get_claims_from_input(args.claim, args.subclaim)
    keywords = _build_keywords_for_claims(claims)
    _dump_output({"keywords": keywords}, args.output)


def run_search(args: argparse.Namespace) -> None:
    claims = _get_claims_from_input(args.claim, args.subclaim) if not args.keywords_json and not args.keywords_file else None
    keywords = _load_keywords(args.keywords_json, args.keywords_file, claims)
    from process.config import ensure_tavily_env
    from process.search import TavilySearcher

    ensure_tavily_env()
    searcher = TavilySearcher()
    articles, stats = searcher.search(keywords)
    _dump_output({"keywords": keywords, "articles": articles, "stats": stats}, args.output)


def run_text_processing(args: argparse.Namespace) -> None:
    if args.article_json or args.article_file:
        article_payload = _load_json_value(args.article_json or args.article_file)
        if isinstance(article_payload, str):
            article_payload = {"content": article_payload}
        if not isinstance(article_payload, dict):
            raise ValueError("Article must be a JSON object (or string content).")
        claim_value = args.claim or args.subclaim or "input_claim"
        keyword_value = args.keyword or "manual_input"
        articles = _build_single_article_payload(article_payload, claim_value, keyword_value)
    elif args.articles_json or args.articles_file:
        articles = _load_articles(args.articles_json, args.articles_file, None)
    else:
        raise ValueError("Provide --article-json/--article-file or --articles-json/--articles-file for text_processing.")
    from process.text_processing import process_articles

    processed, stats, candidates = process_articles(articles)
    _dump_output({"articles": processed, "candidates": candidates, "stats": stats}, args.output)


def run_relevance_ranking(args: argparse.Namespace) -> None:
    candidates = None
    if args.candidates_json or args.candidates_file:
        payload = _load_json_value(args.candidates_json or args.candidates_file)
        if not isinstance(payload, list):
            raise ValueError("Candidates must be a JSON list.")
        candidates = payload
    else:
        claims = _get_claims_from_input(args.claim, args.subclaim)
        keywords = _load_keywords(args.keywords_json, args.keywords_file, claims)
        articles = _load_articles(args.articles_json, args.articles_file, keywords)
        from process.text_processing import process_articles

        _, _, candidates = process_articles(articles)

    from process.config import RELEVANCE_BATCH_SIZE, RELEVANCE_MODEL_NAME
    from process.relevance_ranking import MiniLMRelevanceRanker

    ranker = MiniLMRelevanceRanker(
        model_name=RELEVANCE_MODEL_NAME,
        batch_size=RELEVANCE_BATCH_SIZE,
    )
    ranked, stats = ranker.rank(candidates)  # type: ignore[arg-type]
    _dump_output({"ranked_evidence": ranked, "stats": stats}, args.output)


def run_evidence_packaging(args: argparse.Namespace) -> None:
    ranked_evidence = None
    if args.ranked_json or args.ranked_file:
        ranked_payload = _load_json_value(args.ranked_json or args.ranked_file)
        if not isinstance(ranked_payload, dict):
            raise ValueError("Ranked evidence must be a JSON object.")
        ranked_evidence = ranked_payload

    articles = None
    if args.articles_json or args.articles_file:
        articles = _load_articles(args.articles_json, args.articles_file, None)

    if ranked_evidence is None or articles is None:
        claims = _get_claims_from_input(args.claim, args.subclaim)
        keywords = _load_keywords(args.keywords_json, args.keywords_file, claims)
        raw_articles = _load_articles(args.articles_json, args.articles_file, keywords)
        from process.text_processing import process_articles
        from process.config import RELEVANCE_BATCH_SIZE, RELEVANCE_MODEL_NAME
        from process.relevance_ranking import MiniLMRelevanceRanker

        processed, _, candidates = process_articles(raw_articles)
        ranker = MiniLMRelevanceRanker(
            model_name=RELEVANCE_MODEL_NAME,
            batch_size=RELEVANCE_BATCH_SIZE,
        )
        ranked_evidence, _ = ranker.rank(candidates)
        articles = processed

    from process.config import LLM_EVIDENCE_TOP_K
    from process.evidence_packaging import prepare_llm_evidence

    llm_evidence, stats = prepare_llm_evidence(
        ranked_evidence=ranked_evidence,
        articles=articles,
        top_k=LLM_EVIDENCE_TOP_K,
    )
    _dump_output({"llm_evidence": llm_evidence, "stats": stats}, args.output)


def run_retrieval_quality(args: argparse.Namespace) -> None:
    if not args.original_claim or not args.subclaim:
        raise ValueError("--original-claim and --subclaim are required for retrieval_quality.")
    if not args.evidence_json and not args.evidence_file:
        raise ValueError("--evidence-json or --evidence-file is required for retrieval_quality.")
    keywords = _load_json_value(args.keywords_json or args.keywords_file or "[]")
    if not isinstance(keywords, list):
        raise ValueError("Keywords must be a JSON list.")
    evidence_items = _load_json_value(args.evidence_json or args.evidence_file)
    if not isinstance(evidence_items, list):
        raise ValueError("Evidence must be a JSON list.")

    from process.config import KEYWORDS_MODEL, QUALITY_MAX_ROUNDS, create_llm, ensure_openai_env
    from process.retrieval_quality import RetrievalQualityAgent

    ensure_openai_env()
    agent = RetrievalQualityAgent(create_llm(KEYWORDS_MODEL, temperature=0.2))
    keyword_history = (
        _load_json_value(args.keyword_history_json)
        if args.keyword_history_json
        else [keywords]
    )
    decision = agent.evaluate(
        original_claim=args.original_claim,
        subclaim=args.subclaim,
        current_keywords=keywords,
        evidence_items=evidence_items,
        round_idx=args.round_idx,
        max_rounds=args.max_rounds or QUALITY_MAX_ROUNDS,
        keyword_history=keyword_history,
    )
    _dump_output({"decision": decision}, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test individual modules in the process pipeline.")
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--output", default=None)

    subparsers = parser.add_subparsers(dest="command", required=True)

    common_claim = argparse.ArgumentParser(add_help=False)
    common_claim.add_argument("--claim", default=None)
    common_claim.add_argument("--subclaim", default=None)

    keyword_inputs = argparse.ArgumentParser(add_help=False)
    keyword_inputs.add_argument("--keywords-json", default=None)
    keyword_inputs.add_argument("--keywords-file", default=None)

    articles_inputs = argparse.ArgumentParser(add_help=False)
    articles_inputs.add_argument("--articles-json", default=None)
    articles_inputs.add_argument("--articles-file", default=None)

    subparsers.add_parser("claim_extraction", parents=[common_claim])
    subparsers.add_parser("keyword_generation", parents=[common_claim])
    subparsers.add_parser("search", parents=[common_claim, keyword_inputs])
    text_parser = subparsers.add_parser("text_processing", parents=[common_claim, articles_inputs])
    text_parser.add_argument("--article-json", default=None)
    text_parser.add_argument("--article-file", default=None)
    text_parser.add_argument("--keyword", default=None)

    rank_parser = subparsers.add_parser("relevance_ranking", parents=[common_claim, keyword_inputs, articles_inputs])
    rank_parser.add_argument("--candidates-json", default=None)
    rank_parser.add_argument("--candidates-file", default=None)

    pack_parser = subparsers.add_parser("evidence_packaging", parents=[common_claim, keyword_inputs, articles_inputs])
    pack_parser.add_argument("--ranked-json", default=None)
    pack_parser.add_argument("--ranked-file", default=None)

    quality_parser = subparsers.add_parser("retrieval_quality")
    quality_parser.add_argument("--original-claim", required=True)
    quality_parser.add_argument("--subclaim", required=True)
    quality_parser.add_argument("--keywords-json", default=None)
    quality_parser.add_argument("--keywords-file", default=None)
    quality_parser.add_argument("--evidence-json", default=None)
    quality_parser.add_argument("--evidence-file", default=None)
    quality_parser.add_argument("--keyword-history-json", default=None)
    quality_parser.add_argument("--round-idx", type=int, default=1)
    quality_parser.add_argument("--max-rounds", type=int, default=None)

    return parser


def main() -> None:
    _ensure_project_root_on_path()
    from process.config import get_log_level

    parser = build_parser()
    args = parser.parse_args()

    level_name = (args.log_level or get_log_level()).upper()
    level = getattr(logging, level_name, logging.INFO)
    log_path = args.log_file
    if log_path is None:
        logs_dir = _ensure_logs_dir()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(logs_dir, f"test_interface_{stamp}.log")
    _configure_logging(level, log_path)

    if args.command == "claim_extraction":
        run_claim_extraction(args)
    elif args.command == "keyword_generation":
        run_keyword_generation(args)
    elif args.command == "search":
        run_search(args)
    elif args.command == "text_processing":
        run_text_processing(args)
    elif args.command == "relevance_ranking":
        run_relevance_ranking(args)
    elif args.command == "evidence_packaging":
        run_evidence_packaging(args)
    elif args.command == "retrieval_quality":
        run_retrieval_quality(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
