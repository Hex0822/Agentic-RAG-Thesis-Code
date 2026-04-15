import os
import sys
from typing import Any, Final

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

_PROCESS_DIR = os.path.dirname(__file__)
_ENV_PATH = os.path.join(_PROCESS_DIR, ".env")

LARGE_MODEL: Final[str] = "gpt-5.4"
SMALL_MODEL: Final[str] = "gpt-4o-mini"

LARGE_TEMPERATURE: Final[float] = 0.2
LARGE_MAX_RETRIES: Final[int] = 2
LARGE_TIMEOUT: Final[float] = 60.0

SMALL_TEMPERATURE: Final[float] = 0.2
SMALL_MAX_RETRIES: Final[int] = 2
SMALL_TIMEOUT: Final[float] = 60.0

TAVILY_API_URL: Final[str] = "https://api.tavily.com/search"
TAVILY_SEARCH_DEPTH: Final[str] = "basic"
TAVILY_TOPIC: Final[str] = "general"
TAVILY_MAX_RESULTS: Final[int] = 4
TAVILY_REQUEST_TIMEOUT: Final[float] = 30.0
SUBCLAIM_SEARCH_MAX_WORKERS: Final[int] = 4
QUERY_SEARCH_MAX_WORKERS: Final[int] = 4

RERANKER_MODEL_NAME: Final[str] = "cross-encoder/ms-marco-MiniLM-L-12-v2"
RERANKER_BATCH_SIZE: Final[int] = 32
RERANKER_MAX_WORKERS: Final[int] = 4

# Number of top evidence chunks to keep for each sub-claim when passing to the ccontext stage.
CONTEXT_EVIDENCE_TOP_K: Final[int] = 5
REASONING_MAX_ROUNDS: Final[int] = 3
NESTED_MAX_UNKNOWN_RETRIES_PER_VARIABLE: Final[int] = 3


def load_project_env() -> None:
    load_dotenv(_ENV_PATH)


def ensure_openai_env() -> None:
    load_project_env()
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found.")
        sys.exit(1)


def ensure_tavily_env() -> None:
    load_project_env()
    if not os.getenv("TAVILY_API_KEY"):
        print("ERROR: TAVILY_API_KEY not found.")
        sys.exit(1)


def is_langsmith_tracing_enabled() -> bool:
    load_project_env()
    value = os.getenv("LANGSMITH_TRACING", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def ensure_langsmith_env() -> None:
    load_project_env()
    if not is_langsmith_tracing_enabled():
        return

    if not os.getenv("LANGSMITH_API_KEY", "").strip():
        print("ERROR: LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY not found.")
        sys.exit(1)

    # Keep project naming stable for easier filtering in LangSmith UI.
    os.environ.setdefault("LANGSMITH_PROJECT", get_langsmith_project())


def get_langsmith_project(default: str = "master-thesis-fact-checking") -> str:
    load_project_env()
    project = os.getenv("LANGSMITH_PROJECT", "").strip()
    if project:
        return project
    return default


def build_langsmith_invoke_config(claim: str, run_id: str) -> dict[str, Any]:
    if not is_langsmith_tracing_enabled():
        return {}

    claim_preview = " ".join(claim.strip().split())
    if len(claim_preview) > 120:
        claim_preview = claim_preview[:117] + "..."

    return {
        "run_name": "fact_check_pipeline",
        "tags": ["master-thesis", "fact-checking", f"local_run:{run_id}"],
        "metadata": {
            "claim_preview": claim_preview,
            "claim_length": len(claim),
            "local_run_id": run_id,
        },
    }


def get_tavily_api_key() -> str:
    load_project_env()
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found.")
    return api_key


def create_large_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LARGE_MODEL,
        temperature=LARGE_TEMPERATURE,
        max_retries=LARGE_MAX_RETRIES,
        timeout=LARGE_TIMEOUT,
    )


def create_small_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=SMALL_MODEL,
        temperature=SMALL_TEMPERATURE,
        max_retries=SMALL_MAX_RETRIES,
        timeout=SMALL_TIMEOUT,
    )


def create_search_planner_llm() -> ChatOpenAI:
    # Fixed stage routing: search planner uses the large model.
    return create_large_llm()


def create_nested_planner_llm() -> ChatOpenAI:
    # Fixed stage routing: nested planner uses the large model.
    return create_large_llm()


def create_reasoning_llm() -> ChatOpenAI:
    # Fixed stage routing: reasoning uses the large model.
    return create_large_llm()


def create_quick_reasoning_llm() -> ChatOpenAI:
    # Fixed stage routing: quick reasoning uses the small model.
    return create_small_llm()


def create_nested_decision_llm() -> ChatOpenAI:
    # Fixed stage routing: nested final decision uses the large model.
    return create_large_llm()
