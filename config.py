import os
import sys
from typing import Final

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
