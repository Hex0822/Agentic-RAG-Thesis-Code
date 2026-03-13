import os
import sys
from typing import Final

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

_PROCESS_DIR = os.path.dirname(__file__)
_ENV_PATH = os.path.join(_PROCESS_DIR, ".env")

LARGE_MODEL: Final[str] = "gpt-5.4"
SMALL_MODEL: Final[str] = "gpt-4o-mini"

LLM_TEMPERATURE: Final[float] = 0.0
LLM_MAX_RETRIES: Final[int] = 2
LLM_TIMEOUT: Final[float] = 60.0

# Fixed routing by module (not runtime-selectable).
CLAIM_ANALYSIS_MODEL: Final[str] = SMALL_MODEL


def load_project_env() -> None:
    load_dotenv(_ENV_PATH)


def ensure_openai_env() -> None:
    load_project_env()
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found.")
        sys.exit(1)


def create_llm(
    model: str,
    temperature: float = LLM_TEMPERATURE,
    max_retries: int = LLM_MAX_RETRIES,
    timeout: float | None = LLM_TIMEOUT,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        timeout=timeout,
    )


def create_large_llm() -> ChatOpenAI:
    return create_llm(LARGE_MODEL)


def create_small_llm() -> ChatOpenAI:
    return create_llm(SMALL_MODEL)


def create_claim_analysis_llm() -> ChatOpenAI:
    return create_llm(CLAIM_ANALYSIS_MODEL)
