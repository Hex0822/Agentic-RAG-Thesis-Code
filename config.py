import os
import sys
from typing import Final

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

_PROCESS_DIR = os.path.dirname(__file__)

CLAIM_EXTRACTION_MODEL: Final[str] = "gpt-4o-mini"
KEYWORDS_MODEL: Final[str] = "gpt-5.2"
SEARCH_TOP_K: Final[int] = 2
SEARCH_TOPIC: Final[str] = "general"
SEARCH_DEPTH: Final[str] = "basic"
SEARCH_INCLUDE_ANSWER: Final[bool] = False
SEARCH_INCLUDE_RAW_CONTENT: Final[bool] = False
SEARCH_INCLUDE_IMAGES: Final[bool] = False
LOG_LEVEL: Final[str] = "INFO"
RELEVANCE_MODEL_NAME: Final[str] = os.path.join(
    _PROCESS_DIR,
    "output_minilm_best",
    "minilm-reranker-optimized-v3",
)
RELEVANCE_BATCH_SIZE: Final[int] = 32
LLM_EVIDENCE_TOP_K: Final[int] = 5
QUALITY_MAX_ROUNDS: Final[int] = 3


def ensure_openai_env() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found.")
        sys.exit(1)


def ensure_tavily_env() -> None:
    load_dotenv()
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ ERROR: TAVILY_API_KEY not found.")
        sys.exit(1)


def create_llm(model: str, temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)


def get_log_level() -> str:
    return os.getenv("LOG_LEVEL", LOG_LEVEL).upper()
