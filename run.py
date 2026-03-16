"""Direct run entrypoint (no CLI args)."""

import json
from datetime import datetime
from pathlib import Path

from config import ensure_openai_env, ensure_tavily_env, load_project_env
from pipeline import run_pipeline

INPUT_CLAIM = "The American film, television and theater actress who was a star in the film the Matchmaker and also received the 40th AFI Life Achievement Award was born April 24, 1934."
LOG_DIR = Path(__file__).resolve().parent / "logs"


def main() -> None:
    load_project_env()
    ensure_openai_env()
    ensure_tavily_env()

    claim = INPUT_CLAIM.strip()
    if not claim:
        raise ValueError("INPUT_CLAIM is empty.")

    result = run_pipeline(
        claim,
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = LOG_DIR / f"run_{ts}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved to: {output_path}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
