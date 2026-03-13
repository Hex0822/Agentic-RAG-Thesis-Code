"""Direct run entrypoint (no CLI args)."""

import json

from config import ensure_openai_env, load_project_env
from pipeline import run_pipeline

INPUT_CLAIM = "Elon Musk founded SpaceX in 2002 and later acquired Twitter."
COMPACT_OUTPUT = False


def main() -> None:
    load_project_env()
    ensure_openai_env()

    claim = INPUT_CLAIM.strip()
    if not claim:
        raise ValueError("INPUT_CLAIM is empty.")

    result = run_pipeline(
        claim,
    )
    if COMPACT_OUTPUT:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
