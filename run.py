import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from langsmith import traceable
except Exception:
    def traceable(*_args: Any, **_kwargs: Any):
        def _decorator(func):
            return func

        return _decorator

from config import (
    build_langsmith_invoke_config,
    ensure_langsmith_env,
    ensure_openai_env,
    ensure_tavily_env,
    get_langsmith_project,
    is_langsmith_tracing_enabled,
    load_project_env,
)
from pipeline import run_pipeline, set_progress_log_path
from result import save_final_result

INPUT_CLAIM = "McDonald\u2019s Azerbaijan's official account was promoting the military taking of Nagorno-Karabakh."


LOG_DIR = Path(__file__).resolve().parent / "logs"
RESULTS_TXT_PATH = Path(__file__).resolve().parent / "results.txt"


@traceable(name="fact_check_session", run_type="chain")
def _run_pipeline_traced(
    claim: str,
    invoke_config: dict[str, Any],
) -> dict[str, Any]:
    return run_pipeline(
        claim,
        show_progress=True,
        invoke_config=invoke_config,
    )


def build_nested_context_timeline(result: dict[str, Any]) -> list[dict[str, Any]]:
    context = result.get("context_management", {})
    if not isinstance(context, dict):
        return []
    rounds = context.get("rounds", [])
    if not isinstance(rounds, list):
        return []

    timeline: list[dict[str, Any]] = []
    for item in rounds:
        if not isinstance(item, dict):
            continue
        mode = str(item.get("mode", "")).strip()
        if not mode.startswith("NESTED"):
            continue

        current_var = item.get("current_nested_variable", {})
        current_var_data = current_var if isinstance(current_var, dict) else {}
        search_plan = item.get("search_plan", [])
        quick = item.get("quick_reasoning_output", {})

        timeline.append(
            {
                "round": int(item.get("round", 0)),
                "mode": mode,
                "next_action": str(item.get("next_action", "")).strip(),
                "current_variable": {
                    "variable_id": str(current_var_data.get("variable_id", "")).strip(),
                    "variable_description": str(
                        current_var_data.get("variable_description", "")
                    ).strip(),
                    "query_hint": str(current_var_data.get("query_hint", "")).strip(),
                },
                "resolved_nested_variables": item.get("resolved_nested_variables", []),
                "search_plan": search_plan if isinstance(search_plan, list) else [],
                "quick_reasoning_output": quick if isinstance(quick, dict) else {},
            }
        )
    return timeline


def _append_log_line(log_path: Path, message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


def _append_final_state(log_path: Path, result: dict[str, Any]) -> None:
    final_output: dict[str, Any] = {
        "nested_context_timeline": build_nested_context_timeline(result),
    }
    for key, value in result.items():
        final_output[key] = value

    _append_log_line(log_path, "===== FINAL STATE BEGIN =====")
    final_text = json.dumps(final_output, ensure_ascii=False, indent=2)
    for line in final_text.splitlines():
        _append_log_line(log_path, line)
    _append_log_line(log_path, "===== FINAL STATE END =====")


def _append_result_history(results_path: Path, claim: str, label: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with results_path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}]\n")
        f.write(f"Claim: {claim}\n")
        f.write(f"Result: {label}\n")
        f.write("\n")


def main() -> None:
    load_project_env()
    ensure_openai_env()
    ensure_tavily_env()
    ensure_langsmith_env()

    claim = INPUT_CLAIM.strip()
    if not claim:
        raise ValueError("INPUT_CLAIM is empty.")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_{ts}.log"

    _append_log_line(log_path, "===== RUN START =====")
    _append_log_line(log_path, f"claim: {claim}")
    langsmith_enabled = is_langsmith_tracing_enabled()
    invoke_config = build_langsmith_invoke_config(claim=claim, run_id=ts)
    if langsmith_enabled:
        _append_log_line(log_path, f"langsmith_tracing: enabled (project={get_langsmith_project()})")
    else:
        _append_log_line(log_path, "langsmith_tracing: disabled")

    set_progress_log_path(log_path)
    try:
        result = _run_pipeline_traced(
            claim=claim,
            invoke_config=invoke_config,
        )
    finally:
        set_progress_log_path(None)
    final_label_result = save_final_result(claim, result)
    _append_result_history(
        RESULTS_TXT_PATH,
        final_label_result.get("claim", ""),
        final_label_result.get("label", ""),
    )
    _append_log_line(
        log_path,
        f"final_result_variable: {json.dumps(final_label_result, ensure_ascii=False)}",
    )
    _append_final_state(log_path, result)
    _append_log_line(log_path, "===== RUN END =====")

    print(f"Claim: {final_label_result.get('claim', '')}")
    print(f"Result: {final_label_result.get('label', '')}")
    print(f"Final result variable: {final_label_result}")
    print(f"Saved LOG to: {log_path}")
    print(f"Saved result history to: {RESULTS_TXT_PATH}")


if __name__ == "__main__":
    main()
