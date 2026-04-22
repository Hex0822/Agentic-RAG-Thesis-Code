import argparse
import csv
import json
import traceback
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
DEFAULT_BATCH_OUTPUT_DIR = Path(__file__).resolve().parent


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fact-check pipeline on one or many claims.")
    parser.add_argument(
        "--claim",
        type=str,
        default="",
        help="Single claim string to process. If omitted, INPUT_CLAIM is used unless --input is set.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Batch input path. Supports .json, .jsonl, .csv, or .txt.",
    )
    parser.add_argument(
        "--claim-field",
        type=str,
        default="claim",
        help="Field name containing the claim when --input is .json/.jsonl/.csv.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Batch result CSV path. Default: process/result_<timestamp>.csv",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Only process the first N batch items. 0 means all.",
    )
    return parser.parse_args()


def _normalize_claim(value: Any) -> str:
    return str(value).strip()


def _load_claims_from_json(path: Path, claim_field: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON input must be a list, got {type(data).__name__}.")

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if isinstance(item, str):
            claim = _normalize_claim(item)
        elif isinstance(item, dict):
            claim = _normalize_claim(item.get(claim_field, ""))
        else:
            raise ValueError(f"Unsupported JSON item type at index {idx}: {type(item).__name__}")

        if not claim:
            continue
        rows.append({"row_index": idx, "claim": claim})
    return rows


def _load_claims_from_jsonl(path: Path, claim_field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, str):
                claim = _normalize_claim(item)
            elif isinstance(item, dict):
                claim = _normalize_claim(item.get(claim_field, ""))
            else:
                raise ValueError(f"Unsupported JSONL item type at line {idx + 1}: {type(item).__name__}")

            if not claim:
                continue
            rows.append({"row_index": idx, "claim": claim})
    return rows


def _load_claims_from_csv(path: Path, claim_field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if claim_field not in (reader.fieldnames or []):
            raise ValueError(f"CSV input missing claim field: {claim_field}")
        for idx, item in enumerate(reader):
            claim = _normalize_claim((item or {}).get(claim_field, ""))
            if not claim:
                continue
            rows.append({"row_index": idx, "claim": claim})
    return rows


def _load_claims_from_txt(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f):
            claim = raw_line.strip()
            if not claim:
                continue
            rows.append({"row_index": idx, "claim": claim})
    return rows


def load_batch_claims(path: Path, claim_field: str, max_items: int) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        rows = _load_claims_from_json(path, claim_field)
    elif suffix == ".jsonl":
        rows = _load_claims_from_jsonl(path, claim_field)
    elif suffix == ".csv":
        rows = _load_claims_from_csv(path, claim_field)
    elif suffix == ".txt":
        rows = _load_claims_from_txt(path)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")

    if max_items > 0:
        return rows[:max_items]
    return rows


def build_batch_output_path(output_csv: Path | None, timestamp: str) -> Path:
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        return output_csv
    DEFAULT_BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_BATCH_OUTPUT_DIR / f"result_{timestamp}.csv"


def _build_result_row(
    *,
    row_index: int,
    claim: str,
    label: str = "",
    relationship_type: str = "",
    classification_basis: str = "",
    status: str,
    error: str = "",
    log_path: Path,
) -> dict[str, str]:
    return {
        "row_index": str(row_index),
        "claim": claim,
        "label": label,
        "relationship_type": relationship_type,
        "classification_basis": classification_basis,
        "status": status,
        "error": error,
        "log_path": str(log_path),
    }


def _run_one_claim(
    *,
    claim: str,
    run_id: str,
    log_path: Path,
    append_history: bool,
) -> dict[str, Any]:
    set_progress_log_path(log_path)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _append_log_line(log_path, "===== RUN START =====")
    _append_log_line(log_path, f"claim: {claim}")
    langsmith_enabled = is_langsmith_tracing_enabled()
    invoke_config = build_langsmith_invoke_config(claim=claim, run_id=run_id)
    if langsmith_enabled:
        _append_log_line(log_path, f"langsmith_tracing: enabled (project={get_langsmith_project()})")
    else:
        _append_log_line(log_path, "langsmith_tracing: disabled")

    try:
        result = _run_pipeline_traced(
            claim=claim,
            invoke_config=invoke_config,
        )
        final_label_result = save_final_result(claim, result)
        if append_history:
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
        return {
            "claim": final_label_result.get("claim", ""),
            "label": final_label_result.get("label", ""),
            "relationship_type": str(result.get("relationship_type", "")).strip(),
            "classification_basis": str(result.get("classification_basis", "")).strip(),
            "log_path": log_path,
            "pipeline_result": result,
        }
    except Exception as exc:
        _append_log_line(log_path, f"ERROR: {exc}")
        for line in traceback.format_exc().splitlines():
            _append_log_line(log_path, line)
        _append_log_line(log_path, "===== RUN END =====")
        raise
    finally:
        set_progress_log_path(None)


def write_batch_results_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "row_index",
        "claim",
        "label",
        "relationship_type",
        "classification_basis",
        "status",
        "error",
        "log_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_single_mode(claim: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_{timestamp}.log"
    final_label_result = _run_one_claim(
        claim=claim,
        run_id=timestamp,
        log_path=log_path,
        append_history=True,
    )

    print(f"Claim: {final_label_result.get('claim', '')}")
    print(f"Result: {final_label_result.get('label', '')}")
    print(
        "Final result variable: "
        f"{{'claim': {final_label_result.get('claim', '')!r}, 'label': {final_label_result.get('label', '')!r}}}"
    )
    print(f"Saved LOG to: {log_path}")
    print(f"Saved result history to: {RESULTS_TXT_PATH}")


def run_batch_mode(input_path: Path, claim_field: str, output_csv: Path | None, max_items: int) -> None:
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    claims = load_batch_claims(input_path, claim_field, max_items)
    if not claims:
        raise ValueError(f"No valid claims found in: {input_path}")

    output_path = build_batch_output_path(output_csv, batch_timestamp)
    batch_rows: list[dict[str, str]] = []

    print(f"Loaded {len(claims)} claims from: {input_path}")

    for seq, item in enumerate(claims, start=1):
        row_index = int(item["row_index"])
        claim = str(item["claim"])
        run_id = f"{batch_timestamp}_{row_index:06d}"
        log_path = LOG_DIR / f"run_{run_id}.log"
        print(f"[{seq}/{len(claims)}] Processing claim")
        try:
            result = _run_one_claim(
                claim=claim,
                run_id=run_id,
                log_path=log_path,
                append_history=False,
            )
            batch_rows.append(
                _build_result_row(
                    row_index=row_index,
                    claim=claim,
                    label=str(result.get("label", "")),
                    relationship_type=str(result.get("relationship_type", "")),
                    classification_basis=str(result.get("classification_basis", "")),
                    status="success",
                    log_path=log_path,
                )
            )
        except Exception as exc:
            batch_rows.append(
                _build_result_row(
                    row_index=row_index,
                    claim=claim,
                    status="error",
                    error=str(exc),
                    log_path=log_path,
                )
            )

    write_batch_results_csv(output_path, batch_rows)
    success_count = sum(1 for row in batch_rows if row["status"] == "success")
    error_count = len(batch_rows) - success_count

    print(f"Saved batch results to: {output_path}")
    print(f"Success: {success_count}")
    print(f"Error: {error_count}")


def main() -> None:
    args = parse_args()
    load_project_env()
    ensure_openai_env()
    ensure_tavily_env()
    ensure_langsmith_env()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.input is not None:
        run_batch_mode(
            input_path=args.input,
            claim_field=args.claim_field,
            output_csv=args.output_csv,
            max_items=args.max_items,
        )
        return

    claim = args.claim.strip() or INPUT_CLAIM.strip()
    if not claim:
        raise ValueError("No claim provided. Use --claim or set INPUT_CLAIM.")
    run_single_mode(claim)


if __name__ == "__main__":
    main()
