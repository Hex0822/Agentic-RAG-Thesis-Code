#!/usr/bin/env python3
"""Evaluate process result CSV files against a gold JSON dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PREFERRED_LABEL_ORDER = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a process result CSV against the gold JSON input dataset."
    )
    parser.add_argument(
        "--pred",
        type=Path,
        required=True,
        help="Prediction CSV path, e.g. process/result_<timestamp>.csv",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        required=True,
        help="Gold JSON path used as input for the run.",
    )
    parser.add_argument(
        "--label-field",
        default="label",
        help="Gold label field name in the JSON dataset.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the metrics as JSON.",
    )
    parser.add_argument(
        "--output-comparison-csv",
        type=Path,
        default=None,
        help=(
            "Optional path to save a row-level comparison CSV. "
            "Default: alongside --pred with suffix _comparison.csv."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional explicit label order. Default infers from gold labels.",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Print failed row indices and claims.",
    )
    return parser.parse_args()


def load_gold_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Gold JSON must be a list, got {type(data).__name__}.")
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Gold row {idx} is not an object.")
        rows.append(item)
    return rows


def load_pred_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    required_columns = {"row_index", "label", "status"}
    missing = required_columns - set(rows[0].keys())
    if missing:
        raise ValueError(f"Prediction CSV missing required columns: {sorted(missing)}")
    return rows


def ordered_labels_from_gold(gold_rows: list[dict[str, Any]], label_field: str) -> list[str]:
    labels = {
        str(row.get(label_field, "")).strip()
        for row in gold_rows
        if str(row.get(label_field, "")).strip()
    }
    ordered = [label for label in PREFERRED_LABEL_ORDER if label in labels]
    ordered.extend(sorted(label for label in labels if label not in ordered))
    return ordered


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def default_comparison_csv_path(pred_path: Path) -> Path:
    return pred_path.with_name(f"{pred_path.stem}_comparison.csv")


def build_metrics(
    *,
    pred_rows: list[dict[str, str]],
    gold_rows: list[dict[str, Any]],
    labels: list[str],
    label_field: str,
) -> dict[str, Any]:
    label_set = set(labels)
    aligned_rows: list[dict[str, Any]] = []

    for csv_row in pred_rows:
        raw_index = str(csv_row.get("row_index", "")).strip()
        if not raw_index:
            raise ValueError(f"Missing row_index in prediction row: {csv_row}")
        try:
            row_index = int(raw_index)
        except ValueError as exc:
            raise ValueError(f"Invalid row_index: {raw_index!r}") from exc
        if row_index < 0 or row_index >= len(gold_rows):
            raise ValueError(f"row_index {row_index} is out of range for gold dataset.")

        gold_row = gold_rows[row_index]
        gold_label = str(gold_row.get(label_field, "")).strip()
        pred_label = str(csv_row.get("label", "")).strip()
        status = str(csv_row.get("status", "")).strip().lower()
        claim = str(csv_row.get("claim", "")).strip() or str(gold_row.get("claim", "")).strip()
        gold_adding_type = str(gold_row.get("adding_types", "")).strip()
        gold_claim_types_raw = gold_row.get("claim_types", [])
        gold_claim_types: list[str] = []
        if isinstance(gold_claim_types_raw, list):
            gold_claim_types = [str(item).strip() for item in gold_claim_types_raw if str(item).strip()]
        pred_relationship_type = str(csv_row.get("relationship_type", "")).strip()
        error = str(csv_row.get("error", "")).strip()
        is_success = status == "success" and pred_label in label_set and gold_label in label_set
        is_correct = is_success and gold_label == pred_label

        aligned_rows.append(
            {
                "row_index": row_index,
                "claim": claim,
                "gold": gold_label,
                "pred": pred_label,
                "status": status,
                "is_success": is_success,
                "is_correct": is_correct,
                "gold_adding_type": gold_adding_type,
                "gold_claim_types": gold_claim_types,
                "pred_relationship_type": pred_relationship_type,
                "error": error,
            }
        )

    total = len(aligned_rows)
    success_rows = [row for row in aligned_rows if row["is_success"]]
    error_rows = [row for row in aligned_rows if not row["is_success"]]

    success_correct = sum(row["gold"] == row["pred"] for row in success_rows)
    all_correct = sum(row["gold"] == row["pred"] for row in aligned_rows)

    confusion_matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    for row in success_rows:
        confusion_matrix[row["gold"]][row["pred"]] += 1

    per_class: dict[str, dict[str, Any]] = {}
    supports: dict[str, int] = {}
    for label in labels:
        tp = sum(
            1 for row in success_rows if row["gold"] == label and row["pred"] == label
        )
        fp = sum(
            1 for row in success_rows if row["gold"] != label and row["pred"] == label
        )
        fn = sum(
            1 for row in success_rows if row["gold"] == label and row["pred"] != label
        )
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        support = sum(confusion_matrix[label].values())
        supports[label] = support
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro_f1 = sum(item["f1"] for item in per_class.values()) / len(labels) if labels else 0.0
    weighted_denominator = sum(supports.values())
    weighted_f1 = (
        sum(per_class[label]["f1"] * supports[label] for label in labels) / weighted_denominator
        if weighted_denominator
        else 0.0
    )

    metrics = {
        "total_rows": total,
        "success_rows": len(success_rows),
        "error_rows": len(error_rows),
        "coverage": len(success_rows) / total if total else 0.0,
        "accuracy_success_only": (
            success_correct / len(success_rows) if success_rows else 0.0
        ),
        "accuracy_end_to_end": all_correct / total if total else 0.0,
        "macro_f1_success_only": macro_f1,
        "weighted_f1_success_only": weighted_f1,
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix_success_only": confusion_matrix,
        "error_rows_detail": [
            {
                "row_index": row["row_index"],
                "claim": row["claim"],
                "gold": row["gold"],
                "pred": row["pred"],
                "status": row["status"],
                "pred_relationship_type": row["pred_relationship_type"],
                "gold_adding_type": row["gold_adding_type"],
                "error": row["error"],
            }
            for row in error_rows
        ],
        "comparison_rows": [
            {
                "row_index": row["row_index"],
                "claim": row["claim"],
                "gold_adding_type": row["gold_adding_type"],
                "gold_claim_types": " | ".join(row["gold_claim_types"]),
                "pred_relationship_type": row["pred_relationship_type"],
                "system_label": row["pred"],
                "gold_label": row["gold"],
                "status": row["status"],
                "is_success": row["is_success"],
                "is_correct": row["is_correct"],
                "error": row["error"],
            }
            for row in aligned_rows
        ],
    }
    return metrics


def write_comparison_csv(path: Path, comparison_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_index",
        "claim",
        "gold_adding_type",
        "gold_claim_types",
        "pred_relationship_type",
        "system_label",
        "gold_label",
        "status",
        "is_success",
        "is_correct",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)


def print_metrics(metrics: dict[str, Any], show_errors: bool) -> None:
    print(f"total_rows: {metrics['total_rows']}")
    print(f"success_rows: {metrics['success_rows']}")
    print(f"error_rows: {metrics['error_rows']}")
    print(f"coverage: {metrics['coverage']:.4f}")
    print(f"accuracy_success_only: {metrics['accuracy_success_only']:.4f}")
    print(f"accuracy_end_to_end: {metrics['accuracy_end_to_end']:.4f}")
    print(f"macro_f1_success_only: {metrics['macro_f1_success_only']:.4f}")
    print(f"weighted_f1_success_only: {metrics['weighted_f1_success_only']:.4f}")

    print("\nper-class:")
    for label in metrics["labels"]:
        item = metrics["per_class"][label]
        print(
            f"- {label}: "
            f"precision={item['precision']:.4f}, "
            f"recall={item['recall']:.4f}, "
            f"f1={item['f1']:.4f}, "
            f"support={item['support']}"
        )

    print("\nconfusion_matrix_success_only:")
    labels = metrics["labels"]
    print("\t" + "\t".join(labels))
    for gold_label in labels:
        row = metrics["confusion_matrix_success_only"][gold_label]
        print(gold_label + "\t" + "\t".join(str(row[pred_label]) for pred_label in labels))

    if show_errors:
        print("\nerror_rows_detail:")
        for item in metrics["error_rows_detail"]:
            print(
                f"- row_index={item['row_index']}, status={item['status']}, "
                f"gold={item['gold']}, pred={item['pred']}, claim={item['claim']}"
            )


def main() -> None:
    args = parse_args()
    gold_rows = load_gold_rows(args.gold)
    pred_rows = load_pred_rows(args.pred)
    labels = list(args.labels) if args.labels else ordered_labels_from_gold(gold_rows, args.label_field)
    metrics = build_metrics(
        pred_rows=pred_rows,
        gold_rows=gold_rows,
        labels=labels,
        label_field=args.label_field,
    )

    print_metrics(metrics, show_errors=args.show_errors)

    comparison_csv_path = (
        args.output_comparison_csv
        if args.output_comparison_csv is not None
        else default_comparison_csv_path(args.pred)
    )
    write_comparison_csv(comparison_csv_path, metrics["comparison_rows"])
    print(f"\nSaved comparison CSV: {comparison_csv_path}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nSaved metrics JSON: {args.output_json}")


if __name__ == "__main__":
    main()
