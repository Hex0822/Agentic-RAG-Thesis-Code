"""Store final claim + final reasoning label for experiments."""

from typing import Any

FINAL_RESULT: dict[str, str] = {
    "claim": "",
    "label": "",
}
_DEFAULT_LABEL = "Not Enough Evidence"


def _extract_final_label(pipeline_result: dict[str, Any]) -> str:
    context = pipeline_result.get("context_management", {})
    if isinstance(context, dict):
        rounds = context.get("rounds", [])
        if isinstance(rounds, list):
            for item in reversed(rounds):
                if not isinstance(item, dict):
                    continue
                nested_decision = item.get("nested_decision_output", {})
                if isinstance(nested_decision, dict):
                    nested_label = str(nested_decision.get("label", "")).strip()
                    if nested_label:
                        return nested_label
                reasoning = item.get("reasoning_output", {})
                if not isinstance(reasoning, dict):
                    continue
                label = str(reasoning.get("label", "")).strip()
                if label:
                    return label
        action = str(context.get("overall_next_action", "")).strip().upper()
        if action == "NESTED_BLOCKED":
            return _DEFAULT_LABEL

    nested_decision_output = pipeline_result.get("nested_decision_output", {})
    if isinstance(nested_decision_output, dict):
        nested_label = str(nested_decision_output.get("label", "")).strip()
        if nested_label:
            return nested_label

    reasoning_output = pipeline_result.get("reasoning_output", {})
    if isinstance(reasoning_output, dict):
        label = str(reasoning_output.get("label", "")).strip()
        if label:
            return label
    return _DEFAULT_LABEL


def save_final_result(claim: str, pipeline_result: dict[str, Any]) -> dict[str, str]:
    FINAL_RESULT["claim"] = claim.strip()
    FINAL_RESULT["label"] = _extract_final_label(pipeline_result)
    return dict(FINAL_RESULT)
