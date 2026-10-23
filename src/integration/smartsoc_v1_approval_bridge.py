"""Isolated bridge from existing V1 inference responses to approval logic."""

from typing import Any

from integration.smartsoc_v1_approval_adapter import (
    process_v1_result_for_approval,
)


SAFE_FLAGS = {
    "execution_policy": "LOG_ONLY",
    "automatic_execution": False,
    "automatic_containment": False,
    "endpoint_modified": False,
}


def process_inference_response(
    original_wazuh_alert: dict[str, Any],
    receiver_response: dict[str, Any],
    *,
    reasoning_engine: Any = None,
) -> dict[str, Any]:
    """Reuse a V1 inference response with the preserved original alert."""

    if not isinstance(original_wazuh_alert, dict):
        return {
            **SAFE_FLAGS,
            "status": "BLOCKED",
            "reason": "ORIGINAL_ALERT_MUST_BE_OBJECT",
        }

    if not isinstance(receiver_response, dict):
        return {
            **SAFE_FLAGS,
            "status": "BLOCKED",
            "reason": "RECEIVER_RESPONSE_MUST_BE_OBJECT",
        }

    if receiver_response.get("status") != "INFERENCE_OK":
        return {
            **SAFE_FLAGS,
            "status": "BLOCKED",
            "reason": "V1_INFERENCE_NOT_SUCCESSFUL",
        }

    inference_result = receiver_response.get("inference_result")
    if not isinstance(inference_result, dict):
        return {
            **SAFE_FLAGS,
            "status": "BLOCKED",
            "reason": "INFERENCE_RESULT_MISSING_OR_INVALID",
        }

    try:
        return process_v1_result_for_approval(
            original_wazuh_alert,
            inference_result,
            reasoning_engine=reasoning_engine,
        )
    except Exception as exc:
        return {
            **SAFE_FLAGS,
            "status": "INTEGRATION_ERROR",
            "reason": type(exc).__name__,
        }
