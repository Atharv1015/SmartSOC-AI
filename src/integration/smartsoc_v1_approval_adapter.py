"""V1 inference-to-human-approval adapter. Never executes mitigation."""
from __future__ import annotations

from typing import Any

from autonomous.autonomous_reasoning_v1 import AutonomousReasoningEngine
from integration.local_approval_console_v1 import (
    register_pending_from_reasoning_result,
)

EXECUTION_POLICY = "LOG_ONLY"


def process_v1_result_for_approval(
    original_event: dict[str, Any],
    inference_result: dict[str, Any],
    *,
    reasoning_engine: Any = None,
) -> dict[str, Any]:
    """Evaluate an existing V1 result and register eligible requests only."""
    safe = {
        "execution_policy": EXECUTION_POLICY,
        "automatic_execution": False,
        "automatic_containment": False,
        "endpoint_modified": False,
    }

    if not isinstance(original_event, dict):
        return {**safe, "status": "REJECTED",
                "reason": "EVENT_MUST_BE_OBJECT"}

    if not isinstance(inference_result, dict):
        return {**safe, "status": "REJECTED",
                "reason": "INFERENCE_RESULT_MUST_BE_OBJECT"}

    raw = original_event.get("raw_wazuh_alert")
    if not isinstance(raw, dict):
        raw = original_event.get("wazuh")
    if not isinstance(raw, dict):
        raw = original_event

    # Require original nested Windows event context. Do not invent a PID,
    # process name, or command line from model predictions.
    data = raw.get("data")
    if not isinstance(data, dict):
        data = {}
    win = data.get("win")
    if not isinstance(win, dict):
        win = {}
    eventdata = win.get("eventdata")
    if not isinstance(eventdata, dict):
        eventdata = {}

    rule = raw.get("rule")
    if not isinstance(rule, dict):
        rule = original_event.get("rule")
    if not isinstance(rule, dict):
        rule = {}

    alert_id = (
        original_event.get("alert_id")
        or original_event.get("id")
        or raw.get("id")
    )
    rule_id = rule.get("id") or original_event.get("rule_id")

    if not alert_id or not rule_id:
        return {
            **safe,
            "status": "BLOCKED",
            "reason": "MISSING_ALERT_OR_RULE_ID",
        }

    # Reason only over the existing inference output and the preserved
    # original Wazuh event; do not run inference a second time.
    try:
        engine = reasoning_engine or AutonomousReasoningEngine()
        reasoning = engine.reason(inference_result, raw)
    except Exception as exc:
        return {
            **safe,
            "status": "INTEGRATION_ERROR",
            "reason": type(exc).__name__,
        }

    if not isinstance(reasoning, dict):
        return {**safe, "status": "BLOCKED",
                "reason": "INVALID_REASONING_RESULT"}

    action = str(reasoning.get("mitigation", "NONE")).strip().upper()
    base = {
        **safe,
        "alert_id": str(alert_id),
        "rule_id": str(rule_id),
        "requested_action": action,
    }

    if action != "PROCESS_TERMINATION":
        return {
            **base,
            "status": "NO_APPROVAL_REQUIRED",
            "reason": "NO_SUPPORTED_TERMINATION_REQUESTED",
        }

    process_name = str(
        reasoning.get("process_name")
        or eventdata.get("image")
        or ""
    ).strip()

    process_id = reasoning.get("process_id")
    if process_id in (None, ""):
        process_id = reasoning.get("pid")

    try:
        pid = int(process_id)
    except (TypeError, ValueError):
        pid = 0

    # Bind the proposed action to the exact process in the original alert.
    event_process_name = str(eventdata.get("image") or "").strip()
    event_process_id = eventdata.get("processId")

    if not process_name or pid <= 0 or not event_process_name:
        return {
            **base,
            "status": "BLOCKED",
            "reason": "INCOMPLETE_PROCESS_BINDING",
        }

    try:
        event_pid = int(event_process_id)
    except (TypeError, ValueError):
        event_pid = 0

    if process_name.casefold() != event_process_name.casefold():
        return {
            **base,
            "status": "BLOCKED",
            "reason": "PROCESS_NAME_MISMATCH",
        }

    if event_pid <= 0 or pid != event_pid:
        return {
            **base,
            "status": "BLOCKED",
            "reason": "PROCESS_ID_MISMATCH",
        }

    bound = dict(reasoning)
    bound.update({
        "mitigation": "PROCESS_TERMINATION",
        "process_name": process_name,
        "process_id": pid,
        "pid": pid,
        "alert_id": str(alert_id),
        "rule_id": str(rule_id),
    })

    try:
        registration = register_pending_from_reasoning_result(
            bound, str(alert_id), str(rule_id)
        )
    except Exception as exc:
        return {
            **base,
            "status": "INTEGRATION_ERROR",
            "reason": type(exc).__name__,
        }

    if not isinstance(registration, dict) or not registration.get("registered"):
        reason = (
            registration.get("reason", "APPROVAL_REGISTRATION_FAILED")
            if isinstance(registration, dict)
            else "INVALID_REGISTRATION_RESULT"
        )
        return {**base, "status": "BLOCKED", "reason": reason}

    return {
        **base,
        "status": "PENDING_APPROVAL",
        "approval_request_id": registration.get("request_id"),
        "approval_status": registration.get("status", "PENDING"),
    }


