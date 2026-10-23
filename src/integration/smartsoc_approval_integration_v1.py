"""Safe SmartSOC inference-to-approval integration.

This module never executes mitigation. It only registers eligible
PROCESS_TERMINATION decisions for explicit human review.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autonomous.autonomous_reasoning_v1 import AutonomousReasoningEngine
from integration.local_approval_console_v1 import (
    register_pending_from_reasoning_result,
)
from production_inference_windows_apt_v4_1 import SmartSOCInference

# Reuse the V2 receiver's event conversion contract without starting its server.
from integration.smartsoc_live_receiver_v2 import normalized_to_wazuh_event

POLICY = "LOG_ONLY"
AUTOMATIC_EXECUTION = False
AUTOMATIC_CONTAINMENT = False


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _alert_and_rule(event: dict, wazuh_event: dict) -> tuple[str, str]:
    raw = _dict(event.get("raw_wazuh_alert"))
    legacy = _dict(event.get("wazuh"))
    alert_id = (
        event.get("alert_id")
        or event.get("id")
        or raw.get("id")
        or legacy.get("alert_id")
        or wazuh_event.get("id")
        or ""
    )
    rule = _dict(event.get("rule")) or _dict(wazuh_event.get("rule"))
    rule_id = rule.get("id") or event.get("rule_id") or ""
    return str(alert_id).strip(), str(rule_id).strip()


def process_for_approval(
    event: dict,
    *,
    inference_engine=None,
    reasoning_engine=None,
) -> dict:
    """Infer and reason; register approval only. Never execute an action."""
    if not isinstance(event, dict):
        return {
            "status": "REJECTED",
            "reason": "EVENT_MUST_BE_OBJECT",
            "endpoint_modified": False,
        }

    inference_engine = inference_engine or SmartSOCInference()
    reasoning_engine = reasoning_engine or AutonomousReasoningEngine()

    try:
        wazuh_event = normalized_to_wazuh_event(event)
        inference_result = inference_engine.infer(wazuh_event)
        reasoning_result = reasoning_engine.reason(
            inference_result, wazuh_event
        )
        if not isinstance(reasoning_result, dict):
            raise TypeError("Reasoning result must be a dictionary")

        alert_id, rule_id = _alert_and_rule(event, wazuh_event)
        action = str(reasoning_result.get("mitigation", "NONE")).strip().upper()

        base = {
            "alert_id": alert_id or None,
            "rule_id": rule_id or None,
            "requested_action": action,
            "execution_policy": POLICY,
            "automatic_execution": AUTOMATIC_EXECUTION,
            "automatic_containment": AUTOMATIC_CONTAINMENT,
            "endpoint_modified": False,
        }

        if action != "PROCESS_TERMINATION":
            return {
                **base,
                "status": "NO_APPROVAL_REQUIRED",
                "reason": "NO_SUPPORTED_TERMINATION_REQUESTED",
            }

        process_name = str(reasoning_result.get("process_name") or "").strip()
        process_id = reasoning_result.get("process_id")
        try:
            pid = int(process_id)
        except (TypeError, ValueError):
            pid = 0

        if not alert_id or not rule_id or not process_name or pid <= 0:
            return {
                **base,
                "status": "BLOCKED",
                "reason": "INCOMPLETE_ALERT_RULE_OR_PROCESS_BINDING",
            }

        bound_result = dict(reasoning_result)
        bound_result.update({
            "mitigation": "PROCESS_TERMINATION",
            "process_name": process_name,
            "process_id": pid,
            "pid": pid,
            "alert_id": alert_id,
            "rule_id": rule_id,
        })

        registration = register_pending_from_reasoning_result(
            bound_result, alert_id, rule_id
        )

        if not registration.get("registered"):
            return {
                **base,
                "status": "BLOCKED",
                "reason": registration.get(
                    "reason", "APPROVAL_REGISTRATION_FAILED"
                ),
            }

        return {
            **base,
            "status": "PENDING_APPROVAL",
            "approval_request_id": registration.get("request_id"),
            "approval_status": registration.get("status", "PENDING"),
        }

    except Exception as exc:
        # Fail closed; do not expose internal exception details to callers.
        return {
            "status": "INTEGRATION_ERROR",
            "reason": type(exc).__name__,
            "execution_policy": POLICY,
            "automatic_execution": False,
            "automatic_containment": False,
            "endpoint_modified": False,
        }


if __name__ == "__main__":
    sample_path = PROJECT / "reports" / "integration" / "approval_integration_sample.json"
    if not sample_path.exists():
        raise SystemExit(
            "Sample not found. Create approval_integration_sample.json "
            "with a representative normalized Wazuh event before running."
        )
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    print(json.dumps(process_for_approval(sample), indent=2, default=str))
