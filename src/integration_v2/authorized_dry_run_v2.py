"""Human-authorized SmartSOC v2 dry-run orchestration. No live execution."""

import os
import sys
from pathlib import Path

SRC_DIR = str(Path(__file__).resolve().parents[1])
INTEGRATION_V2_DIR = str(Path(__file__).resolve().parent)

for directory in (SRC_DIR, INTEGRATION_V2_DIR):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from integration import local_approval_console_v1 as approvals
from autonomous_v2 import victim_mitigation_adapter_v2 as adapter
import execution_config_v2 as policy


def _blocked(status, reason):
    return {
        "success": False,
        "verified": False,
        "status": status,
        "reason": reason,
        "victim_request_attempted": False,
        "victim_client_invoked": False,
        "endpoint_modified": False,
    }


def execute_authorized_dry_run(reasoning_result, alert_id=None, rule_id=None, approval_request_id=None):
    """Consume matching human approval, then invoke the v2 dry-run adapter."""

    if not isinstance(reasoning_result, dict):
        return _blocked("REQUEST_BLOCKED", "Reasoning result must be a dictionary.")

    # DRY_RUN uses its own policy checks, not the legacy live-execution validator.
    # The live validator intentionally requires DRY_RUN=False.
    if not policy.DRY_RUN or policy.UNRESTRICTED_AUTONOMOUS_ACTIONS:
        return _blocked("POLICY_BLOCKED", "Dry-run mode is disabled or unrestricted actions are configured.")

    action = str(reasoning_result.get("mitigation", "")).strip().upper()
    if action != "PROCESS_TERMINATION" or not policy.is_action_allowed(action):
        return _blocked("REQUEST_BLOCKED", "Action is not allowlisted.")

    raw_pid = reasoning_result.get("process_id")
    if isinstance(raw_pid, bool):
        return _blocked("REQUEST_BLOCKED", "Invalid PID.")

    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        return _blocked("REQUEST_BLOCKED", "Invalid PID.")

    if pid <= 0:
        return _blocked("REQUEST_BLOCKED", "PID must be positive.")

    process_name = reasoning_result.get("process_name")
    if not isinstance(process_name, str):
        return _blocked("REQUEST_BLOCKED", "Process name is required.")

    process_name = process_name.strip().lower()
    if not policy.is_process_allowed(process_name):
        return _blocked("REQUEST_BLOCKED", "Process is not allowlisted.")

    if len(os.environ.get("SMARTSOC_AGENT_SECRET", "")) < 32:
        return _blocked("CONFIGURATION_BLOCKED", "Host secret is missing or invalid.")

    bound = dict(reasoning_result)
    bound["mitigation"] = action
    bound["process_id"] = pid
    bound["process_name"] = process_name

    bound_alert_id = alert_id or bound.get("alert_id") or bound.get("wazuh_alert_id")
    bound_rule_id = rule_id or bound.get("rule_id") or bound.get("wazuh_rule_id")

    if not bound_alert_id or not bound_rule_id:
        return _blocked("REQUEST_BLOCKED", "Alert ID and rule ID are required.")

    bound["alert_id"] = str(bound_alert_id)
    bound["rule_id"] = str(bound_rule_id)

    # Atomically consume the exact matching approval before adapter invocation.
    authorization = approvals.consume_approval(bound, request_id=approval_request_id)
    if not authorization.get("authorized"):
        result = _blocked(
            "AUTHORIZATION_BLOCKED",
            authorization.get("reason", "Approval required."),
        )
        result["authorization_request_id"] = None
        return result

    # The adapter's HTTP connection must be mocked in tests.
    try:
        result = adapter.execute_dry_run(bound)
    except Exception as exc:
        result = _blocked("ADAPTER_ERROR", str(exc))

    if not isinstance(result, dict):
        result = _blocked("INVALID_ADAPTER_RESULT", "Adapter returned an invalid result.")

    result = dict(result)
    result["authorization_request_id"] = authorization.get("request_id")
    result["authorization_actor"] = authorization.get("actor")
    result["authorization_consumed"] = True
    result["endpoint_modified"] = False
    return result
