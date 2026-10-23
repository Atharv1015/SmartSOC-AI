"""Bridge an exact approved request to the guarded SmartSOC v2 dry-run runner."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = str(Path(__file__).resolve().parents[1])
INTEGRATION_V2_DIR = str(Path(__file__).resolve().parent)

for directory in (SRC_DIR, INTEGRATION_V2_DIR):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from integration import local_approval_console_v1 as approvals
from integration_v2 import authorized_dry_run_v2 as runner


def _blocked(reason, status="AUTHORIZATION_BLOCKED"):
    return {
        "success": False,
        "verified": False,
        "status": status,
        "reason": reason,
        "authorization_consumed": False,
        "victim_request_attempted": False,
        "victim_client_invoked": False,
        "endpoint_modified": False,
    }


def execute_approved_request(request_id):
    """Preflight the selected request, then consume that exact approval once."""
    request_id = str(request_id or "").strip()
    if not request_id:
        return _blocked("INVALID_REQUEST_ID")

    record = approvals.get_approved_request(request_id)
    if not record.get("ok"):
        return _blocked(record.get("reason", "APPROVAL_UNAVAILABLE"))

    # Build the execution input from the database record, not caller-supplied
    # action, process, PID, alert ID, or rule ID.
    bound = {
        "alert_id": record["alert_id"],
        "rule_id": record["rule_id"],
        "mitigation": record["action"],
        "process_name": record["process_name"],
        "process_id": record["pid"],
    }

    try:
        result = runner.execute_authorized_dry_run(
            bound,
            approval_request_id=request_id,
        )
    except Exception:
        # Fail closed; do not expose exception details in the bridge response.
        return _blocked("ORCHESTRATOR_ERROR", "BRIDGE_BLOCKED")

    if not isinstance(result, dict):
        return _blocked("INVALID_ORCHESTRATOR_RESULT", "BRIDGE_BLOCKED")

    result = dict(result)
    result["selected_approval_request_id"] = request_id
    result["endpoint_modified"] = False
    return result
