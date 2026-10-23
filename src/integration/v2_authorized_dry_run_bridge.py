"""Receiver bridge to the canonical, human-authorized V2 runner."""

from integration_v2.authorized_dry_run_v2 import execute_authorized_dry_run


def execute_authorized_v2_dry_run(reasoning_result):
    """Delegate validation, approval, and dry-run execution to V2."""
    if not isinstance(reasoning_result, dict):
        return {
            "success": False,
            "verified": False,
            "status": "REQUEST_BLOCKED",
            "victim_request_attempted": False,
            "victim_client_invoked": False,
            "endpoint_modified": False,
        }

    return execute_authorized_dry_run(
        reasoning_result,
        alert_id=reasoning_result.get("alert_id"),
        rule_id=reasoning_result.get("rule_id"),
    )
