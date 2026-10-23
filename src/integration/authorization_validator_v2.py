"""Fail-closed, event-bound SmartSOC authorization validator.

This module validates an approval record supplied by a trusted caller.
It does not load stale artifacts, execute mitigation, or contact endpoints.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict


MAX_APPROVAL_AGE_SECONDS = 300


def validate_approval(
    approval: Any,
    alert_id: str,
    rule_id: str,
    now: datetime | None = None,
) -> Dict[str, Any]:
    """Validate an event-bound approval record; deny by default."""

    def deny(reason: str) -> Dict[str, Any]:
        return {
            "authorized": False,
            "status": "DENIED",
            "reason": reason,
        }

    if not isinstance(approval, dict):
        return deny("Approval record is missing or malformed.")

    if not isinstance(alert_id, str) or not alert_id.strip():
        return deny("A valid alert ID is required.")

    if not isinstance(rule_id, (str, int)) or not str(rule_id).strip():
        return deny("A valid rule ID is required.")

    if approval.get("decision") != "APPROVED":
        return deny("Explicit APPROVED decision is required.")

    if type(approval.get("analyst_authorized")) is not bool:
        return deny("analyst_authorized must be a Boolean.")

    if approval["analyst_authorized"] is not True:
        return deny("Analyst authorization is not confirmed.")

    if approval.get("alert_id") != alert_id:
        return deny("Approval alert ID does not match the event.")

    if str(approval.get("rule_id", "")) != str(rule_id):
        return deny("Approval rule ID does not match the event.")

    actor = approval.get("authorization_actor")
    if not isinstance(actor, str) or not actor.strip():
        return deny("A recorded authorization actor is required.")

    timestamp = approval.get("authorization_timestamp")
    if not isinstance(timestamp, str) or not timestamp.strip():
        return deny("An authorization timestamp is required.")

    try:
        approved_at = datetime.fromisoformat(
            timestamp.replace("Z", "+00:00")
        )
    except (ValueError, OverflowError):
        return deny("Authorization timestamp is invalid.")

    if approved_at.tzinfo is None or approved_at.utcoffset() is None:
        return deny("Authorization timestamp must include a timezone.")

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        return deny("Validation time must include a timezone.")

    age = (current_time - approved_at).total_seconds()

    if age < 0:
        return deny("Approval timestamp is in the future.")

    if age > MAX_APPROVAL_AGE_SECONDS:
        return deny("Approval has expired.")

    return {
        "authorized": True,
        "status": "APPROVED",
        "reason": "Event-bound approval passed validation.",
        "authorization_actor": actor,
        "alert_id": alert_id,
        "rule_id": str(rule_id),
        "authorization_timestamp": approved_at.isoformat(),
    }
