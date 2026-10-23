"""Offline brute-force detection and Wazuh event normalization."""

from collections import defaultdict
from datetime import datetime, timedelta, timezone


WINDOW = timedelta(minutes=5)
THRESHOLD = 5


def _parse_timestamp(value):
    if not isinstance(value, str) or not value.strip():
        return None

    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def normalize_wazuh_auth_event(alert):
    """Normalize a Wazuh Windows authentication alert.

    Returns None when required authentication fields are unavailable.
    This function does not execute mitigation.
    """
    if not isinstance(alert, dict):
        return None

    data = alert.get("data")
    if not isinstance(data, dict):
        return None

    win = data.get("win")
    if not isinstance(win, dict):
        return None

    system = win.get("system")
    eventdata = win.get("eventdata")
    if not isinstance(system, dict) or not isinstance(eventdata, dict):
        return None

    event_id = str(system.get("eventID", "")).strip()
    if event_id != "4625":
        return None

    timestamp = _parse_timestamp(
        system.get("systemTime") or alert.get("timestamp")
    )
    account = eventdata.get("targetUserName")
    source_ip = eventdata.get("ipAddress")

    if timestamp is None:
        return None
    if not isinstance(account, str) or not account.strip():
        return None
    if not isinstance(source_ip, str) or not source_ip.strip():
        return None

    return {
        "event_id": event_id,
        "timestamp": timestamp,
        "source_ip": source_ip.strip(),
        "target_account": account.strip(),
        "wazuh_alert_id": str(alert.get("id", "")),
        "wazuh_rule_id": str(
            (alert.get("rule") or {}).get("id", "")
        ),
    }


def detect_brute_force(events):
    """Detect repeated failed logons by source IP and target account."""
    buckets = defaultdict(list)

    for event in events:
        if not isinstance(event, dict):
            continue
        if str(event.get("event_id")) != "4625":
            continue

        source_ip = event.get("source_ip")
        account = event.get("target_account")
        timestamp = event.get("timestamp")

        if not source_ip or not account:
            continue
        if not isinstance(timestamp, datetime):
            continue

        key = (str(source_ip), str(account).casefold())
        buckets[key].append(timestamp)

    findings = []

    for (source_ip, account), timestamps in buckets.items():
        timestamps.sort()
        left = 0

        for right, current in enumerate(timestamps):
            while current - timestamps[left] > WINDOW:
                left += 1

            count = right - left + 1

            if count >= THRESHOLD:
                findings.append({
                    "classification": "BRUTE_FORCE_SUSPECTED",
                    "source_ip": source_ip,
                    "target_account": account,
                    "failed_logons": count,
                    "window_minutes": WINDOW.total_seconds() / 60,
                    "mitigation": "NONE",
                })
                break

    return findings
