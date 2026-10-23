"""
SmartSOC Wazuh Live Forwarder v1

Purpose
-------
Reads Wazuh alerts from the Ubuntu Wazuh manager's alerts.json stream
and forwards normalized Windows process/activity events into SmartSOC.

Phase 1 behavior:
    - LIVE ingestion
    - LOG_ONLY
    - NO automatic mitigation
    - NO arbitrary command execution

Architecture
------------
Wazuh Manager
    |
    | alerts.json
    v
Forwarder
    |
    v
SmartSOC inference / reasoning pipeline

The forwarder deliberately does NOT execute endpoint actions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_ALERTS_FILE = "/var/ossec/logs/alerts/alerts.json"

TARGET_AGENT_NAME = "SmartSOC-Windows-Victim"
TARGET_AGENT_ID = "001"

# Relevant Sysmon process-create rules currently observed in the lab.
PROCESS_RULE_IDS = {
    "92031",
    "92033",
}

# Phase 1 is deliberately read-only.
EXECUTION_MODE = "LOG_ONLY"

# Maximum number of alert IDs retained in memory.
MAX_SEEN_ALERTS = 10000


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def safe_int(value: Any) -> Optional[int]:
    """Convert a value to a positive integer when possible."""
    try:
        number = int(str(value).strip())
        return number if number > 0 else None
    except (TypeError, ValueError):
        return None


def normalize_path(value: Any) -> str:
    """Normalize Windows path separators for internal processing."""
    if value is None:
        return ""

    text = str(value).strip()

    if not text:
        return ""

    return text.replace("\\\\", "\\")


def get_nested(data: Dict[str, Any], *keys: str) -> Any:
    """Safely retrieve nested dictionary data."""
    current: Any = data

    for key in keys:
        if not isinstance(current, dict):
            return None

        current = current.get(key)

    return current


# ---------------------------------------------------------------------
# Alert filtering
# ---------------------------------------------------------------------

def is_target_agent(alert: Dict[str, Any]) -> bool:
    """
    Return True only for the intended Windows victim.

    We use both agent ID and name when available.
    """

    agent = alert.get("agent")

    if not isinstance(agent, dict):
        return False

    agent_id = str(agent.get("id", "")).strip()
    agent_name = str(agent.get("name", "")).strip()

    if agent_id == TARGET_AGENT_ID:
        return agent_name == TARGET_AGENT_NAME

    return agent_name == TARGET_AGENT_NAME


def is_process_activity(alert: Dict[str, Any]) -> bool:
    """
    Identify the process-create alerts relevant to SmartSOC.

    Current lab rules:
        92031
        92033
    """

    rule = alert.get("rule")

    if not isinstance(rule, dict):
        return False

    rule_id = str(rule.get("id", "")).strip()

    if rule_id in PROCESS_RULE_IDS:
        return True

    # Additional structural protection:
    # accept Sysmon Event ID 1 only if it contains process data.
    win_system = get_nested(
        alert,
        "data",
        "win",
        "system",
    )

    eventdata = get_nested(
        alert,
        "data",
        "win",
        "eventdata",
    )

    if not isinstance(win_system, dict):
        return False

    if not isinstance(eventdata, dict):
        return False

    event_id = str(win_system.get("eventID", "")).strip()

    return event_id == "1" and bool(eventdata.get("processId"))


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def extract_mitre(alert: Dict[str, Any]) -> list[str]:
    """Extract MITRE technique IDs from the Wazuh rule."""

    rule = alert.get("rule")

    if not isinstance(rule, dict):
        return []

    mitre = rule.get("mitre")

    if not isinstance(mitre, dict):
        return []

    ids = mitre.get("id", [])

    if isinstance(ids, str):
        return [ids]

    if isinstance(ids, list):
        return [
            str(item).strip()
            for item in ids
            if str(item).strip()
        ]

    return []


def normalize_alert(alert: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert a raw Wazuh alert into a stable SmartSOC event.

    No execution information is generated here.
    """

    if not is_target_agent(alert):
        return None

    if not is_process_activity(alert):
        return None

    rule = alert.get("rule", {})
    agent = alert.get("agent", {})

    win = get_nested(
        alert,
        "data",
        "win",
    )

    if not isinstance(win, dict):
        return None

    system = win.get("system", {})
    eventdata = win.get("eventdata", {})

    if not isinstance(system, dict):
        return None

    if not isinstance(eventdata, dict):
        return None

    process_id = safe_int(eventdata.get("processId"))

    if process_id is None:
        return None

    process_image = normalize_path(
        eventdata.get("image")
    )

    parent_process_id = safe_int(
        eventdata.get("parentProcessId")
    )

    parent_image = normalize_path(
        eventdata.get("parentImage")
    )

    command_line = str(
        eventdata.get("commandLine", "")
    ).strip()

    parent_command_line = str(
        eventdata.get("parentCommandLine", "")
    ).strip()

    event = {
        "source": "wazuh_live_forwarder_v1",

        "execution_mode": EXECUTION_MODE,

        "wazuh": {
            "alert_id": str(
                alert.get("id", "")
            ),
            "timestamp": str(
                alert.get("timestamp", "")
            ),
            "rule_id": str(
                rule.get("id", "")
            ),
            "rule_level": rule.get("level"),
            "description": str(
                rule.get("description", "")
            ),
            "groups": rule.get("groups", []),
        },

        "agent": {
            "id": str(
                agent.get("id", "")
            ),
            "name": str(
                agent.get("name", "")
            ),
            "ip": str(
                agent.get("ip", "")
            ),
        },

        "mitre": {
            "techniques": extract_mitre(alert),
            "tactics": (
                rule.get("mitre", {}).get("tactic", [])
                if isinstance(rule.get("mitre"), dict)
                else []
            ),
        },

        "windows": {
            "computer": str(
                system.get("computer", "")
            ),
            "event_id": str(
                system.get("eventID", "")
            ),
            "channel": str(
                system.get("channel", "")
            ),
            "process_id": process_id,
            "process_name": process_image,
            "command_line": command_line,
            "parent_process_id": parent_process_id,
            "parent_process_name": parent_image,
            "parent_command_line": parent_command_line,
            "user": str(
                eventdata.get("user", "")
            ),
            "integrity_level": str(
                eventdata.get("integrityLevel", "")
            ),
            "hashes": str(
                eventdata.get("hashes", "")
            ),
        },
    }

    return event


# ---------------------------------------------------------------------
# SmartSOC handoff
# ---------------------------------------------------------------------

def handoff_to_smartsoc(event: Dict[str, Any]) -> None:
    """
    Phase 1 handoff.

    We intentionally stop at the normalized SmartSOC boundary.
    The event is printed as structured JSON.

    Phase 2 will connect this boundary to the existing inference /
    reasoning pipeline.
    """

    print()
    print("=" * 78)
    print("SMARTSOC LIVE WAZUH EVENT")
    print("=" * 78)

    print(
        json.dumps(
            event,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("-" * 78)
    print("PHASE 1 MODE")
    print("-" * 78)
    print("Execution mode : LOG_ONLY")
    print("Mitigation     : DISABLED")
    print("Endpoint action: NONE")
    print("-" * 78)
    print()


# ---------------------------------------------------------------------
# Alert stream
# ---------------------------------------------------------------------

def follow_file(path: Path) -> Iterable[str]:
    """
    Follow a growing JSON-lines file.

    Existing historical records are skipped by default.
    Only records appended after startup are processed.
    """

    print(f"[+] Opening Wazuh alert stream: {path}")

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        # Start at EOF so historical alerts are not replayed.
        handle.seek(0, os.SEEK_END)

        print("[+] Starting from current end-of-file.")
        print("[+] Waiting for NEW Wazuh alerts...")
        print()

        while True:
            line = handle.readline()

            if not line:
                time.sleep(0.25)
                continue

            yield line.rstrip("\r\n")


def parse_json_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single Wazuh JSON alert."""

    line = line.strip()

    if not line:
        return None

    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as exc:
        print(
            f"[!] Invalid JSON line skipped: {exc}",
            file=sys.stderr,
        )
        return None

    if not isinstance(parsed, dict):
        return None

    return parsed


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------

def run_forwarder(alerts_file: str) -> None:

    path = Path(alerts_file)

    if not path.exists():
        raise FileNotFoundError(
            f"Wazuh alerts file does not exist: {path}"
        )

    seen_alert_ids: Set[str] = set()

    print("=" * 78)
    print("SmartSOC Wazuh Live Forwarder v1")
    print("=" * 78)
    print(f"Alerts file   : {path}")
    print(f"Target agent  : {TARGET_AGENT_NAME} ({TARGET_AGENT_ID})")
    print(f"Process rules : {sorted(PROCESS_RULE_IDS)}")
    print(f"Execution     : {EXECUTION_MODE}")
    print("=" * 78)
    print()

    for line in follow_file(path):

        alert = parse_json_line(line)

        if alert is None:
            continue

        if not is_target_agent(alert):
            continue

        alert_id = str(
            alert.get("id", "")
        ).strip()

        if alert_id and alert_id in seen_alert_ids:
            continue

        event = normalize_alert(alert)

        if event is None:
            continue

        if alert_id:
            seen_alert_ids.add(alert_id)

            if len(seen_alert_ids) > MAX_SEEN_ALERTS:
                # Simple bounded-memory behavior.
                # Remove an arbitrary old ID; ordering is not required
                # for correctness because Wazuh alert IDs are unique.
                seen_alert_ids.pop()

        handoff_to_smartsoc(event)


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "SmartSOC Phase 1 live Wazuh alerts.json forwarder"
        )
    )

    parser.add_argument(
        "--alerts-file",
        default=DEFAULT_ALERTS_FILE,
        help=(
            "Path to Wazuh alerts.json. "
            "Use the Ubuntu manager path when running on Ubuntu."
        ),
    )

    args = parser.parse_args()

    try:
        run_forwarder(args.alerts_file)
        return 0

    except KeyboardInterrupt:
        print()
        print("[+] Forwarder stopped by user.")
        return 0

    except Exception as exc:
        print(
            f"[!] Forwarder error: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())