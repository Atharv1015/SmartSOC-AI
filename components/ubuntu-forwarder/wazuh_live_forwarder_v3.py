#!/usr/bin/env python3
"""SmartSOC Wazuh forwarder v3: raw alert forwarding only."""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

VERSION = "smartsoc-live-wazuh-forwarder-v3"
ALERTS_FILE = "/var/ossec/logs/alerts/alerts.json"
RECEIVER_URL = "http://192.168.56.1:8787/wazuh-event"

TARGET_AGENT_ID = "001"
TARGET_AGENT_NAME = "SmartSOC-Windows-Victim"

RELEVANT_RULE_IDS = {
    "92200",  # Scripting file created under Windows Temp or User folder
    "92027", "92031", "92032", "92033", "92057",
    "100034", "92034", "92035", "92039","100104",
    "91829", "91830", "91832", "91833", "91839", "91841",
    "60228", "60105", "60115", "60203", "60204", "100101",
}

HTTP_TIMEOUT_SECONDS = 30
POLL_INTERVAL_SECONDS = 0.25
MAX_RECENT_ALERT_IDS = 10000
LOG_FILE = Path.home() / "smartsoc-forwarder-v3" / "logs" / "forwarder_v3.log"

recent_ids = set()
recent_order = deque()


def log(message):
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"{timestamp} {message}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def remember_alert(alert_id):
    if not alert_id:
        return True
    if alert_id in recent_ids:
        return False
    recent_ids.add(alert_id)
    recent_order.append(alert_id)
    while len(recent_order) > MAX_RECENT_ALERT_IDS:
        expired = recent_order.popleft()
        recent_ids.discard(expired)
    return True


def target_agent(alert):
    agent = alert.get("agent", {})
    if not isinstance(agent, dict):
        return False
    return (
        str(agent.get("id", "")) == TARGET_AGENT_ID
        or str(agent.get("name", "")) == TARGET_AGENT_NAME
    )


def post_raw_alert(alert):
    """POST a raw Wazuh event with bounded retries."""
    body = json.dumps(
        {"event": alert},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(
            RECEIVER_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "SmartSOC-Wazuh-Forwarder-v3",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=HTTP_TIMEOUT_SECONDS
            ) as response:
                response_body = response.read().decode(
                    "utf-8", errors="replace"
                )
                ok = 200 <= response.status < 300
                log(
                    f"DELIVERY {'OK' if ok else 'FAILED'} "
                    f"http={response.status} attempt={attempt}/{max_attempts} "
                    f"alert_id={alert.get('id', '')} "
                    f"response={response_body[:1000]}"
                )
                if ok:
                    return True
                if response.status < 500:
                    return False

        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            log(
                f"DELIVERY FAILED http={exc.code} "
                f"attempt={attempt}/{max_attempts} "
                f"alert_id={alert.get('id', '')} "
                f"response={response_body[:1000]}"
            )
            if 400 <= exc.code < 500 and exc.code != 429:
                return False

        except Exception as exc:
            log(
                f"DELIVERY FAILED attempt={attempt}/{max_attempts} "
                f"alert_id={alert.get('id', '')} "
                f"error={type(exc).__name__}: {exc}"
            )

        if attempt < max_attempts:
            time.sleep(attempt * 2)

    log(f"DELIVERY ABANDONED after {max_attempts} attempts alert_id={alert.get('id', '')}")
    return False


def process_alert(alert):
    if not isinstance(alert, dict) or not target_agent(alert):
        return

    rule = alert.get("rule", {})
    if not isinstance(rule, dict):
        return

    rule_id = str(rule.get("id", ""))
    if rule_id not in RELEVANT_RULE_IDS:
        return

    alert_id = str(alert.get("id", ""))
    if not remember_alert(alert_id):
        log(f"DUPLICATE_SKIPPED alert_id={alert_id}")
        return

    log(
        f"RELEVANT_ALERT rule_id={rule_id} alert_id={alert_id} "
        f"agent={TARGET_AGENT_NAME}"
    )
    post_raw_alert(alert)


def follow_alerts():
    if not os.path.isfile(ALERTS_FILE):
        raise FileNotFoundError(f"Wazuh alert log not found: {ALERTS_FILE}")

    with open(ALERTS_FILE, "r", encoding="utf-8", errors="replace") as handle:
        handle.seek(0, os.SEEK_END)
        log(f"START version={VERSION} mode=FORWARD_ONLY")
        log(f"Source={ALERTS_FILE}")
        log(f"Receiver={RECEIVER_URL}")
        log("Starting at current EOF; historical alerts will not be replayed.")
        log("Safety: forwarding only; no commands, mitigation, or containment.")

        partial = ""
        while True:
            line = handle.readline()
            if not line:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            line = partial + line
            partial = ""
            line = line.strip()
            if not line:
                continue

            try:
                alert = json.loads(line)
            except json.JSONDecodeError:
                partial = line
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            try:
                process_alert(alert)
            except Exception as exc:
                log(f"PROCESS_ERROR {type(exc).__name__}: {exc}")


def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log(f"Forwarder version: {VERSION}")
    log(f"Python: {sys.version.split()[0]}")
    log(f"Rules allowlisted: {len(RELEVANT_RULE_IDS)}")
    follow_alerts()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("STOPPED_BY_USER")
        raise SystemExit(0)
    except Exception as exc:
        log(f"FATAL {type(exc).__name__}: {exc}")
        raise
