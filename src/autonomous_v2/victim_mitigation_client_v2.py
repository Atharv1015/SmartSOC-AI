"""SmartSOC v2 HMAC client. Initial mitigation requests are dry-run only."""

import hashlib
import hmac
import http.client
import json
import os
import secrets
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration_v2"))

import execution_config_v2 as policy

VICTIM_HOST = "192.168.56.102"
VICTIM_PORT = 8765
TIMEOUT_SECONDS = 5


def build_signature(secret, timestamp, nonce, body):
    message = (
        timestamp.encode("ascii")
        + b"\n"
        + nonce.encode("ascii")
        + b"\n"
        + body
    )
    return hmac.new(
        secret.encode("utf-8"), message, hashlib.sha256
    ).hexdigest()


def request_health():
    connection = http.client.HTTPConnection(
        VICTIM_HOST, VICTIM_PORT, timeout=TIMEOUT_SECONDS
    )
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        raw = response.read().decode("utf-8", errors="replace")
        print("HTTP_STATUS:", response.status)
        try:
            print(json.dumps(json.loads(raw), indent=2))
        except json.JSONDecodeError:
            print(raw)
        return 0 if response.status == 200 else 1
    except OSError as exc:
        print("HEALTH_REQUEST_FAILED:", str(exc))
        return 1
    finally:
        connection.close()


def request_dry_run(pid, process_name):
    if not policy.DRY_RUN:
        print("BLOCKED: v2 policy must remain in dry-run mode.")
        return 2

    if not policy.is_action_allowed("PROCESS_TERMINATION"):
        print("BLOCKED: action is not allowlisted.")
        return 2

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        print("BLOCKED: PID must be a positive integer.")
        return 2

    if not policy.is_process_allowed(process_name):
        print("BLOCKED: process name is not allowlisted.")
        return 2

    secret = os.environ.get("SMARTSOC_AGENT_SECRET", "")
    if len(secret) < 32:
        print("BLOCKED: host secret is missing or invalid.")
        return 2

    request_body = {
        "action": "PROCESS_TERMINATION",
        "target": {
            "pid": pid,
            "process_name": process_name.strip().lower(),
        },
        "context": {
            "client_version": "victim-mitigation-client-v2",
            "requested_mode": "DRY_RUN",
        },
    }
    body = json.dumps(
        request_body, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")

    timestamp = str(int(time.time()))
    nonce = secrets.token_hex(16)
    signature = build_signature(secret, timestamp, nonce, body)

    headers = {
        "Content-Type": "application/json",
        "X-SmartSOC-Timestamp": timestamp,
        "X-SmartSOC-Nonce": nonce,
        "X-SmartSOC-Signature": signature,
    }

    connection = http.client.HTTPConnection(
        VICTIM_HOST, VICTIM_PORT, timeout=TIMEOUT_SECONDS
    )
    try:
        connection.request("POST", "/mitigate", body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read().decode("utf-8", errors="replace")
        print("HTTP_STATUS:", response.status)

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            print("INVALID_JSON_RESPONSE")
            print(raw)
            return 1

        print(json.dumps(result, indent=2))

        if response.status == 200 and result.get("status") == "DRY_RUN_VALIDATED":
            print("DRY_RUN_CONFIRMED: target validated; no termination reported.")
            return 0

        print("DRY_RUN_NOT_CONFIRMED: inspect response and victim audit log.")
        return 1

    except OSError as exc:
        print("MITIGATION_REQUEST_FAILED:", str(exc))
        return 1
    finally:
        connection.close()


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "health":
        return request_health()

    if len(sys.argv) == 4 and sys.argv[1] == "dry-run":
        try:
            pid = int(sys.argv[2])
        except ValueError:
            print("PID must be an integer.")
            return 2
        return request_dry_run(pid, sys.argv[3])

    print("Usage:")
    print("  py -3 src\\autonomous_v2\\victim_mitigation_client_v2.py health")
    print("  py -3 src\\autonomous_v2\\victim_mitigation_client_v2.py dry-run PID notepad.exe")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())