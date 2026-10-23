"""SmartSOC v2 authenticated live adapter for the controlled lab."""

import http.client
import json
import os
import secrets
import sys
import time
from pathlib import Path

SRC_DIR = str(Path(__file__).resolve().parents[1])
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from autonomous_v2.victim_mitigation_client_v2 import (
    build_signature,
    VICTIM_HOST,
    VICTIM_PORT,
    TIMEOUT_SECONDS,
)
import execution_config_v2 as policy


def execute_live(reasoning_result):
    """Send an authenticated LIVE request to the victim agent.

    The victim must independently implement and verify LIVE execution.
    """
    if not isinstance(reasoning_result, dict):
        return {
            "success": False, "verified": False,
            "status": "REQUEST_BLOCKED",
            "error": "Reasoning result must be a dictionary.",
        }

    try:
        invariants_ok = policy.validate_safety_invariants()
    except Exception as exc:
        return {
            "success": False, "verified": False,
            "status": "POLICY_BLOCKED", "error": str(exc),
            "victim_request_attempted": False,
            "victim_client_invoked": False,
            "endpoint_modified": False,
        }

    if (
        policy.DRY_RUN
        or not policy.LIVE_EXECUTION_ENABLED
        or not invariants_ok
        or not policy.CONTROLLED_LAB_MODE
        or not policy.EXECUTION_LAYER_ENABLED
        or not policy.AUTOMATIC_EXECUTION
        or not policy.AUTOMATIC_CONTAINMENT
        or policy.HUMAN_AUTHORIZATION_REQUIRED
        or policy.UNRESTRICTED_AUTONOMOUS_ACTIONS
    ):
        return {
            "success": False, "verified": False,
            "status": "POLICY_BLOCKED",
            "error": "Live execution requires the explicit live gate and valid controlled-lab policy.",
            "victim_request_attempted": False,
            "victim_client_invoked": False,
            "endpoint_modified": False,
        }

    action = str(reasoning_result.get("mitigation", "")).strip().upper()
    if action != "PROCESS_TERMINATION" or not policy.is_action_allowed(action):
        return {
            "success": False, "verified": False,
            "status": "REQUEST_BLOCKED",
            "error": "Action is not approved by the allowlist.",
        }

    raw_pid = reasoning_result.get("process_id")
    if isinstance(raw_pid, bool):
        return {
            "success": False, "verified": False,
            "status": "REQUEST_BLOCKED", "error": "Invalid PID.",
        }

    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        return {
            "success": False, "verified": False,
            "status": "REQUEST_BLOCKED", "error": "Invalid PID.",
        }

    if pid <= 0:
        return {
            "success": False, "verified": False,
            "status": "REQUEST_BLOCKED", "error": "PID must be positive.",
        }

    process_name = reasoning_result.get("process_name")
    if not isinstance(process_name, str):
        return {
            "success": False, "verified": False,
            "status": "REQUEST_BLOCKED",
            "error": "Process name is required.",
        }

   
    process_name = os.path.basename(
        process_name.strip().replace("\\", "/")
    ).lower()

    # Rule 100130 exception is scoped to the explicitly marked lab test.
    # The victim agent independently validates executable and command line.
    _rule_id = str(reasoning_result.get("wazuh_rule_id", ""))
    _lab_override = reasoning_result.get("lab_test_override") is True
    _scoped_defender_lab = (
        _rule_id == "100130"
        and _lab_override
        and process_name == "powershell.exe"
    )
    if not policy.is_process_allowed(process_name) and not _scoped_defender_lab:
        return {
            "success": False, "verified": False,
            "status": "REQUEST_BLOCKED",
            "error": "Process name is not allowlisted for this rule.",
        }


    secret = os.environ.get("SMARTSOC_AGENT_SECRET", "")
    if len(secret) < 32:
        return {
            "success": False, "verified": False,
            "status": "CONFIGURATION_BLOCKED",
            "error": "Host secret is missing or invalid.",
        }

    request_body = {
        "action": action,
        "target": {"pid": pid, "process_name": process_name},
        "context": {
            "client_version": "victim-mitigation-adapter-v2",
            "requested_mode": "LIVE",
            "technique": reasoning_result.get("technique"),
            "wazuh_rule_id": reasoning_result.get("wazuh_rule_id"),
            "lab_test_override": reasoning_result.get("lab_test_override") is True,
            "reason": reasoning_result.get("reason"),
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
    attempted = False

    try:
        attempted = True
        connection.request("POST", "/mitigate", body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read().decode("utf-8", errors="replace")

        try:
            victim_response = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "success": False, "verified": False,
                "status": "INVALID_VICTIM_RESPONSE",
                "http_status": response.status,
                "victim_request_attempted": attempted,
            }

        # Do not count an HTTP 200 or an unverified response as success.
        verified_success = (
            response.status == 200
            and victim_response.get("success") is True
            and victim_response.get("verified") is True
            and victim_response.get("status") == "TERMINATED_VERIFIED"
        )
        return {
            "success": verified_success,
            "verified": verified_success,
            "status": (
                "TERMINATED_VERIFIED"
                if verified_success
                else victim_response.get("status", "MITIGATION_UNVERIFIED")
            ),
            "http_status": response.status,
            "victim_request_attempted": attempted,
            "victim_http_response_received": True,
            "client": "victim-mitigation-adapter-v2",
            "victim_response": victim_response,
        }

    except OSError as exc:
        return {
            "success": False, "verified": False,
            "status": "VICTIM_UNREACHABLE",
            "victim_request_attempted": attempted,
            "victim_http_response_received": False,
            "error": str(exc),
        }
    finally:
        connection.close()


def execute_dry_run(reasoning_result):
    # Authenticated DRY_RUN request only; this function never requests LIVE mode.
    def blocked(status, error):
        return {
            "success": False,
            "verified": False,
            "status": status,
            "error": error,
            "victim_request_attempted": False,
            "victim_http_response_received": False,
            "endpoint_modified": False,
        }

    if not isinstance(reasoning_result, dict):
        return blocked("REQUEST_BLOCKED", "Reasoning result must be a dictionary.")

    if not policy.DRY_RUN or policy.UNRESTRICTED_AUTONOMOUS_ACTIONS:
        return blocked("POLICY_BLOCKED", "Dry-run policy is disabled or unsafe.")

    action = str(reasoning_result.get("mitigation", "")).strip().upper()
    if action != "PROCESS_TERMINATION" or not policy.is_action_allowed(action):
        return blocked("REQUEST_BLOCKED", "Action is not approved by the allowlist.")

    raw_pid = reasoning_result.get("process_id")
    if isinstance(raw_pid, bool):
        return blocked("REQUEST_BLOCKED", "Invalid PID.")
    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        return blocked("REQUEST_BLOCKED", "Invalid PID.")
    if pid <= 0:
        return blocked("REQUEST_BLOCKED", "PID must be positive.")

    process_name = reasoning_result.get("process_name")
    if not isinstance(process_name, str):
        return blocked("REQUEST_BLOCKED", "Process name is required.")
    process_name = os.path.basename(
        process_name.strip().replace("\\", "/")
    ).lower()
    if not policy.is_process_allowed(process_name):
        return blocked("REQUEST_BLOCKED", "Process name is not allowlisted.")

    secret = os.environ.get("SMARTSOC_AGENT_SECRET", "")
    if len(secret) < 32:
        return blocked("CONFIGURATION_BLOCKED", "Host secret is missing or invalid.")

    request_body = {
        "action": action,
        "target": {"pid": pid, "process_name": process_name},
        "context": {
            "client_version": "victim-mitigation-adapter-v2",
            "requested_mode": "DRY_RUN",
            "technique": reasoning_result.get("technique"),
            "wazuh_rule_id": reasoning_result.get("wazuh_rule_id"),
            "reason": reasoning_result.get("reason"),
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

    connection = None
    attempted = False
    try:
        connection = http.client.HTTPConnection(
            VICTIM_HOST, VICTIM_PORT, timeout=TIMEOUT_SECONDS
        )
        attempted = True
        connection.request("POST", "/mitigate", body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read().decode("utf-8", errors="replace")
        try:
            victim_response = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "success": False,
                "verified": False,
                "status": "INVALID_VICTIM_RESPONSE",
                "http_status": response.status,
                "victim_request_attempted": attempted,
                "victim_http_response_received": True,
                "endpoint_modified": False,
            }

        verified = (
            response.status == 200
            and victim_response.get("success") is True
            and victim_response.get("verified") is True
            and victim_response.get("status") == "DRY_RUN_VALIDATED"
        )
        return {
            "success": verified,
            "verified": verified,
            "status": (
                "DRY_RUN_VALIDATED"
                if verified
                else victim_response.get("status", "DRY_RUN_REJECTED")
            ),
            "http_status": response.status,
            "victim_request_attempted": attempted,
            "victim_http_response_received": True,
            "endpoint_modified": False,
            "client": "victim-mitigation-adapter-v2",
            "victim_response": victim_response,
        }
    except OSError as exc:
        return {
            "success": False,
            "verified": False,
            "status": "VICTIM_UNREACHABLE",
            "victim_request_attempted": attempted,
            "victim_http_response_received": False,
            "endpoint_modified": False,
            "error": str(exc),
        }
    finally:
        if connection is not None:
            connection.close()
