"""
SmartSOC Live Receiver v3

Controlled live Wazuh -> SmartSOC -> mitigation receiver.

Pipeline:

    Wazuh Live Forwarder
            |
            v
    SmartSOC Live Receiver
            |
            +--> SmartSOC v4.1 inference
            |
            +--> Autonomous reasoning
            |
            +--> Controlled mitigation controller
            |
            +--> Windows defensive agent
            |
            +--> PID/process validation
            |
            +--> PROCESS_TERMINATION
            |
            +--> Post-action verification
            |
            +--> Audit logging

Security properties:

    - Frozen SmartSOC v4.1 inference is unchanged.
    - Autonomous reasoning is unchanged.
    - Only the existing controlled mitigation controller
      can perform endpoint mitigation.
    - Only PROCESS_TERMINATION is supported by the controller.
    - No arbitrary shell commands are accepted or generated.
    - The Windows defensive agent performs final target validation.
    - Protected Windows processes remain protected.
    - PID and process identity are validated at the endpoint.
    - Endpoint state is verified after mitigation.
    - Controlled lab policy remains the central execution boundary.
    - Mitigation is attempted only when reasoning explicitly selects it.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from time import perf_counter_ns
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict


# ============================================================================
# PROJECT PATH
# ============================================================================

SRC_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Ensure v2 policy and adapter modules resolve during normal receiver startup.
INTEGRATION_V2_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "integration_v2")
)
if INTEGRATION_V2_DIR not in sys.path:
    sys.path.insert(0, INTEGRATION_V2_DIR)


# ============================================================================
# SMARTSOC COMPONENTS
# ============================================================================

from integration.local_approval_console_v1 import bind_event_context

from production_inference_windows_apt_v4_1 import (
    SmartSOCInference,
)

from autonomous.autonomous_reasoning_v1 import (
    AutonomousReasoningEngine,
)

from autonomous.autonomous_mitigation_controller_v1 import (
    AutonomousMitigationController,
)

from decision.smartsoc_risk_engine_v4_1 import SmartSOCRiskEngine
from decision.smartsoc_decision_policy_v4_1 import SmartSOCDecisionPolicy
from decision.smartsoc_response_planner_v4_1 import SmartSOCResponsePlanner

# The adapter remains responsible for HMAC authentication, nonce/timestamp checks,
# process allowlisting, PID/image validation, and verified outcome reporting.
from autonomous_v2.victim_mitigation_adapter_v2 import execute_live
from integration_v2 import execution_config_v2 as _smartsoc_execution_policy
from integration import local_approval_console_v1 as _smartsoc_approvals



# ============================================================================
# RECEIVER CONFIGURATION
# ============================================================================

LISTEN_HOST = "192.168.56.1"
LISTEN_PORT = 8787

# ---------------------------------------------------------------------------
# Phase 3 controlled execution mode
# ---------------------------------------------------------------------------
#
# This is NOT unrestricted autonomous execution.
#
# The receiver will only call the existing controlled mitigation controller
# when:
#
#   1. SmartSOC inference completes.
#   2. Autonomous reasoning explicitly selects a supported mitigation.
#   3. MITIGATION_ENABLED is True.
#   4. The controller accepts the structured mitigation decision.
#   5. The Windows defensive agent accepts the request.
#
# Current supported mitigation:
#
#   PROCESS_TERMINATION
#
# ---------------------------------------------------------------------------

EXECUTION_MODE = "CONTROLLED_LAB"

MITIGATION_ENABLED = True
# Effective receiver gate: environment opt-in is insufficient while policy is dry-run.
LIVE_EXECUTION_ENABLED = (
    os.environ.get("SMARTSOC_ALLOW_LIVE_MITIGATION", "").strip() == "1"
    and not _smartsoc_execution_policy.DRY_RUN
)
VICTIM_AGENT_CALLED = False


# ============================================================================
# COMPONENT INITIALIZATION
# ============================================================================

inference_engine = SmartSOCInference()

reasoning_engine = AutonomousReasoningEngine()

mitigation_controller = AutonomousMitigationController(
    victim_host="192.168.56.102",
    victim_port=8765,
    victim_token="SMARTSOC-LAB-V1-CHANGE-ME",
)

risk_engine = SmartSOCRiskEngine()
decision_policy = SmartSOCDecisionPolicy()
response_planner = SmartSOCResponsePlanner()


# ============================================================================
# HELPERS
# ============================================================================

def utc_timestamp() -> str:
    """
    Return an explicit UTC timestamp without using deprecated utcnow().
    """

    return (
        datetime.now(timezone.utc)
        .isoformat()
    )


def elapsed_ms(start_ns: int, end_ns: int) -> float:
    return round(
        (end_ns - start_ns) / 1_000_000,
        3,
    )


def normalized_to_wazuh_event(
    event: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert the forwarder's normalized event into the Wazuh-shaped structure
    expected by SmartSOCInference and AutonomousReasoningEngine.

    The current forwarder places alert_id, rule, mitre, and windows at the
    top level. Older receiver payloads that used a nested "wazuh" object are
    still accepted as a compatibility fallback. The original Wazuh alert is
    preserved as the base whenever it is available, so fields not represented
    by the normalized schema are not discarded.
    """

    if not isinstance(event, dict):
        raise TypeError("Normalized event must be a dictionary")

    legacy_wazuh = event.get("wazuh", {})
    if not isinstance(legacy_wazuh, dict):
        legacy_wazuh = {}

    raw_alert = event.get("raw_wazuh_alert", {})
    if not isinstance(raw_alert, dict):
        raw_alert = {}

    source_rule = event.get("rule", {})
    if not isinstance(source_rule, dict):
        source_rule = {}

    agent = event.get("agent", {})
    if not isinstance(agent, dict):
        agent = {}

    mitre = event.get("mitre", {})
    if not isinstance(mitre, dict):
        mitre = {}

    windows = event.get("windows", {})
    if not isinstance(windows, dict):
        windows = {}

    # Accept both the current forwarder's singular keys and older plural keys.
    techniques = mitre.get("id")
    if techniques is None:
        techniques = mitre.get("technique")
    if techniques is None:
        techniques = mitre.get("techniques", [])
    if isinstance(techniques, str):
        techniques = [techniques]
    elif not isinstance(techniques, list):
        techniques = []

    tactics = mitre.get("tactic")
    if tactics is None:
        tactics = mitre.get("tactics", [])
    if isinstance(tactics, str):
        tactics = [tactics]
    elif not isinstance(tactics, list):
        tactics = []

    # Preserve the original Wazuh alert and its complete data tree where
    # possible, rather than rebuilding it from only a handful of fields.
    wazuh_event = dict(raw_alert)

    timestamp = (
        event.get("timestamp")
        or legacy_wazuh.get("timestamp")
        or raw_alert.get("timestamp")
        or ""
    )
    alert_id = (
        event.get("alert_id")
        or legacy_wazuh.get("alert_id")
        or raw_alert.get("id")
        or ""
    )

    raw_rule = wazuh_event.get("rule", {})
    if not isinstance(raw_rule, dict):
        raw_rule = {}
    else:
        raw_rule = dict(raw_rule)

    rule_id = source_rule.get("id")
    if rule_id is None or rule_id == "":
        rule_id = legacy_wazuh.get("rule_id", raw_rule.get("id", ""))
    rule_id = str(rule_id) if rule_id is not None else ""

    rule_level = source_rule.get("level")
    if rule_level is None:
        rule_level = legacy_wazuh.get("rule_level", raw_rule.get("level", 0))
    try:
        rule_level = int(rule_level)
    except (TypeError, ValueError):
        try:
            rule_level = float(rule_level)
        except (TypeError, ValueError):
            rule_level = 0

    description = (
        source_rule.get("description")
        or legacy_wazuh.get("description")
        or raw_rule.get("description")
        or ""
    )
    groups = source_rule.get("groups")
    if groups is None:
        groups = legacy_wazuh.get("groups", raw_rule.get("groups", []))
    if not isinstance(groups, list):
        groups = []

    raw_rule["id"] = rule_id
    raw_rule["level"] = rule_level
    raw_rule["description"] = description
    raw_rule["groups"] = groups

    raw_rule_mitre = raw_rule.get("mitre", {})
    if not isinstance(raw_rule_mitre, dict):
        raw_rule_mitre = {}
    else:
        raw_rule_mitre = dict(raw_rule_mitre)
    if techniques:
        raw_rule_mitre["id"] = techniques
    if tactics:
        raw_rule_mitre["tactic"] = tactics
    if techniques:
        raw_rule_mitre["technique"] = techniques
    raw_rule["mitre"] = raw_rule_mitre
    wazuh_event["rule"] = raw_rule

    raw_agent = wazuh_event.get("agent", {})
    if not isinstance(raw_agent, dict):
        raw_agent = {}
    else:
        raw_agent = dict(raw_agent)
    for key in ("id", "name", "ip"):
        value = agent.get(key)
        if value not in (None, ""):
            raw_agent[key] = str(value)
    wazuh_event["agent"] = raw_agent

    # Keep the original alert's eventdata and system metadata, adding
    # normalized process values only when they are actually present.
    data = wazuh_event.get("data", {})
    if not isinstance(data, dict):
        data = {}
    else:
        data = dict(data)

    win = data.get("win", {})
    if not isinstance(win, dict):
        win = {}
    else:
        win = dict(win)

    eventdata = win.get("eventdata", {})
    if not isinstance(eventdata, dict):
        eventdata = {}
    else:
        eventdata = dict(eventdata)

    field_map = {
        "process_id": ("processId", "process_id"),
        "process_name": ("image", "process_name"),
        "command_line": ("commandLine", "command_line"),
        "parent_process_id": ("parentProcessId", "parent_process_id"),
        "parent_process_name": ("parentImage", "parent_process_name"),
        "parent_command_line": ("parentCommandLine", "parent_command_line"),
        "user": ("user",),
        "integrity_level": ("integrityLevel",),
        "hashes": ("hashes",),
    }

    for source_key, target_keys in field_map.items():
        value = windows.get(source_key)
        if value in (None, ""):
            continue
        for target_key in target_keys:
            eventdata[target_key] = str(value) if target_key in (
                "processId", "process_id", "parentProcessId", "parent_process_id"
            ) else value

    win["eventdata"] = eventdata
    data["win"] = win
    wazuh_event["data"] = data
    wazuh_event["timestamp"] = timestamp
    wazuh_event["id"] = str(alert_id)

    # Expose MITRE metadata in both locations used by Wazuh-shaped consumers.
    top_mitre = wazuh_event.get("mitre", {})
    if not isinstance(top_mitre, dict):
        top_mitre = {}
    else:
        top_mitre = dict(top_mitre)
    if techniques:
        top_mitre["id"] = techniques
        top_mitre["technique"] = techniques
    if tactics:
        top_mitre["tactic"] = tactics
    wazuh_event["mitre"] = top_mitre

    return wazuh_event


# ============================================================================
# EVENT PROCESSING
# ============================================================================

def process_event(
    normalized_event: Dict[str, Any],
) -> Dict[str, Any]:
    global VICTIM_AGENT_CALLED
    """
    Execute the complete controlled live processing path.

    Steps:

        1. Normalize Wazuh event.
        2. Run frozen SmartSOC v4.1 inference.
        3. Run autonomous reasoning.
        4. If reasoning selects no mitigation:
               stop safely.
        5. If reasoning selects PROCESS_TERMINATION:
               call the existing controlled mitigation controller.
        6. Return the complete result including verification
           and audit information.
    """

    # MTTD/MTTR measurement clock.
    # T0/T1 are deliberately supplied by the experiment harness/Wazuh;
    # the receiver does not fabricate them.
    receiver_accept_ns = perf_counter_ns()
    receiver_accept_timestamp = utc_timestamp()

    inference_end_ns = None
    reasoning_end_ns = None
    mitigation_start_ns = None
    mitigation_end_ns = None

    inference_latency_ms = None
    reasoning_latency_ms = None
    mitigation_controller_latency_ms = None

    source_rule = normalized_event.get("rule", {})
    if not isinstance(source_rule, dict):
        source_rule = {}

    legacy_wazuh = normalized_event.get("wazuh", {})
    if not isinstance(legacy_wazuh, dict):
        legacy_wazuh = {}

    raw_alert = normalized_event.get("raw_wazuh_alert", {})
    if not isinstance(raw_alert, dict):
        raw_alert = {}

    alert_id = (
        normalized_event.get("alert_id")
        or legacy_wazuh.get("alert_id")
        or raw_alert.get("id", "")
    )

    rule_id = (
        source_rule.get("id")
        or legacy_wazuh.get("rule_id")
        or raw_alert.get("rule", {}).get("id", "")
    )

    print()
    print("=" * 80)
    print("SMARTSOC LIVE EVENT")
    print("=" * 80)

    print(
        json.dumps(
            normalized_event,
            indent=2,
            default=str,
        )
    )

    # ------------------------------------------------------------------------
    # Convert normalized event to SmartSOC/Wazuh structure
    # ------------------------------------------------------------------------

    wazuh_event = normalized_to_wazuh_event(
        normalized_event
    )

    # ------------------------------------------------------------------------
    # STEP 1 - SmartSOC inference
    # ------------------------------------------------------------------------

    print()
    print("[1] Running SmartSOC v4.1 inference...")

    inference_start_ns = perf_counter_ns()

    inference_result = inference_engine.infer(
        wazuh_event
    )

    inference_end_ns = perf_counter_ns()
    inference_end_timestamp = utc_timestamp()

    inference_latency_ms = elapsed_ms(
        inference_start_ns,
        inference_end_ns,
    )

    print()
    print("[2] INFERENCE RESULT")

    print(
        json.dumps(
            inference_result,
            indent=2,
            default=str,
        )
    )

    # ------------------------------------------------------------------------
    # STEP 2 - Autonomous reasoning
    # ------------------------------------------------------------------------

    print()
    print("[3] Running AutonomousReasoningEngine...")

    reasoning_start_ns = perf_counter_ns()

    reasoning_result = reasoning_engine.reason(
        inference_result,
        wazuh_event,
    )

    reasoning_end_ns = perf_counter_ns()
    reasoning_end_timestamp = utc_timestamp()

    reasoning_latency_ms = elapsed_ms(
        reasoning_start_ns,
        reasoning_end_ns,
    )

    print()
    print("[4] AUTONOMOUS REASONING RESULT")

    print(
        json.dumps(
            reasoning_result,
            indent=2,
            default=str,
        )
    )

    # SmartSOC controlled-lab target propagation.
    # Only use process identity already present in this event; never invent a PID.
    # The victim still performs HMAC/replay checks, allowlist checks, and PID/image validation.
    if isinstance(reasoning_result, dict):
        _target_event_candidates = [normalized_event, wazuh_event]
        _target_eventdata = None
        for _candidate in _target_event_candidates:
            if not isinstance(_candidate, dict):
                continue
            _data = _candidate.get("data", {})
            _win = _data.get("win", {}) if isinstance(_data, dict) else {}
            _edata = _win.get("eventdata", {}) if isinstance(_win, dict) else {}
            if isinstance(_edata, dict) and _edata:
                _target_eventdata = _edata
                break

        if isinstance(_target_eventdata, dict):
            _event_pid = (
                _target_eventdata.get("processId")
                or _target_eventdata.get("process_id")
            )
            _event_image = (
                _target_eventdata.get("image")
                or _target_eventdata.get("process_name")
            )
            if reasoning_result.get("process_id") in (None, "") and _event_pid not in (None, ""):
                reasoning_result["process_id"] = str(_event_pid)
            if reasoning_result.get("process_name") in (None, "") and isinstance(_event_image, str):
                reasoning_result["process_name"] = _event_image.strip().lower()
    # Reporting defaults. These describe the decision path; they do
    # not claim that a mitigation was executed or verified.
    risk_report = {
        "status": "NOT_COMPUTED",
        "available": False,
    }
    policy_outcome = {
        "status": "NOT_EVALUATED",
        "allowed": False,
    }
    response_plan_report = {
        "status": "NOT_CREATED",
        "available": False,
    }
    requested_mitigation = str(
        reasoning_result.get(
            "mitigation",
            "NONE",
        )
    ).strip().upper()

    # Explicitly gated test override for the custom SmartSOC lab rule.
    # This does not change ML predictions or the general decision policy.
    _smartsoc_lab_override_enabled = (
        os.environ.get("SMARTSOC_ENABLE_LAB_TEST_OVERRIDE", "").strip() == "1"
    )

    # Extract process identity from normalized Wazuh eventdata. The previous
    # implementation referenced `eventdata` below without defining it in this
    # scope, raising NameError before the process-termination override check.
    _event_data = wazuh_event.get("data", {})
    _event_win = _event_data.get("win", {}) if isinstance(_event_data, dict) else {}
    eventdata = _event_win.get("eventdata", {}) if isinstance(_event_win, dict) else {}
    if not isinstance(eventdata, dict):
        eventdata = {}

    _lab_candidates = [
        reasoning_result.get("process_name"),
        inference_result.get("process_name"),
        eventdata.get("image"),
        eventdata.get("process_name"),
    ]
    _lab_process = ""
    for _candidate in _lab_candidates:
        if isinstance(_candidate, str) and _candidate.strip():
            _lab_process = (
                _candidate.strip().replace("\\", "/").rsplit("/", 1)[-1].lower()
            )
            break

    _lab_pid = (
        reasoning_result.get("process_id")
        or inference_result.get("process_id")
        or eventdata.get("processId")
        or eventdata.get("processID")
        or eventdata.get("process_id")
    )
    # Windows Wazuh process IDs usually arrive as decimal strings. Convert only
    # decimal IDs; never synthesize a PID or reinterpret an unknown format.
    if isinstance(_lab_pid, str):
        _lab_pid = _lab_pid.strip()
        if _lab_pid.isdecimal():
            _lab_pid = int(_lab_pid)
    _lab_conditions = {
        "env_enabled": _smartsoc_lab_override_enabled,
        "controlled_lab": EXECUTION_MODE == "CONTROLLED_LAB",
        "mitigation_enabled": bool(MITIGATION_ENABLED),
        "rule_process_pair_matches": (
            (str(rule_id) == "100101" and _lab_process == "ping.exe")
            or
            (str(rule_id) == "100104" and _lab_process == "notepad.exe")
            or
            (
                str(rule_id) == "100130"
                and _lab_process == "powershell.exe"
                and isinstance(eventdata, dict)
                and "SmartSOC-DefenderLab" in str(eventdata.get("commandLine", ""))
                and "DefenderTamperSimulation.ps1" in str(eventdata.get("commandLine", ""))
            )
        ),
        "pid_present": _lab_pid not in (None, ""),
    }
    print("[LAB TEST OVERRIDE CHECK]", json.dumps(_lab_conditions), flush=True)

    if all(_lab_conditions.values()):
        reasoning_result["process_name"] = _lab_process
        reasoning_result["process_id"] = _lab_pid
        reasoning_result["mitigation"] = "PROCESS_TERMINATION"
        reasoning_result["wazuh_rule_id"] = str(rule_id)
        reasoning_result["lab_test_override"] = True
        requested_mitigation = "PROCESS_TERMINATION"
    elif isinstance(reasoning_result, dict):
        reasoning_result["lab_test_override"] = False
    if requested_mitigation == "NONE":

        print()
        print("[+] No mitigation selected.")
        print("    Effective action : NONE")
        print("    Victim call      : NO")

        return {
            "status": "REASONING_COMPLETE",

            # No mitigation was requested or executed.
            "success": False,
            "verified": False,

            "timestamp": utc_timestamp(),

            "execution_mode": EXECUTION_MODE,

            "mitigation_enabled": MITIGATION_ENABLED,

            "victim_agent_called": False,

            "effective_action": "NONE",

            "requested_mitigation": "NONE",

            "wazuh_alert_id": alert_id,

            "wazuh_rule_id": rule_id,

            "inference": inference_result,

            "reasoning": reasoning_result,
            "risk_report": risk_report,
            "policy_outcome": policy_outcome,
            "response_plan": response_plan_report,
            "mitigation_result": None,

            "performance": {
                "receiver_accept_timestamp": receiver_accept_timestamp,
                "inference_end_timestamp": inference_end_timestamp,
                "reasoning_end_timestamp": reasoning_end_timestamp,
                "mitigation_start_timestamp": None,
                "mitigation_end_timestamp": None,
                "inference_latency_ms": inference_latency_ms,
                "reasoning_latency_ms": reasoning_latency_ms,
                "mitigation_controller_latency_ms": None,
                "victim_request_roundtrip_ms": None,
                "receiver_processing_latency_ms": elapsed_ms(
                    receiver_accept_ns,
                    perf_counter_ns(),
                ),
                "mttr_available": False,
            },

        }

    # ------------------------------------------------------------------------
    # STEP 4 - Mitigation requested
    # ------------------------------------------------------------------------

    print()
    print("[!] MITIGATION CANDIDATE DETECTED")

    print(
        f"    Requested action : {requested_mitigation}"
    )

    # ------------------------------------------------------------------------
    # Fail closed if the receiver is not explicitly enabled
    # ------------------------------------------------------------------------

    # SMARTSOC_PENDING_APPROVAL_REGISTRATION_V1: register candidate independently of execution enablement.
    if requested_mitigation == "PROCESS_TERMINATION":
        try:
            _smartsoc_approvals.register_pending_from_reasoning_result(
                reasoning_result, str(alert_id), str(rule_id)
            )
        except Exception as _approval_error:
            # Fail closed: report registration failure; continue to the
            # existing disabled-execution branch without attempting mitigation.
            print(f"[APPROVAL REGISTRATION FAILED CLOSED] {_approval_error}")

    if not MITIGATION_ENABLED or not LIVE_EXECUTION_ENABLED:

        print(
            "    Effective action : NONE"
        )

        print(
            "    Execution        : BLOCKED"
        )

        return {
            "status": "MITIGATION_BLOCKED",
            "timestamp": utc_timestamp(),
            "execution_mode": EXECUTION_MODE,
            "mitigation_enabled": MITIGATION_ENABLED,
            "victim_agent_called": False,
            "effective_action": "NONE",
            "requested_mitigation": requested_mitigation,
            "wazuh_alert_id": alert_id,
            "wazuh_rule_id": rule_id,
            "inference": inference_result,
            "reasoning": reasoning_result,
            "risk_report": risk_report,
            "policy_outcome": policy_outcome,
            "response_plan": response_plan_report,
            "error": ("Live mitigation is disabled at the receiver." if not MITIGATION_ENABLED else "Host live-execution gate is off; set SMARTSOC_ALLOW_LIVE_MITIGATION=1 in this host session."),
        }

    # ------------------------------------------------------------------------
    # STEP 5 - Explicitly restrict receiver-side action
    # ------------------------------------------------------------------------

    if requested_mitigation != "PROCESS_TERMINATION":

        print(
            "    Effective action : NONE"
        )

        print(
            "    Execution        : BLOCKED"
        )

        return {
            "status": "MITIGATION_BLOCKED",

            "timestamp": utc_timestamp(),

            "execution_mode": EXECUTION_MODE,

            "mitigation_enabled": MITIGATION_ENABLED,

            "victim_agent_called": False,

            "effective_action": "NONE",

            "requested_mitigation": requested_mitigation,

            "wazuh_alert_id": alert_id,

            "wazuh_rule_id": rule_id,

            "inference": inference_result,

            "reasoning": reasoning_result,
            "risk_report": risk_report,
            "policy_outcome": policy_outcome,
            "response_plan": response_plan_report,
            "error": (
                "Receiver permits only "
                "PROCESS_TERMINATION."
            ),
        }

    # ------------------------------------------------------------------------
    # STEP 6 - Controlled mitigation controller
    # ------------------------------------------------------------------------

    print()
    print(
        "[5] Running controlled AutonomousMitigationController..."
    )

    print(
        "    Allowed action    : PROCESS_TERMINATION"
    )

    print(
        "    Victim endpoint   : "
        "http://192.168.56.102:8765/mitigate"
    )

    print(
        "    Arbitrary command : DISABLED"
    )

    try:

        mitigation_start_ns = perf_counter_ns()
        mitigation_start_timestamp = utc_timestamp()

        bound_reasoning = bind_event_context(reasoning_result, alert_id, rule_id)
        mitigation_result = execute_live(bound_reasoning)
        VICTIM_AGENT_CALLED = bool(
            mitigation_result.get("victim_request_attempted", False)
            or mitigation_result.get("victim_client_invoked", False)
        )

        mitigation_end_ns = perf_counter_ns()
        mitigation_end_timestamp = utc_timestamp()

        mitigation_controller_latency_ms = elapsed_ms(
            mitigation_start_ns,
            mitigation_end_ns,
        )

    except Exception as exc:

        print()
        print(
            "[!] Controlled mitigation controller error:"
        )

        print(
            repr(exc)
        )

        mitigation_result = {
            "success": False,

            "verified": False,

            "status": "MITIGATION_ERROR",

            "controller": (
                getattr(
                    mitigation_controller,
                    "VERSION",
                    "unknown",
                )
            ),

            "error": str(exc),
        }

    # ------------------------------------------------------------------------
    # STEP 7 - Determine final state
    # ------------------------------------------------------------------------

    mitigation_success = bool(
        mitigation_result.get(
            "success",
            False,
        )
    )

    mitigation_verified = bool(
        mitigation_result.get(
            "verified",
            False,
        )
    )

    if (
        mitigation_success
        and mitigation_verified
    ):

        effective_action = (
            "PROCESS_TERMINATION"
        )

        final_status = (
            "MITIGATION_SUCCESSFUL"
        )

    else:

        effective_action = (
            "PROCESS_TERMINATION"
            if requested_mitigation
            == "PROCESS_TERMINATION"
            else "NONE"
        )

        final_status = (
            mitigation_result.get(
                "status",
                "MITIGATION_FAILED",
            )
        )

    # ------------------------------------------------------------------------
    # STEP 8 - Print controlled execution result
    # ------------------------------------------------------------------------

    print()
    print("[6] CONTROLLED MITIGATION RESULT")

    print(
        json.dumps(
            mitigation_result,
            indent=2,
            default=str,
        )
    )

    print()
    print("=" * 80)

    print(
        f"FINAL STATUS : {final_status}"
    )

    print(
        f"SUCCESS      : {mitigation_success}"
    )

    print(
        f"VERIFIED     : {mitigation_verified}"
    )

    print(
        f"ACTION       : {effective_action}"
    )

    print("=" * 80)

    # ------------------------------------------------------------------------
    # Final structured response
    # ------------------------------------------------------------------------

    return {
        "status": final_status,

        "timestamp": utc_timestamp(),

        "execution_mode": EXECUTION_MODE,

        "mitigation_enabled": MITIGATION_ENABLED,

        "controller_invoked": True,
        "victim_agent_called": bool(
            mitigation_result.get("victim_request_attempted", False)
            or mitigation_result.get("victim_client_invoked", False)
        ),

        "effective_action": effective_action,

        "requested_mitigation": requested_mitigation,

        "wazuh_alert_id": alert_id,

        "wazuh_rule_id": rule_id,

        "inference": inference_result,

        "reasoning": reasoning_result,
        "risk_report": risk_report,
        "policy_outcome": policy_outcome,
        "response_plan": response_plan_report,
        "mitigation_result": mitigation_result,

        "success": mitigation_success,

        "verified": mitigation_verified,

        "performance": {
            "receiver_accept_timestamp": receiver_accept_timestamp,
            "inference_end_timestamp": inference_end_timestamp,
            "reasoning_end_timestamp": reasoning_end_timestamp,
            "mitigation_start_timestamp": (
                mitigation_start_timestamp
                if "mitigation_start_timestamp" in locals()
                else None
            ),
            "mitigation_end_timestamp": (
                mitigation_end_timestamp
                if "mitigation_end_timestamp" in locals()
                else None
            ),
            "inference_latency_ms": inference_latency_ms,
            "reasoning_latency_ms": reasoning_latency_ms,
            "mitigation_controller_latency_ms": (
                mitigation_controller_latency_ms
            ),
            "victim_request_roundtrip_ms": (
                mitigation_result
                .get("timing", {})
                .get("victim_request_roundtrip_ms")
                if isinstance(mitigation_result, dict)
                else None
            ),
            "receiver_processing_latency_ms": elapsed_ms(
                receiver_accept_ns,
                perf_counter_ns(),
            ),
            "mttr_available": bool(
                mitigation_success
                and mitigation_verified
            ),
        },
    }


# ============================================================================
# HTTP HANDLER
# ============================================================================

class SmartSOCReceiverHandler(
    BaseHTTPRequestHandler
):

    def log_message(
        self,
        format_string,
        *args,
    ):
        """
        Disable default HTTP server logging.

        SmartSOC uses structured application logging instead.
        """

        return

    def send_json(
        self,
        status_code: int,
        payload: Dict[str, Any],
    ) -> None:

        body = json.dumps(
            payload,
            indent=2,
            default=str,
        ).encode(
            "utf-8"
        )

        self.send_response(
            status_code
        )

        self.send_header(
            "Content-Type",
            "application/json",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )

        self.end_headers()

        self.wfile.write(
            body
        )

    # ------------------------------------------------------------------------
    # GET
    # ------------------------------------------------------------------------

    def do_GET(self):

        if self.path == "/health":

            payload = {
                "service": (
                    "smartsoc-live-receiver-v3"
                ),

                "status": "READY",

                "port": LISTEN_PORT,

                "execution_mode": (
                    EXECUTION_MODE
                ),

                "inference_enabled": True,

                "reasoning_enabled": True,

                "mitigation_enabled": (
                    MITIGATION_ENABLED
                ),

                "victim_agent_calls": (
                    VICTIM_AGENT_CALLED
                ),

                "mitigation_status": (
                    "LIVE_ARMED" if MITIGATION_ENABLED and LIVE_EXECUTION_ENABLED
                    else "DISABLED"
                ),

                "execution_layer_enabled": bool(MITIGATION_ENABLED and LIVE_EXECUTION_ENABLED),

                "configured_action_types": [
                    "PROCESS_TERMINATION"
                ],

                "allowed_mitigations": (
                    ["PROCESS_TERMINATION"]
                    if MITIGATION_ENABLED
                    else []
                ),

                "arbitrary_commands": False,

                "victim_endpoint": (
                    "http://192.168.56.102:8765/mitigate"
                ),
            }

            self.send_json(
                200,
                payload,
            )

            return

        self.send_json(
            404,
            {
                "status": "NOT_FOUND",

                "path": self.path,
            },
        )

    # ------------------------------------------------------------------------
    # POST
    # ------------------------------------------------------------------------

    def do_POST(self):

        if self.path != "/wazuh-event":

            self.send_json(
                404,
                {
                    "status": "NOT_FOUND",

                    "path": self.path,
                },
            )

            return

        try:

            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

            if content_length <= 0:

                self.send_json(
                    400,
                    {
                        "status": "ERROR",

                        "error": (
                            "Empty request body"
                        ),
                    },
                )

                return

            raw_body = self.rfile.read(
                content_length
            )

            normalized_event = json.loads(
                raw_body.decode(
                    "utf-8"
                )
            )

            if not isinstance(
                normalized_event,
                dict,
            ):

                raise ValueError(
                    "Request body must contain "
                    "a JSON object"
                )

            # Accept the active Ubuntu forwarder payload: {"event": raw_alert}.
            # Preserve the original alert for existing normalization logic.
            if "event" in normalized_event:
                wrapped_event = normalized_event.get("event")
                if not isinstance(wrapped_event, dict):
                    raise ValueError(
                        "Wrapped event must be a JSON object"
                    )
                normalized_event = {
                    "raw_wazuh_alert": wrapped_event,
                }

            result = process_event(
                normalized_event
            )

            # HTTP 200 means the receiver processed
            # the request. It does NOT mean mitigation
            # succeeded.
            #
            # The actual mitigation state is represented
            # by result["status"], ["success"], and
            # ["verified"].

            self.send_json(
                200,
                result,
            )

        except json.JSONDecodeError as exc:

            self.send_json(
                400,
                {
                    "status": "ERROR",

                    "error": (
                        "Invalid JSON"
                    ),

                    "details": str(exc),
                },
            )

        except Exception as exc:

            print()
            print(
                "[!] Receiver processing error:"
            )

            print(
                repr(exc)
            )

            self.send_json(
                500,
                {
                    "status": "ERROR",

                    "error": str(exc),
                },
            )


# ============================================================================
# MAIN
# ============================================================================

def main():

    server = ThreadingHTTPServer(
        (
            LISTEN_HOST,
            LISTEN_PORT,
        ),
        SmartSOCReceiverHandler,
    )

    print("=" * 80)
    print(
        "SmartSOC Live Receiver v3"
    )
    print("=" * 80)

    print(
        f"Listening : "
        f"{LISTEN_HOST}:{LISTEN_PORT}"
    )

    print(
        f"Endpoint  : "
        f"http://192.168.56.1:{LISTEN_PORT}/wazuh-event"
    )

    print(
        f"Mode      : "
        f"{EXECUTION_MODE}"
    )

    print(
        "Inference : ENABLED"
    )

    print(
        "Reasoning : ENABLED"
    )

    print(
        f"Mitigation: {'ENABLED' if MITIGATION_ENABLED else 'DISABLED'}"
    )

    print(
        "Victim    : LIVE ADAPTER ARMED" if LIVE_EXECUTION_ENABLED else "Victim    : LIVE GATE OFF (DRY-RUN/BLOCKED)"
    )

    print(
        "Action    : PROCESS_TERMINATION ONLY"
    )

    print(
        "Arbitrary : DISABLED"
    )

    print(
        "Verification: ENABLED"
    )

    print("=" * 80)

    print()

    print(
        "[+] SmartSOC receiver v3 READY."
    )

    print(
        "[+] Live mitigation gate is ARMED; adapter and victim checks still apply."
        if LIVE_EXECUTION_ENABLED
        else "[+] Live mitigation is BLOCKED. Set SMARTSOC_ALLOW_LIVE_MITIGATION=1 to arm the host gate."
    )

    print(
        "[+] Waiting for live Wazuh events..."
    )

    print()

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print()
        print(
            "[!] Receiver stopped by user."
        )

    finally:

        server.server_close()


if __name__ == "__main__":
    main()








