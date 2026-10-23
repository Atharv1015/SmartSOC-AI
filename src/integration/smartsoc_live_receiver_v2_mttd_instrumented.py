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


# ============================================================================
# SMARTSOC COMPONENTS
# ============================================================================

from production_inference_windows_apt_v4_1 import (
    SmartSOCInference,
)

from autonomous.autonomous_reasoning_v1 import (
    AutonomousReasoningEngine,
)

from autonomous.autonomous_mitigation_controller_v1 import (
    AutonomousMitigationController,
)


# ============================================================================
# RECEIVER CONFIGURATION
# ============================================================================

LISTEN_HOST = "0.0.0.0"
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

VICTIM_AGENT_CALLED = True


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
    Convert the normalized event produced by
    wazuh_live_forwarder_v2 back into the Wazuh-shaped
    structure expected by SmartSOCInference and
    AutonomousReasoningEngine.
    """

    wazuh = event.get(
        "wazuh",
        {},
    )

    agent = event.get(
        "agent",
        {},
    )

    mitre = event.get(
        "mitre",
        {},
    )

    windows = event.get(
        "windows",
        {},
    )

    techniques = mitre.get(
        "techniques",
        [],
    )

    tactics = mitre.get(
        "tactics",
        [],
    )

    rule_id = str(
        wazuh.get(
            "rule_id",
            "",
        )
    )

    rule_level = wazuh.get(
        "rule_level",
        0,
    )

    description = wazuh.get(
        "description",
        "",
    )

    event_data = {
        "processId": str(
            windows.get(
                "process_id",
                "",
            )
        ),

        "process_id": str(
            windows.get(
                "process_id",
                "",
            )
        ),

        "image": windows.get(
            "process_name",
            "",
        ),

        "process_name": windows.get(
            "process_name",
            "",
        ),

        "commandLine": windows.get(
            "command_line",
            "",
        ),

        "command_line": windows.get(
            "command_line",
            "",
        ),

        "parentProcessId": str(
            windows.get(
                "parent_process_id",
                "",
            )
        ),

        "parent_process_id": str(
            windows.get(
                "parent_process_id",
                "",
            )
        ),

        "parentImage": windows.get(
            "parent_process_name",
            "",
        ),

        "parent_process_name": windows.get(
            "parent_process_name",
            "",
        ),

        "parentCommandLine": windows.get(
            "parent_command_line",
            "",
        ),

        "parent_command_line": windows.get(
            "parent_command_line",
            "",
        ),

        "user": windows.get(
            "user",
            "",
        ),

        "integrityLevel": windows.get(
            "integrity_level",
            "",
        ),

        "hashes": windows.get(
            "hashes",
            "",
        ),
    }

    event_data = {
        key: value
        for key, value in event_data.items()
        if value not in (
            "",
            None,
        )
    }

    return {
        "timestamp": wazuh.get(
            "timestamp",
            "",
        ),

        "id": wazuh.get(
            "alert_id",
            "",
        ),

        "rule": {
            "id": rule_id,

            "level": rule_level,

            "description": description,

            "groups": wazuh.get(
                "groups",
                [],
            ),

            "mitre": {
                "id": techniques,

                "tactic": tactics,
            },
        },

        "agent": {
            "id": agent.get(
                "id",
                "",
            ),

            "name": agent.get(
                "name",
                "",
            ),

            "ip": agent.get(
                "ip",
                "",
            ),
        },

        "data": {
            "win": {
                "eventdata": event_data,
            }
        },

        "mitre": {
            "id": techniques,

            "tactic": tactics,
        },
    }


# ============================================================================
# EVENT PROCESSING
# ============================================================================

def process_event(
    normalized_event: Dict[str, Any],
) -> Dict[str, Any]:
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

    alert_id = (
        normalized_event
        .get(
            "wazuh",
            {},
        )
        .get(
            "alert_id",
            "",
        )
    )

    rule_id = (
        normalized_event
        .get(
            "wazuh",
            {},
        )
        .get(
            "rule_id",
            "",
        )
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

    requested_mitigation = str(
        reasoning_result.get(
            "mitigation",
            "NONE",
        )
    ).strip().upper()

    # ------------------------------------------------------------------------
    # STEP 3 - No mitigation selected
    # ------------------------------------------------------------------------

    if requested_mitigation == "NONE":

        print()
        print("[+] No mitigation selected.")
        print("    Effective action : NONE")
        print("    Victim call      : NO")

        return {
            "status": "REASONING_COMPLETE",

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

    if not MITIGATION_ENABLED:

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

            "error": (
                "Live mitigation is disabled at the receiver."
            ),
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

        mitigation_result = (
            mitigation_controller.execute(
                reasoning_result
            )
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

        "victim_agent_called": True,

        "effective_action": effective_action,

        "requested_mitigation": requested_mitigation,

        "wazuh_alert_id": alert_id,

        "wazuh_rule_id": rule_id,

        "inference": inference_result,

        "reasoning": reasoning_result,

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

                "allowed_mitigations": [
                    "PROCESS_TERMINATION"
                ],

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
        "Mitigation: ENABLED"
    )

    print(
        "Victim    : ENABLED"
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
        "[+] Controlled live mitigation is ENABLED."
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