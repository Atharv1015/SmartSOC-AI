"""
SmartSOC Live Receiver v1

Receives normalized events from the Ubuntu Wazuh live forwarder.

PHASE 2:
    Wazuh -> SmartSOC inference -> reasoning/policy visibility

IMPORTANT:
    This version is LOG_ONLY.
    It does NOT execute mitigation.
    It does NOT call the victim agent.
    It does NOT execute commands.

The existing frozen SmartSOC production inference module is used
without modification.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

# ---------------------------------------------------------------------
# SmartSOC imports
# ---------------------------------------------------------------------

PROJECT_ROOT = r"C:\Users\athar\SmartSOC-AI"
SRC_DIR = PROJECT_ROOT + r"\src"

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from production_inference_windows_apt_v4_1 import SmartSOCInference


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8787

EXECUTION_MODE = "LOG_ONLY"

INFERENCE_ENGINE = SmartSOCInference()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def json_response(
    handler: BaseHTTPRequestHandler,
    status_code: int,
    payload: Dict[str, Any],
) -> None:

    body = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")

    handler.send_response(status_code)

    handler.send_header(
        "Content-Type",
        "application/json; charset=utf-8",
    )

    handler.send_header(
        "Content-Length",
        str(len(body)),
    )

    handler.end_headers()

    handler.wfile.write(body)


def convert_to_wazuh_shape(
    event: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert the normalized forwarder event back into the structure
    expected by the existing production inference interface.

    No model features are modified here.
    """

    wazuh = event.get("wazuh", {})
    agent = event.get("agent", {})
    mitre = event.get("mitre", {})
    windows = event.get("windows", {})

    return {
        "timestamp": wazuh.get("timestamp"),

        "id": wazuh.get("alert_id"),

        "rule": {
            "id": wazuh.get("rule_id"),
            "level": wazuh.get("rule_level"),
            "description": wazuh.get("description"),
            "groups": wazuh.get("groups", []),
            "mitre": {
                "id": mitre.get("techniques", []),
                "tactic": mitre.get("tactics", []),
            },
        },

        "agent": {
            "id": agent.get("id"),
            "name": agent.get("name"),
            "ip": agent.get("ip"),
        },

        "data": {
            "win": {
                "system": {
                    "eventID": windows.get("event_id"),
                    "channel": windows.get("channel"),
                    "computer": windows.get("computer"),
                },

                "eventdata": {
                    "processId": windows.get("process_id"),
                    "image": windows.get("process_name"),
                    "commandLine": windows.get("command_line"),
                    "parentProcessId": windows.get(
                        "parent_process_id"
                    ),
                    "parentImage": windows.get(
                        "parent_process_name"
                    ),
                    "parentCommandLine": windows.get(
                        "parent_command_line"
                    ),
                    "user": windows.get("user"),
                    "integrityLevel": windows.get(
                        "integrity_level"
                    ),
                    "hashes": windows.get("hashes"),
                },
            }
        },
    }


def print_event_summary(
    event: Dict[str, Any],
) -> None:

    wazuh = event.get("wazuh", {})
    agent = event.get("agent", {})
    mitre = event.get("mitre", {})
    windows = event.get("windows", {})

    print()
    print("=" * 80)
    print("SMARTSOC LIVE EVENT RECEIVED")
    print("=" * 80)

    print(
        f"Wazuh Alert ID : {wazuh.get('alert_id')}"
    )

    print(
        f"Rule           : {wazuh.get('rule_id')} "
        f"- {wazuh.get('description')}"
    )

    print(
        f"Agent          : {agent.get('name')} "
        f"({agent.get('ip')})"
    )

    print(
        f"MITRE          : "
        f"{', '.join(mitre.get('techniques', []))}"
    )

    print(
        f"Process        : {windows.get('process_name')}"
    )

    print(
        f"PID            : {windows.get('process_id')}"
    )

    print(
        f"Parent         : {windows.get('parent_process_name')}"
    )

    print(
        f"Parent PID     : {windows.get('parent_process_id')}"
    )

    print(
        f"Command        : {windows.get('command_line')}"
    )

    print("=" * 80)


# ---------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------

class SmartSOCHandler(BaseHTTPRequestHandler):

    def log_message(
        self,
        format: str,
        *args: Any,
    ) -> None:
        """
        Keep default HTTP access logging concise.
        """
        print(
            "[HTTP]",
            format % args,
        )

    def do_GET(self) -> None:

        if self.path == "/health":

            json_response(
                self,
                200,
                {
                    "service": "smartsoc-live-receiver-v1",
                    "status": "READY",
                    "port": LISTEN_PORT,
                    "execution_mode": EXECUTION_MODE,
                    "mitigation_enabled": False,
                    "victim_agent_calls": False,
                },
            )

            return

        json_response(
            self,
            404,
            {
                "error": "not_found",
            },
        )

    def do_POST(self) -> None:

        if self.path != "/wazuh-event":
            json_response(
                self,
                404,
                {
                    "error": "not_found",
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
                raise ValueError(
                    "Request body is empty"
                )

            raw_body = self.rfile.read(
                content_length
            )

            event = json.loads(
                raw_body.decode("utf-8")
            )

            if not isinstance(event, dict):
                raise ValueError(
                    "JSON root must be an object"
                )

            print_event_summary(event)

            # ---------------------------------------------------------
            # Convert to the existing production inference interface.
            # ---------------------------------------------------------

            wazuh_event = convert_to_wazuh_shape(
                event
            )

            print()
            print(
                "[+] Running SmartSOC v4.1 inference..."
            )

            inference_result = INFERENCE_ENGINE.infer(
                wazuh_event
            )

            print()
            print("-" * 80)
            print("SMARTSOC INFERENCE RESULT")
            print("-" * 80)

            print(
                json.dumps(
                    inference_result,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )

            print()
            print("-" * 80)
            print("PHASE 2 EXECUTION POLICY")
            print("-" * 80)
            print(
                "Execution mode      : LOG_ONLY"
            )
            print(
                "Automatic execution : DISABLED"
            )
            print(
                "Victim agent call   : DISABLED"
            )
            print(
                "Process termination : DISABLED"
            )
            print("-" * 80)

            response = {
                "status": "INFERENCE_COMPLETE",

                "execution_mode": EXECUTION_MODE,

                "mitigation_enabled": False,

                "victim_agent_called": False,

                "wazuh_alert_id": event.get(
                    "wazuh",
                    {},
                ).get("alert_id"),

                "inference": inference_result,
            }

            json_response(
                self,
                200,
                response,
            )

        except Exception as exc:

            print(
                f"[!] Receiver error: {exc}",
                file=sys.stderr,
            )

            json_response(
                self,
                500,
                {
                    "status": "ERROR",
                    "error": str(exc),
                },
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:

    print("=" * 80)
    print("SmartSOC Live Receiver v1")
    print("=" * 80)
    print(
        f"Listening : {LISTEN_HOST}:{LISTEN_PORT}"
    )
    print(
        f"Endpoint  : http://<host>:{LISTEN_PORT}/wazuh-event"
    )
    print(
        f"Mode      : {EXECUTION_MODE}"
    )
    print(
        "Mitigation: DISABLED"
    )
    print("=" * 80)

    server = ThreadingHTTPServer(
        (
            LISTEN_HOST,
            LISTEN_PORT,
        ),
        SmartSOCHandler,
    )

    try:

        print()
        print(
            "[+] SmartSOC receiver READY."
        )
        print(
            "[+] Waiting for live Wazuh events..."
        )
        print()

        server.serve_forever()

    except KeyboardInterrupt:

        print()
        print(
            "[+] Receiver stopped."
        )

    finally:

        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )