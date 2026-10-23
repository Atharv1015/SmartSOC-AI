"""
SmartSOC Victim Mitigation Client v1

Purpose:
    Send explicitly structured defensive mitigation requests
    from the SmartSOC SOC controller to the Windows victim agent.

Security properties:
    - No arbitrary shell commands.
    - Only PROCESS_TERMINATION is supported.
    - PID must be a positive integer.
    - Process name must be supplied by the SmartSOC reasoning layer.
    - The Windows victim agent performs final validation.
    - Authentication uses X-SmartSOC-Token.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict


class VictimMitigationClient:
    """
    Controlled client for the SmartSOC Windows defensive agent.
    """

    VERSION = "victim-mitigation-client-v1"

    SUPPORTED_ACTIONS = frozenset({
        "PROCESS_TERMINATION",
    })

    def __init__(
        self,
        victim_host: str = "192.168.56.102",
        victim_port: int = 8765,
        token: str = "SMARTSOC-LAB-V1-CHANGE-ME",
        timeout_seconds: int = 15,
    ) -> None:

        self.victim_host = victim_host
        self.victim_port = int(victim_port)
        self.token = token
        self.timeout_seconds = int(timeout_seconds)

        self.endpoint = (
            f"http://{self.victim_host}:"
            f"{self.victim_port}/mitigate"
        )

    # ------------------------------------------------------------------
    # TARGET VALIDATION
    # ------------------------------------------------------------------

    @staticmethod
    def validate_process_target(
        reasoning_result: Dict[str, Any],
    ) -> tuple[int, str]:

        if not isinstance(reasoning_result, dict):
            raise ValueError(
                "Reasoning result must be a dictionary."
            )

        mitigation = str(
            reasoning_result.get(
                "mitigation",
                "",
            )
        ).strip().upper()

        if mitigation != "PROCESS_TERMINATION":
            raise ValueError(
                "Only PROCESS_TERMINATION is supported."
            )

        if mitigation not in VictimMitigationClient.SUPPORTED_ACTIONS:
            raise ValueError(
                f"Unsupported mitigation action: {mitigation}"
            )

        process_id = reasoning_result.get(
            "process_id"
        )

        if isinstance(process_id, bool):
            raise ValueError(
                "process_id must be an integer."
            )

        try:
            pid = int(process_id)
        except (
            TypeError,
            ValueError,
        ):
            raise ValueError(
                "process_id must be a valid integer."
            )

        if pid <= 0:
            raise ValueError(
                "process_id must be greater than zero."
            )

        process_name = reasoning_result.get(
            "process_name"
        )

        if not isinstance(
            process_name,
            str,
        ):
            raise ValueError(
                "process_name is required."
            )

        process_name = process_name.strip()

        if not process_name:
            raise ValueError(
                "process_name cannot be empty."
            )

        return pid, process_name

    # ------------------------------------------------------------------
    # REQUEST CONSTRUCTION
    # ------------------------------------------------------------------

    def build_request(
        self,
        reasoning_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        pid, process_name = (
            self.validate_process_target(
                reasoning_result
            )
        )

        return {
            "action": "PROCESS_TERMINATION",

            "target": {
                "pid": pid,
                "process_name": process_name,
            },

            "context": {
                "technique": reasoning_result.get(
                    "technique"
                ),

                "confidence": reasoning_result.get(
                    "confidence"
                ),

                "objective": reasoning_result.get(
                    "objective"
                ),

                "wazuh_rule_id": reasoning_result.get(
                    "wazuh_rule_id"
                ),

                "wazuh_rule_level": reasoning_result.get(
                    "wazuh_rule_level"
                ),

                "reason": reasoning_result.get(
                    "reason"
                ),
            },
        }

    # ------------------------------------------------------------------
    # REMOTE MITIGATION
    # ------------------------------------------------------------------

    def execute(
        self,
        reasoning_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        try:

            request_body = self.build_request(
                reasoning_result
            )

        except ValueError as exc:

            return {
                "success": False,
                "verified": False,
                "status": "REQUEST_BLOCKED",
                "error": str(exc),
                "client": self.VERSION,
            }

        payload = json.dumps(
            request_body
        ).encode("utf-8")

        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-SmartSOC-Token": self.token,
            },
        )

        try:

            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:

                raw_response = (
                    response
                    .read()
                    .decode(
                        "utf-8",
                        errors="replace",
                    )
                )

                try:
                    victim_response = json.loads(
                        raw_response
                    )

                except json.JSONDecodeError:

                    return {
                        "success": False,
                        "verified": False,
                        "status": "INVALID_VICTIM_RESPONSE",
                        "http_status": response.status,
                        "error": (
                            "Victim agent returned "
                            "invalid JSON."
                        ),
                        "raw_response": raw_response,
                        "request": request_body,
                    }

                return {
                    "success": bool(
                        victim_response.get(
                            "success"
                        )
                    ),

                    "verified": bool(
                        victim_response.get(
                            "verified"
                        )
                    ),

                    "status": (
                        "MITIGATION_SUCCESSFUL"
                        if victim_response.get(
                            "success"
                        )
                        and victim_response.get(
                            "verified"
                        )
                        else "MITIGATION_FAILED"
                    ),

                    "http_status": response.status,

                    "client": self.VERSION,

                    "endpoint": self.endpoint,

                    "request": request_body,

                    "victim_response": (
                        victim_response
                    ),
                }

        except urllib.error.HTTPError as exc:

            response_body = (
                exc.read()
                .decode(
                    "utf-8",
                    errors="replace",
                )
            )

            return {
                "success": False,
                "verified": False,
                "status": "VICTIM_REJECTED_REQUEST",
                "http_status": exc.code,
                "client": self.VERSION,
                "endpoint": self.endpoint,
                "request": request_body,
                "error": (
                    "Victim agent rejected "
                    "the mitigation request."
                ),
                "victim_response": response_body,
            }

        except urllib.error.URLError as exc:

            return {
                "success": False,
                "verified": False,
                "status": "VICTIM_UNREACHABLE",
                "http_status": None,
                "client": self.VERSION,
                "endpoint": self.endpoint,
                "request": request_body,
                "error": str(exc),
            }

        except TimeoutError:

            return {
                "success": False,
                "verified": False,
                "status": "VICTIM_TIMEOUT",
                "http_status": None,
                "client": self.VERSION,
                "endpoint": self.endpoint,
                "request": request_body,
                "error": (
                    "Victim agent request timed out."
                ),
            }

        except Exception as exc:

            return {
                "success": False,
                "verified": False,
                "status": "CLIENT_ERROR",
                "http_status": None,
                "client": self.VERSION,
                "endpoint": self.endpoint,
                "request": request_body,
                "error": str(exc),
            }


if __name__ == "__main__":

    print("=" * 72)
    print(
        "SmartSOC VICTIM MITIGATION CLIENT v1"
    )
    print("=" * 72)

    client = VictimMitigationClient()

    print()
    print("Client version :", client.VERSION)
    print("Victim endpoint:", client.endpoint)
    print(
        "Allowed actions:",
        sorted(client.SUPPORTED_ACTIONS),
    )

    print()
    print(
        "No mitigation is executed by this "
        "module self-test."
    )

    print("=" * 72)