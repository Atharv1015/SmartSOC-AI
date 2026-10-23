"""
SmartSOC Autonomous Mitigation Controller v3

Controlled autonomous defensive endpoint-mitigation controller.

Pipeline:

    SmartSOC ML
        |
        v
    Autonomous Reasoning
        |
        v
    Central Execution Policy
        |
        v
    Explicit Action Allowlist
        |
        v
    VictimMitigationClient
        |
        v
    Windows Defensive Agent
        |
        v
    PROCESS_TERMINATION
        |
        v
    Endpoint Verification
        |
        v
    Audit Logging

Security properties:

    - Only PROCESS_TERMINATION is supported.
    - Central execution policy is enforced.
    - Controlled laboratory mode is required.
    - Unknown actions are denied.
    - Arbitrary shell commands are never accepted.
    - PID must be a positive integer.
    - Process name must be supplied.
    - Victim agent performs final validation.
    - Endpoint modification requires explicit allowlisting.
    - Unrestricted autonomous actions remain disabled.
"""

from __future__ import annotations
from integration.local_approval_console_v1 import execute_authorized_client_call

import os
import sys
from typing import Any, Dict


# ============================================================================
# PROJECT PATH SETUP
# ============================================================================

# This file is normally executed as:
#
#     python src\autonomous\autonomous_mitigation_controller_v1.py
#
# In that mode Python places src\autonomous on sys.path.
# Add the project's src directory explicitly so that:
#
#     integration.*
#     autonomous.*
#
# can be imported reliably.

CURRENT_FILE = os.path.abspath(__file__)
AUTONOMOUS_DIR = os.path.dirname(CURRENT_FILE)
SRC_DIR = os.path.dirname(AUTONOMOUS_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================================
# IMPORTS
# ============================================================================

from integration.execution_config_v1 import (
    EXECUTION_LAYER_ENABLED,
    AUTOMATIC_EXECUTION,
    AUTOMATIC_CONTAINMENT,
    CONTROLLED_LAB_MODE,
    HUMAN_AUTHORIZATION_REQUIRED,
    is_action_allowed,
    validate_safety_invariants,
)

from autonomous.autonomous_reasoning_v1 import (
    AutonomousReasoningEngine,
)

from autonomous.victim_mitigation_client_v1 import (
    VictimMitigationClient,
)

from autonomous.agent_audit_log_v1 import (
    AgentAuditLogger,
)


# ============================================================================
# CONTROLLER
# ============================================================================

class AutonomousMitigationController:
    """
    Controlled SmartSOC endpoint mitigation controller.

    The controller accepts only structured defensive decisions.

    It does NOT execute arbitrary shell commands.
    """

    VERSION = "autonomous-mitigation-controller-v3"

    SUPPORTED_MITIGATIONS = frozenset({
        "PROCESS_TERMINATION",
    })

    def __init__(
        self,
        victim_host: str = "192.168.56.102",
        victim_port: int = 8765,
        victim_token: str = "SMARTSOC-LAB-V1-CHANGE-ME",
    ) -> None:

        # Fail closed if the centralized safety policy is invalid.
        validate_safety_invariants()

        self.reasoning_engine = AutonomousReasoningEngine()

        self.victim_client = VictimMitigationClient(
            victim_host=victim_host,
            victim_port=victim_port,
            token=victim_token,
        )

        self.audit_logger = AgentAuditLogger()

    # ========================================================================
    # CENTRAL EXECUTION POLICY
    # ========================================================================

    @classmethod
    def validate_execution_policy(
        cls,
        mitigation: str,
    ) -> None:
        """
        Validate the centralized SmartSOC execution policy.

        This method does not contact the victim endpoint.
        """

        normalized = str(
            mitigation
        ).strip().upper()

        # --------------------------------------------------------------
        # Controlled laboratory mode
        # --------------------------------------------------------------

        if not CONTROLLED_LAB_MODE:
            raise ValueError(
                "Controlled laboratory execution mode is disabled."
            )

        # --------------------------------------------------------------
        # Execution layer
        # --------------------------------------------------------------

        if not EXECUTION_LAYER_ENABLED:
            raise ValueError(
                "Execution layer is disabled."
            )

        # --------------------------------------------------------------
        # Automatic execution
        # --------------------------------------------------------------

        if not AUTOMATIC_EXECUTION:
            raise ValueError(
                "Automatic execution is disabled."
            )

        # --------------------------------------------------------------
        # Automatic containment
        # --------------------------------------------------------------
        #
        # Automatic containment is ENABLED for the controlled
        # SmartSOC laboratory.
        #
        # This does NOT permit unrestricted autonomous behavior.
        # The requested mitigation must still:
        #
        #   1. Be supported by this controller.
        #   2. Be explicitly allowlisted centrally.
        #   3. Pass target validation.
        #   4. Be accepted by the victim defensive agent.
        #
        # Currently allowlisted containment:
        #
        #   PROCESS_TERMINATION
        # --------------------------------------------------------------

        if not AUTOMATIC_CONTAINMENT:
            raise ValueError(
                "Automatic containment is disabled."
            )

        # --------------------------------------------------------------
        # Human authorization
        # --------------------------------------------------------------
        # Authorization is enforced immediately before the victim-client
        # call by execute_authorized_client_call(). It requires a valid,
        # unexpired, one-time approval bound to the exact event, action,
        # process name, and PID. Do not bypass that approval bridge.

        # --------------------------------------------------------------
        # Controller allowlist
        # --------------------------------------------------------------

        if normalized not in cls.SUPPORTED_MITIGATIONS:
            raise ValueError(
                f"Unsupported mitigation: {normalized!r}. "
                f"Supported actions: "
                f"{sorted(cls.SUPPORTED_MITIGATIONS)}"
            )

        # --------------------------------------------------------------
        # Central allowlist
        # --------------------------------------------------------------

        if not is_action_allowed(normalized):
            raise ValueError(
                f"Action {normalized!r} is not allowlisted "
                "by the centralized execution policy."
            )

    # ========================================================================
    # REASONING RESULT VALIDATION
    # ========================================================================

    @classmethod
    def validate_reasoning_result(
        cls,
        reasoning_result: Dict[str, Any],
    ) -> None:
        """
        Validate the structured reasoning result.

        No endpoint request is made if validation fails.
        """

        if not isinstance(
            reasoning_result,
            dict,
        ):
            raise ValueError(
                "Reasoning result must be a dictionary."
            )

        # --------------------------------------------------------------
        # Mitigation
        # --------------------------------------------------------------

        mitigation = str(
            reasoning_result.get(
                "mitigation",
                "",
            )
        ).strip().upper()

        cls.validate_execution_policy(
            mitigation
        )

        # --------------------------------------------------------------
        # PID
        # --------------------------------------------------------------

        process_id = reasoning_result.get(
            "process_id"
        )

        if isinstance(
            process_id,
            bool,
        ):
            raise ValueError(
                "process_id must be an integer."
            )

        try:
            pid = int(
                process_id
            )

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

        # --------------------------------------------------------------
        # Process name
        # --------------------------------------------------------------

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

        if not process_name.strip():
            raise ValueError(
                "process_name cannot be empty."
            )

    # ========================================================================
    # EXECUTE MITIGATION
    # ========================================================================

    def execute(
        self,
        reasoning_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a validated defensive mitigation.

        The victim endpoint is contacted only after:

            1. Central safety validation
            2. Controlled-lab validation
            3. Action allowlist validation
            4. Mitigation validation
            5. PID validation
            6. Process-name validation
        """

        # --------------------------------------------------------------
        # 1. Central policy validation
        # --------------------------------------------------------------

        try:

            validate_safety_invariants()

            mitigation = str(
                reasoning_result.get(
                    "mitigation",
                    "",
                )
            ).strip().upper()

            self.validate_execution_policy(
                mitigation
            )

        except Exception as exc:

            return {
                "success": False,
                "verified": False,
                "status": "MITIGATION_BLOCKED",
                "controller": self.VERSION,
                "victim_client_invoked": False,
                "error": str(exc),
            }

        # --------------------------------------------------------------
        # 2. Structured reasoning validation
        # --------------------------------------------------------------

        try:

            self.validate_reasoning_result(
                reasoning_result
            )

        except ValueError as exc:

            return {
                "success": False,
                "verified": False,
                "status": "MITIGATION_BLOCKED",
                "controller": self.VERSION,
                "victim_client_invoked": False,
                "error": str(exc),
            }

        # --------------------------------------------------------------
        # 3. Send structured request to victim
        # --------------------------------------------------------------

        try:

            result = execute_authorized_client_call(self.victim_client, reasoning_result)

        except Exception as exc:

            result = {
                "success": False,
                "verified": False,
                "status": "MITIGATION_ERROR",
                "error": str(exc),
            }

        # --------------------------------------------------------------
        # 4. Audit result
        # --------------------------------------------------------------

        try:

            audit_record = self.audit_logger.log_execution(
                execution_result=result,
                agent_reason=reasoning_result.get(
                    "reason",
                    "Autonomous defensive mitigation.",
                ),
            )

        except Exception as exc:

            audit_record = {
                "audit_error": str(exc),
            }

        # --------------------------------------------------------------
        # 5. Return complete structured result
        # --------------------------------------------------------------

        return {
            "victim_client_invoked": bool(
                result.get("victim_client_invoked", False)
            ),
            "success": bool(
                result.get("success")
            ),

            "verified": bool(
                result.get("verified")
            ),

            "status": result.get(
                "status",
                "UNKNOWN",
            ),

            "controller": self.VERSION,

            "policy": {
                "controlled_lab_mode": (
                    CONTROLLED_LAB_MODE
                ),
                "execution_layer_enabled": (
                    EXECUTION_LAYER_ENABLED
                ),
                "automatic_execution": (
                    AUTOMATIC_EXECUTION
                ),
                "automatic_containment": (
                    AUTOMATIC_CONTAINMENT
                ),
                "human_authorization_required": (
                    HUMAN_AUTHORIZATION_REQUIRED
                ),
                "action_allowlisted": (
                    is_action_allowed(
                        str(
                            reasoning_result.get(
                                "mitigation",
                                "",
                            )
                        )
                    )
                ),
            },

            "victim": {
                "host": (
                    self.victim_client.victim_host
                ),
                "port": (
                    self.victim_client.victim_port
                ),
                "endpoint": (
                    self.victim_client.endpoint
                ),
            },

            "reasoning": reasoning_result,

            "mitigation_result": result,

            "audit": audit_record,
        }

    # ========================================================================
    # REASONING -> MITIGATION
    # ========================================================================

    def process_event(
        self,
        inference_result: Dict[str, Any],
        wazuh_event: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run autonomous reasoning and, when justified,
        perform the controlled defensive mitigation.
        """

        reasoning_result = (
            self.reasoning_engine.reason(
                inference_result=inference_result,
                wazuh_event=wazuh_event,
            )
        )

        mitigation = str(
            reasoning_result.get(
                "mitigation",
                "NONE",
            )
        ).strip().upper()

        # --------------------------------------------------------------
        # No mitigation required
        # --------------------------------------------------------------

        if mitigation == "NONE":

            return {
                "success": True,
                "verified": False,
                "status": "NO_MITIGATION_REQUIRED",
                "controller": self.VERSION,
                "reasoning": reasoning_result,
                "mitigation_result": None,
                "audit": None,
            }

        # --------------------------------------------------------------
        # Unsupported mitigation
        # --------------------------------------------------------------

        if mitigation not in self.SUPPORTED_MITIGATIONS:

            return {
                "success": False,
                "verified": False,
                "status": "MITIGATION_BLOCKED",
                "controller": self.VERSION,
                "reasoning": reasoning_result,
                "error": (
                    f"Mitigation {mitigation!r} is not "
                    "supported by this controller."
                ),
            }

        # --------------------------------------------------------------
        # Controlled mitigation
        # --------------------------------------------------------------

        return self.execute(
            reasoning_result
        )


# ============================================================================
# SELF TEST
# ============================================================================

def self_test() -> None:
    """
    Safe controller self-test.

    The invalid PID guarantees that no endpoint process
    can be terminated.
    """

    print("=" * 72)

    print(
        "SmartSOC AUTONOMOUS MITIGATION CONTROLLER v3"
    )

    print("=" * 72)

    # --------------------------------------------------------------
    # Initialize controller
    # --------------------------------------------------------------

    controller = AutonomousMitigationController()

    print()

    print(
        "Controller version :",
        controller.VERSION,
    )

    print(
        "Victim endpoint    :",
        controller.victim_client.endpoint,
    )

    print(
        "Supported actions  :",
        sorted(
            controller.SUPPORTED_MITIGATIONS
        ),
    )

    # --------------------------------------------------------------
    # Central policy
    # --------------------------------------------------------------

    print()

    print(
        "[CENTRAL POLICY]"
    )

    print(
        "Controlled lab mode       :",
        CONTROLLED_LAB_MODE,
    )

    print(
        "Execution layer enabled   :",
        EXECUTION_LAYER_ENABLED,
    )

    print(
        "Automatic execution       :",
        AUTOMATIC_EXECUTION,
    )

    print(
        "Automatic containment     :",
        AUTOMATIC_CONTAINMENT,
    )

    print(
        "Human authorization       :",
        HUMAN_AUTHORIZATION_REQUIRED,
    )

    print(
        "PROCESS_TERMINATION       :",
        is_action_allowed(
            "PROCESS_TERMINATION"
        ),
    )

    print(
        "ARBITRARY_COMMAND         :",
        is_action_allowed(
            "ARBITRARY_COMMAND"
        ),
    )

    print(
        "Safety invariants         : VALID"
    )

    # --------------------------------------------------------------
    # Invalid PID test
    # --------------------------------------------------------------

    test_reasoning = {
        "mitigation": "PROCESS_TERMINATION",
        "process_id": None,
        "process_name": "notepad.exe",
        "technique": "TEST",
        "confidence": 1.0,
        "objective": "self_test",
        "wazuh_rule_id": "LAB-SELF-TEST",
        "wazuh_rule_level": 1,
        "reason": (
            "Controller self-test. "
            "No endpoint process should be terminated."
        ),
    }

    print()

    print(
        "[SELF-TEST]"
    )

    print(
        "Purpose: validate controller blocking "
        "of an invalid PID."
    )

    result = controller.execute(
        test_reasoning
    )

    print()

    print(
        "Status:",
        result.get("status"),
    )

    print(
        "Success:",
        result.get("success"),
    )

    print(
        "Verified:",
        result.get("verified"),
    )

    print(
        "Error:",
        result.get("error"),
    )

    print()

    print(
        "No endpoint process was terminated "
        "during this self-test."
    )

    print("=" * 72)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    self_test()



