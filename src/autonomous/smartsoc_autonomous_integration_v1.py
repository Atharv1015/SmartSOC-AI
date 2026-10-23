"""
SmartSOC Autonomous Integration v2

Connects:

    Real Wazuh Event
          ↓
    SmartSOC v4.1 Inference
          ↓
    Autonomous Reasoning Engine
          ↓
    Command Generator
          ↓
    Execution Broker
          ↓
    Audit Logger

Current scope:

    - Read-only diagnostic actions
    - Human-auditable execution
    - No endpoint modification
    - No containment
    - No unrestricted autonomous actions
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


# ------------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ------------------------------------------------------------------
# SMARTSOC INFERENCE
# ------------------------------------------------------------------

from production_inference_windows_apt_v4_1 import (
    SmartSOCInference,
)


# ------------------------------------------------------------------
# AUTONOMOUS COMPONENTS
# ------------------------------------------------------------------

from autonomous_reasoning_v1 import (
    AutonomousReasoningEngine,
)

from command_generator_v1 import (
    CommandGenerator,
)

from execution_broker_v1 import (
    ExecutionBroker,
)

from agent_audit_log_v1 import (
    AgentAuditLogger,
)


class SmartSOCAutonomousIntegration:
    """
    Integrates SmartSOC detection with the autonomous
    reasoning and controlled diagnostic execution pipeline.
    """

    VERSION = "smartsoc-autonomous-integration-v2"

    def __init__(self):

        self.inference = SmartSOCInference()

        self.reasoning_engine = (
            AutonomousReasoningEngine()
        )

        self.command_generator = (
            CommandGenerator()
        )

        self.execution_broker = (
            ExecutionBroker(
                timeout_seconds=30
            )
        )

        self.audit_logger = (
            AgentAuditLogger()
        )


    # ------------------------------------------------------------------
    # LOAD WAZUH EVENT
    # ------------------------------------------------------------------

    @staticmethod
    def load_wazuh_event(
        input_path: Path,
    ) -> Dict[str, Any]:

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            data = json.load(handle)

        return data


    # ------------------------------------------------------------------
    # COMPLETE PIPELINE
    # ------------------------------------------------------------------

    def process_event(
        self,
        wazuh_event: Dict[str, Any],
    ) -> Dict[str, Any]:

        # --------------------------------------------------------------
        # STEP 1 — SMARTSOC INFERENCE
        # --------------------------------------------------------------

        print()
        print(
            "[1] Running SmartSOC v4.1 inference..."
        )

        inference_result = (
            self.inference.infer(
                wazuh_event
            )
        )

        print()
        print(
            "[2] SmartSOC detection result"
        )

        print(
            "    Model:",
            inference_result.get(
                "model_version"
            ),
        )

        print(
            "    Predictions:",
            inference_result.get(
                "predictions"
            ),
        )


        # --------------------------------------------------------------
        # STEP 2 — AUTONOMOUS REASONING
        # --------------------------------------------------------------

        print()
        print(
            "[3] Running autonomous reasoning..."
        )

        reasoning_result = (
            self.reasoning_engine.reason(
                inference_result=inference_result,
                wazuh_event=wazuh_event,
            )
        )

        print(
            "    Status:",
            reasoning_result.get(
                "status"
            ),
        )

        print(
            "    Technique:",
            reasoning_result.get(
                "technique"
            ),
        )

        print(
            "    Confidence:",
            reasoning_result.get(
                "confidence"
            ),
        )

        print(
            "    Objective:",
            reasoning_result.get(
                "objective"
            ),
        )

        print(
            "    Reason:",
            reasoning_result.get(
                "reason"
            ),
        )


        # --------------------------------------------------------------
        # NO OBJECTIVE
        # --------------------------------------------------------------

        objective = reasoning_result.get(
            "objective"
        )

        if not objective:

            print()
            print(
                "    No autonomous diagnostic objective selected."
            )

            return {
                "status": "NO_RESPONSE_OBJECTIVE",
                "inference": inference_result,
                "reasoning": reasoning_result,
            }


        # --------------------------------------------------------------
        # STEP 3 — COMMAND GENERATION
        # --------------------------------------------------------------

        print()
        print(
            "[4] Generating autonomous command proposal..."
        )

        proposal = (
            self.command_generator.generate(
                scenario=objective,
                context={
                    "source": "wazuh",
                    "integration": self.VERSION,
                    "inference": inference_result,
                    "reasoning": reasoning_result,
                },
            )
        )

        print(
            "    Status:",
            proposal.get(
                "status"
            ),
        )

        print(
            "    Command:",
            proposal.get(
                "command"
            ),
        )

        if proposal.get(
            "status"
        ) != "PROPOSED":

            return {
                "status": "NO_COMMAND_PROPOSAL",
                "inference": inference_result,
                "reasoning": reasoning_result,
                "proposal": proposal,
            }


        # --------------------------------------------------------------
        # STEP 4 — CONTROLLED EXECUTION
        # --------------------------------------------------------------

        print()
        print(
            "[5] Executing controlled diagnostic response..."
        )

        execution_result = (
            self.execution_broker.execute(
                proposal["command"]
            )
        )

        print(
            "    Success:",
            execution_result[
                "success"
            ],
        )

        print(
            "    Return code:",
            execution_result[
                "return_code"
            ],
        )


        # --------------------------------------------------------------
        # STEP 5 — AUDIT
        # --------------------------------------------------------------

        print()
        print(
            "[6] Writing autonomous audit record..."
        )

        audit_record = (
            self.audit_logger.log_execution(
                execution_result=execution_result,
                agent_reason=proposal[
                    "reason"
                ],
            )
        )

        print(
            "    Audit event:",
            audit_record[
                "event_type"
            ],
        )


        # --------------------------------------------------------------
        # FINAL RESULT
        # --------------------------------------------------------------

        return {
            "status": "EXECUTED",
            "inference": inference_result,
            "reasoning": reasoning_result,
            "proposal": proposal,
            "execution": execution_result,
            "audit": audit_record,
        }


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main() -> None:

    print(
        "=" * 72
    )

    print(
        "SmartSOC REAL WAZUH → AUTONOMOUS INTEGRATION v2"
    )

    print(
        "=" * 72
    )


    input_path = (
        PROJECT_ROOT
        / "wazuh_92031_latest.json"
    )

    print()
    print(
        "[INPUT]"
    )

    print(
        "Wazuh event:",
        input_path,
    )


    if not input_path.exists():

        raise FileNotFoundError(
            f"Wazuh event not found: {input_path}"
        )


    integration = (
        SmartSOCAutonomousIntegration()
    )


    wazuh_event = (
        integration.load_wazuh_event(
            input_path
        )
    )


    result = (
        integration.process_event(
            wazuh_event
        )
    )


    # --------------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------------

    print()
    print(
        "[FINAL RESULT]"
    )

    print(
        "Status:",
        result[
            "status"
        ],
    )


    if result.get(
        "reasoning"
    ):

        print()
        print(
            "[REASONING]"
        )

        print(
            "Technique:",
            result[
                "reasoning"
            ].get(
                "technique"
            ),
        )

        print(
            "Objective:",
            result[
                "reasoning"
            ].get(
                "objective"
            ),
        )


    if result.get(
        "execution"
    ):

        print()
        print(
            "[EXECUTION]"
        )

        print(
            "Success:",
            result[
                "execution"
            ][
                "success"
            ],
        )

        print(
            "Command:",
            result[
                "execution"
            ][
                "command"
            ],
        )

        print()
        print(
            "[STDOUT]"
        )

        print(
            result[
                "execution"
            ][
                "stdout"
            ]
        )


    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()