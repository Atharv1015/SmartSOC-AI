"""
SmartSOC Autonomous Agent v2

Integrates:

    Command Generator
            ↓
    Execution Broker
            ↓
    Windows Endpoint
            ↓
    Audit Logger

This version performs a single controlled autonomous
response cycle for testing.
"""

from __future__ import annotations

from command_generator_v1 import CommandGenerator
from execution_broker_v1 import ExecutionBroker
from agent_audit_log_v1 import AgentAuditLogger


class AutonomousAgent:
    """SmartSOC autonomous-response controller."""

    VERSION = "autonomous-agent-v2"

    def __init__(self):
        self.command_generator = CommandGenerator()

        self.execution_broker = ExecutionBroker(
            timeout_seconds=30
        )

        self.audit_logger = AgentAuditLogger()

    def run_scenario(
        self,
        scenario: str,
        context: dict | None = None,
    ):
        """
        Execute one complete autonomous-response cycle.
        """

        print()
        print("[1] Generating response proposal...")

        proposal = self.command_generator.generate(
            scenario=scenario,
            context=context,
        )

        print(
            "    Status    :",
            proposal["status"],
        )

        print(
            "    Objective :",
            proposal["objective"],
        )

        print(
            "    Command   :",
            proposal["command"],
        )

        if proposal["status"] != "PROPOSED":
            print()
            print("[STOP] No executable proposal generated.")

            return {
                "status": "NO_EXECUTION",
                "proposal": proposal,
                "execution": None,
            }

        command = proposal["command"]

        print()
        print("[2] Sending proposal to execution broker...")

        execution_result = self.execution_broker.execute(
            command
        )

        print(
            "    Success     :",
            execution_result["success"],
        )

        print(
            "    Return code :",
            execution_result["return_code"],
        )

        print(
            "    Timed out   :",
            execution_result["timed_out"],
        )

        print()
        print("[3] Recording execution audit...")

        audit_record = self.audit_logger.log_execution(
            execution_result=execution_result,
            agent_reason=proposal["reason"],
        )

        print(
            "    Audit event :",
            audit_record["event_type"],
        )

        return {
            "status": "EXECUTED",
            "proposal": proposal,
            "execution": execution_result,
            "audit": audit_record,
        }


def main() -> None:
    print("=" * 72)
    print("SmartSOC AUTONOMOUS AGENT v2")
    print("=" * 72)

    agent = AutonomousAgent()

    context = {
        "source": "autonomous-agent-integration-test",
        "technique": "T1087",
        "target": "SmartSOC-Windows-Victim",
    }

    result = agent.run_scenario(
        scenario="identify_execution_context",
        context=context,
    )

    print()
    print("[FINAL RESULT]")
    print("Status:", result["status"])

    if result["execution"]:
        print(
            "Execution success:",
            result["execution"]["success"],
        )

        print(
            "Command:",
            result["execution"]["command"],
        )

        print()
        print("[STDOUT]")
        print(
            result["execution"]["stdout"]
        )

    print("=" * 72)


if __name__ == "__main__":
    main()