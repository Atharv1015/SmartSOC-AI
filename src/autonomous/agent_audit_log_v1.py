"""
SmartSOC Autonomous Agent Audit Logger v1

Records every autonomous-agent execution attempt.

Audit records are written as JSON Lines so that each execution
is independently traceable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIT_DIR = PROJECT_ROOT / "reports" / "autonomous"

AUDIT_FILE = AUDIT_DIR / "agent_execution_audit_v1.jsonl"


class AgentAuditLogger:
    """Write structured autonomous execution audit records."""

    def __init__(self, audit_file: Path = AUDIT_FILE):
        self.audit_file = Path(audit_file)
        self.audit_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def log_execution(
        self,
        execution_result: Dict[str, Any],
        agent_reason: str = "",
    ) -> Dict[str, Any]:
        """
        Record one execution attempt.

        Returns the complete audit record.
        """

        record = {
            "audit_timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "event_type": "AUTONOMOUS_COMMAND_EXECUTION",

            "agent_reason": agent_reason,

            "execution": execution_result,
        }

        with self.audit_file.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        return record


def main() -> None:
    print("=" * 72)
    print("SmartSOC AUTONOMOUS AGENT AUDIT LOGGER v1")
    print("=" * 72)

    logger = AgentAuditLogger()

    test_result = {
        "success": True,
        "return_code": 0,
        "command": [
            "cmd.exe",
            "/c",
            "whoami",
        ],
        "started_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "finished_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "stdout": "atharv\\atharv",
        "stderr": "",
        "timed_out": False,
    }

    record = logger.log_execution(
        execution_result=test_result,
        agent_reason="Initial autonomous execution broker validation",
    )

    print()
    print("[PASS] Audit record written.")
    print()
    print("Audit file:")
    print(logger.audit_file)
    print()
    print("Event type:")
    print(record["event_type"])
    print()
    print("Command:")
    print(record["execution"]["command"])

    print("=" * 72)


if __name__ == "__main__":
    main()