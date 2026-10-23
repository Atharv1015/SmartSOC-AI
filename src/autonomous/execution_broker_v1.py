"""
SmartSOC Autonomous Execution Broker v1

Purpose:
    Controlled execution interface for the autonomous-agent prototype.

Important:
    This module is intentionally isolated from the production
    SmartSOC execution-policy guard during the prototype phase.

    It executes only an explicitly supplied command and records:
      - command
      - start/end time
      - return code
      - stdout
      - stderr
      - success/failure

    The first test should use a harmless diagnostic command.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, List


class ExecutionBroker:
    """Execute a command and return a structured execution result."""

    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds

    def execute(self, command: List[str]) -> Dict[str, Any]:
        """
        Execute a command on the local Windows endpoint.

        Parameters
        ----------
        command:
            Command represented as a list of arguments.

        Returns
        -------
        Dict[str, Any]
            Structured execution result.
        """

        if not isinstance(command, list) or not command:
            raise ValueError("command must be a non-empty list")

        if not all(isinstance(arg, str) for arg in command):
            raise TypeError("all command arguments must be strings")

        started_at = datetime.now(timezone.utc).isoformat()

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                shell=False,
            )

            finished_at = datetime.now(timezone.utc).isoformat()

            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "command": command,
                "started_at": started_at,
                "finished_at": finished_at,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": False,
            }

        except subprocess.TimeoutExpired as exc:
            finished_at = datetime.now(timezone.utc).isoformat()

            return {
                "success": False,
                "return_code": None,
                "command": command,
                "started_at": started_at,
                "finished_at": finished_at,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "timed_out": True,
            }

        except Exception as exc:
            finished_at = datetime.now(timezone.utc).isoformat()

            return {
                "success": False,
                "return_code": None,
                "command": command,
                "started_at": started_at,
                "finished_at": finished_at,
                "stdout": "",
                "stderr": str(exc),
                "timed_out": False,
            }


def main() -> None:
    print("=" * 72)
    print("SmartSOC AUTONOMOUS EXECUTION BROKER v1")
    print("=" * 72)

    broker = ExecutionBroker(timeout_seconds=30)

    # Harmless diagnostic command for the first prototype test.
    command = [
        "cmd.exe",
        "/c",
        "whoami",
    ]

    print()
    print("[TEST]")
    print("Command:", command)
    print()

    result = broker.execute(command)

    print("[RESULT]")
    print("Success     :", result["success"])
    print("Return code :", result["return_code"])
    print("Timed out   :", result["timed_out"])
    print("Started     :", result["started_at"])
    print("Finished    :", result["finished_at"])

    print()
    print("[STDOUT]")
    print(result["stdout"])

    if result["stderr"]:
        print("[STDERR]")
        print(result["stderr"])

    print("=" * 72)


if __name__ == "__main__":
    main()