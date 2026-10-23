"""
SmartSOC Autonomous Mitigation Pipeline Test v3

Controlled containment integration test.

Flow:

    Controlled Wazuh-shaped event
        |
        v
    SmartSOC inference result
        |
        v
    AutonomousReasoningEngine
        |
        v
    Mitigation decision
        |
        v
    Target validation
        |
        v
    AutonomousMitigationController
        |
        v
    VictimMitigationClient
        |
        v
    Windows Defensive Agent v3
        |
        v
    Live PID validation
        |
        v
    Process identity validation
        |
        v
    PROCESS_TERMINATION
        |
        v
    Endpoint verification
        |
        v
    Audit/result

IMPORTANT:

    This file is a controlled integration test.

    It does NOT claim to be a live Wazuh stream.

    A real production event should supply the actual PID and
    process name from Wazuh/Windows telemetry.

Usage:

    python test_autonomous_mitigation_pipeline_v1.py \
        --pid 1234 \
        --process-name notepad.exe
"""

from __future__ import annotations

import argparse
import pprint
import sys
from datetime import datetime, timezone
from pathlib import Path


# ----------------------------------------------------------------------
# PROJECT PATH
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


# ----------------------------------------------------------------------
# SMARTSOC COMPONENTS
# ----------------------------------------------------------------------

from autonomous.autonomous_reasoning_v1 import (
    AutonomousReasoningEngine,
)

from autonomous.autonomous_mitigation_controller_v1 import (
    AutonomousMitigationController,
)


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

VICTIM_HOST = "192.168.56.102"
VICTIM_PORT = 8765
VICTIM_TOKEN = "SMARTSOC-LAB-V1-CHANGE-ME"


# ----------------------------------------------------------------------
# SMARTSOC INFERENCE RESULT
# ----------------------------------------------------------------------

INFERENCE_RESULT = {
    "model_version": "windows-apt-v4.1",

    "predictions": [
        {
            "technique": "T1059.001",
            "confidence": 0.97,
            "threshold": 0.50,
        }
    ],
}


# ----------------------------------------------------------------------
# PROCESS NAME NORMALIZATION
# ----------------------------------------------------------------------

def normalize_process_name(
    process_name: str,
) -> str:
    """
    Normalize a Windows process path/name.

    Example:

        C:\\Windows\\System32\\notepad.exe
            ->
        notepad.exe
    """

    if not isinstance(
        process_name,
        str,
    ):
        return ""

    normalized = process_name.strip()

    if not normalized:
        return ""

    normalized = normalized.replace(
        "/",
        "\\",
    )

    return normalized.rsplit(
        "\\",
        1,
    )[-1].lower()


# ----------------------------------------------------------------------
# ARGUMENTS
# ----------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "SmartSOC controlled autonomous "
            "mitigation pipeline test."
        )
    )

    parser.add_argument(
        "--pid",
        required=True,
        type=int,
        help=(
            "Live PID of the controlled target "
            "process on the victim."
        ),
    )

    parser.add_argument(
        "--process-name",
        required=True,
        type=str,
        help=(
            "Expected live process name, for example "
            "notepad.exe."
        ),
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# TARGET VALIDATION
# ----------------------------------------------------------------------

def validate_cli_target(
    pid: int,
    process_name: str,
) -> tuple[int, str]:

    if pid <= 0:
        raise ValueError(
            "PID must be greater than zero."
        )

    normalized = normalize_process_name(
        process_name
    )

    if not normalized:
        raise ValueError(
            "Process name cannot be empty."
        )

    return pid, normalized


# ----------------------------------------------------------------------
# WAZUH EVENT CONSTRUCTION
# ----------------------------------------------------------------------

def build_wazuh_event(
    pid: int,
    process_name: str,
) -> dict:

    normalized_process_name = (
        normalize_process_name(
            process_name
        )
    )

    return {
        "timestamp":
            datetime.now(
                timezone.utc
            ).strftime(
                "%Y-%m-%dT%H:%M:%S.%f+0000"
            ),

        "agent": {
            "id": "001",
            "name": "SmartSOC-Windows-Victim",
            "ip": "192.168.50.20",
        },

        "rule": {
            "id": "100001",
            "level": 10,
            "description":
                "Controlled SmartSOC autonomous "
                "mitigation integration test",
        },

        "data": {

            "win": {

                "eventdata": {

                    "image":
                        f"C:\\Windows\\System32\\"
                        f"{normalized_process_name}",

                    "parentImage":
                        "C:\\Windows\\System32\\"
                        "WindowsPowerShell\\v1.0\\"
                        "powershell.exe",

                    "commandLine":
                        "powershell.exe "
                        "-EncodedCommand "
                        "CONTROLLED_SMARTSOC_LAB_TEST",

                    "processId":
                        str(pid),

                    "user":
                        "LAB\\SmartSOC",
                }
            }
        },

        "mitre": {

            "id": [
                "T1059.001"
            ],

            "technique": [
                "PowerShell"
            ],

            "tactic": [
                "Execution"
            ],
        },
    }


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main() -> int:

    args = parse_arguments()

    try:

        target_pid, target_process = (
            validate_cli_target(
                args.pid,
                args.process_name,
            )
        )

    except ValueError as exc:

        print()
        print(
            "[BLOCKED] Invalid target:"
        )
        print(
            str(exc)
        )
        return 1

    wazuh_event = build_wazuh_event(
        target_pid,
        target_process,
    )

    print("=" * 78)

    print(
        "SMARTSOC AUTONOMOUS MITIGATION "
        "PIPELINE TEST v3"
    )

    print("=" * 78)

    print()

    print(
        "[TEST TYPE]"
    )

    print(
        "Controlled Wazuh-shaped event"
    )

    print(
        "Live victim endpoint enforcement"
    )

    print()

    print(
        "[TARGET]"
    )

    print(
        "Victim host :",
        VICTIM_HOST,
    )

    print(
        "Victim port :",
        VICTIM_PORT,
    )

    print(
        "Process PID :",
        target_pid,
    )

    print(
        "Process     :",
        target_process,
    )

    # --------------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------------

    print()

    print(
        "[1] SmartSOC inference result"
    )

    pprint.pprint(
        INFERENCE_RESULT,
        sort_dicts=False,
    )

    # --------------------------------------------------------------
    # WAZUH
    # --------------------------------------------------------------

    print()

    print(
        "[2] Wazuh-shaped event"
    )

    print(
        "Rule        :",
        wazuh_event["rule"]["id"],
    )

    print(
        "Rule level  :",
        wazuh_event["rule"]["level"],
    )

    print(
        "Technique   :",
        wazuh_event["mitre"]["id"][0],
    )

    print(
        "Process ID  :",
        target_pid,
    )

    print(
        "Process     :",
        target_process,
    )

    # --------------------------------------------------------------
    # REASONING
    # --------------------------------------------------------------

    print()

    print(
        "[3] Running AutonomousReasoningEngine..."
    )

    reasoning_engine = (
        AutonomousReasoningEngine()
    )

    reasoning_result = (
        reasoning_engine.reason(
            inference_result=INFERENCE_RESULT,
            wazuh_event=wazuh_event,
        )
    )

    print()

    print(
        "[REASONING RESULT]"
    )

    pprint.pprint(
        reasoning_result,
        sort_dicts=False,
    )

    # --------------------------------------------------------------
    # MITIGATION CHECK
    # --------------------------------------------------------------

    mitigation = str(
        reasoning_result.get(
            "mitigation",
            "NONE",
        )
    ).strip().upper()

    if mitigation != "PROCESS_TERMINATION":

        print()

        print(
            "[BLOCKED] Reasoning engine did not "
            "select PROCESS_TERMINATION."
        )

        print(
            "Selected mitigation:",
            mitigation,
        )

        print()

        print(
            "No endpoint action was executed."
        )

        return 1

    # --------------------------------------------------------------
    # TARGET EXTRACTION
    # --------------------------------------------------------------

    process_id = reasoning_result.get(
        "process_id"
    )

    process_name = reasoning_result.get(
        "process_name"
    )

    normalized_process_name = (
        normalize_process_name(
            process_name
        )
    )

    print()

    print(
        "[4] Mitigation decision"
    )

    print(
        "Mitigation          :",
        mitigation,
    )

    print(
        "PID                 :",
        process_id,
    )

    print(
        "Process from event  :",
        process_name,
    )

    print(
        "Normalized process  :",
        normalized_process_name,
    )

    # --------------------------------------------------------------
    # PID VALIDATION
    # --------------------------------------------------------------

    try:

        parsed_pid = int(
            process_id
        )

    except (
        TypeError,
        ValueError,
    ):

        print()

        print(
            "[BLOCKED] Invalid process ID."
        )

        return 1

    if parsed_pid != target_pid:

        print()

        print(
            "[BLOCKED] Reasoning engine PID "
            "does not match event PID."
        )

        print(
            "Expected:",
            target_pid,
        )

        print(
            "Received:",
            parsed_pid,
        )

        return 1

    # --------------------------------------------------------------
    # PROCESS NAME VALIDATION
    # --------------------------------------------------------------

    if normalized_process_name != target_process:

        print()

        print(
            "[BLOCKED] Reasoning engine process "
            "does not match event process."
        )

        print(
            "Expected:",
            target_process,
        )

        print(
            "Received:",
            normalized_process_name,
        )

        return 1

    print()

    print(
        "[PASS] Target validation succeeded."
    )

    print(
        "The PID and process identity propagated "
        "correctly through SmartSOC reasoning."
    )

    # --------------------------------------------------------------
    # CONTROLLER
    # --------------------------------------------------------------

    print()

    print(
        "[5] Running "
        "AutonomousMitigationController..."
    )

    controller = (
        AutonomousMitigationController(
            victim_host=VICTIM_HOST,
            victim_port=VICTIM_PORT,
            victim_token=VICTIM_TOKEN,
        )
    )

    result = controller.execute(
        reasoning_result
    )

    # --------------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------------

    print()

    print(
        "[6] FINAL RESULT"
    )

    print("-" * 78)

    print(
        "Status   :",
        result.get(
            "status"
        ),
    )

    print(
        "Success  :",
        result.get(
            "success"
        ),
    )

    print(
        "Verified :",
        result.get(
            "verified"
        ),
    )

    print()

    print(
        "[COMPLETE RESULT]"
    )

    pprint.pprint(
        result,
        sort_dicts=False,
        width=120,
    )

    print()

    print("=" * 78)

    if (
        result.get(
            "success"
        ) is True
        and
        result.get(
            "verified"
        ) is True
    ):

        print(
            "AUTONOMOUS MITIGATION TEST: PASSED"
        )

        print()

        print(
            "Reasoning -> Controller -> "
            "Victim Agent -> Live PID validation "
            "-> Termination -> Verification succeeded."
        )

        print("=" * 78)

        return 0

    print(
        "AUTONOMOUS MITIGATION TEST: FAILED"
    )

    print("=" * 78)

    return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )