"""
SmartSOC Remote Controller Integration Test

Controlled lab test:
    Reasoning result
        ↓
    Autonomous Mitigation Controller
        ↓
    Victim Mitigation Client
        ↓
    Windows Victim Agent
        ↓
    PROCESS_TERMINATION
        ↓
    Verification

This test requires an explicitly supplied lab PID.
"""

from autonomous_mitigation_controller_v1 import (
    AutonomousMitigationController,
)


def main() -> None:

    print("=" * 72)
    print(
        "SmartSOC REMOTE CONTROLLER INTEGRATION TEST"
    )
    print("=" * 72)

    # --------------------------------------------------------------
    # CONTROLLED LAB TARGET
    # --------------------------------------------------------------

    pid_text = input(
        "\nEnter the PID of the LAB notepad.exe process: "
    ).strip()

    try:
        pid = int(pid_text)
    except ValueError:
        print("[ERROR] PID must be an integer.")
        return

    if pid <= 0:
        print("[ERROR] PID must be greater than zero.")
        return

    reasoning_result = {
        "status": "MITIGATION_SELECTED",

        "reasoning_engine":
            "autonomous-reasoning-v2",

        "technique":
            "T1059.001",

        "confidence":
            0.97,

        "threshold":
            0.5,

        "objective":
            "contain_suspicious_process",

        "mitigation":
            "PROCESS_TERMINATION",

        "process_id":
            pid,

        "process_name":
            "notepad.exe",

        "wazuh_rule_id":
            100001,

        "wazuh_rule_level":
            12,

        "reason":
            (
                "Controlled lab integration test "
                "for remote defensive process "
                "termination."
            ),
    }

    print()
    print("[TEST DECISION]")
    print(
        "    Technique:",
        reasoning_result["technique"],
    )

    print(
        "    Confidence:",
        reasoning_result["confidence"],
    )

    print(
        "    Objective:",
        reasoning_result["objective"],
    )

    print(
        "    Mitigation:",
        reasoning_result["mitigation"],
    )

    print(
        "    PID:",
        reasoning_result["process_id"],
    )

    print(
        "    Process:",
        reasoning_result["process_name"],
    )

    print()
    print(
        "The controller will send a structured "
        "PROCESS_TERMINATION request to the "
        "Windows victim agent."
    )

    confirmation = input(
        "\nType YES to execute this controlled lab test: "
    )

    if confirmation.strip().upper() != "YES":
        print("Test cancelled.")
        return

    controller = (
        AutonomousMitigationController()
    )

    result = controller.process_decision(
        reasoning_result
    )

    print()
    print("=" * 72)
    print("[FINAL RESULT]")
    print("=" * 72)

    print(
        "Status:",
        result.get("status"),
    )

    mitigation = result.get(
        "mitigation",
        {},
    )

    print(
        "Success:",
        mitigation.get("success"),
    )

    print(
        "Verified:",
        mitigation.get("verified"),
    )

    print("=" * 72)


if __name__ == "__main__":
    main()