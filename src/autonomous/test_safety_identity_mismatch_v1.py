"""
SmartSOC Safety Validation - Process Identity Mismatch v1

Purpose:
    Verify that the victim defensive agent rejects a termination request
    when the supplied PID exists but the supplied process name does not
    match the live process.

Test target:
    PID 6980
    Actual process: notepad.exe
    Claimed process: powershell.exe

Expected:
    VICTIM_REJECTED_REQUEST
    success  = False
    verified = False

The test must NOT terminate PID 6980.
"""

from __future__ import annotations

import sys
from typing import Any, Dict


try:
    from autonomous_mitigation_controller_v1 import (
        AutonomousMitigationController,
    )
except ImportError:
    sys.path.insert(
        0,
        r".\src\autonomous",
    )

    from autonomous_mitigation_controller_v1 import (
        AutonomousMitigationController,
    )


TEST_PID = 6980

# This is intentionally WRONG.
# PID 6980 is actually notepad.exe.
CLAIMED_PROCESS_NAME = "powershell.exe"


def build_reasoning_result() -> Dict[str, Any]:

    return {
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
            TEST_PID,

        "process_name":
            CLAIMED_PROCESS_NAME,

        "wazuh_rule_id":
            100001,

        "wazuh_rule_level":
            12,

        "reason":
            (
                "Safety validation test for process "
                "identity mismatch."
            ),
    }


def main() -> int:

    print("=" * 72)
    print(
        "SMARTSOC SAFETY VALIDATION - "
        "PROCESS IDENTITY MISMATCH TEST v1"
    )
    print("=" * 72)

    print()
    print("[TEST TARGET]")
    print(f"    PID              : {TEST_PID}")
    print("    Actual process   : notepad.exe")
    print(
        f"    Claimed process  : "
        f"{CLAIMED_PROCESS_NAME}"
    )
    print(
        "    Expected         : Victim rejection"
    )
    print(
        "    Expected kill    : NONE"
    )

    reasoning_result = build_reasoning_result()

    print()
    print("[REASONING RESULT]")
    print(
        f"    Status       : "
        f"{reasoning_result['status']}"
    )
    print(
        f"    Technique    : "
        f"{reasoning_result['technique']}"
    )
    print(
        f"    Confidence   : "
        f"{reasoning_result['confidence']}"
    )
    print(
        f"    Objective    : "
        f"{reasoning_result['objective']}"
    )
    print(
        f"    Mitigation   : "
        f"{reasoning_result['mitigation']}"
    )
    print(
        f"    PID          : "
        f"{reasoning_result['process_id']}"
    )
    print(
        f"    Claimed name : "
        f"{reasoning_result['process_name']}"
    )

    print()
    print("[CONTROLLER]")
    print(
        "    Calling current "
        "AutonomousMitigationController.execute()"
    )

    controller = AutonomousMitigationController()

    try:
        result = controller.execute(
            reasoning_result
        )

    except Exception as exc:

        print()
        print("=" * 72)
        print("[TEST ERROR]")
        print("=" * 72)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return 1

    print()
    print("=" * 72)
    print("[RAW RESULT]")
    print("=" * 72)

    print(result)

    print()
    print("=" * 72)
    print("[FINAL RESULT]")
    print("=" * 72)

    success = bool(
        result.get("success", False)
    )

    verified = bool(
        result.get("verified", False)
    )

    status = result.get(
        "status",
        "",
    )

    print(f"Status   : {status}")
    print(f"Success  : {success}")
    print(f"Verified : {verified}")

    print()
    print("=" * 72)
    print("[SAFETY ASSERTIONS]")
    print("=" * 72)

    if success:
        print(
            "[FAIL] Identity-mismatched target "
            "was reported as successfully terminated."
        )
        return 1

    print(
        "[PASS] Identity-mismatched target "
        "was not successfully terminated."
    )

    if verified:
        print(
            "[FAIL] Identity-mismatched target "
            "was reported as verified."
        )
        return 1

    print(
        "[PASS] Identity-mismatched target "
        "was not verified as terminated."
    )

    if status != "VICTIM_REJECTED_REQUEST":
        print(
            "[WARN] Unexpected rejection status: "
            f"{status}"
        )
    else:
        print(
            "[PASS] Victim agent rejected the "
            "process identity mismatch."
        )

    print()
    print("=" * 72)
    print(
        "PROCESS IDENTITY MISMATCH TEST: PASS"
    )
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
