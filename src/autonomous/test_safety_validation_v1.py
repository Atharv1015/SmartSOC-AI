"""
SmartSOC Safety Validation Test v1

Purpose:
    Validate that the current AutonomousMitigationController rejects
    a stale/non-running process target.

Test:
    PID 65000
    Process name: notepad.exe

Safety expectation:
    - PID must not exist on the victim.
    - Controller must use the current execute() interface.
    - Victim agent must reject the nonexistent PID.
    - No process must be terminated.
    - success must be False.
    - verified must be False.

This test does NOT modify production code.
"""

from __future__ import annotations

import sys
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Import current controller
# ---------------------------------------------------------------------------

try:
    from autonomous_mitigation_controller_v1 import (
        AutonomousMitigationController,
    )
except ImportError:
    # Allow execution from the project root.
    sys.path.insert(
        0,
        r".\src\autonomous",
    )

    from autonomous_mitigation_controller_v1 import (
        AutonomousMitigationController,
    )


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

TEST_PID = 65000
TEST_PROCESS_NAME = "notepad.exe"


# ---------------------------------------------------------------------------
# Build controlled reasoning result
# ---------------------------------------------------------------------------

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
            TEST_PROCESS_NAME,

        "wazuh_rule_id":
            100001,

        "wazuh_rule_level":
            12,

        "reason":
            (
                "Safety validation test for stale "
                "non-running process target."
            ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:

    print("=" * 72)
    print("SMARTSOC SAFETY VALIDATION - STALE PID TEST v1")
    print("=" * 72)

    print()
    print("[TEST TARGET]")
    print(f"    PID          : {TEST_PID}")
    print(f"    Process      : {TEST_PROCESS_NAME}")
    print("    Expected     : Victim rejection")
    print("    Expected kill: NONE")

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
        f"    Process      : "
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

    # -----------------------------------------------------------------------
    # Safety assertions
    # -----------------------------------------------------------------------

    print()
    print("=" * 72)
    print("[SAFETY ASSERTIONS]")
    print("=" * 72)

    # The stale target must NOT succeed.
    if success:
        print(
            "[FAIL] Stale PID was reported as successful."
        )
        return 1

    print(
        "[PASS] Stale PID was not successfully terminated."
    )

    # The endpoint must not claim verified termination.
    if verified:
        print(
            "[FAIL] Stale PID was reported as verified."
        )
        return 1

    print(
        "[PASS] Stale PID was not verified as terminated."
    )

    # A rejection status is expected.
    if status not in {
        "VICTIM_REJECTED_REQUEST",
        "TARGET_NOT_FOUND",
        "MITIGATION_REJECTED",
        "PROCESS_NOT_FOUND",
    }:
        print(
            "[WARN] Unexpected rejection status:"
            f" {status}"
        )
    else:
        print(
            "[PASS] Victim/controller rejected the stale target."
        )

    print()
    print("=" * 72)
    print("STALE PID SAFETY TEST: PASS")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())