"""
SmartSOC Execution Configuration v3

Controlled autonomous defensive execution policy.

This configuration enables automatic containment in the controlled
SmartSOC laboratory.

Currently enabled containment capability:
    PROCESS_TERMINATION

Important security invariants:
    - Automatic execution is enabled only in controlled lab mode.
    - Automatic containment is enabled.
    - Only explicitly allowlisted defensive actions are permitted.
    - Unrestricted autonomous actions remain disabled.
    - Arbitrary shell/command execution is never allowed.
    - Unknown actions are denied.
    - Endpoint modification requires an explicit allowlist entry.
    - Human authorization is not required for the controlled lab policy.
"""

from __future__ import annotations


# ============================================================================
# EXECUTION POLICY
# ============================================================================

EXECUTION_LAYER_ENABLED = True

# SmartSOC may execute an approved defensive action automatically.
AUTOMATIC_EXECUTION = True

# SmartSOC may automatically perform the explicitly allowlisted containment
# action(s) in the controlled laboratory.
AUTOMATIC_CONTAINMENT = True

# NEVER enable unrestricted autonomous execution.
UNRESTRICTED_AUTONOMOUS_ACTIONS = False


# ============================================================================
# CONTROLLED LABORATORY MODE
# ============================================================================

CONTROLLED_LAB_MODE = True

# No human approval is required for the explicitly allowlisted lab action.
HUMAN_AUTHORIZATION_REQUIRED = True


# ============================================================================
# ALLOWLISTED DEFENSIVE ACTIONS
# ============================================================================

ALLOWED_ACTIONS = frozenset({
    "PROCESS_TERMINATION",
})


# ============================================================================
# DEFAULT-DENY POLICY
# ============================================================================

DENY_UNKNOWN_ACTIONS = True

# Any endpoint-changing action must explicitly appear in ALLOWED_ACTIONS.
ENDPOINT_MODIFICATION_REQUIRES_ALLOWLIST = True


# ============================================================================
# POLICY HELPERS
# ============================================================================

def is_action_allowed(action: str) -> bool:
    """
    Return True only when the requested action is explicitly allowlisted.
    """

    if not isinstance(action, str):
        return False

    normalized = action.strip().upper()

    if not normalized:
        return False

    if DENY_UNKNOWN_ACTIONS:
        return normalized in ALLOWED_ACTIONS

    return normalized in ALLOWED_ACTIONS


def validate_safety_invariants() -> None:
    """
    Validate SmartSOC execution-policy safety invariants.

    Automatic containment is intentionally permitted here because the
    containment capability is constrained by ALLOWED_ACTIONS.
    """

    # ----------------------------------------------------------------------
    # Unrestricted autonomy must NEVER be enabled.
    # ----------------------------------------------------------------------

    if UNRESTRICTED_AUTONOMOUS_ACTIONS:
        raise RuntimeError(
            "SAFETY VIOLATION: unrestricted autonomous actions "
            "must remain disabled."
        )

    # ----------------------------------------------------------------------
    # Controlled laboratory mode is required for automatic execution.
    # ----------------------------------------------------------------------

    if AUTOMATIC_EXECUTION and not CONTROLLED_LAB_MODE:
        raise RuntimeError(
            "SAFETY VIOLATION: automatic execution requires "
            "controlled lab mode."
        )

    # ----------------------------------------------------------------------
    # Automatic containment requires the execution layer.
    # ----------------------------------------------------------------------

    if AUTOMATIC_CONTAINMENT and not EXECUTION_LAYER_ENABLED:
        raise RuntimeError(
            "CONFIGURATION ERROR: automatic containment requires "
            "the execution layer to be enabled."
        )

    # ----------------------------------------------------------------------
    # Automatic containment requires automatic execution.
    # ----------------------------------------------------------------------

    if AUTOMATIC_CONTAINMENT and not AUTOMATIC_EXECUTION:
        raise RuntimeError(
            "CONFIGURATION ERROR: automatic containment requires "
            "automatic execution to be enabled."
        )

    # ----------------------------------------------------------------------
    # Current controlled containment capability.
    # ----------------------------------------------------------------------

    if not is_action_allowed("PROCESS_TERMINATION"):
        raise RuntimeError(
            "CONFIGURATION ERROR: PROCESS_TERMINATION must be "
            "explicitly allowlisted."
        )

    # ----------------------------------------------------------------------
    # Broad/unrestricted containment is never allowed.
    #
    # Automatic containment here means controlled, allowlisted containment.
    # ----------------------------------------------------------------------

    if AUTOMATIC_CONTAINMENT:
        if not ALLOWED_ACTIONS:
            raise RuntimeError(
                "SAFETY VIOLATION: automatic containment cannot operate "
                "with an empty action allowlist."
            )

        for action in ALLOWED_ACTIONS:
            if not isinstance(action, str) or not action.strip():
                raise RuntimeError(
                    "SAFETY VIOLATION: invalid action in allowlist."
                )

    # ----------------------------------------------------------------------
    # Endpoint modification must remain allowlisted.
    # ----------------------------------------------------------------------

    if ENDPOINT_MODIFICATION_REQUIRES_ALLOWLIST:
        if not is_action_allowed("PROCESS_TERMINATION"):
            raise RuntimeError(
                "SAFETY VIOLATION: PROCESS_TERMINATION is not allowlisted."
            )


# ============================================================================
# VALIDATE ON IMPORT
# ============================================================================

validate_safety_invariants()


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("SmartSOC EXECUTION CONFIGURATION v3")
    print("=" * 72)

    print()
    print("[EXECUTION POLICY]")
    print(f"Execution layer enabled   : {EXECUTION_LAYER_ENABLED}")
    print(f"Automatic execution       : {AUTOMATIC_EXECUTION}")
    print(f"Automatic containment     : {AUTOMATIC_CONTAINMENT}")
    print(
        "Unrestricted autonomy     : "
        f"{UNRESTRICTED_AUTONOMOUS_ACTIONS}"
    )

    print()
    print("[CONTROLLED LAB]")
    print(f"Controlled lab mode       : {CONTROLLED_LAB_MODE}")
    print(
        "Human authorization       : "
        f"{HUMAN_AUTHORIZATION_REQUIRED}"
    )

    print()
    print("[ALLOWED ACTIONS]")

    for action in sorted(ALLOWED_ACTIONS):
        print(f"  {action}")

    print()
    print("[POLICY TESTS]")

    print(
        "PROCESS_TERMINATION       : "
        f"{is_action_allowed('PROCESS_TERMINATION')}"
    )

    print(
        "ARBITRARY_COMMAND         : "
        f"{is_action_allowed('ARBITRARY_COMMAND')}"
    )

    print(
        "UNKNOWN_ACTION             : "
        f"{is_action_allowed('UNKNOWN_ACTION')}"
    )

    print()
    print("[SAFETY]")
    print("Safety invariants         : VALID")

    print()
    if AUTOMATIC_CONTAINMENT:
        print("Automatic containment is ENABLED.")
    else:
        print("Automatic containment is DISABLED.")
    print(
        "Containment is restricted to explicitly "
        "allowlisted defensive actions."
    )

    print("=" * 72)


