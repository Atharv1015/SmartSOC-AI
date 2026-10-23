"""SmartSOC controlled-lab execution policy."""

import os

CONTROLLED_LAB_MODE = True
EXECUTION_LAYER_ENABLED = True
AUTOMATIC_EXECUTION = True
AUTOMATIC_CONTAINMENT = True

UNRESTRICTED_AUTONOMOUS_ACTIONS = False
HUMAN_AUTHORIZATION_REQUIRED = False

ALLOWED_ACTIONS = frozenset({"PROCESS_TERMINATION"})
# Explicit controlled-lab process allowlist. Several entries are dual-use.
# Do not add core Windows security/session/service processes to this set.
ALLOWED_PROCESSES = frozenset({"notepad.exe", "ping.exe"})
DRY_RUN = False
LIVE_EXECUTION_ENABLED = os.environ.get("SMARTSOC_ALLOW_LIVE_MITIGATION", "").strip() == "1"
def is_process_allowed(process_name: str) -> bool:
    if not isinstance(process_name, str):
        return False
    normalized = (
        process_name.strip().replace("\\", "/").rsplit("/", 1)[-1].lower()
    )
    return normalized in ALLOWED_PROCESSES


def is_action_allowed(action: str) -> bool:
    return (
        isinstance(action, str)
        and action.strip().upper() in ALLOWED_ACTIONS
    )


def validate_safety_invariants() -> bool:
    return (
        CONTROLLED_LAB_MODE
        and EXECUTION_LAYER_ENABLED
        and AUTOMATIC_EXECUTION
        and AUTOMATIC_CONTAINMENT
        and not UNRESTRICTED_AUTONOMOUS_ACTIONS
        and not HUMAN_AUTHORIZATION_REQUIRED
        and ALLOWED_ACTIONS == frozenset({"PROCESS_TERMINATION"})
        and ALLOWED_PROCESSES == frozenset({"notepad.exe", "ping.exe"})
    )


