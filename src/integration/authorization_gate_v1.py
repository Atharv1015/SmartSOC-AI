import json
from pathlib import Path
from datetime import datetime, timezone

from execution_config_v1 import (
    EXECUTION_LAYER_ENABLED,
    AUTOMATIC_EXECUTION,
    AUTOMATIC_CONTAINMENT,
    UNRESTRICTED_AUTONOMOUS_ACTIONS,
    CONTROLLED_LAB_MODE,
    HUMAN_AUTHORIZATION_REQUIRED,
    validate_safety_invariants,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "integration"
    / "response_plan_v1.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "integration"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "authorization_gate_result_v1.json"
)


def load_response_plan():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Response plan not found: {INPUT_PATH}"
        )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def build_authorization_result(plan):

    risk = plan.get(
        "risk",
        {}
    )

    policy = plan.get(
        "response_policy",
        {}
    )

    execution = plan.get(
        "execution",
        {}
    )

    severity = str(
        risk.get(
            "severity",
            "INFO"
        )
    ).upper()

    risk_score = risk.get(
        "risk_score"
    )

    authorization_required = bool(
        policy.get(
            "authorization_required",
            True
        )
    )

    # --------------------------------------------------------
    # Validate the centralized execution configuration.
    # --------------------------------------------------------

    validate_safety_invariants()

    # --------------------------------------------------------
    # Determine whether the controlled execution layer can
    # proceed without human authorization.
    #
    # Human authorization is explicitly disabled in the
    # controlled lab configuration.
    # --------------------------------------------------------

    if HUMAN_AUTHORIZATION_REQUIRED:

        authorization_status = "PENDING"

        analyst_authorized = False

        execution_allowed = False

        reason = (
            "Human authorization is required by the centralized "
            "execution policy. No analyst authorization has been "
            "provided by this gate."
        )

    else:

        authorization_status = "NOT_REQUIRED"

        analyst_authorized = False

        execution_allowed = (
            EXECUTION_LAYER_ENABLED
            and AUTOMATIC_EXECUTION
            and AUTOMATIC_CONTAINMENT
            and CONTROLLED_LAB_MODE
            and not UNRESTRICTED_AUTONOMOUS_ACTIONS
        )

        if execution_allowed:

            reason = (
                "Human authorization is not required. The controlled "
                "lab execution policy authorizes the downstream "
                "execution layer, subject to the execution policy "
                "guard and action allowlist."
            )

        else:

            reason = (
                "Human authorization is not required, but the "
                "centralized execution configuration does not permit "
                "automatic controlled execution."
            )

    return {
        "schema":
            "SmartSOC-AuthorizationGate-v2",

        "timestamp_utc":
            datetime.now(timezone.utc).isoformat(),

        "mode":
            "CONTROLLED_AUTONOMOUS_AUTHORIZATION_GATE",

        "source": {
            "response_plan":
                str(INPUT_PATH),

            "event_id":
                plan.get(
                    "source",
                    {}
                ).get(
                    "event_id"
                ),

            "rule_id":
                plan.get(
                    "source",
                    {}
                ).get(
                    "rule_id"
                ),
        },

        "risk": {
            "risk_score":
                risk_score,

            "severity":
                severity,

            "detected_techniques":
                risk.get(
                    "detected_techniques",
                    []
                ),
        },

        "response_policy": {
            "authorization_required":
                authorization_required,

            "human_authorization_required":
                HUMAN_AUTHORIZATION_REQUIRED,

            "planned_execution":
                execution,
        },

        "authorization": {
            "required":
                authorization_required,

            "status":
                authorization_status,

            "analyst_authorized":
                analyst_authorized,

            "authorization_timestamp":
                None,

            "authorization_actor":
                None,

            "authorization_reason":
                None,

            "decision_reason":
                reason,
        },

        "execution_control": {
            "execution_allowed":
                execution_allowed,

            "execution_layer_enabled":
                EXECUTION_LAYER_ENABLED,

            "automatic_execution":
                AUTOMATIC_EXECUTION,

            "automatic_containment":
                AUTOMATIC_CONTAINMENT,

            "unrestricted_autonomous_actions":
                UNRESTRICTED_AUTONOMOUS_ACTIONS,

            "controlled_lab_mode":
                CONTROLLED_LAB_MODE,

            "human_authorization_required":
                HUMAN_AUTHORIZATION_REQUIRED,

            "shuffle_called":
                False,

            "caldera_called":
                False,

            "endpoint_modified":
                False,
        },

        "rollback": {
            "required_if_executed":
                True,

            "rollback_prepared":
                False,

            "rollback_executed":
                False,
        },

        "audit": {
            "authorization_gate_evaluated":
                True,

            "analyst_authorization_obtained":
                analyst_authorized,

            "response_execution_performed":
                False,

            "production_model_modified":
                False,

            "production_artifacts_modified":
                False,
        },

        "safety": {
            "read_only":
                True,

            "human_in_the_loop":
                HUMAN_AUTHORIZATION_REQUIRED,

            "execution_policy_authorized":
                execution_allowed,

            "execution_blocked":
                not execution_allowed,

            "execution_layer_enabled":
                EXECUTION_LAYER_ENABLED,

            "automatic_execution":
                AUTOMATIC_EXECUTION,

            "automatic_containment":
                AUTOMATIC_CONTAINMENT,

            "controlled_lab_mode":
                CONTROLLED_LAB_MODE,

            "unrestricted_autonomous_actions":
                UNRESTRICTED_AUTONOMOUS_ACTIONS,
        },
    }


def main():

    print("=" * 72)
    print("SmartSOC CONTROLLED AUTONOMOUS AUTHORIZATION GATE v2")
    print("=" * 72)

    print()
    print("[SAFETY / EXECUTION CONFIGURATION]")

    print(
        f"  Execution layer enabled     : "
        f"{EXECUTION_LAYER_ENABLED}"
    )

    print(
        f"  Automatic execution         : "
        f"{AUTOMATIC_EXECUTION}"
    )

    print(
        f"  Automatic containment       : "
        f"{AUTOMATIC_CONTAINMENT}"
    )

    print(
        f"  Human authorization         : "
        f"{HUMAN_AUTHORIZATION_REQUIRED}"
    )

    print(
        f"  Controlled lab mode         : "
        f"{CONTROLLED_LAB_MODE}"
    )

    print(
        f"  Unrestricted autonomy       : "
        f"{UNRESTRICTED_AUTONOMOUS_ACTIONS}"
    )

    print(
        "  Shuffle calls               : FALSE"
    )

    print(
        "  Caldera calls               : FALSE"
    )

    print(
        "  Endpoint modification       : FALSE"
    )

    print()
    print("[1/3] Loading response plan...")

    plan = load_response_plan()

    print(
        f"[PASS] Input: {INPUT_PATH}"
    )

    print()
    print("[2/3] Evaluating authorization boundary...")

    result = build_authorization_result(
        plan
    )

    print(
        f"[PASS] Risk score : "
        f"{result['risk']['risk_score']}"
    )

    print(
        f"[PASS] Severity   : "
        f"{result['risk']['severity']}"
    )

    print(
        f"[PASS] Techniques : "
        f"{result['risk']['detected_techniques']}"
    )

    print(
        f"[PASS] Required   : "
        f"{result['authorization']['required']}"
    )

    print(
        f"[PASS] Status     : "
        f"{result['authorization']['status']}"
    )

    print(
        f"[PASS] Execution  : "
        f"{result['execution_control']['execution_allowed']}"
    )

    print(
        f"[PASS] Reason     : "
        f"{result['authorization']['decision_reason']}"
    )

    print()
    print("[3/3] Writing authorization artifact...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            result,
            handle,
            indent=2,
        )

    print(
        f"[PASS] {OUTPUT_PATH}"
    )

    print()
    print(
        "[PASS] Authorization Gate completed."
    )

    print(
        "No response action was executed by the authorization gate."
    )


if __name__ == "__main__":
    main()