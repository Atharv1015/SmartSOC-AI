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

AUTH_INPUT = (
    PROJECT_ROOT
    / "reports"
    / "integration"
    / "authorization_decision_v1.json"
)

PLAN_INPUT = (
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
    / "execution_policy_guard_result_v1.json"
)


def load_json(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def evaluate_guard(
    authorization,
    response_plan,
):

    validate_safety_invariants()

    auth = authorization.get(
        "authorization",
        {}
    )

    execution = authorization.get(
        "execution_control",
        {}
    )

    policy = response_plan.get(
        "response_policy",
        {}
    )

    authorization_required_value = policy.get("authorization_required", True)
    authorization_required = (
        authorization_required_value
        if type(authorization_required_value) is bool
        else True
    )

    authorization_status = str(
        auth.get(
            "status",
            ""
        )
    ).upper()

    authorization_decision = str(
        auth.get(
            "decision",
            ""
        )
    ).upper()

    analyst_authorized_value = auth.get("analyst_authorized", False)
    analyst_authorized = (
        analyst_authorized_value
        if type(analyst_authorized_value) is bool
        else False
    )

    policy_execution_allowed_value = execution.get("execution_allowed", False)
    policy_execution_allowed = (
        policy_execution_allowed_value
        if type(policy_execution_allowed_value) is bool
        else False
    )

    # --------------------------------------------------------
    # Authorization condition
    #
    # If human authorization is disabled centrally, the
    # controlled-lab policy itself satisfies this boundary.
    # --------------------------------------------------------

    if HUMAN_AUTHORIZATION_REQUIRED:

        authorization_policy_satisfied = (
            authorization_required
            and analyst_authorized
            and authorization_status == "APPROVED"
        )

    else:

        authorization_policy_satisfied = (
            authorization_status == "AUTHORIZED"
            and authorization_decision == "POLICY_AUTHORIZED"
        )

    # --------------------------------------------------------
    # Centralized execution policy conditions
    # --------------------------------------------------------

    execution_layer_condition = (
        EXECUTION_LAYER_ENABLED
    )

    automatic_execution_condition = (
        AUTOMATIC_EXECUTION
    )

    automatic_containment_condition = (
        AUTOMATIC_CONTAINMENT
    )

    unrestricted_autonomy_condition = (
        not UNRESTRICTED_AUTONOMOUS_ACTIONS
    )

    controlled_lab_condition = (
        CONTROLLED_LAB_MODE
    )

    endpoint_not_modified = True

    shuffle_not_called = True

    caldera_not_called = True

    response_policy_authorization = (
        policy_execution_allowed
    )

    conditions = {
        "authorization_policy_satisfied":
            authorization_policy_satisfied,

        "execution_layer_enabled":
            execution_layer_condition,

        "automatic_execution_enabled":
            automatic_execution_condition,

        "automatic_containment_enabled":
            automatic_containment_condition,

        "unrestricted_autonomous_actions_disabled":
            unrestricted_autonomy_condition,

        "controlled_lab_mode_enabled":
            controlled_lab_condition,

        "endpoint_not_modified":
            endpoint_not_modified,

        "shuffle_not_called_previously":
            shuffle_not_called,

        "caldera_not_called":
            caldera_not_called,

        "response_policy_authorization":
            response_policy_authorization,
    }

    blocking_conditions = [
        name
        for name, passed in conditions.items()
        if not passed
    ]

    execution_allowed = (
        len(blocking_conditions) == 0
    )

    if execution_allowed:

        decision = "ELIGIBLE"

    else:

        decision = "BLOCKED"

    return {
        "decision":
            decision,

        "execution_allowed":
            execution_allowed,

        "conditions":
            conditions,

        "blocking_conditions":
            blocking_conditions,

        "authorization": {
            "authorization_required":
                authorization_required,

            "human_authorization_required":
                HUMAN_AUTHORIZATION_REQUIRED,

            "authorization_status":
                authorization_status,

            "authorization_decision":
                authorization_decision,

            "analyst_authorized":
                analyst_authorized,

            "policy_execution_allowed":
                policy_execution_allowed,
        },
    }


def main():

    print("=" * 72)
    print("SmartSOC EXECUTION POLICY GUARD v2")
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
    print("[1/4] Loading authorization decision...")

    authorization = load_json(
        AUTH_INPUT
    )

    print(
        f"[PASS] {AUTH_INPUT}"
    )

    print()
    print("[2/4] Loading response plan...")

    response_plan = load_json(
        PLAN_INPUT
    )

    print(
        f"[PASS] {PLAN_INPUT}"
    )

    print()
    print("[3/4] Evaluating execution policy...")

    result = evaluate_guard(
        authorization,
        response_plan,
    )

    for name, passed in result[
        "conditions"
    ].items():

        status = (
            "PASS"
            if passed
            else "BLOCK"
        )

        print(
            f"  [{status}] {name}"
        )

    print()
    print(
        f"Decision          : "
        f"{result['decision']}"
    )

    print(
        f"Execution allowed : "
        f"{result['execution_allowed']}"
    )

    if result[
        "blocking_conditions"
    ]:

        print()
        print("Blocking conditions:")

        for condition in result[
            "blocking_conditions"
        ]:

            print(
                f"  - {condition}"
            )

    # --------------------------------------------------------
    # The policy guard evaluates authorization.
    #
    # It does NOT execute endpoint actions.
    # --------------------------------------------------------

    execution_state = {
        "execution_requested":
            False,

        "execution_performed":
            False,

        "shuffle_called":
            False,

        "caldera_called":
            False,

        "endpoint_modified":
            False,
    }

    output = {
        "schema":
            "SmartSOC-ExecutionPolicyGuard-v2",

        "timestamp_utc":
            datetime.now(timezone.utc).isoformat(),

        "mode":
            "CONTROLLED_EXECUTION_POLICY_GUARD",

        "source": {
            "authorization_decision":
                str(AUTH_INPUT),

            "response_plan":
                str(PLAN_INPUT),
        },

        "policy_evaluation":
            result,

        "execution":
            execution_state,

        "safety": {
            "read_only":
                True,

            "execution_layer_enabled":
                EXECUTION_LAYER_ENABLED,

            "automatic_execution":
                AUTOMATIC_EXECUTION,

            "automatic_containment":
                AUTOMATIC_CONTAINMENT,

            "human_authorization_required":
                HUMAN_AUTHORIZATION_REQUIRED,

            "human_authorization_obtained": result["authorization"]["analyst_authorized"],

            "controlled_lab_mode":
                CONTROLLED_LAB_MODE,

            "unrestricted_autonomous_actions":
                UNRESTRICTED_AUTONOMOUS_ACTIONS,

            "policy_authorized":
                result[
                    "execution_allowed"
                ],
        },

        "audit": {
            "policy_guard_evaluated":
                True,

            "response_execution_performed":
                False,

            "production_model_modified":
                False,

            "production_artifacts_modified":
                False,
        },
    }

    print()
    print("[4/4] Writing guard result...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            output,
            handle,
            indent=2,
        )

    print(
        f"[PASS] {OUTPUT_PATH}"
    )

    print()
    print(
        "[PASS] Execution Policy Guard completed."
    )

    print(
        "No response action was executed by the policy guard."
    )


if __name__ == "__main__":
    main()

