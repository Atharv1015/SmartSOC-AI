import json
import sys
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
    / "authorization_gate_result_v1.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "integration"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "authorization_decision_v1.json"
)


VALID_HUMAN_DECISIONS = {
    "APPROVE",
    "DENY",
}


def load_gate_result():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Authorization Gate result not found: {INPUT_PATH}"
        )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def build_decision(
    gate_result,
    decision=None,
):

    validate_safety_invariants()

    authorization = gate_result.get(
        "authorization",
        {}
    )

    risk = gate_result.get(
        "risk",
        {}
    )

    gate_execution = gate_result.get(
        "execution_control",
        {}
    )

    # --------------------------------------------------------
    # CONTROLLED AUTONOMOUS MODE
    #
    # When human authorization is disabled in the centralized
    # execution configuration, the policy itself provides the
    # authorization boundary.
    # --------------------------------------------------------

    if not HUMAN_AUTHORIZATION_REQUIRED:

        status = "AUTHORIZED"

        authorization_decision = "POLICY_AUTHORIZED"

        analyst_authorized = False

        execution_allowed = (
            EXECUTION_LAYER_ENABLED
            and AUTOMATIC_EXECUTION
            and AUTOMATIC_CONTAINMENT
            and CONTROLLED_LAB_MODE
            and not UNRESTRICTED_AUTONOMOUS_ACTIONS
            and bool(
                gate_execution.get(
                    "execution_allowed",
                    False
                )
            )
        )

        if execution_allowed:

            reason = (
                "Human authorization is not required. "
                "Execution is authorized by the centralized "
                "controlled-lab policy. Downstream execution "
                "remains subject to the execution policy guard "
                "and action allowlist."
            )

        else:

            reason = (
                "Human authorization is not required, but the "
                "centralized controlled-lab execution policy "
                "does not currently permit execution."
            )

        authorization_actor = (
            "SMARTSOC_CONTROLLED_LAB_POLICY"
        )

    # --------------------------------------------------------
    # HUMAN-AUTHORIZATION MODE
    # --------------------------------------------------------

    else:

        if decision is None:

            raise ValueError(
                "Human authorization is required. "
                "Provide APPROVE or DENY."
            )

        decision = decision.upper()

        if decision not in VALID_HUMAN_DECISIONS:

            raise ValueError(
                "Decision must be APPROVE or DENY."
            )

        authorization_decision = decision

        if decision == "APPROVE":

            status = "APPROVED"

            analyst_authorized = True

            execution_allowed = (
                EXECUTION_LAYER_ENABLED
                and AUTOMATIC_EXECUTION
                and AUTOMATIC_CONTAINMENT
                and CONTROLLED_LAB_MODE
                and not UNRESTRICTED_AUTONOMOUS_ACTIONS
                and bool(
                    gate_execution.get(
                        "execution_allowed",
                        False
                    )
                )
            )

            if execution_allowed:

                reason = (
                    "Analyst explicitly approved the response "
                    "plan and the centralized controlled-lab "
                    "execution policy permits downstream execution."
                )

            else:

                reason = (
                    "Analyst explicitly approved the response "
                    "plan, but the centralized execution policy "
                    "does not permit downstream execution."
                )

        else:

            status = "DENIED"

            analyst_authorized = False

            execution_allowed = False

            reason = (
                "Analyst explicitly denied the proposed "
                "response plan. Execution is blocked."
            )

        authorization_actor = "LAB_ANALYST"

    return {
        "schema":
            "SmartSOC-AuthorizationDecision-v2",

        "timestamp_utc":
            datetime.now(timezone.utc).isoformat(),

        "mode":
            (
                "CONTROLLED_AUTONOMOUS_AUTHORIZATION"
                if not HUMAN_AUTHORIZATION_REQUIRED
                else "CONTROLLED_HUMAN_AUTHORIZATION"
            ),

        "source": {
            "authorization_gate":
                str(INPUT_PATH),

            "event_id":
                gate_result.get(
                    "source",
                    {}
                ).get(
                    "event_id"
                ),

            "rule_id":
                gate_result.get(
                    "source",
                    {}
                ).get(
                    "rule_id"
                ),
        },

        "risk": {
            "risk_score":
                risk.get(
                    "risk_score"
                ),

            "severity":
                risk.get(
                    "severity"
                ),

            "detected_techniques":
                risk.get(
                    "detected_techniques",
                    []
                ),
        },

        "authorization": {
            "required":
                HUMAN_AUTHORIZATION_REQUIRED,

            "decision":
                authorization_decision,

            "status":
                status,

            "analyst_authorized":
                analyst_authorized,

            "authorization_timestamp":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "authorization_actor":
                authorization_actor,

            "authorization_reason":
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
            "authorization_decision_recorded":
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

            "endpoint_modified":
                False,
        },
    }


def main():

    print("=" * 72)
    print("SmartSOC CONTROLLED AUTHORIZATION DECISION v2")
    print("=" * 72)

    print()
    print("[SAFETY / EXECUTION CONFIGURATION]")

    print(
        f"  Human authorization         : "
        f"{HUMAN_AUTHORIZATION_REQUIRED}"
    )

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

    # --------------------------------------------------------
    # CLI behavior
    #
    # Autonomous controlled-lab mode:
    #     python authorization_decision_v1.py
    #
    # Human authorization mode:
    #     python authorization_decision_v1.py APPROVE
    #     python authorization_decision_v1.py DENY
    # --------------------------------------------------------

    if HUMAN_AUTHORIZATION_REQUIRED:

        if len(sys.argv) != 2:

            print()
            print("Usage:")
            print(
                "  python "
                "authorization_decision_v1.py APPROVE"
            )
            print(
                "  python "
                "authorization_decision_v1.py DENY"
            )

            sys.exit(1)

        decision = sys.argv[1].upper()

    else:

        if len(sys.argv) != 1:

            print()
            print(
                "[ERROR] Human authorization is disabled "
                "in the centralized controlled-lab policy."
            )

            print(
                "[ERROR] Do not provide APPROVE or DENY."
            )

            sys.exit(1)

        decision = None

    print()
    print("[1/3] Loading Authorization Gate result...")

    gate_result = load_gate_result()

    print(
        f"[PASS] Input: {INPUT_PATH}"
    )

    print()
    print("[2/3] Recording authorization decision...")

    result = build_decision(
        gate_result,
        decision,
    )

    print(
        f"[PASS] Decision   : "
        f"{result['authorization']['decision']}"
    )

    print(
        f"[PASS] Status     : "
        f"{result['authorization']['status']}"
    )

    print(
        f"[PASS] Authorized : "
        f"{result['authorization']['analyst_authorized']}"
    )

    print(
        f"[PASS] Execution  : "
        f"{result['execution_control']['execution_allowed']}"
    )

    print(
        f"[PASS] Actor      : "
        f"{result['authorization']['authorization_actor']}"
    )

    print(
        f"[PASS] Reason     : "
        f"{result['authorization']['authorization_reason']}"
    )

    print()
    print("[3/3] Writing authorization decision artifact...")

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
        "[PASS] Authorization decision recorded."
    )

    print(
        "No response action was executed by the authorization decision module."
    )


if __name__ == "__main__":
    main()