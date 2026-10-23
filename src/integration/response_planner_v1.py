import json
from pathlib import Path
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RISK_INPUT = (
    PROJECT_ROOT
    / "reports"
    / "integration"
    / "risk_engine_result.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "integration"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "response_plan_v1.json"
)


def load_risk_result():

    if not RISK_INPUT.exists():
        raise FileNotFoundError(
            f"Risk Engine result not found: {RISK_INPUT}"
        )

    with RISK_INPUT.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def get_detections(data):

    detections = data.get(
        "detections",
        {}
    )

    if not isinstance(detections, dict):
        return []

    techniques = detections.get(
        "techniques",
        []
    )

    if not isinstance(techniques, list):
        return []

    normalized = []

    for item in techniques:

        if not isinstance(item, dict):
            continue

        technique = item.get(
            "technique"
        )

        if not technique:
            continue

        confidence = item.get(
            "confidence"
        )

        threshold = item.get(
            "threshold"
        )

        detected = True

        if confidence is not None and threshold is not None:

            try:
                detected = (
                    float(confidence)
                    >= float(threshold)
                )
            except (
                TypeError,
                ValueError,
            ):
                detected = False

        normalized.append(
            {
                "technique":
                    technique,

                "confidence":
                    confidence,

                "threshold":
                    threshold,

                "detected":
                    detected,
            }
        )

    return normalized


def get_risk(data):

    calculation = data.get(
        "risk_calculation",
        {}
    )

    decision = data.get(
        "decision",
        {}
    )

    if not isinstance(calculation, dict):
        calculation = {}

    if not isinstance(decision, dict):
        decision = {}

    return {
        "risk_score":
            calculation.get(
                "risk_score"
            ),

        "severity":
            str(
                calculation.get(
                    "severity",
                    "INFO"
                )
            ).upper(),

        "recommended_action":
            decision.get(
                "recommended_action"
            ),

        "authorization_required":
            decision.get(
                "authorization_required",
                True
            ),

        "rollback_required":
            decision.get(
                "rollback_required_if_executed",
                True
            ),
    }


def build_evidence_plan(
    severity,
    techniques,
):

    evidence = [
        "Preserve the original Wazuh alert.",
        "Preserve the SmartSOC inference result.",
        "Preserve the Risk Engine result.",
        "Record analyst authorization status.",
    ]

    technique_set = set(techniques)

    if "T1070.004" in technique_set:

        evidence.extend(
            [
                "Review file-deletion telemetry.",
                "Review PowerShell process creation telemetry.",
                "Preserve relevant Sysmon process events.",
            ]
        )

    if "T1059.001" in technique_set:

        evidence.extend(
            [
                "Review PowerShell parent-child "
                "process relationships.",
                "Preserve PowerShell command-line telemetry.",
            ]
        )

    if "T1059.003" in technique_set:

        evidence.append(
            "Review Windows command-shell process telemetry."
        )

    if "T1087" in technique_set:

        evidence.extend(
            [
                "Review account-discovery process telemetry.",
                "Preserve relevant discovery command-line evidence.",
            ]
        )

    if severity in {
        "HIGH",
        "CRITICAL",
    }:

        evidence.extend(
            [
                "Review events immediately before and "
                "after the triggering alert.",
                "Review endpoint telemetry before response.",
                "Review available network telemetry before response.",
            ]
        )

    return list(
        dict.fromkeys(evidence)
    )


def build_response_actions(
    severity,
    techniques,
):

    technique_set = set(techniques)

    actions = []

    if severity in {
        "INFO",
        "LOW",
    }:

        actions.extend(
            [
                {
                    "action_id":
                        "EVIDENCE_REVIEW",

                    "action":
                        "Review alert and supporting telemetry.",

                    "execution_mode":
                        "ANALYST_REVIEW",

                    "authorization_required":
                        False,

                    "destructive":
                        False,
                },
                {
                    "action_id":
                        "CORRELATION",

                    "action":
                        "Correlate related Wazuh events.",

                    "execution_mode":
                        "ANALYST_REVIEW",

                    "authorization_required":
                        False,

                    "destructive":
                        False,
                },
            ]
        )

    elif severity == "MEDIUM":

        actions.extend(
            [
                {
                    "action_id":
                        "EVIDENCE_COLLECTION",

                    "action":
                        "Collect and preserve supporting telemetry.",

                    "execution_mode":
                        "PREPARE_ONLY",

                    "authorization_required":
                        True,

                    "destructive":
                        False,
                },
                {
                    "action_id":
                        "ANALYST_REVIEW",

                    "action":
                        "Perform analyst review before containment.",

                    "execution_mode":
                        "ANALYST_REVIEW",

                    "authorization_required":
                        True,

                    "destructive":
                        False,
                },
            ]
        )

    elif severity in {
        "HIGH",
        "CRITICAL",
    }:

        actions.extend(
            [
                {
                    "action_id":
                        "PRESERVE_EVIDENCE",

                    "action":
                        "Preserve alert, process, and network evidence.",

                    "execution_mode":
                        "PREPARE_ONLY",

                    "authorization_required":
                        True,

                    "destructive":
                        False,
                },
                {
                    "action_id":
                        "ANALYST_REVIEW",

                    "action":
                        "Require analyst authorization before response.",

                    "execution_mode":
                        "ANALYST_REVIEW",

                    "authorization_required":
                        True,

                    "destructive":
                        False,
                },
                {
                    "action_id":
                        "PREPARE_CONTAINMENT",

                    "action":
                        "Prepare endpoint containment workflow.",

                    "execution_mode":
                        "PREPARE_ONLY",

                    "authorization_required":
                        True,

                    "destructive":
                        True,
                },
            ]
        )

    if "T1070.004" in technique_set:

        actions.append(
            {
                "action_id":
                    "FILE_DELETION_INVESTIGATION",

                "action":
                    "Prepare investigation of file-deletion activity.",

                "execution_mode":
                    "PREPARE_ONLY",

                "authorization_required":
                    True,

                "destructive":
                    False,
            }
        )

    if "T1059.001" in technique_set:

        actions.append(
            {
                "action_id":
                    "POWERSHELL_INVESTIGATION",

                "action":
                    "Prepare PowerShell command-line and "
                    "process-tree investigation.",

                "execution_mode":
                    "PREPARE_ONLY",

                "authorization_required":
                    True,

                "destructive":
                    False,
            }
        )

    if "T1087" in technique_set:

        actions.append(
            {
                "action_id":
                    "ACCOUNT_DISCOVERY_INVESTIGATION",

                "action":
                    "Prepare investigation of "
                    "account-discovery activity.",

                "execution_mode":
                    "PREPARE_ONLY",

                "authorization_required":
                    True,

                "destructive":
                    False,
            }
        )

    return actions


def build_plan(data):

    source = data.get(
        "source",
        {}
    )

    if not isinstance(source, dict):
        source = {}

    agent = source.get(
        "agent",
        {}
    )

    if not isinstance(agent, dict):
        agent = {}

    risk = get_risk(
        data
    )

    detections = get_detections(
        data
    )

    detected = [
        item
        for item in detections
        if item["detected"]
    ]

    techniques = [
        item["technique"]
        for item in detected
    ]

    evidence_plan = build_evidence_plan(
        risk["severity"],
        techniques,
    )

    actions = build_response_actions(
        risk["severity"],
        techniques,
    )

    authorization_required = (
        risk["authorization_required"]
        or any(
            action["authorization_required"]
            for action in actions
        )
    )

    return {
        "schema":
            "SmartSOC-ResponsePlan-v1",

        "timestamp_utc":
            datetime.now(timezone.utc).isoformat(),

        "mode":
            "READ_ONLY_RESPONSE_PLANNING",

        "source": {
            "type":
                source.get(
                    "type"
                ),

            "agent": {
                "id":
                    agent.get("id"),

                "name":
                    agent.get("name"),

                "ip":
                    agent.get("ip"),
            },

            "event_id":
                source.get("event_id"),

            "rule_id":
                source.get("rule_id"),

            "rule_level":
                source.get("rule_level"),

            "provider":
                source.get("provider"),

            "channel":
                source.get("channel"),
        },

        "risk": {
            "risk_score":
                risk["risk_score"],

            "severity":
                risk["severity"],

            "recommended_action":
                risk["recommended_action"],

            "detections":
                detected,

            "technique_count":
                len(techniques),

            "detected_techniques":
                techniques,
        },

        "response_policy": {
            "authorization_required":
                authorization_required,

            "automatic_execution":
                False,

            "automatic_containment":
                False,

            "analyst_authorization_obtained":
                False,

            "rollback_required_if_executed":
                True,
        },

        "evidence_collection_plan":
            evidence_plan,

        "proposed_actions":
            actions,

        "execution": {
            "execution_allowed":
                False,

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
        },

        "rollback": {
            "required_if_action_executed":
                True,

            "rollback_prepared":
                False,

            "rollback_executed":
                False,
        },

        "audit": {
            "model_modified":
                False,

            "production_artifacts_modified":
                False,

            "risk_engine_modified":
                False,

            "response_execution_performed":
                False,

            "analyst_authorization_obtained":
                False,
        },

        "safety": {
            "read_only":
                True,

            "no_endpoint_changes":
                True,

            "no_external_execution":
                True,
        },

        "research_note":
            "This artifact is a response plan only. "
            "No response action is executed. "
            "Execution requires separate analyst "
            "authorization and a future controlled "
            "response-execution module.",
    }


def main():

    print("=" * 72)
    print("SmartSOC RESPONSE PLANNER v1")
    print("=" * 72)

    print()
    print("[SAFETY]")
    print("  Read-only planning          : TRUE")
    print("  Automatic execution         : FALSE")
    print("  Automatic containment       : FALSE")
    print("  Shuffle calls               : FALSE")
    print("  Caldera calls               : FALSE")
    print("  Endpoint modification       : FALSE")
    print("  Production model modified   : FALSE")

    print()
    print("[1/4] Loading Risk Engine result...")

    data = load_risk_result()

    print(
        f"[PASS] Input: {RISK_INPUT}"
    )

    print()
    print("[2/4] Reading Risk Engine result...")

    risk = get_risk(
        data
    )

    detections = get_detections(
        data
    )

    detected = [
        item
        for item in detections
        if item["detected"]
    ]

    print(
        f"[PASS] Risk score : "
        f"{risk['risk_score']}"
    )

    print(
        f"[PASS] Severity   : "
        f"{risk['severity']}"
    )

    print(
        f"[PASS] Detections : "
        f"{len(detected)}"
    )

    for item in detected:

        print(
            f"  {item['technique']} "
            f"confidence={item['confidence']} "
            f"threshold={item['threshold']}"
        )

    print()
    print("[3/4] Building response plan...")

    plan = build_plan(
        data
    )

    print(
        f"[PASS] Techniques : "
        f"{plan['risk']['technique_count']}"
    )

    print(
        f"[PASS] Actions    : "
        f"{len(plan['proposed_actions'])}"
    )

    print()
    print("[4/4] Proposed actions:")

    for action in plan["proposed_actions"]:

        print(
            f"  [{action['action_id']}] "
            f"{action['execution_mode']}"
        )

        print(
            f"      {action['action']}"
        )

        print(
            f"      authorization_required="
            f"{action['authorization_required']}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            plan,
            handle,
            indent=2,
        )

    print()
    print(
        f"[PASS] {OUTPUT_PATH}"
    )

    print()
    print(
        "[PASS] Response Planner completed."
    )

    print()
    print(
        "IMPORTANT: No response action was executed."
    )


if __name__ == "__main__":
    main()