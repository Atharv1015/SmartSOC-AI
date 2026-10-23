import argparse
import json
from pathlib import Path
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "reports"
    / "integration"
    / "wazuh_readonly_result.json"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "integration"
    / "risk_engine_result.json"
)


# Transparent policy weights.
CONFIDENCE_WEIGHT = 50.0
RULE_LEVEL_WEIGHT = 10.0
TECHNIQUE_COUNT_WEIGHT = 5.0
THRESHOLD_MARGIN_WEIGHT = 20.0


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(maximum, value))


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_inference(data):
    inference = data.get("inference")

    if not isinstance(inference, dict):
        raise ValueError(
            "Expected 'inference' object in SmartSOC result."
        )

    return inference


def get_predictions(inference):
    predictions = inference.get("predictions", [])

    if not isinstance(predictions, list):
        raise ValueError(
            "Expected 'inference.predictions' to be a list."
        )

    normalized = []

    for item in predictions:

        if not isinstance(item, dict):
            continue

        technique = item.get("technique")

        if not technique:
            continue

        try:
            confidence = float(item.get("confidence"))
        except (TypeError, ValueError):
            confidence = None

        try:
            threshold = float(item.get("threshold"))
        except (TypeError, ValueError):
            threshold = None

        normalized.append(
            {
                "technique": technique,
                "confidence": confidence,
                "threshold": threshold,
            }
        )

    return normalized


def confidence_component(predictions):

    values = [
        item["confidence"]
        for item in predictions
        if item["confidence"] is not None
    ]

    if not values:
        return 0.0, None

    highest = max(values)

    return (
        clamp(highest * CONFIDENCE_WEIGHT),
        highest,
    )


def rule_level_component(rule_level):

    try:
        level = float(rule_level)
    except (TypeError, ValueError):
        return 0.0

    # Wazuh rule levels are normally 0-15.
    normalized = clamp(level / 15.0, 0.0, 1.0)

    return normalized * RULE_LEVEL_WEIGHT


def technique_count_component(count):

    if count <= 0:
        return 0.0

    return min(float(count), TECHNIQUE_COUNT_WEIGHT)


def threshold_margin_component(predictions):

    margins = []

    for item in predictions:

        confidence = item["confidence"]
        threshold = item["threshold"]

        if confidence is None or threshold is None:
            continue

        margins.append(
            max(0.0, confidence - threshold)
        )

    if not margins:
        return 0.0, None

    strongest_margin = max(margins)

    # 0.50 threshold margin = full 20 points.
    normalized = min(
        strongest_margin / 0.50,
        1.0,
    )

    return (
        normalized * THRESHOLD_MARGIN_WEIGHT,
        strongest_margin,
    )


def classify_risk(score):

    if score >= 80:
        return "CRITICAL"

    if score >= 60:
        return "HIGH"

    if score >= 40:
        return "MEDIUM"

    if score >= 20:
        return "LOW"

    return "INFO"


def choose_response_policy(severity):

    if severity == "CRITICAL":
        return {
            "recommended_action":
                "ANALYST_REVIEW_AND_PREPARE_CONTAINMENT",
            "authorization_required": True,
        }

    if severity == "HIGH":
        return {
            "recommended_action":
                "ANALYST_REVIEW_AND_PREPARE_RESPONSE",
            "authorization_required": True,
        }

    if severity == "MEDIUM":
        return {
            "recommended_action":
                "ANALYST_REVIEW_AND_COLLECT_EVIDENCE",
            "authorization_required": True,
        }

    if severity == "LOW":
        return {
            "recommended_action":
                "MONITOR_AND_COLLECT_EVIDENCE",
            "authorization_required": False,
        }

    return {
        "recommended_action":
            "LOG_AND_MONITOR",
        "authorization_required": False,
    }


def build_result(raw_data):

    inference = get_inference(raw_data)

    predictions = get_predictions(inference)

    event_id = inference.get("event_id")
    rule_id = inference.get("rule_id")
    rule_level = inference.get("rule_level")
    provider = raw_data.get("provider")
    channel = raw_data.get("channel")

    confidence_score, highest_confidence = (
        confidence_component(predictions)
    )

    rule_score = rule_level_component(rule_level)

    technique_score = technique_count_component(
        len(predictions)
    )

    margin_score, strongest_margin = (
        threshold_margin_component(predictions)
    )

    raw_score = (
        confidence_score
        + rule_score
        + technique_score
        + margin_score
    )

    risk_score = round(
        clamp(raw_score),
        3,
    )

    severity = classify_risk(risk_score)

    policy = choose_response_policy(severity)

    return {
        "schema": "SmartSOC-RiskEngine-v1",

        "timestamp_utc":
            datetime.now(timezone.utc).isoformat(),

        "mode":
            "READ_ONLY_RISK_ASSESSMENT",

        "source": {
            "type":
                raw_data.get("source", {}).get("type"),

            "agent":
                raw_data.get("source", {}).get("agent"),

            "event_id":
                event_id,

            "rule_id":
                rule_id,

            "rule_level":
                rule_level,

            "provider":
                provider,

            "channel":
                channel,
        },

        "detections": {
            "techniques": [
                {
                    "technique":
                        item["technique"],

                    "confidence":
                        item["confidence"],

                    "threshold":
                        item["threshold"],
                }
                for item in predictions
            ],

            "technique_count":
                len(predictions),

            "highest_model_confidence":
                (
                    round(highest_confidence, 6)
                    if highest_confidence is not None
                    else None
                ),

            "strongest_threshold_margin":
                (
                    round(strongest_margin, 6)
                    if strongest_margin is not None
                    else None
                ),
        },

        "risk_calculation": {
            "confidence_component":
                round(confidence_score, 3),

            "wazuh_rule_level_component":
                round(rule_score, 3),

            "technique_count_component":
                round(technique_score, 3),

            "threshold_margin_component":
                round(margin_score, 3),

            "risk_score":
                risk_score,

            "severity":
                severity,
        },

        "decision": {
            "recommended_action":
                policy["recommended_action"],

            "authorization_required":
                policy["authorization_required"],

            "automatic_execution":
                False,

            "automatic_containment":
                False,

            "rollback_required_if_executed":
                True,

            "shuffle_called":
                False,

            "caldera_called":
                False,

            "endpoint_modified":
                False,
        },

        "audit": {
            "model_modified":
                False,

            "production_artifacts_modified":
                False,

            "automatic_response_performed":
                False,

            "analyst_authorization_obtained":
                False,
        },
    }


def main():

    parser = argparse.ArgumentParser(
        description="SmartSOC v1 read-only risk engine"
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("=" * 72)
    print("SmartSOC v1 READ-ONLY RISK ENGINE")
    print("=" * 72)

    print()
    print("[SAFETY]")
    print("  Mode                  : READ_ONLY_RISK_ASSESSMENT")
    print("  Automatic execution   : FALSE")
    print("  Automatic containment : FALSE")
    print("  Shuffle calls         : FALSE")
    print("  Caldera calls         : FALSE")
    print("  Endpoint modification: FALSE")

    print()
    print("[1/3] Loading inference result...")

    raw_data = load_json(input_path)

    print(f"[PASS] Loaded: {input_path}")

    print()
    print("[2/3] Calculating risk...")

    result = build_result(raw_data)

    print()
    print("-" * 72)
    print("RISK ASSESSMENT")
    print("-" * 72)

    print(
        f"Event ID       : "
        f"{result['source']['event_id']}"
    )

    print(
        f"Rule ID        : "
        f"{result['source']['rule_id']}"
    )

    print(
        f"Wazuh level    : "
        f"{result['source']['rule_level']}"
    )

    print()
    print("Detected techniques:")

    for item in result["detections"]["techniques"]:

        print(
            f"  {item['technique']} "
            f"confidence={item['confidence']:.6f} "
            f"threshold={item['threshold']:.6f}"
        )

    print()
    print(
        f"Highest confidence : "
        f"{result['detections']['highest_model_confidence']}"
    )

    print(
        f"Risk score         : "
        f"{result['risk_calculation']['risk_score']}/100"
    )

    print(
        f"Severity           : "
        f"{result['risk_calculation']['severity']}"
    )

    print()
    print("Risk components:")

    for key, value in result["risk_calculation"].items():

        if key.endswith("_component"):
            print(f"  {key}: {value}")

    print()
    print("Decision:")

    print(
        f"  Recommended action : "
        f"{result['decision']['recommended_action']}"
    )

    print(
        f"  Authorization      : "
        f"{result['decision']['authorization_required']}"
    )

    print(
        f"  Automatic execution: "
        f"{result['decision']['automatic_execution']}"
    )

    print(
        f"  Automatic containment: "
        f"{result['decision']['automatic_containment']}"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            result,
            handle,
            indent=2,
        )

    print()
    print("[3/3] Writing risk assessment...")

    print(
        "[PASS] Result written to:"
    )

    print(
        f"       {output_path}"
    )

    print()
    print(
        "[PASS] Read-only Risk Engine completed."
    )


if __name__ == "__main__":
    main()