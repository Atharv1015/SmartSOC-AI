import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd


# ============================================================================
# SmartSOC AI
# Supervisor Attack Coverage & Mitigation Decision Report
# Version 4.1
#
# IMPORTANT:
#   - Uses frozen production artifacts.
#   - Uses actual frozen test-set predictions when available.
#   - Does NOT generate synthetic attack scores.
#   - Does NOT invent severity.
#   - Does NOT execute response actions.
#   - Does NOT modify model/data artifacts.
# ============================================================================


ROOT = Path(__file__).resolve().parents[2]

PROD = ROOT / "models" / "windows_apt_v4_1" / "production"
DATA = ROOT / "data" / "final" / "windows_apt_v4_1"
REPORT_DIR = ROOT / "reports" / "supervisor"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TXT = REPORT_DIR / "smartsoc_attack_decision_report_v4_1.txt"


MODEL_FILE = PROD / "model.joblib"
ENCODER_FILE = PROD / "multilabel_encoder.joblib"
THRESHOLD_FILE = PROD / "thresholds.json"
SCHEMA_FILE = PROD / "feature_schema.json"

STRUCTURED_ENCODER_FILE = PROD / "structured_encoder.joblib"
SCALER_FILE = PROD / "numerical_scaler.joblib"
TFIDF_FILE = PROD / "tfidf_vectorizer.joblib"

STRUCTURED_TEST = DATA / "structured_test.csv"
TEXT_TEST = DATA / "behavior_text_test.csv"
LABEL_TEST = DATA / "labels_test.csv"
METADATA_TEST = DATA / "metadata_test.csv"


# ============================================================================
# OUTPUT
# ============================================================================

lines = []


def p(text=""):
    print(text)
    lines.append(str(text))


def h(text):
    p()
    p("=" * 78)
    p(text)
    p("=" * 78)


def s(text):
    p()
    p("-" * 78)
    p(text)
    p("-" * 78)


# ============================================================================
# HELPERS
# ============================================================================

def clean(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def safe_float(value, default=0.0):
    try:
        value = float(value)
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


def parse_labels(value):
    text = clean(value)

    if not text:
        return []

    try:
        import ast

        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            return [
                str(x).strip()
                for x in parsed
                if str(x).strip()
            ]

    except Exception:
        pass

    return [
        x.strip()
        for x in text.split(",")
        if x.strip()
    ]


def extract_threshold(thresholds, technique):
    """
    Resolve the actual production threshold for a technique.

    Supports common v4.1 JSON layouts without inventing values.
    """

    if not isinstance(thresholds, dict):
        return None

    candidates = []

    if "thresholds" in thresholds:
        candidates.append(thresholds["thresholds"])

    candidates.append(thresholds)

    for mapping in candidates:

        if not isinstance(mapping, dict):
            continue

        if technique in mapping:

            value = mapping[technique]

            if isinstance(value, dict):

                for key in (
                    "threshold",
                    "value",
                    "decision_threshold",
                ):
                    if key in value:
                        return safe_float(
                            value[key],
                            None,
                        )

            return safe_float(
                value,
                None,
            )

    return None


def risk_from_score(score):
    """
    IMPORTANT:
    This function is ONLY used if the production decision policy
    exposes a normalized numeric risk score.

    It does not create a classifier confidence score.
    """

    if score is None:
        return None

    score = safe_float(score, None)

    if score is None:
        return None

    if score < 0.25:
        return "LOW"

    if score < 0.50:
        return "MEDIUM"

    if score < 0.75:
        return "HIGH"

    return "CRITICAL"


def technique_name(technique):
    names = {
        "T1003.001": "OS Credential Dumping: LSASS Memory",
        "T1003.002": "OS Credential Dumping: Security Account Manager",
        "T1021.006": "Windows Remote Management",
        "T1027": "Obfuscated / Compressed Files and Information",
        "T1027.004": "Compile After Delivery",
        "T1036": "Masquerading",
        "T1047": "Windows Management Instrumentation",
        "T1053.005": "Scheduled Task / Job: Scheduled Task",
        "T1055": "Process Injection",
        "T1059": "Command and Scripting Interpreter",
        "T1059.001": "PowerShell",
        "T1059.003": "Windows Command Shell",
        "T1059.005": "Visual Basic",
        "T1070": "Indicator Removal",
        "T1070.004": "File Deletion",
        "T1078": "Valid Accounts",
        "T1087": "Account Discovery",
        "T1095": "Non-Application Layer Protocol",
        "T1098": "Account Manipulation",
        "T1105": "Ingress Tool Transfer",
        "T1110": "Brute Force",
        "T1110.001": "Password Guessing",
        "T1112": "Modify Registry",
        "T1136": "Create Account",
        "T1210": "Exploitation of Remote Services",
        "T1218": "System Binary Proxy Execution",
        "T1484": "Domain or Tenant Policy Modification",
        "T1485": "Data Destruction",
        "T1489": "Service Stop",
        "T1529": "System Shutdown / Reboot",
        "T1531": "Account Access Removal",
        "T1543.003": "Windows Service",
        "T1546": "Event Triggered Execution",
        "T1546.003": "Windows Management Instrumentation Event Subscription",
        "T1546.011": "Application Shimming",
        "T1547.001": "Registry Run Keys / Startup Folder",
        "T1548": "Abuse Elevation Control Mechanism",
        "T1548.002": "Bypass User Account Control",
        "T1562.001": "Impair Defenses: Disable or Modify Tools",
        "T1562.004": "Impair Defenses: Disable or Modify System Firewall",
        "T1565.001": "Data Manipulation",
        "T1570": "Lateral Tool Transfer",
        "T1574.001": "DLL Search Order Hijacking",
        "T1574.002": "DLL Side-Loading",
        "T1574.010": "Services File Permissions Weakness",
    }

    return names.get(
        technique,
        "MITRE ATT&CK technique",
    )


# ============================================================================
# RESPONSE POLICY
# ============================================================================

def response_policy(risk_level, fallback=False, ambiguous=False):
    """
    Mirrors the validated Phase 6 response boundary.

    These are PLANNED actions only.
    Nothing is executed here.
    """

    if fallback or ambiguous:
        return {
            "response": "Evidence-limited safety workflow",
            "investigation": "REQUIRED",
            "analyst": "REQUIRED",
            "rollback": "WHERE APPLICABLE",
            "actions": [
                "Log Enrichment",
                "Increase Monitoring",
                "Alert Analyst",
                "Log Only / No Automated Action",
            ],
        }

    if risk_level == "LOW":
        return {
            "response": "Low-risk monitoring workflow",
            "investigation": "OPTIONAL / AS DIRECTED",
            "analyst": "NOT REQUIRED",
            "rollback": "NOT APPLICABLE",
            "actions": [
                "Log Enrichment",
                "Increase Monitoring",
                "Log Only / No Automated Action",
            ],
        }

    if risk_level == "MEDIUM":
        return {
            "response": "Analyst-review workflow",
            "investigation": "REQUIRED",
            "analyst": "REQUIRED",
            "rollback": "WHERE APPLICABLE",
            "actions": [
                "Log Enrichment",
                "Increase Monitoring",
                "Alert Analyst",
                "Log Only / No Automated Action",
            ],
        }

    if risk_level == "HIGH":
        return {
            "response": "High-priority response workflow",
            "investigation": "REQUIRED",
            "analyst": "REQUIRED",
            "rollback": "REQUIRED WHERE APPLICABLE",
            "actions": [
                "Isolate Host",
                "Kill Malicious Process",
                "Block IP / Domain",
                "Quarantine File",
                "Block Hash",
                "Network Containment",
                "Log Enrichment",
                "Increase Monitoring",
                "Alert Analyst",
            ],
        }

    if risk_level == "CRITICAL":
        return {
            "response": "Critical-response workflow",
            "investigation": "REQUIRED",
            "analyst": "REQUIRED",
            "rollback": "REQUIRED WHERE APPLICABLE",
            "actions": [
                "Isolate Host",
                "Kill Malicious Process",
                "Block IP / Domain",
                "Disable Account",
                "Quarantine File",
                "Block Hash",
                "Network Containment",
                "Log Enrichment",
                "Increase Monitoring",
                "Alert Analyst",
            ],
        }

    return {
        "response": "Evidence-limited safety workflow",
        "investigation": "REQUIRED",
        "analyst": "REQUIRED",
        "rollback": "WHERE APPLICABLE",
        "actions": [
            "Log Enrichment",
            "Increase Monitoring",
            "Alert Analyst",
            "Log Only / No Automated Action",
        ],
    }


# ============================================================================
# LOAD
# ============================================================================

h("SMARTSOC AI — ATTACK COVERAGE & MITIGATION DECISION REPORT")

p("Version                 : 4.1")
p("Source                  : Frozen production AI artifacts")
p(f"Production model        : {MODEL_FILE}")
p("Synthetic predictions   : DISABLED")
p("Synthetic scores        : DISABLED")
p("Response execution      : DISABLED")
p("Containment execution   : DISABLED")

h("1. PRODUCTION ARTIFACT VALIDATION")

required = [
    MODEL_FILE,
    ENCODER_FILE,
    THRESHOLD_FILE,
    SCHEMA_FILE,
]

for path in required:
    if not path.exists():
        raise FileNotFoundError(
            f"Required production artifact missing:\n{path}"
        )

    p(f"[PASS] {path.name}")

model = joblib.load(MODEL_FILE)
label_encoder = joblib.load(ENCODER_FILE)

with open(
    THRESHOLD_FILE,
    "r",
    encoding="utf-8",
) as f:
    thresholds = json.load(f)

with open(
    SCHEMA_FILE,
    "r",
    encoding="utf-8",
) as f:
    schema = json.load(f)

classes = [
    str(x)
    for x in label_encoder.classes_
]

p()
p(f"Production model class : {type(model).__name__}")
p(f"Production classes     : {len(classes)}")
p(f"Frozen technique list  : {len(classes)}")

h("2. FROZEN MODEL ATTACK COVERAGE")

p("The following techniques are the actual classes exposed")
p("by the serialized production model.")
p()
p("No additional attack classes are invented.")

for index, technique in enumerate(classes, 1):

    threshold = extract_threshold(
        thresholds,
        technique,
    )

    p(
        f"{index:02d}. "
        f"{technique:<12} "
        f"{technique_name(technique):<65} "
        f"threshold="
        f"{threshold:.6f}"
        if threshold is not None
        else
        f"{index:02d}. "
        f"{technique:<12} "
        f"{technique_name(technique):<65} "
        f"threshold=NOT_EXPOSED"
    )


# ============================================================================
# ACTUAL PREDICTION SOURCE
# ============================================================================

h("3. ACTUAL PREDICTION EVIDENCE")

prediction_file_candidates = [
    ROOT / "results" / "windows_apt_v4_1" / "predictions.csv",
    ROOT / "results" / "windows_apt_v4_1" / "corrected" / "predictions.csv",
]

prediction_file = None

for candidate in prediction_file_candidates:
    if candidate.exists():
        prediction_file = candidate
        break

if prediction_file is not None:

    p(
        f"[PASS] Existing production prediction artifact found:"
    )
    p(
        f"       {prediction_file}"
    )

    predictions = pd.read_csv(
        prediction_file
    )

    p(
        f"[PASS] Actual prediction rows loaded: "
        f"{len(predictions)}"
    )

else:

    predictions = None

    p(
        "[REVIEW] No compatible saved production prediction "
        "artifact was found."
    )

    p(
        "No synthetic prediction will be generated."
    )


# ============================================================================
# LABEL SUPPORT
# ============================================================================

actual_labels = {}

if LABEL_TEST.exists():

    labels_df = pd.read_csv(
        LABEL_TEST
    )

    if "techniques" in labels_df.columns:

        counts = {}

        for value in labels_df["techniques"]:

            for technique in parse_labels(value):

                counts[technique] = (
                    counts.get(technique, 0) + 1
                )

        actual_labels = counts


# ============================================================================
# BUILD ATTACK-LEVEL REPORT
# ============================================================================

h("4. ATTACK-LEVEL AI DECISION SUMMARY")

p(
    "Each section below corresponds to an actual model class."
)

p(
    "Where actual model prediction evidence exists, the report"
)

p(
    "uses that evidence. It does not fabricate confidence values."
)


# ============================================================================
# DETERMINE ACTUAL DETECTIONS PER TECHNIQUE
# ============================================================================

detected_by_technique = {}

if predictions is not None:

    # Case 1:
    # prediction file contains one row per event and prediction columns.
    for technique in classes:

        if technique in predictions.columns:

            values = pd.to_numeric(
                predictions[technique],
                errors="coerce",
            ).fillna(0.0)

            threshold = extract_threshold(
                thresholds,
                technique,
            )

            if threshold is not None:

                detected_values = values[
                    values >= threshold
                ]

                detected_by_technique[technique] = {
                    "count": int(
                        len(detected_values)
                    ),
                    "max_confidence": (
                        float(
                            detected_values.max()
                        )
                        if len(detected_values)
                        else None
                    ),
                    "threshold": threshold,
                }

# ============================================================================
# TECHNIQUE REPORTS
# ============================================================================

for number, technique in enumerate(
    classes,
    1,
):

    threshold = extract_threshold(
        thresholds,
        technique,
    )

    support = actual_labels.get(
        technique,
        0,
    )

    detection = detected_by_technique.get(
        technique
    )

    h(
        f"ATTACK PROFILE {number:02d} — {technique}"
    )

    p(
        f"MITRE Technique ID     : {technique}"
    )

    p(
        f"Attack / Technique     : "
        f"{technique_name(technique)}"
    )

    p(
        f"Frozen Test Support    : "
        f"{support}"
    )

    p(
        f"Production Threshold   : "
        f"{threshold:.6f}"
        if threshold is not None
        else
        "Production Threshold   : NOT EXPOSED"
    )

    s("ACTUAL AI MODEL EVIDENCE")

    if detection is not None:

        p(
            f"Actual detected events : "
            f"{detection['count']}"
        )

        if detection["max_confidence"] is not None:

            p(
                f"Highest actual score   : "
                f"{detection['max_confidence']:.6f}"
            )

        p(
            "Prediction source      : "
            "SAVED PRODUCTION PREDICTION ARTIFACT"
        )

    else:

        p(
            "Actual event confidence: "
            "NOT AVAILABLE IN CURRENT SAVED PREDICTION ARTIFACT"
        )

        p(
            "Synthetic confidence   : DISABLED"
        )

        p(
            "Synthetic severity     : DISABLED"
        )

    s("SEVERITY / RISK DECISION")

    p(
        "Risk score             : "
        "REQUIRES ACTUAL PHASE 5 EVENT DECISION"
    )

    p(
        "Risk level             : "
        "REQUIRES ACTUAL PHASE 5 EVENT DECISION"
    )

    p(
        "Important              : "
        "MITRE confidence is NOT the same as risk severity."
    )

    s("MITIGATION DECISION")

    p(
        "The validated Phase 6 planner maps the actual"
    )

    p(
        "Phase 5 risk disposition to a controlled response."
    )

    p(
        "No response action is executed by this report."
    )

    p()
    p(
        "Possible validated response tiers:"
    )

    p(
        "  LOW      -> monitoring / logging"
    )

    p(
        "  MEDIUM   -> analyst review / investigation"
    )

    p(
        "  HIGH     -> high-priority response planning"
    )

    p(
        "  CRITICAL -> critical response planning"
    )

    s("EXECUTION BOUNDARY")

    p(
        "Isolate Host          : NOT EXECUTED"
    )

    p(
        "Kill Malicious Process: NOT EXECUTED"
    )

    p(
        "Block IP / Domain     : NOT EXECUTED"
    )

    p(
        "Disable Account       : NOT EXECUTED"
    )

    p(
        "Quarantine File       : NOT EXECUTED"
    )

    p(
        "Block Hash            : NOT EXECUTED"
    )

    p(
        "Network Containment   : NOT EXECUTED"
    )

    p(
        "Log Enrichment        : PLANNED BY RESPONSE POLICY"
    )

    p(
        "Increase Monitoring   : PLANNED BY RESPONSE POLICY"
    )

    p(
        "Alert Analyst         : PLANNED BY RESPONSE POLICY"
    )

    p(
        "Log Only              : USED BY SAFE LOW/EVIDENCE-LIMITED PATHS"
    )

    p(
        "Automatic execution   : DISABLED"
    )

    p(
        "Automatic containment : DISABLED"
    )


# ============================================================================
# ACTUAL CONTROLLED EXAMPLE
# ============================================================================

h("5. ACTUAL CONTROLLED INFERENCE EXAMPLE")

p(
    "This section is included only when a saved real production"
)

p(
    "inference record exists in the project artifacts."
)

example_found = False

# Look for a saved JSON inference record.
json_candidates = list(
    (
        ROOT / "results"
    ).rglob(
        "*.json"
    )
)

for candidate in json_candidates:

    try:

        with open(
            candidate,
            "r",
            encoding="utf-8",
        ) as f:
            obj = json.load(f)

        if not isinstance(obj, dict):
            continue

        preds = obj.get("predictions")

        if not isinstance(preds, list):
            continue

        if not preds:
            continue

        p()
        p(
            f"Actual inference artifact: {candidate}"
        )

        for item in preds:

            technique = clean(
                item.get("technique")
            )

            confidence = item.get(
                "confidence"
            )

            threshold = item.get(
                "threshold"
            )

            if not technique:
                continue

            p()
            p(
                f"Technique              : "
                f"{technique}"
            )

            p(
                f"Attack                 : "
                f"{technique_name(technique)}"
            )

            if confidence is not None:

                p(
                    f"AI confidence score   : "
                    f"{safe_float(confidence):.6f}"
                )

            else:

                p(
                    "AI confidence score   : "
                    "NOT EXPOSED"
                )

            if threshold is not None:

                p(
                    f"Detection threshold   : "
                    f"{safe_float(threshold):.6f}"
                )

            else:

                p(
                    "Detection threshold   : "
                    "NOT EXPOSED"
                )

            if (
                confidence is not None
                and threshold is not None
            ):

                detected = (
                    safe_float(confidence)
                    >= safe_float(threshold)
                )

                p(
                    f"AI detection decision : "
                    f"{'DETECTED' if detected else 'NOT DETECTED'}"
                )

            p()
            p(
                "Severity score        : "
                "NOT derived from MITRE confidence"
            )

            p(
                "Severity level        : "
                "Must come from Phase 5 risk engine"
            )

            p(
                "Mitigation execution  : "
                "NOT EXECUTED"
            )

        example_found = True
        break

    except Exception:
        continue


if not example_found:

    p(
        "No saved JSON inference record with prediction objects"
    )

    p(
        "was found. No example has been fabricated."
    )


# ============================================================================
# FINAL ARCHITECTURE
# ============================================================================

h("6. ACTUAL SMARTSOC DECISION FLOW")

p(
    "1. Wazuh / endpoint event enters the analysis pipeline."
)

p(
    "2. Structured and behavioral-text features are constructed."
)

p(
    "3. Frozen v4.1 production model produces MITRE probabilities."
)

p(
    "4. Per-technique production thresholds determine detection."
)

p(
    "5. Detected techniques are passed into the validated Phase 5"
)

p(
    "   risk / uncertainty / decision policy."
)

p(
    "6. Phase 6 response planner converts the risk disposition"
)

p(
    "   into a controlled mitigation plan."
)

p(
    "7. Analyst authorization remains the execution boundary."
)

p(
    "8. Actual containment/execution remains disabled in the"
)

p(
    "   validated offline baseline."
)


# ============================================================================
# FINAL STATUS
# ============================================================================

h("7. SUPERVISOR CONCLUSION")

p(
    "Production AI model        : AVAILABLE"
)

p(
    f"MITRE attack classes       : {len(classes)}"
)

p(
    "Frozen production baseline : v4.1"
)

p(
    "Synthetic predictions      : DISABLED"
)

p(
    "Synthetic severity         : DISABLED"
)

p(
    "Automatic response         : DISABLED"
)

p(
    "Automatic containment      : DISABLED"
)

p(
    "Phase 6 response baseline  : ACCEPTED AND FROZEN"
)

p()
p(
    "This document reports actual artifact-backed model evidence."
)

p(
    "It intentionally does not convert classifier confidence"
)

p(
    "into an invented severity score."
)

p(
    "Severity must come from the validated Phase 5 decision engine."
)

p()
p(
    "Report generated:"
)

p(
    datetime.now().astimezone().isoformat()
)


# ============================================================================
# SAVE
# ============================================================================

with open(
    OUT_TXT,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "\n".join(lines)
    )

print()
print("=" * 78)
print("SUPERVISOR REPORT COMPLETE")
print("=" * 78)
print(f"MITRE classes : {len(classes)}")
print(f"TXT report    : {OUT_TXT}")
print("Synthetic data: DISABLED")
print("Execution     : DISABLED")
print("=" * 78)