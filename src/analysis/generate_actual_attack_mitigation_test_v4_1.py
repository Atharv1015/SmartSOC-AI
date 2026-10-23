import json
import os
import sys
import subprocess
from pathlib import Path
from collections import defaultdict
from datetime import datetime


# ======================================================================
# SmartSOC AI
# Actual Attack & Mitigation Profile Report
# Version 4.1
#
# PURPOSE
# -------
# Build a supervisor-facing report from ACTUAL SmartSOC artifacts.
#
# IMPORTANT:
# - No synthetic predictions.
# - No synthetic severity scores.
# - No fabricated attack names.
# - No fabricated mitigation decisions.
# - No response execution.
# - No containment execution.
#
# Primary evidence source:
#   Phase 6.7 offline end-to-end report.
#
# Secondary evidence:
#   Phase 6.1 / 6.3 / 6.4 / 6.5 / 6.6 / 6.8 reports.
#
# The script deliberately refuses to manufacture values that cannot
# be recovered from the serialized production artifacts/reports.
# ======================================================================


VERSION = "4.1"

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models" / "windows_apt_v4_1" / "production"

MODEL_FILE = MODEL_DIR / "model.joblib"
THRESHOLDS_FILE = MODEL_DIR / "thresholds.json"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"

REPORT_DIR = ROOT / "reports"

PHASE_DIR = REPORT_DIR / "phase6"

OUTPUT_DIR = REPORT_DIR / "supervisor"

OUTPUT_FILE = OUTPUT_DIR / "smartsoc_actual_attack_mitigation_profiles_v4_1.txt"


PHASE_SCRIPTS = [
    ROOT / "src" / "analysis" / "phase6_gate6_1_response_policy_spec_v4_1.py",
    ROOT / "src" / "decision" / "smartsoc_response_planner_v4_1.py",
    ROOT / "src" / "analysis" / "phase6_gate6_3_playbook_mapping_v4_1.py",
    ROOT / "src" / "decision" / "smartsoc_analyst_authorization_v4_1.py",
    ROOT / "src" / "analysis" / "phase6_gate6_5_rollback_recovery_v4_1.py",
    ROOT / "src" / "analysis" / "phase6_gate6_6_adversarial_response_safety_v4_1.py",
    ROOT / "src" / "analysis" / "phase6_gate6_7_offline_end_to_end_simulation_v4_1.py",
    ROOT / "src" / "analysis" / "phase6_gate6_8_phase6_acceptance_freeze_v4_1.py",
]


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------

def norm(value):
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return None

        return value

    return value


def clean(value):
    value = norm(value)

    if value is None:
        return None

    if isinstance(value, bool):
        return "YES" if value else "NO"

    return str(value)


def first(d, keys):
    if not isinstance(d, dict):
        return None

    for key in keys:
        if key in d and norm(d[key]) is not None:
            return d[key]

    return None


def recursive_find(obj, wanted_keys):
    """
    Find the first occurrence of one of the wanted keys anywhere
    inside a JSON structure.
    """

    if isinstance(obj, dict):

        for key in wanted_keys:
            if key in obj and norm(obj[key]) is not None:
                return obj[key]

        for value in obj.values():
            result = recursive_find(value, wanted_keys)

            if result is not None:
                return result

    elif isinstance(obj, list):

        for value in obj:
            result = recursive_find(value, wanted_keys)

            if result is not None:
                return result

    return None


def recursive_collect_dicts(obj):
    """
    Return every dictionary found recursively.
    """

    result = []

    if isinstance(obj, dict):

        result.append(obj)

        for value in obj.values():
            result.extend(recursive_collect_dicts(value))

    elif isinstance(obj, list):

        for value in obj:
            result.extend(recursive_collect_dicts(value))

    return result


def load_json(path):
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def find_json_files(directory):
    if not directory.exists():
        return []

    return sorted(directory.rglob("*.json"))


# ----------------------------------------------------------------------
# Phase execution
# ----------------------------------------------------------------------

def run_phase_scripts():
    """
    Execute completed Phase 6 validation scripts.

    Their stdout is intentionally not treated as authoritative data.
    The generated JSON reports are the evidence source.
    """

    print()
    print("=" * 72)
    print("RUNNING COMPLETED PHASE 6 VALIDATION COMPONENTS")
    print("=" * 72)

    results = []

    for script in PHASE_SCRIPTS:

        if not script.exists():
            print(f"[SKIP] Missing component: {script}")
            results.append({
                "script": str(script),
                "status": "MISSING",
            })
            continue

        print()
        print(f"[RUN ] {script}")

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=900,
            )

            if completed.returncode == 0:
                print("[PASS] Component completed")
                status = "PASS"
            else:
                print(
                    "[REVIEW] Component returned non-zero exit code:",
                    completed.returncode,
                )
                status = "REVIEW_REQUIRED"

            results.append({
                "script": str(script),
                "status": status,
                "returncode": completed.returncode,
            })

        except Exception as exc:
            print("[REVIEW] Could not execute component")
            print("        ", repr(exc))

            results.append({
                "script": str(script),
                "status": "ERROR",
                "error": repr(exc),
            })

    return results


# ----------------------------------------------------------------------
# Model inspection
# ----------------------------------------------------------------------

def load_model_classes():
    """
    Read the actual serialized production model.

    The class list is not invented.
    """

    if not MODEL_FILE.exists():
        return [], None

    try:
        import joblib

        model = joblib.load(MODEL_FILE)

    except Exception as exc:
        print("[REVIEW] Production model could not be loaded")
        print("        ", repr(exc))
        return [], None

    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "estimator"):
        classes = getattr(model.estimator, "classes_", None)

    if classes is None:
        return [], model

    output = []

    for item in classes:

        if hasattr(item, "item"):
            try:
                item = item.item()
            except Exception:
                pass

        output.append(str(item))

    return output, model


# ----------------------------------------------------------------------
# Report discovery
# ----------------------------------------------------------------------

def discover_phase_reports():
    reports = {}

    for path in find_json_files(PHASE_DIR):

        data = load_json(path)

        if data is None:
            continue

        reports[str(path)] = data

    return reports


def get_report_by_name(reports, fragment):

    fragment = fragment.lower()

    for path, data in reports.items():

        if fragment in Path(path).name.lower():
            return data

    return None


# ----------------------------------------------------------------------
# Extract actual event-level evidence
# ----------------------------------------------------------------------

def identify_event_records(obj):
    """
    Search the Phase 6.7 report for dictionaries that look like actual
    event inference records.

    This is deliberately conservative.
    """

    candidates = []

    for item in recursive_collect_dicts(obj):

        keys = {str(k).lower() for k in item.keys()}

        class_keys = {
            "predicted_class",
            "prediction",
            "predicted_technique",
            "technique",
            "attack_id",
            "class",
        }

        score_keys = {
            "score",
            "probability",
            "confidence",
            "model_score",
            "severity_score",
        }

        if keys.intersection(class_keys) and keys.intersection(score_keys):
            candidates.append(item)

    return candidates


def extract_actual_profiles(phase67):
    """
    Aggregate actual Phase 6.7 evidence by attack class.

    No synthetic values are generated.
    """

    records = identify_event_records(phase67)

    profiles = defaultdict(list)

    for record in records:

        attack_id = first(
            record,
            [
                "predicted_class",
                "predicted_technique",
                "technique",
                "attack_id",
                "class",
                "prediction",
            ],
        )

        score = first(
            record,
            [
                "severity_score",
                "model_score",
                "score",
                "probability",
                "confidence",
            ],
        )

        if attack_id is None:
            continue

        profiles[str(attack_id)].append({
            "record": record,
            "score": score,
        })

    return profiles


# ----------------------------------------------------------------------
# Actual policy extraction
# ----------------------------------------------------------------------

def find_policy_record(reports, attack_id):
    """
    Search all Phase 6 reports for a record explicitly associated with
    the actual attack/technique identifier.

    No generic four-level policy is assigned here.
    """

    attack_id = str(attack_id).lower()

    matches = []

    for path, report in reports.items():

        for record in recursive_collect_dicts(report):

            text = json.dumps(
                record,
                ensure_ascii=False,
                default=str,
            ).lower()

            if attack_id in text:
                matches.append(record)

    if not matches:
        return None

    # Prefer the most policy-like record.
    policy_keys = [
        "risk_disposition",
        "response_category",
        "analyst_review",
        "containment_candidate",
        "planned_actions",
        "mitigation",
        "authorization_state",
    ]

    for record in matches:

        keys = {str(k).lower() for k in record.keys()}

        if len(keys.intersection(policy_keys)) >= 2:
            return record

    return matches[0]


# ----------------------------------------------------------------------
# Action extraction
# ----------------------------------------------------------------------

ACTION_DEFINITIONS = [
    (
        "Isolate Host",
        [
            "isolate_host",
            "host_isolation",
            "isolation",
        ],
    ),
    (
        "Kill Malicious Process",
        [
            "kill_process",
            "process_kill",
            "terminate_process",
        ],
    ),
    (
        "Block IP / Domain",
        [
            "block_ip",
            "block_domain",
            "ip_domain_block",
            "network_block",
        ],
    ),
    (
        "Disable Account",
        [
            "disable_account",
            "account_disable",
        ],
    ),
    (
        "Quarantine File",
        [
            "quarantine_file",
            "file_quarantine",
        ],
    ),
    (
        "Block Hash",
        [
            "block_hash",
            "hash_block",
        ],
    ),
    (
        "Network Containment",
        [
            "network_containment",
            "containment",
        ],
    ),
    (
        "Log Enrichment",
        [
            "log_enrichment",
            "enrichment",
        ],
    ),
    (
        "Increase Monitoring",
        [
            "increase_monitoring",
            "enhanced_monitoring",
            "monitoring",
        ],
    ),
    (
        "Alert Analyst",
        [
            "alert_analyst",
            "analyst_alert",
        ],
    ),
    (
        "Log Only",
        [
            "log_only",
            "no_automated_action",
        ],
    ),
]


def extract_action_value(record, keywords):

    if not isinstance(record, dict):
        return None

    lower_map = {
        str(k).lower(): v
        for k, v in record.items()
    }

    for keyword in keywords:

        if keyword in lower_map:
            return lower_map[keyword]

    return None


def action_status(record, keywords):

    value = extract_action_value(record, keywords)

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, str):

        v = value.strip().lower()

        if v in {
            "true",
            "yes",
            "required",
            "approved",
            "planned",
            "enabled",
        }:
            return True

        if v in {
            "false",
            "no",
            "disabled",
            "not_required",
            "not planned",
        }:
            return False

    return None


# ----------------------------------------------------------------------
# Attack name resolution
# ----------------------------------------------------------------------

MITRE_NAMES = {
    "T1059": "Command and Scripting Interpreter",
    "T1003": "OS Credential Dumping",
    "T1003.001": "LSASS Memory",
    "T1055": "Process Injection",
    "T1071": "Application Layer Protocol",
    "T1071.001": "Web Protocols",
    "T1071.004": "DNS",
    "T1082": "System Information Discovery",
    "T1083": "File and Directory Discovery",
    "T1057": "Process Discovery",
    "T1105": "Ingress Tool Transfer",
    "T1547": "Boot or Logon Autostart Execution",
    "T1547.001": "Registry Run Keys / Startup Folder",
    "T1562": "Impair Defenses",
    "T1562.001": "Disable or Modify Tools",
    "T1486": "Data Encrypted for Impact",
    "T1047": "Windows Management Instrumentation",
    "T1021": "Remote Services",
    "T1021.001": "Remote Desktop Protocol",
    "T1027": "Obfuscated Files or Information",
    "T1053": "Scheduled Task/Job",
    "T1053.005": "Scheduled Task",
    "T1566": "Phishing",
    "T1566.001": "Spearphishing Attachment",
    "T1566.002": "Spearphishing Link",
}


def attack_name(attack_id):

    attack_id = str(attack_id)

    if attack_id in MITRE_NAMES:
        return MITRE_NAMES[attack_id]

    return "Name not exposed by local artifact"


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------

def checkmark(value):

    if value is True:
        return "[âœ“]"

    if value is False:
        return "[ ]"

    return "[?]"


def actual_or_unavailable(value):

    if value is None:
        return "NOT AVAILABLE IN PRODUCTION EVIDENCE"

    return clean(value)


def build_profile(
    index,
    attack_id,
    actual_records,
    reports,
):
    """
    Build one profile strictly from evidence.
    """

    profile = {}

    representative = None

    if actual_records:

        # Prefer the record with an explicit model score.
        for candidate in actual_records:

            if candidate.get("score") is not None:
                representative = candidate
                break

        if representative is None:
            representative = actual_records[0]

    if representative is None:
        representative_record = {}
        model_score = None
    else:
        representative_record = representative["record"]
        model_score = representative["score"]

    policy_record = find_policy_record(
        reports,
        attack_id,
    )

    if policy_record is None:
        policy_record = {}

    predicted_class = first(
        representative_record,
        [
            "predicted_class",
            "predicted_technique",
            "technique",
            "attack_id",
            "class",
            "prediction",
        ],
    )

    severity_score = first(
        representative_record,
        [
            "severity_score",
            "model_score",
            "score",
            "confidence",
            "probability",
        ],
    )

    if severity_score is None:
        severity_score = model_score

    severity_level = first(
        representative_record,
        [
            "severity_level",
            "risk_level",
            "risk",
            "severity",
        ],
    )

    if severity_level is None:
        severity_level = first(
            policy_record,
            [
                "severity_level",
                "risk_level",
                "risk",
                "severity",
            ],
        )

    risk_disposition = first(
        representative_record,
        [
            "risk_disposition",
            "risk_decision",
            "disposition",
        ],
    )

    if risk_disposition is None:
        risk_disposition = first(
            policy_record,
            [
                "risk_disposition",
                "risk_decision",
                "disposition",
            ],
        )

    analyst_review = first(
        representative_record,
        [
            "analyst_review",
            "requires_analyst",
            "analyst_required",
            "review_required",
        ],
    )

    if analyst_review is None:
        analyst_review = first(
            policy_record,
            [
                "analyst_review",
                "requires_analyst",
                "analyst_required",
                "review_required",
            ],
        )

    response_category = first(
        representative_record,
        [
            "response_category",
            "response",
            "response_type",
        ],
    )

    if response_category is None:
        response_category = first(
            policy_record,
            [
                "response_category",
                "response",
                "response_type",
            ],
        )

    containment_candidate = first(
        representative_record,
        [
            "containment_candidate",
            "containment_required",
            "requires_containment",
        ],
    )

    if containment_candidate is None:
        containment_candidate = first(
            policy_record,
            [
                "containment_candidate",
                "containment_required",
                "requires_containment",
            ],
        )

    rollback = first(
        representative_record,
        [
            "rollback",
            "rollback_required",
        ],
    )

    if rollback is None:
        rollback = first(
            policy_record,
            [
                "rollback",
                "rollback_required",
            ],
        )

    profile["index"] = index
    profile["attack_id"] = str(
        predicted_class
        if predicted_class is not None
        else attack_id
    )
    profile["attack_name"] = attack_name(profile["attack_id"])

    profile["model_score"] = severity_score
    profile["severity_level"] = severity_level
    profile["risk_disposition"] = risk_disposition

    profile["analyst_review"] = analyst_review
    profile["response_category"] = response_category
    profile["containment_candidate"] = containment_candidate
    profile["rollback"] = rollback

    profile["policy_record"] = policy_record
    profile["representative_record"] = representative_record

    profile["actions"] = {}

    for action_name, keywords in ACTION_DEFINITIONS:

        value = action_status(
            representative_record,
            keywords,
        )

        if value is None:
            value = action_status(
                policy_record,
                keywords,
            )

        profile["actions"][action_name] = value

    return profile


def render_profile(profile):

    lines = []

    lines.append("=" * 64)
    lines.append(
        f"ATTACK PROFILE {profile['index']:02d} / 45"
    )
    lines.append("=" * 64)
    lines.append("")

    lines.append(
        f"Attack ID        : {profile['attack_id']}"
    )

    lines.append(
        f"Attack           : {profile['attack_name']}"
    )

    lines.append("")
    lines.append("AI ASSESSMENT")
    lines.append("-------------")

    lines.append(
        f"Predicted Class       : "
        f"{actual_or_unavailable(profile['attack_id'])}"
    )

    lines.append(
        f"Model Score           : "
        f"{actual_or_unavailable(profile['model_score'])}"
    )

    lines.append(
        f"Severity Level        : "
        f"{actual_or_unavailable(profile['severity_level'])}"
    )

    lines.append(
        f"Risk Disposition      : "
        f"{actual_or_unavailable(profile['risk_disposition'])}"
    )

    lines.append("")
    lines.append("DECISION POLICY")
    lines.append("---------------")

    lines.append(
        f"Analyst Review        : "
        f"{actual_or_unavailable(profile['analyst_review'])}"
    )

    lines.append(
        f"Response Required     : "
        f"{actual_or_unavailable(profile['response_category'])}"
    )

    lines.append(
        f"Containment Candidate : "
        f"{actual_or_unavailable(profile['containment_candidate'])}"
    )

    lines.append(
        "Automatic Execution   : DISABLED"
    )

    lines.append(
        "Automatic Containment : DISABLED"
    )

    lines.append("")
    lines.append("RESPONSE PLANNER")
    lines.append("----------------")

    lines.append(
        f"Response Category     : "
        f"{actual_or_unavailable(profile['response_category'])}"
    )

    investigation_value = first(
        profile["representative_record"],
        [
            "investigation",
            "investigation_required",
            "requires_investigation",
        ],
    )

    lines.append(
        f"Investigation         : "
        f"{actual_or_unavailable(investigation_value)}"
    )

    lines.append(
        f"Rollback              : "
        f"{actual_or_unavailable(profile['rollback'])}"
    )

    lines.append("")
    lines.append("PLANNED MITIGATION")
    lines.append("------------------")

    for action_name in [
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
        "Log Only",
    ]:

        lines.append(
            f"{checkmark(profile['actions'][action_name])} "
            f"{action_name}"
        )

    lines.append("")
    lines.append("AUTHORIZATION")
    lines.append("-------------")

    authorization = first(
        profile["representative_record"],
        [
            "authorization_state",
            "authorization",
            "analyst_authorization",
        ],
    )

    lines.append(
        f"Authorization State   : "
        f"{actual_or_unavailable(authorization)}"
    )

    lines.append(
        f"Analyst Approval      : "
        f"{actual_or_unavailable(profile['analyst_review'])}"
    )

    lines.append(
        "AI Authorization      : NOT SUFFICIENT"
    )

    lines.append(
        "Automatic Execution   : DISABLED"
    )

    lines.append("")
    lines.append("EXECUTION STATE")
    lines.append("---------------")

    lines.append(
        "Host Isolation        : NOT EXECUTED"
    )

    lines.append(
        "Process Kill          : NOT EXECUTED"
    )

    lines.append(
        "IP / Domain Block     : NOT EXECUTED"
    )

    lines.append(
        "Account Disable       : NOT EXECUTED"
    )

    lines.append(
        "File Quarantine       : NOT EXECUTED"
    )

    lines.append(
        "Hash Blocking         : NOT EXECUTED"
    )

    lines.append(
        "Network Containment   : NOT EXECUTED"
    )

    lines.append("")
    lines.append("EVIDENCE")
    lines.append("--------")

    lines.append(
        "Inference              : ACTUAL PRODUCTION MODEL"
    )

    lines.append(
        "Score                  : "
        + (
            "ACTUAL MODEL OUTPUT"
            if profile["model_score"] is not None
            else "NOT AVAILABLE IN PRODUCTION EVIDENCE"
        )
    )

    lines.append(
        "Risk                   : "
        + (
            "ACTUAL POLICY OUTPUT"
            if profile["risk_disposition"] is not None
            else "NOT AVAILABLE IN PRODUCTION EVIDENCE"
        )
    )

    lines.append(
        "Response               : "
        + (
            "ACTUAL RESPONSE-PLANNER OUTPUT"
            if profile["response_category"] is not None
            else "NOT AVAILABLE IN PRODUCTION EVIDENCE"
        )
    )

    lines.append(
        "Authorization          : "
        + (
            "ACTUAL AUTHORIZATION OUTPUT"
            if authorization is not None
            else "NOT AVAILABLE IN PRODUCTION EVIDENCE"
        )
    )

    lines.append(
        "Synthetic Data         : NONE"
    )

    lines.append("")

    return lines


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    print("=" * 72)
    print("SmartSOC AI â€” Actual Attack & Mitigation Profile Generator")
    print(f"Version {VERSION}")
    print("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # 1. Execute completed validation components
    # --------------------------------------------------------------

    phase_execution = run_phase_scripts()

    # --------------------------------------------------------------
    # 2. Load production model
    # --------------------------------------------------------------

    print()
    print("[1] Loading actual production model...")

    classes, model = load_model_classes()

    if not classes:

        print("[FAIL] Could not obtain actual production model classes")
        return 2

    print("[PASS] Production model loaded")
    print(
        f"       Actual model classes : {len(classes)}"
    )

    # --------------------------------------------------------------
    # 3. Load phase reports
    # --------------------------------------------------------------

    print()
    print("[2] Loading generated Phase 6 evidence...")

    reports = discover_phase_reports()

    print(
        f"[PASS] JSON evidence reports discovered : "
        f"{len(reports)}"
    )

    phase67 = get_report_by_name(
        reports,
        "phase6_gate6_7_offline_end_to_end_report",
    )

    if phase67 is None:

        print(
            "[FAIL] Phase 6.7 evidence report was not found"
        )

        return 3

    print(
        "[PASS] Phase 6.7 evidence report loaded"
    )

    # --------------------------------------------------------------
    # 4. Extract actual event evidence
    # --------------------------------------------------------------

    print()
    print("[3] Extracting actual model/decision evidence...")

    actual_profiles = extract_actual_profiles(
        phase67
    )

    print(
        f"[PASS] Attack classes with explicit event evidence : "
        f"{len(actual_profiles)}"
    )

    # --------------------------------------------------------------
    # 5. Build profiles
    # --------------------------------------------------------------

    print()
    print("[4] Building attack profiles from actual evidence...")

    profiles = []

    for index, attack_id in enumerate(
        classes,
        start=1,
    ):

        records = actual_profiles.get(
            str(attack_id),
            [],
        )

        profile = build_profile(
            index,
            str(attack_id),
            records,
            reports,
        )

        profiles.append(profile)

    print(
        f"[PASS] Profiles generated : {len(profiles)}"
    )

    # --------------------------------------------------------------
    # 6. Generate professional TXT report
    # --------------------------------------------------------------

    print()
    print("[5] Writing supervisor report...")

    lines = []

    lines.append("=" * 64)
    lines.append(
        "SMARTSOC AI â€” ATTACK & MITIGATION PROFILE REPORT"
    )
    lines.append("=" * 64)
    lines.append("")

    lines.append(
        f"Model Version        : {VERSION}"
    )

    lines.append(
        f"Model Classes        : {len(classes)}"
    )

    lines.append(
        "Inference Source     : Frozen Production Model"
    )

    lines.append(
        "Dataset              : Frozen Test Dataset"
    )

    lines.append(
        "Synthetic Scores     : DISABLED"
    )

    lines.append(
        "Synthetic Predictions: DISABLED"
    )

    lines.append(
        "Execution            : DISABLED"
    )

    lines.append(
        "Containment          : DISABLED"
    )

    lines.append("")
    lines.append(
        "Generated             : "
        + datetime.now().isoformat(
            timespec="seconds"
        )
    )

    lines.append("")
    lines.append("=" * 64)
    lines.append("EVIDENCE POLICY")
    lines.append("=" * 64)
    lines.append("")

    lines.append(
        "Every model class shown below comes directly from the"
    )

    lines.append(
        "serialized SmartSOC production model."
    )

    lines.append("")
    lines.append(
        "Scores, severity, risk, response and authorization values"
    )

    lines.append(
        "are included only when recoverable from actual production"
    )

    lines.append(
        "inference/policy evidence."
    )

    lines.append("")
    lines.append(
        "The report NEVER invents missing values."
    )

    lines.append("")
    lines.append(
        "NOT AVAILABLE IN PRODUCTION EVIDENCE means the current"
    )

    lines.append(
        "serialized/report artifacts do not expose that value."
    )

    lines.append("")
    lines.append(
        "Execution State is deliberately reported as NOT EXECUTED."
    )

    lines.append(
        "The validated SmartSOC Phase 6 architecture does not"
    )

    lines.append(
        "execute containment or response automatically."
    )

    lines.append("")

    # --------------------------------------------------------------
    # 7. Profiles
    # --------------------------------------------------------------

    for profile in profiles:
        lines.extend(
            render_profile(profile)
        )

    # --------------------------------------------------------------
    # 8. Final evidence summary
    # --------------------------------------------------------------

    explicit_scores = sum(
        1
        for profile in profiles
        if profile["model_score"] is not None
    )

    explicit_risk = sum(
        1
        for profile in profiles
        if profile["risk_disposition"] is not None
    )

    explicit_response = sum(
        1
        for profile in profiles
        if profile["response_category"] is not None
    )

    lines.append("=" * 64)
    lines.append(
        "REPORT EVIDENCE SUMMARY"
    )
    lines.append("=" * 64)
    lines.append("")

    lines.append(
        f"Production model classes       : {len(classes)}"
    )

    lines.append(
        f"Profiles generated             : {len(profiles)}"
    )

    lines.append(
        f"Profiles with actual scores    : {explicit_scores}"
    )

    lines.append(
        f"Profiles with actual risk      : {explicit_risk}"
    )

    lines.append(
        f"Profiles with actual response  : {explicit_response}"
    )

    lines.append(
        "Synthetic scores               : 0"
    )

    lines.append(
        "Synthetic predictions          : 0"
    )

    lines.append(
        "Response execution             : 0"
    )

    lines.append(
        "Containment execution          : 0"
    )

    lines.append("")
    lines.append(
        "PHASE VALIDATION STATUS"
    )
    lines.append(
        "------------------------"
    )

    for item in phase_execution:

        lines.append(
            f"{item['status']:18s} "
            f"{Path(item['script']).name}"
        )

    lines.append("")
    lines.append("=" * 64)
    lines.append(
        "END OF SMARTSOC ACTUAL ATTACK & MITIGATION PROFILE REPORT"
    )
    lines.append("=" * 64)

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:

        f.write(
            "\n".join(lines)
        )

    print()
    print("=" * 72)
    print("REPORT COMPLETE")
    print("=" * 72)

    print(
        f"Actual model classes : {len(classes)}"
    )

    print(
        f"Profiles generated   : {len(profiles)}"
    )

    print(
        f"Actual scores found  : {explicit_scores}"
    )

    print(
        f"Actual risk found    : {explicit_risk}"
    )

    print(
        f"Actual responses     : {explicit_response}"
    )

    print(
        "Synthetic data       : NONE"
    )

    print(
        "Execution            : DISABLED"
    )

    print()
    print(
        f"TXT report : {OUTPUT_FILE}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
