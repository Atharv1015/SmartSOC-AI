from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


VERSION = "4.1"

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models" / "windows_apt_v4_1" / "production"
DATA_DIR = ROOT / "data" / "final" / "windows_apt_v4_1"

PHASE67 = ROOT / "src" / "analysis" / "phase6_gate6_7_offline_end_to_end_simulation_v4_1.py"

REPORT_DIR = ROOT / "reports" / "supervisor"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TXT_REPORT = REPORT_DIR / "smartsoc_attack_mitigation_profiles_v4_1.txt"
JSON_REPORT = REPORT_DIR / "smartsoc_attack_mitigation_profiles_v4_1.json"


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any) -> float | None:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def technique_id(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    m = re.search(r"T\d{4}(?:\.\d{3})?", text, re.I)
    return m.group(0).upper() if m else text


def technique_name(value: Any) -> str:
    text = normalize_text(value)

    # Common representation:
    # T1059 — Command and Scripting Interpreter
    # T1059: Command and Scripting Interpreter
    m = re.match(r"^(T\d{4}(?:\.\d{3})?)\s*[-:—]\s*(.+)$", text, re.I)
    if m:
        return m.group(2).strip()

    return text


def first_present(obj: dict, names: list[str], default=None):
    for name in names:
        if name in obj and obj[name] not in (None, ""):
            return obj[name]
    return default


# ----------------------------------------------------------------------
# Locate actual Phase 6.7 evidence
# ----------------------------------------------------------------------

def find_phase67_report() -> Path | None:
    candidates = [
        ROOT / "reports" / "phase6" / "gate6_7"
        / "phase6_gate6_7_offline_end_to_end_report.json",
        ROOT / "reports" / "phase6" / "gate6_7",
    ]

    for path in candidates:
        if path.is_file():
            return path

        if path.is_dir():
            files = sorted(path.glob("*.json"))
            for f in files:
                try:
                    data = read_json(f)
                    text = json.dumps(data).lower()

                    if (
                        "20 events" in text
                        or "events" in data
                        or "event_results" in data
                        or "event_outputs" in data
                    ):
                        return f
                except Exception:
                    continue

    return None


# ----------------------------------------------------------------------
# Extract real event-level evidence from Phase 6.7
# ----------------------------------------------------------------------

def recursively_collect_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from recursively_collect_dicts(value)

    elif isinstance(obj, list):
        for item in obj:
            yield from recursively_collect_dicts(item)


def looks_like_event_prediction(d: dict) -> bool:
    keys = {str(k).lower() for k in d.keys()}

    prediction_keys = {
        "prediction",
        "predicted_class",
        "predicted_label",
        "technique",
        "technique_id",
        "attack",
        "attack_id",
        "class",
        "model_class",
    }

    score_keys = {
        "severity_score",
        "risk_score",
        "score",
        "probability",
        "confidence",
        "max_probability",
        "risk",
    }

    return bool(keys & prediction_keys) and bool(keys & score_keys)


def extract_event_records(report: dict) -> list[dict]:
    records = []

    for d in recursively_collect_dicts(report):
        if looks_like_event_prediction(d):
            records.append(d)

    # Remove duplicate dictionary records.
    unique = []
    seen = set()

    for d in records:
        try:
            marker = json.dumps(d, sort_keys=True, default=str)
        except Exception:
            marker = repr(d)

        if marker not in seen:
            seen.add(marker)
            unique.append(d)

    return unique


# ----------------------------------------------------------------------
# Model artifact inspection
# ----------------------------------------------------------------------

def load_model_classes() -> list[str]:
    import joblib

    model_path = MODEL_DIR / "model.joblib"
    model = joblib.load(model_path)

    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "estimator"):
        classes = getattr(model.estimator, "classes_", None)

    if classes is None:
        raise RuntimeError(
            "Production model does not expose classes_. "
            "Cannot truthfully construct attack profiles."
        )

    return [normalize_text(x) for x in classes]


def load_thresholds() -> dict:
    path = MODEL_DIR / "thresholds.json"

    if not path.exists():
        return {}

    return read_json(path)


def load_feature_schema() -> dict:
    path = MODEL_DIR / "feature_schema.json"

    if not path.exists():
        return {}

    return read_json(path)


# ----------------------------------------------------------------------
# Extract policy semantics from validated Phase 6 components
# ----------------------------------------------------------------------

def import_policy_modules():
    sys.path.insert(0, str(ROOT))

    from src.decision.smartsoc_response_planner_v4_1 import (
        SmartSOCResponsePlanner,
    )

    from src.decision.smartsoc_analyst_authorization_v4_1 import (
        SmartSOCAnalystAuthorization,
    )

    return SmartSOCResponsePlanner, SmartSOCAnalystAuthorization


def discover_policy_constants(module) -> dict:
    result = {}

    for name in dir(module):
        if name.startswith("_"):
            continue

        try:
            value = getattr(module, name)

            if isinstance(value, (dict, list, tuple, str, int, float, bool)):
                result[name] = value

        except Exception:
            pass

    return result


# ----------------------------------------------------------------------
# Real risk / severity extraction
# ----------------------------------------------------------------------

def extract_real_value(
    event: dict,
    names: list[str],
) -> Any:
    for key in names:
        if key in event:
            return event[key]

    # Also search one level of nested objects.
    for value in event.values():
        if isinstance(value, dict):
            for key in names:
                if key in value:
                    return value[key]

    return None


def resolve_risk(event: dict) -> tuple[float | None, str]:
    score = extract_real_value(
        event,
        [
            "severity_score",
            "risk_score",
            "riskScore",
            "score",
            "risk",
        ],
    )

    level = extract_real_value(
        event,
        [
            "severity_level",
            "risk_level",
            "riskLevel",
            "risk_disposition",
            "risk",
        ],
    )

    score_value = safe_float(score)

    level_text = normalize_text(level).upper()

    if level_text in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
        return score_value, level_text

    # If the actual event exposes only score, use the project's
    # validated four-level boundaries.
    #
    # These are the policy boundaries already validated in Phase 5.5.
    if score_value is not None:
        if score_value >= 9.0:
            return score_value, "CRITICAL"
        if score_value >= 7.0:
            return score_value, "HIGH"
        if score_value >= 4.0:
            return score_value, "MEDIUM"
        return score_value, "LOW"

    return None, "NOT EXPOSED"


# ----------------------------------------------------------------------
# Mitigation mapping
#
# IMPORTANT:
# These are POLICY ACTIONS, not executed actions.
# ----------------------------------------------------------------------

def mitigation_for_level(level: str) -> dict:
    level = normalize_text(level).upper()

    actions = {
        "isolate_host": False,
        "kill_malicious_process": False,
        "block_ip_domain": False,
        "disable_account": False,
        "quarantine_file": False,
        "block_hash": False,
        "network_containment": False,
        "log_enrichment": False,
        "increase_monitoring": False,
        "alert_analyst": False,
        "log_only": False,
    }

    if level == "LOW":
        actions.update(
            log_enrichment=True,
            increase_monitoring=True,
            alert_analyst=False,
            log_only=True,
        )

        category = "LOW-RISK MONITORING"

    elif level == "MEDIUM":
        actions.update(
            log_enrichment=True,
            increase_monitoring=True,
            alert_analyst=True,
            log_only=False,
        )

        category = "ANALYST REVIEW"

    elif level == "HIGH":
        actions.update(
            isolate_host=True,
            kill_malicious_process=True,
            block_ip_domain=True,
            quarantine_file=True,
            block_hash=True,
            network_containment=True,
            log_enrichment=True,
            increase_monitoring=True,
            alert_analyst=True,
        )

        category = "HIGH-RISK RESPONSE"

    elif level == "CRITICAL":
        actions.update(
            isolate_host=True,
            kill_malicious_process=True,
            block_ip_domain=True,
            disable_account=True,
            quarantine_file=True,
            block_hash=True,
            network_containment=True,
            log_enrichment=True,
            increase_monitoring=True,
            alert_analyst=True,
        )

        category = "CRITICAL RESPONSE"

    else:
        actions.update(
            alert_analyst=True,
            log_only=True,
        )

        category = "EVIDENCE-LIMITED SAFETY WORKFLOW"

    return {
        "response_category": category,
        "actions": actions,
        "execution": "NOT EXECUTED",
        "containment_execution": "NOT EXECUTED",
        "analyst_approval": (
            "REQUIRED"
            if level in {"MEDIUM", "HIGH", "CRITICAL", "NOT EXPOSED"}
            else "NOT REQUIRED"
        ),
    }


# ----------------------------------------------------------------------
# Attack class resolution
# ----------------------------------------------------------------------

def resolve_attack_fields(event: dict) -> tuple[str, str]:
    raw = extract_real_value(
        event,
        [
            "technique_id",
            "attack_id",
            "predicted_class",
            "predicted_label",
            "prediction",
            "technique",
            "attack",
            "class",
            "model_class",
        ],
    )

    raw_text = normalize_text(raw)

    tid = technique_id(raw_text)

    if not tid:
        # Sometimes technique ID and name are separate.
        tid = technique_id(
            extract_real_value(
                event,
                ["technique_id", "attack_id", "id"],
            )
        )

    name = extract_real_value(
        event,
        [
            "technique_name",
            "attack_name",
            "predicted_name",
            "label_name",
            "technique",
            "attack",
        ],
    )

    name_text = technique_name(normalize_text(name))

    if not name_text and raw_text:
        name_text = technique_name(raw_text)

    return tid or raw_text or "UNKNOWN", name_text or "UNKNOWN"


# ----------------------------------------------------------------------
# Build profiles from actual evidence
# ----------------------------------------------------------------------

def build_profiles(
    model_classes: list[str],
    phase67_events: list[dict],
) -> list[dict]:

    by_class: dict[str, list[dict]] = defaultdict(list)

    for event in phase67_events:
        attack_id, attack_name = resolve_attack_fields(event)

        score, level = resolve_risk(event)

        by_class[attack_id].append(
            {
                "event": event,
                "attack_id": attack_id,
                "attack_name": attack_name,
                "score": score,
                "level": level,
            }
        )

    profiles = []

    for class_name in model_classes:

        cid = technique_id(class_name)

        candidates = []

        if class_name in by_class:
            candidates.extend(by_class[class_name])

        if cid in by_class:
            candidates.extend(by_class[cid])

        # Search by normalized technique ID.
        if cid:
            for key, values in by_class.items():
                if technique_id(key) == cid:
                    candidates.extend(values)

        # Deduplicate candidate events.
        dedup = []
        seen = set()

        for c in candidates:
            marker = id(c["event"])

            if marker not in seen:
                seen.add(marker)
                dedup.append(c)

        candidates = dedup

        real_scores = [
            x["score"]
            for x in candidates
            if x["score"] is not None
        ]

        levels = [
            x["level"]
            for x in candidates
            if x["level"] != "NOT EXPOSED"
        ]

        if candidates:
            # Use the actual observed evidence.
            best = max(
                candidates,
                key=lambda x: (
                    x["score"] is not None,
                    x["score"] if x["score"] is not None else -1,
                ),
            )

            score = (
                max(real_scores)
                if real_scores
                else None
            )

            level = (
                levels[0]
                if levels
                else "NOT EXPOSED"
            )

            attack_name = best["attack_name"]

            evidence_count = len(candidates)

        else:
            score = None
            level = "NOT OBSERVED IN PHASE 6.7 EVIDENCE"
            attack_name = technique_name(class_name) or class_name
            evidence_count = 0

        mitigation = mitigation_for_level(level)

        profiles.append(
            {
                "attack_id": cid or class_name,
                "attack": attack_name,
                "model_class": class_name,
                "severity_score": score,
                "severity_level": level,
                "evidence_events": evidence_count,
                "response_category": mitigation["response_category"],
                "analyst_review": mitigation["analyst_approval"],
                "automatic_response": "DISABLED",
                "automatic_containment": "DISABLED",
                "planned_mitigation_actions": mitigation["actions"],
                "execution_status": {
                    "host_isolation": "NOT EXECUTED",
                    "malicious_process_kill": "NOT EXECUTED",
                    "ip_domain_block": "NOT EXECUTED",
                    "account_disable": "NOT EXECUTED",
                    "file_quarantine": "NOT EXECUTED",
                    "hash_block": "NOT EXECUTED",
                    "network_containment": "NOT EXECUTED",
                },
            }
        )

    return profiles


# ----------------------------------------------------------------------
# Professional TXT report
# ----------------------------------------------------------------------

def check(actions: dict, key: str) -> str:
    return "[✓]" if actions.get(key, False) else "[ ]"


def write_txt(
    profiles: list[dict],
    model_classes: list[str],
    event_count: int,
    phase67_path: Path,
):
    with TXT_REPORT.open("w", encoding="utf-8") as f:

        f.write("=" * 72 + "\n")
        f.write("SMARTSOC AI — ATTACK & MITIGATION PROFILE REPORT\n")
        f.write("=" * 72 + "\n\n")

        f.write(f"Version                 : {VERSION}\n")
        f.write("Evidence source         : FROZEN PRODUCTION ARTIFACTS\n")
        f.write(f"Production classes      : {len(model_classes)}\n")
        f.write(f"Phase 6.7 evidence      : {event_count} event records\n")
        f.write(f"Phase 6.7 report        : {phase67_path}\n")
        f.write("Synthetic predictions   : DISABLED\n")
        f.write("Synthetic scores        : DISABLED\n")
        f.write("Automatic execution     : DISABLED\n")
        f.write("Automatic containment   : DISABLED\n\n")

        f.write("IMPORTANT\n")
        f.write("---------\n")
        f.write(
            "Severity scores and classifications shown below are only "
            "included when they are exposed by actual production "
            "inference evidence. No missing value is invented.\n\n"
        )

        f.write(
            "Planned Mitigation Actions are policy decisions. "
            "They describe what the validated response architecture "
            "prepares for the corresponding risk level. They do not "
            "mean that the action was executed.\n\n"
        )

        for index, p in enumerate(profiles, 1):

            a = p["planned_mitigation_actions"]

            f.write("=" * 60 + "\n")
            f.write(f"SMARTSOC AI — ATTACK PROFILE {index:02d}\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Attack ID        : {p['attack_id']}\n")
            f.write(f"Attack           : {p['attack']}\n")
            f.write(f"Model Class      : {p['model_class']}\n")
            f.write(
                f"Observed Evidence: {p['evidence_events']} event(s)\n\n"
            )

            f.write("AI ASSESSMENT\n")
            f.write("-------------\n")

            if p["severity_score"] is None:
                f.write("Severity Score   : NOT EXPOSED BY ACTUAL EVIDENCE\n")
            else:
                f.write(
                    f"Severity Score   : {p['severity_score']:.4f}\n"
                )

            f.write(f"Severity Level   : {p['severity_level']}\n")
            f.write(f"Risk Disposition : {p['severity_level']}\n\n")

            f.write("AI DECISION\n")
            f.write("-----------\n")
            f.write(
                f"Analyst Review       : {p['analyst_review']}\n"
            )
            f.write(
                f"Response Category    : {p['response_category']}\n"
            )
            f.write(
                f"Automatic Response   : {p['automatic_response']}\n"
            )
            f.write(
                f"Automatic Containment: {p['automatic_containment']}\n\n"
            )

            f.write("MITIGATION DECISION\n")
            f.write("-------------------\n")
            f.write(
                f"The validated response policy maps the "
                f"{p['severity_level']} risk disposition to "
                f"the following planned response actions.\n\n"
            )

            f.write("Planned Mitigation Actions\n")
            f.write("--------------------------\n")

            f.write(
                f"{check(a, 'isolate_host')} Isolate Host\n"
            )
            f.write(
                f"{check(a, 'kill_malicious_process')} Kill Malicious Process\n"
            )
            f.write(
                f"{check(a, 'block_ip_domain')} Block IP / Domain\n"
            )
            f.write(
                f"{check(a, 'disable_account')} Disable Account\n"
            )
            f.write(
                f"{check(a, 'quarantine_file')} Quarantine File\n"
            )
            f.write(
                f"{check(a, 'block_hash')} Block Hash\n"
            )
            f.write(
                f"{check(a, 'network_containment')} Network Containment\n"
            )
            f.write(
                f"{check(a, 'log_enrichment')} Log Enrichment\n"
            )
            f.write(
                f"{check(a, 'increase_monitoring')} Increase Monitoring\n"
            )
            f.write(
                f"{check(a, 'alert_analyst')} Alert Analyst\n"
            )
            f.write(
                f"{check(a, 'log_only')} Log Only / No Automated Action\n"
            )

            f.write("\nAUTHORIZATION\n")
            f.write("-------------\n")
            f.write(
                f"Analyst Approval      : {p['analyst_review']}\n"
            )
            f.write("Automatic Execution   : DISABLED\n")
            f.write("Automatic Containment : DISABLED\n\n")

            f.write("EXECUTION STATUS\n")
            f.write("----------------\n")

            for label, key in [
                ("Host Isolation", "host_isolation"),
                ("Process Kill", "malicious_process_kill"),
                ("IP / Domain Block", "ip_domain_block"),
                ("Account Disable", "account_disable"),
                ("File Quarantine", "file_quarantine"),
                ("Hash Block", "hash_block"),
                ("Network Containment", "network_containment"),
            ]:
                f.write(
                    f"{label:<22}: "
                    f"{p['execution_status'][key]}\n"
                )

            f.write("\n")

        f.write("=" * 72 + "\n")
        f.write("REPORT INTEGRITY\n")
        f.write("=" * 72 + "\n\n")

        f.write(
            "This report does not generate synthetic attack scores, "
            "fake predictions, or simulated execution results.\n"
        )

        f.write(
            "Any severity score shown is derived from actual evidence "
            "available from the validated production inference path.\n"
        )

        f.write(
            "Any attack class with no observed Phase 6.7 event evidence "
            "is explicitly marked as not observed rather than assigned "
            "an invented score.\n"
        )

        f.write(
            "All mitigation actions remain behind the validated "
            "analyst authorization and execution boundary.\n"
        )


# ----------------------------------------------------------------------
# JSON evidence report
# ----------------------------------------------------------------------

def write_json(
    profiles: list[dict],
    model_classes: list[str],
    event_count: int,
    phase67_path: Path,
):
    data = {
        "report": {
            "name": "SmartSOC AI Attack & Mitigation Profile Report",
            "version": VERSION,
            "synthetic_predictions": False,
            "synthetic_scores": False,
            "automatic_execution": False,
            "automatic_containment": False,
        },
        "production_artifacts": {
            "model": str(MODEL_DIR / "model.joblib"),
            "thresholds": str(MODEL_DIR / "thresholds.json"),
            "feature_schema": str(MODEL_DIR / "feature_schema.json"),
        },
        "phase67_evidence": {
            "report": str(phase67_path),
            "event_records": event_count,
        },
        "model": {
            "class_count": len(model_classes),
            "classes": model_classes,
        },
        "attack_profiles": profiles,
    }

    with JSON_REPORT.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:

    print("=" * 72)
    print("SmartSOC AI — Attack & Mitigation Profile Generator")
    print("Version 4.1")
    print("=" * 72)

    print("\n[1] Checking production artifacts...")

    required = [
        MODEL_DIR / "model.joblib",
        MODEL_DIR / "thresholds.json",
        MODEL_DIR / "feature_schema.json",
    ]

    for path in required:
        if not path.exists():
            print(f"[FAIL] Missing artifact: {path}")
            return 1

    print("[PASS] Production artifacts present")

    print("\n[2] Loading actual production model...")

    try:
        model_classes = load_model_classes()
    except Exception as exc:
        print("[FAIL] Could not inspect production model")
        print(f"       {type(exc).__name__}: {exc}")
        return 1

    print(
        f"[PASS] Production model exposes {len(model_classes)} classes"
    )

    print("\n[3] Locating Phase 6.7 evidence...")

    phase67_path = find_phase67_report()

    if phase67_path is None:
        print("[FAIL] Phase 6.7 evidence report not found")
        return 1

    print(f"[PASS] Evidence report found")
    print(f"       {phase67_path}")

    try:
        phase67 = read_json(phase67_path)
    except Exception as exc:
        print("[FAIL] Could not read Phase 6.7 report")
        print(f"       {type(exc).__name__}: {exc}")
        return 1

    print("\n[4] Extracting actual event-level inference evidence...")

    events = extract_event_records(phase67)

    print(
        f"[PASS] Actual event evidence records discovered: {len(events)}"
    )

    if not events:
        print(
            "\n[REVIEW REQUIRED] Phase 6.7 report does not expose "
            "event-level predictions/scores in a machine-readable form."
        )

        print(
            "No synthetic values will be generated."
        )

        return 2

    print("\n[5] Building attack profiles from actual evidence...")

    profiles = build_profiles(
        model_classes=model_classes,
        phase67_events=events,
    )

    observed = sum(
        1 for p in profiles
        if p["evidence_events"] > 0
    )

    scored = sum(
        1 for p in profiles
        if p["severity_score"] is not None
    )

    print(
        f"[PASS] Attack classes profiled: {len(profiles)}"
    )

    print(
        f"[PASS] Classes with actual event evidence: {observed}"
    )

    print(
        f"[PASS] Classes with actual severity score: {scored}"
    )

    print("\n[6] Writing professional supervisor report...")

    write_txt(
        profiles,
        model_classes,
        len(events),
        phase67_path,
    )

    write_json(
        profiles,
        model_classes,
        len(events),
        phase67_path,
    )

    print("[PASS] TXT report generated")
    print(f"       {TXT_REPORT}")

    print("[PASS] JSON evidence report generated")
    print(f"       {JSON_REPORT}")

    print("\n" + "=" * 72)
    print("SUPERVISOR ATTACK PROFILE REPORT COMPLETE")
    print("=" * 72)

    print(f"Production classes : {len(model_classes)}")
    print(f"Evidence events    : {len(events)}")
    print(f"Observed classes   : {observed}")
    print(f"Scored classes     : {scored}")
    print("Synthetic data     : DISABLED")
    print("Execution          : DISABLED")

    return 0


if __name__ == "__main__":
    sys.exit(main())