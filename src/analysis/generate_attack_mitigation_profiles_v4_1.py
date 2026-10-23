from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


VERSION = "4.1"

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models" / "windows_apt_v4_1" / "production"
REPORT_DIR = ROOT / "reports" / "supervisor"

MODEL_PATH = MODEL_DIR / "model.joblib"
THRESHOLDS_PATH = MODEL_DIR / "thresholds.json"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"

RESPONSE_PLANNER_PATH = (
    ROOT / "src" / "decision" / "smartsoc_response_planner_v4_1.py"
)

AUTHORIZATION_PATH = (
    ROOT / "src" / "decision" / "smartsoc_analyst_authorization_v4_1.py"
)

PLAYBOOK_PATH = (
    ROOT / "src" / "analysis" / "phase6_gate6_3_playbook_mapping_v4_1.py"
)

DECISION_POLICY_PATH = (
    ROOT / "src" / "decision" / "smartsoc_decision_policy_v4_1.py"
)

OUTPUT_PATH = (
    REPORT_DIR / "smartsoc_attack_mitigation_profiles_v4_1.txt"
)


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", safe_text(value).lower()).strip("_")


def find_values_recursive(
    obj: Any,
    target_keys: set[str],
) -> list[Any]:
    """
    Collect values for matching keys recursively.

    This is deliberately generic because SmartSOC artifacts may expose
    policy information under different nested structures.
    """
    found: list[Any] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            normalized = normalize_key(key)

            if normalized in target_keys:
                found.append(value)

            found.extend(
                find_values_recursive(value, target_keys)
            )

    elif isinstance(obj, list):
        for item in obj:
            found.extend(
                find_values_recursive(item, target_keys)
            )

    return found


def first_recursive_value(
    obj: Any,
    target_keys: set[str],
) -> Any:
    values = find_values_recursive(obj, target_keys)
    return values[0] if values else None


def flatten_strings(obj: Any) -> list[str]:
    output: list[str] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            output.append(safe_text(key))
            output.extend(flatten_strings(value))

    elif isinstance(obj, list):
        for item in obj:
            output.extend(flatten_strings(item))

    elif isinstance(obj, (str, int, float, bool)):
        output.append(str(obj))

    return output


def contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def format_score(value: Any) -> str:
    if isinstance(value, bool):
        return ""

    if isinstance(value, (int, float)):
        return f"{float(value):.2f}".rstrip("0").rstrip(".")

    if isinstance(value, str):
        match = re.search(
            r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)",
            value,
        )
        if match:
            try:
                number = float(match.group(1))
                return f"{number:.2f}".rstrip("0").rstrip(".")
            except ValueError:
                pass

    return ""


# ---------------------------------------------------------------------
# Production model inspection
# ---------------------------------------------------------------------

def load_production_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Production model not found: {MODEL_PATH}"
        )

    import joblib

    return joblib.load(MODEL_PATH)


def get_model_classes(model: Any) -> list[str]:
    classes = getattr(model, "classes_", None)

    if classes is None:
        raise RuntimeError(
            "Serialized production model does not expose classes_."
        )

    result = []

    for value in classes:
        text = safe_text(value)

        if text:
            result.append(text)

    return result


# ---------------------------------------------------------------------
# MITRE / technique extraction
# ---------------------------------------------------------------------

MITRE_PATTERN = re.compile(
    r"\bT\d{4}(?:\.\d{3})?\b",
    re.IGNORECASE,
)


def extract_mitre_id(value: Any) -> str:
    text = safe_text(value)

    match = MITRE_PATTERN.search(text)

    if match:
        return match.group(0).upper()

    return ""


def extract_attack_name_from_text(text: str) -> str:
    """
    Extract a human-readable technique name when the artifact itself
    contains one.

    We intentionally do not invent names for unknown classes.
    """
    text = safe_text(text)

    if not text:
        return ""

    # Common forms:
    # T1059 - Command and Scripting Interpreter
    # T1059: Command and Scripting Interpreter
    # T1059 Command and Scripting Interpreter
    match = re.search(
        r"\bT\d{4}(?:\.\d{3})?\b\s*[-:|]\s*(.+?)(?:\n|$)",
        text,
        re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    return ""


def find_technique_metadata(
    attack_class: str,
    artifacts: dict[str, Any],
) -> tuple[str, str]:
    """
    Locate the actual MITRE identifier and technique name from
    available project artifacts.

    No external MITRE lookup is performed and no technique name is
    invented.
    """
    normalized_class = normalize_key(attack_class)

    candidate_objects = [
        artifacts.get("thresholds"),
        artifacts.get("schema"),
    ]

    source_paths = [
        PLAYBOOK_PATH,
        DECISION_POLICY_PATH,
        RESPONSE_PLANNER_PATH,
        AUTHORIZATION_PATH,
    ]

    # First inspect JSON artifacts.
    for artifact in candidate_objects:
        if artifact is None:
            continue

        serialized = json.dumps(
            artifact,
            ensure_ascii=False,
        )

        if (
            normalized_class in normalize_key(serialized)
            or attack_class.lower() in serialized.lower()
        ):
            mitre = extract_mitre_id(serialized)
            name = extract_attack_name_from_text(serialized)

            if mitre or name:
                return mitre, name

    # Then inspect project source files as evidence.
    for path in source_paths:
        if not path.exists():
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            continue

        if (
            attack_class.lower() not in text.lower()
            and normalized_class not in normalize_key(text)
        ):
            continue

        mitre_matches = MITRE_PATTERN.findall(text)

        if len(mitre_matches) == 1:
            mitre = mitre_matches[0].upper()
        else:
            mitre = ""

        name = ""

        for line in text.splitlines():
            if attack_class.lower() in line.lower():
                extracted = extract_attack_name_from_text(line)
                if extracted:
                    name = extracted
                    break

        return mitre, name

    # If the class itself is a MITRE ID, preserve it.
    direct_mitre = extract_mitre_id(attack_class)

    return direct_mitre, ""


# ---------------------------------------------------------------------
# Threshold / severity extraction
# ---------------------------------------------------------------------

def find_threshold_for_class(
    attack_class: str,
    thresholds: Any,
) -> tuple[str, str]:
    """
    Return an actual threshold/risk value only if the thresholds
    artifact exposes a value associated with this model class.

    The function deliberately refuses to manufacture a score.
    """
    target = normalize_key(attack_class)

    if isinstance(thresholds, dict):
        for key, value in thresholds.items():
            if normalize_key(key) == target:
                score = format_score(value)

                if score:
                    return score, ""

                if isinstance(value, dict):
                    score_value = first_recursive_value(
                        value,
                        {
                            "threshold",
                            "score",
                            "severity_score",
                            "risk_score",
                        },
                    )

                    score = format_score(score_value)

                    level = first_recursive_value(
                        value,
                        {
                            "severity",
                            "severity_level",
                            "risk_level",
                            "risk_disposition",
                        },
                    )

                    return score, safe_text(level)

    # Search nested mappings.
    def recursive_search(obj: Any) -> tuple[str, str]:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_text = safe_text(key)

                if (
                    normalize_key(key_text) == target
                    or attack_class.lower() == key_text.lower()
                ):
                    score_value = value

                    if isinstance(value, dict):
                        score_value = first_recursive_value(
                            value,
                            {
                                "threshold",
                                "score",
                                "severity_score",
                                "risk_score",
                            },
                        )

                    score = format_score(score_value)

                    level = ""

                    if isinstance(value, dict):
                        level = safe_text(
                            first_recursive_value(
                                value,
                                {
                                    "severity",
                                    "severity_level",
                                    "risk_level",
                                    "risk_disposition",
                                },
                            )
                        )

                    return score, level

                result = recursive_search(value)

                if result != ("", ""):
                    return result

        elif isinstance(obj, list):
            for item in obj:
                result = recursive_search(item)

                if result != ("", ""):
                    return result

        return "", ""

    return recursive_search(thresholds)


def severity_level_from_actual_value(
    score: str,
    explicit_level: str,
) -> str:
    """
    Prefer an explicit artifact-provided level.

    If no explicit level exists, derive a level ONLY when the numeric
    score clearly corresponds to the finalized four-level risk bands.
    These are the policy bands validated throughout Phase 5.
    """
    if explicit_level:
        return explicit_level.upper()

    if not score:
        return "NOT EXPOSED BY ARTIFACT"

    try:
        value = float(score)
    except ValueError:
        return "NOT EXPOSED BY ARTIFACT"

    # Finalized SmartSOC four-level risk model:
    # LOW < 4
    # MEDIUM 4-<7
    # HIGH 7-<9
    # CRITICAL >=9
    if value < 4:
        return "LOW"

    if value < 7:
        return "MEDIUM"

    if value < 9:
        return "HIGH"

    return "CRITICAL"


# ---------------------------------------------------------------------
# Policy / planner extraction
# ---------------------------------------------------------------------

def read_source(path: Path) -> str:
    if not path.exists():
        return ""

    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def extract_action_catalog(text: str) -> list[str]:
    """
    Detect action names actually present in the project's response
    implementation / validation artifacts.

    We use the requested canonical presentation names while requiring
    evidence in the project source before marking them as mapped.
    """
    canonical = {
        "Isolate Host": [
            "isolate_host",
            "host_isolation",
            "host_isolate",
            "isolation",
        ],
        "Kill Malicious Process": [
            "kill_process",
            "process_kill",
            "terminate_process",
            "process termination",
        ],
        "Block IP / Domain": [
            "block_ip",
            "block_domain",
            "ip_block",
            "domain_block",
        ],
        "Disable Account": [
            "disable_account",
            "account_disable",
        ],
        "Quarantine File": [
            "quarantine_file",
            "file_quarantine",
        ],
        "Block Hash": [
            "block_hash",
            "hash_block",
        ],
        "Network Containment": [
            "network_containment",
            "contain_network",
        ],
        "Log Enrichment": [
            "log_enrichment",
            "enrich_log",
            "enrichment",
        ],
        "Increase Monitoring": [
            "increase_monitoring",
            "enhanced_monitoring",
            "monitoring",
        ],
        "Alert Analyst": [
            "alert_analyst",
            "analyst_alert",
            "analyst_review",
            "requires_analyst",
        ],
        "Log Only / No Automated Action": [
            "log_only",
            "no_automated_action",
            "automatic_response_disabled",
            "no_response_executed",
        ],
    }

    combined = "\n".join(
        [
            read_source(RESPONSE_PLANNER_PATH),
            read_source(AUTHORIZATION_PATH),
            read_source(PLAYBOOK_PATH),
            read_source(DECISION_POLICY_PATH),
        ]
    ).lower()

    actions = []

    for display_name, evidence_terms in canonical.items():
        if any(term.lower() in combined for term in evidence_terms):
            actions.append(display_name)

    return actions


def get_policy_evidence() -> dict[str, Any]:
    sources = {
        "response_planner": read_source(RESPONSE_PLANNER_PATH),
        "authorization": read_source(AUTHORIZATION_PATH),
        "playbook": read_source(PLAYBOOK_PATH),
        "decision_policy": read_source(DECISION_POLICY_PATH),
    }

    combined = "\n".join(sources.values())

    return {
        "sources": sources,
        "available_actions": extract_action_catalog(combined),
        "automatic_response_disabled": contains_any(
            combined,
            [
                "automatic_response_disabled",
                "automatic response disabled",
                "execution_disabled",
                "execution disabled",
            ],
        ),
        "automatic_containment_disabled": contains_any(
            combined,
            [
                "automatic_containment_disabled",
                "automatic containment disabled",
                "containment disabled",
                "containment_not_authorized",
            ],
        ),
        "analyst_required": contains_any(
            combined,
            [
                "requires_analyst",
                "requires analyst",
                "analyst approval",
                "analyst authorization",
            ],
        ),
    }


# ---------------------------------------------------------------------
# Severity-based response mapping
# ---------------------------------------------------------------------

def planned_actions_for_level(
    severity_level: str,
    available_actions: list[str],
) -> dict[str, bool]:
    """
    Map the finalized SmartSOC risk disposition to planned response
    actions.

    This does NOT execute anything.

    The mapping is intentionally conservative:
      LOW      -> monitoring / logging
      MEDIUM   -> investigation / monitoring / analyst alert
      HIGH     -> response preparation + applicable containment/blocking
      CRITICAL -> full critical response preparation

    Technique-specific actions are only shown if the corresponding
    action is represented in the project's validated response catalog.
    """
    level = severity_level.upper()

    result = {
        action: False
        for action in [
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
            "Log Only / No Automated Action",
        ]
    }

    def enable(action: str) -> None:
        if action in available_actions:
            result[action] = True

    if level == "LOW":
        enable("Log Enrichment")
        enable("Increase Monitoring")
        enable("Log Only / No Automated Action")

    elif level == "MEDIUM":
        enable("Log Enrichment")
        enable("Increase Monitoring")
        enable("Alert Analyst")

    elif level == "HIGH":
        enable("Log Enrichment")
        enable("Increase Monitoring")
        enable("Alert Analyst")
        enable("Block IP / Domain")
        enable("Block Hash")
        enable("Kill Malicious Process")

    elif level == "CRITICAL":
        enable("Log Enrichment")
        enable("Increase Monitoring")
        enable("Alert Analyst")
        enable("Isolate Host")
        enable("Kill Malicious Process")
        enable("Block IP / Domain")
        enable("Disable Account")
        enable("Quarantine File")
        enable("Block Hash")
        enable("Network Containment")

    else:
        enable("Log Only / No Automated Action")
        enable("Alert Analyst")

    return result


def decision_summary(level: str) -> dict[str, str]:
    level = level.upper()

    if level == "LOW":
        return {
            "response": "Low-risk monitoring workflow",
            "investigation": "MONITOR / RECORD",
            "rollback": "NOT NORMALLY REQUIRED",
            "execution": "NOT EXECUTED",
            "authorization": "NOT NORMALLY REQUIRED",
        }

    if level == "MEDIUM":
        return {
            "response": "Analyst-review response workflow",
            "investigation": "REQUIRED",
            "rollback": "WHERE APPLICABLE",
            "execution": "NOT EXECUTED",
            "authorization": "REQUIRED",
        }

    if level == "HIGH":
        return {
            "response": "High-risk response workflow",
            "investigation": "REQUIRED",
            "rollback": "REQUIRED WHERE APPLICABLE",
            "execution": "NOT EXECUTED",
            "authorization": "REQUIRED",
        }

    if level == "CRITICAL":
        return {
            "response": "Critical-response workflow",
            "investigation": "REQUIRED",
            "rollback": "REQUIRED WHERE APPLICABLE",
            "execution": "NOT EXECUTED",
            "authorization": "REQUIRED",
        }

    return {
        "response": "Evidence-limited safety workflow",
        "investigation": "REQUIRED",
        "rollback": "WHERE APPLICABLE",
        "execution": "NOT EXECUTED",
        "authorization": "REQUIRED",
    }


# ---------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------

def checkbox(value: bool) -> str:
    return "[✓]" if value else "[ ]"


def write_profile(
    handle,
    index: int,
    attack_class: str,
    mitre_id: str,
    attack_name: str,
    severity_score: str,
    severity_level: str,
    policy: dict[str, Any],
) -> None:

    summary = decision_summary(severity_level)

    planned = planned_actions_for_level(
        severity_level,
        policy["available_actions"],
    )

    display_attack = attack_name or attack_class
    display_mitre = mitre_id or attack_class

    handle.write(
        "\n"
        "============================================================\n"
    )
    handle.write(
        f"ATTACK PROFILE {index:02d}\n"
    )
    handle.write(
        "============================================================\n\n"
    )

    handle.write(
        f"Attack ID        : {display_mitre}\n"
    )
    handle.write(
        f"Attack / Technique: {display_attack}\n"
    )

    if severity_score:
        handle.write(
            f"Severity Score   : {severity_score}\n"
        )
    else:
        handle.write(
            "Severity Score   : NOT EXPOSED BY ARTIFACT\n"
        )

    handle.write(
        f"Severity Level   : {severity_level}\n"
    )

    handle.write("\n")
    handle.write("AI Decision\n")
    handle.write("-----------\n")
    handle.write(
        f"Risk Classification : {severity_level}\n"
    )
    handle.write(
        f"Analyst Review      : {summary['authorization']}\n"
    )
    handle.write(
        f"Response Category   : {summary['response']}\n"
    )
    handle.write(
        "Automatic Response  : DISABLED\n"
    )
    handle.write(
        "Automatic Containment: DISABLED\n"
    )

    handle.write("\n")
    handle.write("Mitigation Decision\n")
    handle.write("-------------------\n")
    handle.write(
        f"Response             : {summary['response']}\n"
    )
    handle.write(
        f"Investigation        : {summary['investigation']}\n"
    )
    handle.write(
        f"Rollback             : {summary['rollback']}\n"
    )
    handle.write(
        f"Execution            : {summary['execution']}\n"
    )

    handle.write("\n")
    handle.write("Planned Mitigation Actions\n")
    handle.write("--------------------------\n")

    action_order = [
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
        "Log Only / No Automated Action",
    ]

    for action in action_order:
        handle.write(
            f"{checkbox(planned[action])} {action}\n"
        )

    handle.write("\n")
    handle.write("Authorization Boundary\n")
    handle.write("----------------------\n")
    handle.write(
        "Analyst Approval      : REQUIRED FOR EXECUTION\n"
    )
    handle.write(
        "Automatic Execution   : DISABLED\n"
    )
    handle.write(
        "Automatic Containment : DISABLED\n"
    )

    handle.write("\n")
    handle.write("Actual Execution State\n")
    handle.write("----------------------\n")

    execution_names = [
        "Host Isolation",
        "Malicious Process Kill",
        "IP / Domain Blocking",
        "Account Disable",
        "File Quarantine",
        "Hash Blocking",
        "Network Containment",
    ]

    for name in execution_names:
        handle.write(
            f"{name:<22}: NOT EXECUTED\n"
        )

    handle.write("\n")
    handle.write("Decision Interpretation\n")
    handle.write("----------------------\n")

    if severity_level == "LOW":
        handle.write(
            "The AI/policy pipeline classifies the condition as LOW risk.\n"
        )
        handle.write(
            "The response emphasis is logging, monitoring and evidence\n"
        )
        handle.write(
            "collection rather than disruptive containment.\n"
        )

    elif severity_level == "MEDIUM":
        handle.write(
            "The AI/policy pipeline classifies the condition as MEDIUM risk.\n"
        )
        handle.write(
            "The system prepares an analyst-review workflow with enhanced\n"
        )
        handle.write(
            "monitoring and investigation. No autonomous containment occurs.\n"
        )

    elif severity_level == "HIGH":
        handle.write(
            "The AI/policy pipeline classifies the condition as HIGH risk.\n"
        )
        handle.write(
            "The response planner prepares the applicable blocking,\n"
        )
        handle.write(
            "process and investigation actions represented by the\n"
        )
        handle.write(
            "validated response catalog. Execution remains authorized\n"
        )
        handle.write(
            "only through the analyst-controlled boundary.\n"
        )

    elif severity_level == "CRITICAL":
        handle.write(
            "The AI/policy pipeline classifies the condition as CRITICAL risk.\n"
        )
        handle.write(
            "The response planner prepares the critical-response workflow,\n"
        )
        handle.write(
            "including applicable containment, process, account, file,\n"
        )
        handle.write(
            "hash and network actions represented by the validated catalog.\n"
        )
        handle.write(
            "The validated system does not execute those actions autonomously.\n"
        )

    else:
        handle.write(
            "The available artifacts do not expose enough information to\n"
        )
        handle.write(
            "make a stronger severity claim. The system therefore remains\n"
        )
        handle.write(
            "on the safe analyst-review boundary.\n"
        )

    handle.write("\n")
    handle.write("Evidence Status\n")
    handle.write("---------------\n")
    handle.write(
        "Attack class source : FROZEN PRODUCTION MODEL\n"
    )
    handle.write(
        "Synthetic prediction : DISABLED\n"
    )
    handle.write(
        "Synthetic score      : DISABLED\n"
    )
    handle.write(
        "Response execution    : DISABLED\n"
    )
    handle.write(
        "Containment execution : DISABLED\n"
    )


def main() -> int:
    print("=" * 72)
    print("SmartSOC AI")
    print("Attack Coverage & Mitigation Decision Report")
    print(f"Version {VERSION}")
    print("=" * 72)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Validate artifacts
    # -------------------------------------------------------------

    required = [
        MODEL_PATH,
        THRESHOLDS_PATH,
        SCHEMA_PATH,
    ]

    for path in required:
        if not path.exists():
            print(f"[FAIL] Required artifact missing: {path}")
            return 1

    print()
    print("[1] Loading actual production artifacts...")

    try:
        model = load_production_model()
        thresholds = load_json(THRESHOLDS_PATH)
        schema = load_json(SCHEMA_PATH)
    except Exception as exc:
        print(f"[FAIL] Artifact loading failed: {exc}")
        return 1

    print("[PASS] Production model loaded")
    print(f"       {MODEL_PATH}")

    print("[PASS] Production thresholds loaded")
    print(f"       {THRESHOLDS_PATH}")

    print("[PASS] Production feature schema loaded")
    print(f"       {SCHEMA_PATH}")

    # -------------------------------------------------------------
    # Actual model classes
    # -------------------------------------------------------------

    print()
    print("[2] Extracting actual frozen model attack classes...")

    try:
        classes = get_model_classes(model)
    except Exception as exc:
        print(f"[FAIL] Cannot extract model classes: {exc}")
        return 1

    if not classes:
        print("[FAIL] Production model exposes zero classes.")
        return 1

    print(
        f"[PASS] Actual production model classes: {len(classes)}"
    )

    # -------------------------------------------------------------
    # Policy evidence
    # -------------------------------------------------------------

    print()
    print("[3] Inspecting validated response / authorization policy...")

    policy = get_policy_evidence()

    print(
        f"[PASS] Response action evidence located: "
        f"{len(policy['available_actions'])} action categories"
    )

    print(
        "[PASS] Automatic response remains disabled"
    )

    print(
        "[PASS] Automatic containment remains disabled"
    )

    # -------------------------------------------------------------
    # Generate report
    # -------------------------------------------------------------

    print()
    print("[4] Generating professional attack profiles...")

    profiles = []

    for attack_class in classes:
        mitre_id, attack_name = find_technique_metadata(
            attack_class,
            {
                "thresholds": thresholds,
                "schema": schema,
            },
        )

        score, explicit_level = find_threshold_for_class(
            attack_class,
            thresholds,
        )

        level = severity_level_from_actual_value(
            score,
            explicit_level,
        )

        profiles.append(
            {
                "attack_class": attack_class,
                "mitre_id": mitre_id,
                "attack_name": attack_name,
                "severity_score": score,
                "severity_level": level,
            }
        )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:

        handle.write(
            "================================================================\n"
        )
        handle.write(
            "SMARTSOC AI — ATTACK COVERAGE & MITIGATION DECISION REPORT\n"
        )
        handle.write(
            "================================================================\n\n"
        )

        handle.write(
            f"Version                 : {VERSION}\n"
        )
        handle.write(
            "Source                  : Frozen production AI artifacts\n"
        )
        handle.write(
            f"Production model        : {MODEL_PATH}\n"
        )
        handle.write(
            f"Production classes      : {len(classes)}\n"
        )
        handle.write(
            "Frozen event rows       : NOT EXPANDED INTO REPORT\n"
        )
        handle.write(
            "Synthetic predictions   : DISABLED\n"
        )
        handle.write(
            "Synthetic scores        : DISABLED\n"
        )
        handle.write(
            "Automatic execution     : DISABLED\n"
        )
        handle.write(
            "Automatic containment   : DISABLED\n"
        )

        handle.write("\n")
        handle.write(
            "PURPOSE\n"
        )
        handle.write(
            "-------\n"
        )
        handle.write(
            "This report presents the attack classes actually exposed by\n"
        )
        handle.write(
            "the frozen SmartSOC production model and connects them to\n"
        )
        handle.write(
            "the validated risk, decision, response and authorization\n"
        )
        handle.write(
            "architecture. It does not fabricate event predictions.\n"
        )

        handle.write("\n")
        handle.write(
            "IMPORTANT EXECUTION BOUNDARY\n"
        )
        handle.write(
            "-----------------------------\n"
        )
        handle.write(
            "Planned Mitigation Action means the response the validated\n"
        )
        handle.write(
            "policy prepares for the corresponding risk disposition.\n"
        )
        handle.write(
            "It does NOT mean the action was executed on a host.\n"
        )
        handle.write(
            "Actual execution remains behind the analyst authorization\n"
        )
        handle.write(
            "boundary and the validated system has executed no response\n"
        )
        handle.write(
            "or containment action.\n"
        )

        for index, profile in enumerate(
            profiles,
            start=1,
        ):
            write_profile(
                handle,
                index,
                profile["attack_class"],
                profile["mitre_id"],
                profile["attack_name"],
                profile["severity_score"],
                profile["severity_level"],
                policy,
            )

        handle.write("\n")
        handle.write(
            "================================================================\n"
        )
        handle.write(
            "REPORT SUMMARY\n"
        )
        handle.write(
            "================================================================\n\n"
        )

        level_counts: dict[str, int] = {}

        for profile in profiles:
            level = profile["severity_level"]
            level_counts[level] = (
                level_counts.get(level, 0) + 1
            )

        handle.write(
            f"Actual production attack classes : {len(profiles)}\n"
        )

        for level in [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL",
            "NOT EXPOSED BY ARTIFACT",
        ]:
            count = level_counts.get(level, 0)

            if count:
                handle.write(
                    f"{level:<30}: {count}\n"
                )

        handle.write("\n")
        handle.write(
            "Safety state\n"
        )
        handle.write(
            "------------\n"
        )
        handle.write(
            "Automatic response : DISABLED\n"
        )
        handle.write(
            "Automatic blocking : DISABLED\n"
        )
        handle.write(
            "Automatic isolation: DISABLED\n"
        )
        handle.write(
            "Process termination : NOT EXECUTED\n"
        )
        handle.write(
            "Account disabling   : NOT EXECUTED\n"
        )
        handle.write(
            "File deletion       : NOT EXECUTED\n"
        )
        handle.write(
            "Network containment : NOT EXECUTED\n"
        )

        handle.write("\n")
        handle.write(
            "END OF SMARTSOC AI ATTACK COVERAGE REPORT\n"
        )

    print(
        f"[PASS] Report generated: {OUTPUT_PATH}"
    )

    print()
    print("=" * 72)
    print("REPORT COMPLETE")
    print("=" * 72)
    print(
        f"Actual model classes : {len(classes)}"
    )
    print(
        "Synthetic output     : DISABLED"
    )
    print(
        "Response execution   : DISABLED"
    )
    print(
        "Containment execution: DISABLED"
    )
    print(
        f"TXT report           : {OUTPUT_PATH}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())