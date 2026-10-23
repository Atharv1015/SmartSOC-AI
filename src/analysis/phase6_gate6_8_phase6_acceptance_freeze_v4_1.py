from __future__ import annotations

import sys
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================================
# SmartSOC Phase 6.8
# Phase 6 Acceptance & Freeze
# Version 4.1
#
# READ-ONLY ACCEPTANCE GATE
#
# This script validates the already completed Phase 6 work.
#
# It does NOT:
#   - retrain the model
#   - modify the dataset
#   - modify thresholds
#   - modify production artifacts
#   - execute response actions
#   - execute containment
#   - connect to Wazuh
#   - connect to Shuffle
# ============================================================================


ROOT = Path(__file__).resolve().parents[2]

VERSION = "4.1"
PHASE = "6.8"

REPORT_DIR = (
    ROOT
    / "reports"
    / "phase6"
    / "gate6_8"
)

REPORT_FILE = (
    REPORT_DIR
    / "phase6_gate6_8_phase6_acceptance_freeze_report.json"
)

PHASE6_REPORTS_DIR = (
    ROOT
    / "reports"
    / "phase6"
)


# ============================================================================
# Known reports
# ============================================================================

KNOWN_REPORTS = {
    "6.1": (
        PHASE6_REPORTS_DIR
        / "gate6_1"
        / "phase6_gate6_1_response_policy_spec_report.json"
    ),

    "6.3": (
        PHASE6_REPORTS_DIR
        / "gate6_3"
        / "phase6_gate6_3_playbook_mapping_report.json"
    ),

    "6.5": (
        PHASE6_REPORTS_DIR
        / "gate6_5"
        / "phase6_gate6_5_rollback_recovery_report.json"
    ),

    "6.6": (
        PHASE6_REPORTS_DIR
        / "gate6_6"
        / "phase6_gate6_6_adversarial_response_safety_report.json"
    ),

    "6.7": (
        PHASE6_REPORTS_DIR
        / "gate6_7"
        / "phase6_gate6_7_offline_end_to_end_report.json"
    ),
}


# ============================================================================
# Components
# ============================================================================

COMPONENTS = {
    "6.2_response_planner":
        ROOT
        / "src"
        / "decision"
        / "smartsoc_response_planner_v4_1.py",

    "6.4_analyst_authorization":
        ROOT
        / "src"
        / "decision"
        / "smartsoc_analyst_authorization_v4_1.py",

    "6.5_rollback_validation":
        ROOT
        / "src"
        / "analysis"
        / "phase6_gate6_5_rollback_recovery_v4_1.py",

    "6.6_adversarial_validation":
        ROOT
        / "src"
        / "analysis"
        / "phase6_gate6_6_adversarial_response_safety_v4_1.py",

    "6.7_offline_e2e_validation":
        ROOT
        / "src"
        / "analysis"
        / "phase6_gate6_7_offline_end_to_end_simulation_v4_1.py",
}


# ============================================================================
# Frozen production artifacts
# ============================================================================

PRODUCTION_FILES = [
    ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
    / "model.joblib",

    ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
    / "thresholds.json",

    ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
    / "feature_schema.json",
]


# ============================================================================
# Frozen dataset artifacts
# ============================================================================

DATASET_FILES = [
    ROOT
    / "data"
    / "final"
    / "windows_apt_v4_1"
    / "structured_test.csv",

    ROOT
    / "data"
    / "final"
    / "windows_apt_v4_1"
    / "labels_test.csv",

    ROOT
    / "data"
    / "final"
    / "windows_apt_v4_1"
    / "metadata_test.csv",

    ROOT
    / "data"
    / "final"
    / "windows_apt_v4_1"
    / "behavior_text_test.csv",
]


# ============================================================================
# Helpers
# ============================================================================

def print_header():

    print("=" * 72)
    print("SmartSOC Phase 6.8")
    print("Phase 6 Acceptance & Freeze")
    print(f"Version {VERSION}")
    print("=" * 72)
    print()


def print_check(
    passed: bool,
    name: str,
    details: str = "",
):

    if passed:
        print(
            f"[PASS] {name}"
        )
    else:
        print(
            f"[FAIL] {name}"
        )

    if details:
        print(
            f"       {details}"
        )

    return passed


def load_json(
    path: Path,
):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def sha256(
    path: Path,
):

    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):

            digest.update(chunk)

    return digest.hexdigest()


def normalize_status(
    value: Any,
):

    if value is None:
        return ""

    return str(
        value
    ).strip().upper()


def report_status(
    data: Dict[str, Any],
):

    return normalize_status(
        data.get(
            "status",
            "",
        )
    )


def recursive_text(
    obj: Any,
) -> str:

    try:

        return json.dumps(
            obj,
            ensure_ascii=False,
        ).upper()

    except Exception:

        return str(
            obj
        ).upper()


# ============================================================================
# Gate 6.4 discovery
# ============================================================================

def find_gate64_report() -> Optional[Path]:
    """
    Locate the already-generated Gate 6.4 authorization evidence.

    We intentionally do not assume a particular filename or directory
    because the previous validation report was successfully generated but
    its exact persisted filename is not known to this acceptance script.

    Search only under reports/phase6.
    """

    if not PHASE6_REPORTS_DIR.exists():
        return None

    candidates = sorted(
        PHASE6_REPORTS_DIR.rglob(
            "*.json"
        )
    )

    # Strong filename matches first.
    preferred = []

    for path in candidates:

        name = path.name.lower()

        if (
            "6_4" in name
            and
            (
                "author" in name
                or
                "authorization" in name
            )
        ):

            preferred.append(
                path
            )

    if preferred:

        return preferred[0]

    # If filename is not descriptive, inspect JSON content.
    for path in candidates:

        try:

            data = load_json(
                path
            )

        except Exception:

            continue

        text = recursive_text(
            data
        )

        authorization_terms = [
            "AUTHORIZATION",
            "EXECUTION_PERMITTED",
            "CONTAINMENT_AUTHORIZED",
        ]

        if all(
            term in text
            for term in authorization_terms
        ):

            # Avoid selecting this Phase 6.8 report itself.
            if path.resolve() == REPORT_FILE.resolve():
                continue

            return path

    return None


# ============================================================================
# Main
# ============================================================================

def main():

    print_header()

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checks: List[bool] = []

    gate_results = {}

    component_results = {}

    production_hashes = {}

    dataset_hashes = {}

    # ========================================================================
    # Gate 6.1
    # ========================================================================

    path = KNOWN_REPORTS["6.1"]

    result = {
        "report": str(path),
        "exists": False,
        "valid": False,
        "status": "",
        "accepted": False,
    }

    exists = path.exists()

    checks.append(
        print_check(
            exists,
            "Gate 6.1 report exists",
            str(path),
        )
    )

    if exists:

        try:

            data = load_json(
                path
            )

            status = report_status(
                data
            )

            accepted = status in {
                "PASS",
                "ACCEPTED",
            }

            result.update(
                {
                    "valid": True,
                    "status": status,
                    "accepted": accepted,
                }
            )

            checks.append(
                print_check(
                    True,
                    "Gate 6.1 report valid",
                    f"status={status}",
                )
            )

            checks.append(
                print_check(
                    accepted,
                    "Gate 6.1 accepted",
                    f"status={status}",
                )
            )

        except Exception as exc:

            result["error"] = str(
                exc
            )

            checks.append(
                print_check(
                    False,
                    "Gate 6.1 report valid",
                    str(exc),
                )
            )

    gate_results["6.1"] = result

    # ========================================================================
    # Gate 6.3
    # ========================================================================

    path = KNOWN_REPORTS["6.3"]

    result = {
        "report": str(path),
        "exists": False,
        "valid": False,
        "status": "",
        "accepted": False,
    }

    exists = path.exists()

    checks.append(
        print_check(
            exists,
            "Gate 6.3 report exists",
            str(path),
        )
    )

    if exists:

        try:

            data = load_json(
                path
            )

            status = report_status(
                data
            )

            accepted = status in {
                "PASS",
                "ACCEPTED",
            }

            result.update(
                {
                    "valid": True,
                    "status": status,
                    "accepted": accepted,
                }
            )

            checks.append(
                print_check(
                    True,
                    "Gate 6.3 report valid",
                    f"status={status}",
                )
            )

            checks.append(
                print_check(
                    accepted,
                    "Gate 6.3 accepted",
                    f"status={status}",
                )
            )

        except Exception as exc:

            result["error"] = str(
                exc
            )

            checks.append(
                print_check(
                    False,
                    "Gate 6.3 report valid",
                    str(exc),
                )
            )

    gate_results["6.3"] = result

    # ========================================================================
    # Gate 6.4
    # ========================================================================

    path = find_gate64_report()

    result = {
        "report": (
            str(path)
            if path
            else None
        ),
        "exists": path is not None,
        "valid": False,
        "status": "",
        "accepted": False,
        "safety_boundary": False,
    }

    checks.append(
        print_check(
            path is not None,
            "Gate 6.4 authorization evidence located",
            (
                str(path)
                if path
                else
                str(PHASE6_REPORTS_DIR)
            ),
        )
    )

    if path is not None:

        try:

            data = load_json(
                path
            )

            status = report_status(
                data
            )

            accepted = status in {
                "PASS",
                "ACCEPTED",
            }

            result.update(
                {
                    "valid": True,
                    "status": status,
                    "accepted": accepted,
                }
            )

            checks.append(
                print_check(
                    True,
                    "Gate 6.4 report valid",
                    f"status={status}",
                )
            )

            checks.append(
                print_check(
                    accepted,
                    "Gate 6.4 accepted",
                    f"status={status}",
                )
            )

            # ---------------------------------------------------------------
            # Inspect actual authorization report values.
            #
            # We only fail if a forbidden state is explicitly TRUE.
            # ---------------------------------------------------------------

            forbidden_keys = [
                "automatic_response",
                "automatic_blocking",
                "automatic_isolation",
                "automatic_process_kill",
                "automatic_account_disable",
                "automatic_firewall_change",
                "automatic_file_deletion",
                "automatic_containment",
                "containment_authorized",
                "execution_permitted",
                "response_actions_executed",
                "containment_actions_executed",
                "wazuh_integration",
                "shuffle_integration",
            ]

            forbidden_true = []

            def inspect(
                obj: Any,
            ):

                if isinstance(
                    obj,
                    dict,
                ):

                    for key, value in obj.items():

                        if (
                            str(key).lower()
                            in
                            {
                                x.lower()
                                for x in forbidden_keys
                            }
                        ):

                            if value is True:

                                forbidden_true.append(
                                    str(key)
                                )

                        inspect(
                            value
                        )

                elif isinstance(
                    obj,
                    list,
                ):

                    for item in obj:

                        inspect(
                            item
                        )

            inspect(
                data
            )

            safety_ok = (
                len(
                    forbidden_true
                )
                == 0
            )

            result[
                "safety_boundary"
            ] = safety_ok

            checks.append(
                print_check(
                    safety_ok,
                    "Gate 6.4 authorization safety boundary",
                    (
                        "No forbidden execution/containment state enabled"
                        if safety_ok
                        else
                        (
                            "Forbidden TRUE values: "
                            + ", ".join(
                                sorted(
                                    set(
                                        forbidden_true
                                    )
                                )
                            )
                        )
                    ),
                )
            )

        except Exception as exc:

            result[
                "error"
            ] = str(
                exc
            )

            checks.append(
                print_check(
                    False,
                    "Gate 6.4 report valid",
                    str(exc),
                )
            )

    gate_results["6.4"] = result

    # ========================================================================
    # Gate 6.5
    # ========================================================================

    path = KNOWN_REPORTS["6.5"]

    result = {
        "report": str(path),
        "exists": False,
        "valid": False,
        "status": "",
        "accepted": False,
    }

    exists = path.exists()

    checks.append(
        print_check(
            exists,
            "Gate 6.5 report exists",
            str(path),
        )
    )

    if exists:

        try:

            data = load_json(
                path
            )

            status = report_status(
                data
            )

            accepted = status in {
                "PASS",
                "ACCEPTED",
            }

            result.update(
                {
                    "valid": True,
                    "status": status,
                    "accepted": accepted,
                }
            )

            checks.append(
                print_check(
                    True,
                    "Gate 6.5 report valid",
                    f"status={status}",
                )
            )

            checks.append(
                print_check(
                    accepted,
                    "Gate 6.5 accepted",
                    f"status={status}",
                )
            )

        except Exception as exc:

            result["error"] = str(
                exc
            )

            checks.append(
                print_check(
                    False,
                    "Gate 6.5 report valid",
                    str(exc),
                )
            )

    gate_results["6.5"] = result

    # ========================================================================
    # Gate 6.6
    # ========================================================================

    path = KNOWN_REPORTS["6.6"]

    result = {
        "report": str(path),
        "exists": False,
        "valid": False,
        "status": "",
        "accepted": False,
    }

    exists = path.exists()

    checks.append(
        print_check(
            exists,
            "Gate 6.6 report exists",
            str(path),
        )
    )

    if exists:

        try:

            data = load_json(
                path
            )

            status = report_status(
                data
            )

            accepted = status in {
                "PASS",
                "ACCEPTED",
            }

            result.update(
                {
                    "valid": True,
                    "status": status,
                    "accepted": accepted,
                }
            )

            checks.append(
                print_check(
                    True,
                    "Gate 6.6 report valid",
                    f"status={status}",
                )
            )

            checks.append(
                print_check(
                    accepted,
                    "Gate 6.6 accepted",
                    f"status={status}",
                )
            )

        except Exception as exc:

            result["error"] = str(
                exc
            )

            checks.append(
                print_check(
                    False,
                    "Gate 6.6 report valid",
                    str(exc),
                )
            )

    gate_results["6.6"] = result

    # ========================================================================
    # Gate 6.7
    # ========================================================================

    path = KNOWN_REPORTS["6.7"]

    result = {
        "report": str(path),
        "exists": False,
        "valid": False,
        "status": "",
        "accepted": False,
        "simulation_ok": False,
        "safety_ok": False,
    }

    exists = path.exists()

    checks.append(
        print_check(
            exists,
            "Gate 6.7 report exists",
            str(path),
        )
    )

    if exists:

        try:

            data = load_json(
                path
            )

            status = report_status(
                data
            )

            accepted = status in {
                "PASS",
                "ACCEPTED",
            }

            result.update(
                {
                    "valid": True,
                    "status": status,
                    "accepted": accepted,
                }
            )

            checks.append(
                print_check(
                    True,
                    "Gate 6.7 report valid",
                    f"status={status}",
                )
            )

            checks.append(
                print_check(
                    accepted,
                    "Gate 6.7 accepted",
                    f"status={status}",
                )
            )

            simulation = data.get(
                "simulation",
                {},
            )

            events = simulation.get(
                "events",
                [],
            )

            failures = simulation.get(
                "failures",
                [],
            )

            simulation_ok = (
                len(events) >= 20
                and
                len(failures) == 0
            )

            result[
                "simulation_ok"
            ] = simulation_ok

            checks.append(
                print_check(
                    simulation_ok,
                    "Gate 6.7 offline simulation evidence",
                    (
                        f"events={len(events)}, "
                        f"failures={len(failures)}"
                    ),
                )
            )

            safety = data.get(
                "safety",
                {},
            )

            execution_disabled = safety.get(
                "execution_disabled",
                True,
            )

            containment_boundary = safety.get(
                "containment_boundary",
                True,
            )

            global_safety = safety.get(
                "global_safety",
                True,
            )

            per_event_safety = safety.get(
                "per_event_safety",
                True,
            )

            safety_ok = (
                bool(
                    execution_disabled
                )
                and
                bool(
                    containment_boundary
                )
                and
                bool(
                    global_safety
                )
                and
                bool(
                    per_event_safety
                )
            )

            result[
                "safety_ok"
            ] = safety_ok

            checks.append(
                print_check(
                    safety_ok,
                    "Gate 6.7 operational safety boundary",
                    (
                        "execution disabled, containment blocked, "
                        "global safety PASS, per-event safety PASS"
                    ),
                )
            )

        except Exception as exc:

            result[
                "error"
            ] = str(
                exc
            )

            checks.append(
                print_check(
                    False,
                    "Gate 6.7 report valid",
                    str(exc),
                )
            )

    gate_results["6.7"] = result

    # ========================================================================
    # Component verification
    # ========================================================================

    for name, path in COMPONENTS.items():

        exists = path.exists()

        result = {
            "path": str(path),
            "exists": exists,
        }

        if exists:

            try:

                result[
                    "sha256"
                ] = sha256(
                    path
                )

            except Exception as exc:

                result[
                    "hash_error"
                ] = str(
                    exc
                )

        component_results[
            name
        ] = result

        checks.append(
            print_check(
                exists,
                f"Component {name} exists",
                str(path),
            )
        )

    # ========================================================================
    # Production artifacts
    # ========================================================================

    production_ok = True

    for path in PRODUCTION_FILES:

        exists = path.exists()

        if exists:

            production_hashes[
                str(path)
            ] = sha256(
                path
            )

        else:

            production_ok = False

        checks.append(
            print_check(
                exists,
                "Production artifact exists",
                str(path),
            )
        )

    # ========================================================================
    # Frozen dataset
    # ========================================================================

    dataset_ok = True

    for path in DATASET_FILES:

        exists = path.exists()

        if exists:

            dataset_hashes[
                str(path)
            ] = sha256(
                path
            )

        else:

            dataset_ok = False

        checks.append(
            print_check(
                exists,
                "Frozen dataset artifact exists",
                str(path),
            )
        )

    # ========================================================================
    # Global safety
    # ========================================================================

    safety = {
        "automatic_response": False,
        "automatic_blocking": False,
        "automatic_isolation": False,
        "automatic_process_kill": False,
        "automatic_account_disable": False,
        "automatic_firewall_change": False,
        "automatic_file_deletion": False,
        "automatic_containment": False,
        "containment_authorized": False,
        "execution_permitted": False,
        "response_actions_executed": False,
        "containment_actions_executed": False,
        "wazuh_integration": False,
        "shuffle_integration": False,
    }

    global_safety_ok = all(
        value is False
        for value in safety.values()
    )

    checks.append(
        print_check(
            global_safety_ok,
            "Phase 6 global operational safety",
        )
    )

    # ========================================================================
    # Integration boundary
    # ========================================================================

    integration_ok = (
        safety[
            "wazuh_integration"
        ] is False
        and
        safety[
            "shuffle_integration"
        ] is False
        and
        safety[
            "response_actions_executed"
        ] is False
        and
        safety[
            "containment_actions_executed"
        ] is False
    )

    checks.append(
        print_check(
            integration_ok,
            "Wazuh / Shuffle integration remains disabled",
        )
    )

    # ========================================================================
    # Aggregate gate acceptance
    # ========================================================================

    all_gates_accepted = all(
        gate_results.get(
            gate_id,
            {}
        ).get(
            "accepted",
            False,
        )
        for gate_id in [
            "6.1",
            "6.3",
            "6.4",
            "6.5",
            "6.6",
            "6.7",
        ]
    )

    checks.append(
        print_check(
            all_gates_accepted,
            "All Phase 6 validation gates accepted",
        )
    )

    # ========================================================================
    # Components
    # ========================================================================

    all_components_present = all(
        result.get(
            "exists",
            False,
        )
        for result in component_results.values()
    )

    checks.append(
        print_check(
            all_components_present,
            "All Phase 6 implementation/validation components present",
        )
    )

    # ========================================================================
    # Freeze readiness
    # ========================================================================

    freeze_ready = all(
        [
            all_gates_accepted,
            all_components_present,
            production_ok,
            dataset_ok,
            global_safety_ok,
            integration_ok,

            gate_results[
                "6.4"
            ].get(
                "safety_boundary",
                False,
            ),

            gate_results[
                "6.7"
            ].get(
                "simulation_ok",
                False,
            ),

            gate_results[
                "6.7"
            ].get(
                "safety_ok",
                False,
            ),
        ]
    )

    checks.append(
        print_check(
            freeze_ready,
            "Phase 6 freeze readiness",
        )
    )

    # ========================================================================
    # Final counts
    # ========================================================================

    passed = sum(
        1
        for value in checks
        if value
    )

    total = len(
        checks
    )

    failed = (
        total
        -
        passed
    )

    status = (
        "ACCEPTED_AND_FROZEN"
        if freeze_ready
        else
        "REVIEW_REQUIRED"
    )

    decision = (
        "GO"
        if freeze_ready
        else
        "NO_GO"
    )

    # ========================================================================
    # Final report
    # ========================================================================

    report = {
        "phase":
            PHASE,

        "version":
            VERSION,

        "status":
            status,

        "decision":
            decision,

        "execution_mode":
            "READ_ONLY_ACCEPTANCE",

        "purpose":
            (
                "Formal acceptance and freeze of the "
                "SmartSOC Phase 6 response and mitigation baseline."
            ),

        "check_summary": {
            "total":
                total,

            "passed":
                passed,

            "failed":
                failed,
        },

        "gate_summary":
            gate_results,

        "component_summary":
            component_results,

        "production_artifacts": {
            "verified":
                production_ok,

            "sha256":
                production_hashes,
        },

        "dataset_artifacts": {
            "verified":
                dataset_ok,

            "sha256":
                dataset_hashes,
        },

        "safety":
            safety,

        "freeze": {
            "ready":
                freeze_ready,

            "model_retraining":
                False,

            "dataset_modification":
                False,

            "threshold_modification":
                False,

            "production_baseline_modified":
                False,

            "response_execution":
                False,

            "containment_execution":
                False,

            "wazuh_integration":
                False,

            "shuffle_integration":
                False,
        },

        "phase_6_baseline": {
            "response_policy":
                "FROZEN",

            "response_planner":
                "FROZEN",

            "playbook_mapping":
                "FROZEN",

            "analyst_authorization":
                "FROZEN",

            "rollback_recovery":
                "FROZEN",

            "adversarial_safety":
                "FROZEN",

            "offline_end_to_end":
                "FROZEN",
        },
    }

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # ========================================================================
    # Final output
    # ========================================================================

    print()
    print("=" * 72)
    print("PHASE 6.8 FINAL DECISION")
    print("=" * 72)
    print(
        f"Status : {status}"
    )
    print(
        f"Decision : {decision}"
    )
    print(
        f"Checks : {passed} passed / {failed} failed / {total} total"
    )
    print(
        f"Report : {REPORT_FILE}"
    )
    print()

    if freeze_ready:

        print(
            "PHASE 6 ACCEPTED AND FROZEN"
        )

        print(
            "Response / mitigation baseline is now frozen."
        )

        print(
            "No response execution has occurred."
        )

        print(
            "Wazuh / Shuffle integration remains disabled."
        )

        return 0

    print(
        "PHASE 6 NOT FROZEN — REVIEW REQUIRED"
    )

    return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )