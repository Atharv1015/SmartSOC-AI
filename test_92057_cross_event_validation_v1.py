"""
SmartSOC Stage 6
92057 Cross-Event Validation

Purpose
-------
Validate whether the system_level scaling/model behavior observed
for Wazuh rule 92057 is isolated or systematic across other real
Wazuh events available in the SmartSOC investigation.

READ-ONLY DIAGNOSTIC.

NO:
- production modification
- model retraining
- threshold modification
- victim contact
- mitigation
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import production_inference_windows_apt_v4_1_aligned as canonical


REPORT_DIR = (
    PROJECT_ROOT /
    "reports" /
    "integration"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_FILE = (
    REPORT_DIR /
    "92057_cross_event_validation_v1.json"
)


# ============================================================
# EVENT SOURCE FILES
# ============================================================

# The exact real-event forensic scripts previously used in the
# investigation contain RAW_WAZUH_EVENT assignments.
#
# We search the project for these scripts rather than inventing
# new event data.

CANDIDATE_FILES = [
    PROJECT_ROOT /
    "test_92057_pipeline_forensics_v3.py",

    PROJECT_ROOT /
    "test_92057_scaler_forensics_v1.py",
]


# ============================================================
# HELPERS
# ============================================================

def section(title):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def extract_raw_events_from_file(path):

    if not path.exists():
        return []

    source = path.read_text(
        encoding="utf-8"
    )

    try:
        tree = ast.parse(
            source,
            filename=str(path),
        )
    except Exception:
        return []

    events = []

    for node in tree.body:

        if not isinstance(
            node,
            ast.Assign,
        ):
            continue

        for target in node.targets:

            if not (
                isinstance(
                    target,
                    ast.Name,
                )
                and target.id
                == "RAW_WAZUH_EVENT"
            ):
                continue

            try:

                event = ast.literal_eval(
                    node.value
                )

                if isinstance(
                    event,
                    dict,
                ):

                    events.append(
                        event
                    )

            except Exception:
                pass

    return events


def recursive_find_json_files():

    roots = [
        PROJECT_ROOT /
        "reports",

        PROJECT_ROOT /
        "tests",

        PROJECT_ROOT /
        "src",
    ]

    found = []

    for root in roots:

        if not root.exists():
            continue

        try:

            for path in root.rglob(
                "*.json"
            ):

                name = path.name.lower()

                if (
                    "92057" in name
                    or "92033" in name
                    or "92031" in name
                    or "92021" in name
                    or "wazuh" in name
                ):

                    found.append(
                        path
                    )

        except Exception:
            pass

    return found


def load_events_from_json(path):

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

    except Exception:

        return []

    if isinstance(
        data,
        dict,
    ):

        # Direct Wazuh event.
        if (
            "rule" in data
            and "agent" in data
        ):

            return [data]

        # Common wrappers.
        for key in [
            "event",
            "alert",
            "wazuh_event",
            "result",
        ]:

            value = data.get(key)

            if isinstance(
                value,
                dict,
            ):

                if (
                    "rule" in value
                    and "agent" in value
                ):

                    return [value]

            if isinstance(
                value,
                list,
            ):

                valid = [
                    x
                    for x in value
                    if (
                        isinstance(
                            x,
                            dict,
                        )
                        and "rule" in x
                    )
                ]

                if valid:
                    return valid

    if isinstance(
        data,
        list,
    ):

        return [
            x
            for x in data
            if (
                isinstance(
                    x,
                    dict,
                )
                and "rule" in x
            )
        ]

    return []


def event_rule_id(event):

    try:
        return str(
            event[
                "rule"
            ][
                "id"
            ]
        )
    except Exception:
        return "UNKNOWN"


def event_mitre(event):

    try:

        mitre = event[
            "rule"
        ][
            "mitre"
        ]

        if isinstance(
            mitre,
            dict,
        ):

            value = mitre.get(
                "id",
                [],
            )

            if isinstance(
                value,
                list,
            ):
                return [
                    str(x)
                    for x in value
                ]

            return [
                str(value)
            ]

        if isinstance(
            mitre,
            list,
        ):

            return [
                str(x)
                for x in mitre
            ]

    except Exception:
        pass

    return []


def event_description(event):

    try:
        return str(
            event[
                "rule"
            ].get(
                "description",
                "",
            )
        )
    except Exception:
        return ""


def get_top_scores(
    prediction,
    limit=5,
):

    scores = []

    for item in prediction.get(
        "all_scores",
        [],
    ):

        try:

            scores.append(
                {
                    "technique":
                        str(
                            item[
                                "technique"
                            ]
                        ),

                    "confidence":
                        float(
                            item[
                                "confidence"
                            ]
                        ),

                    "threshold":
                        float(
                            item[
                                "threshold"
                            ]
                        ),

                    "detected":
                        bool(
                            item[
                                "detected"
                            ]
                        ),
                }
            )

        except Exception:
            pass

    scores.sort(
        key=lambda x:
            x["confidence"],
        reverse=True,
    )

    return scores[:limit]


def get_score(
    prediction,
    technique,
):

    for item in prediction.get(
        "all_scores",
        [],
    ):

        if (
            str(
                item[
                    "technique"
                ]
            )
            == technique
        ):

            return {
                "confidence":
                    float(
                        item[
                            "confidence"
                        ]
                    ),

                "threshold":
                    float(
                        item[
                            "threshold"
                        ]
                    ),

                "detected":
                    bool(
                        item[
                            "detected"
                        ]
                    ),
            }

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "SmartSOC Stage 6 - 92057 CROSS-EVENT VALIDATION"
    )

    print()
    print(
        "PURPOSE"
    )

    print(
        "Determine whether the system_level behavior "
        "observed for rule 92057 is isolated or "
        "systematic across available real Wazuh events."
    )

    print()
    print(
        "SAFETY"
    )

    print(
        "Production modification : NO"
    )

    print(
        "Retraining              : NO"
    )

    print(
        "Threshold modification  : NO"
    )

    print(
        "Victim contact           : NO"
    )

    print(
        "Mitigation               : NO"
    )

    # ========================================================
    # LOAD ARTIFACTS
    # ========================================================

    section(
        "1. LOAD FROZEN PRODUCTION ARTIFACTS"
    )

    artifacts = (
        canonical.load_artifacts()
    )

    model = artifacts[
        "model"
    ]

    scaler = artifacts[
        "scaler"
    ]

    print(
        "Model:",
        type(model).__name__
    )

    print(
        "Scaler:",
        type(scaler).__name__
    )

    print(
        "Features:",
        model.n_features_in_
    )

    # ========================================================
    # COLLECT EVENTS
    # ========================================================

    section(
        "2. COLLECT AVAILABLE REAL WAZUH EVENTS"
    )

    events = []

    # --------------------------------------------------------
    # Python forensic scripts
    # --------------------------------------------------------

    for path in CANDIDATE_FILES:

        for event in (
            extract_raw_events_from_file(
                path
            )
        ):

            events.append(
                (
                    str(path),
                    event,
                )
            )

    # --------------------------------------------------------
    # JSON reports / event files
    # --------------------------------------------------------

    for path in recursive_find_json_files():

        for event in (
            load_events_from_json(
                path
            )
        ):

            events.append(
                (
                    str(path),
                    event,
                )
            )

    # --------------------------------------------------------
    # Deduplicate using rule + timestamp + alert id
    # --------------------------------------------------------

    unique = {}

    for source, event in events:

        try:

            alert_id = str(
                event.get(
                    "id",
                    "",
                )
            )

        except Exception:

            alert_id = ""

        try:

            timestamp = str(
                event.get(
                    "timestamp",
                    "",
                )
            )

        except Exception:

            timestamp = ""

        key = (
            event_rule_id(event),
            timestamp,
            alert_id,
        )

        if key not in unique:

            unique[key] = (
                source,
                event,
            )

    events = list(
        unique.values()
    )

    print(
        "Unique candidate events:",
        len(events)
    )

    for source, event in events:

        print(
            f"Rule={event_rule_id(event):<8} "
            f"MITRE={event_mitre(event)} "
            f"source={source}"
        )

    # ========================================================
    # ANALYZE EVENTS
    # ========================================================

    section(
        "3. CROSS-EVENT FEATURE + INFERENCE ANALYSIS"
    )

    numerical_features = [
        str(x)
        for x in canonical.NUMERICAL_FEATURES
    ]

    system_position = (
        numerical_features.index(
            "system_level"
        )
    )

    matrix_index = (
        canonical.EXPECTED_CATEGORICAL_ENCODED
        + system_position
    )

    results = []

    for event_number, (
        source,
        event,
    ) in enumerate(
        events,
        start=1,
    ):

        rule_id = event_rule_id(
            event
        )

        mitre = event_mitre(
            event
        )

        print()
        print(
            "-" * 100
        )

        print(
            f"EVENT {event_number}"
        )

        print(
            "Rule:",
            rule_id
        )

        print(
            "MITRE:",
            mitre
        )

        print(
            "Description:",
            event_description(
                event
            )
        )

        try:

            normalized = (
                canonical.normalize_wazuh_event(
                    event
                )
            )

            feature_result = (
                canonical.build_feature_matrix(
                    normalized,
                    artifacts,
                )
            )

            matrix = feature_result[
                "matrix"
            ]

            numerical_raw = (
                feature_result[
                    "numerical_array"
                ]
            )

            numerical_scaled = (
                feature_result[
                    "numerical_scaled"
                ]
            )

            prediction = (
                canonical.predict(
                    feature_result,
                    artifacts,
                )
            )

            raw_system_level = float(
                numerical_raw[
                    0,
                    system_position
                ]
            )

            scaled_system_level = float(
                numerical_scaled[
                    0,
                    system_position
                ]
            )

            matrix_system_level = float(
                matrix[
                    0,
                    matrix_index
                ]
            )

            t1070 = get_score(
                prediction,
                "T1070",
            )

            t1059 = get_score(
                prediction,
                "T1059.001",
            )

            top_scores = get_top_scores(
                prediction,
                limit=5,
            )

            print(
                "system_level raw:",
                raw_system_level
            )

            print(
                "system_level scaled:",
                scaled_system_level
            )

            print(
                "system_level matrix:",
                matrix_system_level
            )

            print(
                "T1070:",
                t1070
            )

            print(
                "T1059.001:",
                t1059
            )

            print(
                "Top predictions:"
            )

            for item in top_scores:

                print(
                    f"  {item['technique']:<12} "
                    f"{item['confidence']:.12f} "
                    f"detected={item['detected']}"
                )

            result = {

                "source":
                    source,

                "event_number":
                    event_number,

                "rule_id":
                    rule_id,

                "mitre":
                    mitre,

                "description":
                    event_description(
                        event
                    ),

                "timestamp":
                    event.get(
                        "timestamp"
                    ),

                "alert_id":
                    event.get(
                        "id"
                    ),

                "system_level": {

                    "raw":
                        raw_system_level,

                    "scaled":
                        scaled_system_level,

                    "matrix":
                        matrix_system_level,

                    "matrix_index":
                        matrix_index,
                },

                "T1070":
                    t1070,

                "T1059.001":
                    t1059,

                "top_predictions":
                    top_scores,
            }

            results.append(
                result
            )

        except Exception as exc:

            print(
                "[SKIP] Event could not be "
                "processed:",
                repr(exc),
            )

    # ========================================================
    # PREFERRED EVENTS
    # ========================================================

    section(
        "4. PREFERRED REAL-EVENT SUMMARY"
    )

    preferred_ids = {
        "92057",
        "92033",
        "92031",
        "92021",
    }

    preferred = [
        result
        for result in results
        if result[
            "rule_id"
        ] in preferred_ids
    ]

    preferred.sort(
        key=lambda x:
            int(
                x["rule_id"]
            )
            if x[
                "rule_id"
            ].isdigit()
            else 999999
    )

    if not preferred:

        print(
            "No preferred-event set was found "
            "in the available local artifacts."
        )

    for result in preferred:

        print()
        print(
            f"Rule {result['rule_id']}"
        )

        print(
            "  MITRE:",
            result["mitre"]
        )

        print(
            "  raw system_level:",
            result[
                "system_level"
            ][
                "raw"
            ]
        )

        print(
            "  scaled system_level:",
            result[
                "system_level"
            ][
                "scaled"
            ]
        )

        print(
            "  T1070:",
            result[
                "T1070"
            ]
        )

        print(
            "  T1059.001:",
            result[
                "T1059.001"
            ]
        )

        print(
            "  Top:",
            [
                (
                    x["technique"],
                    round(
                        x["confidence"],
                        6,
                    ),
                )
                for x in result[
                    "top_predictions"
                ]
            ]
        )

    # ========================================================
    # SCALE ANALYSIS
    # ========================================================

    section(
        "5. CROSS-EVENT SYSTEM_LEVEL MAGNITUDE ANALYSIS"
    )

    scaled_values = []

    for result in results:

        value = float(
            result[
                "system_level"
            ][
                "scaled"
            ]
        )

        scaled_values.append(
            value
        )

        print(
            f"Rule {result['rule_id']:<8} "
            f"raw={result['system_level']['raw']:>10.4f} "
            f"scaled={value:>15.6f}"
        )

    if scaled_values:

        scaled_array = np.asarray(
            scaled_values,
            dtype=float,
        )

        print()
        print(
            "Cross-event scaled system_level:"
        )

        print(
            "  min:",
            float(
                np.min(
                    scaled_array
                )
            )
        )

        print(
            "  max:",
            float(
                np.max(
                    scaled_array
                )
            )
        )

        print(
            "  mean:",
            float(
                np.mean(
                    scaled_array
                )
            )
        )

    # ========================================================
    # CLASSIFICATION CONSISTENCY
    # ========================================================

    section(
        "6. CLASSIFICATION CONSISTENCY"
    )

    t1070_wins = 0
    t1059_wins = 0
    other_wins = 0

    classification_rows = []

    for result in results:

        top = result[
            "top_predictions"
        ]

        if not top:

            winner = "NONE"

        else:

            winner = top[0][
                "technique"
            ]

        if winner == "T1070":

            t1070_wins += 1

        elif winner == "T1059.001":

            t1059_wins += 1

        else:

            other_wins += 1

        classification_rows.append(
            {
                "rule_id":
                    result[
                        "rule_id"
                    ],

                "mitre":
                    result[
                        "mitre"
                    ],

                "winner":
                    winner,

                "raw_system_level":
                    result[
                        "system_level"
                    ][
                        "raw"
                    ],

                "scaled_system_level":
                    result[
                        "system_level"
                    ][
                        "scaled"
                    ],
            }
        )

        print(
            f"Rule {result['rule_id']:<8} "
            f"winner={winner:<12} "
            f"raw={result['system_level']['raw']:>8.3f} "
            f"scaled={result['system_level']['scaled']:>12.3f}"
        )

    # ========================================================
    # 92057 SPECIFIC CHECK
    # ========================================================

    section(
        "7. 92057 SPECIFIC FINAL CHECK"
    )

    result_92057 = [
        x
        for x in results
        if x[
            "rule_id"
        ] == "92057"
    ]

    if result_92057:

        item = result_92057[0]

        print(
            "92057 raw system_level:",
            item[
                "system_level"
            ][
                "raw"
            ]
        )

        print(
            "92057 scaled system_level:",
            item[
                "system_level"
            ][
                "scaled"
            ]
        )

        print(
            "92057 winner:",
            item[
                "top_predictions"
            ][0][
                "technique"
            ]
            if item[
                "top_predictions"
            ]
            else "NONE"
        )

    else:

        print(
            "92057 was not found."
        )

    # ========================================================
    # FINAL DIAGNOSTIC DECISION
    # ========================================================

    section(
        "8. STAGE 6 FINAL DIAGNOSTIC ASSESSMENT"
    )

    assessment = []

    if not results:

        assessment.append(
            "INSUFFICIENT_DATA: no real Wazuh "
            "events were available locally."
        )

        decision = (
            "INSUFFICIENT_DATA"
        )

    else:

        # Determine whether scaled values are
        # consistently extreme.
        extreme_count = sum(
            1
            for result in results
            if abs(
                float(
                    result[
                        "system_level"
                    ][
                        "scaled"
                    ]
                )
            ) >= 10
        )

        extreme_fraction = (
            extreme_count
            / len(results)
        )

        print(
            "Events analyzed:",
            len(results)
        )

        print(
            "Events with |scaled system_level| >= 10:",
            extreme_count
        )

        print(
            "Extreme-value fraction:",
            extreme_fraction
        )

        if extreme_fraction >= 0.5:

            assessment.append(
                "SYSTEMATIC_SCALING_PATTERN: "
                "at least half of the available "
                "real events have an extreme transformed "
                "system_level magnitude."
            )

            decision = (
                "SYSTEMATIC_SCALING_PATTERN"
            )

        elif extreme_count > 0:

            assessment.append(
                "MIXED_SCALING_PATTERN: "
                "some real events have extreme "
                "system_level magnitudes, but the "
                "behavior is not universal."
            )

            decision = (
                "MIXED_SCALING_PATTERN"
            )

        else:

            assessment.append(
                "NO_EXTREME_SCALING_PATTERN: "
                "the available real events do not "
                "show extreme transformed system_level."
            )

            decision = (
                "NO_EXTREME_SCALING_PATTERN"
            )

        # Check whether 92057 is actually classified
        # against its Wazuh MITRE label.
        for result in results:

            if (
                result[
                    "rule_id"
                ]
                != "92057"
            ):
                continue

            expected = set(
                result[
                    "mitre"
                ]
            )

            predicted = {
                x[
                    "technique"
                ]
                for x in result[
                    "top_predictions"
                ]
                if x[
                    "detected"
                ]
            }

            if (
                expected
                and not (
                    expected
                    & predicted
                )
            ):

                assessment.append(
                    "92057_LABEL_MISMATCH: "
                    "the expected Wazuh MITRE technique "
                    "is not among the detected top predictions."
                )

            else:

                assessment.append(
                    "92057_LABEL_ALIGNMENT: "
                    "the expected Wazuh MITRE technique "
                    "appears among detected predictions."
                )

    for item in assessment:

        print(
            "-",
            item
        )

    # ========================================================
    # IMPORTANT INTERPRETATION
    # ========================================================

    section(
        "9. INTERPRETATION BOUNDARY"
    )

    print(
        "This experiment does NOT prove that the scaler "
        "is incorrectly implemented."
    )

    print(
        "It tests whether the observed transformed "
        "system_level behavior is repeated across "
        "available real Wazuh events."
    )

    print(
        "A final production change must NOT be made "
        "solely from this diagnostic."
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {

        "experiment":
            "92057_cross_event_validation_v1",

        "diagnostic_only":
            True,

        "production_modified":
            False,

        "model_retrained":
            False,

        "thresholds_modified":
            False,

        "victim_contacted":
            False,

        "mitigation_executed":
            False,

        "artifact_types": {

            "model":
                type(model).__name__,

            "scaler":
                type(scaler).__name__,
        },

        "matrix_layout": {

            "categorical":
                {
                    "start": 0,
                    "count": 162,
                },

            "numerical":
                {
                    "start": 162,
                    "count": 58,
                },

            "tfidf":
                {
                    "start": 220,
                    "count": 6699,
                },

            "total":
                6919,
        },

        "system_level_matrix_index":
            matrix_index,

        "events_analyzed":
            results,

        "preferred_events":
            preferred,

        "classification_summary": {

            "T1070_wins":
                t1070_wins,

            "T1059.001_wins":
                t1059_wins,

            "other_wins":
                other_wins,
        },

        "classification_rows":
            classification_rows,

        "final_decision":
            decision,

        "assessment":
            assessment,

        "interpretation_boundary":
            [
                "Diagnostic only.",
                "No production change was made.",
                "Cross-event evidence does not by itself "
                "justify retraining or feature modification.",
            ],
    }

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            default=str,
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    section(
        "10. STAGE 6 COMPLETE"
    )

    print(
        "Decision:",
        decision
    )

    print(
        "Production modified: NO"
    )

    print(
        "Victim contacted: NO"
    )

    print(
        "Mitigation executed: NO"
    )

    print()
    print(
        "Report:"
    )

    print(
        REPORT_FILE
    )


if __name__ == "__main__":

    main()