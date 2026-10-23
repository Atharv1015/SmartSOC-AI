import sys
import json
import numpy as np

sys.path.insert(0, r".\src")

from production_inference_windows_apt_v4_1 import SmartSOCInference
import production_inference_windows_apt_v4_1_aligned as aligned


RAW_WAZUH_EVENT = {
    "timestamp": "2026-10-04T11:52:58.841+0000",

    "agent": {
        "id": "001",
        "name": "SmartSOC-Windows-Victim",
        "ip": "192.168.50.20"
    },

    "rule": {
        "id": "92057",
        "level": 12,
        "description": (
            "Powershell.exe spawned a powershell process "
            "which executed a base64 encoded command"
        ),
        "groups": [
            "sysmon",
            "sysmon_eid1_detections",
            "windows"
        ],
        "mitre": {
            "id": ["T1059.001"],
            "tactic": ["Execution"]
        }
    },

    "data": {
        "win": {
            "eventdata": {
                "image":
                    r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",

                "parentImage":
                    r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",

                "commandLine":
                    r'"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" '
                    r'-EncodedCommand '
                    r'UwB0AGEAcgB0AC0AUAByAG8AYwBlAHMAcwAgAG4AbwB0AGUAcABhAGQALgBlAHgAZQA=',

                "parentCommandLine":
                    r'"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"',

                "processId": "5768",
                "parentProcessId": "1940",

                "user":
                    r"WINDOWS-10\vboxuser",

                "integrityLevel": "High",

                "hashes":
                    "SHA256=9785001B0DCF755EDDB8AF294A373C0B87B2498660F724E76C4D53F9C217C7A3"
            }
        }
    }
}


TARGET_TECHNIQUES = {
    "T1070",
    "T1059.001",
}


def get_score(prediction_result, technique):
    """
    Extract one technique's probability from aligned.predict().
    """

    for item in prediction_result.get("all_scores", []):

        if item.get("technique") == technique:
            return float(item.get("confidence", 0.0))

    return 0.0


def run_prediction(matrix, artifacts):
    """
    Run the frozen model directly against an in-memory
    ablated feature matrix.

    No model/artifact modification occurs.
    """

    model = artifacts["model"]
    label_encoder = artifacts["label_encoder"]
    thresholds = artifacts["thresholds"]

    probabilities = model.predict_proba(matrix)

    # OneVsRest normally returns (n_samples, n_classes).
    probabilities = np.asarray(probabilities)

    if probabilities.ndim != 2:
        raise RuntimeError(
            f"Unexpected probability shape: {probabilities.shape}"
        )

    row = probabilities[0]

    classes = list(label_encoder.classes_)

    if len(classes) != len(row):
        raise RuntimeError(
            "Probability/class mismatch: "
            f"{len(row)} probabilities vs "
            f"{len(classes)} classes"
        )

    all_scores = []

    for index, technique in enumerate(classes):

        confidence = float(row[index])

        threshold = float(
            thresholds.get(
                technique,
                0.5
            )
        )

        all_scores.append(
            {
                "technique": technique,
                "confidence": confidence,
                "threshold": threshold,
                "detected": confidence >= threshold,
            }
        )

    all_scores.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )

    detected = [
        item
        for item in all_scores
        if item["detected"]
    ]

    return {
        "detected": detected,
        "all_scores": all_scores,
    }


def print_target_scores(name, result):
    """
    Print T1070 and T1059.001 for one experiment.
    """

    t1070 = get_score(
        result,
        "T1070"
    )

    t1059 = get_score(
        result,
        "T1059.001"
    )

    winner = (
        "T1070"
        if t1070 >= t1059
        else "T1059.001"
    )

    print(
        f"{name:32s} "
        f"T1070={t1070:.9f} "
        f"T1059.001={t1059:.9f} "
        f"winner={winner}"
    )

    return {
        "T1070": t1070,
        "T1059.001": t1059,
        "winner": winner,
    }


print("=" * 90)
print("SmartSOC v4.1 — 92057 FEATURE ABLATION")
print("=" * 90)

print("\nIMPORTANT:")
print("This is a diagnostic only.")
print("Frozen model/artifacts are NOT modified.")
print("All ablations are performed on in-memory feature matrices.")


# ------------------------------------------------------------------
# 1. LOAD PRODUCTION ENGINE
# ------------------------------------------------------------------

print("\n[1] LOADING PRODUCTION ENGINE")

engine = SmartSOCInference()

artifacts = engine.artifacts

print("Production artifacts loaded.")

print(
    "Model features:",
    artifacts["model"].n_features_in_
)

print(
    "MITRE techniques:",
    len(artifacts["label_encoder"].classes_)
)


# ------------------------------------------------------------------
# 2. NORMALIZE + BUILD EXACT FEATURES
# ------------------------------------------------------------------

print("\n[2] BUILDING EXACT v4.1 FEATURE REPRESENTATION")

normalized = aligned.normalize_wazuh_event(
    RAW_WAZUH_EVENT
)

feature_result = aligned.build_feature_matrix(
    normalized,
    artifacts
)

original_matrix = feature_result["matrix"].copy()

print(
    "Original matrix shape:",
    original_matrix.shape
)

print(
    "Original non-zero:",
    original_matrix.nnz
)


# ------------------------------------------------------------------
# 3. FEATURE BLOCK INFORMATION
# ------------------------------------------------------------------

# v4.1 feature layout:
#
#   categorical encoded = 162
#   numerical scaled    = 58
#   TF-IDF              = 6699
#
# Total = 6919

CATEGORICAL_START = 0
CATEGORICAL_END = 162

NUMERICAL_START = 162
NUMERICAL_END = 162 + 58

TFIDF_START = NUMERICAL_END
TFIDF_END = 6919

print("\nFeature layout:")
print(
    f"  categorical : [{CATEGORICAL_START}:{CATEGORICAL_END}]"
)

print(
    f"  numerical   : [{NUMERICAL_START}:{NUMERICAL_END}]"
)

print(
    f"  TF-IDF      : [{TFIDF_START}:{TFIDF_END}]"
)


# ------------------------------------------------------------------
# 4. ORIGINAL PREDICTION
# ------------------------------------------------------------------

print("\n[3] BASELINE")

baseline_result = aligned.predict(
    feature_result,
    artifacts
)

baseline_scores = print_target_scores(
    "BASELINE — ALL FEATURES",
    baseline_result
)


# ------------------------------------------------------------------
# 5. ABLATION HELPER
# ------------------------------------------------------------------

def run_ablation(
    name,
    zero_start=None,
    zero_end=None,
    zero_indices=None,
):
    """
    Create a copy of the original sparse matrix,
    zero selected columns, and run the frozen model.

    The original matrix is never changed.
    """

    matrix = original_matrix.copy()

    if zero_start is not None and zero_end is not None:

        matrix[:, zero_start:zero_end] = 0

    if zero_indices is not None:

        for index in zero_indices:
            matrix[:, index] = 0

    matrix.eliminate_zeros()

    result = run_prediction(
        matrix,
        artifacts
    )

    scores = print_target_scores(
        name,
        result
    )

    return {
        "name": name,
        "non_zero_features": int(matrix.nnz),
        "scores": scores,
        "top_predictions": result["all_scores"][:5],
    }


# ------------------------------------------------------------------
# 6. SIX ABLATIONS
# ------------------------------------------------------------------

results = []

print("\n[4] FEATURE ABLATIONS")
print("-" * 90)


# ABLATION 1
# Remove TF-IDF completely.

results.append(
    run_ablation(
        "ABLATION 1 — NO TF-IDF",
        TFIDF_START,
        TFIDF_END,
    )
)


# ABLATION 2
# Remove categorical features completely.

results.append(
    run_ablation(
        "ABLATION 2 — NO CATEGORICAL",
        CATEGORICAL_START,
        CATEGORICAL_END,
    )
)


# ABLATION 3
# Remove numerical features completely.

results.append(
    run_ablation(
        "ABLATION 3 — NO NUMERICAL",
        NUMERICAL_START,
        NUMERICAL_END,
    )
)


# ABLATION 4
# Remove PowerShell-related engineered numerical features.
#
# These are the known feature names in the 58-dimensional
# numerical feature block.

numerical_features = list(
    feature_result["features"].keys()
)

# Determine numerical feature ordering from the production
# numerical dataframe rather than guessing.

numerical_columns = list(
    feature_result["numerical_df"].columns
)

print("\nNumerical feature order:")
for index, name in enumerate(numerical_columns):
    print(
        f"  {index:02d}: {name}"
    )


powershell_feature_names = {
    "uses_powershell",
    "has_encoded_command",
    "has_obfuscation_hint",
    "administrative_powershell_context",
}

powershell_indices = []

for index, name in enumerate(numerical_columns):

    if name in powershell_feature_names:

        powershell_indices.append(
            NUMERICAL_START + index
        )


results.append(
    run_ablation(
        "ABLATION 4 — NO POWERSHELL SIGNALS",
        zero_indices=powershell_indices,
    )
)


# ABLATION 5
# Remove process identity from categorical block.
#
# The exact categorical encoding is one-hot encoded.
# We therefore remove the entire categorical block here
# as a conservative diagnostic rather than manually guessing
# encoded columns.

# This is intentionally equivalent to Ablation 2 and is not
# counted as an independent result. Instead we perform a
# targeted numerical process-context ablation below.

process_context_names = {
    "same_process_parent",
    "strong_behavior_signal",
    "suspicious_parent_wazuh",
}

process_context_indices = []

for index, name in enumerate(numerical_columns):

    if name in process_context_names:

        process_context_indices.append(
            NUMERICAL_START + index
        )


results.append(
    run_ablation(
        "ABLATION 5 — NO PROCESS-CONTEXT SIGNALS",
        zero_indices=process_context_indices,
    )
)


# ABLATION 6
# Remove encoded/obfuscation signals specifically.

encoded_feature_names = {
    "has_encoded_command",
    "has_obfuscation_hint",
}

encoded_indices = []

for index, name in enumerate(numerical_columns):

    if name in encoded_feature_names:

        encoded_indices.append(
            NUMERICAL_START + index
        )


results.append(
    run_ablation(
        "ABLATION 6 — NO ENCODED/OBFUSCATION",
        zero_indices=encoded_indices,
    )
)


# ------------------------------------------------------------------
# 7. SUMMARY TABLE
# ------------------------------------------------------------------

print("\n" + "=" * 90)
print("ABLATION SUMMARY")
print("=" * 90)

print(
    f"{'Experiment':32s} "
    f"{'T1070':>14s} "
    f"{'T1059.001':>14s} "
    f"{'Winner':>14s}"
)

print("-" * 90)

print(
    f"{'BASELINE':32s} "
    f"{baseline_scores['T1070']:14.9f} "
    f"{baseline_scores['T1059.001']:14.9f} "
    f"{baseline_scores['winner']:>14s}"
)

for item in results:

    scores = item["scores"]

    print(
        f"{item['name']:32s} "
        f"{scores['T1070']:14.9f} "
        f"{scores['T1059.001']:14.9f} "
        f"{scores['winner']:>14s}"
    )


# ------------------------------------------------------------------
# 8. TOP PREDICTIONS FOR EACH ABLATION
# ------------------------------------------------------------------

print("\n" + "=" * 90)
print("TOP-5 PREDICTIONS PER ABLATION")
print("=" * 90)

for item in results:

    print(
        "\n" + item["name"]
    )

    print(
        "Non-zero features:",
        item["non_zero_features"]
    )

    for prediction in item["top_predictions"]:

        print(
            f"  {prediction['technique']:12s} "
            f"{prediction['confidence']:.9f}"
        )


# ------------------------------------------------------------------
# 9. SAVE DIAGNOSTIC JSON
# ------------------------------------------------------------------

output = {
    "event": {
        "rule_id": "92057",
        "ground_truth": "T1059.001",
    },

    "model": {
        "version": "v4.1",
        "features": 6919,
        "techniques": 45,
    },

    "baseline": baseline_scores,

    "ablations": results,

    "feature_layout": {
        "categorical": 162,
        "numerical": 58,
        "tfidf": 6699,
        "total": 6919,
    },

    "safety": {
        "production_files_modified": False,
        "model_modified": False,
        "thresholds_modified": False,
        "artifacts_modified": False,
    },
}


output_path = (
    r".\reports\integration"
    r"\92057_feature_ablation_v1.json"
)

import os

os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        output,
        handle,
        indent=2,
        default=str
    )


print("\n" + "=" * 90)
print("ABLATION COMPLETE")
print("=" * 90)

print(
    "\nDiagnostic report written to:"
)

print(
    output_path
)

print(
    "\nProduction model/artifacts were not modified."
)

print(
    "Do NOT change production inference based on this result yet."
)