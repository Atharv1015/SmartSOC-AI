import sys
import json
import inspect
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


print("=" * 80)
print("SmartSOC v4.1 — 92057 EXACT PIPELINE FORENSICS")
print("=" * 80)


# ------------------------------------------------------------------
# 1. LOAD THE EXISTING PRODUCTION ENGINE
# ------------------------------------------------------------------

print("\n[1] LOADING EXISTING PRODUCTION ENGINE")

engine = SmartSOCInference()

print("Production engine loaded.")


# ------------------------------------------------------------------
# 2. INSPECT ENGINE ARTIFACT ATTRIBUTES
# ------------------------------------------------------------------

print("\n[2] ENGINE ATTRIBUTES")

candidate_names = [
    "artifacts",
    "model",
    "structured_encoder",
    "numerical_scaler",
    "tfidf_vectorizer",
    "multilabel_encoder",
    "feature_schema",
    "thresholds",
    "manifest",
]

for name in candidate_names:

    if hasattr(engine, name):

        value = getattr(engine, name)

        print(f"\n{name}:")
        print("  type:", type(value))

        if isinstance(value, dict):
            print("  keys:", list(value.keys()))

        elif hasattr(value, "shape"):
            print("  shape:", value.shape)

        elif hasattr(value, "classes_"):
            try:
                print("  classes:", len(value.classes_))
            except Exception:
                pass

        elif hasattr(value, "n_features_in_"):
            print("  n_features_in_:", value.n_features_in_)

    else:

        print(f"\n{name}: NOT FOUND")


# ------------------------------------------------------------------
# 3. FIND THE ACTUAL ARTIFACT DICTIONARY
# ------------------------------------------------------------------

print("\n[3] SEARCHING FOR ARTIFACT DICTIONARY")

artifacts = None

if hasattr(engine, "artifacts"):
    artifacts = engine.artifacts

if artifacts is None:

    print("engine.artifacts not present.")

    print("\nAvailable instance attributes:")

    for name in sorted(vars(engine).keys()):
        print(" ", name)

else:

    print("Artifact dictionary found.")
    print("Artifact keys:")

    for key in artifacts.keys():
        print(" ", key)


# ------------------------------------------------------------------
# 4. NORMALIZE
# ------------------------------------------------------------------

print("\n[4] NORMALIZATION")

normalized = aligned.normalize_wazuh_event(RAW_WAZUH_EVENT)

print(json.dumps(normalized, indent=2, default=str))


# ------------------------------------------------------------------
# 5. DERIVE FEATURES
# ------------------------------------------------------------------

print("\n[5] DERIVED FEATURES")

features = aligned.derive_features(normalized)

for key in sorted(features.keys()):
    print(
        f"  {key:35s}: {repr(features[key])}"
    )


# ------------------------------------------------------------------
# 6. BEHAVIOR TEXT
# ------------------------------------------------------------------

print("\n[6] BEHAVIOR TEXT")

behavior_text = aligned.build_behavior_text(
    normalized,
    features
)

print(behavior_text)


# ------------------------------------------------------------------
# 7. BUILD EXACT PRODUCTION FEATURE RESULT
# ------------------------------------------------------------------

if artifacts is None:

    print("\n[7] ERROR")
    print("Could not locate engine.artifacts.")
    print("No production feature reconstruction performed.")

else:

    print("\n[7] EXACT build_feature_matrix()")

    feature_result = aligned.build_feature_matrix(
        normalized,
        artifacts
    )

    print("\nFeature-result type:")
    print(type(feature_result))

    if isinstance(feature_result, dict):

        print("\nFeature-result keys:")

        for key in feature_result.keys():
            print(" ", key)

    print("\nFeature-result:")

    print(
        json.dumps(
            {
                k: (
                    {
                        "type": str(type(v)),
                        "shape": list(v.shape)
                        if hasattr(v, "shape")
                        else None
                    }
                    if hasattr(v, "shape")
                    else v
                )
                for k, v in feature_result.items()
            },
            indent=2,
            default=str
        )
    )


    # --------------------------------------------------------------
    # 8. FEATURE MATRIX INSPECTION
    # --------------------------------------------------------------

    print("\n[8] FEATURE MATRIX INSPECTION")

    matrix = None

    if isinstance(feature_result, dict):

        for key in [
            "feature_matrix",
            "X",
            "features",
            "final_features",
            "matrix",
        ]:

            if key in feature_result:

                candidate = feature_result[key]

                if hasattr(candidate, "shape"):

                    matrix = candidate

                    print(
                        f"Using feature-result key: {key}"
                    )

                    break

    if matrix is None:

        print(
            "Could not automatically locate final feature matrix."
        )

    else:

        arr = np.asarray(matrix)

        print("Shape       :", arr.shape)
        print(
            "Non-zero    :",
            int(np.count_nonzero(arr))
        )
        print(
            "Minimum     :",
            float(np.min(arr))
        )
        print(
            "Maximum     :",
            float(np.max(arr))
        )
        print(
            "Mean        :",
            float(np.mean(arr))
        )


    # --------------------------------------------------------------
    # 9. RUN EXACT PRODUCTION PREDICT()
    # --------------------------------------------------------------

    print("\n[9] EXACT aligned.predict()")

    prediction_result = aligned.predict(
        feature_result,
        artifacts
    )

    print(
        json.dumps(
            prediction_result,
            indent=2,
            default=str
        )
    )


    # --------------------------------------------------------------
    # 10. BUILD EXACT RESULT
    # --------------------------------------------------------------

    print("\n[10] EXACT build_result_json()")

    final_result = aligned.build_result_json(
        normalized,
        feature_result,
        prediction_result,
        artifacts
    )

    print(
        json.dumps(
            final_result,
            indent=2,
            default=str
        )
    )


# ------------------------------------------------------------------
# 11. PRODUCTION WRAPPER COMPARISON
# ------------------------------------------------------------------

print("\n[11] PRODUCTION WRAPPER RESULT")

wrapper_result = engine.infer(RAW_WAZUH_EVENT)

print(
    json.dumps(
        wrapper_result,
        indent=2,
        default=str
    )
)


# ------------------------------------------------------------------
# 12. FINAL
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("EXACT PIPELINE FORENSICS COMPLETE")
print("=" * 80)