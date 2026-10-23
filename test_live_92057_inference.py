import sys
import json

sys.path.insert(0, r".\src")

import production_inference_windows_apt_v4_1_aligned as aligned


event = {
    "source": "wazuh_live_forwarder_v2",
    "execution_mode": "LOG_ONLY",

    "wazuh": {
        "alert_id": "1791114778.1034574",
        "timestamp": "2026-10-04T11:52:58.841+0000",
        "rule_id": "92057",
        "rule_level": 12,
        "description": (
            "Powershell.exe spawned a powershell process which "
            "executed a base64 encoded command"
        ),
        "groups": [
            "sysmon",
            "sysmon_eid1_detections",
            "windows"
        ]
    },

    "agent": {
        "id": "001",
        "name": "SmartSOC-Windows-Victim",
        "ip": "192.168.50.20"
    },

    "mitre": {
        "techniques": ["T1059.001"],
        "tactics": ["Execution"]
    },

    "windows": {
        "computer": "",
        "event_id": "",
        "channel": "",
        "process_id": "5768",

        "process_name":
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",

        "command_line":
            r'"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" '
            r'-EncodedCommand '
            r'UwB0AGEAcgB0AC0AUAByAG8AYwBlAHMAcwAgAG4AbwB0AGUAcABhAGQALgBl'
            r'AHgAZQA=',

        "parent_process_id": "1940",

        "parent_process_name":
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",

        "parent_command_line":
            r'"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"',

        "user": r"WINDOWS-10\vboxuser",
        "integrity_level": "High",

        "hashes":
            "SHA256=9785001B0DCF755EDDB8AF294A373C0B87B2498660F724E76C4D53F9C217C7A3"
    }
}


print("=" * 80)
print("SmartSOC v4.1 — ALIGNED 92057 PIPELINE DIAGNOSTIC")
print("=" * 80)

# ------------------------------------------------------------------
# 1. Normalize
# ------------------------------------------------------------------

print("\n[1] NORMALIZATION")

normalized = aligned.normalize_wazuh_event(event)

print(json.dumps(normalized, indent=2, default=str))


# ------------------------------------------------------------------
# 2. Derive features
# ------------------------------------------------------------------

print("\n[2] FEATURE DERIVATION")

features = aligned.derive_features(normalized)

print("Feature count:", len(features))

print("\nImportant derived features:")

for key, value in features.items():
    key_lower = str(key).lower()

    if any(term in key_lower for term in [
        "process",
        "parent",
        "command",
        "powershell",
        "encoded",
        "script",
        "mitre",
        "rule",
        "description",
        "behavior"
    ]):
        print(f"{key}: {value}")


# ------------------------------------------------------------------
# 3. Load frozen production artifacts
# ------------------------------------------------------------------

print("\n[3] LOADING PRODUCTION ARTIFACTS")

artifacts = aligned.load_artifacts()

print("Artifacts loaded successfully.")

if "model" in artifacts:
    print("Model:", type(artifacts["model"]).__name__)

if "label_encoder" in artifacts:
    encoder = artifacts["label_encoder"]

    try:
        print(
            "Label encoder classes:",
            len(encoder.classes_)
        )
    except Exception:
        pass


# ------------------------------------------------------------------
# 4. Build feature matrix
# ------------------------------------------------------------------

print("\n[4] BUILDING FEATURE MATRIX")

feature_result = aligned.build_feature_matrix(
    normalized,
    artifacts
)

print(
    "Feature matrix keys:",
    list(feature_result.keys())
)

for key, value in feature_result.items():

    if key == "X":
        try:
            print(
                "X shape:",
                value.shape
            )
        except Exception:
            print("X:", type(value).__name__)

    elif key not in {
        "vectorizer",
        "structured_encoder",
        "numerical_scaler"
    }:
        print(f"{key}: {value}")


# ------------------------------------------------------------------
# 5. Predict
# ------------------------------------------------------------------

print("\n[5] RUNNING FROZEN MODEL PREDICTION")

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


# ------------------------------------------------------------------
# 6. Final result
# ------------------------------------------------------------------

print("\n[6] BUILDING FINAL RESULT")

result = aligned.build_result_json(
    normalized,
    feature_result,
    prediction_result,
    artifacts
)

print(
    json.dumps(
        result,
        indent=2,
        default=str
    )
)


# ------------------------------------------------------------------
# 7. Direct comparison
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

print(
    "Wazuh MITRE:",
    event["mitre"]["techniques"]
)

print(
    "SmartSOC predictions:"
)

for prediction in result.get("predictions", []):
    print(
        "  Technique   :",
        prediction.get("technique")
    )
    print(
        "  Confidence  :",
        prediction.get("confidence")
    )
    print(
        "  Threshold   :",
        prediction.get("threshold")
    )
    print(
        "  Detected    :",
        prediction.get("detected")
    )
    print()


print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)