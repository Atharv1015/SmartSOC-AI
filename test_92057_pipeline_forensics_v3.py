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


print("=" * 80)
print("SmartSOC v4.1 — 92057 EXACT PIPELINE FORENSICS v3")
print("=" * 80)


# ------------------------------------------------------------------
# 1. LOAD PRODUCTION ENGINE
# ------------------------------------------------------------------

print("\n[1] LOAD PRODUCTION ENGINE")

engine = SmartSOCInference()

artifacts = engine.artifacts

print("Artifacts loaded:")
print(" ", list(artifacts.keys()))


# ------------------------------------------------------------------
# 2. NORMALIZE
# ------------------------------------------------------------------

print("\n[2] NORMALIZATION")

normalized = aligned.normalize_wazuh_event(
    RAW_WAZUH_EVENT
)

print(
    json.dumps(
        normalized,
        indent=2,
        default=str
    )
)


# ------------------------------------------------------------------
# 3. DERIVE FEATURES
# ------------------------------------------------------------------

print("\n[3] DERIVED FEATURES")

features = aligned.derive_features(
    normalized
)

for key in sorted(features.keys()):
    print(
        f"  {key:35s}: {repr(features[key])}"
    )


# ------------------------------------------------------------------
# 4. BUILD BEHAVIOR TEXT
# ------------------------------------------------------------------

print("\n[4] BEHAVIOR TEXT")

behavior_text = aligned.build_behavior_text(
    normalized,
    features
)

print(behavior_text)


# ------------------------------------------------------------------
# 5. BUILD EXACT FEATURE RESULT
# ------------------------------------------------------------------

print("\n[5] EXACT FEATURE MATRIX")

feature_result = aligned.build_feature_matrix(
    normalized,
    artifacts
)

matrix = feature_result["matrix"]
tfidf = feature_result["tfidf_features"]

print("Matrix type :", type(matrix))
print("Matrix shape:", matrix.shape)

print("TF-IDF type :", type(tfidf))
print("TF-IDF shape:", tfidf.shape)


# ------------------------------------------------------------------
# 6. SPARSE MATRIX STATISTICS
# ------------------------------------------------------------------

print("\n[6] SPARSE FEATURE STATISTICS")

print(
    "Final matrix non-zero:",
    matrix.nnz
)

print(
    "TF-IDF non-zero:",
    tfidf.nnz
)

print(
    "Final matrix dtype:",
    matrix.dtype
)

print(
    "TF-IDF dtype:",
    tfidf.dtype
)


# ------------------------------------------------------------------
# 7. TF-IDF TERMS ACTUALLY PRESENT
# ------------------------------------------------------------------

print("\n[7] TF-IDF TERMS PRESENT IN EVENT")

vectorizer = artifacts["tfidf"]

tfidf_row = tfidf.getrow(0)

feature_names = vectorizer.get_feature_names_out()

terms_present = []

for index, value in zip(
    tfidf_row.indices,
    tfidf_row.data
):

    terms_present.append(
        (
            feature_names[index],
            float(value),
            int(index)
        )
    )


terms_present.sort(
    key=lambda x: x[1],
    reverse=True
)

print(
    "Non-zero TF-IDF terms:",
    len(terms_present)
)

for term, value, index in terms_present[:50]:

    print(
        f"  {term:40s} "
        f"value={value:.8f} "
        f"index={index}"
    )


# ------------------------------------------------------------------
# 8. IMPORTANT TERM CHECK
# ------------------------------------------------------------------

print("\n[8] IMPORTANT TF-IDF TERM CHECK")

important_terms = [
    "powershell",
    "encodedcommand",
    "start-process",
    "notepad",
    "downloadstring",
    "invoke-webrequest",
    "base64",
]

vocabulary = vectorizer.vocabulary_

for term in important_terms:

    idx = vocabulary.get(term)

    if idx is None:

        print(
            f"  {term:25s}: NOT IN VOCABULARY"
        )

        continue

    value = tfidf[0, idx]

    print(
        f"  {term:25s}: "
        f"index={idx} "
        f"value={float(value):.8f}"
    )


# ------------------------------------------------------------------
# 9. EXACT ALIGNED PREDICTION
# ------------------------------------------------------------------

print("\n[9] EXACT ALIGNED predict()")

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
# 10. EXACT RESULT JSON
# ------------------------------------------------------------------

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
# 11. PRODUCTION WRAPPER
# ------------------------------------------------------------------

print("\n[11] SmartSOCInference.infer()")

wrapper_result = engine.infer(
    RAW_WAZUH_EVENT
)

print(
    json.dumps(
        wrapper_result,
        indent=2,
        default=str
    )
)


# ------------------------------------------------------------------
# 12. DIRECT COMPARISON
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("DIRECT COMPARISON")
print("=" * 80)

print("\nAligned prediction:")

for item in prediction_result.get(
    "predictions",
    []
):

    print(
        f"  {item.get('technique'):12s} "
        f"{item.get('confidence')}"
    )


print("\nProduction wrapper prediction:")

for item in wrapper_result.get(
    "predictions",
    []
):

    print(
        f"  {item.get('technique'):12s} "
        f"{item.get('confidence')}"
    )


print("\n" + "=" * 80)
print("FORENSICS COMPLETE")
print("=" * 80)