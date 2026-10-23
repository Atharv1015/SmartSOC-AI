import sys
import json
import numpy as np
import joblib

sys.path.insert(0, r".\src")

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
print("SmartSOC v4.1 — 92057 FEATURE FORENSICS")
print("=" * 80)


# ------------------------------------------------------------------
# 1. NORMALIZE RAW WAZUH EVENT
# ------------------------------------------------------------------

print("\n[1] NORMALIZING RAW WAZUH EVENT")

normalized = aligned.normalize_wazuh_event(RAW_WAZUH_EVENT)

print(json.dumps(normalized, indent=2, default=str))


# ------------------------------------------------------------------
# 2. DERIVE FEATURES
# ------------------------------------------------------------------

print("\n[2] DERIVING FEATURES")

derived = aligned.derive_features(normalized)

print("\nDerived feature keys:")

for key in sorted(derived.keys()):
    print(f"  {key}: {repr(derived[key])}")


# ------------------------------------------------------------------
# 3. BUILD BEHAVIOR TEXT
# ------------------------------------------------------------------

print("\n[3] BEHAVIOR TEXT")

try:
    behavior_text = aligned.build_behavior_text(
        normalized,
        derived
    )
except TypeError:
    # Some aligned versions may expect keyword arguments.
    behavior_text = aligned.build_behavior_text(
        normalized,
        features=derived
    )

print(behavior_text)


# ------------------------------------------------------------------
# 4. IMPORTANT TOKEN CHECK
# ------------------------------------------------------------------

print("\n[4] IMPORTANT TOKEN CHECK")

tokens = [
    "powershell",
    "encodedcommand",
    "start-process",
    "notepad",
    "downloadstring",
    "invoke-webrequest",
    "base64",
    "net.exe",
    "net1.exe",
]

behavior_lower = str(behavior_text).lower()

for token in tokens:
    print(
        f"{token:20s}: "
        f"{'FOUND' if token.lower() in behavior_lower else 'NOT FOUND'}"
    )


# ------------------------------------------------------------------
# 5. BUILD FEATURE MATRIX
# ------------------------------------------------------------------

print("\n[5] BUILDING 6919-DIMENSION FEATURE MATRIX")

X, metadata = aligned.build_feature_matrix([normalized])

print("\nFeature shape:")
print(X.shape)

print("\nMetadata:")
print(json.dumps(metadata, indent=2, default=str))


# ------------------------------------------------------------------
# 6. FEATURE STATISTICS
# ------------------------------------------------------------------

print("\n[6] FEATURE STATISTICS")

x = np.asarray(X[0])

print("Total features :", len(x))
print("Non-zero       :", int(np.count_nonzero(x)))
print("Minimum        :", float(np.min(x)))
print("Maximum        :", float(np.max(x)))
print("Mean           :", float(np.mean(x)))


# ------------------------------------------------------------------
# 7. TF-IDF VOCABULARY CHECK
# ------------------------------------------------------------------

print("\n[7] TF-IDF VOCABULARY CHECK")

vectorizer_path = (
    r".\models\windows_apt_v4_1\production\tfidf_vectorizer.joblib"
)

vectorizer = joblib.load(vectorizer_path)

print("Vocabulary size:", len(vectorizer.vocabulary_))

print("\nImportant vocabulary terms:")

for token in tokens:

    token_lower = token.lower()

    if token_lower in vectorizer.vocabulary_:

        print(
            f"  {token:20s} -> "
            f"index {vectorizer.vocabulary_[token_lower]}"
        )

    else:

        print(
            f"  {token:20s} -> NOT IN VOCABULARY"
        )


# ------------------------------------------------------------------
# 8. FINAL
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("FEATURE FORENSICS COMPLETE")
print("=" * 80)