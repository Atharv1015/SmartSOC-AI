import sys
import json

sys.path.insert(0, r".\src")

from production_inference_windows_apt_v4_1 import SmartSOCInference


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
print("SmartSOC v4.1 — RAW WAZUH 92057 REPRODUCTION")
print("=" * 80)

print("\n[1] INPUT EVENT")
print(json.dumps(RAW_WAZUH_EVENT, indent=2))

print("\n[2] LOADING EXISTING PRODUCTION INFERENCE")

engine = SmartSOCInference()

print("Production inference loaded.")

print("\n[3] RUNNING SmartSOCInference.infer()")

result = engine.infer(RAW_WAZUH_EVENT)

print("\n[4] COMPLETE RESULT")
print(json.dumps(result, indent=2, default=str))

print("\n" + "=" * 80)
print("KEY COMPARISON")
print("=" * 80)

print("\nWazuh/MITRE:")
print("  T1059.001")

print("\nSmartSOC predictions:")

for item in result.get("predictions", []):
    print(
        "  {}  confidence={}  threshold={}".format(
            item.get("technique"),
            item.get("confidence"),
            item.get("threshold")
        )
    )

print("\nSelected probability scores:")

for item in result.get("top_probability_scores", []):
    if item.get("technique") in {
        "T1059.001",
        "T1070",
        "T1070.004",
        "T1489"
    }:
        print(
            "  {}  confidence={}  detected={}".format(
                item.get("technique"),
                item.get("confidence"),
                item.get("detected")
            )
        )

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
