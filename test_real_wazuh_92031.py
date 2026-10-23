import json
import sys

sys.path.insert(0, r"C:\Users\athar\SmartSOC-AI\src")

from production_inference_windows_apt_v4_1 import SmartSOCInference


EVENT_FILE = r"C:\Users\athar\SmartSOC-AI\wazuh_92031.json"


def main():
    with open(EVENT_FILE, "r", encoding="utf-8") as f:
        wazuh_event = json.load(f)

    print("=" * 78)
    print("REAL WAZUH 92031 -> SMARTSOC v4.1")
    print("=" * 78)

    print(f"Rule ID     : {wazuh_event.get('rule', {}).get('id')}")
    print(f"Rule Level  : {wazuh_event.get('rule', {}).get('level')}")
    print(f"Agent       : {wazuh_event.get('agent', {}).get('name')}")
    print(f"Agent IP    : {wazuh_event.get('agent', {}).get('ip')}")

    engine = SmartSOCInference()

    result = engine.infer(wazuh_event)

    print()
    print("INFER_OK")
    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()