import os
import json
import sys
from pathlib import Path
from typing import Any

import requests


WEBHOOK_URL = os.environ.get("SMARTSOC_SHUFFLE_WEBHOOK_URL", "").strip()

PROJECT_ROOT = Path(r"C:\Users\athar\SmartSOC-AI")
INPUT_FILE = PROJECT_ROOT / "reports" / "integration" / "response_plan_v1.json"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "integration"
OUTPUT_FILE = OUTPUT_DIR / "shuffle_dry_run_result_v1.json"


def find_first_value(data: Any, target_key: str) -> Any:
    """Recursively find the first occurrence of a key."""
    if isinstance(data, dict):
        if target_key in data:
            return data[target_key]

        for value in data.values():
            result = find_first_value(value, target_key)
            if result is not None:
                return result

    elif isinstance(data, list):
        for item in data:
            result = find_first_value(item, target_key)
            if result is not None:
                return result

    return None


def main() -> int:
    if not INPUT_FILE.exists():
        print(f"[ERROR] Response plan not found: {INPUT_FILE}")
        return 1

    try:
        with INPUT_FILE.open("r", encoding="utf-8") as f:
            response_plan = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid response-plan JSON: {exc}")
        return 1

    risk_score = find_first_value(response_plan, "risk_score")
    severity = find_first_value(response_plan, "severity")

    # Preserve the existing response plan and explicitly mark this
    # integration as dry-run / log-only.
    payload = {
        "source": "SmartSOC",
        "integration": "Shuffle",
        "mode": "DRY_RUN",
        "execution_policy": "LOG_ONLY",
        "risk_score": risk_score,
        "severity": severity,
        "response_plan": response_plan,
        "safety": {
            "automatic_execution": False,
            "automatic_containment": False,
            "endpoint_modified": False,
            "shuffle_called": True,
            "caldera_called": False,
        },
    }

    print("[INFO] Sending SmartSOC response plan to Shuffle...")
    print("[INFO] Shuffle webhook configured.")
    print(f"[INFO] Risk score: {risk_score}")
    print(f"[INFO] Severity: {severity}")
    print("[INFO] Mode: DRY_RUN")
    print("[INFO] Policy: LOG_ONLY")

    try:
        if not WEBHOOK_URL.startswith(("http://", "https://")):
            print("[ERROR] Shuffle webhook is not configured.")
            return 1

        response = requests.post(
            WEBHOOK_URL,
            json=payload,
            timeout=15,
        )
    except requests.RequestException as exc:
        print(f"[ERROR] Shuffle request failed: {exc}")
        return 2

    result = {
        "schema": "SmartSOC-Shuffle-DryRun-v1",
        "http_status": response.status_code,
        "shuffle_response": response.text,
        "webhook_configured": bool(WEBHOOK_URL),
        "mode": "DRY_RUN",
        "execution_policy": "LOG_ONLY",
        "risk_score": risk_score,
        "severity": severity,
        "automatic_execution": False,
        "automatic_containment": False,
        "endpoint_modified": False,
        "caldera_called": False,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[INFO] HTTP status: {response.status_code}")
    print(f"[INFO] Shuffle response: {response.text}")
    print(f"[INFO] Evidence written to: {OUTPUT_FILE}")

    if response.ok:
        print("[SUCCESS] SmartSOC -> Shuffle delivery succeeded.")
        print("[SAFETY] DRY_RUN / LOG_ONLY. No endpoint action performed.")
        return 0

    print("[ERROR] Shuffle rejected the request.")
    return 3


if __name__ == "__main__":
    sys.exit(main())