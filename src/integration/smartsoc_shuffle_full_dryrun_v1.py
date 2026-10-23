import os
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests


PROJECT_ROOT = Path(r"C:\Users\athar\SmartSOC-AI")
INTEGRATION_DIR = PROJECT_ROOT / "reports" / "integration"
SRC_DIR = PROJECT_ROOT / "src" / "integration"

WAZUH_RESULT = INTEGRATION_DIR / "wazuh_readonly_result.json"
RISK_RESULT = INTEGRATION_DIR / "risk_engine_result.json"
RESPONSE_PLAN = INTEGRATION_DIR / "response_plan_v1.json"
AUTH_GATE_RESULT = INTEGRATION_DIR / "authorization_gate_result_v1.json"
AUTH_DECISION_RESULT = INTEGRATION_DIR / "authorization_decision_v1.json"
POLICY_GUARD_RESULT = INTEGRATION_DIR / "execution_policy_guard_result_v1.json"

OUTPUT_FILE = INTEGRATION_DIR / "smartsoc_shuffle_full_dryrun_v1.json"

WEBHOOK_URL = os.environ.get("SMARTSOC_SHUFFLE_WEBHOOK_URL", "").strip()

RISK_ENGINE_SCRIPT = SRC_DIR / "risk_engine_v1.py"
RESPONSE_PLANNER_SCRIPT = SRC_DIR / "response_planner_v1.py"
AUTH_GATE_SCRIPT = SRC_DIR / "authorization_gate_v1.py"
AUTH_DECISION_SCRIPT = SRC_DIR / "authorization_decision_v1.py"
POLICY_GUARD_SCRIPT = SRC_DIR / "execution_policy_guard_v1.py"


def run_script(script_path: Path, *args: str) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    command = [sys.executable, str(script_path), *args]

    print(f"[RUN] {script_path.name} {' '.join(args)}")

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if completed.stdout:
        print(completed.stdout.rstrip())

    if completed.stderr:
        print(completed.stderr.rstrip())

    if completed.returncode != 0:
        raise RuntimeError(
            f"{script_path.name} failed with exit code {completed.returncode}"
        )


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    print("=" * 72)
    print("SmartSOC -> Shuffle Full Dry-Run Integration")
    print("=" * 72)

    INTEGRATION_DIR.mkdir(parents=True, exist_ok=True)

    if not WAZUH_RESULT.exists():
        print(f"[ERROR] Missing Wazuh/SmartSOC result: {WAZUH_RESULT}")
        print("[ERROR] Run the existing read-only Wazuh inference first.")
        return 1

    try:
        # Existing SmartSOC pipeline components.
        run_script(RISK_ENGINE_SCRIPT)
        run_script(RESPONSE_PLANNER_SCRIPT)
        run_script(AUTH_GATE_SCRIPT)

        # Intentionally DENY for this dry-run.
        run_script(AUTH_DECISION_SCRIPT, "DENY")

        run_script(POLICY_GUARD_SCRIPT)

        # Load the resulting artifacts.
        wazuh_result = load_json(WAZUH_RESULT)
        risk_result = load_json(RISK_RESULT)
        response_plan = load_json(RESPONSE_PLAN)
        auth_gate = load_json(AUTH_GATE_RESULT)
        auth_decision = load_json(AUTH_DECISION_RESULT)
        policy_guard = load_json(POLICY_GUARD_RESULT)

    except Exception as exc:
        print(f"[ERROR] Pipeline stage failed: {exc}")
        return 2

    risk_calculation = risk_result.get("risk_calculation", {})
    risk_decision = risk_result.get("decision", {})

    risk_score = risk_calculation.get("risk_score")
    severity = risk_calculation.get("severity")

    # Correct authorization schema:
    # authorization.status
    # authorization.analyst_authorized
    authorization = auth_decision.get("authorization", {})

    authorization_status = authorization.get("status")
    analyst_authorized = authorization.get("analyst_authorized")

    # Correct execution-control schema:
    # execution_control.execution_allowed
    execution_control = auth_decision.get("execution_control", {})

    auth_execution_allowed = execution_control.get("execution_allowed")

    # Prefer the policy guard's final execution decision when available.
    execution_allowed = policy_guard.get("execution_allowed")

    if execution_allowed is None:
        execution_allowed = (
            policy_guard.get("decision", "").upper() == "ALLOWED"
        )

    payload = {
        "schema": "SmartSOC-Shuffle-Full-DryRun-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": "SmartSOC",
        "integration": "Shuffle",
        "mode": "DRY_RUN",
        "execution_policy": "LOG_ONLY",

        "summary": {
            "risk_score": risk_score,
            "severity": severity,
            "recommended_action": risk_decision.get("recommended_action"),

            "authorization_required": authorization.get("required"),
            "authorization_status": authorization_status,
            "analyst_authorized": analyst_authorized,

            "authorization_execution_allowed": auth_execution_allowed,
            "policy_guard_execution_allowed": execution_allowed,
        },

        "wazuh_inference": wazuh_result,
        "risk_engine": risk_result,
        "response_plan": response_plan,
        "authorization_gate": auth_gate,
        "authorization_decision": auth_decision,
        "execution_policy_guard": policy_guard,

        "safety": {
            "automatic_execution": False,
            "automatic_containment": False,
            "endpoint_modified": False,
            "shuffle_called": True,
            "caldera_called": False,
            "dry_run": True,
            "log_only": True,
        },
    }

    print()
    print("[INFO] Final SmartSOC decision data")
    print(f"[INFO] Risk score                     : {risk_score}")
    print(f"[INFO] Severity                       : {severity}")
    print(f"[INFO] Authorization required        : {authorization.get('required')}")
    print(f"[INFO] Authorization status          : {authorization_status}")
    print(f"[INFO] Analyst authorized            : {analyst_authorized}")
    print(f"[INFO] Authorization execution       : {auth_execution_allowed}")
    print(f"[INFO] Policy guard execution        : {execution_allowed}")
    print(f"[INFO] Shuffle mode                   : DRY_RUN / LOG_ONLY")

    print()
    print("[INFO] Sending full SmartSOC result to Shuffle...")

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
        print(f"[ERROR] Shuffle connection failed: {exc}")
        return 3

    evidence = {
        "schema": "SmartSOC-Shuffle-Full-DryRun-Evidence-v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),

        "webhook_configured": bool(WEBHOOK_URL),
        "http_status": response.status_code,
        "shuffle_response": response.text,

        "mode": "DRY_RUN",
        "execution_policy": "LOG_ONLY",

        "risk_score": risk_score,
        "severity": severity,

        "authorization_required": authorization.get("required"),
        "authorization_status": authorization_status,
        "analyst_authorized": analyst_authorized,

        "authorization_execution_allowed": auth_execution_allowed,
        "policy_guard_execution_allowed": execution_allowed,

        "automatic_execution": False,
        "automatic_containment": False,
        "endpoint_modified": False,
        "caldera_called": False,
        "dry_run": True,
        "log_only": True,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    print()
    print(f"[INFO] HTTP status              : {response.status_code}")
    print(f"[INFO] Shuffle response         : {response.text}")
    print(f"[INFO] Evidence file             : {OUTPUT_FILE}")

    if not response.ok:
        print("[ERROR] Shuffle rejected the full SmartSOC payload.")
        return 4

    print()
    print("[SUCCESS] Full SmartSOC -> Shuffle dry-run succeeded.")
    print(f"[RESULT] Authorization status   : {authorization_status}")
    print(f"[RESULT] Analyst authorized     : {analyst_authorized}")
    print(f"[RESULT] Execution allowed      : {execution_allowed}")
    print("[SAFETY] Authorization was explicitly DENY.")
    print("[SAFETY] No response action was executed.")
    print("[SAFETY] No endpoint containment was performed.")
    print("[SAFETY] Caldera was not called.")
    print("[SAFETY] Production ML artifacts were not modified.")

    return 0


if __name__ == "__main__":
    sys.exit(main())