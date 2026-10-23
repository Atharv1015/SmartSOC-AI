from pathlib import Path
from datetime import datetime
import re
import subprocess
import sys
import os

ROOT = Path(r"C:\Users\athar\SmartSOC-AI")
CONFIG = ROOT / "src" / "integration_v2" / "execution_config_v2.py"
RUNNER = ROOT / "src" / "integration_v2" / "authorized_dry_run_v2.py"
ADAPTER = ROOT / "src" / "autonomous_v2" / "victim_mitigation_adapter_v2.py"

def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)

for path in (CONFIG, RUNNER, ADAPTER):
    if not path.is_file():
        fail(f"Expected file not found: {path}")

if os.environ.get("SMARTSOC_ALLOW_LIVE_MITIGATION", "").strip() == "1":
    fail("SMARTSOC_ALLOW_LIVE_MITIGATION=1 is set in this shell. Disable it before patching/testing.")

config_text = CONFIG.read_text(encoding="utf-8-sig")
if not re.search(r"(?m)^DRY_RUN\s*=\s*True\s*$", config_text):
    fail("execution_config_v2.py must have DRY_RUN = True before applying this patch.")

runner_text = RUNNER.read_text(encoding="utf-8-sig")
start_marker = "    try:\n        policy.validate_safety_invariants()"
action_marker = '    action = str(reasoning_result.get("mitigation", "")).strip().upper()'
start = runner_text.find(start_marker)
end = runner_text.find(action_marker, start + len(start_marker))
if start < 0 or end < 0:
    fail("Could not locate the dry-run policy block in authorized_dry_run_v2.py. No files changed.")
if runner_text.find(start_marker, start + 1) >= 0:
    fail("Multiple candidate policy blocks found in authorized_dry_run_v2.py. No files changed.")

new_policy_block = (
    "    # DRY_RUN uses its own policy checks, not the legacy live-execution validator.\n"
    "    # The live validator intentionally requires DRY_RUN=False.\n"
    "    if not policy.DRY_RUN or policy.UNRESTRICTED_AUTONOMOUS_ACTIONS:\n"
    '        return _blocked("POLICY_BLOCKED", "Dry-run mode is disabled or unrestricted actions are configured.")\n\n'
)
updated_runner = runner_text[:start] + new_policy_block + runner_text[end:]

adapter_text = ADAPTER.read_text(encoding="utf-8-sig")
adapter_marker = "def execute_dry_run(reasoning_result):"
adapter_start = adapter_text.find(adapter_marker)
if adapter_start < 0:
    fail("Could not locate execute_dry_run in victim_mitigation_adapter_v2.py. No files changed.")
if adapter_text.find(adapter_marker, adapter_start + 1) >= 0:
    fail("Multiple execute_dry_run definitions found. No files changed.")

new_dry_run_function = r"""def execute_dry_run(reasoning_result):
    # Authenticated DRY_RUN request only; this function never requests LIVE mode.
    def blocked(status, error):
        return {
            "success": False,
            "verified": False,
            "status": status,
            "error": error,
            "victim_request_attempted": False,
            "victim_http_response_received": False,
            "endpoint_modified": False,
        }

    if not isinstance(reasoning_result, dict):
        return blocked("REQUEST_BLOCKED", "Reasoning result must be a dictionary.")

    if not policy.DRY_RUN or policy.UNRESTRICTED_AUTONOMOUS_ACTIONS:
        return blocked("POLICY_BLOCKED", "Dry-run policy is disabled or unsafe.")

    action = str(reasoning_result.get("mitigation", "")).strip().upper()
    if action != "PROCESS_TERMINATION" or not policy.is_action_allowed(action):
        return blocked("REQUEST_BLOCKED", "Action is not approved by the allowlist.")

    raw_pid = reasoning_result.get("process_id")
    if isinstance(raw_pid, bool):
        return blocked("REQUEST_BLOCKED", "Invalid PID.")
    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        return blocked("REQUEST_BLOCKED", "Invalid PID.")
    if pid <= 0:
        return blocked("REQUEST_BLOCKED", "PID must be positive.")

    process_name = reasoning_result.get("process_name")
    if not isinstance(process_name, str):
        return blocked("REQUEST_BLOCKED", "Process name is required.")
    process_name = os.path.basename(
        process_name.strip().replace("\\", "/")
    ).lower()
    if not policy.is_process_allowed(process_name):
        return blocked("REQUEST_BLOCKED", "Process name is not allowlisted.")

    secret = os.environ.get("SMARTSOC_AGENT_SECRET", "")
    if len(secret) < 32:
        return blocked("CONFIGURATION_BLOCKED", "Host secret is missing or invalid.")

    request_body = {
        "action": action,
        "target": {"pid": pid, "process_name": process_name},
        "context": {
            "client_version": "victim-mitigation-adapter-v2",
            "requested_mode": "DRY_RUN",
            "technique": reasoning_result.get("technique"),
            "wazuh_rule_id": reasoning_result.get("wazuh_rule_id"),
            "reason": reasoning_result.get("reason"),
        },
    }
    body = json.dumps(
        request_body, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    timestamp = str(int(time.time()))
    nonce = secrets.token_hex(16)
    signature = build_signature(secret, timestamp, nonce, body)
    headers = {
        "Content-Type": "application/json",
        "X-SmartSOC-Timestamp": timestamp,
        "X-SmartSOC-Nonce": nonce,
        "X-SmartSOC-Signature": signature,
    }

    connection = None
    attempted = False
    try:
        connection = http.client.HTTPConnection(
            VICTIM_HOST, VICTIM_PORT, timeout=TIMEOUT_SECONDS
        )
        attempted = True
        connection.request("POST", "/mitigate", body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read().decode("utf-8", errors="replace")
        try:
            victim_response = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "success": False,
                "verified": False,
                "status": "INVALID_VICTIM_RESPONSE",
                "http_status": response.status,
                "victim_request_attempted": attempted,
                "victim_http_response_received": True,
                "endpoint_modified": False,
            }

        verified = (
            response.status == 200
            and victim_response.get("success") is True
            and victim_response.get("verified") is True
            and victim_response.get("status") == "DRY_RUN_VALIDATED"
        )
        return {
            "success": verified,
            "verified": verified,
            "status": (
                "DRY_RUN_VALIDATED"
                if verified
                else victim_response.get("status", "DRY_RUN_REJECTED")
            ),
            "http_status": response.status,
            "victim_request_attempted": attempted,
            "victim_http_response_received": True,
            "endpoint_modified": False,
            "client": "victim-mitigation-adapter-v2",
            "victim_response": victim_response,
        }
    except OSError as exc:
        return {
            "success": False,
            "verified": False,
            "status": "VICTIM_UNREACHABLE",
            "victim_request_attempted": attempted,
            "victim_http_response_received": False,
            "endpoint_modified": False,
            "error": str(exc),
        }
    finally:
        if connection is not None:
            connection.close()
"""
updated_adapter = adapter_text[:adapter_start] + new_dry_run_function.rstrip() + "\n"

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backups = []
for path in (RUNNER, ADAPTER):
    backup = path.with_name(path.name + f".bak_{stamp}")
    backup.write_bytes(path.read_bytes())
    backups.append(backup)

RUNNER.write_text(updated_runner, encoding="utf-8", newline="")
ADAPTER.write_text(updated_adapter, encoding="utf-8", newline="")

print("PATCH APPLIED")
for backup in backups:
    print(f"Backup: {backup}")

compiled = subprocess.run(
    [sys.executable, "-m", "py_compile", str(RUNNER), str(ADAPTER), str(CONFIG)],
    cwd=ROOT,
)
if compiled.returncode:
    fail(f"Python compile failed ({compiled.returncode}). Backups are available above.")

print("\n=== Run targeted dry-run tests (adapter/HTTP mocked by tests) ===")
tests = [
    str(ROOT / "tests" / "test_approval_bridge_v1.py"),
    str(ROOT / "tests" / "test_authorized_dry_run_v2.py"),
    str(ROOT / "tests" / "test_victim_mitigation_adapter_v2.py"),
]
result = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests], cwd=ROOT)
print("\nLive mitigation was not enabled; this script did not start the receiver.")
raise SystemExit(result.returncode)
