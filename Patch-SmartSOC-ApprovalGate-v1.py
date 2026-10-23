#!/usr/bin/env python3
# SmartSOC approval-gate wiring patch (fail-closed, DRY_RUN preserved).
# Run from project root: py .\Patch-SmartSOC-ApprovalGate-v1.py
# Does not turn DRY_RUN off, arm live mitigation, alter HMAC secrets, or send a victim request.

from __future__ import annotations
import datetime
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "src" / "integration_v2" / "execution_config_v2.py"
ADAPTER = ROOT / "src" / "autonomous_v2" / "victim_mitigation_adapter_v2.py"
APPROVAL = ROOT / "src" / "integration" / "local_approval_console_v1.py"
RECEIVER = ROOT / "src" / "integration" / "smartsoc_live_receiver_v2.py"

def fail(msg: str) -> None:
    print(f"PATCH ABORTED: {msg}", file=sys.stderr)
    raise SystemExit(2)

def read(path: Path) -> str:
    if not path.is_file():
        fail(f"Required file not found: {path}")
    return path.read_text(encoding="utf-8")

def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        fail(f"{label}: expected exactly one matching block; found {count}. No files have been written.")
    return text.replace(old, new, 1)

config = read(CONFIG)
adapter = read(ADAPTER)
approval = read(APPROVAL)
receiver = read(RECEIVER)

config_new = replace_once(
    config,
    'AUTOMATIC_EXECUTION = True\nAUTOMATIC_CONTAINMENT = True\n\nUNRESTRICTED_AUTONOMOUS_ACTIONS = False\nHUMAN_AUTHORIZATION_REQUIRED = False\n',
    'AUTOMATIC_EXECUTION = False\nAUTOMATIC_CONTAINMENT = False\n\nUNRESTRICTED_AUTONOMOUS_ACTIONS = False\nHUMAN_AUTHORIZATION_REQUIRED = True\n',
    "execution policy defaults",
)

config_new = replace_once(
    config_new,
    '''def validate_safety_invariants() -> bool:
    return (
        CONTROLLED_LAB_MODE
        and EXECUTION_LAYER_ENABLED
        and AUTOMATIC_EXECUTION
        and AUTOMATIC_CONTAINMENT
        and not UNRESTRICTED_AUTONOMOUS_ACTIONS
        and not DRY_RUN
        and ALLOWED_ACTIONS == frozenset({"PROCESS_TERMINATION"})
        and ALLOWED_PROCESSES == frozenset({"notepad.exe", "ping.exe"})
    )
''',
    '''def validate_safety_invariants() -> bool:
    # This validates bounded policy; it does not authorize an individual action.
    return (
        CONTROLLED_LAB_MODE
        and not UNRESTRICTED_AUTONOMOUS_ACTIONS
        and HUMAN_AUTHORIZATION_REQUIRED
        and not AUTOMATIC_EXECUTION
        and not AUTOMATIC_CONTAINMENT
        and ALLOWED_ACTIONS == frozenset({"PROCESS_TERMINATION"})
        and ALLOWED_PROCESSES == frozenset({"notepad.exe", "ping.exe"})
    )
''',
    "safety invariant function",
)

approval_new = replace_once(
    approval,
    '''    result=client.execute(reasoning_result)
    result=dict(result) if isinstance(result,dict) else {"success":False,"verified":False,"status":"INVALID_CLIENT_RESULT"}
    result.update(victim_client_invoked=True,authorization_request_id=auth["request_id"],authorization_actor=auth["actor"])
''',
    '''    authorized_request = dict(reasoning_result)
    authorized_request["_smartsoc_authorization"] = {
        "request_id": auth["request_id"],
        "actor": auth["actor"],
        "consumed": True,
    }
    if callable(client):
        raw_result = client(authorized_request)
    else:
        raw_result = client.execute(authorized_request)
    result=dict(raw_result) if isinstance(raw_result,dict) else {"success":False,"verified":False,"status":"INVALID_CLIENT_RESULT"}
    result.update(victim_client_invoked=bool(result.get("victim_request_attempted", False) or result.get("victim_client_invoked", False)),authorization_request_id=auth["request_id"],authorization_actor=auth["actor"])
''',
    "approval wrapper",
)

adapter_new = replace_once(
    adapter,
    '''    try:
        policy.validate_safety_invariants()
    except Exception as exc:
        return {
            "success": False, "verified": False,
            "status": "POLICY_BLOCKED", "error": str(exc),
        }

    if (
        policy.DRY_RUN
        or not policy.EXECUTION_LAYER_ENABLED
        or not policy.AUTOMATIC_EXECUTION
        or not policy.AUTOMATIC_CONTAINMENT
        or policy.HUMAN_AUTHORIZATION_REQUIRED
        or policy.UNRESTRICTED_AUTONOMOUS_ACTIONS
    ):
        return {
            "success": False, "verified": False,
            "status": "POLICY_BLOCKED",
            "error": "Live policy invariants are not satisfied.",
        }
''',
    '''    authorization = reasoning_result.get("_smartsoc_authorization")
    if (
        not isinstance(authorization, dict)
        or authorization.get("consumed") is not True
        or not str(authorization.get("request_id", "")).strip()
        or not str(authorization.get("actor", "")).strip()
    ):
        return {
            "success": False, "verified": False,
            "status": "AUTHORIZATION_BLOCKED",
            "error": "A consumed, event-bound human approval is required.",
            "victim_request_attempted": False,
            "victim_client_invoked": False,
            "endpoint_modified": False,
        }

    try:
        invariants_ok = policy.validate_safety_invariants()
    except Exception as exc:
        return {
            "success": False, "verified": False,
            "status": "POLICY_BLOCKED", "error": str(exc),
            "victim_request_attempted": False,
            "victim_client_invoked": False,
            "endpoint_modified": False,
        }

    if (
        policy.DRY_RUN
        or not invariants_ok
        or not policy.EXECUTION_LAYER_ENABLED
        or not policy.HUMAN_AUTHORIZATION_REQUIRED
        or policy.AUTOMATIC_EXECUTION
        or policy.AUTOMATIC_CONTAINMENT
        or policy.UNRESTRICTED_AUTONOMOUS_ACTIONS
    ):
        return {
            "success": False, "verified": False,
            "status": "POLICY_BLOCKED",
            "error": "Live execution is disabled by policy; DRY_RUN must remain enabled during Phase A.",
            "victim_request_attempted": False,
            "victim_client_invoked": False,
            "endpoint_modified": False,
        }
''',
    "live adapter policy gate",
)

receiver_new = replace_once(
    receiver,
    'from autonomous_v2.victim_mitigation_adapter_v2 import execute_live\n',
    'from autonomous_v2.victim_mitigation_adapter_v2 import execute_live\nfrom integration_v2 import execution_config_v2 as _smartsoc_execution_policy\n',
    "receiver policy import",
)

receiver_new = replace_once(
    receiver_new,
    '''        mitigation_result = execute_live(
            bind_event_context(reasoning_result, alert_id, rule_id)
        )
''',
    '''        bound_reasoning = bind_event_context(reasoning_result, alert_id, rule_id)
        if _smartsoc_execution_policy.DRY_RUN:
            # Phase A is deliberately non-invasive: no victim HTTP request is sent.
            mitigation_result = {
                "success": False,
                "verified": False,
                "status": "DRY_RUN_ONLY",
                "reason": "DRY_RUN policy is enabled; victim endpoint was not called.",
                "victim_request_attempted": False,
                "victim_client_invoked": False,
                "endpoint_modified": False,
            }
        else:
            # Phase B: consume a matching, unexpired, one-time analyst approval first.
            mitigation_result = _smartsoc_approvals.execute_authorized_client_call(
                execute_live, bound_reasoning
            )
''',
    "receiver mitigation call site",
)

if "DRY_RUN = True" not in config_new:
    fail("Safety check failed: DRY_RUN must remain True.")
if "HUMAN_AUTHORIZATION_REQUIRED = True" not in config_new:
    fail("Safety check failed: human approval must remain required.")
if "execute_authorized_client_call(" not in receiver_new:
    fail("Safety check failed: receiver approval wrapper not wired.")
if '"status": "DRY_RUN_ONLY"' not in receiver_new:
    fail("Safety check failed: dry-run receiver path not present.")
if 'authorization.get("consumed") is not True' not in adapter_new:
    fail("Safety check failed: adapter approval marker check not present.")

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
changes = [(CONFIG, config, config_new), (APPROVAL, approval, approval_new),
           (ADAPTER, adapter, adapter_new), (RECEIVER, receiver, receiver_new)]
backups = []
written = []
try:
    for path, old, new in changes:
        backup = path.with_name(path.name + f".bak_approval_gate_{stamp}")
        shutil.copy2(path, backup)
        backups.append((path, backup, old))
    for path, old, new in changes:
        path.write_text(new, encoding="utf-8", newline="")
        written.append(path)
except Exception as exc:
    for path in reversed(written):
        original = next(old for pth, _bak, old in backups if pth == path)
        path.write_text(original, encoding="utf-8", newline="")
    fail(f"Write failed and changed files were restored: {exc}")

print("PATCH APPLIED: SmartSOC approval gate wired.")
print("Backups:")
for path, backup, _old in backups:
    print(f"  {backup}")
print()
print("Safety state intentionally preserved:")
print("  DRY_RUN=True; receiver will not send victim HTTP requests during Phase A")
print("  HUMAN_AUTHORIZATION_REQUIRED=True")
print("  AUTOMATIC_EXECUTION=False")
print("  AUTOMATIC_CONTAINMENT=False")
print("  Process allowlist unchanged; arbitrary commands remain disallowed.")
print()
print("NEXT: run the scenario suite and inspect its output. This patch is not evidence of live mitigation.")
