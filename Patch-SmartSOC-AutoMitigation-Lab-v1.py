#!/usr/bin/env python3
"""Enable bounded automatic SmartSOC mitigation in the controlled lab.

WARNING: DRY_RUN=False plus no per-event approval permits termination of allowlisted
processes on the lab victim. Keep the victim disposable and isolated. This patch keeps
the explicit receiver startup gate, HMAC, PROCESS_TERMINATION-only action allowlist,
notepad.exe/ping.exe process allowlist, and arbitrary-command prohibition.
"""
from pathlib import Path
import datetime, shutil, sys

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "src/integration_v2/execution_config_v2.py"
ADAPTER = ROOT / "src/autonomous_v2/victim_mitigation_adapter_v2.py"
RECEIVER = ROOT / "src/integration/smartsoc_live_receiver_v2.py"

def abort(msg):
    print("PATCH ABORTED: " + msg, file=sys.stderr)
    raise SystemExit(2)

def read(path):
    if not path.is_file():
        abort("Required file not found: " + str(path))
    return path.read_text(encoding="utf-8")

def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        abort(label + ": expected one exact block, found " + str(n) + "; no files written.")
    return text.replace(old, new, 1)

config = read(CONFIG)
adapter = read(ADAPTER)
receiver = read(RECEIVER)

old_flags = (
    "AUTOMATIC_EXECUTION = False\nAUTOMATIC_CONTAINMENT = False\n\n"
    "UNRESTRICTED_AUTONOMOUS_ACTIONS = False\nHUMAN_AUTHORIZATION_REQUIRED = True\n"
)
new_flags = (
    "AUTOMATIC_EXECUTION = True\nAUTOMATIC_CONTAINMENT = True\n\n"
    "UNRESTRICTED_AUTONOMOUS_ACTIONS = False\nHUMAN_AUTHORIZATION_REQUIRED = False\n"
)
if old_flags in config:
    config_new = replace_once(config, old_flags, new_flags, "policy flags")
elif new_flags in config:
    config_new = config
else:
    abort("Could not identify policy flags.")
config_new = replace_once(config_new, "DRY_RUN = True", "DRY_RUN = False", "DRY_RUN flag")

old_invariant_approval = (
    "def validate_safety_invariants() -> bool:\n"
    "    # This validates bounded policy; it does not authorize an individual action.\n"
    "    return (\n"
    "        CONTROLLED_LAB_MODE\n"
    "        and not UNRESTRICTED_AUTONOMOUS_ACTIONS\n"
    "        and HUMAN_AUTHORIZATION_REQUIRED\n"
    "        and not AUTOMATIC_EXECUTION\n"
    "        and not AUTOMATIC_CONTAINMENT\n"
    '        and ALLOWED_ACTIONS == frozenset({"PROCESS_TERMINATION"})\n'
    '        and ALLOWED_PROCESSES == frozenset({"notepad.exe", "ping.exe"})\n'
    "    )\n"
)
old_invariant_original = (
    "def validate_safety_invariants() -> bool:\n"
    "    return (\n"
    "        CONTROLLED_LAB_MODE\n"
    "        and EXECUTION_LAYER_ENABLED\n"
    "        and AUTOMATIC_EXECUTION\n"
    "        and AUTOMATIC_CONTAINMENT\n"
    "        and not UNRESTRICTED_AUTONOMOUS_ACTIONS\n"
    "        and not DRY_RUN\n"
    '        and ALLOWED_ACTIONS == frozenset({"PROCESS_TERMINATION"})\n'
    '        and ALLOWED_PROCESSES == frozenset({"notepad.exe", "ping.exe"})\n'
    "    )\n"
)
new_invariant = (
    "def validate_safety_invariants() -> bool:\n"
    "    # Bounded lab policy only; individual requests still pass action/PID/process checks.\n"
    "    return (\n"
    "        CONTROLLED_LAB_MODE\n"
    "        and EXECUTION_LAYER_ENABLED\n"
    "        and AUTOMATIC_EXECUTION\n"
    "        and AUTOMATIC_CONTAINMENT\n"
    "        and not UNRESTRICTED_AUTONOMOUS_ACTIONS\n"
    "        and not HUMAN_AUTHORIZATION_REQUIRED\n"
    "        and not DRY_RUN\n"
    '        and ALLOWED_ACTIONS == frozenset({"PROCESS_TERMINATION"})\n'
    '        and ALLOWED_PROCESSES == frozenset({"notepad.exe", "ping.exe"})\n'
    "    )\n"
)
if old_invariant_approval in config_new:
    config_new = replace_once(config_new, old_invariant_approval, new_invariant, "approval invariant")
elif old_invariant_original in config_new:
    config_new = replace_once(config_new, old_invariant_original, new_invariant, "original invariant")
elif new_invariant not in config_new:
    abort("Could not identify validate_safety_invariants().")

old_approval_gate = (
    '    authorization = reasoning_result.get("_smartsoc_authorization")\n'
    "    if (\n"
    "        not isinstance(authorization, dict)\n"
    '        or authorization.get("consumed") is not True\n'
    '        or not str(authorization.get("request_id", "")).strip()\n'
    '        or not str(authorization.get("actor", "")).strip()\n'
    "    ):\n"
    "        return {\n"
    '            "success": False, "verified": False,\n'
    '            "status": "AUTHORIZATION_BLOCKED",\n'
    '            "error": "A consumed, event-bound human approval is required.",\n'
    '            "victim_request_attempted": False,\n'
    '            "victim_client_invoked": False,\n'
    '            "endpoint_modified": False,\n'
    "        }\n\n"
    "    try:\n"
    "        invariants_ok = policy.validate_safety_invariants()\n"
    "    except Exception as exc:\n"
    "        return {\n"
    '            "success": False, "verified": False,\n'
    '            "status": "POLICY_BLOCKED", "error": str(exc),\n'
    '            "victim_request_attempted": False,\n'
    '            "victim_client_invoked": False,\n'
    '            "endpoint_modified": False,\n'
    "        }\n\n"
    "    if (\n"
    "        policy.DRY_RUN\n"
    "        or not invariants_ok\n"
    "        or not policy.EXECUTION_LAYER_ENABLED\n"
    "        or not policy.HUMAN_AUTHORIZATION_REQUIRED\n"
    "        or policy.AUTOMATIC_EXECUTION\n"
    "        or policy.AUTOMATIC_CONTAINMENT\n"
    "        or policy.UNRESTRICTED_AUTONOMOUS_ACTIONS\n"
    "    ):\n"
    "        return {\n"
    '            "success": False, "verified": False,\n'
    '            "status": "POLICY_BLOCKED",\n'
    '            "error": "Live execution is disabled by policy; DRY_RUN must remain enabled during Phase A.",\n'
    '            "victim_request_attempted": False,\n'
    '            "victim_client_invoked": False,\n'
    '            "endpoint_modified": False,\n'
    "        }\n"
)
new_auto_gate = (
    "    try:\n"
    "        invariants_ok = policy.validate_safety_invariants()\n"
    "    except Exception as exc:\n"
    "        return {\n"
    '            "success": False, "verified": False,\n'
    '            "status": "POLICY_BLOCKED", "error": str(exc),\n'
    '            "victim_request_attempted": False,\n'
    '            "victim_client_invoked": False,\n'
    '            "endpoint_modified": False,\n'
    "        }\n\n"
    "    if (\n"
    "        policy.DRY_RUN\n"
    "        or not invariants_ok\n"
    "        or not policy.CONTROLLED_LAB_MODE\n"
    "        or not policy.EXECUTION_LAYER_ENABLED\n"
    "        or not policy.AUTOMATIC_EXECUTION\n"
    "        or not policy.AUTOMATIC_CONTAINMENT\n"
    "        or policy.HUMAN_AUTHORIZATION_REQUIRED\n"
    "        or policy.UNRESTRICTED_AUTONOMOUS_ACTIONS\n"
    "    ):\n"
    "        return {\n"
    '            "success": False, "verified": False,\n'
    '            "status": "POLICY_BLOCKED",\n'
    '            "error": "Automatic controlled-lab policy invariants are not satisfied.",\n'
    '            "victim_request_attempted": False,\n'
    '            "victim_client_invoked": False,\n'
    '            "endpoint_modified": False,\n'
    "        }\n"
)
if old_approval_gate in adapter:
    adapter_new = replace_once(adapter, old_approval_gate, new_auto_gate, "adapter approval gate")
elif new_auto_gate in adapter:
    adapter_new = adapter
else:
    old_original_gate = (
        "    try:\n"
        "        policy.validate_safety_invariants()\n"
        "    except Exception as exc:\n"
        "        return {\n"
        '            "success": False, "verified": False,\n'
        '            "status": "POLICY_BLOCKED", "error": str(exc),\n'
        "        }\n\n"
        "    if (\n"
        "        policy.DRY_RUN\n"
        "        or not policy.EXECUTION_LAYER_ENABLED\n"
        "        or not policy.AUTOMATIC_EXECUTION\n"
        "        or not policy.AUTOMATIC_CONTAINMENT\n"
        "        or policy.HUMAN_AUTHORIZATION_REQUIRED\n"
        "        or policy.UNRESTRICTED_AUTONOMOUS_ACTIONS\n"
        "    ):\n"
        "        return {\n"
        '            "success": False, "verified": False,\n'
        '            "status": "POLICY_BLOCKED",\n'
        '            "error": "Live policy invariants are not satisfied.",\n'
        "        }\n"
    )
    if old_original_gate in adapter:
        adapter_new = replace_once(adapter, old_original_gate, new_auto_gate, "adapter original gate")
    else:
        abort("Could not identify adapter policy gate.")

dryrun_branch = (
    "        bound_reasoning = bind_event_context(reasoning_result, alert_id, rule_id)\n"
    "        if _smartsoc_execution_policy.DRY_RUN:\n"
    "            # Phase A is deliberately non-invasive: no victim HTTP request is sent.\n"
    "            mitigation_result = {\n"
    '                "success": False,\n'
    '                "verified": False,\n'
    '                "status": "DRY_RUN_ONLY",\n'
    '                "reason": "DRY_RUN policy is enabled; victim endpoint was not called.",\n'
    '                "victim_request_attempted": False,\n'
    '                "victim_client_invoked": False,\n'
    '                "endpoint_modified": False,\n'
    "            }\n"
    "        else:\n"
    "            # Phase B: consume a matching, unexpired, one-time analyst approval first.\n"
    "            mitigation_result = _smartsoc_approvals.execute_authorized_client_call(\n"
    "                execute_live, bound_reasoning\n"
    "            )\n"
)
direct_branch = (
    "        bound_reasoning = bind_event_context(reasoning_result, alert_id, rule_id)\n"
    "        mitigation_result = execute_live(bound_reasoning)\n"
)
if dryrun_branch in receiver:
    receiver_new = replace_once(receiver, dryrun_branch, direct_branch, "receiver approval wrapper")
else:
    original_call = (
        "        mitigation_result = execute_live(\n"
        "            bind_event_context(reasoning_result, alert_id, rule_id)\n"
        "        )\n"
    )
    if original_call in receiver:
        receiver_new = replace_once(receiver, original_call, direct_branch, "receiver original call")
    elif direct_branch in receiver:
        receiver_new = receiver
    else:
        abort("Could not identify receiver mitigation call block.")

for required in [
    "EXECUTION_LAYER_ENABLED = True",
    "AUTOMATIC_EXECUTION = True",
    "AUTOMATIC_CONTAINMENT = True",
    "HUMAN_AUTHORIZATION_REQUIRED = False",
    "UNRESTRICTED_AUTONOMOUS_ACTIONS = False",
    "DRY_RUN = False",
    'ALLOWED_ACTIONS = frozenset({"PROCESS_TERMINATION"})',
    'ALLOWED_PROCESSES = frozenset({"notepad.exe", "ping.exe"})',
]:
    if required not in config_new:
        abort("Final policy assertion failed: " + required)
if "SMARTSOC_ALLOW_LIVE_MITIGATION" not in receiver_new:
    abort("Explicit receiver live-startup gate not found; refusing to patch.")
if "execute_live(bound_reasoning)" not in receiver_new:
    abort("Expected direct automatic adapter call not found.")

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
changes = [(CONFIG, config, config_new), (ADAPTER, adapter, adapter_new), (RECEIVER, receiver, receiver_new)]
backups, written = [], []
try:
    for path, old, new in changes:
        backup = path.with_name(path.name + ".bak_auto_lab_" + stamp)
        shutil.copy2(path, backup)
        backups.append((path, backup, old))
    for path, old, new in changes:
        path.write_text(new, encoding="utf-8", newline="")
        written.append(path)
except Exception as exc:
    for path in reversed(written):
        original = next(old for pth, _bak, old in backups if pth == path)
        path.write_text(original, encoding="utf-8", newline="")
    abort("Write failed; changed files restored: " + str(exc))

print("PATCH APPLIED: automatic mitigation policy enabled for the controlled lab.")
print("Backups:")
for _path, backup, _old in backups:
    print("  " + str(backup))
print("\nPolicy: EXECUTION_LAYER_ENABLED=True; AUTOMATIC_EXECUTION=True; AUTOMATIC_CONTAINMENT=True; HUMAN_AUTHORIZATION_REQUIRED=False; DRY_RUN=False.")
print('Action allowlist remains PROCESS_TERMINATION; process allowlist remains notepad.exe and ping.exe.')
print("Arbitrary commands remain prohibited. Receiver still requires SMARTSOC_ALLOW_LIVE_MITIGATION=1 at startup.")
print("This patch does not prove endpoint authentication or live mitigation; test on a disposable lab victim only.")
