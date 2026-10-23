#!/usr/bin/env python3
# Separate SmartSOC dry-run regression mode from explicitly gated live mode.
# Keeps automatic policy flags, restores DRY_RUN for scenario tests, and requires
# SMARTSOC_ALLOW_LIVE_MITIGATION=1 before live adapter execution. No HTTP is sent.

from pathlib import Path
import datetime
import shutil
import sys

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "src" / "integration_v2" / "execution_config_v2.py"
ADAPTER = ROOT / "src" / "autonomous_v2" / "victim_mitigation_adapter_v2.py"
APPROVAL = ROOT / "src" / "integration" / "local_approval_console_v1.py"

def abort(msg):
    print("PATCH ABORTED: " + msg, file=sys.stderr)
    raise SystemExit(2)

def read(path):
    if not path.is_file():
        abort("Required file not found: " + str(path))
    return path.read_text(encoding="utf-8")

def block(lines):
    return "\n".join(lines) + "\n"

def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        abort(label + ": expected exactly one source block, found " + str(count) + ". No files written.")
    return text.replace(old, new, 1)

config = read(CONFIG)
adapter = read(ADAPTER)
approval = read(APPROVAL)

# Restore the test-mode default; automatic policy flags remain unchanged.
if "DRY_RUN = False" in config:
    config_new = replace_once(config, "DRY_RUN = False", "DRY_RUN = True", "restore DRY_RUN default")
elif "DRY_RUN = True" in config:
    config_new = config
else:
    abort("Could not identify DRY_RUN setting.")

if "import os\n" not in config_new:
    config_new = replace_once(
        config_new,
        '"""SmartSOC controlled-lab execution policy."""\n',
        '"""SmartSOC controlled-lab execution policy."""\n\nimport os\n',
        "add os import",
    )

if "LIVE_EXECUTION_ENABLED =" not in config_new:
    config_new = replace_once(
        config_new,
        "DRY_RUN = True",
        'DRY_RUN = True\nLIVE_EXECUTION_ENABLED = os.environ.get("SMARTSOC_ALLOW_LIVE_MITIGATION", "").strip() == "1"',
        "add independent live gate",
    )

old_invariant = block([
    "def validate_safety_invariants() -> bool:",
    "    # Bounded lab policy only; individual requests still pass action/PID/process checks.",
    "    return (",
    "        CONTROLLED_LAB_MODE",
    "        and EXECUTION_LAYER_ENABLED",
    "        and AUTOMATIC_EXECUTION",
    "        and AUTOMATIC_CONTAINMENT",
    "        and not UNRESTRICTED_AUTONOMOUS_ACTIONS",
    "        and not HUMAN_AUTHORIZATION_REQUIRED",
    "        and not DRY_RUN",
    '        and ALLOWED_ACTIONS == frozenset({"PROCESS_TERMINATION"})',
    '        and ALLOWED_PROCESSES == frozenset({"notepad.exe", "ping.exe"})',
    "    )",
])
new_invariant = block([
    "def validate_safety_invariants() -> bool:",
    "    # Validate bounded policy; DRY_RUN is a separate execution mode.",
    "    return (",
    "        CONTROLLED_LAB_MODE",
    "        and EXECUTION_LAYER_ENABLED",
    "        and AUTOMATIC_EXECUTION",
    "        and AUTOMATIC_CONTAINMENT",
    "        and not UNRESTRICTED_AUTONOMOUS_ACTIONS",
    "        and not HUMAN_AUTHORIZATION_REQUIRED",
    '        and ALLOWED_ACTIONS == frozenset({"PROCESS_TERMINATION"})',
    '        and ALLOWED_PROCESSES == frozenset({"notepad.exe", "ping.exe"})',
    "    )",
])
if old_invariant in config_new:
    config_new = replace_once(config_new, old_invariant, new_invariant, "separate policy invariant from DRY_RUN")
elif new_invariant not in config_new:
    abort("Could not identify validate_safety_invariants() block.")

old_live_gate = block([
    "    if (",
    "        policy.DRY_RUN",
    "        or not invariants_ok",
    "        or not policy.CONTROLLED_LAB_MODE",
    "        or not policy.EXECUTION_LAYER_ENABLED",
    "        or not policy.AUTOMATIC_EXECUTION",
    "        or not policy.AUTOMATIC_CONTAINMENT",
    "        or policy.HUMAN_AUTHORIZATION_REQUIRED",
    "        or policy.UNRESTRICTED_AUTONOMOUS_ACTIONS",
    "    ):",
    "        return {",
    '            "success": False, "verified": False,',
    '            "status": "POLICY_BLOCKED",',
    '            "error": "Automatic controlled-lab policy invariants are not satisfied.",',
    '            "victim_request_attempted": False,',
    '            "victim_client_invoked": False,',
    '            "endpoint_modified": False,',
    "        }",
])
new_live_gate = block([
    "    if (",
    "        not policy.LIVE_EXECUTION_ENABLED",
    "        or not invariants_ok",
    "        or not policy.CONTROLLED_LAB_MODE",
    "        or not policy.EXECUTION_LAYER_ENABLED",
    "        or not policy.AUTOMATIC_EXECUTION",
    "        or not policy.AUTOMATIC_CONTAINMENT",
    "        or policy.HUMAN_AUTHORIZATION_REQUIRED",
    "        or policy.UNRESTRICTED_AUTONOMOUS_ACTIONS",
    "    ):",
    "        return {",
    '            "success": False, "verified": False,',
    '            "status": "POLICY_BLOCKED",',
    '            "error": "Live execution requires the explicit live gate and valid controlled-lab policy.",',
    '            "victim_request_attempted": False,',
    '            "victim_client_invoked": False,',
    '            "endpoint_modified": False,',
    "        }",
])
if old_live_gate in adapter:
    adapter_new = replace_once(adapter, old_live_gate, new_live_gate, "separate live adapter gate")
elif "not policy.LIVE_EXECUTION_ENABLED" in adapter:
    adapter_new = adapter
else:
    old_original_gate = block([
        "    try:",
        "        policy.validate_safety_invariants()",
        "    except Exception as exc:",
        "        return {",
        '            "success": False, "verified": False,',
        '            "status": "POLICY_BLOCKED", "error": str(exc),',
        "        }",
        "",
        "    if (",
        "        policy.DRY_RUN",
        "        or not policy.EXECUTION_LAYER_ENABLED",
        "        or not policy.AUTOMATIC_EXECUTION",
        "        or not policy.AUTOMATIC_CONTAINMENT",
        "        or policy.HUMAN_AUTHORIZATION_REQUIRED",
        "        or policy.UNRESTRICTED_AUTONOMOUS_ACTIONS",
        "    ):",
        "        return {",
        '            "success": False, "verified": False,',
        '            "status": "POLICY_BLOCKED",',
        '            "error": "Live policy invariants are not satisfied.",',
        "        }",
    ])
    new_original_gate = block([
        "    try:",
        "        invariants_ok = policy.validate_safety_invariants()",
        "    except Exception as exc:",
        "        return {",
        '            "success": False, "verified": False,',
        '            "status": "POLICY_BLOCKED", "error": str(exc),',
        '            "victim_request_attempted": False,',
        '            "victim_client_invoked": False,',
        '            "endpoint_modified": False,',
        "        }",
        "",
        "    if (",
        "        not policy.LIVE_EXECUTION_ENABLED",
        "        or not invariants_ok",
        "        or not policy.CONTROLLED_LAB_MODE",
        "        or not policy.EXECUTION_LAYER_ENABLED",
        "        or not policy.AUTOMATIC_EXECUTION",
        "        or not policy.AUTOMATIC_CONTAINMENT",
        "        or policy.HUMAN_AUTHORIZATION_REQUIRED",
        "        or policy.UNRESTRICTED_AUTONOMOUS_ACTIONS",
        "    ):",
        "        return {",
        '            "success": False, "verified": False,',
        '            "status": "POLICY_BLOCKED",',
        '            "error": "Live execution requires the explicit live gate and valid controlled-lab policy.",',
        '            "victim_request_attempted": False,',
        '            "victim_client_invoked": False,',
        '            "endpoint_modified": False,',
        "        }",
    ])
    if old_original_gate in adapter:
        adapter_new = replace_once(adapter, old_original_gate, new_original_gate, "replace original live adapter gate")
    else:
        abort("Could not identify execute_live policy gate.")

# Preserve the established .execute() client interface. Mock objects can also be callable;
# dispatching to client(...) first breaks tests and may call the wrong mock.
old_dispatch = block([
    "    if callable(client):",
    "        raw_result = client(authorized_request)",
    "    else:",
    "        raw_result = client.execute(authorized_request)",
])
new_dispatch = block([
    "    execute_method = getattr(client, 'execute', None)",
    "    if callable(execute_method):",
    "        raw_result = execute_method(authorized_request)",
    "    elif callable(client):",
    "        raw_result = client(authorized_request)",
    "    else:",
    "        return {",
    '            "success": False, "verified": False,',
    '            "status": "INVALID_CLIENT",',
    '            "victim_client_invoked": False,',
    '            "endpoint_modified": False,',
    "        }",
])
if old_dispatch in approval:
    approval_new = replace_once(approval, old_dispatch, new_dispatch, "restore approval client dispatch")
elif "execute_method = getattr(client, 'execute', None)" in approval:
    approval_new = approval
else:
    abort("Could not identify approval client dispatch block.")

for required in (
    "AUTOMATIC_EXECUTION = True",
    "AUTOMATIC_CONTAINMENT = True",
    "HUMAN_AUTHORIZATION_REQUIRED = False",
    "UNRESTRICTED_AUTONOMOUS_ACTIONS = False",
    "DRY_RUN = True",
    'LIVE_EXECUTION_ENABLED = os.environ.get("SMARTSOC_ALLOW_LIVE_MITIGATION", "").strip() == "1"',
):
    if required not in config_new:
        abort("Final config assertion failed: " + required)
if "not policy.LIVE_EXECUTION_ENABLED" not in adapter_new:
    abort("Live adapter does not enforce separate live opt-in.")
live_section = adapter_new[adapter_new.find("def execute_live"):adapter_new.find("def execute_dry_run") if "def execute_dry_run" in adapter_new else len(adapter_new)]
if "policy.DRY_RUN" in live_section:
    abort("execute_live still depends on DRY_RUN.")
if "execute_method = getattr(client, 'execute', None)" not in approval_new:
    abort("Approval client dispatch fix missing.")

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
changes = [(CONFIG, config, config_new), (ADAPTER, adapter, adapter_new), (APPROVAL, approval, approval_new)]
backups, written = [], []
try:
    for path, old, new in changes:
        backup = path.with_name(path.name + ".bak_auto_regression_fix_" + stamp)
        shutil.copy2(path, backup)
        backups.append((path, backup, old))
    for path, old, new in changes:
        path.write_text(new, encoding="utf-8", newline="")
        written.append(path)
except Exception as exc:
    for path in reversed(written):
        original = next(old for pth, _backup, old in backups if pth == path)
        path.write_text(original, encoding="utf-8", newline="")
    abort("Write failed; changed files were restored: " + str(exc))

print("PATCH APPLIED: dry-run regression mode and live opt-in are separated.")
print("Backups:")
for _path, backup, _old in backups:
    print("  " + str(backup))
print()
print("Automatic policy flags remain enabled; human authorization remains disabled.")
print("DRY_RUN=True for the regression suite.")
print("Live execution requires SMARTSOC_ALLOW_LIVE_MITIGATION=1 in receiver startup environment.")
print("No receiver was started and no victim request was sent.")
