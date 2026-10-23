from __future__ import annotations

import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RECEIVER = ROOT / "src" / "integration" / "smartsoc_live_receiver_v2.py"

def abort(message: str) -> None:
    print(f"PATCH ABORTED: {message}")
    raise SystemExit(2)

if not (ROOT / "tests").is_dir():
    abort("Run this script from the SmartSOC-AI repository root.")
if not RECEIVER.is_file():
    abort(f"Receiver source not found: {RECEIVER}")

source = RECEIVER.read_text(encoding="utf-8")
marker = "SMARTSOC_PENDING_APPROVAL_REGISTRATION_V1"

if marker in source:
    print("PATCH ALREADY APPLIED: pending-approval registration is present.")
    raise SystemExit(0)

# Insert the approval-store import at module scope.
import_line = "from integration import local_approval_console_v1 as _smartsoc_approvals"
if import_line not in source:
    import_matches = list(re.finditer(
        r"^(?:from\s+\S+\s+import\s+.+|import\s+.+)$",
        source,
        re.M,
    ))
    if not import_matches:
        abort("Could not find a safe module-level import insertion point.")
    insert_at = import_matches[-1].end()
    source = source[:insert_at] + "\n" + import_line + source[insert_at:]

# Insert at the execution gate, where reasoning_result, alert_id, rule_id and
# requested_mitigation are already available. This records only a PENDING
# human-approval request; it never invokes the controller or endpoint client.
gate = re.search(
    r"(?m)^([ \t]*)if not MITIGATION_ENABLED or not LIVE_EXECUTION_ENABLED\s*:",
    source,
)
if not gate:
    abort("Could not find the receiver execution gate; no source file was changed.")

indent = gate.group(1)
registration = (
    f"{indent}# {marker}: register candidate independently of execution enablement.\n"
    f"{indent}if requested_mitigation == \"PROCESS_TERMINATION\":\n"
    f"{indent}    try:\n"
    f"{indent}        _smartsoc_approvals.register_pending_from_reasoning_result(\n"
    f"{indent}            reasoning_result, str(alert_id), str(rule_id)\n"
    f"{indent}        )\n"
    f"{indent}    except Exception as _approval_error:\n"
    f"{indent}        # Fail closed: report registration failure; continue to the\n"
    f"{indent}        # existing disabled-execution branch without attempting mitigation.\n"
    f"{indent}        print(f\"[APPROVAL REGISTRATION FAILED CLOSED] {{_approval_error}}\")\n\n"
)
source = source[:gate.start()] + registration + source[gate.start():]

try:
    compile(source, str(RECEIVER), "exec")
except SyntaxError as exc:
    abort(f"Proposed receiver patch does not compile: {exc}")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = RECEIVER.with_name(RECEIVER.name + f".bak_pending_approval_{stamp}")
shutil.copy2(RECEIVER, backup)
RECEIVER.write_text(source, encoding="utf-8", newline="\n")

print("PATCH APPLIED")
print(f"Backup: {backup}")
print("Changed: src/integration/smartsoc_live_receiver_v2.py")
print("Safety: records a PENDING approval only; does not enable live mitigation.")

compiled = subprocess.run(
    [sys.executable, "-m", "py_compile", str(RECEIVER)],
    cwd=str(ROOT),
)
if compiled.returncode:
    shutil.copy2(backup, RECEIVER)
    abort("py_compile failed; original receiver restored.")

tests = [
    "tests/test_receiver_approval_bridge_workflow_v1.py",
    "tests/test_receiver_fail_closed_v1.py",
]
targeted = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", *tests],
    cwd=str(ROOT),
)
if targeted.returncode:
    print("\nThe targeted tests still fail. Keep the backup; do not enable live mitigation.")
    raise SystemExit(targeted.returncode)

print("\nTargeted receiver tests passed.")
print("Run the full suite:")
print('powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\\Run-SmartSOC-ScenarioSuite.ps1"')
