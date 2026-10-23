from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ADAPTER = ROOT / "src" / "autonomous_v2" / "victim_mitigation_adapter_v2.py"
RECEIVER = ROOT / "src" / "integration" / "smartsoc_live_receiver_v2.py"

ADAPTER_OLD = '''    if (
        not policy.LIVE_EXECUTION_ENABLED
        or not invariants_ok
        or not policy.CONTROLLED_LAB_MODE
'''
ADAPTER_NEW = '''    if (
        policy.DRY_RUN
        or not policy.LIVE_EXECUTION_ENABLED
        or not invariants_ok
        or not policy.CONTROLLED_LAB_MODE
'''

RECEIVER_OLD = '''LIVE_EXECUTION_ENABLED = os.environ.get("SMARTSOC_ALLOW_LIVE_MITIGATION", "").strip() == "1"'''
RECEIVER_NEW = '''# Effective receiver gate: environment opt-in is insufficient while policy is dry-run.
LIVE_EXECUTION_ENABLED = (
    os.environ.get("SMARTSOC_ALLOW_LIVE_MITIGATION", "").strip() == "1"
    and not _smartsoc_execution_policy.DRY_RUN
)'''

def main() -> int:
    missing = [str(p) for p in (ADAPTER, RECEIVER) if not p.is_file()]
    if missing:
        print("ERROR: required file(s) not found:")
        print("\\n".join(missing))
        print("Run this patch from the SmartSOC-AI repository root. No files changed.")
        return 2

    adapter_source = ADAPTER.read_text(encoding="utf-8")
    receiver_source = RECEIVER.read_text(encoding="utf-8")

    adapter_count = adapter_source.count(ADAPTER_OLD)
    receiver_count = receiver_source.count(RECEIVER_OLD)

    if adapter_count != 1:
        print(f"ERROR: expected one adapter live-gate block; found {adapter_count}. No files changed.")
        return 3
    if receiver_count != 1:
        print(f"ERROR: expected one receiver live-gate assignment; found {receiver_count}. No files changed.")
        return 4
    if "and not _smartsoc_execution_policy.DRY_RUN" in receiver_source:
        print("ERROR: receiver already appears to include the dry-run gate. No files changed.")
        return 5

    adapter_updated = adapter_source.replace(ADAPTER_OLD, ADAPTER_NEW, 1)
    receiver_updated = receiver_source.replace(RECEIVER_OLD, RECEIVER_NEW, 1)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backups = [
        (ADAPTER, ADAPTER.with_name(ADAPTER.name + f".bak_dry_run_live_gate_{stamp}"), adapter_source),
        (RECEIVER, RECEIVER.with_name(RECEIVER.name + f".bak_dry_run_live_gate_{stamp}"), receiver_source),
    ]

    try:
        for target, backup, original in backups:
            shutil.copy2(target, backup)
        ADAPTER.write_text(adapter_updated, encoding="utf-8")
        RECEIVER.write_text(receiver_updated, encoding="utf-8")

        saved_adapter = ADAPTER.read_text(encoding="utf-8")
        saved_receiver = RECEIVER.read_text(encoding="utf-8")
        if (
            ADAPTER_NEW not in saved_adapter
            or RECEIVER_NEW not in saved_receiver
            or ADAPTER_OLD in saved_adapter
            or RECEIVER_OLD in saved_receiver
        ):
            raise RuntimeError("post-write verification failed")
    except Exception as exc:
        for target, backup, original in backups:
            try:
                target.write_text(original, encoding="utf-8")
            except Exception:
                pass
        print(f"ERROR: patch failed ({exc}); original source content was restored where possible.")
        return 6

    print("PATCH APPLIED")
    print(f"Adapter: {ADAPTER}")
    print(f"Backup : {backups[0][1]}")
    print(f"Receiver: {RECEIVER}")
    print(f"Backup  : {backups[1][1]}")
    print("Changes:")
    print("  - execute_live() rejects requests whenever policy.DRY_RUN is True")
    print("  - receiver effective live gate requires environment opt-in AND DRY_RUN=False")
    print("  - receiver health/handler/startup status reuse the effective gate")
    print("No receiver was started and no mitigation request was sent.")
    print("This patch does not enable live execution.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
