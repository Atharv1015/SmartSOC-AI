from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "src" / "integration" / "local_approval_console_v1.py"

def replace_update_call(lines: list[str], start: int, end: int, indent: str) -> list[str]:
    replacement = [
        indent + "result.update(",
        indent + "    victim_client_invoked=True,",
        indent + '    authorization_request_id=auth["request_id"],',
        indent + '    authorization_actor=auth["actor"],',
        indent + ")",
    ]
    return lines[:start] + replacement + lines[end + 1:]

def main() -> int:
    if not TARGET.is_file():
        print(f"ERROR: target file not found: {TARGET}")
        print("Place this patch in the SmartSOC-AI repository root and run it there.")
        return 2

    source = TARGET.read_text(encoding="utf-8")
    lines = source.splitlines()
    try:
        fn_start = next(i for i, line in enumerate(lines)
                        if line.lstrip().startswith("def execute_authorized_client_call("))
    except StopIteration:
        print("ERROR: execute_authorized_client_call() not found. No files changed.")
        return 3

    try:
        fn_end = next(i for i in range(fn_start + 1, len(lines))
                      if lines[i].startswith("def run_console("))
    except StopIteration:
        print("ERROR: run_console() boundary not found. No files changed.")
        return 4

    candidates = [
        i for i in range(fn_start, fn_end)
        if "result.update(" in lines[i] and "victim_client_invoked" in "\n".join(lines[i:min(i+8, fn_end)])
    ]
    if len(candidates) != 1:
        print(f"ERROR: expected one result.update() invocation-tracking block, found {len(candidates)}.")
        print("No files changed. Inspect execute_authorized_client_call() before adapting.")
        return 5

    start = candidates[0]
    indent = lines[start][:len(lines[start]) - len(lines[start].lstrip())]

    # Find the end of this update call across one or multiple lines.
    end = start
    paren_balance = 0
    started = False
    for i in range(start, fn_end):
        line = lines[i]
        # Parentheses in this call are structural; the current source block has no
        # parentheses inside string literals that affect balance.
        paren_balance += line.count("(") - line.count(")")
        started = True
        end = i
        if started and paren_balance == 0:
            break
    else:
        print("ERROR: could not find the end of result.update(). No files changed.")
        return 6

    old_block = "\n".join(lines[start:end + 1])
    if "authorization_request_id" not in old_block or "authorization_actor" not in old_block:
        print("ERROR: invocation block does not match expected authorization metadata. No files changed.")
        return 7

    updated_lines = replace_update_call(lines, start, end, indent)
    updated = "\n".join(updated_lines) + ("\n" if source.endswith("\n") else "")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_name(TARGET.name + f".bak_invocation_flag_{stamp}")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(updated, encoding="utf-8")

    check = TARGET.read_text(encoding="utf-8")
    if "victim_client_invoked=True" not in check or old_block in check:
        shutil.copy2(backup, TARGET)
        print("ERROR: verification failed; original file restored.")
        return 8

    print("PATCH APPLIED")
    print(f"Target : {TARGET}")
    print(f"Backup : {backup}")
    print("Fix    : successful authorized client calls are now marked victim_client_invoked=True.")
    print("Approval consumption and client dispatch logic were not changed.")
    print("No receiver was started and no HTTP/live mitigation request was sent.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
