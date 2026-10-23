from __future__ import annotations

import datetime as dt
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "src" / "integration" / "local_approval_console_v1.py"


def main() -> int:
    if not TARGET.is_file():
        print(f"ERROR: target not found: {TARGET}")
        return 2

    source = TARGET.read_text(encoding="utf-8")

    # Limit all edits to execute_authorized_client_call() so unrelated code stays untouched.
    fn_match = re.search(
        r"(?ms)^def execute_authorized_client_call\(client,\s*reasoning_result\):\n.*?(?=^def run_console\(\):)",
        source,
    )
    if not fn_match:
        print("ERROR: Could not locate execute_authorized_client_call() through run_console(). No files changed.")
        return 3

    fn = fn_match.group(0)

    # Remove the private metadata injection. Tests and the existing adapter interface expect
    # the original event fields only; authorization is already consumed and validated locally.
    metadata_pattern = re.compile(
        r"(?m)^    authorized_request = dict\(reasoning_result\)\n"
        r"    authorized_request\[\"_smartsoc_authorization\"\]\s*=\s*\{\n"
        r"        \"request_id\": auth\[\"request_id\"\],\n"
        r"        \"actor\": auth\[\"actor\"\],\n"
        r"        \"consumed\": True,\n"
        r"    \}\n"
    )
    fn_new, metadata_count = metadata_pattern.subn(
        "    authorized_request = dict(reasoning_result)\n", fn, count=1
    )
    if metadata_count != 1:
        print(f"ERROR: expected one authorization-metadata injection block, found {metadata_count}. No files changed.")
        return 4

    # Set the invocation flag after a client method/callable returns, independent of whether
    # a mock/adapter includes its own invocation flag. Do not alter success/verified semantics.
    update_pattern = re.compile(
        r"(?m)^    result\.update\(victim_client_invoked=.*?\n"
    )
    fn_new2, update_count = update_pattern.subn(
        '    result.update(victim_client_invoked=True, authorization_request_id=auth["request_id"], authorization_actor=auth["actor"])\n',
        fn_new,
        count=1,
    )
    if update_count != 1:
        print(f"ERROR: expected one invocation-tracking result.update() line, found {update_count}. No files changed.")
        return 5

    updated = source[:fn_match.start()] + fn_new2 + source[fn_match.end():]
    if updated == source:
        print("ERROR: patch produced no changes. No files written.")
        return 6

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_name(TARGET.name + f".bak_approval_client_contract_{stamp}")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(updated, encoding="utf-8")

    # Verify saved file; restore backup if the intended transformation is not present.
    saved = TARGET.read_text(encoding="utf-8")
    saved_fn_match = re.search(
        r"(?ms)^def execute_authorized_client_call\(client,\s*reasoning_result\):\n.*?(?=^def run_console\(\):)",
        saved,
    )
    saved_fn = saved_fn_match.group(0) if saved_fn_match else ""
    if (
        'authorized_request["_smartsoc_authorization"]' in saved_fn
        or "victim_client_invoked=True" not in saved_fn
        or 'authorized_request = dict(reasoning_result)' not in saved_fn
    ):
        shutil.copy2(backup, TARGET)
        print("ERROR: post-write verification failed; original file restored.")
        return 7

    print("PATCH APPLIED")
    print(f"Target : {TARGET}")
    print(f"Backup : {backup}")
    print("Changes:")
    print("  - passes the original event dictionary to client.execute()/callable")
    print("  - records victim_client_invoked=True after the client returns")
    print("  - keeps approval consumption and replay protection unchanged")
    print("No receiver was started; no HTTP/live mitigation request was sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
