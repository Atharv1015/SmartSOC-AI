from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "src" / "integration" / "local_approval_console_v1.py"

OLD = "\n".join([
    '    authorized_request = dict(reasoning_result)',
    '    authorized_request["_smartsoc_authorization"] = {',
    '        "request_id": auth["request_id"],',
    '        "actor": auth["actor"],',
    '        "consumed": True,',
    '    }',
    '    execute_method = getattr(client, \'execute\', None)',
    '    if callable(execute_method):',
    '        raw_result = execute_method(authorized_request)',
    '    elif callable(client):',
    '        raw_result = client(authorized_request)',
    '    else:',
    '        return {',
    '            "success": False, "verified": False,',
    '            "status": "INVALID_CLIENT",',
    '            "victim_client_invoked": False,',
    '            "endpoint_modified": False,',
    '        }',
    '    result=dict(raw_result) if isinstance(raw_result,dict) else {"success":False,"verified":False,"status":"INVALID_CLIENT_RESULT"}',
    '',
    '    result.update(victim_client_invoked=bool(result.get("victim_request_attempted", False) or result.get("victim_client_invoked", False)),authorization_request_id=auth["request_id"],authorization_actor=auth["actor"])',
    '    return result',
])

NEW = "\n".join([
    '    # Approval is consumed above before any client call. Pass a clean copy of',
    '    # the original event to preserve the established client interface; do not',
    '    # add private authorization metadata to the event payload.',
    '    authorized_request = dict(reasoning_result)',
    '    execute_method = getattr(client, \'execute\', None)',
    '    if callable(execute_method):',
    '        raw_result = execute_method(authorized_request)',
    '    elif callable(client):',
    '        raw_result = client(authorized_request)',
    '    else:',
    '        return {',
    '            "success": False, "verified": False,',
    '            "status": "INVALID_CLIENT",',
    '            "victim_client_invoked": False,',
    '            "endpoint_modified": False,',
    '        }',
    '    result=dict(raw_result) if isinstance(raw_result,dict) else {"success":False,"verified":False,"status":"INVALID_CLIENT_RESULT"}',
    '',
    '    # A returned client result proves the client method/callable was invoked;',
    '    # endpoint_modified and verified remain based on the actual adapter result.',
    '    result.update(',
    '        victim_client_invoked=True,',
    '        authorization_request_id=auth["request_id"],',
    '        authorization_actor=auth["actor"],',
    '    )',
    '    return result',
])

def main() -> int:
    if not TARGET.is_file():
        print(f"ERROR: target file not found: {TARGET}")
        print("Place this patch in the SmartSOC-AI repository root and run it there.")
        return 2

    source = TARGET.read_text(encoding="utf-8")
    count = source.count(OLD)
    if count != 1:
        print(f"ERROR: expected exactly one target block, found {count}. No files changed.")
        print("Patch aborted safely; inspect the current function before adapting.")
        return 3

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_name(TARGET.name + f".bak_invocation_flag_{stamp}")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(source.replace(OLD, NEW, 1), encoding="utf-8")

    check = TARGET.read_text(encoding="utf-8")
    if NEW not in check or OLD in check:
        shutil.copy2(backup, TARGET)
        print("ERROR: verification failed; original file restored.")
        return 4

    print("PATCH APPLIED")
    print(f"Target : {TARGET}")
    print(f"Backup : {backup}")
    print("Changes:")
    print("  - records victim_client_invoked=True only after the authorized client returns")
    print("  - passes the original event fields without injecting private metadata")
    print("  - approval consumption still occurs before client invocation")
    print("No receiver was started; no HTTP/live mitigation request was sent.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
