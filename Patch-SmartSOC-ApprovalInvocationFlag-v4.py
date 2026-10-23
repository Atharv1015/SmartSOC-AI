from __future__ import annotations

import datetime as dt
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "src" / "integration" / "local_approval_console_v1.py"


def find_function(source: str) -> tuple[int, int, str] | None:
    match = re.search(
        r"(?m)^def execute_authorized_client_call\(client,\s*reasoning_result\):\s*$",
        source,
    )
    if not match:
        return None
    next_def = re.search(r"(?m)^def run_console\(\):\s*$", source[match.end():])
    if not next_def:
        return None
    end = match.end() + next_def.start()
    return match.start(), end, source[match.start():end]


def remove_authorization_injection(function: str) -> tuple[str, int]:
    # Remove the current multi-line private metadata block, allowing spacing variations.
    patterns = [
        re.compile(
            r'(?m)^    authorized_request\s*=\s*dict\(reasoning_result\)\s*\n'
            r'    authorized_request\["_smartsoc_authorization"\]\s*=\s*\{\s*\n'
            r'        "request_id"\s*:\s*auth\["request_id"\]\s*,\s*\n'
            r'        "actor"\s*:\s*auth\["actor"\]\s*,\s*\n'
            r'        "consumed"\s*:\s*True\s*,?\s*\n'
            r'    \}\s*\n'
        ),
        re.compile(
            r'(?m)^    authorized_request\s*=\s*dict\(reasoning_result\)\s*\n'
            r'    authorized_request\["_smartsoc_authorization"\]\s*=\s*\{.*?^    \}\s*\n',
            re.S,
        ),
    ]
    for pattern in patterns:
        updated, count = pattern.subn(
            "    authorized_request = dict(reasoning_result)\n", function, count=1
        )
        if count:
            return updated, count
    # If a previous attempt already removed the injection, leave the function as-is.
    if 'authorized_request = dict(reasoning_result)' in function and \
       'authorized_request["_smartsoc_authorization"]' not in function:
        return function, 0
    return function, -1


def replace_result_update(function: str) -> tuple[str, int]:
    matches = list(re.finditer(r"(?m)^    result\.update\s*\(", function))
    if len(matches) != 1:
        return function, -1

    start = matches[0].start()
    open_paren = function.find("(", matches[0].start())
    depth = 0
    quote = None
    escaped = False
    end = None
    for i in range(open_paren, len(function)):
        ch = function[i]
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return function, -1

    replacement = (
        '    result.update(\n'
        '        victim_client_invoked=True,\n'
        '        authorization_request_id=auth["request_id"],\n'
        '        authorization_actor=auth["actor"],\n'
        '    )'
    )
    return function[:start] + replacement + function[end:], 1


def main() -> int:
    if not TARGET.is_file():
        print(f"ERROR: target file not found: {TARGET}")
        return 2

    source = TARGET.read_text(encoding="utf-8")
    found = find_function(source)
    if not found:
        print("ERROR: could not isolate execute_authorized_client_call(). No files changed.")
        return 3

    start, end, function = found
    function, metadata_status = remove_authorization_injection(function)
    if metadata_status == -1:
        print("ERROR: could not safely normalize authorized_request metadata. No files changed.")
        return 4

    function, update_status = replace_result_update(function)
    if update_status != 1:
        print(f"ERROR: expected exactly one result.update() call in the function; status={update_status}. No files changed.")
        return 5

    if 'authorized_request["_smartsoc_authorization"]' in function:
        print("ERROR: authorization metadata injection remains. No files changed.")
        return 6

    updated = source[:start] + function + source[end:]
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_name(TARGET.name + f".bak_approval_client_contract_{stamp}")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(updated, encoding="utf-8")

    saved = TARGET.read_text(encoding="utf-8")
    verify = find_function(saved)
    if not verify:
        shutil.copy2(backup, TARGET)
        print("ERROR: post-write function verification failed; original restored.")
        return 7
    saved_fn = verify[2]
    if (
        'authorized_request["_smartsoc_authorization"]' in saved_fn
        or "victim_client_invoked=True" not in saved_fn
        or 'authorized_request = dict(reasoning_result)' not in saved_fn
    ):
        shutil.copy2(backup, TARGET)
        print("ERROR: post-write content verification failed; original restored.")
        return 8

    print("PATCH APPLIED")
    print(f"Target : {TARGET}")
    print(f"Backup : {backup}")
    print("Changes:")
    print("  - handles both single-line and multi-line result.update() blocks")
    print("  - passes the original event dictionary to the client")
    print("  - records victim_client_invoked=True after the client call returns")
    print("  - preserves approval consumption and replay protection")
    print("No receiver was started; no HTTP/live mitigation request was sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
