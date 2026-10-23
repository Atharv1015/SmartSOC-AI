from pathlib import Path

approval_path = Path("src/integration/local_approval_console_v1.py")
runner_path = Path("src/integration_v2/authorized_dry_run_v2.py")

approval = approval_path.read_text(encoding="utf-8")

old_signature = "def consume_approval(d):"
new_signature = '''def get_approved_request(request_id):
    """Read one approval without consuming or changing its status."""
    request_id = str(request_id or "").strip()
    if not request_id:
        return {"ok": False, "reason": "INVALID_REQUEST_ID"}

    with _db() as c:
        row = c.execute(
            "SELECT * FROM requests WHERE request_id=?",
            (request_id,),
        ).fetchone()

    if not row:
        return {"ok": False, "reason": "NOT_FOUND"}

    if row["status"] != "APPROVED":
        return {"ok": False, "reason": "NOT_APPROVED", "status": row["status"]}

    try:
        expired = _parse(row["expires_at"]) <= _now()
    except (TypeError, ValueError):
        return {"ok": False, "reason": "INVALID_EXPIRY"}

    if expired:
        return {"ok": False, "reason": "EXPIRED"}

    return {
        "ok": True,
        "request_id": row["request_id"],
        "alert_id": row["alert_id"],
        "rule_id": row["rule_id"],
        "action": row["action"],
        "process_name": row["process_name"],
        "pid": row["pid"],
        "status": row["status"],
        "expires_at": row["expires_at"],
    }


def consume_approval(d, request_id=None):'''

if approval.count(old_signature) != 1:
    raise SystemExit("STOP: expected consume_approval signature not found exactly once.")

approval = approval.replace(old_signature, new_signature, 1)

old_query = '''        row=c.execute("SELECT * FROM requests WHERE alert_id=? AND rule_id=? AND action=? AND process_name=? AND pid=? AND status='APPROVED' AND expires_at>? ORDER BY decided_at DESC LIMIT 1",(b["alert_id"],b["rule_id"],b["action"],b["process_name"],b["pid"],_iso(t))).fetchone()'''

new_query = '''        row=c.execute("SELECT * FROM requests WHERE alert_id=? AND rule_id=? AND action=? AND process_name=? AND pid=? AND status='APPROVED' AND expires_at>? AND (? IS NULL OR request_id=?) ORDER BY decided_at DESC LIMIT 1",(b["alert_id"],b["rule_id"],b["action"],b["process_name"],b["pid"],_iso(t),request_id,request_id)).fetchone()'''

if approval.count(old_query) != 1:
    raise SystemExit("STOP: expected approval lookup query not found exactly once.")

approval = approval.replace(old_query, new_query, 1)
approval_path.write_text(approval, encoding="utf-8")

runner = runner_path.read_text(encoding="utf-8")

old_runner_signature = "def execute_authorized_dry_run(reasoning_result, alert_id=None, rule_id=None):"
new_runner_signature = "def execute_authorized_dry_run(reasoning_result, alert_id=None, rule_id=None, approval_request_id=None):"

if runner.count(old_runner_signature) != 1:
    raise SystemExit("STOP: expected orchestrator signature not found exactly once.")

runner = runner.replace(old_runner_signature, new_runner_signature, 1)

old_consume = "authorization = approvals.consume_approval(bound)"
new_consume = "authorization = approvals.consume_approval(bound, request_id=approval_request_id)"

if runner.count(old_consume) != 1:
    raise SystemExit("STOP: expected orchestrator approval call not found exactly once.")

runner = runner.replace(old_consume, new_consume, 1)
runner_path.write_text(runner, encoding="utf-8")

print("PATCH_OK: exact-request approval lookup and consumption are enabled.")
