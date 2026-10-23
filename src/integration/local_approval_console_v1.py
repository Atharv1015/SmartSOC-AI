
from __future__ import annotations
import os, sqlite3, uuid, getpass
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

ACTION = "PROCESS_TERMINATION"
TTL_SECONDS = 300

def _now():
    return datetime.now(timezone.utc)
def _iso(value):
    return value.astimezone(timezone.utc).isoformat()
def _parse(value):
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError("timezone required")
    return dt.astimezone(timezone.utc)
def _root():
    return Path(__file__).resolve().parents[2]
def db_path():
    return Path(os.environ["SMARTSOC_APPROVAL_DB"]).resolve() if os.getenv("SMARTSOC_APPROVAL_DB") else _root()/"reports/authorization/local_approval.sqlite3"
def token_path():
    return Path(os.environ["SMARTSOC_APPROVAL_TOKEN_FILE"]).resolve() if os.getenv("SMARTSOC_APPROVAL_TOKEN_FILE") else _root()/".secrets/smartsoc_approval_console.token"
def _connect():
    p=db_path(); p.parent.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(str(p),timeout=10,isolation_level=None); c.row_factory=sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE IF NOT EXISTS requests(request_id TEXT PRIMARY KEY, alert_id TEXT NOT NULL, rule_id TEXT NOT NULL, action TEXT NOT NULL, process_name TEXT NOT NULL, pid INTEGER NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, expires_at TEXT NOT NULL, actor TEXT, decided_at TEXT, consumed_at TEXT)")
    c.execute("CREATE INDEX IF NOT EXISTS request_binding ON requests(alert_id,rule_id,action,process_name,pid,status)")
    return c
@contextmanager
def _db():
    c=_connect()
    try:
        yield c
    finally:
        c.close()
def _val(d,*keys):
    for key in keys:
        if d.get(key) is not None and str(d[key]).strip(): return d[key]
    return None
def _binding(d,alert_id=None,rule_id=None):
    if not isinstance(d,dict): return None
    alert=alert_id or _val(d,"alert_id","wazuh_alert_id","id")
    rule=rule_id or _val(d,"rule_id","wazuh_rule_id")
    mit=d.get("mitigation")
    m=mit if isinstance(mit,dict) else d
    action=_val(m,"action","type","mitigation_type","mitigation_action","recommended_action")
    if not action and isinstance(mit,str): action=mit
    pid=_val(m,"pid","process_id","target_pid")
    proc=_val(m,"process_name","process","target_process")
    target=d.get("target")
    if isinstance(target,dict):
        pid=pid or _val(target,"pid","process_id")
        proc=proc or _val(target,"process_name","process")
    try: pid=int(pid)
    except (TypeError,ValueError): return None
    if not all((alert,rule,action,proc)) or pid<=0 or str(action).upper()!=ACTION: return None
    return {"alert_id":str(alert),"rule_id":str(rule),"action":ACTION,"process_name":str(proc).strip().lower(),"pid":pid}
def register_pending_from_reasoning_result(d,alert_id,rule_id):
    b=_binding(d,alert_id,rule_id)
    if not b: return {"registered":False,"reason":"INCOMPLETE_OR_UNSUPPORTED_BINDING"}
    t=_now()
    with _db() as c:
        row=c.execute("SELECT request_id FROM requests WHERE alert_id=? AND rule_id=? AND action=? AND process_name=? AND pid=? AND status='PENDING' AND expires_at>? ORDER BY created_at DESC LIMIT 1",(b["alert_id"],b["rule_id"],b["action"],b["process_name"],b["pid"],_iso(t))).fetchone()
        if row: return {"registered":True,"request_id":row["request_id"],"status":"PENDING"}
        rid=str(uuid.uuid4())
        c.execute("INSERT INTO requests VALUES(?,?,?,?,?,?,'PENDING',?,?,NULL,NULL,NULL)",(rid,b["alert_id"],b["rule_id"],b["action"],b["process_name"],b["pid"],_iso(t),_iso(t+timedelta(seconds=TTL_SECONDS))))
    return {"registered":True,"request_id":rid,"status":"PENDING"}
def bind_event_context(d,alert_id,rule_id):
    if not isinstance(d,dict): return d
    out=dict(d); out["alert_id"]=str(alert_id or ""); out["rule_id"]=str(rule_id or ""); return out
def list_requests():
    t=_iso(_now())
    with _db() as c:
        c.execute("UPDATE requests SET status='EXPIRED' WHERE status IN ('PENDING','APPROVED') AND expires_at<=?",(t,))
        return [dict(r) for r in c.execute("SELECT * FROM requests WHERE status IN ('PENDING','APPROVED') ORDER BY created_at DESC LIMIT 200")]
def decide_request(request_id,decision,actor):
    decision=str(decision).upper(); actor=str(actor).strip(); t=_now()
    if decision not in ("APPROVED","DENIED") or not actor or len(actor)>128: return {"ok":False,"reason":"INVALID_INPUT"}
    with _db() as c:
        c.execute("BEGIN IMMEDIATE")
        row=c.execute("SELECT * FROM requests WHERE request_id=?",(str(request_id),)).fetchone()
        if not row: c.execute("ROLLBACK"); return {"ok":False,"reason":"NOT_FOUND"}
        if row["status"]!="PENDING": c.execute("ROLLBACK"); return {"ok":False,"reason":"NOT_PENDING","status":row["status"]}
        if _parse(row["expires_at"])<=t:
            c.execute("UPDATE requests SET status='EXPIRED' WHERE request_id=?",(str(request_id),)); c.execute("COMMIT")
            return {"ok":False,"reason":"EXPIRED"}
        c.execute("UPDATE requests SET status=?,actor=?,decided_at=? WHERE request_id=? AND status='PENDING'",(decision,actor,_iso(t),str(request_id)))
        c.execute("COMMIT")
    return {"ok":True,"status":decision,"request_id":str(request_id),"actor":actor}
def get_approved_request(request_id):
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


def consume_approval(d, request_id=None):
    b=_binding(d)
    if not b: return {"authorized":False,"reason":"INVALID_BINDING"}
    t=_now()
    with _db() as c:
        c.execute("BEGIN IMMEDIATE")
        c.execute("UPDATE requests SET status='EXPIRED' WHERE status IN ('PENDING','APPROVED') AND expires_at<=?",(_iso(t),))
        row=c.execute("SELECT * FROM requests WHERE alert_id=? AND rule_id=? AND action=? AND process_name=? AND pid=? AND status='APPROVED' AND expires_at>? AND (? IS NULL OR request_id=?) ORDER BY decided_at DESC LIMIT 1",(b["alert_id"],b["rule_id"],b["action"],b["process_name"],b["pid"],_iso(t),request_id,request_id)).fetchone()
        if not row: c.execute("ROLLBACK"); return {"authorized":False,"reason":"NO_MATCHING_APPROVAL"}
        n=c.execute("UPDATE requests SET status='CONSUMED',consumed_at=? WHERE request_id=? AND status='APPROVED'",(_iso(t),row["request_id"])).rowcount
        if n!=1: c.execute("ROLLBACK"); return {"authorized":False,"reason":"ALREADY_CONSUMED"}
        c.execute("COMMIT")
    return {"authorized":True,"request_id":row["request_id"],"actor":row["actor"]}
def execute_authorized_client_call(client,reasoning_result):
    auth=consume_approval(reasoning_result)
    if not auth.get("authorized"):
        return {"success":False,"verified":False,"status":"AUTHORIZATION_BLOCKED","reason":auth.get("reason"),"victim_client_invoked":False,"endpoint_modified":False}
    authorized_request = dict(reasoning_result)
    execute_method = getattr(client, 'execute', None)
    if callable(execute_method):
        raw_result = execute_method(authorized_request)
    elif callable(client):
        raw_result = client(authorized_request)
    else:
        return {
            "success": False, "verified": False,
            "status": "INVALID_CLIENT",
            "victim_client_invoked": False,
            "endpoint_modified": False,
        }
    result=dict(raw_result) if isinstance(raw_result,dict) else {"success":False,"verified":False,"status":"INVALID_CLIENT_RESULT"}
    result.update(
        victim_client_invoked=True,
        authorization_request_id=auth["request_id"],
        authorization_actor=auth["actor"],
    )
    return result
def run_console():
    expected=token_path().read_text(encoding="utf-8").strip() if token_path().exists() else ""
    supplied=getpass.getpass("SmartSOC local approval token: ")
    if len(expected)<32 or not __import__("hmac").compare_digest(supplied,expected):
        raise SystemExit("Authentication failed.")
    actor=input("Operator label for audit (not a separate identity proof): ").strip()
    if not actor or len(actor)>128: raise SystemExit("Invalid operator label.")
    print("Authenticated local approval console. Mitigation remains disabled.")
    while True:
        requests=list_requests()
        print("\\nActive requests:")
        if not requests: print("  (none)")
        for i,r in enumerate(requests,1):
            print(f"{i}. {r['status']} alert={r['alert_id']} rule={r['rule_id']} action={r['action']} process={r['process_name']} pid={r['pid']} expires={r['expires_at']} id={r['request_id']}")
        choice=input("\\nEnter request number to decide, R to refresh, Q to quit: ").strip().lower()
        if choice=="q": break
        if choice=="r": continue
        try: item=requests[int(choice)-1]
        except (ValueError,IndexError): print("Invalid selection."); continue
        if item["status"]!="PENDING": print("Only PENDING requests can be decided."); continue
        decision=input("Type APPROVE or DENY: ").strip().upper()
        if decision not in ("APPROVE","DENY"): print("No decision recorded."); continue
        if decision=="APPROVE":
            confirm=input("To confirm this exact alert/rule/process/PID, type APPROVE: ").strip()
            if confirm!="APPROVE": print("Approval cancelled."); continue
        result=decide_request(item["request_id"],"APPROVED" if decision=="APPROVE" else "DENIED",actor)
        print(result)
if __name__=="__main__": run_console()
