#!/usr/bin/env python3
"""Read-only SmartSOC master audit. Run from the project root; no mitigation/webhook calls."""
from __future__ import annotations
import ast, json, os, re, socket, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "src").is_dir():
    print("ERROR: Run from SmartSOC-AI project root (folder containing src).")
    print("Current directory:", ROOT)
    raise SystemExit(2)
OUTDIR = ROOT / "reports" / "audit"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / f"smartsoc_master_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
REPORT = []

def out(s=""):
    s = str(s)
    REPORT.append(s)
    print(s)

def section(s):
    out()
    out("=" * 78)
    out(s)
    out("=" * 78)

def command(args):
    try:
        p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=8)
        return ((p.stdout or "") + (p.stderr or "")).strip() or f"(no output; exit={p.returncode})"
    except Exception as e:
        return f"(unavailable: {e})"

KEY_FILES = [
 "src/integration/smartsoc_live_receiver_v2.py",
 "src/integration/execution_config_v1.py",
 "src/integration_v2/execution_config_v2.py",
 "src/integration/authorization_gate_v1.py",
 "src/integration/authorization_decision_v1.py",
 "src/integration/execution_policy_guard_v1.py",
 "src/integration/authorization_validator_v2.py",
 "src/integration/response_planner_v1.py",
 "src/decision/smartsoc_risk_engine_v4_1.py",
 "src/decision/smartsoc_decision_policy_v4_1.py",
 "src/decision/smartsoc_response_planner_v4_1.py",
 "src/autonomous/autonomous_mitigation_controller_v1.py",
 "src/autonomous_v2/victim_mitigation_adapter_v2.py",
]
POLICY = re.compile(r"(MITIGATION_ENABLED|VICTIM_AGENT_CALLED|EXECUTION_LAYER_ENABLED|AUTOMATIC_EXECUTION|AUTOMATIC_CONTAINMENT|UNRESTRICTED_AUTONOMOUS_ACTIONS|CONTROLLED_LAB_MODE|HUMAN_AUTHORIZATION_REQUIRED|ALLOWED_ACTIONS|ALLOWED_PROCESSES|DRY_RUN|validate_safety_invariants|validate_execution_policy|MITIGATION_BLOCKED|victim_client_invoked|authorization_required|analyst_authorized|containment_authorized|automatic_response|automatic_blocking|automatic_isolation|requires_approval|fallback_can_directly_contain|webhook|victim_token)", re.I)
SECRET = re.compile(r"(token|secret|password|api.?key|credential)\s*=", re.I)

def audit_py(path):
    rel = path.relative_to(ROOT).as_posix()
    section("PYTHON SOURCE: " + rel)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    out(f"Bytes={path.stat().st_size}; lines={len(lines)}")
    try:
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out(f"  {type(node).__name__} {node.name} (line {node.lineno})")
            elif isinstance(node, ast.Assign):
                names = [x.id for x in node.targets if isinstance(x, ast.Name)]
                if names and any(re.search(r"(ENABLED|EXECUTION|CONTAINMENT|AUTHORIZATION|APPROVAL|ALLOW|DENY|DRY_RUN|LAB_MODE|VICTIM|PORT|ACTION|POLICY|MITIGATION|AUTOMATIC)", n, re.I) for n in names):
                    try: value = repr(ast.literal_eval(node.value))[:250]
                    except Exception: value = ast.unparse(node.value)[:180]
                    if SECRET.search(" ".join(names)): value = "<redacted>"
                    out(f"  POLICY CONSTANT line {node.lineno}: {', '.join(names)} = {value}")
    except SyntaxError as e:
        out(f"AST parse error: {e}")
    out("-- policy/safety matches (context; likely secrets redacted) --")
    hits = [i for i, line in enumerate(lines) if POLICY.search(line)]
    shown = set()
    for i in hits[:100]:
        for j in range(max(0, i-1), min(len(lines), i+2)):
            if j in shown: continue
            shown.add(j)
            line = lines[j]
            if SECRET.search(line):
                line = SECRET.sub("<redacted assignment>", line)
            out(f"  {j+1:5}: {line[:260]}")
        out("  ...")

def audit_json(path):
    section("JSON ARTIFACT: " + path.relative_to(ROOT).as_posix())
    try: data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        out("JSON parse error: " + str(e)); return
    out("Top-level keys: " + ", ".join(map(str, data.keys())) if isinstance(data, dict) else f"Root type: {type(data).__name__}")
    keys = re.compile(r"(schema|timestamp|source|response_policy|policy|authorization|execution_control|safety|risk|decision|status|event_id|alert_id|rule_id|execution_allowed|analyst_authorized|authorization_required|automatic_execution|automatic_containment|human_in_the_loop|execution_blocked|risk_score|severity|recommended_action)", re.I)
    def walk(obj, prefix="", depth=0):
        if depth > 4 or not isinstance(obj, dict): return
        for k,v in obj.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            if keys.search(str(k)):
                out(f"  {name}: {json.dumps(v, ensure_ascii=False)[:350]}" if not isinstance(v, (dict,list)) else f"  {name}: <{type(v).__name__}, {len(v)} items>")
            if isinstance(v,dict): walk(v,name,depth+1)
            elif isinstance(v,list) and len(v) <= 12:
                for n,item in enumerate(v):
                    if isinstance(item,dict): walk(item,f"{name}[{n}]",depth+1)
    if isinstance(data,dict): walk(data)

section("SMARTSOC MASTER AUDIT")
out("Root: " + str(ROOT))
out("Local time: " + datetime.now().astimezone().isoformat())
out("UTC: " + datetime.now(timezone.utc).isoformat())
out("Python: " + sys.version.replace(os.linesep, " "))
out("MODE: READ-ONLY. No source/config edits, webhook calls, victim endpoint calls, or mitigation.")
section("GIT STATUS")
out(command(["git","status","--short"]))
section("RECENT COMMITS")
out(command(["git","log","-5","--oneline"]))
section("KEY FILE CHECK")
for rel in KEY_FILES:
    p = ROOT / rel
    out(("FOUND  " if p.is_file() else "MISSING") + rel)
for rel in KEY_FILES:
    p = ROOT / rel
    if p.is_file() and p.suffix == ".py":
        audit_py(p)

for folder, suffixes in [(ROOT/"src",{".py"}),(ROOT/"tests",{".py"}),(ROOT/"reports"/"integration",{".json"}),(ROOT/"models",None)]:
    section("INVENTORY: " + str(folder.relative_to(ROOT)))
    if not folder.exists():
        out("(directory missing)"); continue
    fs = [p for p in folder.rglob("*") if p.is_file() and (suffixes is None or p.suffix.lower() in suffixes)]
    out(f"File count: {len(fs)}")
    for p in sorted(fs)[:160]:
        out(f"  {p.relative_to(ROOT).as_posix()} ({p.stat().st_size} bytes)")
    if len(fs)>160: out(f"  ... {len(fs)-160} files omitted")

for rel in ["reports/integration/response_plan_v1.json","reports/integration/authorization_gate_result_v1.json","reports/integration/authorization_decision_v1.json","reports/integration/execution_policy_guard_result_v1.json","reports/integration/risk_engine_result.json","reports/integration/shuffle_dry_run_result_v1.json"]:
    p=ROOT/rel
    if p.is_file(): audit_json(p)

for port in (8787,8788):
    section(f"LOCAL PORT CHECK: {port}")
    try:
        import psutil
        matches=[f"pid={c.pid}, {c.laddr.ip}:{c.laddr.port}" for c in psutil.net_connections(kind="tcp") if c.laddr and c.laddr.port==port and c.status=="LISTEN"]
        out("\n".join(matches) if matches else "No listener found (or permissions restrict inspection).")
    except ImportError:
        s=socket.socket(); s.settimeout(.4)
        try: out("127.0.0.1 TCP accepts connection" if s.connect_ex(("127.0.0.1",port))==0 else "No localhost listener detected; service may bind another interface.")
        finally: s.close()
    except Exception as e: out(f"Listener inspection unavailable: {e}")

section("POLICY REUSE CHECKLIST (recommendations only; no changes applied)")
for item in [
 "Treat previous source/config files as policy evidence; check timestamps and current code before reuse.",
 "Keep CONTROLLED_LAB_MODE=True, DRY_RUN=True, and UNRESTRICTED_AUTONOMOUS_ACTIONS=False during audit.",
 "Keep EXECUTION_LAYER_ENABLED, AUTOMATIC_EXECUTION, and AUTOMATIC_CONTAINMENT=False until separately reviewed and tested.",
 "Keep action/process allowlists narrow and deny unknown actions.",
 "Automate triage, enrichment, evidence collection, and notifications separately from disruptive endpoint actions.",
 "Do not trust stale JSON reports as live policy; regenerate only intentionally.",
 "An actor string in an approval record does not authenticate the approver.",
 "Do not enable receiver mitigation while placeholder credentials remain.",
 "'victim_client_invoked' indicates code-path invocation, not confirmed remote delivery.",
 "This script only reports current state; it does not configure or modify SmartSOC."
]: out("- " + item)

section("AUDIT COMPLETE")
out("Report path: " + str(OUTFILE))
OUTFILE.write_text("\n".join(REPORT) + "\n", encoding="utf-8")
print("\nFinished. Review report for private IPs or secrets before sharing.")
