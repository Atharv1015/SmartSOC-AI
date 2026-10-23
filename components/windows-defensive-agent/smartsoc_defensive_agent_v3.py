"""
SmartSOC Defensive Agent v3 — HMAC-compatible controlled-lab endpoint.

Preserves the existing victim endpoint: 0.0.0.0:8765
Routes:
  GET  /health
  POST /mitigate

Only PROCESS_TERMINATION for explicitly allowlisted lab processes is supported.
No arbitrary command execution. The caller must provide an HMAC-signed request
using SMARTSOC_AGENT_SECRET and the SmartSOC v2 canonical signing format.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import subprocess
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

AGENT_VERSION = "smartsoc-defensive-agent-v3-hmac"
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8765
MAX_BODY_BYTES = 16_384
MAX_CLOCK_SKEW_SECONDS = 120
MAX_NONCES = 10_000
AUDIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "logs", "smartsoc_defensive_agent_v3_audit.jsonl")

ALLOWED_ACTIONS = frozenset({"PROCESS_TERMINATION"})
# Keep this tight and aligned with the current SmartSOC controlled-lab policy.
ALLOWED_PROCESS_NAMES = frozenset({"notepad.exe", "ping.exe"})
ALLOWED_CLIENT_IPS = frozenset({"192.168.56.1", "127.0.0.1", "::1"})
PROTECTED_PROCESS_NAMES = frozenset({
    "system", "system idle process", "smss.exe", "csrss.exe",
    "wininit.exe", "services.exe", "lsass.exe", "winlogon.exe",
    "svchost.exe", "registry", "memory compression",
})

_nonce_lock = threading.Lock()
_seen_nonces: OrderedDict[str, float] = OrderedDict()
_audit_lock = threading.Lock()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_process_name(value: Optional[str]) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip().replace("/", "\\")
    return value.rsplit("\\", 1)[-1].lower()


def json_response(action: str, target: Dict[str, Any], success: bool,
                  verified: bool, message: str, **extra: Any) -> Dict[str, Any]:
    status = "TERMINATED_VERIFIED" if success and verified else (
        "MITIGATION_REJECTED" if not success else "MITIGATION_UNVERIFIED"
    )
    return {
        "agent": AGENT_VERSION,
        "timestamp": utc_timestamp(),
        "action": action,
        "target": target,
        "success": bool(success),
        "verified": bool(verified),
        "status": status,
        "message": message,
        **extra,
    }


def audit(event: Dict[str, Any]) -> None:
    record = {"timestamp": utc_timestamp(), **event}
    try:
        os.makedirs(os.path.dirname(AUDIT_PATH), exist_ok=True)
        with _audit_lock:
            with open(AUDIT_PATH, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        # Do not leak secrets or request signatures in logs.
        print("[AUDIT_WRITE_FAILED]", type(exc).__name__, str(exc))


def build_signature(secret: str, timestamp: str, nonce: str, body: bytes) -> str:
    message = timestamp.encode("ascii") + b"\n" + nonce.encode("ascii") + b"\n" + body
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def remember_nonce(nonce: str, now: float) -> bool:
    """Return False for a replay; expire old nonce entries."""
    with _nonce_lock:
        expired = [key for key, seen_at in _seen_nonces.items()
                   if now - seen_at > MAX_CLOCK_SKEW_SECONDS]
        for key in expired:
            _seen_nonces.pop(key, None)
        if nonce in _seen_nonces:
            return False
        _seen_nonces[nonce] = now
        while len(_seen_nonces) > MAX_NONCES:
            _seen_nonces.popitem(last=False)
        return True


class DefensiveExecutor:
    def get_process(self, pid: int) -> Dict[str, Any]:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10, shell=False,
        )
        stdout = result.stdout.strip()
        if not stdout or "INFO:" in stdout:
            return {"running": False, "pid": pid}
        # tasklist CSV: "image.exe","PID",...
        try:
            import csv
            import io
            row = next(csv.reader(io.StringIO(stdout)))
            name = row[0] if row else ""
        except Exception:
            name = ""
        if not name:
            return {"running": False, "pid": pid, "inspection_error": "Unable to parse tasklist output"}
        return {"running": True, "pid": pid, "process_name": name,
                "normalized_process_name": normalize_process_name(name)}

    def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = str(request.get("action", ""))
        target = request.get("target")
        if not isinstance(target, dict):
            target = {}
        raw_pid = target.get("pid")
        raw_name = target.get("process_name")
        name = normalize_process_name(raw_name)
        target_out = {"pid": raw_pid, "process_name": name}

        if action not in ALLOWED_ACTIONS:
            return json_response(action, target_out, False, False, "Action is not allowlisted.")
        if isinstance(raw_pid, bool) or not isinstance(raw_pid, int) or raw_pid <= 0:
            return json_response(action, target_out, False, False, "PID must be a positive integer.")
        if name not in ALLOWED_PROCESS_NAMES:
            return json_response(action, target_out, False, False, "Process name is not allowlisted.")
        if name in PROTECTED_PROCESS_NAMES:
            return json_response(action, target_out, False, False, "Protected process denied.")

        before = self.get_process(raw_pid)
        if not before.get("running"):
            result = json_response(action, target_out, False, False, "Target process is not running.",
 pre_state=before)
            audit({"event": "MITIGATION_REJECTED", "result": result})
            return result
        actual = before.get("normalized_process_name", "")
        if actual != name:
            result = json_response(action, target_out, False, False,
                                   "Live process identity does not match requested process.",
                                   pre_state=before, actual_process_name=actual)
            audit({"event": "IDENTITY_MISMATCH", "result": result})
            return result
        try:
            terminated = subprocess.run(
                ["taskkill", "/PID", str(raw_pid), "/F"],
                capture_output=True, text=True, timeout=15, shell=False,
            )
            after = self.get_process(raw_pid)
            verified = not after.get("running", False)
            success = terminated.returncode == 0 and verified
            result = json_response(
                action, target_out, success, verified,
                "Process terminated and exit verified." if success else "Termination could not be verified.",
                pre_state=before,
                execution={"return_code": terminated.returncode,
                           "stdout": terminated.stdout[-2000:],
                           "stderr": terminated.stderr[-2000:]},
                post_state=after,
            )
            audit({"event": "MITIGATION_RESULT", "result": result})
            return result
        except Exception as exc:
            result = json_response(action, target_out, False, False,
                                   "Defensive termination failed.", error=f"{type(exc).__name__}: {exc}")
            audit({"event": "MITIGATION_ERROR", "result": result})
            return result


class SmartSOCRequestHandler(BaseHTTPRequestHandler):
    executor = DefensiveExecutor()
    server_version = "SmartSOCDefensiveAgent/3"

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[HTTP]", fmt % args)

    def send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def client_allowed(self) -> bool:
        ip = self.client_address[0]
        if ip.startswith("::ffff:"):
            ip = ip[7:]
        return ip in ALLOWED_CLIENT_IPS

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_json(404, {"status": "NOT_FOUND", "error": "Endpoint not found."})
            return
        secret_ok = len(os.environ.get("SMARTSOC_AGENT_SECRET", "")) >= 32
        gate_on = os.environ.get("SMARTSOC_ALLOW_TERMINATION", "").strip() == "1"
        self.send_json(200, {
            "status": "HEALTHY" if secret_ok and gate_on else "DEGRADED",
            "agent": AGENT_VERSION,
            "host": LISTEN_HOST,
            "port": LISTEN_PORT,
            "mode": "CONTROLLED_LAB",
            "execution_enabled": bool(secret_ok and gate_on),
            "authentication": "HMAC-SHA256",
            "allowed_actions": sorted(ALLOWED_ACTIONS),
            "allowed_processes": sorted(ALLOWED_PROCESS_NAMES),
            "allowed_clients": sorted(ALLOWED_CLIENT_IPS),
            "arbitrary_commands": False,
            "replay_protection": True,
            "process_identity_validation": True,
            "post_action_verification": True,
        })

    def do_POST(self) -> None:
        if self.path != "/mitigate":
            self.send_json(404, {"status": "NOT_FOUND", "error": "Endpoint not found."})
            return
        if not self.client_allowed():
            audit({"event": "CLIENT_IP_DENIED", "client_ip": self.client_address[0]})
            self.send_json(403, {"status": "CLIENT_IP_DENIED", "success": False, "verified": False})
            return

        secret = os.environ.get("SMARTSOC_AGENT_SECRET", "")
        if len(secret) < 32 or os.environ.get("SMARTSOC_ALLOW_TERMINATION", "").strip() != "1":
            self.send_json(503, {"status": "EXECUTION_DISABLED", "success": False, "verified": False})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            self.send_json(400, {"status": "INVALID_CONTENT_LENGTH", "success": False, "verified": False})
            return
        body = self.rfile.read(length)
        timestamp = self.headers.get("X-SmartSOC-Timestamp", "")
        nonce = self.headers.get("X-SmartSOC-Nonce", "")
        supplied_signature = self.headers.get("X-SmartSOC-Signature", "")

        try:
            timestamp_int = int(timestamp)
        except (ValueError, TypeError):
            timestamp_int = 0
        now = time.time()
        if not timestamp_int or abs(now - timestamp_int) > MAX_CLOCK_SKEW_SECONDS:
            self.send_json(401, {"status": "TIMESTAMP_REJECTED", "success": False, "verified": False})
            return
        if not nonce or len(nonce) > 128 or not supplied_signature:
            self.send_json(401, {"status": "AUTH_HEADERS_MISSING", "success": False, "verified": False})
            return
        expected_signature = build_signature(secret, timestamp, nonce, body)
        if not hmac.compare_digest(supplied_signature, expected_signature):
            audit({"event": "AUTHENTICATION_FAILED", "client_ip": self.client_address[0]})
            self.send_json(401, {"status": "AUTHENTICATION_FAILED", "success": False, "verified": False})
            return
        if not remember_nonce(nonce, now):
            self.send_json(409, {"status": "REPLAY_REJECTED", "success": False, "verified": False})
            return

        try:
            request = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json(400, {"status": "INVALID_JSON", "success": False, "verified": False})
            return
        if not isinstance(request, dict):
            self.send_json(400, {"status": "INVALID_REQUEST", "success": False, "verified": False})
            return

        result = self.executor.execute(request)
        audit({"event": "REQUEST_PROCESSED", "client_ip": self.client_address[0],
               "action": request.get("action"), "status": result.get("status"),
               "success": result.get("success"), "verified": result.get("verified")})
        self.send_json(200 if result.get("success") and result.get("verified") else 400, result)


def main() -> None:
    secret = os.environ.get("SMARTSOC_AGENT_SECRET", "")
    gate_on = os.environ.get("SMARTSOC_ALLOW_TERMINATION", "").strip() == "1"
    print("=" * 68)
    print("SmartSOC Defensive Agent v3 (HMAC-compatible)")
    print(f"Listen: {LISTEN_HOST}:{LISTEN_PORT}")
    print(f"Allowed actions: {sorted(ALLOWED_ACTIONS)}")
    print(f"Allowed processes: {sorted(ALLOWED_PROCESS_NAMES)}")
    print(f"HMAC secret configured: {len(secret) >= 32}")
    print(f"Live termination gate enabled: {gate_on}")
    print("Arbitrary commands: DISABLED")
    if len(secret) < 32 or not gate_on:
        print("WARNING: /mitigate will remain disabled until the secret and gate are set.")
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), SmartSOCRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSmartSOC Defensive Agent stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
