from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BIND_HOST = "192.168.56.1"
PORT = 8790
ALLOWED_VICTIM_IP = "192.168.56.102"
MAX_BYTES = 50 * 1024 * 1024
MAX_SKEW_SECONDS = 60
ROOT = Path(__file__).resolve().parent.parent / "containment" / "evidence"
MANIFEST = ROOT / "manifest.jsonl"
SAFE_CASE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,180}$")
nonces: dict[str, float] = {}
nonce_lock = threading.Lock()
manifest_lock = threading.Lock()


def shared_secret() -> bytes:
    value = os.environ.get("SMARTSOC_EVIDENCE_SECRET", "")
    if len(value) < 32:
        raise RuntimeError("SMARTSOC_EVIDENCE_SECRET must be set to at least 32 characters")
    return value.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "SmartSOC-Evidence/1.0"

    def log_message(self, fmt, *args):
        print(f"[evidence] {self.client_address[0]} {fmt % args}")

    def reply(self, status: int, payload: dict):
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        if self.path != "/evidence":
            return self.reply(404, {"status": "NOT_FOUND"})
        if self.client_address[0] != ALLOWED_VICTIM_IP:
            return self.reply(403, {"status": "CLIENT_IP_DENIED"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_BYTES:
            return self.reply(413, {"status": "INVALID_SIZE", "max_bytes": MAX_BYTES})

        case_id = self.headers.get("X-SmartSOC-Case", "")
        filename = self.headers.get("X-SmartSOC-Filename", "")
        claimed_hash = self.headers.get("X-SmartSOC-SHA256", "").lower()
        timestamp = self.headers.get("X-SmartSOC-Timestamp", "")
        nonce = self.headers.get("X-SmartSOC-Nonce", "")
        signature = self.headers.get("X-SmartSOC-Signature", "").lower()
        if (not SAFE_CASE.fullmatch(case_id) or not SAFE_NAME.fullmatch(filename)
                or not re.fullmatch(r"[a-f0-9]{64}", claimed_hash)
                or not timestamp.isdigit() or not re.fullmatch(r"[a-f0-9]{32}", nonce)):
            return self.reply(400, {"status": "INVALID_HEADERS"})
        if abs(int(time.time()) - int(timestamp)) > MAX_SKEW_SECONDS:
            return self.reply(401, {"status": "REQUEST_EXPIRED"})

        body = self.rfile.read(length)
        if len(body) != length:
            return self.reply(400, {"status": "INCOMPLETE_BODY"})
        actual_hash = hashlib.sha256(body).hexdigest()
        if not hmac.compare_digest(actual_hash, claimed_hash):
            return self.reply(400, {"status": "HASH_MISMATCH"})
        signed = (timestamp.encode() + b"\n" + nonce.encode() + b"\n" +
                  case_id.encode() + b"\n" + filename.encode() + b"\n" +
                  claimed_hash.encode() + b"\n" + body)
        expected = hmac.new(shared_secret(), signed, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            return self.reply(401, {"status": "SIGNATURE_INVALID"})

        now = time.time()
        with nonce_lock:
            for key, seen in list(nonces.items()):
                if now - seen > 2 * MAX_SKEW_SECONDS:
                    nonces.pop(key, None)
            if nonce in nonces:
                return self.reply(401, {"status": "REPLAY_DETECTED"})
            nonces[nonce] = now

        case_dir = ROOT / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        stored_name = f"{actual_hash}_{filename}"
        target = case_dir / stored_name
        temp = case_dir / f".{nonce}.part"
        try:
            with temp.open("xb") as stream:
                stream.write(body)
            os.replace(temp, target)
            record = {
                "received_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "case_id": case_id,
                "source_ip": self.client_address[0],
                "original_name": filename,
                "stored_name": stored_name,
                "size_bytes": len(body),
                "sha256": actual_hash,
                "handling": "STAGED_ONLY_NEVER_EXECUTED",
            }
            with manifest_lock, MANIFEST.open("a", encoding="utf-8") as log:
                log.write(json.dumps(record, separators=(",", ":")) + "\n")
        except Exception:
            temp.unlink(missing_ok=True)
            raise
        self.reply(201, {"status": "STAGED", "case_id": case_id,
                         "stored_name": stored_name, "sha256": actual_hash,
                         "size_bytes": len(body)})


def main():
    shared_secret()
    ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((BIND_HOST, PORT), Handler)
    print(f"Listening on http://{BIND_HOST}:{PORT}/evidence")
    print(f"Evidence folder: {ROOT}")
    print("Policy: host-only sender allowlist, HMAC, 50 MiB limit; files are staged and never executed.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()



