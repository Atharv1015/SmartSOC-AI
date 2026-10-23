import json
import os
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT / "reports" / "integration"
AUDIT_FILE = REPORT_DIR / "smartsoc_receiver_v1_audit.jsonl"
HOST = "192.168.56.1"
PORT = 8788
MAX_BODY_BYTES = 1024 * 1024

sys.path.insert(0, str(PROJECT / "src"))
import production_inference_windows_apt_v4_1_aligned as inference
from integration.smartsoc_v1_approval_bridge import process_inference_response

REPORT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS = inference.load_artifacts()
inference.validate_artifacts(ARTIFACTS)
APPROVAL_HOOK_ENABLED = os.getenv('SMARTSOC_V1_APPROVAL_ENABLED', '0').strip() == '1'

def audit(record):
    record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with AUDIT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, data):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {
                "status": "ok",
                "inference_loaded": True,
                "execution_policy": "LOG_ONLY",
                "automatic_execution": False,
                "automatic_containment": False
            })
        else:
            self.send_json(404, {"error": "not_found"})

    def do_POST(self):
        if self.path != "/wazuh-event":
            self.send_json(404, {"error": "not_found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                self.send_json(413, {"error": "invalid_or_oversized_body"})
                return

            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                self.send_json(400, {"error": "JSON object required"})
                return

            event = payload.get("event", payload)
            if not isinstance(event, dict):
                self.send_json(400, {"error": "event must be an object"})
                return

            normalized_event = inference.normalize_wazuh_event(event)
            features = inference.build_feature_matrix(normalized_event, ARTIFACTS)
            prediction = inference.predict(features, ARTIFACTS)
            result = inference.build_result_json(
                normalized_event, features, prediction, ARTIFACTS
            )

            agent = event.get("agent", {})
            if not isinstance(agent, dict):
                agent = {}

            response = {
                "status": "INFERENCE_OK",
                "agent_name": agent.get("name", "unknown"),
                "inference_result": result,
                "execution_policy": "LOG_ONLY",
                "automatic_execution": False,
                "automatic_containment": False,
                "endpoint_modified": False
            }

            rule = event.get("rule", {})
            if not isinstance(rule, dict):
                rule = {}

            response["alert_id"] = event.get("id") or event.get("alert_id")
            response["rule_id"] = rule.get("id") or event.get("rule_id")

            if APPROVAL_HOOK_ENABLED:
                approval_result = process_inference_response(
                    event,
                    {
                        "status": "INFERENCE_OK",
                        "inference_result": result
                    }
                )
                response["approval"] = approval_result
            audit({
                "status": "INFERENCE_OK",
                "agent_name": agent.get("name", "unknown"),
                "rule_id": rule.get("id") if isinstance(rule, dict) else None,
                "execution_policy": "LOG_ONLY",
                "automatic_execution": False,
                "automatic_containment": False,
                "endpoint_modified": False,
                "alert_id": response.get("alert_id"),
                "approval": response.get("approval", {"status": "DISABLED"}),
                "result": result
            })
            self.send_json(200, response)

        except Exception as exc:
            audit({
                "status": "INFERENCE_ERROR",
                "error": str(exc),
                "automatic_execution": False,
                "automatic_containment": False,
                "endpoint_modified": False
            })
            self.send_json(500, {
                "status": "INFERENCE_ERROR",
                "error": str(exc),
                "automatic_execution": False,
                "automatic_containment": False
            })

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))

if __name__ == "__main__":
    print("SMARTSOC_RECEIVER_V1_STARTING")
    print("Listening on %s:%s" % (HOST, PORT))
    print("Policy: LOG_ONLY; no mitigation or containment")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


