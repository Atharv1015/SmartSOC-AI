import http.client
import json
import secrets
import time

HOST = "192.168.56.102"
PORT = 8765
BODY = json.dumps({
    "action": "PROCESS_TERMINATION",
    "target": {"pid": 2147483647, "process_name": "notepad.exe"},
    "context": {"requested_mode": "AUTH_NEGATIVE_TEST"}
}, separators=(",", ":")).encode("utf-8")

def run_test(name, timestamp):
    nonce = secrets.token_hex(16)
    headers = {
        "Content-Type": "application/json",
        "X-SmartSOC-Timestamp": str(timestamp),
        "X-SmartSOC-Nonce": nonce,
        "X-SmartSOC-Signature": "0" * 64,
    }
    conn = http.client.HTTPConnection(HOST, PORT, timeout=5)
    try:
        conn.request("POST", "/mitigate", body=BODY, headers=headers)
        response = conn.getresponse()
        raw = response.read().decode("utf-8", errors="replace")
        print(f"\nTEST: {name}")
        print(f"HTTP_STATUS: {response.status}")
        print(f"RESPONSE: {raw or '<empty>'}")
    except OSError as exc:
        print(f"\nTEST: {name}\nREQUEST_FAILED: {exc}")
    finally:
        conn.close()

if __name__ == "__main__":
    now = int(time.time())
    print("HOST_UNIX_TIME:", now)
    run_test("FRESH_TIMESTAMP_INVALID_SIGNATURE", now)
    run_test("STALE_TIMESTAMP", now - 300)
