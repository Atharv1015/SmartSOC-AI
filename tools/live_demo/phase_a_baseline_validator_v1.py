import csv
import json
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "integration"
AUDIT_FILE = REPORT_DIR / "smartsoc_receiver_v1_audit.jsonl"
WINDOW = timedelta(hours=24)

def get_json(url):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))

def check(name, passed, detail):
    return {
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "detail": str(detail),
    }

def main():
    now = datetime.now(timezone.utc)
    results = []
    receiver = None
    victim = None

    try:
        receiver = get_json("http://192.168.56.1:8788/health")
        results.append(check(
            "receiver_healthy",
            receiver.get("status") == "ok",
            receiver.get("status"),
        ))
        results.append(check(
            "inference_loaded",
            receiver.get("inference_loaded") is True,
            receiver.get("inference_loaded"),
        ))
        results.append(check(
            "receiver_log_only",
            receiver.get("execution_policy") == "LOG_ONLY",
            receiver.get("execution_policy"),
        ))
        results.append(check(
            "receiver_automatic_execution_disabled",
            receiver.get("automatic_execution") is False,
            receiver.get("automatic_execution"),
        ))
        results.append(check(
            "receiver_automatic_containment_disabled",
            receiver.get("automatic_containment") is False,
            receiver.get("automatic_containment"),
        ))
    except Exception as exc:
        results.append(check("receiver_health_reachable", False, repr(exc)))

    try:
        victim = get_json("http://192.168.56.102:8765/health")
        results.append(check(
            "victim_healthy",
            victim.get("status") == "HEALTHY",
            victim.get("status"),
        ))
        results.append(check(
            "victim_execution_disabled",
            victim.get("execution_enabled") is False,
            victim.get("execution_enabled"),
        ))
        results.append(check(
            "victim_arbitrary_commands_disabled",
            victim.get("arbitrary_commands") is False,
            victim.get("arbitrary_commands"),
        ))
        results.append(check(
            "victim_process_allowlist",
            victim.get("allowed_processes") == ["notepad.exe"],
            victim.get("allowed_processes"),
        ))
    except Exception as exc:
        results.append(check("victim_health_reachable", False, repr(exc)))

    recent = []
    malformed = 0
    if AUDIT_FILE.exists():
        for line in AUDIT_FILE.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                stamp = datetime.fromisoformat(
                    record["timestamp_utc"].replace("Z", "+00:00")
                )
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
                if now - WINDOW <= stamp.astimezone(timezone.utc) <= now:
                    recent.append(record)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                malformed += 1

    valid_events = [
        r for r in recent
        if r.get("status") == "INFERENCE_OK"
        and r.get("execution_policy") == "LOG_ONLY"
        and r.get("automatic_execution") is False
        and r.get("automatic_containment") is False
        and r.get("endpoint_modified") is False
    ]

    rule_ids = {
        str(r.get("rule_id"))
        for r in valid_events
    }
    results.append(check(
        "recent_safe_inference_evidence",
        len(valid_events) >= 1,
        f"{len(valid_events)} qualifying records in last 24h",
    ))
    results.append(check(
        "expected_test_rules_seen",
        {"92031", "92033"}.issubset(rule_ids),
        f"Observed rule IDs: {sorted(rule_ids)}",
    ))

    schema_ok = any(
        r.get("result", {}).get("model_version") == "v4.1"
        and r.get("result", {}).get("feature_count") == 6919
        and r.get("result", {}).get("mitre_technique_count") == 45
        for r in valid_events
    )
    results.append(check(
        "production_inference_schema",
        schema_ok,
        "Requires model v4.1, 6919 features, and 45 MITRE techniques",
    ))

    results.append(check(
        "receiver_audit_present",
        AUDIT_FILE.exists(),
        str(AUDIT_FILE),
    ))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    report = {
        "report_type": "PHASE_A_BASELINE_ONLY",
        "generated_at_utc": now.isoformat(),
        "scope": "Read-only health and recent receiver audit checks",
        "mitigation_requests_sent": 0,
        "actual_mitigation_tested": False,
        "checks_passed": sum(r["status"] == "PASS" for r in results),
        "checks_failed": sum(r["status"] == "FAIL" for r in results),
        "checks": results,
        "recent_qualifying_inference_records": len(valid_events),
        "malformed_or_unparseable_audit_lines": malformed,
        "receiver_health": receiver,
        "victim_health": victim,
        "limitations": [
            "This is not the complete ten-scenario Phase A suite.",
            "This script does not verify HMAC/replay rejection scenarios.",
            "This script does not inspect the victim audit file.",
            "A PASS here is not proof of successful real mitigation.",
        ],
    }

    json_path = REPORT_DIR / f"phase_a_baseline_{stamp}.json"
    csv_path = REPORT_DIR / f"phase_a_baseline_{stamp}.csv"

    json_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        writer.writerows(results)

    print("PHASE A BASELINE REPORT")
    print("Overall:", "PASS" if report["checks_failed"] == 0 else "FAIL")
    print(f"Checks passed: {report['checks_passed']}")
    print(f"Checks failed: {report['checks_failed']}")
    print("JSON:", json_path)
    print("CSV :", csv_path)
    for item in results:
        print(f'{item["status"]:4} | {item["check"]} | {item["detail"]}')

    return 0 if report["checks_failed"] == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
