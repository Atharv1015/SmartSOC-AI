import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detection.brute_force_detector import (
    detect_brute_force,
    normalize_wazuh_auth_event,
)


BASE = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


def normalized_event(offset_seconds, event_id="4625",
                     source_ip="192.0.2.25", account="lab-user"):
    return {
        "event_id": event_id,
        "timestamp": BASE + timedelta(seconds=offset_seconds),
        "source_ip": source_ip,
        "target_account": account,
    }


def wazuh_event(offset_seconds, event_id="4625",
                source_ip="192.0.2.25", account="lab-user"):
    timestamp = BASE + timedelta(seconds=offset_seconds)
    return {
        "id": f"offline-{offset_seconds}-{account}",
        "timestamp": timestamp.isoformat(),
        "rule": {"id": "60122", "level": 5},
        "data": {
            "win": {
                "system": {
                    "eventID": event_id,
                    "systemTime": timestamp.isoformat(),
                },
                "eventdata": {
                    "targetUserName": account,
                    "ipAddress": source_ip,
                },
            }
        },
    }


class TestBruteForceDetector(unittest.TestCase):

    def test_five_failures_within_window_detected(self):
        result = detect_brute_force([
            normalized_event(i * 30) for i in range(5)
        ])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["failed_logons"], 5)
        self.assertEqual(result[0]["mitigation"], "NONE")

    def test_four_failures_not_detected(self):
        self.assertEqual(detect_brute_force([
            normalized_event(i * 30) for i in range(4)
        ]), [])

    def test_failures_outside_window_not_detected(self):
        self.assertEqual(detect_brute_force([
            normalized_event(i * 120) for i in range(5)
        ]), [])

    def test_successful_logons_not_counted(self):
        self.assertEqual(detect_brute_force([
            normalized_event(i * 30, event_id="4624")
            for i in range(6)
        ]), [])

    def test_different_accounts_are_separate(self):
        self.assertEqual(detect_brute_force([
            normalized_event(i * 30, account=f"user-{i}")
            for i in range(5)
        ]), [])

    def test_different_source_ips_are_separate(self):
        self.assertEqual(detect_brute_force([
            normalized_event(i * 30, source_ip=f"192.0.2.{i + 1}")
            for i in range(5)
        ]), [])

    def test_missing_fields_are_ignored(self):
        self.assertEqual(detect_brute_force([
            {"event_id": "4625", "timestamp": BASE},
            normalized_event(0),
        ]), [])

    def test_non_dictionary_input_is_ignored(self):
        self.assertEqual(detect_brute_force([None, "invalid"]), [])

    def test_detection_never_requests_mitigation(self):
        result = detect_brute_force([
            normalized_event(i * 30) for i in range(5)
        ])
        self.assertTrue(result)
        self.assertTrue(all(x["mitigation"] == "NONE" for x in result))

    def test_wazuh_4625_event_normalizes(self):
        raw = wazuh_event(0)
        result = normalize_wazuh_auth_event(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["event_id"], "4625")
        self.assertEqual(result["source_ip"], "192.0.2.25")
        self.assertEqual(result["target_account"], "lab-user")
        self.assertEqual(result["wazuh_rule_id"], "60122")

    def test_wazuh_successful_logon_is_not_normalized(self):
        self.assertIsNone(normalize_wazuh_auth_event(
            wazuh_event(0, event_id="4624")
        ))

    def test_missing_source_ip_is_rejected(self):
        raw = wazuh_event(0, source_ip="")
        self.assertIsNone(normalize_wazuh_auth_event(raw))

    def test_missing_account_is_rejected(self):
        raw = wazuh_event(0, account="")
        self.assertIsNone(normalize_wazuh_auth_event(raw))

    def test_invalid_timestamp_is_rejected(self):
        raw = wazuh_event(0)
        raw["data"]["win"]["system"]["systemTime"] = "not-a-timestamp"
        raw["timestamp"] = "also-invalid"
        self.assertIsNone(normalize_wazuh_auth_event(raw))

    def test_wazuh_shaped_events_reach_detector(self):
        alerts = [wazuh_event(i * 30) for i in range(5)]
        normalized = [
            normalize_wazuh_auth_event(alert) for alert in alerts
        ]
        self.assertTrue(all(item is not None for item in normalized))
        findings = detect_brute_force(normalized)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["classification"],
                         "BRUTE_FORCE_SUSPECTED")
        self.assertEqual(findings[0]["mitigation"], "NONE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
