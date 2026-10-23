import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src" / "integration"),
)

from authorization_validator_v2 import validate_approval


NOW = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)


def valid_record():
    return {
        "decision": "APPROVED",
        "analyst_authorized": True,
        "alert_id": "TEST-ALERT-001",
        "rule_id": "92031",
        "authorization_actor": "synthetic-test-analyst",
        "authorization_timestamp": (
            NOW - timedelta(seconds=30)
        ).isoformat(),
    }


class AuthorizationValidatorTests(unittest.TestCase):
    def assert_denied(self, approval, alert="TEST-ALERT-001", rule="92031"):
        result = validate_approval(approval, alert, rule, now=NOW)
        self.assertIs(result["authorized"], False, result)
        self.assertEqual(result["status"], "DENIED", result)

    def test_valid_approval(self):
        result = validate_approval(
            valid_record(), "TEST-ALERT-001", "92031", now=NOW
        )
        self.assertIs(result["authorized"], True, result)

    def test_missing_record(self):
        self.assert_denied(None)

    def test_denied_decision(self):
        record = valid_record()
        record["decision"] = "DENY"
        self.assert_denied(record)

    def test_string_true_rejected(self):
        record = valid_record()
        record["analyst_authorized"] = "true"
        self.assert_denied(record)

    def test_boolean_false_rejected(self):
        record = valid_record()
        record["analyst_authorized"] = False
        self.assert_denied(record)

    def test_alert_mismatch(self):
        self.assert_denied(valid_record(), alert="OTHER-ALERT")

    def test_rule_mismatch(self):
        self.assert_denied(valid_record(), rule="99999")

    def test_expired_approval(self):
        record = valid_record()
        record["authorization_timestamp"] = (
            NOW - timedelta(seconds=301)
        ).isoformat()
        self.assert_denied(record)

    def test_future_approval(self):
        record = valid_record()
        record["authorization_timestamp"] = (
            NOW + timedelta(seconds=1)
        ).isoformat()
        self.assert_denied(record)

    def test_invalid_timestamp(self):
        record = valid_record()
        record["authorization_timestamp"] = "not-a-timestamp"
        self.assert_denied(record)

    def test_timezone_required(self):
        record = valid_record()
        record["authorization_timestamp"] = "2026-10-11T11:59:30"
        self.assert_denied(record)

    def test_actor_required(self):
        record = valid_record()
        record["authorization_actor"] = ""
        self.assert_denied(record)

    def test_alert_id_required(self):
        self.assert_denied(valid_record(), alert="")

    def test_integer_one_rejected(self):
        record = valid_record()
        record["analyst_authorized"] = 1
        self.assert_denied(record)


if __name__ == "__main__":
    unittest.main(verbosity=2)
