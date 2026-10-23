import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
INTEGRATION_V2 = SRC / "integration_v2"

for directory in (str(SRC), str(INTEGRATION_V2)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from integration import local_approval_console_v1 as approvals
from integration_v2 import authorized_dry_run_v2 as runner
import execution_config_v2 as policy


class AuthorizedDryRunV2Tests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_db = os.environ.get("SMARTSOC_APPROVAL_DB")
        self.old_secret = os.environ.get("SMARTSOC_AGENT_SECRET")
        os.environ["SMARTSOC_APPROVAL_DB"] = str(
            Path(self.tmp.name) / "approval.sqlite3"
        )
        os.environ["SMARTSOC_AGENT_SECRET"] = "S" * 64

        self.event = {
            "alert_id": "SYNTHETIC-V2-AUTH-001",
            "rule_id": "92031",
            "mitigation": "PROCESS_TERMINATION",
            "process_name": "notepad.exe",
            "process_id": 4321,
            "technique": "T1087",
            "reason": "Synthetic authorization integration test",
        }

        self.adapter_result = {
            "success": True,
            "verified": True,
            "status": "DRY_RUN_VALIDATED",
            "victim_request_attempted": True,
            "victim_http_response_received": True,
        }

        self.adapter_patch = patch.object(
            runner.adapter,
            "execute_dry_run",
            return_value=dict(self.adapter_result),
        )
        self.mock_adapter = self.adapter_patch.start()
        self.addCleanup(self.adapter_patch.stop)

    def tearDown(self):
        if self.old_db is None:
            os.environ.pop("SMARTSOC_APPROVAL_DB", None)
        else:
            os.environ["SMARTSOC_APPROVAL_DB"] = self.old_db

        if self.old_secret is None:
            os.environ.pop("SMARTSOC_AGENT_SECRET", None)
        else:
            os.environ["SMARTSOC_AGENT_SECRET"] = self.old_secret

        self.tmp.cleanup()

    def approve_event(self, event=None):
        event = event or self.event
        pending = approvals.register_pending_from_reasoning_result(
            event, event["alert_id"], event["rule_id"]
        )
        self.assertTrue(pending["registered"])
        decision = approvals.decide_request(
            pending["request_id"], "APPROVED", "synthetic-test-operator"
        )
        self.assertTrue(decision["ok"])
        return pending["request_id"]

    def test_unapproved_event_never_reaches_adapter(self):
        result = runner.execute_authorized_dry_run(self.event)
        self.assertEqual(result["status"], "AUTHORIZATION_BLOCKED")
        self.assertFalse(result["victim_request_attempted"])
        self.mock_adapter.assert_not_called()

    def test_approved_event_invokes_adapter_once(self):
        request_id = self.approve_event()
        result = runner.execute_authorized_dry_run(self.event)

        self.assertEqual(result["status"], "DRY_RUN_VALIDATED")
        self.assertTrue(result["authorization_consumed"])
        self.assertEqual(result["authorization_request_id"], request_id)
        self.assertEqual(result["authorization_actor"], "synthetic-test-operator")
        self.assertFalse(result["endpoint_modified"])
        self.mock_adapter.assert_called_once()

    def test_replay_is_blocked_after_approval_consumed(self):
        self.approve_event()

        first = runner.execute_authorized_dry_run(self.event)
        second = runner.execute_authorized_dry_run(self.event)

        self.assertEqual(first["status"], "DRY_RUN_VALIDATED")
        self.assertEqual(second["status"], "AUTHORIZATION_BLOCKED")
        self.assertEqual(self.mock_adapter.call_count, 1)

    def test_denied_event_never_reaches_adapter(self):
        pending = approvals.register_pending_from_reasoning_result(
            self.event, self.event["alert_id"], self.event["rule_id"]
        )
        self.assertTrue(pending["registered"])
        decision = approvals.decide_request(
            pending["request_id"], "DENIED", "synthetic-test-operator"
        )
        self.assertTrue(decision["ok"])

        result = runner.execute_authorized_dry_run(self.event)
        self.assertEqual(result["status"], "AUTHORIZATION_BLOCKED")
        self.mock_adapter.assert_not_called()

    def test_unsafe_policy_never_consumes_approval_or_calls_adapter(self):
        self.approve_event()

        with patch.object(policy, "DRY_RUN", False):
            result = runner.execute_authorized_dry_run(self.event)

        self.assertEqual(result["status"], "POLICY_BLOCKED")
        self.mock_adapter.assert_not_called()

        # The approval remains available because policy validation failed first.
        self.assertEqual(
            approvals.consume_approval(self.event)["authorized"],
            True,
        )

    def test_disallowed_process_is_rejected_before_approval_consumption(self):
        event = dict(self.event, process_name="powershell.exe")
        self.approve_event(event)

        result = runner.execute_authorized_dry_run(event)
        self.assertEqual(result["status"], "REQUEST_BLOCKED")
        self.mock_adapter.assert_not_called()

        # This check confirms preflight rejection did not consume approval.
        self.assertTrue(approvals.consume_approval(event)["authorized"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
