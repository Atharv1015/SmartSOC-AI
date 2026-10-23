import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for path in (str(SRC), str(SRC / "integration"), str(SRC / "integration_v2")):
    if path not in sys.path:
        sys.path.insert(0, path)

from integration import local_approval_console_v1 as approvals


class ReceiverApprovalBoundaryTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_db = os.environ.get("SMARTSOC_APPROVAL_DB")
        os.environ["SMARTSOC_APPROVAL_DB"] = str(
            Path(self.tmp.name) / "approval.sqlite3"
        )

        self.event = {
            "alert_id": "SYNTHETIC-AUTH-001",
            "rule_id": "92031",
            "mitigation": "PROCESS_TERMINATION",
            "process_name": "notepad.exe",
            "process_id": 4321,
        }

        self.client = Mock()
        self.client.execute.return_value = {
            "success": True,
            "verified": True,
            "status": "MOCKED_DRY_RUN",
        }

    def tearDown(self):
        if self.old_db is None:
            os.environ.pop("SMARTSOC_APPROVAL_DB", None)
        else:
            os.environ["SMARTSOC_APPROVAL_DB"] = self.old_db
        self.tmp.cleanup()

    def test_unapproved_event_cannot_reach_client(self):
        result = approvals.execute_authorized_client_call(
            self.client, self.event
        )

        self.assertEqual(result["status"], "AUTHORIZATION_BLOCKED")
        self.assertFalse(result["victim_client_invoked"])
        self.client.execute.assert_not_called()

    def test_approved_event_calls_mocked_client_once(self):
        pending = approvals.register_pending_from_reasoning_result(
            self.event, "SYNTHETIC-AUTH-001", "92031"
        )
        self.assertTrue(pending["registered"])

        decision = approvals.decide_request(
            pending["request_id"], "APPROVED", "synthetic-test-operator"
        )
        self.assertTrue(decision["ok"])

        result = approvals.execute_authorized_client_call(
            self.client, self.event
        )

        self.assertEqual(result["status"], "MOCKED_DRY_RUN")
        self.assertTrue(result["victim_client_invoked"])
        self.client.execute.assert_called_once_with(self.event)

        replay = approvals.execute_authorized_client_call(
            self.client, self.event
        )
        self.assertEqual(replay["status"], "AUTHORIZATION_BLOCKED")
        self.assertEqual(self.client.execute.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
