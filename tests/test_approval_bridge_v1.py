import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
V2 = SRC / "integration_v2"

for directory in (str(SRC), str(V2)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from integration import local_approval_console_v1 as approvals
from integration_v2 import approval_bridge_v1 as bridge
from integration_v2 import authorized_dry_run_v2 as runner


class ApprovalBridgeV1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_db = os.environ.get("SMARTSOC_APPROVAL_DB")
        self.old_secret = os.environ.get("SMARTSOC_AGENT_SECRET")

        os.environ["SMARTSOC_APPROVAL_DB"] = str(
            Path(self.tmp.name) / "approval.sqlite3"
        )
        os.environ["SMARTSOC_AGENT_SECRET"] = "S" * 64

        self.event = {
            "alert_id": "BRIDGE-SYNTHETIC-001",
            "rule_id": "92031",
            "mitigation": "PROCESS_TERMINATION",
            "process_name": "notepad.exe",
            "process_id": 4321,
        }

        self.adapter_patch = patch.object(
            runner.adapter,
            "execute_dry_run",
            return_value={
                "success": True,
                "verified": True,
                "status": "DRY_RUN_VALIDATED",
                "victim_request_attempted": True,
                "victim_http_response_received": True,
            },
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

    def create_request(self, decision="APPROVED"):
        pending = approvals.register_pending_from_reasoning_result(
            self.event, self.event["alert_id"], self.event["rule_id"]
        )
        self.assertTrue(pending["registered"])

        decided = approvals.decide_request(
            pending["request_id"], decision, "synthetic-operator"
        )
        self.assertTrue(decided["ok"])
        return pending["request_id"]

    def test_pending_request_is_blocked(self):
        pending = approvals.register_pending_from_reasoning_result(
            self.event, self.event["alert_id"], self.event["rule_id"]
        )
        result = bridge.execute_approved_request(pending["request_id"])

        self.assertEqual(result["status"], "AUTHORIZATION_BLOCKED")
        self.mock_adapter.assert_not_called()

    def test_denied_request_is_blocked(self):
        request_id = self.create_request("DENIED")
        result = bridge.execute_approved_request(request_id)

        self.assertEqual(result["status"], "AUTHORIZATION_BLOCKED")
        self.mock_adapter.assert_not_called()

    def test_unknown_request_is_blocked(self):
        result = bridge.execute_approved_request("unknown-request-id")

        self.assertEqual(result["status"], "AUTHORIZATION_BLOCKED")
        self.mock_adapter.assert_not_called()

    def test_approved_request_uses_mocked_adapter_once(self):
        request_id = self.create_request()
        result = bridge.execute_approved_request(request_id)

        self.assertEqual(result["status"], "DRY_RUN_VALIDATED")
        self.assertEqual(result["authorization_request_id"], request_id)
        self.assertEqual(result["selected_approval_request_id"], request_id)
        self.assertTrue(result["authorization_consumed"])
        self.assertFalse(result["endpoint_modified"])
        self.mock_adapter.assert_called_once()

    def test_replay_is_blocked(self):
        request_id = self.create_request()

        first = bridge.execute_approved_request(request_id)
        second = bridge.execute_approved_request(request_id)

        self.assertEqual(first["status"], "DRY_RUN_VALIDATED")
        self.assertEqual(second["status"], "AUTHORIZATION_BLOCKED")
        self.assertEqual(self.mock_adapter.call_count, 1)

    def test_selected_request_id_is_consumed_not_another_matching_approval(self):
        first_id = self.create_request()

        # A second request with the same binding is now possible because the
        # first request is already APPROVED, not PENDING.
        second_id = self.create_request()
        self.assertNotEqual(first_id, second_id)

        result = bridge.execute_approved_request(first_id)
        self.assertEqual(result["authorization_request_id"], first_id)

        with approvals._db() as connection:
            first = connection.execute(
                "SELECT status FROM requests WHERE request_id=?", (first_id,)
            ).fetchone()
            second = connection.execute(
                "SELECT status FROM requests WHERE request_id=?", (second_id,)
            ).fetchone()

        self.assertEqual(first["status"], "CONSUMED")
        self.assertEqual(second["status"], "APPROVED")
        self.mock_adapter.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
