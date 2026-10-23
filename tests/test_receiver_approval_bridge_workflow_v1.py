import os
import sys
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
V2 = SRC / "integration_v2"

for directory in (str(SRC), str(V2)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from integration import smartsoc_live_receiver_v2 as receiver
from integration import local_approval_console_v1 as approvals
from integration_v2 import approval_bridge_v1 as bridge
from integration_v2 import authorized_dry_run_v2 as runner


class ReceiverApprovalBridgeWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="smartsoc-bridge-workflow-")
        self.db_path = Path(self.temp_dir) / "approval.sqlite3"

        self.old_db = os.environ.get("SMARTSOC_APPROVAL_DB")
        self.old_secret = os.environ.get("SMARTSOC_AGENT_SECRET")
        os.environ["SMARTSOC_APPROVAL_DB"] = str(self.db_path)
        os.environ["SMARTSOC_AGENT_SECRET"] = "S" * 64

        self.event = {
            "alert_id": "SYNTHETIC-BRIDGE-WORKFLOW-001",
            "rule": {"id": "92031", "level": 10},
            "data": {
                "win": {
                    "eventdata": {
                        "image": "notepad.exe",
                        "parentImage": r"C:\Windows\explorer.exe",
                        "commandLine": "notepad.exe",
                    }
                }
            },
        }

        self.inference = Mock()
        self.inference.infer.return_value = {
            "status": "INFER_OK",
            "confidence": 0.95,
            "technique": "T1087",
        }

        self.reasoning = Mock()
        self.reasoning.reason.return_value = {
            "mitigation": "PROCESS_TERMINATION",
            "process_name": "notepad.exe",
            "process_id": 4321,
            "technique": "T1087",
            "confidence": 0.95,
            "reason": "Synthetic bridge workflow test",
        }

        self.controller = Mock()

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

        print(f"\nWorkflow test database retained at: {self.db_path}")

    def submit_receiver_event(self):
        with (
            patch.object(receiver, "inference_engine", self.inference),
            patch.object(receiver, "reasoning_engine", self.reasoning),
            patch.object(receiver, "mitigation_controller", self.controller),
            patch.object(receiver, "MITIGATION_ENABLED", False),
        ):
            result = receiver.process_event(self.event)

        self.assertEqual(result["status"], "MITIGATION_BLOCKED", result)
        self.assertEqual(result["effective_action"], "NONE")
        self.assertFalse(result["victim_agent_called"])
        self.controller.execute.assert_not_called()
        self.mock_adapter.assert_not_called()
        return result

    def test_receiver_pending_approval_can_be_approved_and_bridged_once(self):
        self.submit_receiver_event()

        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                """
                SELECT request_id, alert_id, rule_id, action,
                       process_name, pid, status
                FROM requests
                WHERE alert_id = ?
                """,
                ("SYNTHETIC-BRIDGE-WORKFLOW-001",),
            ).fetchone()

        self.assertIsNotNone(row, "Receiver did not register an approval.")
        request_id, alert_id, rule_id, action, process_name, pid, status = row
        self.assertEqual(
            (alert_id, rule_id, action, process_name, pid, status),
            (
                "SYNTHETIC-BRIDGE-WORKFLOW-001",
                "92031",
                "PROCESS_TERMINATION",
                "notepad.exe",
                4321,
                "PENDING",
            ),
        )

        approved = approvals.decide_request(
            request_id, "APPROVED", "synthetic-test-operator"
        )
        self.assertTrue(approved["ok"], approved)

        result = bridge.execute_approved_request(request_id)
        self.assertEqual(result["status"], "DRY_RUN_VALIDATED", result)
        self.assertEqual(result["authorization_request_id"], request_id)
        self.assertTrue(result["authorization_consumed"])
        self.assertFalse(result["endpoint_modified"])

        # The adapter is mocked: no HTTP connection reaches an endpoint.
        self.mock_adapter.assert_called_once()

        replay = bridge.execute_approved_request(request_id)
        self.assertEqual(replay["status"], "AUTHORIZATION_BLOCKED", replay)
        self.assertEqual(self.mock_adapter.call_count, 1)

        # Receiver's legacy controller remains unused throughout.
        self.controller.execute.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
