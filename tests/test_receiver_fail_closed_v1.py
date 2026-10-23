
import os
import sys
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from integration import smartsoc_live_receiver_v2 as receiver


class ReceiverFailClosedTests(unittest.TestCase):

    def setUp(self):
        # Retain isolated test databases to avoid Windows SQLite
        # file-lock cleanup failures during diagnosis.
        self.test_dir = Path(
            tempfile.mkdtemp(prefix="smartsoc-receiver-test-")
        )
        self.test_db = self.test_dir / "receiver-test.sqlite3"

        self.old_db = os.environ.get("SMARTSOC_APPROVAL_DB")
        os.environ["SMARTSOC_APPROVAL_DB"] = str(self.test_db)

        self.inference = Mock()
        self.inference.infer.return_value = {
            "status": "INFER_OK",
            "confidence": 0.95,
            "technique": "T1087",
        }

        self.reasoning = Mock()
        self.controller = Mock()

        self.reasoning_result = {
            "mitigation": "PROCESS_TERMINATION",
            "process_name": "synthetic-test.exe",
            "process_id": 4321,
            "technique": "T1087",
            "confidence": 0.95,
            "reason": "Synthetic receiver integration test",
        }

        self.event = {
            "alert_id": "SYNTHETIC-RECEIVER-001",
            "rule": {
                "id": "92031",
                "level": 10,
            },
            "data": {
                "win": {
                    "eventdata": {
                        "image": "synthetic-test.exe",
                        "parentImage": r"C:\Windows\explorer.exe",
                        "commandLine": "synthetic-test.exe",
                    }
                }
            },
        }

    def tearDown(self):
        if self.old_db is None:
            os.environ.pop("SMARTSOC_APPROVAL_DB", None)
        else:
            os.environ["SMARTSOC_APPROVAL_DB"] = self.old_db

        # Intentionally retain the isolated DB for inspection.
        print(f"\nReceiver test DB retained at: {self.test_db}")

    def run_event(self, mitigation):
        result = dict(
            self.reasoning_result,
            mitigation=mitigation,
        )
        self.reasoning.reason.return_value = result

        with (
            patch.object(receiver, "inference_engine", self.inference),
            patch.object(receiver, "reasoning_engine", self.reasoning),
            patch.object(
                receiver,
                "mitigation_controller",
                self.controller,
            ),
            patch.object(receiver, "MITIGATION_ENABLED", False),
        ):
            return receiver.process_event(self.event)

    def test_disabled_receiver_registers_pending_and_blocks_execution(self):
        result = self.run_event("PROCESS_TERMINATION")

        # Receiver must fail closed.
        self.assertEqual(result["status"], "MITIGATION_BLOCKED", result)
        self.assertEqual(result["effective_action"], "NONE")
        self.assertFalse(result["mitigation_enabled"])
        self.assertFalse(result["victim_agent_called"])

        # Neither the controller nor endpoint client may be invoked.
        self.controller.execute.assert_not_called()

        # Verify that the pending approval record was registered
        # in this test's isolated SQLite database.
        self.assertTrue(
            self.test_db.exists(),
            f"Expected test database was not created: {self.test_db}",
        )

        with sqlite3.connect(str(self.test_db), timeout=5) as conn:
            row = conn.execute(
                """
                SELECT
                    alert_id,
                    rule_id,
                    action,
                    process_name,
                    pid,
                    status
                FROM requests
                WHERE alert_id = ?
                """,
                ("SYNTHETIC-RECEIVER-001",),
            ).fetchone()

        self.assertEqual(
            row,
            (
                "SYNTHETIC-RECEIVER-001",
                "92031",
                "PROCESS_TERMINATION",
                "synthetic-test.exe",
                4321,
                "PENDING",
            ),
            f"Unexpected approval record: {row}",
        )

    def test_no_mitigation_returns_without_controller_call(self):
        result = self.run_event("NONE")

        self.assertEqual(result["status"], "REASONING_COMPLETE", result)
        self.assertEqual(result["effective_action"], "NONE")
        self.assertFalse(result["victim_agent_called"])
        self.controller.execute.assert_not_called()

        # A NONE decision must not create a pending termination request.
        if self.test_db.exists():
            with sqlite3.connect(str(self.test_db), timeout=5) as conn:
                rows = conn.execute(
                    """
                    SELECT request_id
                    FROM requests
                    WHERE alert_id = ?
                    """,
                    ("SYNTHETIC-RECEIVER-001",),
                ).fetchall()

            self.assertEqual(rows, [])

    if __name__ == "__main__":
        unittest.main(verbosity=2)
