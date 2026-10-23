
import os
import sys
import sqlite3
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from integration import local_approval_console_v1 as approval


class ApprovalRegistrationIsolationTests(unittest.TestCase):

    def setUp(self):
        # Use a unique temporary directory for each test.
        # Retain it after the test so Windows file-lock issues can be diagnosed.
        self.test_dir = Path(
            tempfile.mkdtemp(prefix="smartsoc-approval-test-")
        )
        self.test_db = self.test_dir / "approval-test.sqlite3"

        self.old_db = os.environ.get("SMARTSOC_APPROVAL_DB")
        os.environ["SMARTSOC_APPROVAL_DB"] = str(self.test_db)

    def tearDown(self):
        # Restore the environment variable.
        if self.old_db is None:
            os.environ.pop("SMARTSOC_APPROVAL_DB", None)
        else:
            os.environ["SMARTSOC_APPROVAL_DB"] = self.old_db

        # Intentionally retain the isolated database for diagnosis.
        # Do not delete files while Windows reports them as locked.
        print(
            f"\nTemporary test database retained at: {self.test_db}"
        )

    def test_registers_pending_request_in_temporary_database(self):
        result = approval.register_pending_from_reasoning_result(
            {
                "mitigation": "PROCESS_TERMINATION",
                "process_name": "synthetic-test.exe",
                "process_id": 4321,
            },
            alert_id="SYNTHETIC-DB-TEST-001",
            rule_id="92031",
        )

        self.assertTrue(result["registered"], result)
        self.assertEqual(result["status"], "PENDING")
        self.assertTrue(self.test_db.exists())

        # The inspection connection is closed before the test finishes.
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
                WHERE request_id = ?
                """,
                (result["request_id"],),
            ).fetchone()

        self.assertEqual(
            row,
            (
                "SYNTHETIC-DB-TEST-001",
                "92031",
                "PROCESS_TERMINATION",
                "synthetic-test.exe",
                4321,
                "PENDING",
            ),
        )

    def test_incomplete_candidate_is_not_registered(self):
        result = approval.register_pending_from_reasoning_result(
            {"mitigation": "NONE"},
            alert_id="SYNTHETIC-DB-TEST-002",
            rule_id="92031",
        )

        self.assertFalse(result["registered"])
        self.assertEqual(
            result["reason"],
            "INCOMPLETE_OR_UNSUPPORTED_BINDING",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
