import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from autonomous import autonomous_mitigation_controller_v1 as controller


class ControllerExecutionBoundaryTests(unittest.TestCase):
    def setUp(self):
        # Avoid the controller initializer and all external dependencies.
        self.instance = object.__new__(
            controller.AutonomousMitigationController
        )
        self.client = Mock()
        self.instance.victim_client = self.client
        self.instance.audit_logger = Mock()

        self.reasoning_result = {
            "mitigation": "PROCESS_TERMINATION",
            "process_name": "synthetic-test.exe",
            "process_id": 4321,
            "reason": "Synthetic policy-boundary test",
        }

    def test_disabled_policy_blocks_before_client_call(self):
        with patch.object(
            controller.AutonomousMitigationController,
            "validate_execution_policy",
            side_effect=ValueError("Execution policy disabled"),
        ), patch.object(
            controller,
            "execute_authorized_client_call",
        ) as approval_wrapper:
            result = self.instance.execute(self.reasoning_result)

        self.assertEqual(result["status"], "MITIGATION_BLOCKED", result)
        self.assertFalse(result["success"], result)
        self.assertFalse(result["verified"], result)
        self.assertFalse(result["victim_client_invoked"], result)
        approval_wrapper.assert_not_called()
        self.client.execute.assert_not_called()

    def test_policy_rejection_prevents_approval_wrapper_call(self):
        with patch.object(
            controller.AutonomousMitigationController,
            "validate_execution_policy",
            side_effect=ValueError("Execution policy disabled"),
        ), patch.object(
            controller,
            "execute_authorized_client_call",
        ) as approval_wrapper:
            result = self.instance.execute(self.reasoning_result)

        self.assertEqual(result["status"], "MITIGATION_BLOCKED", result)
        self.assertFalse(result["success"], result)
        self.assertFalse(result["victim_client_invoked"], result)
        approval_wrapper.assert_not_called()
        self.client.execute.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)

