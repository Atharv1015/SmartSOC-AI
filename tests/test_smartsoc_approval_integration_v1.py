import unittest
from unittest.mock import Mock, patch

from integration import smartsoc_approval_integration_v1 as integration


class SmartSOCApprovalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.event = {
            "alert_id": "test-alert-001",
            "rule": {"id": "92031"},
        }
        self.wazuh_event = {
            "id": "test-alert-001",
            "rule": {"id": "92031"},
        }
        self.inference = Mock()
        self.inference.infer.return_value = {"prediction": "test"}
        self.reasoning = Mock()
        self.reasoning.reason.return_value = {
            "mitigation": "PROCESS_TERMINATION",
            "process_name": "notepad.exe",
            "process_id": 1234,
        }

    @patch.object(integration, "normalized_to_wazuh_event")
    @patch.object(integration, "register_pending_from_reasoning_result")
    def test_valid_request_registers_pending_approval(self, register, normalize):
        normalize.return_value = self.wazuh_event
        register.return_value = {
            "registered": True,
            "request_id": "mock-request-001",
            "status": "PENDING",
        }

        result = integration.process_for_approval(
            self.event,
            inference_engine=self.inference,
            reasoning_engine=self.reasoning,
        )

        self.assertEqual(result["status"], "PENDING_APPROVAL")
        self.assertEqual(result["approval_request_id"], "mock-request-001")
        self.assertEqual(result["execution_policy"], "LOG_ONLY")
        self.assertFalse(result["automatic_execution"])
        self.assertFalse(result["automatic_containment"])
        self.assertFalse(result["endpoint_modified"])
        register.assert_called_once()

        bound = register.call_args.args[0]
        self.assertEqual(bound["alert_id"], "test-alert-001")
        self.assertEqual(bound["rule_id"], "92031")
        self.assertEqual(bound["process_name"], "notepad.exe")
        self.assertEqual(bound["pid"], 1234)

    @patch.object(integration, "normalized_to_wazuh_event")
    @patch.object(integration, "register_pending_from_reasoning_result")
    def test_missing_pid_fails_closed(self, register, normalize):
        normalize.return_value = self.wazuh_event
        self.reasoning.reason.return_value = {
            "mitigation": "PROCESS_TERMINATION",
            "process_name": "notepad.exe",
            "process_id": None,
        }

        result = integration.process_for_approval(
            self.event,
            inference_engine=self.inference,
            reasoning_engine=self.reasoning,
        )

        self.assertEqual(result["status"], "BLOCKED")
        self.assertFalse(result["endpoint_modified"])
        register.assert_not_called()

    @patch.object(integration, "normalized_to_wazuh_event")
    @patch.object(integration, "register_pending_from_reasoning_result")
    def test_unsupported_action_never_registers(self, register, normalize):
        normalize.return_value = self.wazuh_event
        self.reasoning.reason.return_value = {
            "mitigation": "FILE_QUARANTINE",
            "process_name": "notepad.exe",
            "process_id": 1234,
        }

        result = integration.process_for_approval(
            self.event,
            inference_engine=self.inference,
            reasoning_engine=self.reasoning,
        )

        self.assertEqual(result["status"], "NO_APPROVAL_REQUIRED")
        register.assert_not_called()

    @patch.object(integration, "normalized_to_wazuh_event")
    @patch.object(integration, "register_pending_from_reasoning_result")
    def test_missing_alert_id_fails_closed(self, register, normalize):
        normalize.return_value = {"rule": {"id": "92031"}}

        result = integration.process_for_approval(
            {"rule": {"id": "92031"}},
            inference_engine=self.inference,
            reasoning_engine=self.reasoning,
        )

        self.assertEqual(result["status"], "BLOCKED")
        register.assert_not_called()

    @patch.object(integration, "normalized_to_wazuh_event")
    @patch.object(integration, "register_pending_from_reasoning_result")
    def test_registration_failure_fails_closed(self, register, normalize):
        normalize.return_value = self.wazuh_event
        register.return_value = {
            "registered": False,
            "reason": "INCOMPLETE_OR_UNSUPPORTED_BINDING",
        }

        result = integration.process_for_approval(
            self.event,
            inference_engine=self.inference,
            reasoning_engine=self.reasoning,
        )

        self.assertEqual(result["status"], "BLOCKED")
        self.assertFalse(result["endpoint_modified"])

    def test_non_object_event_is_rejected(self):
        result = integration.process_for_approval(
            ["not", "an", "event"],
            inference_engine=self.inference,
            reasoning_engine=self.reasoning,
        )

        self.assertEqual(result["status"], "REJECTED")
        self.assertFalse(result["endpoint_modified"])

    @patch.object(integration, "normalized_to_wazuh_event")
    def test_inference_exception_fails_closed(self, normalize):
        normalize.return_value = self.wazuh_event
        self.inference.infer.side_effect = RuntimeError("mock failure")

        result = integration.process_for_approval(
            self.event,
            inference_engine=self.inference,
            reasoning_engine=self.reasoning,
        )

        self.assertEqual(result["status"], "INTEGRATION_ERROR")
        self.assertFalse(result["automatic_execution"])
        self.assertFalse(result["automatic_containment"])
        self.assertFalse(result["endpoint_modified"])
        self.reasoning.reason.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
