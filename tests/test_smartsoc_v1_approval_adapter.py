import unittest
from unittest.mock import patch

from integration.smartsoc_v1_approval_adapter import (
    process_v1_result_for_approval,
)

class FakeReasoning:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def reason(self, inference_result, wazuh_event):
        self.calls.append((inference_result, wazuh_event))
        return self.result


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.event = {
            "alert_id": "adapter-test-001",
            "rule": {"id": "100034", "level": 8},
            "raw_wazuh_alert": {
                "id": "adapter-test-001",
                "rule": {"id": "100034", "level": 8},
                "data": {
                    "win": {
                        "eventdata": {
                            "image": "notepad.exe",
                            "processId": "4242",
                            "parentImage": "powershell.exe",
                            "commandLine": "notepad.exe",
                        }
                    }
                },
            },
        }
        self.inference = {
            "predictions": [
                {"technique": "T1059.001",
                 "confidence": 0.91, "threshold": 0.5}
            ]
        }

    def test_no_action_does_not_register(self):
        engine = FakeReasoning({"mitigation": "NONE"})
        with patch(
            "integration.smartsoc_v1_approval_adapter."
            "register_pending_from_reasoning_result"
        ) as register:
            result = process_v1_result_for_approval(
                self.event, self.inference, reasoning_engine=engine
            )
        self.assertEqual(result["status"], "NO_APPROVAL_REQUIRED")
        register.assert_not_called()
        self.assertIs(engine.calls[0][0], self.inference)
        self.assertIs(engine.calls[0][1], self.event["raw_wazuh_alert"])

    def test_missing_alert_id_blocks(self):
        event = {
            "rule": {"id": "100034"},
            "raw_wazuh_alert": {"rule": {"id": "100034"}},
        }
        engine = FakeReasoning({"mitigation": "PROCESS_TERMINATION"})
        with patch(
            "integration.smartsoc_v1_approval_adapter."
            "register_pending_from_reasoning_result"
        ) as register:
            result = process_v1_result_for_approval(
                event, self.inference, reasoning_engine=engine
            )
        self.assertEqual(result["status"], "BLOCKED")
        register.assert_not_called()
        self.assertEqual(engine.calls, [])

    def test_missing_process_identity_blocks(self):
        engine = FakeReasoning({
            "mitigation": "PROCESS_TERMINATION",
            "process_name": "notepad.exe",
            "process_id": None,
        })
        with patch(
            "integration.smartsoc_v1_approval_adapter."
            "register_pending_from_reasoning_result"
        ) as register:
            result = process_v1_result_for_approval(
                self.event, self.inference, reasoning_engine=engine
            )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["reason"], "INCOMPLETE_PROCESS_BINDING")
        register.assert_not_called()

    def test_eligible_request_registers_only(self):
        engine = FakeReasoning({
            "mitigation": "PROCESS_TERMINATION",
            "process_name": "notepad.exe",
            "process_id": 4242,
        })
        with patch(
            "integration.smartsoc_v1_approval_adapter."
            "register_pending_from_reasoning_result",
            return_value={
                "registered": True,
                "request_id": "isolated-test-request",
                "status": "PENDING",
            },
        ) as register:
            result = process_v1_result_for_approval(
                self.event, self.inference, reasoning_engine=engine
            )
        self.assertEqual(result["status"], "PENDING_APPROVAL")
        self.assertEqual(
            result["approval_request_id"], "isolated-test-request"
        )
        register.assert_called_once()
        self.assertFalse(result["automatic_execution"])
        self.assertFalse(result["automatic_containment"])
        self.assertFalse(result["endpoint_modified"])

    def test_process_name_mismatch_blocks(self):
        engine = FakeReasoning({
            "mitigation": "PROCESS_TERMINATION",
            "process_name": "powershell.exe",
            "process_id": 4242,
        })
        with patch(
            "integration.smartsoc_v1_approval_adapter."
            "register_pending_from_reasoning_result"
        ) as register:
            result = process_v1_result_for_approval(
                self.event, self.inference, reasoning_engine=engine
            )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["reason"], "PROCESS_NAME_MISMATCH")
        register.assert_not_called()

    def test_process_id_mismatch_blocks(self):
        engine = FakeReasoning({
            "mitigation": "PROCESS_TERMINATION",
            "process_name": "notepad.exe",
            "process_id": 9999,
        })
        with patch(
            "integration.smartsoc_v1_approval_adapter."
            "register_pending_from_reasoning_result"
        ) as register:
            result = process_v1_result_for_approval(
                self.event, self.inference, reasoning_engine=engine
            )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["reason"], "PROCESS_ID_MISMATCH")
        register.assert_not_called()
    def test_reasoning_exception_fails_closed(self):
        class BrokenReasoning:
            def reason(self, *_):
                raise RuntimeError("test exception")

        with patch(
            "integration.smartsoc_v1_approval_adapter."
            "register_pending_from_reasoning_result"
        ) as register:
            result = process_v1_result_for_approval(
                self.event, self.inference,
                reasoning_engine=BrokenReasoning()
            )
        self.assertEqual(result["status"], "INTEGRATION_ERROR")
        register.assert_not_called()
        self.assertFalse(result["endpoint_modified"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

