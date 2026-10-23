import unittest
from unittest.mock import patch

from integration.smartsoc_v1_approval_bridge import (
    process_inference_response,
)


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.original_alert = {
            "id": "BRIDGE-SYNTHETIC-001",
            "rule": {"id": "100001"},
            "data": {"win": {"eventdata": {
                "image": "notepad.exe",
                "processId": "4242",
            }}},
        }
        self.receiver_response = {
            "status": "INFERENCE_OK",
            "inference_result": {
                "predictions": [
                    {
                        "technique": "T1059",
                        "confidence": 0.99,
                        "threshold": 0.50,
                    }
                ]
            },
            "execution_policy": "LOG_ONLY",
            "automatic_execution": False,
            "automatic_containment": False,
            "endpoint_modified": False,
        }

    def test_passes_original_alert_and_existing_inference(self):
        expected = {
            "status": "PENDING_APPROVAL",
            "approval_request_id": "SYNTHETIC-ONLY",
            "execution_policy": "LOG_ONLY",
            "automatic_execution": False,
            "automatic_containment": False,
            "endpoint_modified": False,
        }
        with patch(
            "integration.smartsoc_v1_approval_bridge."
            "process_v1_result_for_approval",
            return_value=expected,
        ) as adapter:
            result = process_inference_response(
                self.original_alert, self.receiver_response
            )

        self.assertEqual(result, expected)
        adapter.assert_called_once_with(
            self.original_alert,
            self.receiver_response["inference_result"],
            reasoning_engine=None,
        )

    def test_rejects_failed_inference(self):
        response = {"status": "INFERENCE_ERROR"}
        with patch(
            "integration.smartsoc_v1_approval_bridge."
            "process_v1_result_for_approval"
        ) as adapter:
            result = process_inference_response(
                self.original_alert, response
            )

        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["reason"], "V1_INFERENCE_NOT_SUCCESSFUL")
        adapter.assert_not_called()

    def test_rejects_missing_inference_result(self):
        with patch(
            "integration.smartsoc_v1_approval_bridge."
            "process_v1_result_for_approval"
        ) as adapter:
            result = process_inference_response(
                self.original_alert, {"status": "INFERENCE_OK"}
            )

        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(
            result["reason"], "INFERENCE_RESULT_MISSING_OR_INVALID"
        )
        adapter.assert_not_called()

    def test_rejects_non_object_original_alert(self):
        result = process_inference_response(
            None, self.receiver_response
        )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(
            result["reason"], "ORIGINAL_ALERT_MUST_BE_OBJECT"
        )


if __name__ == "__main__":
    unittest.main()
