import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
INTEGRATION_V2 = SRC / "integration_v2"

for path in (str(SRC), str(INTEGRATION_V2)):
    if path not in sys.path:
        sys.path.insert(0, path)

from autonomous_v2 import victim_mitigation_adapter_v2 as adapter
import execution_config_v2 as policy


class TestVictimMitigationAdapterV2(unittest.TestCase):

    def setUp(self):
        self.original_secret = os.environ.get("SMARTSOC_AGENT_SECRET")
        os.environ["SMARTSOC_AGENT_SECRET"] = "S" * 64

    def tearDown(self):
        if self.original_secret is None:
            os.environ.pop("SMARTSOC_AGENT_SECRET", None)
        else:
            os.environ["SMARTSOC_AGENT_SECRET"] = self.original_secret

    @staticmethod
    def valid_reasoning_result():
        return {
            "mitigation": "PROCESS_TERMINATION",
            "process_id": 4321,
            "process_name": "notepad.exe",
            "technique": "T1087",
            "wazuh_rule_id": "92031",
            "reason": "Synthetic adapter unit test",
        }

    def test_valid_request_uses_mocked_http_and_dry_run_payload(self):
        response = Mock()
        response.status = 200
        response.read.return_value = json.dumps({
            "success": True,
            "verified": True,
            "status": "DRY_RUN_VALIDATED",
        }).encode("utf-8")

        connection = Mock()
        connection.getresponse.return_value = response

        with patch.object(
            adapter.http.client,
            "HTTPConnection",
            return_value=connection,
        ) as http_connection:
            result = adapter.execute_dry_run(
                self.valid_reasoning_result()
            )

        self.assertTrue(result["success"])
        self.assertTrue(result["verified"])
        self.assertEqual(result["status"], "DRY_RUN_VALIDATED")
        self.assertTrue(result["victim_request_attempted"])
        self.assertTrue(result["victim_http_response_received"])

        http_connection.assert_called_once_with(
            adapter.VICTIM_HOST,
            adapter.VICTIM_PORT,
            timeout=adapter.TIMEOUT_SECONDS,
        )

        connection.request.assert_called_once()
        args, kwargs = connection.request.call_args

        self.assertEqual(args[0], "POST")
        self.assertEqual(args[1], "/mitigate")

        payload = json.loads(kwargs["body"].decode("utf-8"))
        self.assertEqual(payload["action"], "PROCESS_TERMINATION")
        self.assertEqual(
            payload["context"]["requested_mode"],
            "DRY_RUN",
        )
        self.assertEqual(payload["target"]["pid"], 4321)
        self.assertEqual(
            payload["target"]["process_name"],
            "notepad.exe",
        )
        self.assertIn("X-SmartSOC-Signature", kwargs["headers"])
        connection.close.assert_called_once()

    def test_invalid_action_is_rejected_without_http(self):
        reasoning = self.valid_reasoning_result()
        reasoning["mitigation"] = "ARBITRARY_COMMAND"

        with patch.object(adapter.http.client, "HTTPConnection") as http:
            result = adapter.execute_dry_run(reasoning)

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "REQUEST_BLOCKED")
        http.assert_not_called()

    def test_missing_secret_is_rejected_without_http(self):
        os.environ.pop("SMARTSOC_AGENT_SECRET", None)

        with patch.object(adapter.http.client, "HTTPConnection") as http:
            result = adapter.execute_dry_run(
                self.valid_reasoning_result()
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "CONFIGURATION_BLOCKED")
        http.assert_not_called()

    def test_invalid_policy_is_blocked_without_http(self):
        with patch.object(policy, "DRY_RUN", False):
            with patch.object(
                adapter.http.client, "HTTPConnection"
            ) as http:
                result = adapter.execute_dry_run(
                    self.valid_reasoning_result()
                )

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "POLICY_BLOCKED")
        http.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
