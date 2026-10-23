import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from autonomous import autonomous_mitigation_controller_v1 as controller


class ControllerExecutionPolicyTests(unittest.TestCase):
    ACTION = "PROCESS_TERMINATION"

    def assert_policy_blocked(self, expected_message, **overrides):
        settings = {
            "CONTROLLED_LAB_MODE": True,
            "EXECUTION_LAYER_ENABLED": True,
            "AUTOMATIC_EXECUTION": True,
            "AUTOMATIC_CONTAINMENT": True,
            "HUMAN_AUTHORIZATION_REQUIRED": False,
        }
        settings.update(overrides)

        with patch.multiple(controller, **settings):
            with self.assertRaisesRegex(ValueError, expected_message):
                controller.AutonomousMitigationController.validate_execution_policy(
                    self.ACTION
                )

    def test_current_configuration_passes_policy_validation(self):
        # Authorization is enforced by the approval bridge before
        # the victim client is invoked, not by this policy validator.
        controller.AutonomousMitigationController.validate_execution_policy(
            self.ACTION
        )

    def test_lab_mode_disabled_blocks_execution(self):
        self.assert_policy_blocked(
            "Controlled laboratory execution mode is disabled",
            CONTROLLED_LAB_MODE=False,
        )

    def test_execution_layer_disabled_blocks_execution(self):
        self.assert_policy_blocked(
            "Execution layer is disabled",
            EXECUTION_LAYER_ENABLED=False,
        )

    def test_automatic_execution_disabled_blocks_execution(self):
        self.assert_policy_blocked(
            "Automatic execution is disabled",
            AUTOMATIC_EXECUTION=False,
        )

    def test_automatic_containment_disabled_blocks_execution(self):
        self.assert_policy_blocked(
            "Automatic containment is disabled",
            AUTOMATIC_CONTAINMENT=False,
        )

    def test_human_authorization_is_retained_for_approval_bridge(self):
        # The policy validator must not disable the human-approval
        # requirement; the bridge enforces it per request.
        with patch.multiple(
            controller,
            CONTROLLED_LAB_MODE=True,
            EXECUTION_LAYER_ENABLED=True,
            AUTOMATIC_EXECUTION=True,
            AUTOMATIC_CONTAINMENT=True,
            HUMAN_AUTHORIZATION_REQUIRED=True,
        ):
            controller.AutonomousMitigationController.validate_execution_policy(
                self.ACTION
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

