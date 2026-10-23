"""
SmartSOC Autonomous Reasoning Engine v2

Purpose:
    Convert SmartSOC ML detections and Wazuh context into
    a structured defensive decision.

Pipeline:

    Wazuh Event
        ↓
    SmartSOC ML Detection
        ↓
    Contextual Reasoning
        ↓
    Defensive Objective
        ↓
    Mitigation Decision

Important design principle:

    Detection does not automatically imply mitigation.

The reasoning layer evaluates:
    - MITRE technique
    - ML confidence
    - Wazuh rule
    - process context
    - parent process context
    - command-line context

Current defensive capabilities:
    - PROCESS_TERMINATION
    - FILE_QUARANTINE
    - ACCOUNT_CONTAINMENT

Current T1087 behavior:
    Investigation only.

No endpoint action is performed by this module.
"""

from __future__ import annotations

from typing import Any, Dict


class AutonomousReasoningEngine:
    """
    Context-aware defensive reasoning engine.
    """

    VERSION = "autonomous-reasoning-v2"

    SUPPORTED_MITIGATIONS = {
        "PROCESS_TERMINATION",
        "FILE_QUARANTINE",
        "ACCOUNT_CONTAINMENT",
    }

    def reason(
        self,
        inference_result: Dict[str, Any],
        wazuh_event: Dict[str, Any],
    ) -> Dict[str, Any]:

        # --------------------------------------------------------------
        # 1. Extract model detections
        # --------------------------------------------------------------

        predictions = inference_result.get(
            "predictions",
            [],
        )

        detected_predictions = [
            item
            for item in predictions
            if isinstance(item, dict)
            and item.get("confidence", 0.0)
            >= item.get("threshold", 1.0)
        ]

        if not detected_predictions:

            # Diagnostics only: preserve the original decision.
            # Candidate scores are evidence for review, not detections.
            candidate_scores = inference_result.get(
                "top_probability_scores", []
            )
            if not isinstance(candidate_scores, list):
                candidate_scores = []

            rule = wazuh_event.get("rule", {})
            if not isinstance(rule, dict):
                rule = {}

            data = wazuh_event.get("data", {})
            if not isinstance(data, dict):
                data = {}

            win = data.get("win", {})
            if not isinstance(win, dict):
                win = {}

            eventdata = win.get("eventdata", {})
            if not isinstance(eventdata, dict):
                eventdata = {}

            return {
                "status": "NO_DETECTION",
                "reasoning_engine": self.VERSION,
                "technique": None,
                "confidence": None,
                "threshold": None,
                "objective": None,
                "mitigation": "NONE",
                "reason": (
                    "No MITRE technique satisfied "
                    "its configured detection threshold."
                ),
                "diagnostics": {
                    "prediction_count": len(predictions)
                    if isinstance(predictions, list) else 0,
                    "candidate_scores": candidate_scores[:5],
                    "wazuh_rule_id": rule.get("id"),
                    "wazuh_rule_description": rule.get("description"),
                    "process_name": eventdata.get("image"),
                    "parent_process": eventdata.get("parentImage"),
                    "process_id": eventdata.get("processId"),
                    "command_line_present": bool(
                        eventdata.get("commandLine")
                    ),
                },
            }

        # --------------------------------------------------------------
        # 2. Select highest-confidence detection
        # --------------------------------------------------------------

        selected = max(
            detected_predictions,
            key=lambda item: item.get(
                "confidence",
                0.0,
            ),
        )

        technique = selected.get(
            "technique"
        )

        confidence = selected.get(
            "confidence",
            0.0,
        )

        threshold = selected.get(
            "threshold"
        )

        # --------------------------------------------------------------
        # 3. Extract Wazuh context
        # --------------------------------------------------------------

        rule = wazuh_event.get(
            "rule",
            {},
        )

        data = wazuh_event.get(
            "data",
            {},
        )

        win = data.get(
            "win",
            {},
        )

        eventdata = win.get(
            "eventdata",
            {},
        )

        rule_id = rule.get(
            "id"
        )

        rule_level = rule.get(
            "level"
        )

        rule_description = rule.get(
            "description"
        )

        process_name = eventdata.get(
            "image"
        )

        parent_process = eventdata.get(
            "parentImage"
        )

        command_line = eventdata.get(
            "commandLine"
        )

        process_id = eventdata.get(
            "processId"
        )

        user = eventdata.get(
            "user"
        )

        # --------------------------------------------------------------
        # 4. Normalize context
        # --------------------------------------------------------------

        process_name_text = (
            str(process_name).lower()
            if process_name
            else ""
        )

        parent_process_text = (
            str(parent_process).lower()
            if parent_process
            else ""
        )

        command_line_text = (
            str(command_line).lower()
            if command_line
            else ""
        )

        # --------------------------------------------------------------
        # 5. Default defensive decision
        # --------------------------------------------------------------

        objective = None
        mitigation = "NONE"

        reason = (
            "Detection was confirmed, but the available "
            "context does not justify automatic endpoint "
            "modification."
        )

        # --------------------------------------------------------------
        # 6. T1087 — Account Discovery
        # --------------------------------------------------------------

        if technique == "T1087":

            objective = (
                "investigate_account_discovery"
            )

            mitigation = "NONE"

            reason = (
                "T1087 Account Discovery was detected "
                "above the configured confidence threshold. "
                "The observed net.exe/net1.exe activity can "
                "represent legitimate administrative behavior, "
                "so process termination is not automatically "
                "justified from this event alone."
            )

        # --------------------------------------------------------------
        # 7. Suspicious process context
        #
        # This branch intentionally requires additional evidence.
        # It does not trigger merely because a process exists.
        # --------------------------------------------------------------

        elif (
            technique
            and
            process_name_text
            and
            (
                "powershell" in parent_process_text
                or
                "wscript" in parent_process_text
                or
                "cscript" in parent_process_text
            )
            and
            any(
                indicator in command_line_text
                for indicator in (
                    "encodedcommand",
                    "downloadstring",
                    "invoke-webrequest",
                    "start-bitstransfer",
                )
            )
        ):

            objective = (
                "contain_suspicious_process"
            )

            mitigation = (
                "PROCESS_TERMINATION"
            )

            reason = (
                "The detected activity contains additional "
                "execution-context indicators associated with "
                "suspicious script execution. The defensive "
                "objective is process containment."
            )

        # --------------------------------------------------------------
        # 8. Generic detection
        # --------------------------------------------------------------

        else:

            objective = (
                "investigate_detected_activity"
            )

            mitigation = "NONE"

            reason = (
                "A MITRE technique was detected, but the "
                "available telemetry is insufficient to select "
                "an automatic endpoint modification safely."
            )

        # --------------------------------------------------------------
        # 9. Final structured decision
        # --------------------------------------------------------------

        return {
            "status": (
                "MITIGATION_SELECTED"
                if mitigation != "NONE"
                else "OBJECTIVE_SELECTED"
            ),
            "reasoning_engine": self.VERSION,
            "technique": technique,
            "confidence": confidence,
            "threshold": threshold,
            "wazuh_rule_id": rule_id,
            "wazuh_rule_level": rule_level,
            "wazuh_rule_description": rule_description,
            "process_name": process_name,
            "parent_process": parent_process,
            "command_line": command_line,
            "process_id": process_id,
            "user": user,
            "objective": objective,
            "mitigation": mitigation,
            "reason": reason,
        }


# ----------------------------------------------------------------------
# SELF TEST
# ----------------------------------------------------------------------

def main() -> None:

    print("=" * 72)
    print(
        "SmartSOC AUTONOMOUS REASONING ENGINE v2"
    )
    print("=" * 72)

    engine = AutonomousReasoningEngine()

    # --------------------------------------------------------------
    # Test 1 — Current real Wazuh T1087-style event
    # --------------------------------------------------------------

    inference_result = {
        "predictions": [
            {
                "technique": "T1087",
                "confidence": 0.918637,
                "threshold": 0.5,
            }
        ]
    }

    wazuh_event = {
        "rule": {
            "id": "92031",
            "level": 3,
            "description": (
                "Discovery activity executed"
            ),
        },
        "data": {
            "win": {
                "eventdata": {
                    "image": (
                        "C:\\Windows\\System32\\net1.exe"
                    ),
                    "parentImage": (
                        "C:\\Windows\\System32\\net.exe"
                    ),
                    "commandLine": (
                        "net user"
                    ),
                    "processId": "1234",
                }
            }
        },
    }

    result = engine.reason(
        inference_result=inference_result,
        wazuh_event=wazuh_event,
    )

    print()
    print("[TEST 1] T1087 Account Discovery")
    print()

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------------------
    # Test 2 — Suspicious script execution
    # --------------------------------------------------------------

    suspicious_inference = {
        "predictions": [
            {
                "technique": "T1059.001",
                "confidence": 0.97,
                "threshold": 0.5,
            }
        ]
    }

    suspicious_event = {
        "rule": {
            "id": "100001",
            "level": 12,
            "description": (
                "Suspicious PowerShell execution"
            ),
        },
        "data": {
            "win": {
                "eventdata": {
                    "image": (
                        "C:\\Windows\\Temp\\lab-test.exe"
                    ),
                    "parentImage": (
                        "C:\\Windows\\System32\\WindowsPowerShell\\"
                        "v1.0\\powershell.exe"
                    ),
                    "commandLine": (
                        "powershell.exe "
                        "-EncodedCommand "
                        "LAB_TEST"
                    ),
                    "processId": "5678",
                }
            }
        },
    }

    suspicious_result = engine.reason(
        inference_result=suspicious_inference,
        wazuh_event=suspicious_event,
    )

    print()
    print(
        "[TEST 2] Suspicious script execution"
    )
    print()

    for key, value in suspicious_result.items():

        print(
            f"{key}: {value}"
        )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()