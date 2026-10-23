from pathlib import Path

path = Path(r"src/autonomous/autonomous_reasoning_v1.py")
source = path.read_text(encoding="utf-8")

old = '''        if not detected_predictions:

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
            }
'''

new = '''        if not detected_predictions:

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
'''

if old not in source:
    raise SystemExit(
        "PATCH ABORTED: expected code block was not found. "
        "No source changes were made."
    )

path.write_text(source.replace(old, new, 1), encoding="utf-8")
print("PATCH APPLIED: diagnostics added; decision logic unchanged.")
