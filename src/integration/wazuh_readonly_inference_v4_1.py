"""
SmartSOC v4.1
Read-only Wazuh -> production ML inference adapter.

IMPORTANT:
- READ ONLY
- NO automatic execution
- NO containment
- NO Shuffle calls
- NO Caldera calls
- NO endpoint modification
- Production model artifacts are loaded but never modified.

Input:
    Wazuh JSON alert exported to a local file.

Pipeline:
    Wazuh JSON
        -> normalize_wazuh_event()
        -> build_feature_matrix()
        -> predict()
        -> read-only result JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# Import the EXISTING production inference pipeline.
# Do not duplicate its preprocessing logic.
# ---------------------------------------------------------------------------

from production_inference_windows_apt_v4_1_aligned import (  # noqa: E402
    load_artifacts,
    validate_artifacts,
    normalize_wazuh_event,
    build_feature_matrix,
    predict,
    build_result_json,
)


# ---------------------------------------------------------------------------
# Read-only safety declaration
# ---------------------------------------------------------------------------

READ_ONLY = True
AUTOMATIC_EXECUTION = False
AUTOMATIC_CONTAINMENT = False


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Wazuh alert file does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Input path is not a file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            "The input JSON must contain one Wazuh alert object."
        )

    return data


# ---------------------------------------------------------------------------
# Read-only result wrapper
# ---------------------------------------------------------------------------

def build_readonly_result(
    result: Dict[str, Any],
    raw_event: Dict[str, Any],
) -> Dict[str, Any]:

    output = {
        "smartsoc_mode": "READ_ONLY_INFERENCE",

        "safety": {
            "read_only": READ_ONLY,
            "automatic_execution": AUTOMATIC_EXECUTION,
            "automatic_containment": AUTOMATIC_CONTAINMENT,
            "shuffle_called": False,
            "caldera_called": False,
            "endpoint_modified": False,
        },

        "source": {
            "type": "wazuh_alert_json",
            "agent": (
                raw_event.get("agent")
                if isinstance(raw_event.get("agent"), dict)
                else {}
            ),
        },

        "inference": result,
    }

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "SmartSOC v4.1 read-only Wazuh inference adapter"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to a single Wazuh JSON alert.",
    )

    parser.add_argument(
        "--output",
        default="reports/integration/wazuh_readonly_result.json",
        help="Output JSON path.",
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()

    output_path = (
        PROJECT_ROOT / args.output
    ).resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("SmartSOC v4.1 READ-ONLY Wazuh Integration")
    print("=" * 72)

    print()
    print("[SAFETY]")
    print("  Mode                 : READ_ONLY_INFERENCE")
    print("  Automatic execution  : FALSE")
    print("  Automatic containment : FALSE")
    print("  Shuffle calls        : FALSE")
    print("  Caldera calls        : FALSE")
    print("  Endpoint modification: FALSE")

    # -----------------------------------------------------------------------
    # Load production artifacts.
    # -----------------------------------------------------------------------

    print()
    print("[1/5] Loading production artifacts...")

    artifacts = load_artifacts()

    print("[PASS] Production artifacts loaded.")

    # -----------------------------------------------------------------------
    # Validate production artifact dimensions.
    # -----------------------------------------------------------------------

    print()
    print("[2/5] Validating production artifacts...")

    validate_artifacts(
        artifacts
    )

    print("[PASS] Production artifact validation completed.")

    # -----------------------------------------------------------------------
    # Load Wazuh alert.
    # -----------------------------------------------------------------------

    print()
    print("[3/5] Loading Wazuh alert...")

    raw_event = load_json_file(
        input_path
    )

    print(
        f"[PASS] Wazuh alert loaded: {input_path}"
    )

    # -----------------------------------------------------------------------
    # Normalize the exact Wazuh structure expected by the model.
    # -----------------------------------------------------------------------

    print()
    print("[4/5] Normalizing Wazuh event...")

    event = normalize_wazuh_event(
        raw_event
    )

    print("[PASS] Wazuh event normalized.")

    print()
    print("Normalized event:")
    print(
        json.dumps(
            event,
            indent=2,
            ensure_ascii=False,
        )
    )

    # -----------------------------------------------------------------------
    # Build EXACT production feature matrix.
    # -----------------------------------------------------------------------

    print()
    print("[5/5] Running production read-only inference...")

    feature_result = build_feature_matrix(
        event,
        artifacts,
    )

    matrix = feature_result["matrix"]

    print()
    print(
        f"[PASS] Final feature matrix: {matrix.shape}"
    )

    if matrix.shape[1] != 6919:
        raise RuntimeError(
            "Production feature dimension mismatch. "
            f"Expected 6919, got {matrix.shape[1]}."
        )

    # -----------------------------------------------------------------------
    # Prediction.
    # -----------------------------------------------------------------------

    prediction_result = predict(
        feature_result,
        artifacts,
    )

    # -----------------------------------------------------------------------
    # Build existing production result format.
    # -----------------------------------------------------------------------

    result = build_result_json(
        event,
        feature_result,
        prediction_result,
        artifacts,
    )

    # -----------------------------------------------------------------------
    # Add explicit read-only safety wrapper.
    # -----------------------------------------------------------------------

    readonly_result = build_readonly_result(
        result,
        raw_event,
    )

    # -----------------------------------------------------------------------
    # Write result.
    # -----------------------------------------------------------------------

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            readonly_result,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------------------------
    # Human-readable summary.
    # -----------------------------------------------------------------------

    detected = prediction_result.get(
        "detected",
        [],
    )

    all_scores = prediction_result.get(
        "all_scores",
        [],
    )

    print()
    print("=" * 72)
    print("READ-ONLY INFERENCE RESULT")
    print("=" * 72)

    print()
    print(
        f"Event ID       : {event.get('event_id', '')}"
    )

    print(
        f"Rule ID        : {event.get('rule_id', '')}"
    )

    print(
        f"Rule level     : {event.get('rule_level', '')}"
    )

    print(
        f"Provider       : {event.get('provider', '')}"
    )

    print(
        f"Channel        : {event.get('channel', '')}"
    )

    print(
        f"Process        : {result.get('process_name', '')}"
    )

    print(
        f"Parent process : "
        f"{result.get('parent_process_name', '')}"
    )

    print()
    print("Detected techniques:")

    if detected:

        for item in detected:
            print(
                "  "
                f"{item['technique']:<12} "
                f"confidence={item['confidence']:.6f} "
                f"threshold={item['threshold']:.6f}"
            )

    else:
        print("  None")

    print()
    print("Top probability scores:")

    for item in all_scores[:10]:

        marker = (
            "*"
            if item.get("detected")
            else " "
        )

        print(
            "  "
            f"{marker} "
            f"{item['technique']:<12} "
            f"{item['confidence']:.6f} "
            f"threshold={item['threshold']:.6f}"
        )

    print()
    print("Production dimensions:")
    print("  Categorical : 162")
    print("  Numerical   : 58")
    print("  TF-IDF      : 6699")
    print("  Final       : 6919")
    print("  Techniques  : 45")

    print()
    print("Safety:")
    print("  READ-ONLY            : TRUE")
    print("  AUTOMATIC EXECUTION  : FALSE")
    print("  AUTOMATIC CONTAINMENT: FALSE")

    print()
    print(
        f"[PASS] Result written to:\n"
        f"       {output_path}"
    )

    print()
    print(
        "[PASS] SmartSOC read-only inference completed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )