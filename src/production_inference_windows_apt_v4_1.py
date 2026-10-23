"""
SmartSOC v4.1 Production Inference API
======================================

Compatibility wrapper around the validated v4.1-aligned
production inference pipeline.

IMPORTANT:
- Uses the existing frozen v4.1 production model.
- Uses the existing aligned preprocessing pipeline.
- Does NOT retrain or modify the model.
- Does NOT modify production artifacts.
- Preserves the existing SmartSOCInference().infer(event) API.
"""

from pathlib import Path
import json
import sys
import traceback


# ============================================================================
# IMPORT THE VALIDATED ALIGNED PIPELINE
# ============================================================================

try:
    from production_inference_windows_apt_v4_1_aligned import (
        load_artifacts as _aligned_load_artifacts,
        validate_artifacts as _aligned_validate_artifacts,
        normalize_wazuh_event as _aligned_normalize_wazuh_event,
        build_feature_matrix as _aligned_build_feature_matrix,
        predict as _aligned_predict,
        build_result_json as _aligned_build_result_json,
        build_demo_event as _aligned_build_demo_event,
    )

except ImportError as exc:
    print()
    print("=" * 72)
    print("ERROR: Could not import aligned v4.1 production pipeline")
    print("=" * 72)
    print()
    print(str(exc))
    print()
    print(
        "Make sure this file exists:"
    )
    print(
        "src\\production_inference_windows_apt_v4_1_aligned.py"
    )
    print()
    raise


# ============================================================================
# PUBLIC COMPATIBILITY FUNCTIONS
# ============================================================================

def load_artifacts():
    """
    Load the frozen v4.1 production artifacts using the
    validated aligned implementation.
    """
    return _aligned_load_artifacts()


def validate_artifacts(artifacts):
    """
    Validate the frozen v4.1 production artifacts.
    """
    return _aligned_validate_artifacts(artifacts)


def normalize_wazuh_event(event):
    """
    Normalize a Wazuh event using the canonical v4.1-aligned
    normalization implementation.
    """
    return _aligned_normalize_wazuh_event(event)


def build_feature_matrix(event, artifacts):
    """
    Build the exact v4.1-aligned 6919-dimensional feature matrix.

    The aligned implementation performs:
        162 categorical features
        58 numerical features
        6699 TF-IDF features
        --------------------------------
        6919 final features
    """
    return _aligned_build_feature_matrix(
        event,
        artifacts,
    )


def predict(feature_result, artifacts):
    """
    Run the frozen production model using the aligned
    v4.1 feature representation.
    """
    return _aligned_predict(
        feature_result,
        artifacts,
    )


def build_result_json(
    event,
    feature_result,
    prediction_result,
    artifacts,
):
    """
    Build the standard SmartSOC v4.1 inference result.
    """
    return _aligned_build_result_json(
        event,
        feature_result,
        prediction_result,
        artifacts,
    )


# ============================================================================
# PRODUCTION INFERENCE API
# ============================================================================

class SmartSOCInference:

    def __init__(self):

        print()
        print("=" * 72)
        print("SmartSOC v4.1 Production Inference")
        print("=" * 72)
        print()

        print("[1/3] Loading aligned production artifacts...")

        self.artifacts = _aligned_load_artifacts()

        print()
        print("[2/3] Validating aligned production artifacts...")

        _aligned_validate_artifacts(
            self.artifacts
        )

        # --------------------------------------------------------------------
        # Version validation
        # --------------------------------------------------------------------

        manifest = self.artifacts["manifest"]

        version = manifest.get(
            "version",
            "",
        )

        if version != "v4.1":
            raise RuntimeError(
                "Production artifact version is "
                f"{version}, expected v4.1."
            )

        # --------------------------------------------------------------------
        # Model feature validation
        # --------------------------------------------------------------------

        model_features = getattr(
            self.artifacts["model"],
            "n_features_in_",
            None,
        )

        if model_features != 6919:
            raise RuntimeError(
                "Expected 6919 model features, "
                f"found {model_features}."
            )

        # --------------------------------------------------------------------
        # MITRE technique validation
        # --------------------------------------------------------------------

        technique_count = len(
            self.artifacts[
                "label_encoder"
            ].classes_
        )

        if technique_count != 45:
            raise RuntimeError(
                "Expected 45 MITRE techniques, "
                f"found {technique_count}."
            )

        print()
        print("[3/3] Production model validation complete.")

        print()
        print("[OK] SmartSOC v4.1 production model validated.")
        print(
            f"[OK] Model features      : {model_features}"
        )
        print(
            f"[OK] MITRE techniques    : {technique_count}"
        )
        print(
            "[OK] Canonical aligned pipeline enabled."
        )

    # ========================================================================
    # INFERENCE
    # ========================================================================

    def infer(
        self,
        wazuh_event,
    ):
        """
        Run SmartSOC v4.1 inference on a Wazuh event.

        Pipeline:

            Wazuh JSON
                ↓
            aligned normalization
                ↓
            exact feature derivation
                ↓
            categorical encoding
                ↓
            numerical scaling
                ↓
            TF-IDF
                ↓
            6919-dimensional feature vector
                ↓
            frozen OneVsRest model
                ↓
            MITRE ATT&CK predictions
        """

        # --------------------------------------------------------------------
        # 1. Normalize
        # --------------------------------------------------------------------

        normalized = _aligned_normalize_wazuh_event(
            wazuh_event
        )

        # --------------------------------------------------------------------
        # 2. Build exact v4.1-aligned features
        # --------------------------------------------------------------------

        feature_result = _aligned_build_feature_matrix(
            normalized,
            self.artifacts,
        )

        # --------------------------------------------------------------------
        # 3. Run frozen production model
        # --------------------------------------------------------------------

        prediction_result = _aligned_predict(
            feature_result,
            self.artifacts,
        )

        # --------------------------------------------------------------------
        # 4. Build standard result
        # --------------------------------------------------------------------

        output = _aligned_build_result_json(
            normalized,
            feature_result,
            prediction_result,
            self.artifacts,
        )

        return output


# ============================================================================
# INPUT LOADER
# ============================================================================

def load_input_event(
    input_path,
):
    """
    Load a single Wazuh JSON event from disk.
    """

    path = Path(
        input_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        event = json.load(
            handle
        )

    if not isinstance(
        event,
        dict,
    ):
        raise ValueError(
            "Input JSON must contain one Wazuh event object."
        )

    return event


# ============================================================================
# RESULT DISPLAY
# ============================================================================

def print_result(
    result,
):
    """
    Print a concise SmartSOC inference result.
    """

    print()
    print("=" * 72)
    print("SMARTSOC v4.1 INFERENCE RESULT")
    print("=" * 72)

    print(
        f"Event ID       : {result.get('event_id', '')}"
    )

    print(
        f"Rule ID        : {result.get('rule_id', '')}"
    )

    print(
        f"Rule level     : {result.get('rule_level', '')}"
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

    predictions = result.get(
        "predictions",
        [],
    )

    if not predictions:

        print(
            "  None"
        )

    else:

        for prediction in predictions:

            print(
                "  "
                f"{prediction['technique']:12s} "
                f"confidence="
                f"{prediction['confidence']:.6f} "
                f"threshold="
                f"{prediction['threshold']:.6f}"
            )

    print()

    print("Top probability scores:")

    top_scores = result.get(
        "top_probability_scores",
        [],
    )

    for score in top_scores:

        marker = (
            "*"
            if score.get(
                "detected",
                False,
            )
            else " "
        )

        print(
            f"  {marker} "
            f"{score['technique']:12s} "
            f"{score['confidence']:.6f} "
            f"threshold="
            f"{score['threshold']:.6f}"
        )

    print()

    print("Production dimensions:")
    print(
        f"  Features   : "
        f"{result.get('feature_count')}"
    )
    print(
        f"  Techniques : "
        f"{result.get('mitre_technique_count')}"
    )

    print()

    print(
        "[OK] SmartSOC v4.1 inference completed."
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 72)
    print("SmartSOC v4.1 Production Inference API")
    print("=" * 72)

    # ------------------------------------------------------------------------
    # Determine input
    # ------------------------------------------------------------------------

    if (
        len(sys.argv) >= 3
        and sys.argv[1] == "--input"
    ):

        input_path = sys.argv[2]

        print()
        print(
            f"Input event: {input_path}"
        )

        event = load_input_event(
            input_path
        )

    elif len(sys.argv) == 1:

        print()
        print(
            "No --input supplied."
        )

        print(
            "Using the aligned module's demo event."
        )

        event = _aligned_build_demo_event()

    else:

        print()
        print(
            "Usage:"
        )

        print(
            "  python "
            "production_inference_windows_apt_v4_1.py "
            "--input <wazuh_json>"
        )

        print()

        return 1

    # ------------------------------------------------------------------------
    # Initialize inference engine
    # ------------------------------------------------------------------------

    try:

        engine = SmartSOCInference()

        # --------------------------------------------------------------------
        # Run inference
        # --------------------------------------------------------------------

        result = engine.infer(
            event
        )

        # --------------------------------------------------------------------
        # Display result
        # --------------------------------------------------------------------

        print_result(
            result
        )

        return 0

    except Exception as exc:

        print()
        print("=" * 72)
        print("SMARTSOC INFERENCE ERROR")
        print("=" * 72)

        print()
        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()
        traceback.print_exc()

        return 1


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )