import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# SmartSOC Windows-APT Model B
# Real Inference Adapter v1
# ============================================================
#
# Purpose:
#
#   Accept a Wazuh/Sysmon-style event
#   -> create SmartSOC structured features
#   -> create behavior text
#   -> load calibrated Model B
#   -> predict MITRE ATT&CK techniques
#   -> apply technique-specific thresholds
#   -> return normalized AI result
#
# This script DOES NOT:
#   - kill processes
#   - isolate endpoints
#   - modify firewall
#   - execute response commands
#
# It only performs inference.
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "windows_apt_v3"
)

MODEL_PATH = (
    MODEL_DIR
    / "multilabel_logistic_v3_calibrated.joblib"
)

TFIDF_PATH = (
    MODEL_DIR
    / "tfidf_vectorizer_calibrated.joblib"
)

ENCODER_PATH = (
    MODEL_DIR
    / "multilabel_encoder_calibrated.joblib"
)

THRESHOLD_PATH = (
    MODEL_DIR
    / "technique_thresholds.json"
)

STRUCTURED_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "windows_apt_v3"
    / "structured_features.csv"
)

TEXT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "windows_apt_v3"
    / "behavior_text.csv"
)


# ============================================================
# REQUIRED FEATURE LIST
# ============================================================
#
# This MUST remain aligned with the feature-engineering
# pipeline used during training.
#
# ============================================================

STRUCTURED_FEATURES = [
    "event_id",
    "system_level",
    "is_sysmon",
    "is_security_channel",

    "process_name",
    "parent_process_name",
    "process_extension",
    "parent_extension",

    "has_process_image",
    "has_parent_image",
    "has_user",
    "has_parent_user",

    "is_system_user",
    "is_admin_user",

    "integrity_high",
    "integrity_medium",

    "command_length",
    "parent_command_length",

    "has_command_line",
    "has_parent_command",

    "uses_powershell",
    "uses_cmd",
    "uses_net",
    "uses_whoami",
    "uses_schtasks",
    "uses_reg",
    "uses_rundll32",
    "uses_wmic",
    "uses_certutil",
    "uses_bitsadmin",
    "uses_ipconfig",
    "uses_netstat",
    "uses_tasklist",
    "uses_sc",
    "uses_mshta",
    "uses_regsvr32",
    "uses_cscript",
    "uses_wscript",
    "uses_wevtutil",
    "uses_vssadmin",

    "has_encoded_command",
    "has_download_behavior",
    "has_remote_execution_hint",
    "has_registry_operation",
    "has_file_deletion",
    "has_obfuscation_hint",

    "same_process_parent",
    "suspicious_parent_wazuh",

    "has_hash",
    "has_sha256",
    "has_md5",

    "has_source_ip",
    "has_destination_ip",
    "has_source_port",
    "has_destination_port",

    "has_protocol",
    "is_network_event",

    "has_target_filename",
    "has_image_loaded",
    "has_target_object",
]


# ============================================================
# LOAD ARTIFACTS
# ============================================================

def load_artifacts():

    print("\nLoading SmartSOC Model B artifacts...")

    required = [
        MODEL_PATH,
        TFIDF_PATH,
        ENCODER_PATH,
        THRESHOLD_PATH,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required artifact not found:\n{path}"
            )

    model = joblib.load(
        MODEL_PATH
    )

    tfidf = joblib.load(
        TFIDF_PATH
    )

    encoder = joblib.load(
        ENCODER_PATH
    )

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        thresholds = json.load(f)

    print("Model loaded.")
    print("TF-IDF loaded.")
    print("Encoder loaded.")
    print("Thresholds loaded.")

    return (
        model,
        tfidf,
        encoder,
        thresholds
    )


# ============================================================
# SAFE STRING CONVERSION
# ============================================================

def safe_string(value):

    if value is None:
        return ""

    if isinstance(
        value,
        float
    ) and np.isnan(value):

        return ""

    return str(value)


# ============================================================
# BOOLEAN CONVERSION
# ============================================================

def bool_value(value):

    if isinstance(
        value,
        bool
    ):

        return int(value)

    if value is None:
        return 0

    value = safe_string(
        value
    ).strip().lower()

    return int(
        value in [
            "1",
            "true",
            "yes",
            "y",
            "present"
        ]
    )


# ============================================================
# TEXT SEARCH HELPER
# ============================================================

def contains_any(
    text,
    patterns
):

    text = safe_string(
        text
    ).lower()

    return int(
        any(
            pattern.lower() in text
            for pattern in patterns
        )
    )


# ============================================================
# EXTRACT EVENT FIELDS
# ============================================================

def extract_event_data(event):

    """
    Extract useful values from either:

        1. Flattened SmartSOC event
        2. Typical Wazuh JSON event
        3. Sysmon-like event
    """

    # --------------------------------------------------------
    # Convert entire event to searchable text
    # --------------------------------------------------------

    event_text = json.dumps(
        event,
        ensure_ascii=False
    )

    # --------------------------------------------------------
    # Common process information
    # --------------------------------------------------------

    process_name = (
        event.get("process_name")
        or event.get("process")
        or event.get("image")
        or event.get(
            "win",
            {}
        ).get(
            "eventdata",
            {}
        ).get(
            "image"
        )
        or ""
    )

    parent_process_name = (
        event.get("parent_process_name")
        or event.get("parent_image")
        or event.get(
            "win",
            {}
        ).get(
            "eventdata",
            {}
        ).get(
            "parentImage"
        )
        or ""
    )

    command_line = (
        event.get("command_line")
        or event.get("command")
        or event.get(
            "win",
            {}
        ).get(
            "eventdata",
            {}
        ).get(
            "commandLine"
        )
        or ""
    )

    parent_command_line = (
        event.get("parent_command_line")
        or event.get(
            "win",
            {}
        ).get(
            "eventdata",
            {}
        ).get(
            "parentCommandLine"
        )
        or ""
    )

    user = (
        event.get("user")
        or event.get(
            "win",
            {}
        ).get(
            "eventdata",
            {}
        ).get(
            "user"
        )
        or ""
    )

    # --------------------------------------------------------
    # Event ID
    # --------------------------------------------------------

    event_id = (
        event.get("event_id")
        or event.get(
            "eventID"
        )
        or event.get(
            "win",
            {}
        ).get(
            "system",
            {}
        ).get(
            "eventID"
        )
        or 0
    )

    try:

        event_id = int(
            event_id
        )

    except (
        ValueError,
        TypeError
    ):

        event_id = 0

    # --------------------------------------------------------
    # Wazuh rule level
    # --------------------------------------------------------

    rule_level = (
        event.get("rule_level")
        or event.get(
            "level"
        )
        or event.get(
            "rule",
            {}
        ).get(
            "level"
        )
        or 0
    )

    try:

        rule_level = int(
            rule_level
        )

    except (
        ValueError,
        TypeError
    ):

        rule_level = 0

    # --------------------------------------------------------
    # System information
    # --------------------------------------------------------

    channel = (
        event.get("channel")
        or event.get(
            "win",
            {}
        ).get(
            "system",
            {}
        ).get(
            "channel"
        )
        or ""
    )

    integrity = (
        event.get("integrity")
        or event.get(
            "win",
            {}
        ).get(
            "eventdata",
            {}
        ).get(
            "integrityLevel"
        )
        or ""
    )

    source_ip = (
        event.get("source_ip")
        or event.get(
            "srcip"
        )
        or ""
    )

    destination_ip = (
        event.get("destination_ip")
        or event.get(
            "dstip"
        )
        or ""
    )

    source_port = (
        event.get("source_port")
        or event.get(
            "srcport"
        )
        or ""
    )

    destination_port = (
        event.get("destination_port")
        or event.get(
            "dstport"
        )
        or ""
    )

    protocol = (
        event.get("protocol")
        or ""
    )

    target_filename = (
        event.get("target_filename")
        or event.get(
            "win",
            {}
        ).get(
            "eventdata",
            {}
        ).get(
            "targetFilename"
        )
        or ""
    )

    target_object = (
        event.get("target_object")
        or event.get(
            "win",
            {}
        ).get(
            "eventdata",
            {}
        ).get(
            "objectName"
        )
        or ""
    )

    image_loaded = (
        event.get("image_loaded")
        or event.get(
            "win",
            {}
        ).get(
            "eventdata",
            {}
        ).get(
            "imageLoaded"
        )
        or ""
    )

    sha256 = (
        event.get("sha256")
        or event.get("sha256_hash")
        or ""
    )

    md5 = (
        event.get("md5")
        or event.get("md5_hash")
        or ""
    )

    # --------------------------------------------------------
    # Return normalized event
    # --------------------------------------------------------

    return {
        "event_id":
            event_id,

        "rule_level":
            rule_level,

        "channel":
            safe_string(
                channel
            ),

        "process_name":
            safe_string(
                process_name
            ),

        "parent_process_name":
            safe_string(
                parent_process_name
            ),

        "command_line":
            safe_string(
                command_line
            ),

        "parent_command_line":
            safe_string(
                parent_command_line
            ),

        "user":
            safe_string(
                user
            ),

        "integrity":
            safe_string(
                integrity
            ),

        "source_ip":
            safe_string(
                source_ip
            ),

        "destination_ip":
            safe_string(
                destination_ip
            ),

        "source_port":
            safe_string(
                source_port
            ),

        "destination_port":
            safe_string(
                destination_port
            ),

        "protocol":
            safe_string(
                protocol
            ),

        "target_filename":
            safe_string(
                target_filename
            ),

        "target_object":
            safe_string(
                target_object
            ),

        "image_loaded":
            safe_string(
                image_loaded
            ),

        "sha256":
            safe_string(
                sha256
            ),

        "md5":
            safe_string(
                md5
            ),

        "event_text":
            event_text
    }


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_structured_features(
    data
):

    process_name = (
        data["process_name"]
    )

    parent_process_name = (
        data["parent_process_name"]
    )

    command_line = (
        data["command_line"]
    )

    parent_command_line = (
        data["parent_command_line"]
    )

    user = (
        data["user"]
    )

    integrity = (
        data["integrity"]
    )

    event_text = (
        data["event_text"]
    )

    process_lower = process_name.lower()
    parent_lower = parent_process_name.lower()
    command_lower = command_line.lower()

    # --------------------------------------------------------
    # Extensions
    # --------------------------------------------------------

    process_extension = (
        Path(
            process_name
        ).suffix.lower()
    )

    parent_extension = (
        Path(
            parent_process_name
        ).suffix.lower()
    )

    # --------------------------------------------------------
    # Basic indicators
    # --------------------------------------------------------

    features = {

        "event_id":
            data["event_id"],

        "system_level":
            data["rule_level"],

        "is_sysmon":
            int(
                "sysmon"
                in
                data["channel"].lower()
                or
                "sysmon"
                in
                event_text.lower()
            ),

        "is_security_channel":
            int(
                "security"
                in
                data["channel"].lower()
            ),

        "process_name":
            process_name,

        "parent_process_name":
            parent_process_name,

        "process_extension":
            process_extension,

        "parent_extension":
            parent_extension,

        "has_process_image":
            int(
                bool(
                    process_name
                )
            ),

        "has_parent_image":
            int(
                bool(
                    parent_process_name
                )
            ),

        "has_user":
            int(
                bool(
                    user
                )
            ),

        "has_parent_user":
            int(
                bool(
                    data.get(
                        "parent_user",
                        ""
                    )
                )
            ),

        "is_system_user":
            int(
                user.lower()
                in [
                    "system",
                    "nt authority\\system",
                    "local system"
                ]
            ),

        "is_admin_user":
            int(
                "admin"
                in
                user.lower()
            ),

        "integrity_high":
            int(
                "high"
                in
                integrity.lower()
            ),

        "integrity_medium":
            int(
                "medium"
                in
                integrity.lower()
            ),

        "command_length":
            len(command_line),

        "parent_command_length":
            len(parent_command_line),

        "has_command_line":
            int(
                bool(
                    command_line
                )
            ),

        "has_parent_command":
            int(
                bool(
                    parent_command_line
                )
            ),

        # ----------------------------------------------------
        # Command indicators
        # ----------------------------------------------------

        "uses_powershell":
            contains_any(
                command_lower,
                [
                    "powershell",
                    "pwsh"
                ]
            ),

        "uses_cmd":
            contains_any(
                command_lower,
                [
                    "cmd.exe",
                    "cmd "
                ]
            ),

        "uses_net":
            contains_any(
                command_lower,
                [
                    " net ",
                    "net.exe"
                ]
            ),

        "uses_whoami":
            contains_any(
                command_lower,
                [
                    "whoami"
                ]
            ),

        "uses_schtasks":
            contains_any(
                command_lower,
                [
                    "schtasks"
                ]
            ),

        "uses_reg":
            contains_any(
                command_lower,
                [
                    "reg.exe",
                    " reg "
                ]
            ),

        "uses_rundll32":
            contains_any(
                command_lower,
                [
                    "rundll32"
                ]
            ),

        "uses_wmic":
            contains_any(
                command_lower,
                [
                    "wmic"
                ]
            ),

        "uses_certutil":
            contains_any(
                command_lower,
                [
                    "certutil"
                ]
            ),

        "uses_bitsadmin":
            contains_any(
                command_lower,
                [
                    "bitsadmin"
                ]
            ),

        "uses_ipconfig":
            contains_any(
                command_lower,
                [
                    "ipconfig"
                ]
            ),

        "uses_netstat":
            contains_any(
                command_lower,
                [
                    "netstat"
                ]
            ),

        "uses_tasklist":
            contains_any(
                command_lower,
                [
                    "tasklist"
                ]
            ),

        "uses_sc":
            contains_any(
                command_lower,
                [
                    "sc.exe",
                    " sc "
                ]
            ),

        "uses_mshta":
            contains_any(
                command_lower,
                [
                    "mshta"
                ]
            ),

        "uses_regsvr32":
            contains_any(
                command_lower,
                [
                    "regsvr32"
                ]
            ),

        "uses_cscript":
            contains_any(
                command_lower,
                [
                    "cscript"
                ]
            ),

        "uses_wscript":
            contains_any(
                command_lower,
                [
                    "wscript"
                ]
            ),

        "uses_wevtutil":
            contains_any(
                command_lower,
                [
                    "wevtutil"
                ]
            ),

        "uses_vssadmin":
            contains_any(
                command_lower,
                [
                    "vssadmin"
                ]
            ),

        # ----------------------------------------------------
        # Suspicious behavior
        # ----------------------------------------------------

        "has_encoded_command":
            contains_any(
                command_lower,
                [
                    "-enc",
                    "-encodedcommand",
                    "frombase64string",
                    "base64"
                ]
            ),

        "has_download_behavior":
            contains_any(
                command_lower,
                [
                    "invoke-webrequest",
                    "iwr ",
                    "wget ",
                    "curl ",
                    "bitsadmin",
                    "downloadstring",
                    "downloadfile",
                    "certutil"
                ]
            ),

        "has_remote_execution_hint":
            contains_any(
                command_lower,
                [
                    "winrm",
                    "psexec",
                    "invoke-command",
                    "enter-pssession",
                    "wmic /node",
                    "\\\\"
                ]
            ),

        "has_registry_operation":
            contains_any(
                command_lower,
                [
                    "reg add",
                    "reg delete",
                    "reg.exe",
                    "registry"
                ]
            ),

        "has_file_deletion":
            contains_any(
                command_lower,
                [
                    "del ",
                    "erase ",
                    "remove-item",
                    "rm ",
                    "cipher /w",
                    "wevtutil cl"
                ]
            ),

        "has_obfuscation_hint":
            contains_any(
                command_lower,
                [
                    "encodedcommand",
                    "frombase64string",
                    "-nop",
                    "-w hidden",
                    "invoke-expression",
                    "iex ",
                    "obfuscat"
                ]
            ),

        "same_process_parent":
            int(
                process_lower
                == parent_lower
                and
                bool(process_lower)
            ),

        "suspicious_parent_wazuh":
            int(
                "wazuh"
                in
                parent_lower
            ),

        # ----------------------------------------------------
        # Hashes
        # ----------------------------------------------------

        "has_hash":
            int(
                bool(
                    data["sha256"]
                    or
                    data["md5"]
                )
            ),

        "has_sha256":
            int(
                bool(
                    data["sha256"]
                )
            ),

        "has_md5":
            int(
                bool(
                    data["md5"]
                )
            ),

        # ----------------------------------------------------
        # Network indicators
        # ----------------------------------------------------

        "has_source_ip":
            int(
                bool(
                    data["source_ip"]
                )
            ),

        "has_destination_ip":
            int(
                bool(
                    data["destination_ip"]
                )
            ),

        "has_source_port":
            int(
                bool(
                    data["source_port"]
                )
            ),

        "has_destination_port":
            int(
                bool(
                    data["destination_port"]
                )
            ),

        "has_protocol":
            int(
                bool(
                    data["protocol"]
                )
            ),

        "is_network_event":
            int(
                bool(
                    data["source_ip"]
                    or
                    data["destination_ip"]
                    or
                    data["source_port"]
                    or
                    data["destination_port"]
                )
            ),

        # ----------------------------------------------------
        # File / object indicators
        # ----------------------------------------------------

        "has_target_filename":
            int(
                bool(
                    data["target_filename"]
                )
            ),

        "has_image_loaded":
            int(
                bool(
                    data["image_loaded"]
                )
            ),

        "has_target_object":
            int(
                bool(
                    data["target_object"]
                )
            ),
    }

    # --------------------------------------------------------
    # Ensure exact feature order
    # --------------------------------------------------------

    output = {}

    for feature in STRUCTURED_FEATURES:

        output[feature] = features.get(
            feature,
            0
        )

    return pd.DataFrame(
        [output],
        columns=STRUCTURED_FEATURES
    )


# ============================================================
# BEHAVIOR TEXT
# ============================================================

def build_behavior_text(
    data
):

    pieces = []

    fields = [
        data["process_name"],
        data["parent_process_name"],
        data["command_line"],
        data["parent_command_line"],
        data["user"],
        data["target_filename"],
        data["target_object"],
        data["image_loaded"],
        data["source_ip"],
        data["destination_ip"],
        data["protocol"],
        data["event_text"],
    ]

    for value in fields:

        value = safe_string(
            value
        ).strip()

        if value:

            pieces.append(
                value
            )

    return " ".join(
        pieces
    )


# ============================================================
# MODEL INPUT VALIDATION
# ============================================================

def validate_feature_dimensions(
    structured_matrix,
    text_matrix,
    model
):

    structured_count = (
        structured_matrix.shape[1]
    )

    text_count = (
        text_matrix.shape[1]
    )

    total_count = (
        structured_count
        +
        text_count
    )

    print(
        "\nFeature dimensions:"
    )

    print(
        "Structured:",
        structured_count
    )

    print(
        "TF-IDF:",
        text_count
    )

    print(
        "Combined:",
        total_count
    )

    # --------------------------------------------------------
    # Try to determine model expected dimensions
    # --------------------------------------------------------

    expected = None

    if hasattr(
        model,
        "estimators_"
    ):

        if model.estimators_:

            estimator = (
                model.estimators_[0]
            )

            if hasattr(
                estimator,
                "n_features_in_"
            ):

                expected = (
                    estimator.n_features_in_
                )

    elif hasattr(
        model,
        "n_features_in_"
    ):

        expected = (
            model.n_features_in_
        )

    if expected is not None:

        print(
            "Model expects:",
            expected
        )

        if expected != total_count:

            raise ValueError(
                "\nMODEL FEATURE MISMATCH\n"
                f"Model expects {expected} features, "
                f"but inference generated {total_count}."
            )


# ============================================================
# PREDICTION
# ============================================================

def predict_event(
    event
):

    # --------------------------------------------------------
    # Load model artifacts
    # --------------------------------------------------------

    (
        model,
        tfidf,
        encoder,
        thresholds
    ) = load_artifacts()

    # --------------------------------------------------------
    # Extract normalized event
    # --------------------------------------------------------

    data = extract_event_data(
        event
    )

    # --------------------------------------------------------
    # Structured features
    # --------------------------------------------------------

    structured_df = (
        build_structured_features(
            data
        )
    )

    # --------------------------------------------------------
    # Behavior text
    # --------------------------------------------------------

    behavior_text = (
        build_behavior_text(
            data
        )
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    text_matrix = (
        tfidf.transform(
            [behavior_text]
        )
    )

    # --------------------------------------------------------
    # Structured matrix
    # --------------------------------------------------------
    #
    # The training pipeline used numerical features.
    # Convert to float32 and CSR format.
    # --------------------------------------------------------

    from scipy.sparse import csr_matrix, hstack

    structured_matrix = csr_matrix(
        structured_df.astype(
            np.float32
        ).values
    )

    text_matrix = text_matrix.tocsr()

    combined_matrix = hstack(
        [
            structured_matrix,
            text_matrix
        ],
        format="csr"
    )

    combined_matrix.sum_duplicates()

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    validate_feature_dimensions(
        structured_matrix,
        text_matrix,
        model
    )

    # --------------------------------------------------------
    # Predict probabilities
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            combined_matrix
        )
    )

    # --------------------------------------------------------
    # Multi-label One-vs-Rest handling
    # --------------------------------------------------------

    if isinstance(
        probabilities,
        list
    ):

        probability_vector = np.array(
            [
                float(
                    p[0, 1]
                )
                if np.ndim(p) > 1
                and
                p.shape[1] > 1
                else float(
                    np.ravel(p)[-1]
                )
                for p in probabilities
            ]
        )

    else:

        probability_vector = np.asarray(
            probabilities
        )

        if probability_vector.ndim == 2:

            if (
                probability_vector.shape[0] == 1
            ):

                probability_vector = (
                    probability_vector[0]
                )

            else:

                probability_vector = (
                    probability_vector[0]
                )

    # --------------------------------------------------------
    # Class names
    # --------------------------------------------------------

    try:

        classes = list(
            encoder.classes_
        )

    except AttributeError:

        classes = list(
            encoder
        )

    # --------------------------------------------------------
    # Apply calibrated thresholds
    # --------------------------------------------------------

    predictions = []

    prediction_details = []

    for index, technique in enumerate(
        classes
    ):

        if index >= len(
            probability_vector
        ):

            break

        probability = float(
            probability_vector[index]
        )

        # JSON keys are technique IDs.
        # Some calibration files may store
        # nested dictionaries.

        threshold_value = thresholds.get(
            technique,
            0.50
        )

        if isinstance(
            threshold_value,
            dict
        ):

            threshold_value = (
                threshold_value.get(
                    "threshold",
                    0.50
                )
            )

        threshold_value = float(
            threshold_value
        )

        is_predicted = (
            probability
            >=
            threshold_value
        )

        prediction_details.append(
            {
                "technique":
                    technique,

                "probability":
                    round(
                        probability,
                        6
                    ),

                "threshold":
                    round(
                        threshold_value,
                        6
                    ),

                "predicted":
                    bool(
                        is_predicted
                    )
            }
        )

        if is_predicted:

            predictions.append(
                technique
            )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------
    #
    # If no threshold is crossed, report the strongest
    # candidate for investigation but clearly mark that
    # it did not cross the calibrated threshold.
    # --------------------------------------------------------

    strongest_index = None

    if len(
        probability_vector
    ) > 0:

        strongest_index = int(
            np.argmax(
                probability_vector
            )
        )

    strongest_candidate = None

    if strongest_index is not None:

        strongest_candidate = {
            "technique":
                classes[
                    strongest_index
                ],

            "probability":
                round(
                    float(
                        probability_vector[
                            strongest_index
                        ]
                    ),
                    6
                )
        }

    # --------------------------------------------------------
    # Overall AI confidence
    # --------------------------------------------------------

    if predictions:

        selected_probabilities = [

            item["probability"]

            for item in
            prediction_details

            if item["technique"]
            in
            predictions
        ]

        ai_confidence = max(
            selected_probabilities
        )

    else:

        ai_confidence = 0.0

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {

        "model":
            "SmartSOC Windows-APT Model B v3",

        "model_type":
            "Calibrated Multi-Label MITRE ATT&CK Classifier",

        "predicted_techniques":
            predictions,

        "ai_confidence":
            round(
                ai_confidence,
                6
            ),

        "prediction_count":
            len(
                predictions
            ),

        "strongest_candidate":
            strongest_candidate,

        "behavior_text":
            behavior_text,

        "event": {

            "event_id":
                data["event_id"],

            "wazuh_rule_level":
                data["rule_level"],

            "process":
                data["process_name"],

            "parent_process":
                data["parent_process_name"],

            "command_line":
                data["command_line"],

            "source_ip":
                data["source_ip"],

            "destination_ip":
                data["destination_ip"],

            "sha256":
                data["sha256"],

            "md5":
                data["md5"]
        },

        "prediction_details":
            prediction_details
    }

    return result


# ============================================================
# DEMO EVENTS
# ============================================================

DEMO_EVENTS = [

    # ========================================================
    # NORMAL POWERHELL
    # ========================================================

    {
        "event_id": 1,

        "rule_level": 3,

        "channel":
            "Microsoft-Windows-Sysmon/Operational",

        "process_name":
            "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",

        "parent_process_name":
            "C:\\Windows\\explorer.exe",

        "command_line":
            "powershell.exe Get-Process",

        "user":
            "Administrator",

        "integrity":
            "Medium",

        "source_ip":
            "",

        "destination_ip":
            "",

        "protocol":
            "",

        "sha256":
            "",

        "md5":
            ""
    },

    # ========================================================
    # SUSPICIOUS ENCODED POWERSHELL
    # ========================================================

    {
        "event_id": 1,

        "rule_level": 9,

        "channel":
            "Microsoft-Windows-Sysmon/Operational",

        "process_name":
            "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",

        "parent_process_name":
            "C:\\Users\\Public\\suspicious.exe",

        "command_line":
            "powershell.exe -nop -w hidden -encodedcommand SQBFAFgA",

        "user":
            "Administrator",

        "integrity":
            "High",

        "destination_ip":
            "203.0.113.10",

        "destination_port":
            "443",

        "protocol":
            "TCP",

        "sha256":
            "DEMO-SHA256",

        "md5":
            ""
    }
]


# ============================================================
# RUN DEMO
# ============================================================

def run_demo():

    print(
        "\n"
        + "=" * 75
    )

    print(
        " SmartSOC Windows-APT Model B"
    )

    print(
        " Real Inference Adapter"
    )

    print(
        "=" * 75
    )

    for number, event in enumerate(
        DEMO_EVENTS,
        start=1
    ):

        print(
            "\n"
            + "-" * 75
        )

        print(
            f"DEMO EVENT {number}"
        )

        print(
            "-" * 75
        )

        try:

            result = predict_event(
                event
            )

            print(
                json.dumps(
                    result,
                    indent=4
                )
            )

        except Exception as error:

            print(
                "\nINFERENCE ERROR:"
            )

            print(
                str(error)
            )

            print(
                "\nThis normally indicates that the"
            )

            print(
                "inference feature construction does not"
            )

            print(
                "exactly match the training pipeline."
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_demo()