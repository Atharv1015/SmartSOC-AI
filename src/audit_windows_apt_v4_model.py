from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "windows_apt_v4"
MODEL_DIR = ROOT / "models" / "windows_apt_v4" / "calibrated"

STRUCTURED = DATA_DIR / "structured_train.csv"
TEXT = DATA_DIR / "behavior_text_train.csv"
LABELS = DATA_DIR / "labels_train.csv"
METADATA = DATA_DIR / "metadata_train.csv"

MODEL_PATH = MODEL_DIR / "multilabel_logistic_v4_calibrated.joblib"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer_v4_calibrated.joblib"
ENCODER_PATH = MODEL_DIR / "multilabel_encoder_v4_calibrated.joblib"
SCHEMA_PATH = MODEL_DIR / "feature_schema_v4_calibrated.json"


# ============================================================
# HELPERS
# ============================================================

def header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )


def parse_techniques(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def contains_any(series, terms):
    result = pd.Series(
        False,
        index=series.index
    )

    for term in terms:
        result |= series.str.contains(
            term,
            regex=False,
            na=False
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    header(
        "SmartSOC Windows-APT Model B v4 - CORRECTED MODEL AUDIT"
    )

    # --------------------------------------------------------
    # FILE CHECK
    # --------------------------------------------------------

    print("\nChecking required files...")

    for path in [
        STRUCTURED,
        TEXT,
        LABELS,
        METADATA,
        MODEL_PATH,
        TFIDF_PATH,
        ENCODER_PATH,
        SCHEMA_PATH,
    ]:
        require_file(path)
        print("[OK]", path)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading datasets...")

    structured = pd.read_csv(STRUCTURED)
    text = pd.read_csv(TEXT)
    labels = pd.read_csv(LABELS)
    metadata = pd.read_csv(METADATA)

    print("\nRows:")
    print("Structured :", len(structured))
    print("Text       :", len(text))
    print("Labels     :", len(labels))
    print("Metadata   :", len(metadata))

    # --------------------------------------------------------
    # CHECK ALIGNMENT
    # --------------------------------------------------------

    header("DATASET ALIGNMENT")

    lengths = {
        "structured": len(structured),
        "text": len(text),
        "labels": len(labels),
        "metadata": len(metadata),
    }

    print("\nDataset lengths:")

    for name, value in lengths.items():
        print(
            f"{name:<12}: {value}"
        )

    if len(set(lengths.values())) != 1:
        raise ValueError(
            "Dataset row counts are not aligned."
        )

    print(
        "\n[OK] All datasets contain the same number of rows."
    )

    # --------------------------------------------------------
    # LABEL STRUCTURE
    # --------------------------------------------------------

    header("LABEL STRUCTURE")

    print(
        "Label columns:",
        list(labels.columns)
    )

    if "techniques" not in labels.columns:
        raise ValueError(
            "Expected 'techniques' column."
        )

    parsed_labels = labels["techniques"].apply(
        parse_techniques
    )

    print(
        "\nTotal labeled events:",
        len(parsed_labels)
    )

    print(
        "Events with labels:",
        (parsed_labels.str.len() > 0).sum()
    )

    print(
        "Events without labels:",
        (parsed_labels.str.len() == 0).sum()
    )

    # --------------------------------------------------------
    # TECHNIQUE PREVALENCE
    # --------------------------------------------------------

    header("TECHNIQUE PREVALENCE")

    technique_counts = {}

    for technique_list in parsed_labels:

        for technique in technique_list:

            technique_counts[technique] = (
                technique_counts.get(
                    technique,
                    0
                ) + 1
            )

    technique_counts = dict(
        sorted(
            technique_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    print()

    for technique, count in technique_counts.items():

        percentage = (
            count / len(labels)
        ) * 100

        print(
            f"{technique:<12}"
            f"{count:>7} "
            f"({percentage:>7.3f}%)"
        )

    # --------------------------------------------------------
    # MODEL ARTIFACTS
    # --------------------------------------------------------

    header("MODEL INFORMATION")

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
        SCHEMA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        schema = json.load(f)

    techniques = list(
        encoder.classes_
    )

    print(
        "Model type        :",
        type(model).__name__
    )

    print(
        "MITRE techniques  :",
        len(techniques)
    )

    print(
        "TF-IDF vocabulary :",
        len(tfidf.vocabulary_)
    )

    print(
        "Expected features :",
        schema.get(
            "final_feature_count"
        )
    )

    # --------------------------------------------------------
    # TEXT COLUMN
    # --------------------------------------------------------

    header("BEHAVIOR TEXT")

    print(
        "Text columns:",
        list(text.columns)
    )

    if "behavior_text_normalized" not in text.columns:
        raise ValueError(
            "Expected behavior_text_normalized."
        )

    behavior = (
        text["behavior_text_normalized"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    empty_count = (
        behavior.str.strip().eq("").sum()
    )

    print(
        "Empty behavior rows:",
        empty_count
    )

    print(
        "Non-empty behavior rows:",
        len(behavior) - empty_count
    )

    # --------------------------------------------------------
    # POWERSHELL AUDIT
    # --------------------------------------------------------

    header("POWERSHELL LABEL AUDIT")

    powershell_mask = behavior.str.contains(
        "powershell",
        regex=False,
        na=False
    )

    print(
        "PowerShell events:",
        powershell_mask.sum()
    )

    ps_label_counts = {}

    for technique_list in parsed_labels[
        powershell_mask
    ]:

        for technique in technique_list:

            ps_label_counts[technique] = (
                ps_label_counts.get(
                    technique,
                    0
                ) + 1
            )

    print(
        "\nTechniques associated with PowerShell:"
    )

    for technique, count in sorted(
        ps_label_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"{technique:<12}: {count}"
        )

    # --------------------------------------------------------
    # ADMINISTRATIVE POWERSHELL
    # --------------------------------------------------------

    header(
        "ADMINISTRATIVE POWERSHELL AUDIT"
    )

    admin_terms = [
        "get-service",
        "get-process",
        "get-computerinfo",
        "get-wmiobject",
        "get-ciminstance",
        "get-eventlog",
        "get-childitem",
        "get-itemproperty",
        "get-command",
        "get-help",
        "get-content",
        "test-connection",
    ]

    admin_mask = (
        powershell_mask
        & contains_any(
            behavior,
            admin_terms
        )
    )

    print(
        "Administrative-looking PowerShell:",
        admin_mask.sum()
    )

    admin_counts = {}

    for technique_list in parsed_labels[
        admin_mask
    ]:

        for technique in technique_list:

            admin_counts[technique] = (
                admin_counts.get(
                    technique,
                    0
                ) + 1
            )

    print(
        "\nLabels attached to administrative-looking PowerShell:"
    )

    for technique, count in sorted(
        admin_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"{technique:<12}: {count}"
        )

    # --------------------------------------------------------
    # T1070.004 AUDIT
    # --------------------------------------------------------

    header(
        "T1070.004 LABEL AUDIT"
    )

    t1070_mask = parsed_labels.apply(
        lambda x: "T1070.004" in x
    )

    print(
        "T1070.004 events:",
        t1070_mask.sum()
    )

    print(
        "T1070.004 + PowerShell:",
        (
            t1070_mask
            & powershell_mask
        ).sum()
    )

    print(
        "T1070.004 + admin PowerShell:",
        (
            t1070_mask
            & admin_mask
        ).sum()
    )

    if powershell_mask.sum() > 0:

        rate = (
            (
                t1070_mask
                & powershell_mask
            ).sum()
            / powershell_mask.sum()
        )

        print(
            "PowerShell -> T1070.004 rate:",
            f"{rate:.4f}"
        )

    # --------------------------------------------------------
    # DESTRUCTIVE BEHAVIOR
    # --------------------------------------------------------

    header(
        "DESTRUCTIVE BEHAVIOR AUDIT"
    )

    destructive_terms = [
        "mass file",
        "file destruction",
        "file deletion",
        "delete files",
        "deleting files",
        "encrypt files",
        "file encryption",
        "ransomware",
        "shadow copy",
        "shadow copies",
        "vssadmin",
        "cipher",
        "destroy",
        "destruction",
    ]

    destructive_mask = contains_any(
        behavior,
        destructive_terms
    )

    print(
        "Potential destructive events:",
        destructive_mask.sum()
    )

    destructive_counts = {}

    for technique_list in parsed_labels[
        destructive_mask
    ]:

        for technique in technique_list:

            destructive_counts[technique] = (
                destructive_counts.get(
                    technique,
                    0
                ) + 1
            )

    print(
        "\nTechniques associated with destructive behavior:"
    )

    for technique, count in sorted(
        destructive_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"{technique:<12}: {count}"
        )

    for technique in [
        "T1485",
        "T1489",
        "T1565.001",
        "T1070.004",
    ]:

        mask = parsed_labels.apply(
            lambda x: technique in x
        )

        print(
            f"\n{technique}"
        )

        print(
            "Total:",
            mask.sum()
        )

        print(
            "Destructive text:",
            (
                mask
                & destructive_mask
            ).sum()
        )

    # --------------------------------------------------------
    # T1485 TRAINING SIGNAL
    # --------------------------------------------------------

    header(
        "T1485 TRAINING SIGNAL"
    )

    t1485_mask = parsed_labels.apply(
        lambda x: "T1485" in x
    )

    print(
        "T1485 positive events:",
        t1485_mask.sum()
    )

    print(
        "T1485 + destructive text:",
        (
            t1485_mask
            & destructive_mask
        ).sum()
    )

    print(
        "T1485 + PowerShell:",
        (
            t1485_mask
            & powershell_mask
        ).sum()
    )

    print(
        "\nSample T1485 behavior:"
    )

    samples = behavior[
        t1485_mask
    ].head(20)

    for i, sample in enumerate(
        samples,
        start=1
    ):

        print(
            f"{i:02d}. {sample[:500]}"
        )

    # --------------------------------------------------------
    # CONFLICTING SIGNATURES
    # --------------------------------------------------------

    header(
        "CONFLICTING BEHAVIOR SIGNATURES"
    )

    signature_df = pd.DataFrame({
        "behavior": behavior,
        "labels": labels["techniques"]
    })

    signature_groups = (
        signature_df
        .groupby("behavior")["labels"]
        .nunique()
    )

    conflicts = signature_groups[
        signature_groups > 1
    ]

    print(
        "Total unique behavior signatures:",
        behavior.nunique()
    )

    print(
        "Conflicting signatures:",
        len(conflicts)
    )

    if len(conflicts) > 0:

        print(
            "\nTop conflicting signatures:"
        )

        for signature in conflicts.index[:10]:

            group = signature_df[
                signature_df["behavior"]
                == signature
            ]

            print(
                "\nSIGNATURE:"
            )

            print(
                signature[:1000]
            )

            print(
                "\nLABELS:"
            )

            print(
                group["labels"]
                .value_counts()
                .to_string()
            )

    # --------------------------------------------------------
    # MODEL COEFFICIENTS
    # --------------------------------------------------------

    header(
        "MODEL COEFFICIENT AUDIT"
    )

    tfidf_names = np.array(
        tfidf.get_feature_names_out()
    )

    estimators = model.estimators_

    for target in [
        "T1059.001",
        "T1070.004",
        "T1105",
        "T1485",
        "T1565.001",
    ]:

        if target not in techniques:
            continue

        index = techniques.index(
            target
        )

        estimator = estimators[index]

        coefficients = (
            estimator.coef_.ravel()
        )

        print(
            f"\n--- {target} ---"
        )

        max_abs = np.max(
            np.abs(coefficients)
        )

        print(
            "Maximum absolute coefficient:",
            f"{max_abs:.8f}"
        )

        top_positive = np.argsort(
            coefficients
        )[-15:][::-1]

        print(
            "\nTop positive TF-IDF features:"
        )

        for idx in top_positive:

            if idx >= len(tfidf_names):
                continue

            print(
                f"{tfidf_names[idx]:<45}"
                f"{coefficients[idx]: .6f}"
            )

    # --------------------------------------------------------
    # IMPORTANT MODEL CHECK
    # --------------------------------------------------------

    header(
        "MODEL QUALITY CHECK"
    )

    for target in [
        "T1485",
        "T1565.001",
        "T1070.004",
        "T1059.001",
    ]:

        if target not in techniques:
            continue

        idx = techniques.index(
            target
        )

        coef = (
            model.estimators_[idx]
            .coef_.ravel()
        )

        print(
            f"{target:<12}"
            f" max|coef|="
            f"{np.max(np.abs(coef)):.8f}"
        )

    print(
        """
==============================================================
AUDIT INTERPRETATION
==============================================================

Pay particular attention to:

1. T1070.004 attached to ordinary PowerShell.
2. T1485 positive examples.
3. T1485 examples containing actual destructive language.
4. Conflicting behavior signatures.
5. Maximum coefficient magnitude for T1485.
6. Whether T1485 has enough meaningful training examples.

If T1485 has many positive labels but essentially zero model
coefficients, the problem is probably not inference code.

It indicates that the current feature representation does not
provide the classifier with useful discriminative information
for T1485.

If T1070.004 is heavily associated with benign/admin PowerShell,
the dataset contains a potentially problematic label correlation.

Do not deploy automated containment until these two issues are
resolved.
"""
    )


if __name__ == "__main__":
    main()