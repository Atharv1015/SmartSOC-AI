import json
from pathlib import Path


# ============================================================
# SmartSOC Risk & Decision Engine v2
# ============================================================
#
# Purpose:
#   Combine:
#       1. AI prediction
#       2. MITRE ATT&CK techniques
#       3. Wazuh severity
#       4. VirusTotal intelligence
#       5. AbuseIPDB intelligence
#       6. Endpoint behavioral indicators
#
#   into:
#
#       Risk Score 0-100
#       Severity
#       Decision
#       Recommended Response
#
# IMPORTANT:
#   This version DOES NOT execute endpoint actions.
#   It only generates a safe decision.
#
# Decisions:
#
#   PASS
#   ALERT
#   CONTAIN
#   AUTOMATED_RESPONSE
#
# ============================================================


# ============================================================
# DIRECTORIES
# ============================================================

MODEL_DIR = Path("models/windows_apt_v3")
RESULT_DIR = Path("results/risk_engine")

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DECISION THRESHOLDS
# ============================================================

PASS_THRESHOLD = 30
ALERT_THRESHOLD = 45
CONTAIN_THRESHOLD = 70
AUTOMATED_THRESHOLD = 85


# ============================================================
# MITRE TECHNIQUE WEIGHTS
# ============================================================
#
# These are SmartSOC policy weights.
#
# They are NOT official MITRE severity values.
#
# Common administrative techniques have lower weights.
# Techniques associated with stronger malicious behavior
# receive higher weights.
#
# ============================================================

TECHNIQUE_WEIGHTS = {

    # --------------------------------------------------------
    # Command and scripting
    # --------------------------------------------------------

    "T1059": 5,
    "T1059.001": 5,       # PowerShell
    "T1059.003": 5,       # Windows Command Shell
    "T1059.005": 7,       # Visual Basic

    # --------------------------------------------------------
    # Discovery
    # --------------------------------------------------------

    "T1087": 7,           # Account Discovery
    "T1003.001": 25,      # LSASS Memory
    "T1003.002": 25,

    # --------------------------------------------------------
    # Valid accounts
    # --------------------------------------------------------

    "T1078": 18,

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    "T1053.005": 18,
    "T1543.003": 22,
    "T1547.001": 25,

    # --------------------------------------------------------
    # Defense evasion
    # --------------------------------------------------------

    "T1027": 15,
    "T1027.004": 22,
    "T1070": 10,
    "T1070.004": 18,

    "T1562.001": 25,
    "T1562.004": 25,

    # --------------------------------------------------------
    # Process injection
    # --------------------------------------------------------

    "T1055": 30,

    # --------------------------------------------------------
    # Lateral movement
    # --------------------------------------------------------

    "T1021.006": 22,
    "T1210": 30,
    "T1570": 22,

    # --------------------------------------------------------
    # Command and Control
    # --------------------------------------------------------

    "T1105": 22,
    "T1095": 25,

    # --------------------------------------------------------
    # Privilege escalation
    # --------------------------------------------------------

    "T1548": 20,
    "T1548.002": 25,

    # --------------------------------------------------------
    # DLL / execution hijacking
    # --------------------------------------------------------

    "T1574.001": 28,
    "T1574.002": 28,
    "T1574.010": 28,

    # --------------------------------------------------------
    # Credential attacks
    # --------------------------------------------------------

    "T1110": 22,
    "T1110.001": 22,

    # --------------------------------------------------------
    # Registry
    # --------------------------------------------------------

    "T1112": 12,

    # --------------------------------------------------------
    # Account manipulation
    # --------------------------------------------------------

    "T1136": 20,

    # --------------------------------------------------------
    # Impact
    # --------------------------------------------------------

    "T1485": 35,
    "T1489": 35,
    "T1490": 35,
    "T1529": 35,
    "T1531": 35,
    "T1565.001": 35,
}


# ============================================================
# AI CONFIDENCE SCORE
# ============================================================

def confidence_score(confidence):
    """
    Convert AI confidence into risk contribution.

    Important:
        Low/moderate AI confidence does NOT automatically
        create a high-risk event.

    This prevents ordinary administrative activity from
    being escalated merely because the model produced
    a moderate prediction confidence.
    """

    confidence = max(
        0.0,
        min(
            1.0,
            float(confidence)
        )
    )

    if confidence < 0.70:
        return 0

    if confidence < 0.80:
        return 5

    if confidence < 0.90:
        return 12

    if confidence < 0.95:
        return 20

    return 30


# ============================================================
# WAZUH SCORE
# ============================================================

def wazuh_score(level):
    """
    Convert Wazuh rule level into a supporting risk score.

    Wazuh severity is evidence, not absolute truth.
    """

    level = float(level)

    if level <= 3:
        return 2

    if level <= 5:
        return 5

    if level <= 7:
        return 10

    if level <= 9:
        return 15

    if level <= 11:
        return 20

    if level <= 13:
        return 25

    return 30


# ============================================================
# VIRUSTOTAL SCORE
# ============================================================

def virustotal_score(
    malicious=0,
    suspicious=0,
    total_engines=0
):
    """
    Convert VirusTotal results into a risk contribution.

    malicious:
        Number of engines identifying the artifact as malicious.

    suspicious:
        Number of engines identifying the artifact as suspicious.

    total_engines:
        Total engines participating in the scan.
    """

    malicious = max(
        0,
        int(malicious)
    )

    suspicious = max(
        0,
        int(suspicious)
    )

    total_engines = max(
        0,
        int(total_engines)
    )

    if total_engines == 0:
        return 0

    malicious_ratio = (
        malicious /
        total_engines
    )

    suspicious_ratio = (
        suspicious /
        total_engines
    )

    score = (
        malicious_ratio * 25
        +
        suspicious_ratio * 10
    )

    return round(
        min(
            30,
            score
        ),
        2
    )


# ============================================================
# ABUSEIPDB SCORE
# ============================================================

def abuseipdb_score(
    confidence_score_value=0
):
    """
    Convert AbuseIPDB confidence score into
    SmartSOC risk contribution.
    """

    confidence_score_value = max(
        0,
        min(
            100,
            float(
                confidence_score_value
            )
        )
    )

    return round(
        confidence_score_value * 0.20,
        2
    )


# ============================================================
# BEHAVIORAL SCORE
# ============================================================

def behavioral_score(
    encoded_command=False,
    download_behavior=False,
    remote_execution=False,
    registry_operation=False,
    file_deletion=False,
    suspicious_parent=False,
    process_injection=False
):
    """
    Score endpoint behavioral indicators.

    These indicators are important because a technique
    by itself may be legitimate.

    Example:

        PowerShell

    is not automatically malicious.

    But:

        PowerShell
        + encoded command
        + download
        + suspicious parent

    is much stronger evidence.
    """

    score = 0

    if encoded_command:
        score += 8

    if download_behavior:
        score += 8

    if remote_execution:
        score += 10

    if registry_operation:
        score += 5

    if file_deletion:
        score += 10

    if suspicious_parent:
        score += 8

    if process_injection:
        score += 15

    return min(
        score,
        35
    )


# ============================================================
# TECHNIQUE SCORE
# ============================================================

def technique_score(techniques):
    """
    Calculate risk contribution from predicted MITRE
    ATT&CK techniques.

    The highest-risk technique receives its full weight.

    Additional techniques contribute 25% of their weight.

    This prevents a large number of predictions from
    automatically producing a 100 score.
    """

    if techniques is None:
        return 0

    if isinstance(
        techniques,
        str
    ):

        techniques = [
            x.strip()
            for x in techniques.split(",")
            if x.strip()
        ]

    if not techniques:
        return 0

    scores = []

    for technique in techniques:

        score = TECHNIQUE_WEIGHTS.get(
            technique,
            5
        )

        scores.append(
            score
        )

    if not scores:
        return 0

    scores.sort(
        reverse=True
    )

    total = scores[0]

    for extra in scores[1:]:

        total += (
            extra * 0.25
        )

    return round(
        min(
            40,
            total
        ),
        2
    )


# ============================================================
# HIGH-RISK TECHNIQUE CHECK
# ============================================================

def has_high_risk_technique(
    techniques
):
    """
    Determine whether the event contains a technique
    that requires stronger response consideration.
    """

    if not techniques:
        return False

    if isinstance(
        techniques,
        str
    ):

        techniques = [
            x.strip()
            for x in techniques.split(",")
            if x.strip()
        ]

    for technique in techniques:

        if TECHNIQUE_WEIGHTS.get(
            technique,
            0
        ) >= 30:

            return True

    return False


# ============================================================
# MAIN RISK CALCULATION
# ============================================================

def calculate_risk(
    techniques=None,
    ai_confidence=0.0,
    wazuh_level=0,

    vt_malicious=0,
    vt_suspicious=0,
    vt_total_engines=0,

    abuseipdb_confidence=0,

    encoded_command=False,
    download_behavior=False,
    remote_execution=False,
    registry_operation=False,
    file_deletion=False,
    suspicious_parent=False,
    process_injection=False
):

    # ========================================================
    # INDIVIDUAL EVIDENCE SCORES
    # ========================================================

    technique_component = technique_score(
        techniques
    )

    ai_component = confidence_score(
        ai_confidence
    )

    wazuh_component = wazuh_score(
        wazuh_level
    )

    vt_component = virustotal_score(
        malicious=vt_malicious,
        suspicious=vt_suspicious,
        total_engines=vt_total_engines
    )

    abuse_component = abuseipdb_score(
        abuseipdb_confidence
    )

    behavior_component = behavioral_score(
        encoded_command=encoded_command,
        download_behavior=download_behavior,
        remote_execution=remote_execution,
        registry_operation=registry_operation,
        file_deletion=file_deletion,
        suspicious_parent=suspicious_parent,
        process_injection=process_injection
    )

    # ========================================================
    # RAW RISK
    # ========================================================

    raw_score = (
        technique_component
        +
        ai_component
        +
        wazuh_component
        +
        vt_component
        +
        abuse_component
        +
        behavior_component
    )

    risk_score = round(
        min(
            100,
            raw_score
        ),
        2
    )

    # ========================================================
    # EVIDENCE CONDITIONS
    # ========================================================

    strong_ai = (
        float(ai_confidence) >= 0.90
    )

    strong_threat_intelligence = (
        vt_malicious >= 3
        or
        abuseipdb_confidence >= 80
    )

    destructive_behavior = (
        file_deletion
        or
        process_injection
        or
        remote_execution
    )

    high_risk_technique = (
        has_high_risk_technique(
            techniques
        )
    )

    # ========================================================
    # INITIAL SEVERITY
    # ========================================================

    if risk_score >= AUTOMATED_THRESHOLD:

        severity = "CRITICAL"
        decision = "AUTOMATED_RESPONSE"

    elif risk_score >= CONTAIN_THRESHOLD:

        severity = "HIGH"
        decision = "CONTAIN"

    elif risk_score >= ALERT_THRESHOLD:

        severity = "MEDIUM"
        decision = "ALERT"

    else:

        severity = "LOW"
        decision = "PASS"

    # ========================================================
    # AUTOMATED RESPONSE SAFETY GATE
    # ========================================================
    #
    # Automated response requires at least TWO independent
    # strong evidence conditions.
    #
    # This prevents:
    #
    #     AI alone
    #
    # from killing/isolation actions.
    #
    # ========================================================

    evidence_conditions = sum([
        strong_ai,
        strong_threat_intelligence,
        destructive_behavior,
        high_risk_technique
    ])

    if decision == "AUTOMATED_RESPONSE":

        if evidence_conditions < 2:

            decision = "CONTAIN"
            severity = "HIGH"

    # ========================================================
    # ADDITIONAL LOW-RISK SAFETY RULE
    # ========================================================
    #
    # A normal administrative technique with:
    #
    #   - low AI confidence
    #   - low Wazuh level
    #   - no TI evidence
    #   - no suspicious behavior
    #
    # should remain LOW.
    #
    # ========================================================

    no_external_intelligence = (
        vt_malicious == 0
        and
        vt_suspicious == 0
        and
        abuseipdb_confidence == 0
    )

    no_suspicious_behavior = not any([
        encoded_command,
        download_behavior,
        remote_execution,
        registry_operation,
        file_deletion,
        suspicious_parent,
        process_injection
    ])

    low_ai = (
        float(ai_confidence) < 0.70
    )

    low_wazuh = (
        float(wazuh_level) <= 5
    )

    if (
        low_ai
        and
        low_wazuh
        and
        no_external_intelligence
        and
        no_suspicious_behavior
    ):

        risk_score = min(
            risk_score,
            20
        )

        severity = "LOW"
        decision = "PASS"

    # ========================================================
    # RESPONSE ACTION
    # ========================================================

    if decision == "PASS":

        action = "MONITOR"

    elif decision == "ALERT":

        action = "CREATE_ALERT"

    elif decision == "CONTAIN":

        action = "ISOLATE_ENDPOINT"

    else:

        action = "AUTOMATED_RESPONSE"

    # ========================================================
    # BUILD RESULT
    # ========================================================

    result = {

        "risk_score":
            risk_score,

        "severity":
            severity,

        "decision":
            decision,

        "recommended_action":
            action,

        "evidence": {

            "techniques":
                techniques or [],

            "ai_confidence":
                round(
                    float(
                        ai_confidence
                    ),
                    4
                ),

            "technique_score":
                technique_component,

            "ai_score":
                ai_component,

            "wazuh_level":
                wazuh_level,

            "wazuh_score":
                wazuh_component,

            "virustotal_score":
                vt_component,

            "abuseipdb_score":
                abuse_component,

            "behavioral_score":
                behavior_component

        },

        "safety": {

            "strong_ai":
                strong_ai,

            "strong_threat_intelligence":
                strong_threat_intelligence,

            "destructive_behavior":
                destructive_behavior,

            "high_risk_technique":
                high_risk_technique,

            "independent_evidence_count":
                evidence_conditions

        }

    }

    return result


# ============================================================
# DEMONSTRATION
# ============================================================

def run_demo():

    print(
        "\n"
        + "=" * 75
    )

    print(
        " SmartSOC Risk & Decision Engine v2"
    )

    print(
        "=" * 75
    )

    test_cases = [

        # ====================================================
        # CASE 1
        # Normal PowerShell
        # ====================================================

        {
            "name":
                "Normal PowerShell administration",

            "techniques":
                ["T1059.001"],

            "ai_confidence":
                0.62,

            "wazuh_level":
                3
        },

        # ====================================================
        # CASE 2
        # Encoded PowerShell
        # ====================================================

        {
            "name":
                "Suspicious encoded PowerShell",

            "techniques":
                [
                    "T1059.001",
                    "T1027"
                ],

            "ai_confidence":
                0.91,

            "wazuh_level":
                9,

            "encoded_command":
                True,

            "download_behavior":
                True
        },

        # ====================================================
        # CASE 3
        # Malicious download
        # ====================================================

        {
            "name":
                "Malicious download from suspicious IP",

            "techniques":
                [
                    "T1105",
                    "T1059.003"
                ],

            "ai_confidence":
                0.95,

            "wazuh_level":
                12,

            "vt_malicious":
                8,

            "vt_suspicious":
                1,

            "vt_total_engines":
                70,

            "abuseipdb_confidence":
                92,

            "download_behavior":
                True,

            "remote_execution":
                True
        },

        # ====================================================
        # CASE 4
        # Ransomware-like behavior
        # ====================================================

        {
            "name":
                "Potential ransomware-like behavior",

            "techniques":
                [
                    "T1485",
                    "T1565.001"
                ],

            "ai_confidence":
                0.97,

            "wazuh_level":
                15,

            "vt_malicious":
                12,

            "vt_suspicious":
                2,

            "vt_total_engines":
                70,

            "abuseipdb_confidence":
                95,

            "file_deletion":
                True,

            "process_injection":
                True,

            "suspicious_parent":
                True
        }
    ]

    # ========================================================
    # RUN TEST CASES
    # ========================================================

    for case in test_cases:

        case = case.copy()

        name = case.pop(
            "name"
        )

        result = calculate_risk(
            **case
        )

        print(
            "\n"
            + "-" * 75
        )

        print(
            "CASE:",
            name
        )

        print(
            "-" * 75
        )

        print(
            json.dumps(
                result,
                indent=4
            )
        )

    print(
        "\n"
        + "=" * 75
    )

    print(
        " Risk Engine Demonstration Complete"
    )

    print(
        "=" * 75
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_demo()