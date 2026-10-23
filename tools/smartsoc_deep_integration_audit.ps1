# SmartSOC Deep Integration Audit (read-only)
# Purpose: collect the key integration, policy, authorization, client, and report evidence
# in one run. This script does NOT import project modules, call endpoints/webhooks,
# start/stop services, change configuration, or execute mitigation.

$ErrorActionPreference = "Continue"
$ProjectRoot = "C:\Users\athar\SmartSOC-AI"

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    Write-Error "Project root not found: $ProjectRoot"
    exit 1
}

Set-Location -LiteralPath $ProjectRoot

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$AuditDir = Join-Path $ProjectRoot "reports\audit"
New-Item -ItemType Directory -Path $AuditDir -Force | Out-Null
$ReportPath = Join-Path $AuditDir "smartsoc_deep_integration_audit_$Timestamp.txt"

$Report = [System.Collections.Generic.List[string]]::new()

function Add-Line {
    param([AllowEmptyString()][string]$Text = "")
    $script:Report.Add($Text)
}

function Add-Section {
    param([string]$Title)
    Add-Line ""
    Add-Line ("=" * 100)
    Add-Line $Title
    Add-Line ("=" * 100)
}

function Add-FileMetadata {
    param([string]$RelativePath)
    $full = Join-Path $ProjectRoot $RelativePath
    Add-Line ""
    Add-Line "--- FILE: $RelativePath ---"
    if (Test-Path -LiteralPath $full -PathType Leaf) {
        $item = Get-Item -LiteralPath $full
        Add-Line ("Exists: YES | Bytes: {0} | LastWriteTime: {1}" -f $item.Length, $item.LastWriteTime.ToString("s"))
    } else {
        Add-Line "Exists: NO"
    }
}

function Add-SourceEvidence {
    param(
        [string]$RelativePath,
        [string[]]$Patterns,
        [int]$MaxMatches = 100
    )
    $full = Join-Path $ProjectRoot $RelativePath
    Add-FileMetadata $RelativePath
    if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
        return
    }

    # Read as text only. Do not dot-source/import/execute project files.
    $content = Get-Content -LiteralPath $full -ErrorAction SilentlyContinue
    if ($null -eq $content) {
        Add-Line "Could not read file."
        return
    }

    $matches = Select-String -LiteralPath $full -Pattern $Patterns -ErrorAction SilentlyContinue |
        Select-Object -First $MaxMatches

    if (-not $matches) {
        Add-Line "No matching evidence lines found for the configured patterns."
    } else {
        foreach ($m in $matches) {
            $line = $m.Line.Trim()
            # Redact common secret/token assignments in displayed evidence.
            if ($line -match '(?i)(secret|token|password|api[_-]?key|webhook)' -and
                $line -match '(:|=)' -and
                $line -notmatch '(?i)(required|missing|invalid|placeholder|change-me|environment|getenv|os\.environ|process\.env)') {
                $line = [regex]::Replace($line, '(?i)(secret|token|password|api[_-]?key|webhook)(\s*["'']?\s*[:=]\s*["'']?)[^,"''\s)}]+', '$1$2[REDACTED]')
            }
            Add-Line ("{0,5}: {1}" -f $m.LineNumber, $line)
        }
    }
}

Add-Section "SMARTSOC DEEP INTEGRATION AUDIT"
Add-Line "Generated: $(Get-Date -Format o)"
Add-Line "Project root: $ProjectRoot"
Add-Line "Mode: READ-ONLY STATIC INSPECTION"
Add-Line "No project Python module was imported or executed by this script."
Add-Line "No endpoint, webhook, victim agent, or mitigation function was called."
Add-Line "No source/config files were modified."

Add-Section "1. ENVIRONMENT AND PROJECT STATUS"
Add-Line ("PowerShell: {0}" -f $PSVersionTable.PSVersion)
try {
    $pyVersion = & python --version 2>&1
    Add-Line ("Python: {0}" -f ($pyVersion -join " "))
} catch {
    Add-Line "Python version command unavailable."
}
try {
    $gitResult = & git -C $ProjectRoot rev-parse --show-toplevel 2>&1
    if ($LASTEXITCODE -eq 0) { Add-Line ("Git root: " + ($gitResult -join " ")) }
    else { Add-Line "Git repository: NOT DETECTED (this is informational; audit continues)." }
} catch {
    Add-Line "Git command unavailable."
}

Add-Section "2. RECEIVER: IMPORTS, FLAGS, ACTION SELECTION, GATES, RESULTS"
Add-SourceEvidence ".\src\integration\smartsoc_live_receiver_v2.py" @(
    '^(from |import )',
    '^(LISTEN_HOST|LISTEN_PORT|EXECUTION_MODE|MITIGATION_ENABLED|VICTIM_AGENT_CALLED)\s*=',
    'mitigation_controller\s*=',
    'requested_mitigation\s*=',
    'execute_dry_run',
    'validate_approval',
    'risk_engine\.calculate',
    'decision_policy\.evaluate',
    'response_planner\.evaluate',
    'if not MITIGATION_ENABLED',
    'mitigation_controller\.execute',
    'victim_agent_called',
    'effective_action\s*=',
    '"controller_invoked"',
    '"success"',
    '"verified"'
) 160

Add-Section "3. V1 EXECUTION POLICY: REAL SETTINGS AND INVARIANTS"
Add-SourceEvidence ".\src\integration\execution_config_v1.py" @(
    '^(EXECUTION_LAYER_ENABLED|AUTOMATIC_EXECUTION|AUTOMATIC_CONTAINMENT|UNRESTRICTED_AUTONOMOUS_ACTIONS|CONTROLLED_LAB_MODE|HUMAN_AUTHORIZATION_REQUIRED|ALLOWED_ACTIONS|DENY_UNKNOWN_ACTIONS|ENDPOINT_MODIFICATION_REQUIRES_ALLOWLIST)\s*=',
    '^def ',
    'if ',
    'validate_safety_invariants'
) 140

Add-Section "4. V2 EXECUTION POLICY: REAL SETTINGS AND INVARIANTS"
Add-SourceEvidence ".\src\integration_v2\execution_config_v2.py" @(
    '^(CONTROLLED_LAB_MODE|EXECUTION_LAYER_ENABLED|AUTOMATIC_EXECUTION|AUTOMATIC_CONTAINMENT|UNRESTRICTED_AUTONOMOUS_ACTIONS|HUMAN_AUTHORIZATION_REQUIRED|ALLOWED_ACTIONS|ALLOWED_PROCESSES|DRY_RUN)\s*=',
    '^def ',
    'if ',
    'validate_safety_invariants'
) 100

Add-Section "5. V1 CONTROLLER: POLICY GATES, VALIDATION, CLIENT CALL, RESULT"
Add-SourceEvidence ".\src\autonomous\autonomous_mitigation_controller_v1.py" @(
    '^(from |import )',
    '^class ',
    '^    def ',
    'validate_execution_policy',
    'validate_reasoning_result',
    'validate_safety_invariants',
    'EXECUTION_LAYER_ENABLED',
    'AUTOMATIC_EXECUTION',
    'AUTOMATIC_CONTAINMENT',
    'HUMAN_AUTHORIZATION_REQUIRED',
    'victim_client\.execute',
    'victim_client_invoked',
    'PROCESS_TERMINATION',
    'authorization_validator',
    'validate_approval'
) 180

Add-Section "6. V1 VICTIM CLIENT: TRANSPORT AND RESPONSE HANDLING (STATIC SEARCH)"
Add-SourceEvidence ".\src\autonomous\victim_mitigation_client_v1.py" @(
    '^(from |import )',
    '^class ',
    '^    def ',
    'HTTPConnection',
    'HTTPSConnection',
    'request\(',
    'PROCESS_TERMINATION',
    'token',
    'authorization',
    'success',
    'verified',
    'response\.status',
    'victim_response'
) 160

Add-Section "7. V2 ADAPTER: DRY-RUN POLICY AND NETWORK CALL SITE (STATIC SEARCH)"
Add-SourceEvidence ".\src\autonomous_v2\victim_mitigation_adapter_v2.py" @(
    '^(from |import )',
    '^def ',
    'policy\.DRY_RUN',
    'EXECUTION_LAYER_ENABLED',
    'AUTOMATIC_EXECUTION',
    'AUTOMATIC_CONTAINMENT',
    'HUMAN_AUTHORIZATION_REQUIRED',
    'is_action_allowed',
    'is_process_allowed',
    'SMARTSOC_AGENT_SECRET',
    'HTTPConnection',
    'request\(',
    'requested_mode',
    'victim_request_attempted',
    'victim_http_response_received'
) 150

Add-Section "8. AUTHORIZATION VALIDATOR: DEFINITIONS AND RECEIVER INTEGRATION EVIDENCE"
Add-SourceEvidence ".\src\integration\authorization_validator_v2.py" @(
    '^(from |import )',
    '^def ',
    'validate_approval',
    'analyst_authorized',
    'authorization_actor',
    'authorization_timestamp',
    'alert_id',
    'rule_id',
    'timezone',
    'timedelta'
) 100
Add-Line ""
Add-Line "Receiver references to authorization validation:"
$receiverPath = Join-Path $ProjectRoot ".\src\integration\smartsoc_live_receiver_v2.py"
if (Test-Path -LiteralPath $receiverPath) {
    $authRefs = Select-String -LiteralPath $receiverPath -Pattern 'authorization_validator_v2|validate_approval|authorization_validator' -SimpleMatch
    if ($authRefs) {
        foreach ($m in $authRefs) { Add-Line ("{0,5}: {1}" -f $m.LineNumber, $m.Line.Trim()) }
    } else {
        Add-Line "No direct literal references to authorization_validator_v2 / validate_approval found in receiver."
    }
}

Add-Section "9. V4.1 RISK, DECISION POLICY, RESPONSE PLANNER"
Add-SourceEvidence ".\src\decision\smartsoc_risk_engine_v4_1.py" @(
    '^class ',
    '^def ',
    'calculate_risk_level',
    'risk_score',
    'LOW',
    'MEDIUM',
    'HIGH',
    'CRITICAL',
    'automatic',
    'containment'
) 100
Add-SourceEvidence ".\src\decision\smartsoc_decision_policy_v4_1.py" @(
    '^class ',
    '^def ',
    'containment_authorized',
    'analyst',
    'fallback',
    'ambiguous',
    'automatic',
    'self_test'
) 100
Add-SourceEvidence ".\src\decision\smartsoc_response_planner_v4_1.py" @(
    '^class ',
    '^def ',
    'ACTION_CATALOG',
    'requires_approval',
    'automatic',
    'containment_authorized',
    'PREPARE_HOST_ISOLATION',
    'self_test'
) 120

Add-Section "10. LEGACY AUTHORIZATION AND EXECUTION GUARD"
Add-SourceEvidence ".\src\integration\authorization_gate_v1.py" @(
    '^(from |import )',
    '^def ',
    'HUMAN_AUTHORIZATION_REQUIRED',
    'authorization_required',
    'authorization_status',
    'execution_allowed',
    'analyst_authorized'
) 100
Add-SourceEvidence ".\src\integration\authorization_decision_v1.py" @(
    '^(from |import )',
    '^def ',
    'APPROVE',
    'DENY',
    'LAB_ANALYST',
    'analyst_authorized',
    'authorization_status',
    'execution_allowed'
) 100
Add-SourceEvidence ".\src\integration\execution_policy_guard_v1.py" @(
    '^(from |import )',
    '^def ',
    'HUMAN_AUTHORIZATION_REQUIRED',
    'authorization_required',
    'authorization_status',
    'authorization_decision',
    'execution_allowed',
    'AUTOMATIC_EXECUTION',
    'AUTOMATIC_CONTAINMENT'
) 120

Add-Section "11. TEST INVENTORY AND SAFE BASELINE ARTIFACTS"
$testRoot = Join-Path $ProjectRoot "tests"
if (Test-Path -LiteralPath $testRoot) {
    $testFiles = Get-ChildItem -LiteralPath $testRoot -Recurse -File -Filter "test_*.py" -ErrorAction SilentlyContinue
    Add-Line ("Test files found: {0}" -f @($testFiles).Count)
    foreach ($f in $testFiles) {
        Add-Line ("{0} | {1} bytes | {2}" -f $f.FullName.Substring($ProjectRoot.Length + 1), $f.Length, $f.LastWriteTime.ToString("s"))
    }
} else { Add-Line "tests directory not found." }

Add-Line ""
Add-Line "Selected integration JSON artifact inventory (metadata only; JSON files are NOT executed):"
$artifactRoot = Join-Path $ProjectRoot "reports\integration"
if (Test-Path -LiteralPath $artifactRoot) {
    Get-ChildItem -LiteralPath $artifactRoot -File -Filter "*.json" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 60 |
        ForEach-Object {
            Add-Line ("{0} | {1} bytes | {2}" -f $_.Name, $_.Length, $_.LastWriteTime.ToString("s"))
        }
} else { Add-Line "reports\integration directory not found." }

Add-Section "12. STALE/CONFLICTING ARTIFACT CHECK (SELECTED JSON FIELDS)"
$selectedJson = @(
    ".\reports\integration\response_plan_v1.json",
    ".\reports\integration\authorization_gate_result_v1.json",
    ".\reports\integration\authorization_decision_v1.json",
    ".\reports\integration\execution_policy_guard_result_v1.json",
    ".\reports\integration\risk_engine_result.json",
    ".\reports\integration\shuffle_dry_run_result_v1.json"
)
foreach ($rel in $selectedJson) {
    $full = Join-Path $ProjectRoot $rel
    Add-Line ""
    Add-Line "--- $rel ---"
    if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
        Add-Line "Missing"
        continue
    }
    try {
        $jsonText = Get-Content -LiteralPath $full -Raw
        $obj = $jsonText | ConvertFrom-Json -ErrorAction Stop
        Add-Line ("LastWriteTime: " + (Get-Item -LiteralPath $full).LastWriteTime.ToString("s"))
        foreach ($prop in @("timestamp","event_id","alert_id","rule_id","status","risk_score","severity","authorization_required","human_authorization_required","analyst_authorized","execution_allowed","automatic_execution","automatic_containment","execution_mode","mode","http_status")) {
            $p = $obj.PSObject.Properties[$prop]
            if ($null -ne $p) {
                $val = [string]$p.Value
                if ($val.Length -gt 180) { $val = $val.Substring(0,180) + "..." }
                Add-Line ("{0}: {1}" -f $prop, $val)
            }
        }
        foreach ($nestedName in @("authorization","execution","policy","safety")) {
            $nested = $obj.PSObject.Properties[$nestedName]
            if ($null -ne $nested -and $null -ne $nested.Value) {
                foreach ($nestedProp in @("authorization_required","human_authorization_required","analyst_authorized","execution_allowed","automatic_execution","automatic_containment","status","decision")) {
                    $np = $nested.Value.PSObject.Properties[$nestedProp]
                    if ($null -ne $np) {
                        Add-Line ("{0}.{1}: {2}" -f $nestedName, $nestedProp, [string]$np.Value)
                    }
                }
            }
        }
    } catch {
        Add-Line ("JSON parse error: " + $_.Exception.Message)
    }
}

Add-Section "13. CONFIG / SECRET HYGIENE (NAMES AND PLACEHOLDER SIGNALS ONLY)"
$scanRoots = @(".\src\integration", ".\src\integration_v2", ".\src\autonomous", ".\src\autonomous_v2")
foreach ($root in $scanRoots) {
    $fullRoot = Join-Path $ProjectRoot $root
    if (-not (Test-Path -LiteralPath $fullRoot)) { continue }
    Get-ChildItem -LiteralPath $fullRoot -Recurse -File -Include "*.py","*.json","*.env","*.yaml","*.yml" -ErrorAction SilentlyContinue |
        Select-String -Pattern '(?i)CHANGE-ME|placeholder|SMARTSOC_AGENT_SECRET|victim_token\s*=|webhook' -ErrorAction SilentlyContinue |
        Select-Object -First 80 |
        ForEach-Object {
            $line = $_.Line.Trim()
            $line = [regex]::Replace($line, '(?i)(victim_token\s*=\s*)["''][^"'']*["'']', '$1"[REDACTED_OR_PLACEHOLDER]"')
            $line = [regex]::Replace($line, '(?i)(webhook.{0,30}[:=]\s*)["''][^"'']*["'']', '$1"[REDACTED]"')
            Add-Line ("{0}: {1}" -f $_.Path.Substring($ProjectRoot.Length + 1), $line)
        }
}

Add-Section "14. FINAL AUDIT NOTES"
Add-Line "This report is static evidence, not proof of end-to-end runtime behavior."
Add-Line "Imports/call sites are searched as source text; this script does not import project modules."
Add-Line "No endpoint or webhook requests were made by this audit."
Add-Line "Keep MITIGATION_ENABLED=False, execution disabled, automatic containment disabled, and V2 DRY_RUN=True while validating."
Add-Line "V2 DRY_RUN adapter may send an HTTP request for endpoint-side validation when invoked; do not invoke it for a network-free audit."
Add-Line "Review timestamps before trusting reports\integration JSON files; old artifacts may contradict current policy."
Add-Line "Recommended next stage: write local-only unit/integration tests with mocked clients and explicit assertions that no network client is invoked when policy denies."

$Report | Set-Content -LiteralPath $ReportPath -Encoding UTF8

Write-Host ""
Write-Host "SmartSOC Deep Integration Audit complete." -ForegroundColor Green
Write-Host ("Report: {0}" -f $ReportPath) -ForegroundColor Cyan
Write-Host ("Report lines: {0}" -f $Report.Count)
Write-Host "Read-only static inspection only; no endpoints/webhooks were contacted."
Write-Host ""
Write-Host "Preview of key sections:"
$Report | Select-String -Pattern "SMARTSOC DEEP INTEGRATION AUDIT|RECEIVER:|V1 EXECUTION POLICY|V2 EXECUTION POLICY|V1 CONTROLLER:|V1 VICTIM CLIENT:|V2 ADAPTER:|AUTHORIZATION VALIDATOR:|FINAL AUDIT NOTES|Exists: NO|No direct literal references" |
    Select-Object -First 80 |
    ForEach-Object { $_.Line }
