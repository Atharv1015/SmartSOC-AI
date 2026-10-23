
$ErrorActionPreference = "Continue"
$Root = (Get-Location).Path
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Report = Join-Path $Root "reports\integration\MASTER_AUDIT_$Stamp.txt"

New-Item -ItemType Directory -Force -Path (Split-Path $Report) | Out-Null
"SMARTSOC MASTER AUDIT" | Set-Content -LiteralPath $Report
"Generated: $(Get-Date -Format o)" | Add-Content $Report
"Root: $Root" | Add-Content $Report

function Section([string]$Name) {
    "`r`n" + ("=" * 90) + "`r`n$Name`r`n" + ("=" * 90) |
        Add-Content -LiteralPath $Report
}

function Scan-Files([string]$Path, [string[]]$Patterns) {
    if (-not (Test-Path -LiteralPath $Path)) {
        "MISSING PATH: $Path" | Add-Content $Report
        return
    }

    Get-ChildItem -LiteralPath $Path -Recurse -File -Filter "*.py" |
        Where-Object { $_.FullName -notmatch '\\(\.venv[^\\]*|venv|__pycache__|site-packages)\\' } |
        Select-String -Pattern $Patterns -ErrorAction SilentlyContinue |
        ForEach-Object {
            $line = $_.Line.Trim()
            if ($line.Length -gt 350) { $line = $line.Substring(0,350) + "..." }
            "$($_.Path.Replace($Root, '.')) :$($_.LineNumber): $line" |
                Add-Content -LiteralPath $Report
        }
}

Section "1. PROJECT FILE INVENTORY"
Get-ChildItem -LiteralPath $Root -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object {
        $_.FullName -notmatch '\\(\.git|\.venv[^\\]*|venv|__pycache__|site-packages)\\' -and
        ($_.Extension -in ".py",".json",".yaml",".yml",".toml",".ini",".ps1",".md" -or
         $_.Name -in "pytest.ini","pyproject.toml","requirements.txt")
    } |
    ForEach-Object {
        $_.FullName.Replace($Root, ".")
    } | Add-Content $Report

Section "2. EXECUTION POLICY, AUTHORIZATION AND DISPATCH"
$PolicyPatterns = @(
    'EXECUTION_LAYER_ENABLED',
    'AUTOMATIC_EXECUTION',
    'AUTOMATIC_CONTAINMENT',
    'CONTROLLED_LAB_MODE',
    'HUMAN_AUTHORIZATION_REQUIRED',
    'UNRESTRICTED_AUTONOMOUS_ACTIONS',
    'DRY_RUN',
    'MITIGATION_ENABLED',
    'ALLOWED_ACTIONS',
    'ALLOWED_PROCESSES',
    'execution_allowed',
    'analyst_authorized',
    'response_policy_authorization',
    'authorization_policy_satisfied',
    'authorization_decision',
    'execution_policy_guard',
    'register_pending_from_reasoning_result',
    'consume_approval',
    'execute_authorized_client_call',
    'validate_execution_policy',
    'mitigation_controller.execute',
    'victim_client_invoked',
    'endpoint_modified'
)
Scan-Files (Join-Path $Root "src") $PolicyPatterns

Section "3. TESTS RELATED TO AUTHORIZATION AND EXECUTION"
Scan-Files (Join-Path $Root "tests") @(
    'authorization',
    'execution_policy',
    'MITIGATION_BLOCKED',
    'AUTHORIZATION_BLOCKED',
    'NO_MATCHING_APPROVAL',
    'victim_client',
    'endpoint_modified',
    'MITIGATION_ENABLED',
    'DENIED',
    'DRY_RUN'
)

Section "4. INTEGRATION ARTIFACT SUMMARIES"
$ArtifactDir = Join-Path $Root "reports\integration"
$ArtifactNames = @(
    "authorization_decision_v1.json",
    "authorization_gate_result_v1.json",
    "execution_policy_guard_result_v1.json",
    "response_plan_v1.json",
    "execution_policy_guard_result.json",
    "authorization_decision.json"
)

foreach ($Name in $ArtifactNames) {
    $Path = Join-Path $ArtifactDir $Name
    if (Test-Path -LiteralPath $Path) {
        "`r`n--- $Name ---" | Add-Content $Report
        try {
            $Obj = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
            $Obj | ConvertTo-Json -Depth 12 |
                Add-Content -LiteralPath $Report
        } catch {
            "Could not parse JSON: $($_.Exception.Message)" |
                Add-Content $Report
        }
    }
}

Section "5. LIVE RECEIVER AND ENDPOINT CONFIGURATION REFERENCES"
$Receiver = Join-Path $Root "src\integration\smartsoc_live_receiver_v2.py"
if (Test-Path -LiteralPath $Receiver) {
    Select-String -LiteralPath $Receiver -Pattern `
        '^MITIGATION_ENABLED\s*=',
        'def .*health',
        'execution_layer_enabled',
        'allowed_mitigations',
        'register_pending_from_reasoning_result',
        'mitigation_controller.execute',
        'execution_policy_guard',
        'authorization_decision',
        'victim_agent_called',
        'effective_action' |
        ForEach-Object {
            "$($_.Path.Replace($Root,'.')):$($_.LineNumber): $($_.Line.Trim())"
        } | Add-Content $Report
} else {
    "Live receiver file not found at expected path." | Add-Content $Report
}

$EndpointPatterns = @(
    'HOST\s*=',
    'PORT\s*=',
    'ALLOWED_CLIENTS',
    'ALLOWED_ACTIONS',
    'ALLOWED_PROCESSES',
    'SMARTSOC_ALLOW_TERMINATION',
    'execution_enabled',
    'DRY_RUN',
    'TerminateProcess',
    'PROCESS_TERMINATION'
)
Scan-Files (Join-Path $Root "src") $EndpointPatterns

Section "6. PYTHON AND TEST ENVIRONMENT"
Get-Command python -ErrorAction SilentlyContinue |
    Select-Object Source, Version | Format-List | Out-String |
    Add-Content $Report

$VenvPython = Join-Path $Root ".venv-compat\Scripts\python.exe"
if (Test-Path -LiteralPath $VenvPython) {
    "Project Python: $VenvPython" | Add-Content $Report
    & $VenvPython --version 2>&1 | Out-String | Add-Content $Report

    "`r`n--- Pytest collection summary ---" | Add-Content $Report
    & $VenvPython -m pytest --collect-only -q 2>&1 |
        Select-Object -Last 35 | Out-String |
        Add-Content $Report
} else {
    "Project virtual environment not found: $VenvPython" |
        Add-Content $Report
}

Section "7. GIT CHANGE SUMMARY"
if (Get-Command git -ErrorAction SilentlyContinue) {
    git status --short 2>&1 | Out-String | Add-Content $Report
    git diff --stat 2>&1 | Out-String | Add-Content $Report
} else {
    "Git is not installed or not on PATH." | Add-Content $Report
}

Section "8. AUDIT COMPLETION"
"Report: $Report" | Add-Content $Report
"Audit completed: $(Get-Date -Format o)" | Add-Content $Report

Write-Host "`nMASTER AUDIT COMPLETE" -ForegroundColor Green
Write-Host "Report saved to: $Report" -ForegroundColor Cyan
Write-Host "`nLatest report files:"
Get-Item -LiteralPath $Report | Select-Object FullName, Length, LastWriteTime
Write-Host "`nThe script only gathers information; it does not modify project source files or enable endpoint execution."
