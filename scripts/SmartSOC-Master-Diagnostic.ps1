[CmdletBinding()]
param(
    [ValidateSet("None", "Focused", "All")]
    [string]$Tests = "Focused",

    [string[]]$ExtraPath = @(),

    [switch]$IncludeGitStatus
)

$ErrorActionPreference = "Continue"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ReportRoot = Join-Path $ProjectRoot "reports\diagnostics"
$RunDir = Join-Path $ReportRoot "master_diagnostic_$Stamp"
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null

$BundlePath = Join-Path $RunDir "inspection_bundle.txt"
$SummaryPath = Join-Path $RunDir "diagnostic_summary.txt"
$TestOutputPath = Join-Path $RunDir "test_output.txt"
$ManifestPath = Join-Path $RunDir "manifest.txt"

$script:Bundle = [System.Collections.Generic.List[string]]::new()
$script:Summary = [System.Collections.Generic.List[string]]::new()
$script:Manifest = [System.Collections.Generic.List[string]]::new()

function Add-Section([string]$Title) {
    $script:Bundle.Add("")
    $script:Bundle.Add(("=" * 90))
    $script:Bundle.Add($Title)
    $script:Bundle.Add(("=" * 90))
}

function Protect-Text([string]$Text) {
    # Redact common secret/token/password assignments before saving or displaying file content.
    $patterns = @(
        '(?im)(SMARTSOC_AGENT_SECRET\s*["'']?\s*[:=]\s*["'']?)[^"''\r\n,;}\s]+',
        '(?im)((?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password|passwd)\s*["'']?\s*[:=]\s*["'']?)[^"''\r\n,;}\s]+',
        '(?im)(Authorization\s*:\s*Bearer\s+)[A-Za-z0-9._~+/-]+=*'
    )
    foreach ($pattern in $patterns) {
        $Text = [regex]::Replace($Text, $pattern, '$1[REDACTED]')
    }
    return $Text
}

function Add-FileToBundle([string]$RelativePath) {
    $full = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
        $script:Manifest.Add("MISSING`t$RelativePath")
        return
    }

    try {
        $resolved = (Resolve-Path -LiteralPath $full).Path
        # Only inspect files inside the project root.
        if (-not $resolved.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            $script:Manifest.Add("SKIPPED_OUTSIDE_PROJECT`t$RelativePath")
            return
        }

        $raw = Get-Content -LiteralPath $resolved -Raw -ErrorAction Stop
        $safe = Protect-Text $raw
        Add-Section "FILE: $RelativePath"
        $script:Bundle.Add($safe)
        $script:Manifest.Add("INCLUDED`t$RelativePath`t$((Get-Item -LiteralPath $resolved).Length) bytes")
    }
    catch {
        $script:Manifest.Add("ERROR`t$RelativePath`t$($_.Exception.Message)")
    }
}

$defaultFiles = @(
    "src\integration\smartsoc_live_receiver_v2.py",
    "src\integration\v2_authorized_dry_run_bridge.py",
    "src\integration\local_approval_console_v1.py",
    "src\integration_v2\authorized_dry_run_v2.py",
    "src\integration_v2\execution_config_v2.py",
    "src\integration_v2\approval_bridge_v1.py",
    "src\autonomous_v2\victim_mitigation_adapter_v2.py",
    "src\autonomous_v2\victim_mitigation_client_v2.py",
    "tests\test_local_approval_console_v1.py",
    "tests\test_approval_bridge_v1.py",
    "tests\test_approval_registration_isolation_v1.py",
    "tests\test_receiver_approval_boundary_v1.py",
    "tests\test_receiver_approval_bridge_workflow_v1.py",
    "tests\test_receiver_fail_closed_v1.py",
    "tests\test_authorized_dry_run_v2.py",
    "tests\test_victim_mitigation_adapter_v2.py"
)

$allFiles = @($defaultFiles + $ExtraPath) | Select-Object -Unique

$script:Summary.Add("SmartSOC Master Diagnostic")
$script:Summary.Add("Timestamp: $(Get-Date -Format o)")
$script:Summary.Add("Project root: $ProjectRoot")
$script:Summary.Add("Test mode: $Tests")
$script:Summary.Add("Safety: this script only reads selected files and runs Python compile/tests.")
$script:Summary.Add("It does NOT start/stop services, send victim requests, approve requests, or execute mitigation.")
$script:Summary.Add("")

Add-Section "RUNTIME / ENVIRONMENT"
$script:Bundle.Add("Timestamp: $(Get-Date -Format o)")
$script:Bundle.Add("Project root: $ProjectRoot")
$script:Bundle.Add("PowerShell: $($PSVersionTable.PSVersion)")
$script:Bundle.Add("Computer: $env:COMPUTERNAME")
$script:Bundle.Add("SMARTSOC_AGENT_SECRET present: $([bool]($env:SMARTSOC_AGENT_SECRET -and $env:SMARTSOC_AGENT_SECRET.Length -ge 32))")
$script:Bundle.Add("Secret value intentionally not collected.")
$script:Manifest.Add("ROOT`t$ProjectRoot")

$pythonCandidates = @(
    (Join-Path $ProjectRoot ".venv-compat\Scripts\python.exe"),
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe")
)
$PythonExe = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $PythonExe) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $PythonExe = $cmd.Source }
}
if ($PythonExe) {
    $script:Bundle.Add("Python executable: $PythonExe")
    try {
        $version = & $PythonExe --version 2>&1
        $script:Bundle.Add("Python version: $version")
    } catch {
        $script:Bundle.Add("Python version check failed: $($_.Exception.Message)")
    }
} else {
    $script:Bundle.Add("Python executable: NOT FOUND")
    $script:Summary.Add("WARNING: Python was not found; compile/tests skipped.")
}

if ($IncludeGitStatus) {
    Add-Section "GIT STATUS"
    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($git) {
        $gitText = & git -C $ProjectRoot status --short --branch 2>&1 | Out-String
        $script:Bundle.Add((Protect-Text $gitText))
    } else {
        $script:Bundle.Add("Git executable not found.")
    }
}

Add-Section "SAFETY CONFIGURATION SNAPSHOT"
$policyPath = Join-Path $ProjectRoot "src\integration_v2\execution_config_v2.py"
if (Test-Path -LiteralPath $policyPath) {
    Select-String -LiteralPath $policyPath -Pattern '^\s*(CONTROLLED_LAB_MODE|EXECUTION_LAYER_ENABLED|AUTOMATIC_EXECUTION|AUTOMATIC_CONTAINMENT|UNRESTRICTED_AUTONOMOUS_ACTIONS|HUMAN_AUTHORIZATION_REQUIRED|ALLOWED_ACTIONS|ALLOWED_PROCESSES|DRY_RUN)\s*=' |
        ForEach-Object { $script:Bundle.Add($_.Line) }
} else {
    $script:Bundle.Add("Policy file missing.")
}

Add-Section "SELECTED SOURCE AND TEST FILES"
foreach ($file in $allFiles) {
    Add-FileToBundle $file
}

$compileTargets = @(
    "src\integration\smartsoc_live_receiver_v2.py",
    "src\integration\v2_authorized_dry_run_bridge.py",
    "src\integration\local_approval_console_v1.py",
    "src\integration_v2\authorized_dry_run_v2.py",
    "src\integration_v2\execution_config_v2.py",
    "src\integration_v2\approval_bridge_v1.py",
    "src\autonomous_v2\victim_mitigation_adapter_v2.py",
    "src\autonomous_v2\victim_mitigation_client_v2.py"
)
$existingCompileTargets = @($compileTargets | Where-Object { Test-Path -LiteralPath (Join-Path $ProjectRoot $_) -PathType Leaf })

$testLines = [System.Collections.Generic.List[string]]::new()
$testLines.Add("SmartSOC diagnostic test run: $(Get-Date -Format o)")
$testLines.Add("Test mode: $Tests")
$testLines.Add("No receiver/agent services are started by this script.")
$testLines.Add("")

if ($PythonExe -and $Tests -ne "None") {
    if ($existingCompileTargets.Count -gt 0) {
        $testLines.Add("=== PY_COMPILE ===")
        $compileArgs = @("-m", "py_compile") + @($existingCompileTargets | ForEach-Object { Join-Path $ProjectRoot $_ })
        $compileOutput = & $PythonExe @compileArgs 2>&1
        $compileExit = $LASTEXITCODE
        $testLines.Add(($compileOutput | Out-String))
        $testLines.Add("py_compile exit code: $compileExit")
        $script:Summary.Add("py_compile exit code: $compileExit")
    } else {
        $testLines.Add("No compile targets found.")
    }

    if ($Tests -eq "Focused") {
        $testTargets = @(
            "tests\test_local_approval_console_v1.py",
            "tests\test_approval_bridge_v1.py",
            "tests\test_approval_registration_isolation_v1.py",
            "tests\test_receiver_approval_boundary_v1.py",
            "tests\test_receiver_approval_bridge_workflow_v1.py",
            "tests\test_receiver_fail_closed_v1.py",
            "tests\test_authorized_dry_run_v2.py",
            "tests\test_victim_mitigation_adapter_v2.py"
        )
        $testTargets = @($testTargets | Where-Object { Test-Path -LiteralPath (Join-Path $ProjectRoot $_) -PathType Leaf })
        $testArgs = @("-m", "pytest", "-q") + @($testTargets | ForEach-Object { Join-Path $ProjectRoot $_ })
    } else {
        $testArgs = @("-m", "pytest", "-q", (Join-Path $ProjectRoot "tests"))
    }

    $testLines.Add("")
    $testLines.Add("=== PYTEST ===")
    $testLines.Add("Command: $PythonExe $($testArgs -join ' ')")
    $pytestOutput = & $PythonExe @testArgs 2>&1
    $pytestExit = $LASTEXITCODE
    $testLines.Add((Protect-Text ($pytestOutput | Out-String)))
    $testLines.Add("pytest exit code: $pytestExit")
    $script:Summary.Add("pytest exit code: $pytestExit")
} elseif ($Tests -eq "None") {
    $testLines.Add("Tests intentionally skipped (-Tests None).")
} else {
    $testLines.Add("Tests skipped because Python was not found.")
}

$testLines | Set-Content -LiteralPath $TestOutputPath -Encoding UTF8
$script:Bundle.Add("")
$script:Bundle.Add("=== TEST OUTPUT SUMMARY ===")
$script:Bundle.Add((Get-Content -LiteralPath $TestOutputPath -Raw))

$script:Bundle | Set-Content -LiteralPath $BundlePath -Encoding UTF8
$script:Manifest | Set-Content -LiteralPath $ManifestPath -Encoding UTF8
$script:Summary.Add("")
$script:Summary.Add("Output folder: $RunDir")
$script:Summary.Add("Bundle: $BundlePath")
$script:Summary.Add("Test output: $TestOutputPath")
$script:Summary.Add("Manifest: $ManifestPath")
$script:Summary.Add("Review/redact the bundle before sharing it externally.")
$script:Summary | Set-Content -LiteralPath $SummaryPath -Encoding UTF8

$zipPath = Join-Path $ReportRoot "master_diagnostic_$Stamp.zip"
try {
    Compress-Archive -Path (Join-Path $RunDir "*") -DestinationPath $zipPath -Force
    Add-Content -LiteralPath $SummaryPath -Value "ZIP bundle: $zipPath"
} catch {
    Add-Content -LiteralPath $SummaryPath -Value "ZIP creation failed: $($_.Exception.Message)"
}

Write-Host ""
Write-Host "SmartSOC master diagnostic finished." -ForegroundColor Cyan
Write-Host "Report folder: $RunDir"
Write-Host "Bundle:        $BundlePath"
Write-Host "Test output:   $TestOutputPath"
Write-Host "ZIP:           $zipPath"
Write-Host ""
Write-Host "No mitigation or containment action was executed." -ForegroundColor Yellow
Write-Host "Inspect/redact generated files before sharing them."
