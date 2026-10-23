# SmartSOC Scenario Suite Runner
# Run from any PowerShell location:
#   .\Run-SmartSOC-ScenarioSuite.ps1
#
# This runner intentionally executes only the maintained tests/ suite.
# It excludes root-level forensic scripts and archived snapshots from pytest
# discovery because the supplied preflight showed import-time failures there.
# It does not start the receiver, send network requests, or invoke live mitigation.

$ErrorActionPreference = "Stop"

$Root = "C:\Users\athar\SmartSOC-AI"
if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
    throw "SmartSOC repository not found at: $Root. Edit `$Root at the top of this script."
}

Set-Location -LiteralPath $Root

$TestsDir = Join-Path $Root "tests"
if (-not (Test-Path -LiteralPath $TestsDir -PathType Container)) {
    throw "Maintained tests directory not found: $TestsDir"
}

$Receiver = Join-Path $Root "src\integration\smartsoc_live_receiver_v2.py"
if (-not (Test-Path -LiteralPath $Receiver -PathType Leaf)) {
    throw "Receiver file not found: $Receiver"
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ReportDir = Join-Path $Root "reports\diagnostics\scenario_suite_$Stamp"
New-Item -ItemType Directory -Path $ReportDir -Force | Out-Null
$TranscriptPath = Join-Path $ReportDir "scenario_suite_output.txt"
$ResultsPath = Join-Path $ReportDir "scenario_suite_summary.json"

Write-Host "`n=== SMARTSOC SCENARIO SUITE ===" -ForegroundColor Cyan
Write-Host "Repository : $Root"
Write-Host "Test scope : $TestsDir"
Write-Host "Report dir : $ReportDir"
Write-Host "Safety     : no receiver start; no live mitigation request"

# Make this runner fail closed if the shell has live execution armed.
if ($env:SMARTSOC_ALLOW_LIVE_MITIGATION -eq "1") {
    Write-Warning "SMARTSOC_ALLOW_LIVE_MITIGATION=1 in this shell."
    Write-Warning "The test suite is not started. Disarm this shell and rerun."
    throw "Refusing to run while the shell live-mitigation gate is armed."
}

Write-Host "`n[1/3] Compile receiver..."
python -m py_compile $Receiver
$CompileExit = $LASTEXITCODE
if ($CompileExit -ne 0) {
    throw "Receiver compilation failed with exit code $CompileExit"
}
Write-Host "PASS: receiver compiles" -ForegroundColor Green

Write-Host "`n[2/3] Collect maintained tests..."
$CollectOutput = & python -m pytest $TestsDir --collect-only -q 2>&1
$CollectExit = $LASTEXITCODE
$CollectOutput | Tee-Object -FilePath (Join-Path $ReportDir "collection.txt")
if ($CollectExit -ne 0) {
    throw "Test collection failed. Review $ReportDir\collection.txt"
}

Write-Host "`n[3/3] Execute maintained tests..."
$RunOutput = & python -m pytest $TestsDir -q 2>&1
$RunExit = $LASTEXITCODE
$RunOutput | Tee-Object -FilePath $TranscriptPath

$Summary = [PSCustomObject]@{
    timestamp = (Get-Date).ToString("o")
    repository = $Root
    test_scope = $TestsDir
    receiver_compile_exit_code = $CompileExit
    collection_exit_code = $CollectExit
    test_exit_code = $RunExit
    result = if ($RunExit -eq 0) { "PASS" } else { "FAIL" }
    live_mitigation_requested = $false
    notes = "Tests under tests/ only. Root-level forensic scripts and archived snapshots were excluded from discovery."
    output_file = $TranscriptPath
}
$Summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ResultsPath -Encoding UTF8

Write-Host "`n=== SUMMARY ===" -ForegroundColor Cyan
$Summary | Format-List

if ($RunExit -ne 0) {
    Write-Host "Scenario suite has failing tests. Use the first failure in the output to guide the next targeted patch." -ForegroundColor Yellow
    exit $RunExit
}

Write-Host "ALL MAINTAINED TESTS PASSED." -ForegroundColor Green
Write-Host "This validates the test suite, not live endpoint mitigation."
exit 0
