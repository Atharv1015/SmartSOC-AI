# SmartSOC targeted dry-run configuration patch
# Run from any PowerShell location. Creates a timestamped backup before editing.
$ErrorActionPreference = "Stop"
$Project = "C:\Users\athar\SmartSOC-AI"
$Config = Join-Path $Project "src\integration_v2\execution_config_v2.py"

if (-not (Test-Path -LiteralPath $Config)) {
    throw "Cannot find expected configuration file: $Config"
}

if ($env:SMARTSOC_ALLOW_LIVE_MITIGATION -eq "1") {
    throw "Safety stop: SMARTSOC_ALLOW_LIVE_MITIGATION is currently 1 in this shell. Disable it before running tests; this patch does not enable live mitigation."
}

$Content = Get-Content -LiteralPath $Config -Raw
$Pattern = '(?m)^DRY_RUN\s*=\s*False\s*(?=\r?$)'
$Matches = [regex]::Matches($Content, $Pattern)

if ($Matches.Count -eq 0) {
    if ($Content -match '(?m)^DRY_RUN\s*=\s*True\s*$') {
        Write-Host "No change needed: DRY_RUN is already True." -ForegroundColor Green
    } else {
        throw "Expected a single top-level 'DRY_RUN = False' line was not found. No file changed; inspect $Config manually."
    }
} elseif ($Matches.Count -ne 1) {
    throw "Found $($Matches.Count) matching DRY_RUN assignments. No file changed; inspect $Config manually."
} else {
    $Backup = "$Config.bak_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Copy-Item -LiteralPath $Config -Destination $Backup
    $Updated = [regex]::Replace($Content, $Pattern, 'DRY_RUN = True', 1)
    Set-Content -LiteralPath $Config -Value $Updated -Encoding utf8
    Write-Host "Patched DRY_RUN = True" -ForegroundColor Green
    Write-Host "Backup: $Backup"
}

Write-Host "`n=== Verify config ==="
Select-String -LiteralPath $Config -Pattern '^\s*DRY_RUN\s*=|^\s*LIVE_EXECUTION_ENABLED\s*='

Write-Host "`n=== Compile affected Python files ==="
Set-Location $Project
python -m py_compile `
  .\src\integration_v2\execution_config_v2.py `
  .\src\integration_v2\authorized_dry_run_v2.py `
  .\src\autonomous_v2\victim_mitigation_adapter_v2.py
if ($LASTEXITCODE -ne 0) {
    throw "Python compile check failed with exit code $LASTEXITCODE"
}

Write-Host "`n=== Run targeted dry-run tests ==="
python -m pytest -q `
  .\tests\test_approval_bridge_v1.py `
  .\tests\test_authorized_dry_run_v2.py `
  .\tests\test_victim_mitigation_adapter_v2.py
$TestExit = $LASTEXITCODE

Write-Host "`nTargeted test exit code: $TestExit"
Write-Host "Live mitigation was not enabled and the receiver was not started."
Write-Host "If tests still fail, paste the full failure summary; the receiver SQLite tests need a separate targeted fix."
exit $TestExit
