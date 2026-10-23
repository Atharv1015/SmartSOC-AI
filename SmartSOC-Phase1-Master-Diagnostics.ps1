#requires -Version 5.1
<#
SmartSOC Phase 1 Master Diagnostics
- Read-only by default.
- Probes known host/victim TCP ports and common read-only health routes.
- Collects source/config/test evidence without printing secret key material.
- Runs the maintained pytest suite when available.
- DOES NOT invoke live mitigation or send a process-termination request.
  The exact authorized execution API/payload must be verified from this repository first.
#>

[CmdletBinding()]
param(
    [string]$Root = "C:\Users\athar\SmartSOC-AI",
    [string]$ReceiverHost = "192.168.56.1",
    [int]$ReceiverPort = 8787,
    [string]$VictimHost = "192.168.56.102",
    [int]$VictimPort = 8765,
    [switch]$SkipTests,
    [switch]$SkipHttpProbes
)

$ErrorActionPreference = "Continue"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$reportDir = Join-Path $Root "reports\diagnostics\phase1_master_$stamp"
New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
$report = Join-Path $reportDir "phase1_master_report.txt"
$findings = [System.Collections.Generic.List[string]]::new()

function Add-Section([string]$Title) {
    $findings.Add("")
    $findings.Add(("=" * 78))
    $findings.Add($Title)
    $findings.Add(("=" * 78))
}
function Add-Line([string]$Text) { $findings.Add($Text) }

function Test-TcpPort([string]$HostName, [int]$Port) {
    $client = $null
    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $task = $client.ConnectAsync($HostName, $Port)
        if (-not $task.Wait(2500)) { return "TIMEOUT" }
        if ($client.Connected) { return "OPEN" }
        return "CLOSED"
    } catch { return ("FAILED: " + $_.Exception.Message) }
    finally { if ($client) { $client.Dispose() } }
}

function Invoke-ReadOnlyProbe([string]$BaseUrl) {
    $routes = @("/", "/health", "/healthz", "/status", "/ready", "/v1/health")
    foreach ($route in $routes) {
        $url = $BaseUrl.TrimEnd("/") + $route
        try {
            $r = Invoke-WebRequest -Uri $url -Method Get -TimeoutSec 3 -UseBasicParsing
            $body = [string]$r.Content
            if ($body.Length -gt 1200) { $body = $body.Substring(0,1200) + "...[truncated]" }
            Add-Line ("GET {0} => HTTP {1}" -f $url, [int]$r.StatusCode)
            if ($body) { Add-Line ("  BODY: " + ($body -replace "`r?`n", " ")) }
        } catch {
            $status = "NO RESPONSE"
            try {
                if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
                    $status = "HTTP " + [int]$_.Exception.Response.StatusCode
                } else { $status = $_.Exception.Message }
            } catch { $status = $_.Exception.Message }
            Add-Line ("GET {0} => {1}" -f $url, $status)
        }
    }
}

Add-Section "RUN SUMMARY"
Add-Line "Timestamp: $(Get-Date -Format o)"
Add-Line "Computer: $env:COMPUTERNAME"
Add-Line "User: $env:USERNAME"
Add-Line "Root: $Root"
Add-Line "Receiver target: ${ReceiverHost}:$ReceiverPort"
Add-Line "Victim target: ${VictimHost}:$VictimPort"
Add-Line "Mode: READ-ONLY diagnostics; no live mitigation request will be sent."

if (-not (Test-Path -LiteralPath $Root)) {
    Add-Line "ERROR: Project root does not exist: $Root"
    $findings | Set-Content -LiteralPath $report -Encoding UTF8
    Write-Host "Project root not found. Report: $report" -ForegroundColor Red
    exit 2
}
Set-Location -LiteralPath $Root

Add-Section "1. HOST / VICTIM CONNECTIVITY"
Add-Line ("Receiver TCP {0}:{1} => {2}" -f $ReceiverHost, $ReceiverPort, (Test-TcpPort $ReceiverHost $ReceiverPort))
Add-Line ("Victim TCP {0}:{1} => {2}" -f $VictimHost, $VictimPort, (Test-TcpPort $VictimHost $VictimPort))
Add-Line "Local IPv4 addresses:"
try {
    Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop |
        Where-Object { $_.IPAddress -notlike "127.*" -and $_.AddressState -eq "Preferred" } |
        ForEach-Object { Add-Line ("  {0} ({1})" -f $_.IPAddress, $_.InterfaceAlias) }
} catch { Add-Line ("  Unable to enumerate local addresses: " + $_.Exception.Message) }

if (-not $SkipHttpProbes) {
    Add-Section "2. READ-ONLY HTTP ROUTE DISCOVERY"
    Invoke-ReadOnlyProbe ("http://{0}:{1}" -f $ReceiverHost, $ReceiverPort)
    Invoke-ReadOnlyProbe ("http://{0}:{1}" -f $VictimHost, $VictimPort)
}

Add-Section "3. VICTIM PROCESS SNAPSHOT (LOCAL ONLY)"
Add-Line "This process list is from the machine running this script; it is NOT assumed to be the remote victim."
try {
    Get-Process -Name notepad -ErrorAction SilentlyContinue |
        Select-Object Id, ProcessName, StartTime, Path |
        ForEach-Object { Add-Line ("  PID={0}; Name={1}; StartTime={2}; Path={3}" -f $_.Id, $_.ProcessName, $_.StartTime, $_.Path) }
    if (-not (Get-Process -Name notepad -ErrorAction SilentlyContinue)) {
        Add-Line "  No local notepad.exe process found."
    }
} catch { Add-Line ("  Process query error: " + $_.Exception.Message) }

Add-Section "4. CONFIGURATION / POLICY / AUTHENTICATION EVIDENCE"
$searchRoots = @(
    (Join-Path $Root "src"),
    (Join-Path $Root "config"),
    (Join-Path $Root "configs"),
    (Join-Path $Root "scripts"),
    (Join-Path $Root "tests")
) | Where-Object { Test-Path -LiteralPath $_ }

$interestingFiles = @()
foreach ($p in $searchRoots) {
    $interestingFiles += Get-ChildItem -LiteralPath $p -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -match '(policy|auth|hmac|victim|receiver|mitigation|audit|scenario|execution|key|config)' -and
            $_.Extension -in @('.py','.ps1','.json','.yaml','.yml','.toml','.ini','.env','.md')
        }
}
$interestingFiles = $interestingFiles | Sort-Object FullName -Unique
Add-Line ("Candidate files discovered: {0}" -f $interestingFiles.Count)
$interestingFiles | ForEach-Object { Add-Line ("  " + $_.FullName) }

$pattern = '(HMAC|SHA-?256|fingerprint|key_id|key_fingerprint|authorization|authorized|execution_policy|PROCESS_TERMINATION|automatic_execution|automatic_containment|endpoint_modified|health|audit|DRY_RUN|LIVE_ARMED|LIVE_ENABLED|notepad\.exe|pending approval)'
$evidenceCount = 0
foreach ($f in $interestingFiles) {
    if ($evidenceCount -ge 220) { break }
    try {
        $matches = Select-String -LiteralPath $f.FullName -Pattern $pattern -CaseSensitive:$false -ErrorAction SilentlyContinue
        foreach ($m in $matches) {
            if ($evidenceCount -ge 220) { break }
            $line = $m.Line.Trim()
            # Never print likely secret values from .env/config lines.
            if ($line -match '(?i)(secret|token|password|private.?key|hmac.?key|shared.?key)\s*[:=]') {
                $line = ($line -replace '([:=]\s*).+$', '$1[REDACTED]')
            }
            if ($line.Length -gt 260) { $line = $line.Substring(0,260) + "...[truncated]" }
            Add-Line ("{0}:{1}: {2}" -f $f.FullName, $m.LineNumber, $line)
            $evidenceCount++
        }
    } catch {}
}
Add-Line ("Evidence lines captured: {0}" -f $evidenceCount)

Add-Section "5. KEY FILE FINGERPRINTS (HASH ONLY; NO KEY CONTENT)"
$keyFiles = @()
foreach ($p in $searchRoots) {
    $keyFiles += Get-ChildItem -LiteralPath $p -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '(?i)(hmac|shared.?key|secret.?key|auth.?key)' -and $_.Length -le 65536 }
}
$keyFiles = $keyFiles | Sort-Object FullName -Unique
if (-not $keyFiles) {
    Add-Line "No obvious key files found under src/config/scripts/tests. Keys may be environment variables or stored outside this project."
} else {
    foreach ($kf in $keyFiles) {
        try {
            $hash = (Get-FileHash -LiteralPath $kf.FullName -Algorithm SHA256).Hash
            Add-Line ("FILE={0}; SHA256_FILE_FINGERPRINT={1}" -f $kf.FullName, $hash)
            Add-Line "  NOTE: this hashes the file bytes. It is comparable only if both sides use the exact same key-file format."
        } catch { Add-Line ("HASH ERROR {0}: {1}" -f $kf.FullName, $_.Exception.Message) }
    }
}
Add-Line "Remote victim key fingerprint is not inferred from a local host file. It must be returned by the victim's existing authenticated/health mechanism or checked on the victim itself."

Add-Section "6. AUDIT / REPORT / LOG FILES"
$logRoots = @(
    (Join-Path $Root "reports"),
    (Join-Path $Root "logs"),
    (Join-Path $Root "src")
) | Where-Object { Test-Path -LiteralPath $_ }
$logFiles = @()
foreach ($p in $logRoots) {
    $logFiles += Get-ChildItem -LiteralPath $p -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '(?i)(audit|mitigation|authorization|execution|receiver|scenario|risk|gate|policy|integration)' -and $_.Length -le 5MB }
}
$logFiles = $logFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 40
if (-not $logFiles) { Add-Line "No matching audit/report/log files found." }
foreach ($lf in $logFiles) {
    Add-Line ("FILE={0}; Modified={1}; Bytes={2}" -f $lf.FullName, $lf.LastWriteTime.ToString("o"), $lf.Length)
    try {
        $tail = Get-Content -LiteralPath $lf.FullName -Tail 4 -ErrorAction SilentlyContinue
        foreach ($ln in $tail) {
            $safe = [string]$ln
            $safe = $safe -replace '(?i)(hmac.?key|secret|token|password)\s*["'']?\s*[:=]\s*["'']?[^,"''\s}]+', '$1=[REDACTED]'
            if ($safe.Length -gt 280) { $safe = $safe.Substring(0,280) + "...[truncated]" }
            Add-Line ("  TAIL: " + $safe)
        }
    } catch {}
}

Add-Section "7. TEST HARNESS / TEST SUITE"
$pytest = Get-Command pytest -ErrorAction SilentlyContinue
$python = Get-Command python -ErrorAction SilentlyContinue
Add-Line ("pytest command found: " + [bool]$pytest)
Add-Line ("python command found: " + [bool]$python)
$testFiles = Get-ChildItem -LiteralPath $Root -File -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^(test_.*|.*_test)\.py$' -and $_.FullName -notmatch '\\(\.venv|venv|node_modules|\.git)\\' } |
    Select-Object -First 80
Add-Line ("Python test files found: {0}" -f @($testFiles).Count)
$testFiles | ForEach-Object { Add-Line ("  " + $_.FullName) }

if (-not $SkipTests) {
    $runner = Join-Path $Root "Run-SmartSOC-ScenarioSuite.ps1"
    Add-Section "8. MAINTAINED SCENARIO SUITE"
    if (Test-Path -LiteralPath $runner) {
        Add-Line "Running maintained scenario suite. This is the repository's regression test suite, not proof of live endpoint mitigation."
        $testOutput = Join-Path $reportDir "scenario_suite_output.txt"
        try {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $runner *>&1 |
                Tee-Object -FilePath $testOutput | ForEach-Object { Add-Line ([string]$_) }
            Add-Line ("Scenario suite output file: " + $testOutput)
            Add-Line ("Scenario suite exit code: " + $LASTEXITCODE)
        } catch { Add-Line ("Scenario suite launch error: " + $_.Exception.Message) }
    } else {
        Add-Line "Run-SmartSOC-ScenarioSuite.ps1 not found; suite not run."
    }
} else {
    Add-Section "8. MAINTAINED SCENARIO SUITE"
    Add-Line "Skipped by -SkipTests."
}

Add-Section "9. AUTOMATIC MITIGATION SAFETY BOUNDARY"
Add-Line "No POST request, process termination, live mitigation, or policy modification was performed by this script."
Add-Line "Reason: the existing endpoint's exact authenticated request schema and authorization contract must be read from the project's implementation; guessing it could bypass or invalidate the safety gates."
Add-Line "This report gathers the code/routes/logs needed to implement the next patch against the actual API."
Add-Line "Phase 1 live-baseline proof still requires: remote victim health + reported key fingerprint, policy state, a request through the existing authorized path, remote process disappearance, and matching audit event."
Add-Line "Fail-closed live tests still require a dedicated safe test harness that confirms rejected requests do not reach the process-termination adapter."

$findings | Set-Content -LiteralPath $report -Encoding UTF8
Write-Host ""
Write-Host "SmartSOC Phase 1 diagnostics complete." -ForegroundColor Green
Write-Host "Report: $report" -ForegroundColor Cyan
Write-Host "No live mitigation request was sent." -ForegroundColor Yellow
Write-Host "Review the report and share it for a precise patch." -ForegroundColor Yellow
