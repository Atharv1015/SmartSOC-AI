$ErrorActionPreference = "Stop"

$path = ".\src\integration\smartsoc_live_receiver_v2.py"
if (-not (Test-Path -LiteralPath $path)) {
    throw "Receiver file not found: $path"
}

$source = Get-Content -LiteralPath $path -Raw

$pattern = '(?s)"rule_process_pair_matches"\s*:\s*\(.*?\)\s*,\s*(?="pid_present"\s*:)'
$matches = [regex]::Matches($source, $pattern)

if ($matches.Count -ne 1) {
    throw "Expected exactly one rule_process_pair_matches block; found $($matches.Count). No changes made."
}

$newBlock = @"
"rule_process_pair_matches": (
            (str(rule_id) == "100101" and _lab_process == "ping.exe")
            or
            (str(rule_id) == "100104" and _lab_process == "notepad.exe")
            or
            (
                str(rule_id) == "100130"
                and _lab_process == "powershell.exe"
                and isinstance(eventdata, dict)
                and "SmartSOC-DefenderLab" in str(eventdata.get("commandLine", ""))
                and "DefenderTamperSimulation.ps1" in str(eventdata.get("commandLine", ""))
            )
        ),
"@

$replacement = $newBlock.TrimEnd() + "`r`n        "
$updated = [regex]::Replace(
    $source,
    $pattern,
    [System.Text.RegularExpressions.MatchEvaluator]{
        param($match)
        return $replacement
    },
    1
)

if ($updated -eq $source) {
    throw "No change produced. Original receiver remains unchanged."
}

# Validate the candidate source before writing it.
$temp = Join-Path $env:TEMP "smartsoc_receiver_patch_candidate.py"
Set-Content -LiteralPath $temp -Value $updated -Encoding UTF8

$python = Get-Command python -ErrorAction Stop
& $python.Source -m py_compile $temp
if ($LASTEXITCODE -ne 0) {
    Remove-Item -LiteralPath $temp -Force -ErrorAction SilentlyContinue
    throw "Candidate syntax validation failed. Receiver was not modified."
}

Remove-Item -LiteralPath $temp -Force
Write-Host "PATCH_CANDIDATE_SYNTAX_OK"
Write-Host "Candidate validated. Receiver has NOT been modified."
