$ErrorActionPreference = "Stop"
$root = "C:\Users\athar\SmartSOC-AI"
$receiver = Join-Path $root "src\integration\smartsoc_live_receiver_v2.py"
$adapter = Join-Path $root "src\autonomous_v2\victim_mitigation_adapter_v2.py"
foreach ($p in @($receiver, $adapter)) { if (-not (Test-Path -LiteralPath $p)) { throw "Missing source: $p" } }

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backup = Join-Path $root "backups\defender_lab_patch_$stamp"
New-Item -ItemType Directory -Path $backup -Force | Out-Null
Copy-Item $receiver (Join-Path $backup "smartsoc_live_receiver_v2.py")
Copy-Item $adapter (Join-Path $backup "victim_mitigation_adapter_v2.py")

$r = Get-Content -LiteralPath $receiver -Raw
$old = @'
        "rule_process_pair_matches": (
            (str(rule_id) == "100101" and _lab_process == "ping.exe")
            or
            (str(rule_id) == "100104" and _lab_process == "notepad.exe")
        ),
        "pid_present": _lab_pid not in (None, ""),
'@
$new = @'
        "rule_process_pair_matches": (
            (str(rule_id) == "100101" and _lab_process == "ping.exe")
            or
            (str(rule_id) == "100104" and _lab_process == "notepad.exe")
            or
            (
                str(rule_id) == "100130"
                and _lab_process == "powershell.exe"
                and isinstance(eventdata.get("commandLine"), str)
                and "SmartSOC-DefenderLab" in eventdata.get("commandLine", "")
                and "DefenderTamperSimulation.ps1" in eventdata.get("commandLine", "")
            )
        ),
        "pid_present": _lab_pid not in (None, ""),
'@
if (-not $r.Contains($old)) { throw "Receiver expected override block not found; no receiver changes applied." }
$r = $r.Replace($old, $new)
$old2 = @'
        reasoning_result["lab_test_override"] = True
        requested_mitigation = "PROCESS_TERMINATION"
'@
$new2 = @'
        reasoning_result["lab_test_override"] = True
        reasoning_result["lab_expected_script"] = (
            "SmartSOC-DefenderLab\\DefenderTamperSimulation.ps1"
            if str(rule_id) == "100130" and _lab_process == "powershell.exe"
            else None
        )
        requested_mitigation = "PROCESS_TERMINATION"
'@
if (-not $r.Contains($old2)) { throw "Receiver override assignment block not found." }

$a = Get-Content -LiteralPath $adapter -Raw
$old3 = @'
    if not policy.is_process_allowed(process_name):
        return {
            "success": False, "verified": False,
            "status": "REQUEST_BLOCKED",
            "error": "Process name is not allowlisted.",
        }
'@
$new3 = @'
    _lab_simulation_target = (
        reasoning_result.get("lab_test_override") is True
        and str(reasoning_result.get("wazuh_rule_id", "")) == "100130"
        and process_name == "powershell.exe"
        and reasoning_result.get("lab_expected_script")
            == "SmartSOC-DefenderLab\\DefenderTamperSimulation.ps1"
    )
    if not policy.is_process_allowed(process_name) and not _lab_simulation_target:
        return {
            "success": False, "verified": False,
            "status": "REQUEST_BLOCKED",
            "error": "Process name is not allowlisted for this request.",
        }
'@
if (-not $a.Contains($old3)) { throw "Adapter allowlist block not found." }
$a = $a.Replace($old3, $new3)
$old4 = @'
            "wazuh_rule_id": reasoning_result.get("wazuh_rule_id"),
            "reason": reasoning_result.get("reason"),
'@
$new4 = @'
            "wazuh_rule_id": reasoning_result.get("wazuh_rule_id"),
            "reason": reasoning_result.get("reason"),
            "lab_test_override": reasoning_result.get("lab_test_override") is True,
            "lab_expected_script": reasoning_result.get("lab_expected_script"),
'@
if (-not $a.Contains($old4)) { throw "Adapter request context block not found." }

# Validate all replacement anchors before writing either file.
$r = $r.Replace($old2, $new2)
$a = $a.Replace($old4, $new4)
Set-Content -LiteralPath $receiver -Value $r -Encoding UTF8
Set-Content -LiteralPath $adapter -Value $a -Encoding UTF8
Write-Host "PATCHED receiver and adapter."
Write-Host "Global ALLOWED_PROCESSES unchanged."
Write-Host "Backups: $backup"
Write-Host "Apply victim patch on the victim VM before testing."
