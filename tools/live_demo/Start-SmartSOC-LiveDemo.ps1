$ErrorActionPreference = "Stop"
Set-Location C:\Users\athar\SmartSOC-AI

$dir = Join-Path (Get-Location).Path "tools\live_demo"

# Avoid launching a duplicate receiver.
try {
    $health = Invoke-RestMethod `
        -Uri "http://192.168.56.1:8788/health" `
        -TimeoutSec 2

    if ($health.status -eq "ok") {
        throw "Receiver is already running. Stop its existing instance with Ctrl+C, then rerun this launcher."
    }
} catch {
    if ($_.Exception.Message -like "Receiver is already running*") {
        throw
    }
    # Connection failure is expected when the receiver is stopped.
}

& wt.exe -w 0 new-tab --title "1 - SmartSOC Receiver" `
    powershell.exe -NoExit -ExecutionPolicy Bypass `
    -File (Join-Path $dir "run_receiver.ps1")

Start-Sleep -Seconds 2

& wt.exe -w 0 new-tab --title "2 - Ubuntu Forwarder" `
    powershell.exe -NoExit -ExecutionPolicy Bypass `
    -File (Join-Path $dir "run_ubuntu_forwarder.ps1")

& wt.exe -w 0 new-tab --title "3 - Victim Audit SSH" `
    powershell.exe -NoExit -ExecutionPolicy Bypass `
    -File (Join-Path $dir "watch_victim_audit.ps1")

& wt.exe -w 0 new-tab --title "4 - Phase A Tests" `
    powershell.exe -NoExit -ExecutionPolicy Bypass `
    -File (Join-Path $dir "phase_a_workspace.ps1")

Write-Host "SmartSOC demo tabs requested. Check each tab for startup errors."
