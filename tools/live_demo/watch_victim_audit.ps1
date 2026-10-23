$ErrorActionPreference = "Continue"
Write-Host "=== WINDOWS VICTIM DEFENSIVE AGENT AUDIT ===" -ForegroundColor Cyan
Write-Host "The agent remains running on the victim VM."
Write-Host "Streaming its audit file over SSH..."
ssh.exe -tt vboxuser@192.168.56.102 "powershell.exe -NoLogo -NoProfile -Command `"Get-Content -LiteralPath 'C:\SmartSOC-Agent-v2\logs\defensive_agent_v2_audit.jsonl' -Wait`""
Write-Host "SSH/audit session ended. Check the error above." -ForegroundColor Yellow
