$ErrorActionPreference = "Stop"
Set-Location C:\Users\athar\SmartSOC-AI
Write-Host "=== SMARTSOC RECEIVER + AI INFERENCE ===" -ForegroundColor Cyan
Write-Host "Endpoint: http://192.168.56.1:8788/wazuh-event"
Write-Host "Policy: LOG_ONLY; no mitigation or containment" -ForegroundColor Yellow
py -3 .\src\integration\smartsoc_receiver_v1.py
