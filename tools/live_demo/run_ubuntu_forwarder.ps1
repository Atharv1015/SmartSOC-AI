$ErrorActionPreference = "Continue"
Write-Host "=== UBUNTU WAZUH LIVE FORWARDER ===" -ForegroundColor Cyan
Write-Host "SSH target: smartsoc@192.168.56.101"
ssh.exe -tt smartsoc@192.168.56.101 "cd ~/smartsoc-forwarder-v3 && python3 -u wazuh_live_forwarder_v3.py"
Write-Host "SSH/forwarder session ended. Check the error above." -ForegroundColor Yellow
