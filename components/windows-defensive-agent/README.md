# Windows Defensive Agent — `smartsoc_defensive_agent_v2.py`

The filename comes from the supplied victim VM (`C:\SmartSOC-Agent-v2\smartsoc_defensive_agent_v2.py`). The source itself declares `AGENT_VERSION = "smartsoc-defensive-agent-v3-hmac"`; preserve the supplied filename when deploying so the existing local path and documentation stay aligned.

## Responsibility

A small Python HTTP server running on the Windows victim. It provides `GET /health` and `POST /mitigate` on `0.0.0.0:8765`. The file is a repository-managed copy; adding it to GitHub does not install or start it on a VM.

## Controls in the supplied source

| Control | Source configuration |
|---|---|
| Listen address / port | `0.0.0.0:8765` |
| Supported routes | `GET /health`, `POST /mitigate` |
| Action allowlist | `PROCESS_TERMINATION` only |
| Process allowlist | `notepad.exe`, `ping.exe` |
| Client IP allowlist | `192.168.56.1`, `127.0.0.1`, `::1` |
| Authentication | HMAC-SHA256 over timestamp, nonce, and exact raw request body |
| Clock-skew limit | 120 seconds |
| Maximum body | 16,384 bytes |
| Replay protection | In-memory nonce tracking |
| Target checks | PID must be a positive integer, process must be running, and its live image name must match the request |
| Execution | `taskkill` as an argument list with `shell=False` |
| Verification | Checks process state after attempted termination |
| Audit file | `logs/smartsoc_defensive_agent_v3_audit.jsonl` relative to the agent script directory |
| Environment gates | `SMARTSOC_AGENT_SECRET` must be at least 32 characters and `SMARTSOC_ALLOW_TERMINATION` must equal `1` for POST execution to proceed |

The agent returns a structured action result and writes audit events for processed or rejected actions. Do not equate an HTTP response or a healthy `/health` state with a verified process action.

## Deployment and startup steps

**Safety first:** leave `SMARTSOC_ALLOW_TERMINATION` unset or set to a value other than `1` during initial installation and validation. Do not set the live gate merely to make health report `HEALTHY`. Do not run a live termination test until the victim VM, target, policy, and shared secret have been checked and the test has been explicitly authorized.

1. **Back up the currently deployed file.** On the victim VM, open PowerShell:

   ```powershell
   $AgentDir = 'C:\SmartSOC-Agent-v2'
   $AgentFile = Join-Path $AgentDir 'smartsoc_defensive_agent_v2.py'
   if (Test-Path -LiteralPath $AgentFile) {
       $Backup = "$AgentFile.before-readme-component-update-$(Get-Date -Format yyyyMMdd_HHmmss)"
       Copy-Item -LiteralPath $AgentFile -Destination $Backup
       Write-Host "Backup created at $Backup"
   }
   ```

2. **Copy the repository-managed file into the victim VM deliberately.** Use your approved transfer path to place `components/windows-defensive-agent/smartsoc_defensive_agent_v2.py` at `C:\SmartSOC-Agent-v2\smartsoc_defensive_agent_v2.py`. If the repository clone is already available on the victim, copy from its local checkout. Do not assume pushing to GitHub deploys this file to the VM.

3. **Verify syntax without executing the agent.**

   ```powershell
   py -3 -m py_compile C:\SmartSOC-Agent-v2\smartsoc_defensive_agent_v2.py
   ```

4. **Start for local health-only validation with execution disabled.** In a dedicated PowerShell window on the victim:

   ```powershell
   Set-Location C:\SmartSOC-Agent-v2
   Remove-Item Env:SMARTSOC_ALLOW_TERMINATION -ErrorAction SilentlyContinue
   py -3 .\smartsoc_defensive_agent_v2.py
   ```

   Do not set the shared secret in the code. Provision `SMARTSOC_AGENT_SECRET` through your established secure local mechanism. With the execution gate unset, the POST endpoint must not perform termination. The health endpoint can report `DEGRADED`; that is expected while the execution gate is off.

5. **Check health from the victim itself.** In a second PowerShell window:

   ```powershell
   Invoke-RestMethod -Uri 'http://127.0.0.1:8765/health' | ConvertTo-Json -Depth 5
   ```

   Confirm the reported agent/version, port, allowed action/processes, and execution flag. Keep Windows Firewall configured to restrict network access to the host-only lab network and expected client. The code binds to all interfaces (`0.0.0.0`); the app-level client allowlist is not a substitute for firewall rules.

6. **Stop the foreground agent.** Use `Ctrl+C` in the PowerShell window. A Windows service or unattended startup unit is not created by this repository component.

## Audit watcher mismatch

The checked-in `tools/live_demo/watch_victim_audit.ps1` currently requests `C:\SmartSOC-Agent-v2\logs\defensive_agent_v2_audit.jsonl`. This supplied agent writes `C:\SmartSOC-Agent-v2\logs\smartsoc_defensive_agent_v3_audit.jsonl` if run from that directory. Verify which agent file is actually running before changing the watcher; this component documentation does not modify the existing watcher.

## Security notes

- `0.0.0.0` binds on all network interfaces; restrict connectivity at the VM network and Windows Firewall layers.
- HMAC authenticates and protects message integrity; HTTP does not encrypt content.
- Keep the secret private, identical across the intended sender and receiver, and outside Git.
- Nonces are kept in memory. Restarting the agent resets that replay cache.
- Only use the action on a disposable, isolated, explicitly authorized lab victim. Process termination is not general containment.
- The committed SmartSOC-side execution policy is a separate control that must be reviewed independently.
