# Windows Defensive Agent v3

This folder contains a cleaned repository copy of the supplied `smartsoc_defensive_agent_v2.py` console output. The source declares `AGENT_VERSION = "smartsoc-defensive-agent-v3-hmac"`; it is therefore named here for its internal version identifier. The terminal-wrapped strings were rejoined so the copy can be parsed as Python. Runtime behavior/settings have not intentionally been changed.

## Responsibility

The agent offers `GET /health` and `POST /mitigate` on `0.0.0.0:8765`. It validates the client address, request size, timestamp, HMAC-SHA256 signature and nonce; checks that the requested action and target process are allowlisted; validates PID/process identity; invokes `taskkill` without a shell; checks post-action state; and records audit events.

## Controls visible in the supplied code

| Control | Configured behavior |
|---|---|
| Listen endpoint | `0.0.0.0:8765` |
| Routes | `GET /health`, `POST /mitigate` |
| Action allowlist | `PROCESS_TERMINATION` |
| Process allowlist | `notepad.exe`, `ping.exe` |
| Client IPs | `192.168.56.1`, `127.0.0.1`, `::1` |
| HMAC | HMAC-SHA256 over timestamp, nonce and exact body |
| Clock skew | 120 seconds |
| Request body | Up to 16,384 bytes |
| Replay protection | In-memory nonce tracking |
| Audit | `logs/smartsoc_defensive_agent_v3_audit.jsonl` relative to this script's directory |
| Live gate | `SMARTSOC_ALLOW_TERMINATION=1` and `SMARTSOC_AGENT_SECRET` length at least 32 characters |

## Deployment notes

The supplied audit watcher in `tools/live_demo/watch_victim_audit.ps1` currently requests `C:\SmartSOC-Agent-v2\logs\defensive_agent_v2_audit.jsonl`, while this agent source writes `logs/smartsoc_defensive_agent_v3_audit.jsonl` relative to the script. Confirm the actual version installed on the VM and align the watcher in a separate tested change.

Do not assume that adding this file to GitHub deploys it to the Windows victim. The supplied machine path was `C:\SmartSOC-Agent-v2`; any deployment/copy operation should be deliberate and backed up first.

## Security notes

- `0.0.0.0` binds on all interfaces; use Windows Firewall and isolated lab networking to restrict access.
- HTTP is unencrypted. HMAC authenticates signed request content but does not encrypt it.
- Provision the same strong `SMARTSOC_AGENT_SECRET` on the intended client and agent via a secure local mechanism. Never commit the value.
- Keep `SMARTSOC_ALLOW_TERMINATION` unset/off while installing, reviewing, and running negative tests.
- A successful health response is not proof a process was terminated. Check the mitigation response and audit trail.
- Verify the committed SmartSOC-side policy independently; it may have its own live-execution gates.
