# SmartSOC-AI Lab Deployment Guide

This guide describes a cautious, manual rollout for the three roles identified in the checked-in lab scripts. Confirm the actual VM names, operating systems, addresses, and network adapter mode against the hypervisor before using these values.

| Role | Address found in scripts | Main purpose |
|---|---|---|
| SmartSOC Windows host | `192.168.56.1` | Receiver, inference, policy and client adapter |
| Ubuntu/Wazuh VM | `192.168.56.101` | Wazuh alert source and forwarder |
| Windows victim VM | `192.168.56.102` | Endpoint telemetry and defensive-agent endpoint |

## Order of operations

1. Confirm all machines are on the intended isolated lab network. Restrict the victim port `8765` and receiver ports `8787`/`8788` at the network and host-firewall layers.
2. Choose the receiver generation deliberately. The supplied v3 forwarder targets `192.168.56.1:8787/wazuh-event`; the current `run_receiver.ps1` starts a different receiver and documents port `8788`. Do not proceed until the listener, route and forwarder agree.
3. Start and validate the receiver in its intended non-destructive/log-only configuration.
4. Deploy and syntax-check the Ubuntu forwarder. Confirm the Wazuh alert file is readable, then start the forwarder in the foreground and observe logs.
5. Deploy and syntax-check the victim agent with its live gate disabled. Test only `/health` locally and inspect the response; do not send a termination request during initial validation.
6. Verify the host's configured client address matches the victim agent's client-IP allowlist, while keeping the endpoint reachable only from the isolated lab segment.
7. Run relevant rejection/authorization unit tests, review component logs, and document results.
8. Only after explicit review and authorization should a separate controlled-lab test plan consider live endpoint actions. This README/package update does not enable those actions.

## Secrets and configuration

- `SMARTSOC_AGENT_SECRET` is a shared secret used by the sender and agent. It must not be committed, pasted into documentation, placed in screenshots, or stored in source code.
- The victim service checks `SMARTSOC_ALLOW_TERMINATION=1` before processing POST mitigation requests. Keep the variable unset or off during deployment and negative testing.
- The SmartSOC receiver/policy has separate gates. Review the effective entry point's policy; do not assume the victim-side gate is the only control.
- Preserve endpoint protection, Windows Firewall, and protected-process rules.

## Evidence and troubleshooting

Record the timestamp, VM and component versions, receiver endpoint, Wazuh agent/rule, sanitized alert ID, decision result, agent health response, and audit result. Do not publish raw security alerts or audit records until usernames, real IPs, file paths, tokens, and other sensitive context have been reviewed.

If event delivery fails, verify the receiver version/route/port first, then check forwarder logs. If a victim-agent request is rejected, inspect the returned status and audit log; never weaken allowlists or gates simply to make a demonstration pass.
