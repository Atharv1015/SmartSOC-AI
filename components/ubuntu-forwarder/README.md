# Ubuntu / Wazuh Forwarder v3

This folder contains the repository-managed copy of `wazuh_live_forwarder_v3.py` shared from the lab.

## Responsibility

The forwarder tails Wazuh's `/var/ossec/logs/alerts/alerts.json`, filters alerts for the configured victim agent and relevant rule IDs, suppresses recent duplicate alert IDs in memory, and POSTs the original alert to the configured SmartSOC receiver. It does not execute commands, terminate processes, or contain endpoints.

## Configuration in this version

| Setting | Current value |
|---|---|
| Wazuh alerts source | `/var/ossec/logs/alerts/alerts.json` |
| Receiver | `http://192.168.56.1:8787/wazuh-event` |
| Target agent ID | `001` |
| Target agent name | `SmartSOC-Windows-Victim` |
| Poll interval | `0.25` seconds |
| HTTP timeout | `30` seconds |
| Delivery attempts | Up to 3 |
| Deduplication | In-memory, up to 10,000 recent IDs |
| Log file | `~/smartsoc-forwarder-v3/logs/forwarder_v3.log` |

## Deployment notes

The checked-in Windows launcher currently SSHes to `smartsoc@192.168.56.101` and starts a copy under `~/smartsoc-forwarder-v3`. This repository copy does not automatically deploy itself to Ubuntu. Keep deployed and committed files synchronized or update the launcher in a separate tested change.

Before starting the process, confirm the alert file is readable and that the selected SmartSOC receiver is listening at the configured endpoint. The forwarder starts at the current end of the alert file and does not replay historical events. Retried delivery is bounded; there is no persistent retry queue.

## Integration mismatch to resolve

The forwarder sends to `192.168.56.1:8787/wazuh-event`, while `tools/live_demo/run_receiver.ps1` starts `smartsoc_receiver_v1.py` and documents port `8788` in `LOG_ONLY` mode. Confirm the intended receiver generation and route before running the demo. No receiver, port, or live-response configuration was changed by adding this component folder.

## Safety

This component forwards telemetry only. The HTTP channel is not encrypted; keep it on an isolated, access-controlled lab network. Do not place credentials or sensitive event exports in this folder.
