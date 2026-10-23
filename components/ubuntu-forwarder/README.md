# Ubuntu / Wazuh Forwarder v3

**Source file:** `wazuh_live_forwarder_v3.py`
**Role:** read selected Wazuh alerts and send raw alert JSON to the SmartSOC receiver. The forwarder does not execute commands, terminate processes, or perform containment.

## Configuration in the supplied source

| Setting | Value |
|---|---|
| Alert input | `/var/ossec/logs/alerts/alerts.json` |
| Receiver | `http://192.168.56.1:8787/wazuh-event` |
| Target Wazuh agent ID | `001` |
| Target agent name | `SmartSOC-Windows-Victim` |
| Poll interval | `0.25` seconds |
| HTTP timeout | `30` seconds |
| Delivery attempts | Up to 3 attempts with delays between retryable failures |
| Duplicate handling | In-memory queue/set capped at 10,000 alert IDs |
| Log output | `~/smartsoc-forwarder-v3/logs/forwarder_v3.log` |
| Startup position | Starts at current end of file; does not replay historical alerts |

The source uses an explicit Wazuh rule-ID allowlist. An alert must match the target agent and one of the listed rule IDs before forwarding.

## Deployment steps

These steps assume the Ubuntu VM is `192.168.56.101`, and that the repository has been cloned on that VM under `~/SmartSOC-AI`. If your checkout path is different, adjust only the source path in the copy command.

1. **Check the Wazuh alert source and Python.**

   ```bash
   python3 --version
   test -r /var/ossec/logs/alerts/alerts.json && echo "Wazuh alerts readable"
   ```

   If the alert file is unreadable, resolve the Wazuh service/account permissions first; do not run the forwarder as root by default just to bypass permissions.

2. **Copy the version-controlled script into the path expected by the existing launcher.**

   ```bash
   mkdir -p ~/smartsoc-forwarder-v3/logs
   cp ~/SmartSOC-AI/components/ubuntu-forwarder/wazuh_live_forwarder_v3.py \
      ~/smartsoc-forwarder-v3/wazuh_live_forwarder_v3.py
   ```

   This keeps the repository source organized separately while matching `tools/live_demo/run_ubuntu_forwarder.ps1`, which SSHes to `smartsoc@192.168.56.101` and runs from `~/smartsoc-forwarder-v3`.

3. **Check syntax without starting event forwarding.**

   ```bash
   python3 -m py_compile ~/smartsoc-forwarder-v3/wazuh_live_forwarder_v3.py
   ```

4. **Verify the receiver endpoint before launch.**

   The configured destination is `192.168.56.1:8787/wazuh-event`. Confirm the intended receiver process is listening on this address and route, from the isolated lab network. A TCP reachability check only confirms that a port is reachable; it does not prove the event route accepts the payload.

5. **Start the forwarder in the foreground.**

   ```bash
   cd ~/smartsoc-forwarder-v3
   python3 -u wazuh_live_forwarder_v3.py
   ```

   Use the existing SSH-based launcher if preferred. The supplied scripts run the process interactively; this component package does not install a systemd service.

6. **Inspect the log.**

   ```bash
   tail -f ~/smartsoc-forwarder-v3/logs/forwarder_v3.log
   ```

   Expected log categories include `START`, `RELEVANT_ALERT`, `DELIVERY OK`, `DELIVERY FAILED`, `DUPLICATE_SKIPPED`, and `DELIVERY ABANDONED`. No new matching alerts means it may remain idle.

## Stop and restart

Stop a foreground process with `Ctrl+C`. After a restart, duplicate history is reset because it is kept in memory. The script tails from the current end of the alert file and does not recover delivery failures from a persistent queue.

## Troubleshooting

- **`Wazuh alert log not found`** — verify Wazuh is installed and the configured `ALERTS_FILE` exists.
- **No relevant alerts** — verify target agent ID/name and rule IDs in real Wazuh records.
- **Connection refused / timeout** — verify host-only networking and that the receiver is listening on `192.168.56.1:8787`.
- **HTTP 404 / route error** — verify that the selected receiver actually implements `POST /wazuh-event`.
- **Delivery abandoned** — inspect receiver logs and response body before retrying; the forwarder has no durable queue.

## Integration mismatch requiring attention

The checked-in `tools/live_demo/run_receiver.ps1` currently starts `smartsoc_receiver_v1.py` and describes a LOG_ONLY endpoint at port `8788`, while this v3 forwarder targets port `8787` and route `/wazuh-event`. The v2 receiver source is configured for `8787`. Select and test the intended receiver/route/port pairing before a demonstration. This documentation and component copy do not change the launcher or runtime configuration.

## Security

The event post uses HTTP, not TLS. Keep it on an isolated, controlled lab network and sanitize forwarded-event logs before public sharing. This forwarder contains no mitigation code.
