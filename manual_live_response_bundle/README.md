# SmartSOC manual response bundle

Copy the entire `manual_live_response_bundle` folder into the root of `C:\Users\athar\SmartSOC-AI` before running. This is a manually run, lab-only file-response helper. It does not patch or start the current SmartSOC receiver, defensive-agent process, or Ubuntu forwarder.

## Host (SmartSOC-AI)

1. Use the same `SMARTSOC_EVIDENCE_SECRET` value already provisioned for the defensive agent. Do not paste it into chat or put it in source code. In a PowerShell window, set it for that window only:

   ```powershell
   $env:SMARTSOC_EVIDENCE_SECRET = '<your existing shared secret>'
   py -3 .\manual_live_response_bundle\evidence_receiver.py
   ```

2. The receiver listens only on `192.168.56.1:8790`, accepts only `192.168.56.102`, and writes received files under `manual_live_response_bundle\containment\evidence\<case-id>`. It writes a JSONL manifest. It never opens or executes evidence.

## Victim VM

1. Set the same secret in the victim PowerShell window (do not send it in chat).
2. After an alert, obtain the exact file path and PID from Wazuh/Windows event details. Run:

   ```powershell
   .\Submit-SmartSOC-Evidence.ps1 -FilePath 'C:\Users\vboxuser\Downloads\sample.exe' -CaseId 'CAPSTONE-001' -ProcessId 1234
   ```

3. The script verifies that the PID image path exactly matches the file, stops that PID, uploads the file with HMAC and SHA-256, checks the host's returned hash, then removes the victim copy only after that check succeeds.

## Limits

- This manual helper is not wired into the current alert pipeline or the defensive agent. It does not automatically discover downloads.
- Terminating a process acts after process start; it cannot guarantee the file did not execute earlier.
- HTTP is unencrypted. Use only on the isolated host-only lab network. HMAC authenticates and detects tampering; it does not encrypt contents.
- Removal is not secure disk erasure. The host copy and manifest are the preserved investigation evidence.
- The existing live receiver's dry-run/approval path is unchanged by these files. Do not describe this helper as automatic SmartSOC containment.



