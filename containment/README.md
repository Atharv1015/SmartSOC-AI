# Containment evidence staging

Use this folder for files and scripts deliberately collected by an administrator for investigation. It is not connected to the receiver or forwarder and does not automatically download, quarantine, inspect, extract, or execute content.

- evidence/: collected files and scripts
- manifests/: collection records with case ID, UTC time, source alert ID, source host/path, collector, reason, and SHA-256 hash

Treat contents as untrusted. Do not open or run them on a production host. Analyze them only in an isolated lab. Preserve originals and record any derived copies separately.
