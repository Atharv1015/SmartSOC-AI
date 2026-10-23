# Repository structure and organization policy

## Stable project areas

```text
SmartSOC-AI/
├── README.md
├── src/                             # Core SmartSOC Python modules
│   ├── inference/                   # Inference paths
│   ├── decision/                    # Risk, response planning and decision policies
│   ├── integration/                 # Receiver generations and integrations
│   ├── integration_v2/              # v2 policy / authorization bridges
│   ├── autonomous/                  # Earlier controller/reasoning implementation
│   └── autonomous_v2/               # SmartSOC-side victim client and adapter
├── components/
│   ├── ubuntu-forwarder/            # Ubuntu/Wazuh forwarder v3 package copy
│   └── windows-defensive-agent/     # Windows defensive-agent v3 package copy
├── tests/                           # Focused maintained test suite
├── tools/
│   ├── live_demo/                   # Demo launchers and audit watcher
│   └── ...                          # Diagnostics and utility scripts
├── data/                            # Selected integration samples and manifests
├── manual_live_response_bundle/     # Separate manual evidence helper
├── containment/                     # Evidence staging documentation
└── docs/                            # Architecture and project documentation
```

## Why legacy root scripts are not bulk-moved in this pass

The repository has many top-level patch, report-generation, diagnostic and forensic scripts. Some derive the project root from `Path(__file__).resolve().parent`; some PowerShell launchers use relative paths or hard-code the current Windows checkout path. Moving these files without updating those path assumptions and testing the scripts can silently break them.

For this change, the external VM components and documentation are placed in dedicated folders. Legacy tools stay at their current paths until a separate path-aware refactor can update project-root discovery and validate each supported invocation. This avoids changing response functionality as part of a documentation/organization change.

## Source-of-truth and deployment distinction

The files in `components/` are repository-managed copies from the supplied lab source. Adding or editing a file in the Git repository does not deploy it to Ubuntu or the victim Windows VM. The existing launch scripts still target the paths described in each component README. Synchronize deployment copies deliberately and test any launcher changes separately.


## Component deployment guides

The two folders under `components/` contain the source copies and step-by-step instructions. Deployment is a separate operation: publishing or cloning the repository does not install the code on the Ubuntu or victim VM. Review `docs/DEPLOYMENT_GUIDE.md` and each component README before running them.
