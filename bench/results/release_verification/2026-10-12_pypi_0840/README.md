# mojolearn 0.8.40

Source commit db9dc78a72ace5261b4ea826aab3b4288c97475f; release tooling at db9dc78a72ace5261b4ea826aab3b4288c97475f. Published 2026-10-12. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Every wheel smoke below verifies the installed wheel against it.

## Wheels (light route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| macosx_11_0_arm64 | a2ff79ab289c59d31b79f2212283090212cd73d1c04c41a0ad0e8cfbd7dd3a31 | PASSED, 12 jobs | pypi via alpha-api-0.8.40-macos-20261012 (source db9dc78a72ac, tooling db9dc78a72ac) |

Finish line: macOS: pip install mojolearn==0.8.40 on this Mac: OK; not yet published: linux, nvidia, amd.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
