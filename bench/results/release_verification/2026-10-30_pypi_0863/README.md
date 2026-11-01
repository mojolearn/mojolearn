# mojolearn 0.8.63

Source commit 3c026af7c1926c731295b71242545165cba62ebe; release tooling at 6d972b2221f6b882ed827e4e9b0c5391cb667229. Published 2026-10-30. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Identity and behavior were checked before merge on NVIDIA, AMD, Apple and CPU; this release built, checked artifact integrity and published (preverified route). The macOS wheel passed `verify --self-test` in a fresh venv.

## Wheels (preverified route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| mojolearn-core manylinux_2_35_x86_64 | d2576743301cfba0c10e4925ab85103b781ac9e2bbb93e8b266b2449c3ad2f08 | preverified, 0 jobs | pypi via alpha-api-0.8.63-linux-20261031 (source 3c026af7c192, tooling 2024e0a25773) |
| mojolearn-nvidia manylinux_2_35_x86_64 | 312cdc605d012e97f27da4c80237357cce02690e8ae6cbe9c656f566946855a5 | preverified, 0 jobs | pypi via alpha-api-0.8.63-nvidia-20261031 (source 3c026af7c192, tooling 2024e0a25773) |
| mojolearn-amd manylinux_2_35_x86_64 | 9119af63fedb2f7a9b2df75af5006e3f9e0a67f3206370ca90aca6ca17e764e8 | preverified, 0 jobs | pypi via alpha-api-0.8.63-amd-20261031 (source 3c026af7c192, tooling 2024e0a25773) |
| mojolearn-nvidia-ptx manylinux_2_35_x86_64 | 8b3ef57e597c4bc78ce4a0ac84ff43f33ab2d63f2a248144988d51ac592554b8 | preverified, 0 jobs | pypi via alpha-api-0.8.63-nvidia-ptx-20261031 (source 3c026af7c192, tooling 2024e0a25773) |
| mojolearn manylinux_2_35_x86_64 | 36ee5ac1c53db1c632e6e20159213d42ff78eb281ba06a004cdba602bc773fbb | preverified, 0 jobs | pypi via alpha-api-0.8.63-linux-meta-20261031 (source 3c026af7c192, tooling 2024e0a25773) |
| macosx_26_0_arm64 | 832efee85dc5c915be9d74098f88e62a613395957caa6607dcea636edf93700f | PASSED, 0 jobs | pypi via alpha-api-0.8.63-macos-20261031 (source 3c026af7c192, tooling 2024e0a25773) |

Finish line: macOS: pip install mojolearn==0.8.63 on this Mac: OK; mojolearn-core: pip resolves mojolearn-core==0.8.63 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia: pip resolves mojolearn-nvidia==0.8.63 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia-ptx: pip resolves mojolearn-nvidia-ptx==0.8.63 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-amd: pip resolves mojolearn-amd==0.8.63 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn: pip resolves mojolearn==0.8.63 for manylinux_2_35_x86_64; bytes are the published wheel's; pip install mojolearn-core==0.8.63 on manylinux_2_35_x86_64 resolves mojolearn-core; pip install mojolearn==0.8.63 on manylinux_2_35_x86_64 resolves mojolearn-core, mojolearn-nvidia, mojolearn-amd, mojolearn-nvidia-ptx, mojolearn; pip install mojolearn-nvidia==0.8.63 on manylinux_2_35_x86_64 resolves mojolearn-core, mojolearn-nvidia, mojolearn-nvidia-ptx; pip install mojolearn-amd==0.8.63 on manylinux_2_35_x86_64 resolves mojolearn-core, mojolearn-amd.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
