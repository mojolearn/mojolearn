# mojolearn 0.8.41

Source commit 32a1c4bbc3904117b1c055f620a0b31031cf4b81; release tooling at 86dbf5f732dcea4ef7e16c396cb2e049238160a6. Published 2026-10-12. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Every wheel smoke below verifies the installed wheel against it.

## Wheels (light route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | abc6e425067a3d38d8633f2830e690be1cf21a4bdaa34529a06bd23610a457f7 | PASSED, 12 jobs | pypi via alpha-api-0.8.41-linux-20261012 (source 32a1c4bbc390, tooling 86dbf5f732dc) |
| mojolearn-nvidia manylinux_2_35_x86_64 | c345b27b4ac8cc3bd5c91377ed0a3e78d2864796df69e68775635ecf635b46de | PASSED, 12 jobs | pypi via alpha-api-0.8.41-nvidia-20261012 (source 32a1c4bbc390, tooling 86dbf5f732dc) |
| mojolearn-amd manylinux_2_35_x86_64 | cd12c89bbca9f7a978608eb56d4c5eded06d0695c6cf12db3a305b0602e57a58 | PASSED, 12 jobs | pypi via alpha-api-0.8.41-amd-20261012 (source 32a1c4bbc390, tooling 32a1c4bbc390) |
| macosx_11_0_arm64 | 48f734fac68992450b9630b66ba9e3d47e179033c6a563b24c07d5aa4b6fef64 | PASSED, 12 jobs | pypi via alpha-api-0.8.41-macos-20261012 (source 32a1c4bbc390, tooling 32a1c4bbc390) |

Finish line: macOS: pip install mojolearn==0.8.41 on this Mac: OK; Linux: pip resolves mojolearn==0.8.41 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia: pip resolves mojolearn-nvidia==0.8.41 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-amd: pip resolves mojolearn-amd==0.8.41 for manylinux_2_35_x86_64; bytes are the published wheel's.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
