# mojolearn 0.8.51

Source commit 958cd351aa6ec8456d2c42ca33a4cd6900595c5f; release tooling at 958cd351aa6ec8456d2c42ca33a4cd6900595c5f. Published 2026-10-17. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Identity and behavior were checked before merge on NVIDIA, AMD, Apple and CPU; this release built, checked artifact integrity and published (preverified route). The macOS wheel passed `verify --self-test` in a fresh venv.

## Wheels (preverified route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | 225f6bc294af26f450c2bf517cf32748591f65cb373f2ea89cf207ccfb23ee15 | preverified, 0 jobs | pypi via alpha-api-0.8.51-linux-20261017 (source 958cd351aa6e, tooling 958cd351aa6e) |
| mojolearn-nvidia manylinux_2_35_x86_64 | 293e914645bd90ba2262c1bd75b0a0187e8d087b9af55941f1dea11df4b7ad5a | preverified, 0 jobs | pypi via alpha-api-0.8.51-nvidia-20261017 (source 958cd351aa6e, tooling 958cd351aa6e) |
| mojolearn-amd manylinux_2_35_x86_64 | 7d00dad07361b861a84f9476d8d804b0e1719fca16ce0820bde851922083246b | preverified, 0 jobs | pypi via alpha-api-0.8.51-amd-20261017 (source 958cd351aa6e, tooling 958cd351aa6e) |
| macosx_11_0_arm64 | 7bc1fe155b64d26dd763b2c85e5fce7e8db2cffd763c44960972540eb5855fb9 | PASSED, 0 jobs | pypi via alpha-api-0.8.51-macos-20261017 (source 958cd351aa6e, tooling 958cd351aa6e) |

Finish line: macOS: pip install mojolearn==0.8.51 on this Mac: OK; Linux: pip resolves mojolearn==0.8.51 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia: pip resolves mojolearn-nvidia==0.8.51 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-amd: pip resolves mojolearn-amd==0.8.51 for manylinux_2_35_x86_64; bytes are the published wheel's.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
