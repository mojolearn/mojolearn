# mojolearn 0.8.52

Source commit 95c41f70bff4bf195a2616150e012cf832e13230; release tooling at ef242756af2c3080263e991972975d7593f6a2de. Published 2026-10-18. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Identity and behavior were checked before merge on NVIDIA, AMD, Apple and CPU; this release built, checked artifact integrity and published (preverified route). The macOS wheel passed `verify --self-test` in a fresh venv.

## Wheels (preverified route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | c0e3d7b9e300ad953e96877db1190c843651184c4d27e363372a186dda02411f | preverified, 0 jobs | pypi via alpha-api-0.8.52-linux-20261018 (source 95c41f70bff4, tooling ef242756af2c) |
| mojolearn-nvidia manylinux_2_35_x86_64 | b9975b051982bcd29248b302a4717f2c7dc961e089cc4976be7fb0640a4579c8 | preverified, 0 jobs | pypi via alpha-api-0.8.52-nvidia-20261018 (source 95c41f70bff4, tooling ef242756af2c) |
| mojolearn-amd manylinux_2_35_x86_64 | aa44d3e8da13061c8527a9fef3fcb809814a29ba93fd205af6d6cc10bbec1b1e | preverified, 0 jobs | pypi via alpha-api-0.8.52-amd-20261018 (source 95c41f70bff4, tooling ef242756af2c) |
| macosx_11_0_arm64 | 0e0e7d66e17cfb5d40d131639db1f83bfdf642d03e45d69d8c71a009d25e59f7 | PASSED, 0 jobs | pypi via alpha-api-0.8.52-macos-20261018 (source 95c41f70bff4, tooling 95c41f70bff4) |

Finish line: macOS: pip install mojolearn==0.8.52 on this Mac: OK; Linux: pip resolves mojolearn==0.8.52 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia: pip resolves mojolearn-nvidia==0.8.52 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-amd: pip resolves mojolearn-amd==0.8.52 for manylinux_2_35_x86_64; bytes are the published wheel's.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
