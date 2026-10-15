# mojolearn 0.8.45

Source commit b8b7cf6f608c40f21d0976f2e80f4f49dd9aae9c; release tooling at f4c365b4ca35056ac920e09905fcd7a13f6d7a8f. Published 2026-10-14. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Identity and behavior were checked before merge on NVIDIA, AMD, Apple and CPU; this release built, checked artifact integrity and published (preverified route). The macOS wheel passed `verify --self-test` in a fresh venv.

## Wheels (preverified route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | 452b17a8b7a7f2bbd58ffa72e8e367107eb35d599ff44bc2c2a026ce07803e3f | preverified, 0 jobs | pypi via alpha-api-0.8.45-linux-20261014 (source b8b7cf6f608c, tooling b8b7cf6f608c) |
| mojolearn-nvidia manylinux_2_35_x86_64 | 092aed52995873469261169fb7c3f8c0df7b896fc2f356312719be6400db75fe | preverified, 0 jobs | pypi via alpha-api-0.8.45-nvidia-20261014 (source b8b7cf6f608c, tooling b8b7cf6f608c) |
| mojolearn-amd manylinux_2_35_x86_64 | 3894ae566cc730bd3ddcdce169bf6f50795e2b1cbde7fa3d9662ceedc160549b | preverified, 0 jobs | pypi via alpha-api-0.8.45-amd-20261014 (source b8b7cf6f608c, tooling b8b7cf6f608c) |
| macosx_11_0_arm64 | 8fcf2435a991647ff19d005bd24db66cd90e4a6fdd6dca0f1a160e59046a4710 | PASSED, 0 jobs | pypi via alpha-api-0.8.45-macos-20261015 (source b8b7cf6f608c, tooling f4c365b4ca35) |

Finish line: macOS: pip install mojolearn==0.8.45 on this Mac: OK; Linux: pip resolves mojolearn==0.8.45 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia: pip resolves mojolearn-nvidia==0.8.45 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-amd: pip resolves mojolearn-amd==0.8.45 for manylinux_2_35_x86_64; bytes are the published wheel's.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
