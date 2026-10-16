# mojolearn 0.8.48

Source commit 5f636ccbff6568ce097d7ec7a857fc84ad596706; release tooling at 5f636ccbff6568ce097d7ec7a857fc84ad596706. Published 2026-10-16. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Identity and behavior were checked before merge on NVIDIA, AMD, Apple and CPU; this release built, checked artifact integrity and published (preverified route). The macOS wheel passed `verify --self-test` in a fresh venv.

## Wheels (preverified route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | 1867964bcf568bc36a2ca5ad1ec71e29201f1350712e43ee3c4b15e4ecc705f8 | preverified, 0 jobs | pypi via alpha-api-0.8.48-linux-20261016 (source 5f636ccbff65, tooling 5f636ccbff65) |
| mojolearn-nvidia manylinux_2_35_x86_64 | 4493b4ddce50ece940ada49c69e86a49f795b9b6905e8db80d870ff1932dced1 | preverified, 0 jobs | pypi via alpha-api-0.8.48-nvidia-20261016 (source 5f636ccbff65, tooling 5f636ccbff65) |
| mojolearn-amd manylinux_2_35_x86_64 | 62675dacb7afe7c897143ee67d08bf5bd6a21ede57c34b048a96ce6f6b7e932e | preverified, 0 jobs | pypi via alpha-api-0.8.48-amd-20261016 (source 5f636ccbff65, tooling 5f636ccbff65) |
| macosx_11_0_arm64 | 2904dc50d315a0104971cda6b2c987370d77f0f6213b935f5dbe2ec4de823329 | PASSED, 0 jobs | pypi via alpha-api-0.8.48-macos-20261016 (source 5f636ccbff65, tooling 5f636ccbff65) |

Finish line: macOS: pip install mojolearn==0.8.48 on this Mac: OK; Linux: pip resolves mojolearn==0.8.48 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia: pip resolves mojolearn-nvidia==0.8.48 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-amd: pip resolves mojolearn-amd==0.8.48 for manylinux_2_35_x86_64; bytes are the published wheel's.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
