# mojolearn 0.8.44

Source commit b6c49ce419c47b224b5d257440dadc6210d33151; release tooling at 2f0295555bb692ee90851f7365f14007f0588076. Published 2026-10-13. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Identity and behavior were checked before merge on NVIDIA, AMD, Apple and CPU; this release built, checked artifact integrity and published (preverified route). The macOS wheel passed `verify --self-test` in a fresh venv.

## Wheels (preverified route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | e892e8d165a608c9fdf1b4ab989df6ad90cdc2fd320bb67d78bfa01f79e441c8 | preverified, 0 jobs | pypi via alpha-api-0.8.44-linux-20261013 (source b6c49ce419c4, tooling 2f0295555bb6) |
| mojolearn-nvidia manylinux_2_35_x86_64 | a1d465f100517956198269c76d810a372e327b7318ea9accf3afc08b567f7449 | preverified, 0 jobs | pypi via alpha-api-0.8.44-nvidia-20261013 (source b6c49ce419c4, tooling 2f0295555bb6) |
| mojolearn-amd manylinux_2_35_x86_64 | fcb7cbd27f92bc58d008d10a5f41115428274de1a1211913d6544bce0f02264e | preverified, 0 jobs | pypi via alpha-api-0.8.44-amd-20261013 (source b6c49ce419c4, tooling 2f0295555bb6) |
| macosx_11_0_arm64 | cc450c2220077b5c00307b7c98b91782808017354595cfe848021ba01f7b3655 | PASSED, 0 jobs | pypi via alpha-api-0.8.44-macos-20261013 (source b6c49ce419c4, tooling 284052f0f514) |

Finish line: macOS: pip install mojolearn==0.8.44 on this Mac: OK; Linux: pip resolves mojolearn==0.8.44 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia: pip resolves mojolearn-nvidia==0.8.44 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-amd: pip resolves mojolearn-amd==0.8.44 for manylinux_2_35_x86_64; bytes are the published wheel's.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
