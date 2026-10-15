# mojolearn 0.8.46

Source commit dedd866a183e127b5e117f66641dda764753d450; release tooling at dedd866a183e127b5e117f66641dda764753d450. Published 2026-10-15. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Identity and behavior were checked before merge on NVIDIA, AMD, Apple and CPU; this release built, checked artifact integrity and published (preverified route). The macOS wheel passed `verify --self-test` in a fresh venv.

## Wheels (preverified route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | ccc122848a66a0c946a2c70aeb4c41ca5428bd205ce05846c85cbde8809107b2 | preverified, 0 jobs | pypi via alpha-api-0.8.46-linux-20261015 (source dedd866a183e, tooling dedd866a183e) |
| mojolearn-nvidia manylinux_2_35_x86_64 | e79d2389301931fe8df79d12ee04a65bbbae913568b8d81f19657093fdae8f51 | preverified, 0 jobs | pypi via alpha-api-0.8.46-nvidia-20261015 (source dedd866a183e, tooling dedd866a183e) |
| mojolearn-amd manylinux_2_35_x86_64 | 660918f4f012bb92b178b0cc3cce8fc534c2b7ea70a84d16c8bf753558845755 | preverified, 0 jobs | pypi via alpha-api-0.8.46-amd-20261015 (source dedd866a183e, tooling dedd866a183e) |
| macosx_11_0_arm64 | b62801a43264d4ef38a087e9fd98647e9ab2225b4cf8e41b3d2086f913ca7172 | PASSED, 0 jobs | pypi via alpha-api-0.8.46-macos-20261015 (source dedd866a183e, tooling dedd866a183e) |

Finish line: macOS: pip install mojolearn==0.8.46 on this Mac: OK; Linux: pip resolves mojolearn==0.8.46 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia: pip resolves mojolearn-nvidia==0.8.46 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-amd: pip resolves mojolearn-amd==0.8.46 for manylinux_2_35_x86_64; bytes are the published wheel's.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
