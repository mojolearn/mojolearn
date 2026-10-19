# mojolearn 0.8.53

Source commit fdf04b801ca818639e9f5cf1e68a46019ce0b338; release tooling at bb1c87bf2e30164a62852586c1a4e4acfcc487cf. Published 2026-10-19. See CHANGELOG.md.

## Identity

One identity check per release: the admitted reference table (python/mojolearn/verify_reference/table.json, NVIDIA == AMD, tools/admit_identity_columns.sh). Identity and behavior were checked before merge on NVIDIA, AMD, Apple and CPU; this release built, checked artifact integrity and published (preverified route). The macOS wheel passed `verify --self-test` in a fresh venv.

## Wheels (preverified route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | a66c137db63d25a31f97e6e6317a1c9ef32bbac953b42db4ae44db5b7b2ca291 | preverified, 0 jobs | pypi via alpha-api-0.8.53-linux-20261019 (source fdf04b801ca8, tooling bb1c87bf2e30) |
| mojolearn-nvidia manylinux_2_35_x86_64 | 824047b5fa623f1e2c0b70d01f67cec69391973fb437e4db0502d26e19ee3e6c | preverified, 0 jobs | pypi via alpha-api-0.8.53-nvidia-20261019 (source fdf04b801ca8, tooling bb1c87bf2e30) |
| mojolearn-amd manylinux_2_35_x86_64 | 238ca266afea382136bd22f37550440ab134e90e1f97b347a92088262e74aad1 | preverified, 0 jobs | pypi via alpha-api-0.8.53-amd-20261019 (source fdf04b801ca8, tooling bb1c87bf2e30) |
| mojolearn-nvidia-ptx manylinux_2_35_x86_64 | 3727c10bf1bb3062705312acf4cc2f0d849ba2d4586f64d577a5834cb0d93f2e | preverified, 0 jobs | already on PyPI |
| macosx_11_0_arm64 | 96003edbba03fe533690a0c21b8b4958c83237c9a44684e74d40271881415420 | PASSED, 0 jobs | already on PyPI |

Finish line: macOS: pip install mojolearn==0.8.53 on this Mac: OK; Linux: pip resolves mojolearn==0.8.53 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia: pip resolves mojolearn-nvidia==0.8.53 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia-ptx: pip resolves mojolearn-nvidia-ptx==0.8.53 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-amd: pip resolves mojolearn-amd==0.8.53 for manylinux_2_35_x86_64; bytes are the published wheel's.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
