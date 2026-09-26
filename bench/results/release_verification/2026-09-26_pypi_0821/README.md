# mojolearn 0.8.21

Source commit b950daaa39778f9b9171f9f3f69d01d8da474bfe; release tooling at b950daaa39778f9b9171f9f3f69d01d8da474bfe. Published 2026-09-26. See CHANGELOG.md.

## Release verification (the changed lanes, one fit per cell)

| column | lanes | cells | complete |
|---|---|---|---|
| Apple Metal | 37 | 111 | yes |
| NVIDIA (installed wheel) | 0 | 0 | no |
| AMD (installed wheel) | 0 | 0 | no |

Every column together (`tools/identity_break.py --diff`, diff-columns.txt):


- metal/column.json sha256 a1712185f5423faf017fc624cdf2f2474940d681003ed1017c4c5865a6fcfcac

## Wheels (light route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | e9a5269267158d1fbe4da8098ff73e6ebfffa12d57244e44fed9fc90c2f054a1 | ?, 0 jobs | not yet published |
| mojolearn-nvidia manylinux_2_34_x86_64.manylinux_2_35_x86_64 | b274217e05e645b15db94cac948f4b38e830a24857e8ef21038479e63f131aa7 | ?, 0 jobs | not yet published |
| mojolearn-amd manylinux_2_34_x86_64.manylinux_2_35_x86_64 | 2501e5807c465815cc4e6f142ae77614f38d8ba7451d36c2ab74f7fa36ef588f | ?, 0 jobs | not yet published |
| macosx_11_0_arm64 | 1aad52d5af8eb77df8866f05e0e24ee30efca6735deeded531fd34435a8a733a | PASSED, 13 jobs | pypi via alpha-api-0.8.21-macos-20260926 (source b950daaa3977, tooling b950daaa3977) |

Finish line: macOS: pip install mojolearn==0.8.21 on this Mac: OK; not yet published: linux, nvidia, amd.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
