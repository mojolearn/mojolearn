# mojolearn 0.8.20

Source commit a6bb47ae7e2a613c07c2b3b717a7165973f9ce71; release tooling at a6bb47ae7e2a613c07c2b3b717a7165973f9ce71. Published 2026-09-26. See CHANGELOG.md.

## Release verification (the changed lanes, one fit per cell)

| column | lanes | cells | complete |
|---|---|---|---|
| Apple Metal | 209 | 627 | yes |
| NVIDIA (installed wheel) | 0 | 0 | no |
| AMD (installed wheel) | 0 | 0 | no |

Every column together (`tools/identity_break.py --diff`, diff-columns.txt):


- metal/column.json sha256 5664b20e98a01e8c52d62033b7fcd2bdce4633ec4c7b18d97e7ab716de23bf36

## Wheels (light route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| macosx_11_0_arm64 | d281f89cf9836b70b48d17977ed06af675d3f1a16acd46fe4aadfe543f10127d | PASSED, 13 jobs | pypi via alpha-api-0.8.20-macos-20260926 (source a6bb47ae7e2a, tooling a6bb47ae7e2a) |

Finish line: macOS: pip install mojolearn==0.8.20 on this Mac: OK; not yet published: linux.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
