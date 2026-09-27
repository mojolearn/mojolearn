# mojolearn 0.8.23

Source commit d38e13aff90d8f1efd475c48934c363cf03cfc5a; release tooling at d38e13aff90d8f1efd475c48934c363cf03cfc5a. Published 2026-09-26. See CHANGELOG.md.

## Release verification (the changed lanes, one fit per cell)

| column | lanes | cells | complete |
|---|---|---|---|
| Apple Metal | 209 | 627 | yes |
| NVIDIA (installed wheel) | 0 | 0 | no |
| AMD (installed wheel) | 0 | 0 | no |

Every column together (`tools/identity_break.py --diff`, diff-columns.txt):


- metal/column.json sha256 067e936feb2567b3e45388d95c13c64c81397a98a163ce99213a301143556a97

## Wheels (light route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| macosx_11_0_arm64 | 0a2c5f9cd65365dd120806fcf54075f0661c082d26725da11557d8225544f16d | PASSED, 13 jobs | pypi via alpha-api-0.8.23-macos-20260927 (source d38e13aff90d, tooling d38e13aff90d) |

Finish line: macOS: pip install mojolearn==0.8.23 on this Mac: OK; not yet published: linux, nvidia, amd.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
