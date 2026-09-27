# mojolearn 0.8.24

Source commit c8654671b8484adb3862d16f7f0307cc39df0c42; release tooling at c8654671b8484adb3862d16f7f0307cc39df0c42. Published 2026-09-27. See CHANGELOG.md.

## Release verification (the changed lanes, one fit per cell)

| column | lanes | cells | complete |
|---|---|---|---|
| Apple Metal | 202 | 606 | yes |
| NVIDIA (installed wheel) | 0 | 0 | no |
| AMD (installed wheel) | 0 | 0 | no |

Every column together (`tools/identity_break.py --diff`, diff-columns.txt):


- metal/column.json sha256 8651d1e020e91dd8b852a7c4b66df1bf495abecf20e03c8f1bf472cf53b53db8

## Wheels (light route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| macosx_11_0_arm64 | 47c6390e575ddb7bd3bbd5cfac613ea33fc0eb21d87809a17a9a81e6b0cdc8c5 | PASSED, 13 jobs | pypi via alpha-api-0.8.24-macos-20260927 (source c8654671b848, tooling c8654671b848) |

Finish line: macOS: pip install mojolearn==0.8.24 on this Mac: OK; not yet published: linux, nvidia, amd.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
