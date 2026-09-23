# mojolearn 0.8.16

Source commit a62361472007b4b720407b54289afde5e69195b7. Published 2026-09-23. See CHANGELOG.md.

## Release verification (CPU and Apple GPU, `pixi run -e test release-check`)

| column | lanes | cells | complete |
|---|---|---|---|
| CPU | 215 | 645 | yes |
| Apple Metal | 209 | 627 | yes |

CPU against Metal (`tools/identity_break.py --diff`):

    summary: IDENTICAL=627, ONE-COLUMN=18
    summary (infer/model): IDENTICAL=978, N/A=291, ONE-COLUMN=21
    summary (batch): N/A=645
    summary (rlpair): N/A=60
    summary (batchgrad): IDENTICAL=48, N/A=597
    summary (batchscale): IDENTICAL=81, N/A=558, ONE-COLUMN=6
    summary (ragged): IDENTICAL=54, N/A=585, ONE-COLUMN=6
    summary (stepfull): IDENTICAL=54, N/A=591

- cpu/column.json sha256 0a739b38a7891fa91a02ac4c6acc9a14417eadc3f2dd7425b5b31f2e35ca6b5f
- metal/column.json sha256 4578d8af1003b361a74651fe7b9fe5b1c0b0296ff3ad11bfcdab71e35d6dd4fb

## Wheels (light route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | e81b1555b601a3d3445707211aa5590eee3ac638770da88aaa5f9a5284865f00 | PASSED, 13 jobs | pypi via alpha-api-0.8.16-linux-20260923 |
| macosx_11_0_arm64 | 986d697babad3f676ce539cd6d7ab8169cc69a3dc0ba6696168dede415f01f6f | PASSED, 13 jobs | pypi via alpha-api-0.8.16-macos-20260923 |

`pip install mojolearn==0.8.16` on this Mac: pip install mojolearn==0.8.16 on this Mac: OK.
