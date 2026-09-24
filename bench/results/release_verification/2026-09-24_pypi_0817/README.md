# mojolearn 0.8.17

Source commit 5cc3f4e311b4e1c1bdf18f0e63d9e95cae913eab. Published 2026-09-24. See CHANGELOG.md.

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

- cpu/column.json sha256 69e01c1a30dbc0447d78853dbba01e5b0c29e18cbefc31bed01ef13ab53823ab
- metal/column.json sha256 a0e86470d8bea0a99b3d73104084f9ab3b959497bcdc1aac5eb7975cccb404e5

## Wheels (light route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | dd3899f0e28d5e1c64f3827735937a7be1731b04f37758a02b4bd13b1be66c69 | PASSED, 13 jobs | pypi via alpha-api-0.8.17-linux-20260924 |
| macosx_11_0_arm64 | b34730762d0646dac6a19a009130b86decca097e250d8e3b409a55d56762029e | PASSED, 13 jobs | pypi via alpha-api-0.8.17-macos-20260924 |

`pip install mojolearn==0.8.17` on this Mac: pip install mojolearn==0.8.17 on this Mac: OK.
