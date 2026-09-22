# mojolearn 0.8.13

Source commit 1eab38968604907476cc80272b50bcf6a756d568. Fixes the prediction
bindings' interpreter-lock race (quantile regression, extra trees, random forest).

## Release verification (CPU and Apple GPU, side by side, 516 s)

| column | lanes | cells | IDENTICAL | N/A | DIVERGENT | skipped probes |
|---|---|---|---|---|---|---|
| CPU | 214 | 642 | 1878 | 2508 | 0 | 642 batch, 60 rlpair |
| Apple M4 Metal | 208 | 624 | 1839 | 2421 | 0 | 624 batch, 54 rlpair |

- cpu/column.json sha256 434e1976c17fa5de3e79483cd838b6f5b0e10b4035f3dbc09bdbfb437fec13be
- metal/column.json sha256 e63aa5b1cbe83c325acd5d26fd57cebd2ff758a15b8d05877d2a02d85260aaa1

## Wheels (light route)

| platform | sha256 | smoke | tag |
|---|---|---|---|
| manylinux_2_35_x86_64 | f4a865cb86988a0d4a7ec616a00c038b16b1ac9a0b20af0af0d971e35954c43a | NVIDIA H100, 13 jobs PASSED | alpha-api-0.8.13-linux-20260921 |
| macosx_11_0_arm64 | 6a5f43f77f104baabd91fbde7f3cccd86bb198e0026b73c60bc384bcec52a962 | Apple M4, PASSED | alpha-api-0.8.13-macos-20260921 |

The Linux wheel packs the sm_89, sm_90a and gfx942 builds of the source commit; the 32 host
bindings are byte-identical across them and auditwheel repair passed.
`pip install mojolearn==0.8.13` on the M4: version 0.8.13, vendor metal, no other package required.
