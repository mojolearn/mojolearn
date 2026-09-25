# mojolearn 0.8.19

Source commit 69a519c1522d245f197d7eb1d2d06e1b0cb6ad0e. Published 2026-09-25. See CHANGELOG.md.

## Release verification (the changed lanes, one fit per cell)

| column | lanes | cells | complete |
|---|---|---|---|
| Apple Metal | 209 | 627 | yes |
| NVIDIA (installed wheel) | 209 | 627 | yes |
| AMD (installed wheel) | 209 | 627 | yes |

Every column together (`tools/identity_break.py --diff`, diff-columns.txt):

    summary: IDENTICAL=627
    summary (infer/model): IDENTICAL=984, N/A=270
    summary (batch): N/A=627
    summary (rlpair): N/A=54
    summary (batchgrad): IDENTICAL=48, N/A=579
    summary (batchscale): IDENTICAL=81, N/A=546
    summary (ragged): IDENTICAL=54, N/A=573
    summary (stepfull): IDENTICAL=54, N/A=573

- metal/column.json sha256 f7c9de6183be4a3fa83b8a93f3fbb4b41744042ecffdb86a1bdfbf71958df616
- column-cuda.json sha256 033eee28be5888b8304b0ede54027e3f4874650ff12b4b1aad4a403967ce205e
- column-hip.json sha256 c49c62eb377f89795201d34996d2090a1528569182e752ec7c3de30609692d92

## Wheels (light route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | 2835939fdf8475fd8cd16a5a7cee75a29a15affd05b60f13f228c54be36460ce | PASSED, 13 jobs | already on PyPI |
| macosx_11_0_arm64 | 36f94ee23a4cab7e9547c9292415f65e4f830c4f44557045d12b3e5362a0fd69 | PASSED, 13 jobs | pypi via alpha-api-0.8.19-macos-20260925 |

`pip install mojolearn==0.8.19` on this Mac: pip install mojolearn==0.8.19 on this Mac: OK.
