# mojolearn 0.8.24

Source commit c8654671b8484adb3862d16f7f0307cc39df0c42; release tooling at f228a1eaa0a0882b3c709058b129b27f4310e97d. Published 2026-09-27. See CHANGELOG.md.

## Release verification (the changed lanes, one fit per cell)

| column | lanes | cells | complete |
|---|---|---|---|
| Apple Metal | 202 | 606 | yes |
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

- metal/column.json sha256 8651d1e020e91dd8b852a7c4b66df1bf495abecf20e03c8f1bf472cf53b53db8
- column-cuda.json sha256 2eaa84e66b4e80d1f919a4282d9d59388d50dfdb1ccb483cc9fe34521b48877c
- column-hip.json sha256 562d8c221f19adf72ab512f15110f6d3358b33a59f24c6d66ce75aa12b6e40b6

## Wheels (light route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | ed0517a2e50cd3a7fcf0eb74d6427012a1ec7a0d1002f26f15a125a9c4eeedcc | PASSED, 13 jobs | pypi via alpha-api-0.8.24-linux-20260927 (source c8654671b848, tooling f228a1eaa0a0) |
| mojolearn-nvidia manylinux_2_35_x86_64 | ba1f3562c8fda97d21fbb55bfd0ca06f2ebe3682001da0e8c6c2ef8acda9adf7 | PASSED, 13 jobs | pypi via alpha-api-0.8.24-nvidia-20260927 (source c8654671b848, tooling f228a1eaa0a0) |
| mojolearn-amd manylinux_2_35_x86_64 | 0b709dee9308d75555e27c6cfdee3d15e3eaf513443fad7bcdfad36d26f52675 | PASSED, 13 jobs | pypi via alpha-api-0.8.24-amd-20260927 (source c8654671b848, tooling f228a1eaa0a0) |
| macosx_11_0_arm64 | 47c6390e575ddb7bd3bbd5cfac613ea33fc0eb21d87809a17a9a81e6b0cdc8c5 | PASSED, 13 jobs | pypi via alpha-api-0.8.24-macos-20260927 (source c8654671b848, tooling c8654671b848) |

Finish line: macOS: pip install mojolearn==0.8.24 on this Mac: OK; Linux: pip resolves mojolearn==0.8.24 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia: pip resolves mojolearn-nvidia==0.8.24 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-amd: pip resolves mojolearn-amd==0.8.24 for manylinux_2_35_x86_64; bytes are the published wheel's.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
