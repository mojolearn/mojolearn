# mojolearn 0.8.18

Source commit f2293183c729059331cb31afcb8319f528143e55. Published 2026-09-25. See CHANGELOG.md.

## Release verification (CPU and Apple GPU, `pixi run -e test release-check`)

| column | lanes | cells | complete |
|---|---|---|---|
| CPU | 212 | 636 | yes |
| Apple Metal | 206 | 618 | yes |

CPU against Metal (`tools/identity_break.py --diff`):

    summary: IDENTICAL=618, ONE-COLUMN=18
    summary (infer/model): IDENTICAL=972, N/A=279, ONE-COLUMN=21
    summary (batch): N/A=636
    summary (rlpair): N/A=60
    summary (batchgrad): IDENTICAL=48, N/A=588
    summary (batchscale): IDENTICAL=81, N/A=549, ONE-COLUMN=6
    summary (ragged): IDENTICAL=54, N/A=576, ONE-COLUMN=6
    summary (stepfull): IDENTICAL=54, N/A=582

- cpu/column.json sha256 14a13c5351b1ab870848330fec644226cead991ab55c730185d8174b47e1ec22
- metal/column.json sha256 5d85edfd7e104a611d194a2325fe56e69e9751957e1f2d8afedd6de3d633eae0

## Wheels (light route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | c160fb6d5101f1d0ff6a4747ff11c5ed2bbaacfd522506de64ab262f6b9e3836 | PASSED, 13 jobs | pypi via alpha-api-0.8.18-linux-20260925 |
| macosx_11_0_arm64 | 8e696ed569c3ba411619c2a622a46223d79279ed8bf37ee888d3d473380063e9 | PASSED, 13 jobs | pypi via alpha-api-0.8.18-macos-20260925 |

`pip install mojolearn==0.8.18` on this Mac: pip install mojolearn==0.8.18 on this Mac: OK.
