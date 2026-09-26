# mojolearn 0.8.20

Source commit a6bb47ae7e2a613c07c2b3b717a7165973f9ce71; release tooling at eaa6edc718eead7862d2fd0d418e155b31e824d6. Published 2026-09-26. See CHANGELOG.md.

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

- metal/column.json sha256 5664b20e98a01e8c52d62033b7fcd2bdce4633ec4c7b18d97e7ab716de23bf36
- column-cuda.json sha256 cd03b55e28034269fa9ae266e1f5f26fbea562487b2c2031bb1dcf4d8a5fefd5
- column-hip.json sha256 871d77032f81b30edf2687882bb95c48881d6f19ecf42ef224603b08c6b4aa2e

## Wheels (light route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | 2598fc8d2d6f1556605d8663bb1768529fded38c4dfb0fa59b92054206b62245 | PASSED, 13 jobs | pypi via alpha-api-0.8.20-linux-20260926 (source a6bb47ae7e2a, tooling eaa6edc718ee) |
| macosx_11_0_arm64 | d281f89cf9836b70b48d17977ed06af675d3f1a16acd46fe4aadfe543f10127d | PASSED, 13 jobs | pypi via alpha-api-0.8.20-macos-20260926 (source a6bb47ae7e2a, tooling a6bb47ae7e2a) |

Finish line: macOS: pip install mojolearn==0.8.20 on this Mac: OK; Linux: pip resolves mojolearn==0.8.20 for manylinux_2_35_x86_64; bytes are the published wheel's.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
