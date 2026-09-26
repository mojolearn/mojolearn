# mojolearn 0.8.22

Source commit f02ecb8b0d123297184bae958570f4e62b1dcd7d; release tooling at daa0808b261f82de87b9d935242201aa05fc89b7. Published 2026-09-26. See CHANGELOG.md.

## Release verification (the changed lanes, one fit per cell)

| column | lanes | cells | complete |
|---|---|---|---|
| Apple Metal | 2 | 6 | yes |
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

- metal/column.json sha256 7e7524ecb1d0f7af27875c739ba6989d45c60dc5855dafe8ef5ac7969d8f76f2
- column-cuda.json sha256 1f01ca15f1c8cac7c71d950b89352a9f8f14f0a938c046a8446e8c00007e8825
- column-hip.json sha256 295b533593e911379132712bd4a722cdd7c4af61540a8da8069bb416b11894b6

## Wheels (light route)

| platform | sha256 | smoke | published |
|---|---|---|---|
| manylinux_2_35_x86_64 | b3f320079d48ea2dac460635695dc83f8a9dfcc9c1773ed3ed59b818fafbf0ff | PASSED, 13 jobs | pypi via alpha-api-0.8.22-linux-20260926 (source f02ecb8b0d12, tooling daa0808b261f) |
| mojolearn-nvidia manylinux_2_35_x86_64 | 8015d981ad3539e1718d715d970ad5ad1027ad974aa5982be2f35d11cc46ecef | PASSED, 13 jobs | pypi via alpha-api-0.8.22-nvidia-20260926 (source f02ecb8b0d12, tooling f02ecb8b0d12) |
| mojolearn-amd manylinux_2_35_x86_64 | 7c7dbafa03fec1c0fcb569dd478a0fe8caf9d9f088248cb1e423b19115c94eac | PASSED, 13 jobs | pypi via alpha-api-0.8.22-amd-20260926 (source f02ecb8b0d12, tooling daa0808b261f) |
| macosx_11_0_arm64 | bd02a7dc043e679e317c93ca6b88969f2f1e3c648903a808c6948ca53d7f897d | PASSED, 13 jobs | pypi via alpha-api-0.8.22-macos-20260926 (source f02ecb8b0d12, tooling f02ecb8b0d12) |

Finish line: macOS: pip install mojolearn==0.8.22 on this Mac: OK; Linux: pip resolves mojolearn==0.8.22 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-nvidia: pip resolves mojolearn-nvidia==0.8.22 for manylinux_2_35_x86_64; bytes are the published wheel's; mojolearn-amd: pip resolves mojolearn-amd==0.8.22 for manylinux_2_35_x86_64; bytes are the published wheel's.

Provenance of every leg and column (built, or taken and why): release-provenance.json.
