# Reference columns at ca8ea1f8d64b (2026-10-10)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 2) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` from every committed column.
Source commit: `ca8ea1f8d64bb09179a2bd6bd52c11b0a43b64ec`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [DIVERGENT=7,REFUSED=1,STABLE=3943] | `923a9bcf67cdb35d` |
| `amd-mi325x-gfx942/amd-mi325x-gfx942.identical.json` | class=amd vendor=amd-mi325x-gfx942 [DIVERGENT=7,REFUSED=1,STABLE=3943] | `e0f5122391d478ac` |
| `cpu-apple-m4/cpu.identical.json` | class=cpu vendor=cpu-apple-m4 [N/A=297,REFUSED=19,STABLE=3635] | `6194a64fb7b9f65c` |
| `apple-m4/apple-m4.identical.json` | class=apple vendor=apple-m4 [DIVERGENT=7,REFUSED=1,STABLE=3943] | `aede75580fb88e9a` |

Builder summary: `admission-summary.json`. Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
