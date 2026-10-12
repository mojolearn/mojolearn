# Reference columns at a8fd6d926f58 (2026-10-12)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 1) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` from every committed column.
Source commit: `a8fd6d926f58`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `nvidia-rtx4090-sm_89/nvidia-rtx4090-sm_89.lanes-6c0cf436.identical.json` | class=nvidia vendor=nvidia-rtx4090-sm_89 [DIVERGENT=7,REFUSED=1,STABLE=3943] | `0738d0293165fff1` |
| `amd-mi325x-gfx942/amd-mi325x-gfx942.lanes-6c0cf436.identical.json` | class=amd vendor=amd-mi325x-gfx942 [DIVERGENT=7,REFUSED=1,STABLE=3943] | `1cb7f61f23d22331` |

Builder summary: `admission-summary.json`. Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
