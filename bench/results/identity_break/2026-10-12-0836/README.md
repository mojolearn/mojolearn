# Reference columns at 5b0dfc1efc22 (2026-10-12)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 1) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` from every committed column.
Source commit: `5b0dfc1efc22`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.lanes-6c0cf436.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [DIVERGENT=7,REFUSED=1,STABLE=3943] | `eeb08d5d87e37aea` |
| `amd-mi325x-gfx942/amd-mi325x-gfx942.lanes-6c0cf436.identical.json` | class=amd vendor=amd-mi325x-gfx942 [DIVERGENT=7,REFUSED=1,STABLE=3943] | `cc5d02b92633696f` |

Builder summary: `admission-summary.json`. Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
