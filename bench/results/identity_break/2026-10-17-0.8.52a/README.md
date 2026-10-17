# Reference columns at f6d0e6eda181 (2026-10-17)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 1) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` from every committed column.
Source commit: `f6d0e6eda181`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.lanes-e2a6ca3f.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [DIVERGENT=25,REFUSED=1,STABLE=4015] | `2a3a5943bdcf9512` |
| `amd-mi325x-gfx942/amd-mi325x-gfx942.lanes-e2a6ca3f.identical.json` | class=amd vendor=amd-mi325x-gfx942 [DIVERGENT=25,REFUSED=1,STABLE=4015] | `8f5bab164a7ce699` |

Builder summary: `admission-summary.json`. Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
