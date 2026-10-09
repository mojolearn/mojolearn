# Reference columns at 4df610f3b5c7 (2026-10-09)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 2) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` from every committed column.
Source commit: `4df610f3b5c79ea7b7600783484e251f07988553`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.lanes-433ad668.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [STABLE=18] | `3316a9726e47ce63` |
| `amd-mi325x-gfx942/amd-mi325x-gfx942.lanes-433ad668.identical.json` | class=amd vendor=amd-mi325x-gfx942 [STABLE=18] | `c3d2f544b790f112` |
| `cpu-apple-m4/cpu.lanes-433ad668.identical.json` | class=cpu vendor=cpu-apple-m4 [STABLE=18] | `9aceac62624bd6a4` |

Builder summary: `admission-summary.json`. Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
