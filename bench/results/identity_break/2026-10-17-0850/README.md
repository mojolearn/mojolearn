# Reference columns at 19d137fb848d (2026-10-17)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 1) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` from every committed column.
Source commit: `19d137fb848d`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.lanes-3b43b63e.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [REFUSED=2763,STABLE=1188] | `0098ba149eb3c6b8` |
| `amd-mi325x-gfx942/amd-mi325x-gfx942.lanes-3b43b63e.identical.json` | class=amd vendor=amd-mi325x-gfx942 [REFUSED=2763,STABLE=1188] | `5dd3ed75d49bb836` |

Builder summary: `admission-summary.json`. Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
