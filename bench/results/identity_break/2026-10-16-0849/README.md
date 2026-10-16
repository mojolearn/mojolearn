# Reference columns at 5f636ccbff65 (2026-10-16)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 1) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` from every committed column.
Source commit: `5f636ccbff65`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `amd-mi300x-gfx942/amd-mi300x-gfx942.lanes-127b645d.identical.json` | class=amd vendor=amd-mi300x-gfx942 [STABLE=225] | `d18b58cb98340c4c` |
| `apple-m4/apple-m4.lanes-127b645d.identical.json` | class=apple vendor=apple-m4 [STABLE=225] | `bd1dac65a5ebc0a4` |
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.lanes-127b645d.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [STABLE=225] | `35e6a422ac7ac86b` |

Builder summary: `admission-summary.json`. Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
