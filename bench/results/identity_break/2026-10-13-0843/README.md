# Reference columns at c0db3dad7be4 (2026-10-13)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 1) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` from every committed column.
Source commit: `c0db3dad7be4`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.lanes-96798356.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [STABLE=621] | `1fdc135c03caaf3f` |
| `amd-mi325x-gfx942/amd-mi325x-gfx942.lanes-96798356.identical.json` | class=amd vendor=amd-mi325x-gfx942 [STABLE=621] | `2af3fb550c7a4b6f` |

Builder summary: `admission-summary.json`. Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
