# Reference columns at 6b882cf81512 (2026-10-09)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 2) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` from every committed column.
Source commit: `6b882cf8151232df582f4aa74d331d507318d0c2`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.lanes-7a9bfbd1.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [STABLE=90] | `b85711628c74555e` |
| `amd-mi325x-gfx942/amd-mi325x-gfx942.lanes-7a9bfbd1.identical.json` | class=amd vendor=amd-mi325x-gfx942 [STABLE=90] | `3cf148d1a13321d0` |
| `cpu-apple-m4/cpu.lanes-7a9bfbd1.identical.json` | class=cpu vendor=cpu-apple-m4 [STABLE=90] | `60603f73e6075dfd` |

Builder summary: `admission-summary.json`. Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
