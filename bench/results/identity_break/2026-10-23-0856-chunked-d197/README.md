# Reference columns at d19734e06dc0 (2026-10-23)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 1) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` for these lanes, preserving every other row:
`training-chunked-lm-head`. Original source records are retained whole.
Source commit: `d19734e06dc0`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `amd-mi325x-gfx942/amd-mi325x-gfx942.lanes-48eaadc5.identical.json` | class=amd vendor=amd-mi325x-gfx942 [STABLE=9] | `8657ce7bcb57bbd6` |
| `nvidia-rtx6000ada-sm89/nvidia-rtx6000ada-sm89.lanes-48eaadc5.identical.json` | class=nvidia vendor=nvidia-rtx6000ada-sm89 [STABLE=9] | `bc8e761d4e233e63` |
| `apple-m2-pro/apple-m2-pro.lanes-48eaadc5.identical.json` | class=apple vendor=apple-m2-pro [STABLE=9] | `53cf71e2f22ccfaa` |
| `cpu-amd-epyc-9575f-64-core-processor/cpu.lanes-48eaadc5.identical.json` | class=cpu vendor=cpu-amd-epyc-9575f-64-core-processor [STABLE=9] | `cd177696d0558415` |

Builder summary: `admission-summary.json`; selected GPU agreement: `agreement-summary.json`.
Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
