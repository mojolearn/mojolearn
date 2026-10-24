# Reference columns at f83d83933f17 (2026-10-24)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 1) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` for these lanes, preserving every other row:
`par-kernel-ridge`. Original source records are retained whole.
Source commit: `f83d83933f17`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `apple-m4/apple-m4.lanes-f29d921d.identical.json` | class=apple vendor=apple-m4 [STABLE=9] | `6d09856f484606b6` |
| `nvidia-rtx4090-sm89/nvidia-rtx4090-sm89.lanes-f29d921d.identical.json` | class=nvidia vendor=nvidia-rtx4090-sm89 [STABLE=9] | `eee19f1a59211b52` |
| `amd-mi325x-gfx942/amd-mi325x-gfx942.lanes-f29d921d.identical.json` | class=amd vendor=amd-mi325x-gfx942 [STABLE=9] | `350a14c0093200d5` |
| `cpu-amd-epyc-9575f-64-core-processor/cpu.lanes-f29d921d.identical.json` | class=cpu vendor=cpu-amd-epyc-9575f-64-core-processor [N/A=9] | `907d234a7f6419d3` |

Builder summary: `admission-summary.json`; selected GPU and cpu agreement: `agreement-summary.json`.
Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
