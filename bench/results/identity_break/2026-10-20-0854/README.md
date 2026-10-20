# Reference columns at 661016f7c769 (2026-10-20)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 1) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` from every committed column.
Source commit: `661016f7c769`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `amd-mi325x-gfx942/amd-mi325x-gfx942.lanes-e2a6ca3f.identical.json` | class=amd vendor=amd-mi325x-gfx942 [REFUSED=1,STABLE=4040] | `912ff1ec69a47a19` |
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.s0of4.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [STABLE=1017] | `f6914c937b60a075` |
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.s2of4.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [REFUSED=1,STABLE=1007] | `b03f03969c9fc366` |
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.s3of4.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [STABLE=1008] | `e6836b76b8a57db8` |
| `amd-mi325x-gfx942/amd-mi325x-gfx942.lanes-9044db46.identical.json` | class=amd vendor=amd-mi325x-gfx942 [STABLE=36] | `7c5297e06e39c1b3` |
| `nvidia-rtx4090-sm_89/nvidia-rtx4090-sm_89.lanes-9044db46.identical.json` | class=nvidia vendor=nvidia-rtx4090-sm_89 [STABLE=36] | `6a5eb54f33560d93` |

Builder summary: `admission-summary.json`. Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
