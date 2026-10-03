# Reference columns at e4e9004af95b (2026-10-03)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 2) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` from every committed column.
Source commit: `e4e9004af95b1ea87457130ebb209637c51a9bed`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `amd-mi325x-gfx942/amd-mi325x-gfx942.lanes-378b36ef.identical.json` | class=amd vendor=amd-mi325x-gfx942 [REFUSED=9,STABLE=72] | `a8df7641e5238b4f` |
| `apple-m4/apple-m4.lanes-378b36ef.identical.json` | class=apple vendor=apple-m4 [STABLE=81] | `03fc04b7714f8fd1` |
| `cpu-amd-epyc-9575f-64-core-processor/cpu.lanes-378b36ef.identical.json` | class=cpu vendor=cpu-amd-epyc-9575f-64-core-processor [STABLE=81] | `49f0c8ceca69e054` |
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.lanes-378b36ef.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [STABLE=81] | `5bfe5745c376e981` |

Builder summary: `admission-summary.json`. Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
