# Reference columns at d813426e2239 (2026-10-17)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 1) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` from every committed column.
Source commit: `d813426e2239`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.lanes-52cc2a7a.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [REFUSED=1,STABLE=35] | `7f69107b4d2b5303` |
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.lanes-e2a6ca3f.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [REFUSED=2845,STABLE=1196] | `a4b378dc80d23cf0` |
| `amd-mi325x-gfx942/amd-mi325x-gfx942.lanes-52cc2a7a.identical.json` | class=amd vendor=amd-mi325x-gfx942 [REFUSED=1,STABLE=35] | `95aa5a6ff28ce0fc` |
| `amd-mi325x-gfx942/amd-mi325x-gfx942.lanes-e2a6ca3f.identical.json` | class=amd vendor=amd-mi325x-gfx942 [REFUSED=2845,STABLE=1196] | `9de56b6d75bf538e` |

Builder summary: `admission-summary.json`. Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
