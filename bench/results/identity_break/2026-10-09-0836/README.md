# Reference columns at 64cb8caa9b7c (2026-10-09)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 2) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` from every committed column.
Source commit: `64cb8caa9b7c155e96431d38c9785bb47ce53981`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [DIVERGENT=8,STABLE=3943] | `f6fbfc77cffaa359` |
| `amd-mi325x-gfx942/amd-mi325x-gfx942.identical.json` | class=amd vendor=amd-mi325x-gfx942 [DIVERGENT=8,STABLE=3943] | `b687a72ab8f60bb7` |
| `cpu-apple-m4/cpu.identical.json` | class=cpu vendor=cpu-apple-m4 [N/A=297,REFUSED=18,STABLE=3636] | `271ed52b999a350e` |

Builder summary: `admission-summary.json`. Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
