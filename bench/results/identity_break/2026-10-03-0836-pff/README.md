# Reference columns at 72a65643653c (2026-10-03)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 2) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` from every committed column.
Source commit: `72a65643653c679b080ebf1eabd70449ccf4f097`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `amd-mi325x-gfx942/amd-mi325x-gfx942.lanes-34f82557.identical.json` | class=amd vendor=amd-mi325x-gfx942 [STABLE=9] | `6ad4417dee084430` |
| `apple-m4/apple-m4.lanes-34f82557.identical.json` | class=apple vendor=apple-m4 [STABLE=9] | `ed4bd773fe9cae8a` |
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.lanes-34f82557.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [STABLE=9] | `33a11a56ea104601` |

Builder summary: `admission-summary.json`. Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
