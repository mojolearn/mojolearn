# Reference columns at 4840cb1db47a (2026-10-03)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 2) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` from every committed column.
Source commit: `4840cb1db47a96c67c74b318e3e8512e81f02889`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `amd-mi325x-gfx942/amd-mi325x-gfx942.identical.json` | class=amd vendor=amd-mi325x-gfx942 [REFUSED=63,STABLE=3888] | `f0f1a41d30c3bffd` |
| `apple-m4/apple-m4.s0of1.identical.json` | class=apple vendor=apple-m4 [REFUSED=72,STABLE=3879] | `6b37729637a92fcc` |
| `cpu-amd-epyc-9575f-64-core-processor/cpu.s0of4.identical.json` | class=cpu vendor=cpu-amd-epyc-9575f-64-core-processor [N/A=63,REFUSED=27,STABLE=900] | `a78d6ecad3e3c016` |
| `cpu-amd-epyc-9575f-64-core-processor/cpu.s1of4.identical.json` | class=cpu vendor=cpu-amd-epyc-9575f-64-core-processor [N/A=63,STABLE=927] | `6bbe053e601e095b` |
| `cpu-amd-epyc-9575f-64-core-processor/cpu.s2of4.identical.json` | class=cpu vendor=cpu-amd-epyc-9575f-64-core-processor [N/A=90,REFUSED=9,STABLE=891] | `0a70baf6ad2a2432` |
| `cpu-amd-epyc-9575f-64-core-processor/cpu.s3of4.identical.json` | class=cpu vendor=cpu-amd-epyc-9575f-64-core-processor [N/A=81,REFUSED=18,STABLE=882] | `2670095f551e3712` |
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.lanes-e42d881e.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [REFUSED=9,STABLE=972] | `ed734b5937a15a06` |
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.s0of4.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [REFUSED=27,STABLE=963] | `316147b54d8f6bf0` |
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.s1of4.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [STABLE=990] | `17c9e8c3471af3d9` |
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.s3of4.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [REFUSED=18,STABLE=963] | `5078100678081de4` |

Builder summary: `admission-summary.json`. Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
