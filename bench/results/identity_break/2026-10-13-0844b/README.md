# Reference columns at 8f9ef96d6c60 (2026-10-13)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 1) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` from every committed column.
Source commit: `8f9ef96d6c60`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.lanes-0d3ea030.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [STABLE=54] | `4a47bf8675812ceb` |
| `amd-mi325x-gfx942/amd-mi325x-gfx942.lanes-0d3ea030.identical.json` | class=amd vendor=amd-mi325x-gfx942 [STABLE=54] | `4aaf652aa6dd348d` |
| `apple-m4/apple-m4.lanes-0d3ea030.identical.json` | class=apple vendor=apple-m4 [STABLE=54] | `6030770630b9dd77` |
| `cpu-apple-m4/cpu.lanes-0d3ea030.identical.json` | class=cpu vendor=cpu-apple-m4 [STABLE=54] | `dd88d5f0991230ba` |

Builder summary: `admission-summary.json`. Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).

Lanes: x-decomp-eigh-wide-{linalg,fastica,rotation,factor-analysis,mincovdet,kpca} (eigh at n = 72 and 128),
REPEATS=1, 54 cells per record. Records: NVIDIA L40S at 8f9ef96d6, AMD MI325X at f55fbd40f, Apple M4 and
cpu-apple-m4 at a0d92171c (the code is identical across the three commits: tools/identity_code_equal.py).
NVIDIA == AMD == Apple == cpu on all 54 cells (486 of 486 fields); 0 conflicts.
