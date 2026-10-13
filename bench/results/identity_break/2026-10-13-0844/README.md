# Reference columns at 7fc9462a12ea (2026-10-13)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 1) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` from every committed column.
Source commit: `7fc9462a12ea`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `nvidia-l40s-sm_89/nvidia-l40s-sm_89.lanes-6759ae6a.identical.json` | class=nvidia vendor=nvidia-l40s-sm_89 [STABLE=72] | `164ff1b604a31c35` |
| `amd-mi325x-gfx942/amd-mi325x-gfx942.lanes-6759ae6a.identical.json` | class=amd vendor=amd-mi325x-gfx942 [STABLE=72] | `49f73246e5e580f0` |

Builder summary: `admission-summary.json`. Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).

Lanes re-recorded (explicit `LANES=`, REPEATS=1; every other lane keeps its admitted rows):
gbdt-feature-freq, gbdt-tensor-ctr-tables, par-feature-freq (the two-level FeatureFreq fold, 736feed37),
linalg-eigh, x-decomp-fastica, x-decomp-factor-analysis, x-decomp-robust-cov, x-neighbors-kpca
(host eigh tridiagonal route, 8d5b96fcb). NVIDIA == AMD on all 72 cells; the three FeatureFreq lanes now
equal the Apple column (2026-10-10-0837d, 2026-10-12-verify-repairs).
