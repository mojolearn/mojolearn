# Reference columns at ba151faf37fb (2026-11-04)

Columns recorded by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 1) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` for these lanes, preserving every other row:
`gmm,gmm-sample`. Original source records are retained whole.
Source commit: `ba151faf37fb`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `bench/results/identity_break/retained/616d3e2e7a8508da551ac1edd9a7d3053b1f7c707f6dd5ee38439485f9e4de31/apple-m4.lanes-a0701798.identical.json` | class=apple vendor=apple-m4 [STABLE=18] | `616d3e2e7a8508da` |

Builder summary: `admission-summary.json`; selected GPU and cpu agreement: `agreement-summary.json`.
Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
