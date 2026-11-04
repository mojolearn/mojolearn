# Reference columns at 87a42d01a789 (2026-11-04)

Columns recorded by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 1) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` for these lanes, preserving every other row:
`gmm,gmm-sample`. Original source records are retained whole.
Source commit: `87a42d01a789`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `bench/results/identity_break/retained/420a6264eb35399d88ed47b170b390e5eaf7ff37db22ef2827a582b3af1e8b3a/apple-m4.lanes-47373e7a.identical.json` | class=apple vendor=apple-m4 [STABLE=3060] | `420a6264eb35399d` |

Builder summary: `admission-summary.json`; selected GPU and cpu agreement: `agreement-summary.json`.
Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
