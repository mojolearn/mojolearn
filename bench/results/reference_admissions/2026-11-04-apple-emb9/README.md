# Reference columns at bb28ac829684 (2026-11-04)

Columns recorded by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 1) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` for these lanes, preserving every other row:
`bpe-trainer,bpe-vocabulary,hf-causal-lm,hf-causal-lm-stream,hf-causal-lm-text,hf-model-surface,hf-tokenizer,tokenized-corpus,tokenizer`. Original source records are retained whole.
Source commit: `bb28ac829684`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `bench/results/identity_break/retained/5b8ab7cbd03dc94064d0d6f54134be898987b892136a8f7b3c1cfb45b8d3f272/apple-m4.lanes-b12d9b04.identical.json` | class=apple vendor=apple-m4 [STABLE=81] | `5b8ab7cbd03dc940` |

Builder summary: `admission-summary.json`; selected GPU and cpu agreement: `agreement-summary.json`.
Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
