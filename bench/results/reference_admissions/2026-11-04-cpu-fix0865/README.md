# Reference columns at 58f304919890 (2026-11-04)

Columns recorded by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 1) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` for these lanes, preserving every other row:
`bpe-trainer,bpe-vocabulary,hf-causal-lm,hf-causal-lm-stream,hf-causal-lm-text,hf-model-surface,hf-tokenizer,tokenized-corpus,tokenizer`. Original source records are retained whole.
Source commit: `58f304919890`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `bench/results/identity_break/retained/a43ec73cd12d9d2e89c47a884a5678ece90f0cf690159a992b542d88559929fc/cpu.lanes-290621ed.identical.json` | class=cpu vendor=cpu-amd-epyc-75f3-32-core-processor [STABLE=81] | `a43ec73cd12d9d2e` |

Builder summary: `admission-summary.json`; selected GPU and cpu agreement: `agreement-summary.json`.
Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
