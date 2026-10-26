# Reference columns at 439199dcddf5 (2026-10-26)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 1) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` for these lanes, preserving every other row:
`hf-causal-lm-stream,hf-checkpoint,hf-model-surface,par-samba,par-samba-clip,sequence-onnx-import,sequence-program-save-load,sequence-tape-program`. Original source records are retained whole.
Source commit: `439199dcddf5`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `amd-mi325x-gfx942/amd-mi325x-gfx942.lanes-3e6af02b.identical.json` | class=amd vendor=amd-mi325x-gfx942 [STABLE=72] | `96ef06c3d7aa8520` |
| `apple-m4/apple-m4.lanes-3e6af02b.identical.json` | class=apple vendor=apple-m4 [STABLE=72] | `cd82e2eddc55d31d` |
| `cpu-amd-epyc-9575f-64-core-processor/cpu.lanes-3e6af02b.identical.json` | class=cpu vendor=cpu-amd-epyc-9575f-64-core-processor [STABLE=72] | `8daed1638369e7c5` |
| `nvidia-ptx-rtx4090-sm80/nvidia-ptx-rtx4090-sm80.lanes-3e6af02b.identical.json` | class=ptx vendor=nvidia-ptx-rtx4090-sm80 [STABLE=72] | `e864d81612174be6` |
| `nvidia-rtx4090-sm89/nvidia-rtx4090-sm89.lanes-3e6af02b.identical.json` | class=nvidia vendor=nvidia-rtx4090-sm89 [STABLE=72] | `084a76a7624ccc43` |

Builder summary: `admission-summary.json`; selected GPU and cpu agreement: `agreement-summary.json`.
Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
