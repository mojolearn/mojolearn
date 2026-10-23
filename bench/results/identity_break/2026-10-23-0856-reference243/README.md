# Reference columns at 243b2478d425 (2026-10-23)

Written by `tools/record_identity_column.sh` (identity_break.py --json, identical tier,
default fixture, one device, every part, every fixture, --repeats 1) and admitted by
`tools/admit_identity_columns.sh`, which regenerated
`python/mojolearn/verify_reference/table.json` for these lanes, preserving every other row:
`classical-float16-sequences,cross-vendor-fold,et-clf-weighted,et-reg-weighted,gbdt-class-weights,gbdt-multiclass-offgrid,gp-lml-theta,hf-causal-lm-stream,hf-causal-lm-text,hf-checkpoint,hf-model-surface,ivf-short-fill,logistic-multiclass,mamba2-decode-session,mamba3-decode-session,native-schedule-series,optim-io-capability,optim-owner-lifecycle,optim-resident-io,safetensors-widen-f16,samba-optimizer-state,saved-model-host-infer-trees,sequence-adafactor-device,sequence-adamax,sequence-device-release,sequence-device-tensor,sequence-export-roundtrip,sequence-lamb-device,sequence-layernorm-backward,sequence-lr-schedulers,sequence-moe-backward,sequence-nadam,sequence-onnx-import,sequence-optimizer-state,sequence-program-save-load,sequence-tape-program,sequence-tape-sequential,training-device-cross-entropy,training-mlp-sessions,transformer-decode-load-state,x-decomp-rsvd-rank-deficient,x-huber-outliers,x-logistic-cv-proba,x-metrics-classification-report,x-metrics-permutation-test-score,x-sgd-clf-proba`. Original source records are retained whole.
Source commit: `243b2478d425`.

| record | class / vendor / cells | sha256 |
|---|---|---|
| `amd-mi325x-gfx942/amd-mi325x-gfx942.lanes-758a5371.identical.json` | class=amd vendor=amd-mi325x-gfx942 [STABLE=423] | `f7b1c2b01a336671` |
| `nvidia-rtx6000ada-sm89/nvidia-rtx6000ada-sm89.lanes-758a5371.identical.json` | class=nvidia vendor=nvidia-rtx6000ada-sm89 [STABLE=423] | `31b8c697bfbbe3bc` |
| `apple-m2-pro/apple-m2-pro.lanes-758a5371.identical.json` | class=apple vendor=apple-m2-pro [STABLE=423] | `d6acbae9f69d0fdc` |
| `cpu-amd-epyc-9575f-64-core-processor/cpu.lanes-758a5371.identical.json` | class=cpu vendor=cpu-amd-epyc-9575f-64-core-processor [REFUSED=27,STABLE=396] | `bf8af50de5cfe92e` |

Builder summary: `admission-summary.json`; selected GPU agreement: `agreement-summary.json`.
Routine-profile reference parts from this
commit vs. still older: `reference-report.json` (tools/identity_columns.py report).
