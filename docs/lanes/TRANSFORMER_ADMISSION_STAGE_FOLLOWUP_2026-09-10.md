# Transformer admission: stage localization follow-up

The retained [full-table comparisons](../../bench/results/transformer_admission_2026-09-10/README.md)
show that both our FP32 output and Torch FP32 fail the original 5e-4/1e-5
threshold against FP64 using the same production FP32 RoPE constants. This
rules out claiming that shared constants alone fix admission. It does not
identify the dominant remaining arithmetic stage.

The opt-in `--stage-errors` diagnostic adds three kinds of measurements using
the original benchmark's `LlamaEager` methods without rewriting its arithmetic:

- Cumulative Torch FP32 versus matched FP64 errors at norm1, attention output
  projection, first residual, norm2, MLP down projection and final residual.
- Local FP64 replay of attention/output projection, norm2 and MLP using each
  stage's actual FP32 input, plus isolated residual-add rounding. These hold
  incoming activation words fixed to expose error created inside that stage.
- The local FP64 output versus the fully FP64 pipeline output. Both use FP64
  stage arithmetic; their difference exposes propagation of upstream errors.

The reported `passed` fields retain the original output tolerance for consistent
diagnostic summaries. They do not establish a new intermediate-stage contract.
These are Torch self-comparisons, not direct measurements of our intermediate
stages. No production arithmetic, benchmark default, tolerance or timing arm
changes. No new opponent measurement or performance ratio is produced.

Root-only host checks (no binding or Torch import):

```sh
python -m unittest discover -s tools/tests -p test_transformer_admission_report.py -v
python -m py_compile tools/transformer_admission_diagnose.py tools/transformer_admission_stages.py
```

Numerical follow-up, only when root grants an existing GPU slot: retain the
original companion spec, own `.npy` output and decompressed native full RoPE
log from the linked evidence, then run each original shape separately:

```sh
python tools/transformer_admission_diagnose.py --spec /path/to/original_speed_torch_seq.py --shape narrow --arm reference --output /path/to/narrow.npy --full-rope-log /path/to/rope_full.log --reference64 --stage-errors
python tools/transformer_admission_diagnose.py --spec /path/to/original_speed_torch_seq.py --shape wide --arm reference --output /path/to/wide.npy --full-rope-log /path/to/rope_full.log --reference64 --stage-errors
```

Source is proposed; these new checks have not been run by the author. Host
reporting checks cannot qualify the GPU stage comparisons. Full production
intermediate exports would still be needed to attribute our own residuals.
