# Numeric feature fraction — initial implementation evidence

2026-09-10, Apple M4 / Mojo 1.0.0 ed45d567, Python 3.13.15,
NumPy 2.5.2, sklearn 1.8.0. Initial source baseline `7620749b`;
shared RF/ET helper landed separately at `0e012669`.

All three GBDT bindings built with `MOJOLEARN_SKIP_BUILD_GATE=1`; the existing
broad build gate was skipped in favor of focused checks. Raw build logs remain.
`python_tests.log`: 98 parameter/ABI/protocol checks passed.
`native-fast.log`: mixed 1/4/8-bit packed-bin oracle over 55 original features,
constant/excluded-first columns, sampler count/default-RNG checks, and ordinary
versus prepared fits across three policies passed. Initial compile failure
(non-copyable layout ternary) is preserved separately and was fixed.

`public.smoke.log`: 147 small fits/check cells pass across three modes, three
growth policies, RMSE/Logloss, default/half/tiny-positive fractions, repeats,
adapter model/raw-prediction agreement and weighted evaluation with class
weights and both eligibility controls. Enabled models differ from full-feature
models. Eighteen default fingerprints match the pre-build public capture;
another 18 default models/predictions in `default_before.json` and
`default_after.json` match byte-for-byte on a separate fixture with a constant
column. `sampled_before_reuse.json` retains 18 mixed-layout model/prediction
fingerprints for subsequent allocation reuse changes.

The public smoke emitted 10 `Context leak detected, CoreAnalytics returned
false` diagnostics, while assertions and process exit succeeded. Their origin
and resource impact remain unresolved; this run does not qualify long-run
stability. No CUDA/HIP identity result is established by these local checks.

## Exploratory timing, not a promotion result

`timing_probe.py` / `timing.json` / `timing.log`: public whole-fit time for
8,192 training rows, 32 features, eight RMSE trees, depth six, 32 borders.
One warmup per arm, then three rounds with rotated arm order in one process.
Fractions are 1, 0.5 and 0.25. A 1,024-row heldout set records independent
RMSE; sampling changes the learner and quality, so this is not an equal-model
speed comparison. The reference spread cutoff is 1.1.

| Mode / policy | Half / full median time | Quarter / full | Reference spread |
| --- | ---: | ---: | ---: |
| FAST symmetric | 1.136 | 1.116 | 1.199 — unstable |
| FAST Depthwise | 1.138 | 1.125 | 1.080 |
| FAST Lossguide | 1.052 | 1.101 | 1.027 |
| IDENTICAL symmetric | 1.115 | 1.101 | 1.033 |
| IDENTICAL Depthwise | 1.030 | 1.021 | 1.154 — unstable |
| IDENTICAL Lossguide | 1.025 | 1.034 | 1.024 |

Ratios above one are slower. This small workload establishes no sampling
speedup; two of six windows are unstable, and small differences remain within
observed noise even in some other windows. Heldout RMSE also worsens under
sampling on this fixture (full approximately 0.44–0.48, half 0.65–0.67,
quarter 0.89–0.90). No default changes. The user requested AMD/NVIDIA evidence
before performance decisions; these local measurements remain exploratory.

The initial path allocates projection/staging and clears cached search
workspaces per sampled tree. Next investigate capacity reuse and original-layout
mapped histogram access. Packing happens once per sampled tree, not per depth
or leaf. A shape-aware allocation improvement must preserve the same sampled
model bits and be compared against this implementation in one process.
