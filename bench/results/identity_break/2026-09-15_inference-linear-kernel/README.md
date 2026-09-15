# Saved-model CPU inference: scalers, lasso and elasticnet, kernel methods, and the ols, ridge and logistic variants

lane/inference-linear-svm, 2026-09-15. Apple M4, one core, shared machine.
No GPU box was rented. Lanes (17): ols-no-intercept, ols-weighted,
ridge-no-intercept, logistic-l1, logistic-elasticnet,
logistic-unpenalized-no-intercept, standard-scaler, standard-scaler-no-mean,
standard-scaler-no-std, minmax-scaler, minmax-scaler-clip, lasso, elasticnet,
elasticnet-l2end-no-intercept, kernel-ridge, nystroem, rbf-sampler.
Fixtures: base, odd, denormal.

## Files

| file | what it is |
|---|---|
| `apple-m4-metal.json` | identity_break column on the M4 Metal IDENTICAL set, 2 repeats |
| `cpu-apple-m4.json` | the CPU column: host bindings estimators, preprocessing, solver, kernel_methods, core, linalg built from this tree, 2 repeats |
| `cpu-apple-m4-sabotage.json` | the same six families built with `-D MOJOLEARN_HOST_SABOTAGE=1`, 1 repeat |
| `diff_cpu_vs_166-lanes_require4.txt` | `identity_break --diff` of the CPU column against the 166-lane record's Apple, H100 and MI325X columns, `--require-columns 4 --owed-json`. The three GPU columns were filtered to base, odd and denormal (the fixtures the CPU column ran) and the copies read from `<scratchpad>/inf/cols3/`; the cells are the committed hashes unchanged. Exit 0. |
| `owed_cells.json` | the 33 model cells no GPU record hashes (the new `save` formats of 11 lanes on 3 fixtures) |
| `diff_metal_vs_cpu.txt` | Metal column against the CPU column. Exit 0. |
| `diff_cpu-sabotage_vs_166-lanes.txt` | the sabotage column against the three GPU columns. Exit 1, as it must. |
| `classical_host_record_metal.txt` | `tools/classical_host_gate.py record` on the Metal set: `bench/results/classical_host/2026-09-15-apple-m4-linear-kernel` |
| `classical_host_check_cpu.txt` | `check` of that recording through `mojolearn.host_model` on the CPU-only package copy and the host set, with the three 166-lane columns as `--gpu-column` |
| `classical_host_check_sabotage.txt` | the same check on the sabotage set, `--expect-mismatch` |
| `classical_host_check_sabotage_BEFORE_transform_arm.txt` | the sabotage check before the scaler transform arm existed: the five scaler lanes read EQUAL, the negative control that could not fail |
| `builds.txt` | build durations and the estimators host binding size before (origin/main) and after |

## Verdict lines

- CPU vs 166-lane columns, require 4: `summary: IDENTICAL=51`,
  `summary (infer/model): IDENTICAL=69, OWED=33`, `summary (batch): IDENTICAL=51`.
- Metal vs CPU: `IDENTICAL=51`, infer/model `IDENTICAL=102`, batch `IDENTICAL=51`.
- Sabotage vs 166-lane columns: `DIVERGENT=51`, infer/model `DIVERGENT=69, ONE-COLUMN=33`,
  batch `DIVERGENT=51`.
- Saved-model check on the CPU host set: `gate verdict IDENTICAL (51 fixtures, 3 GPU columns, exit 0)`;
  every infer cell EQUAL to all three GPU columns.
- Saved-model check on the sabotage set: `gate verdict EXPECTED MISMATCH SEEN`, identity_hash
  DIFFER on all 17 lanes, 3 fixtures each. Before the scaler transform arm
  (preprocessing/host/scaler_oracle.mojo) the five scaler lanes read EQUAL there.
- Installed test wheel (`classical_host_check_installed_wheel.txt`): a local
  `pip wheel` of `python/` staged with the estimators, core and linalg host
  bindings built from this tree, installed with `pip install --target` into an
  isolated directory with no identical set (vendor cpu), and checked with
  `--package-root ''`: `gate verdict IDENTICAL (51 fixtures, 3 GPU columns, exit 0)`,
  135 surfaces and 153 GPU column comparisons EQUAL. The first attempt staged
  estimators alone and exited 2: a saved ElasticNet or Lasso predicts through
  `_mojolearn.transpose_f32` (the column-major design), which the core host
  binding serves. Core ships in every wheel, so this is a dependency to state,
  not a gap.

## Wheel size

The estimators host binding, the one shipped binary this change grows, is
534,192 bytes at origin/main and 604,592 bytes here (+70,400 bytes, macOS
arm64, one-core builds). The scaler transform sabotage arm is compile-time
only; the production binary is the same size before and after it. No
reference-only binding (preprocessing, solver, kernel_methods) enters the wheel.

## Owed to the release record

The NVIDIA and AMD saved-model recordings of these 17 lanes, and the 33 model
cells in `owed_cells.json`, which no GPU column has hashed.
