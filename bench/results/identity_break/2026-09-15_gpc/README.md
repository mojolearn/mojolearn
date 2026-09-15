# GaussianProcessClassifier identity lanes (2026-09-15)

Branch lane/gaussian-process-classifier. Lanes `gpc` (binary,
`ConstantKernel(1.0) * RBF(1.0)`) and `gpc-multiclass` (one-vs-rest over three
classes, `ConstantKernel(2.0) * Matern(1.0, nu=1.5)`), 256 training rows of four
columns, 64 held-out rows, all nine fixtures. The `gp` lane rides along to show
the regressor did not move.

## Apple M4, Metal (`apple-m4.json`, `apple-m4.txt`)

Built on the Mac through the shared slot wrapper at one core
(`bash bindings/build_gp.sh`, identical), two repeats:

- train 27 of 27 cells STABLE (gp, gpc, gpc-multiclass on nine fixtures), infer
  27 STABLE, model 18 STABLE (the gp lane has no save, 9 n/a), batch 27 STABLE.
- The gp lane's cells against the 166-lane record's three GPU columns
  (`2026-09-14_166-lanes`): IDENTICAL x4 on every train, infer and batch cell.
- In one process on the Mac, the Metal fit and the CPU host binding's fit
  (`_mojolearn_gp_host`, built on the Mac) agree bit for bit on `L_`, `pi_`,
  `W_sr_`, `n_iter_` and the likelihood for both lanes' base fixture, and
  `mojolearn.host_model` on the saved file predicts the Metal probabilities and
  labels bit for bit. Rows asked alone equal the same rows in the batch.
- `python -m mojolearn.tests.test_gpc_surface` on Metal: 6 of 6 PASS.

## Reference check against scikit-learn 1.9.0 (`ref_compare.txt`)

`sk_ref.py` (bench env) fits `sklearn.gaussian_process.GaussianProcessClassifier`
with `optimizer=None` and the same kernels (bounds fixed) on float64 copies of
the lanes' float32 rows; `ref_compare.py` fits mojolearn on Metal and compares.
We did not tune to match.

| lane | labels agreeing | largest probability difference | largest likelihood difference |
|---|---|---|---|
| gpc | 576 of 576 | 5.8e-5 (`wide`), at most 2.3e-6 elsewhere | 5.6e-5 |
| gpc-multiclass | 576 of 576 | 1.6e-4 (`wide`), at most 1.1e-5 elsewhere (`denormal`) | 5.5e-5 |

The differences are float32 arithmetic against float64 (DEVIATION 2831) and the
float32 reading of the stop rule (DEVIATION 2830). On `wide` the four columns
the lanes read are scaled by 1e-4 to 4e-3, so every kernel cell is within about
1e-5 of 1 and float32's spacing near 1 (1.2e-7) is the limit.

## x86 CPU, RunPod (`runpod_cpu/2026-09-15_172819-gaussian-process-classifier`)

One CPU pod (AMD EPYC 7713, 8 vCPU) at commit a3357cd98, core and gp host
families built production and `-D MOJOLEARN_HOST_SABOTAGE=1`, deleted and
verified gone (spend $0.011).

- `cpu-x86.json`, two repeats: 27 train, 27 infer, 18 model and 27 batch cells
  STABLE.
- `diff.metal-cpu.txt`, Apple M4 Metal against x86 CPU on gpc and
  gpc-multiclass: train IDENTICAL=18, infer and model IDENTICAL=36, batch
  IDENTICAL=18.
- `diff.gate.txt`, the CPU gate's diff (the 166-lane record's three GPU columns
  and the CPU column, `--require-columns 4 --owed-json owed.json`): the gp lane
  IDENTICAL x4 on every train, infer and batch cell; the gpc lanes OWED=72 cell
  parts (18 train, 36 infer and model, 18 batch), written to `owed.json`.
- `diff.host-sabotage.txt`: DIVERGENT on all 27 train, 45 infer and model and 27
  batch cells. `owed_sabotage_check.txt`: 72 of 72 owed cell parts moved.
- `cpu-x86.batch-sabotage.txt` (`MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1`):
  batch_moved=27 of 27; `diff.batch-sabotage.txt` reads BATCH_MOVED on every
  gpc batch cell.
- `test_gpc_surface.txt`: 6 of 6 PASS on the host binding. `pytest.txt`: 136
  passed, 3 failed, none of them GP: the mixture exports test fails on
  origin/main too, and the two recording-presence tests read `bench/results`,
  which the leg does not ship.

## First RunPod CPU leg (failed, kept as `runpod_cpu/2026-09-15_172413-gaussian-process-classifier.failed-no-core-host-binding`)

Built only the gp host family. On a CPU-only box the classifier's label decoding
reads `gather_i64` from the core binding (`_mojolearn` routes to
`_mojolearn_core_host`, which ships in the wheel), so every gpc cell read
REFUSED by name. The gp lanes ran, and the batch sabotage column read
BATCH_MOVED on all nine gp fixtures. Pod deleted and verified gone (spend
$0.011).
