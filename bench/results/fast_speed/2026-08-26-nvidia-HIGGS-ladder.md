# The HIGGS load ladder: the gap does NOT close with load, and the prediction was wrong

H100 80GB HBM3, FAST on both sides, GPU arms only (standing rule: on NVIDIA
and AMD we compare against the vendor's GPU path, never their CPU path).
HIGGS 11,000,000 x 28, binary. Two rungs, nested prefixes of the same data,
both scored against the SAME fixed 500,000-row tail. Three timed rounds
after one untimed warm-up, ours and every opponent alternating inside ONE
process. Mode witness 6 FAST / 0 IDENTICAL.

## The prediction, and what happened to it

Written into `2026-08-25-nvidia-trees-PARTIAL.md` BEFORE this ran, so it
could not be retrofitted:

> per-row work faster than theirs, fixed per-tree cost 5.7x theirs,
> therefore the gaps should narrow with load and the forest gap should
> narrow hardest.

**Both halves are wrong against the vendors' GPU learners.** That earlier
measurement was against CatBoost's **CPU** arm, and the relationship
reverses on the GPU: we have the LOW fixed cost and the HIGH marginal cost.
The forest gap did not narrow at all.

## gbdt-symmetric vs catboost-gpu

| rows | ours | catboost-gpu | |
|---|---|---|---|
| 1,000,000 | 636.6 ms | 1182.3 ms | **1.86x FASTER** |
| 5,000,000 | 2998.7 ms | 2687.3 ms | **1.12x slower** |

We WIN at a million rows and LOSE at five. Decomposed as `t = fixed +
marginal * rows`:

| | fixed | marginal per 1M rows |
|---|---|---|
| ours | **46.1 ms** | 590.5 ms |
| catboost-gpu | 806.0 ms | **376.3 ms** |
| ours / theirs | **0.06x** | 1.57x |

**Our fixed cost is SEVENTEEN TIMES LOWER than CatBoost's. Our marginal
cost is 1.57x higher.** CatBoost's GPU learner pays a large setup that
amortizes away; ours barely has one, and then does more work per row.

**Crossover at ~3.55M rows.** Below it we are faster, above it they are.
That is the single most useful sentence this lane has produced, and it
replaces "we are 1.17x behind CatBoost", which was one point on a curve
mistaken for the curve.

**THE FIT IS THROUGH TWO POINTS AND IS THEREFORE EXACT AND UNVALIDATED.**
A linear model through two rungs cannot be wrong at those rungs and cannot
be checked between them. A 2,000,000-row rung would test it, and until one
runs the crossover is an extrapolation from two measurements, not a
measured quantity.

## rf vs cuml-rf-gpu

| rows | ours | cuml-rf-gpu | |
|---|---|---|---|
| 1,000,000 | 6188.4 ms | 3079.5 ms | 2.01x slower |
| 5,000,000 | 14698.9 ms | 7057.5 ms | 2.08x slower |

**FLAT.** The ratio does not move between one million rows and five, which
says the forest gap is not a fixed-cost story at all:

| | fixed | marginal per 1M rows |
|---|---|---|
| ours | 4060.8 ms | 2127.6 ms |
| cuml-rf-gpu | 2084.9 ms | 994.5 ms |
| ours / theirs | 1.95x | 2.14x |

Both components are about 2x theirs, which is exactly why the ratio is
flat. `ensemble/` is a PORT OF cuML's forest and it is uniformly twice the
cost of the original, per row and per fit. There is no size at which this
closes on its own. It closes when the kernels get faster.

## et: cuML ships no ExtraTrees, so there is no legal opponent

| rows | ours |
|---|---|
| 1,000,000 | 4146.7 ms |
| 5,000,000 | 14446.1 ms |

`cuml-et-gpu` refused by name. Under the GPU-path-only rule scikit-learn on
the host CPU is not a substitute, so this lane has NO opponent on NVIDIA.
That is a finding about the vendor's GPU coverage, not a gap in this run.

## Accuracy: the fits are the same fits

This is what makes the timings mean anything. Same data, same held-out tail,
both rungs:

| lane | rows | ours logloss | theirs | ours AUC | theirs |
|---|---|---|---|---|---|
| gbdt-symmetric | 1M | **0.542133** | 0.542202 | **0.800447** | 0.800372 |
| gbdt-symmetric | 5M | **0.542247** | 0.542424 | **0.800668** | 0.800664 |
| rf | 1M | 0.535736 | **0.535689** | 0.809906 | **0.809834** |
| rf | 5M | 0.538850 | **0.538814** | 0.812881 | **0.812976** |

Symmetric boosting: we are very slightly BETTER than CatBoost on logloss at
both rungs. Random forest: identical to cuML to within 5e-5 on every metric.
Nobody is buying speed with accuracy in either direction.

## What is owed

* **A 2,000,000-row rung**, to turn the crossover from a two-point
  extrapolation into a measurement.
* **`ensemble/` needs a kernel pass, not a size.** 2.14x marginal against
  the library it was ported from is the number to attack.
* **`lightgbm-cuda` was skipped on this leg** (`--no-lgbm-cuda`) because its
  source build is 15-30 minutes of a 60-minute lease. It is not an opponent
  in these three lanes anyway, but the lossguide lane still owes it.
* **depthwise and lossguide have not climbed the ladder** -- this leg ran
  symmetric, rf and et only.
