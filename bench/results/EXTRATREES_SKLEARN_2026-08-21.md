# ExtraTrees on the GPU against scikit-learn: 0.57-1.30x their ten cores, at matched quality

2026-08-21/22, M4 base (10 GPU cores, 10 CPU cores, 16GB), scikit-learn 1.9.0,
numpy 2.5.2. Arms INTERLEAVED IN ONE PROCESS via Mojo's Python interop
(`extratrees/bench/sklearn_interleaved.mojo`), because this box drifts -- 1.7x
measured on the same binary and the same fixture twenty minutes apart. Only the
ratio inside a window is comparable; the absolute numbers here are not
comparable to the ones in earlier sections of this file's history.

**What is being compared.** Our GPU fit against scikit-learn's CPU fit, on the
same bytes, at the same parameters. No GPU implementation of this estimator
runs on this machine at all -- cuML's ExtraTrees is an unlanded design (FEA
#8133) and cuML does not run on Apple silicon; scikit-learn's tree estimators
have no Array API dispatch. This is a claim about what a user on this machine
can reach, not a claim about beating cuML.

**Three arms.** scikit-learn's own default is `n_jobs=None`, one thread, and
that is the arm whose parameters match ours exactly. A user who types
`n_jobs=-1` gets all ten cores. Reporting only the first would be choosing the
flattering comparison, so both run, alternating with ours, in the same window.
THE TEN-CORE COLUMN IS THE HEADLINE.

Both sides: `bootstrap=False`, `criterion` matched, `min_samples_split=2`,
`min_samples_leaf=1`, same `max_depth`, same `max_features`, same tree count.
`random_state` is set on both, but the RNGs are different designs (sequential
xorshift per split there, counter-based and keyed here), so the forests are
not the same forest and no bit-identity is claimed. Accuracy/MSE and NODE COUNT
are what say the speed was not bought with a worse model.

## Classification -- covtype 581,012 x 54, 7 classes, 10 trees, depth 12, 3 reps

| `max_features` | ours ms/tree | vs 1 core | **vs 10 cores** | our acc | their acc | our nodes | their nodes |
|---|---|---|---|---|---|---|---|
| 5 (log2) | 106-121 | 1.96-2.15x | 0.63-0.71x | 0.686-0.692 | 0.660-0.716 | 20,964-24,302 | 20,558-24,052 |
| 7 (sqrt, their default) | 113-118 | 2.47-2.54x | 0.80-0.84x | 0.679-0.707 | 0.683-0.736 | 23,726-25,016 | 22,428-24,434 |
| 14 | 152-158 | 3.17-3.51x | **1.00-1.03x** | 0.730-0.757 | 0.737-0.749 | 27,468-30,118 | 27,962-29,446 |
| 27 | 216-221 | 3.57-3.97x | **1.11-1.23x** | 0.770-0.775 | 0.762-0.776 | 29,446-31,694 | 30,928-32,872 |
| 54 (all) | 368-392 | 3.74-3.95x | **1.18-1.22x** | 0.791-0.800 | 0.795-0.801 | 32,920-33,964 | 32,966-34,776 |

The node counts match theirs at every setting, which is what makes the speed
column mean anything. Before DEVIATION 205 they did not: at `max_features=5`
we built 3,798-5,818 nodes against their 20,558-24,052, and "won" the speed
column at 0.20-0.26x by building a smaller and worse tree.

## Classification -- depth, 10 trees, 2 reps

| depth | sqrt: **vs 10 cores** | our acc / theirs | all: **vs 10 cores** | our acc / theirs |
|---|---|---|---|---|
| 8 | 0.83-0.86x | 0.638-0.654 / 0.654-0.665 | **1.21-1.30x** | 0.727-0.729 / 0.730-0.731 |
| 12 | 0.80-0.84x | 0.679-0.707 / 0.683-0.736 | **1.18-1.22x** | 0.791-0.800 / 0.795-0.801 |
| 16 | 0.73-0.74x | 0.764-0.770 / 0.752-0.769 | 0.96-1.00x | 0.864-0.872 / 0.868-0.871 |
| 20 | 0.57-0.60x | 0.839-0.840 / 0.835-0.840 | 0.91-0.95x | 0.931-0.937 / 0.931 |

**We degrade with depth**, and the reason is structural: cost tracks LEVEL
COUNT, because each level is a separate launch sequence with a host round
trip, and deep levels have many small nodes. At depth 20 with 7 sampled
columns there is not enough work per launch to fill ten GPU cores.

## Classification -- epsilon 200,000 x 2,000, `max_features`=44, 10 trees, 2 reps

| depth | ours ms/tree | vs 1 core | vs 10 cores | our acc / theirs | our nodes / theirs |
|---|---|---|---|---|---|
| 10 | 377-391 | 1.84-2.12x | 0.58-0.59x | 0.723-0.724 / 0.718-0.726 | 12,790-13,800 / 12,518-12,678 |
| 14 | 495-521 | 1.82-2.01x | 0.54-0.62x | 0.839-0.847 / 0.827-0.834 | 74,338-81,436 / 68,168-71,450 |

## Regression -- covtype 581,012 rows, column 0 as target, 53 features, depth 12

Target is column 0 (Elevation) because that is where it is, not because of
anything it showed.

| `max_features` | ours ms/tree | vs 1 core | **vs 10 cores** | our MSE | their MSE |
|---|---|---|---|---|---|
| 7 (sqrt) | 165-421 | 1.8-3.3x | 0.69-0.84x | 22,051-22,722 | 21,645-22,348 |
| 27 | 155-186 | 7.7-9.8x | **2.60-2.87x** | 19,043-19,100 | 18,674-18,922 |
| 53 (all) | 232-250 | 8.5-15.3x | **3.42-3.48x** | 17,715-17,735 | 17,985-18,295 |

At `max_features=53` our MSE is LOWER than theirs. This is the widest margin
against scikit-learn in this lane.

## What decides the ratio, in one sentence

**The sampled-column count**, because it is `gridDim.y` in every kernel here:
at 5 columns the grid is starved and their ten cores win; at 27 or more it is
fed and we win. Depth works the other way, for the same reason from the other
side: more levels, each with a host round trip, and fewer rows per node.

## The scale caveat, stated beside the wins

At **100,000 rows** the same code is 0.06-0.29x against their ten cores, for
both objectives. Every winning number above is at 581,012 rows or 200,000 x
2,000. The GPU is starved below a few hundred thousand rows and this comparison
should not be quoted without that.

## Reproducing

    pixi run -e bench mojo run -I . extratrees/bench/sklearn_interleaved.mojo \
        ~/.cache/mojolearn covtype 581012 54 7 10 12 3 log2,sqrt,k14,k27,all
    # add the token `regression` for the ExtraTreesRegressor arms

`bench/results` convention: one file per recorded run, never edited in place to
change a number.
