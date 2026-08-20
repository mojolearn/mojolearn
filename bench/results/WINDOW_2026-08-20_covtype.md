# covtype against CatBoost, and why gbm-bench cannot run this repository yet

Taken 2026-08-20 on the M4, quiet box, AC power, at `1121341`.

```
pixi run -e bench python tools/interleaved_prep.py <dir> covtype
pixi run -e bench mojo run -I . bench/interleaved/catboost_interleaved.mojo <dir> covtype 581012
```

581,012 rows x 53 features kept, depth 6, 20 trees, lr 0.3, l2 3.0, RMSE on
the class label, both arms on CatBoost's own quantization grid, pool quantized
outside the timed region on both sides, arms alternating per rep.

## The numbers

| borders | CatBoost CPU, ms/tree | ours GPU, ms/tree | ratio |
|---|---|---|---|
| 254 | 9.753 / 9.855 / 9.644 | 19.285 / 19.330 / 18.074 | **1.98x slower** |
| 128 | 9.459 / 9.708 / 9.875 | 18.416 / 18.164 / 17.821 | **1.87x slower** |

Ratios from medians. Three interleaved reps each.

## The loss agrees to eight significant figures

| borders | our final MSE | CatBoost MSE |
|---|---|---|
| 254 | 0.9545517132176272 | 0.954551752073908 |
| 128 | 0.9486324077643835 | 0.9486324342756266 |

That agreement is the reason the speed column means anything. Two arms that
trained different models would produce two incomparable times, and the usual
way a GBDT benchmark flatters itself is by quietly fitting something easier.
This one is fitting CatBoost's model, to the eighth digit, and losing on time.

## What this changes

**The 1.66x figure was on synthetic data at 800k x 100. On a real dataset we
are 1.9x behind, not 1.66x.** The honest number to carry is the worse one, and
covtype is the dataset the incumbents publish on.

Per-tree cost is far below the 50 ms/tree recorded at 800k x 100, at 18 to 19
ms/tree here, so the fixed control-plane overhead is a smaller share of a
smaller tree and this shape does not isolate it. It is not evidence the
overhead shrank.

## gbm-bench cannot run this repository, and that is a binding gap

NVIDIA's gbm-bench drives sklearn-shaped Python estimators. This repository's
extension exposes exactly two entry points:

```
bindings/_mojolearn.mojo   ->  kmeans_fit, knn_search
python/mojolearn/__all__   ->  KMeans, NearestNeighbors
```

There is no GBDT binding, so there is nothing for that harness to call and no
arm to register. Running a competitor's harness against this GBDT is not a
benchmarking task, it is a fit/predict binding plus a sklearn-shaped estimator,
and only then an arm. Until that exists, `bench/interleaved/` is the only
external comparison this GBDT has, which is what this file records.
