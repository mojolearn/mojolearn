# Trees and gradient boosting

Tree ensembles are the usual first choice for tabular data: they need no feature scaling, handle
interactions between features, and work for classification, regression, ranking and anomaly detection.

| Use | Estimators |
|---|---|
| Random forests and Extra Trees | `RandomForestClassifier`, `RandomForestRegressor`, `ExtraTreesClassifier`, `ExtraTreesRegressor` |
| Gradient boosting, scikit-learn interface | `GradientBoostingClassifier` (two classes), `GradientBoostingRegressor` |
| Gradient boosting, every loss | `GradientBoosting`: `MultiClass`, `Logloss`, `RMSE`, `Quantile`, `Poisson`, `Tweedie`, `YetiRank`, `PairLogit` and more |
| Single trees | `DecisionTreeClassifier`, `DecisionTreeRegressor` |
| Other ensembles | `AdaBoostClassifier`, `BaggingClassifier`, `DARTClassifier`, `VotingClassifier`, `StackingClassifier` |
| Anomaly detection | `IsolationForest` |

## A random forest

```python
import numpy as np
import mojolearn

rng = np.random.default_rng(0)
X = rng.random((10_000, 8), dtype=np.float32)
y = (X[:, 0] > 0.5).astype(np.int64)

forest = mojolearn.RandomForestClassifier(n_estimators=100).fit(X, y)
proba = forest.predict_proba(X[:5])
```

Random forests follow cuML's defaults: split thresholds are searched over at most `n_bins` quantiles per
feature. `random_state=None` means seed 0, so a fit is repeatable without setting a seed.

## Gradient boosting

`GradientBoostingClassifier` and `GradientBoostingRegressor` keep the scikit-learn interface.
`GradientBoosting` exposes every loss; its defaults follow CatBoost's GPU learner. `predict` returns raw
scores, and `predict_proba` returns probabilities for classification losses.

```python
reg = mojolearn.GradientBoostingRegressor(n_estimators=200).fit(X, X[:, 0] * 2 + X[:, 1])
print(reg.score(X, X[:, 0] * 2 + X[:, 1]))

labels = rng.integers(0, 3, 10_000).astype(np.float32)   # class codes 0, 1, 2
multi = mojolearn.GradientBoosting(loss="MultiClass", n_estimators=200).fit(X, labels)
proba = multi.predict_proba(X[:5])
```

## Anomaly detection

```python
detector = mojolearn.IsolationForest(n_estimators=100).fit(X)
flags = detector.predict(X)            # -1 for anomalies, 1 for inliers
scores = detector.score_samples(X)
```

## Saving a model

Forests, `GradientBoosting` and `IsolationForest` save to a file and load back on any supported device:

```python
forest.save("forest.npz")
loaded = mojolearn.RandomForestClassifier.load("forest.npz")
```

See [Saving and loading models](saving_models.md).

## Devices and numeric modes

In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs. `identical` is the
default. On Apple GPUs you can choose `fast` instead, which is tuned for speed and does not promise
identical bits: call `mojolearn.set_numeric_mode()` with `"fast"`, or pass `numeric_mode="fast"` to an
estimator that takes it, such as `RandomForestClassifier`. See [Numeric modes](numeric_modes.md).

## See also

- [Choosing an estimator](choosing_an_estimator.md)
- API reference: [Forests and trees](../api/trees.rst), [Gradient boosting](../api/boosting.rst)
