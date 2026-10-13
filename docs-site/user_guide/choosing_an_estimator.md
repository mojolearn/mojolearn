# Choosing an estimator

Start from the task. Each row names the estimators to try first and links the guide page for that family.
Every estimator follows the scikit-learn conventions: construct it with hyperparameters, then call `fit`,
`predict`, `predict_proba`, `transform` or `score`. Inputs are float32 arrays.

| Task | Try first | Also available | Guide |
|---|---|---|---|
| Classify rows of tabular data | `RandomForestClassifier`, `GradientBoostingClassifier` (two classes) | `ExtraTreesClassifier`, `LogisticRegression`, `SVC`, `KNeighborsClassifier` | [Trees and gradient boosting](trees.md) |
| Classify into more than two classes with boosting | `GradientBoosting` with `loss="MultiClass"` | `RandomForestClassifier` | [Trees and gradient boosting](trees.md) |
| Predict a number | `GradientBoostingRegressor`, `RandomForestRegressor` | `Ridge`, `LinearRegression`, `SVR`, `KernelRidge` | [Trees and gradient boosting](trees.md), [Linear models](linear.md) |
| Rank items within groups | `GradientBoosting` with `loss="YetiRank"` or `loss="PairLogit"` | `GradientBoosting` with `loss="QueryRMSE"` | [Trees and gradient boosting](trees.md) |
| Fit an interpretable linear model | `Ridge`, `LogisticRegression` | `Lasso`, `ElasticNet`, `RidgeCV`, `LassoCV` | [Linear models](linear.md) |
| Predict with an uncertainty estimate | `GaussianProcessRegressor` | `GaussianProcessClassifier`, `BayesianRidge` | [Gaussian processes](gaussian_process.md) |
| Group similar rows | `KMeans` | `MiniBatchKMeans`, `GaussianMixture`, `DBSCAN`, `HDBSCAN` | [Clustering](clustering.md) |
| Reduce the number of features | `PCA` | `TruncatedSVD`, `IncrementalPCA`, `NMF`, `FastICA` | [Decomposition and embeddings](decomposition.md) |
| Draw a 2-D map of the data | `UMAP` | `TSNE`, `Isomap`, `SpectralEmbedding` | [Decomposition and embeddings](decomposition.md) |
| Find the nearest rows exactly | `NearestNeighbors` | `KNeighborsClassifier`, `KNeighborsRegressor` | [Neighbors and vector search](neighbors.md) |
| Search millions of vectors quickly | `IVFIndex`, `CagraIndex` | `IVFPQIndex`, `IVFSQIndex` | [Neighbors and vector search](neighbors.md) |
| Flag unusual rows | `IsolationForest` | `LocalOutlierFactor`, `OneClassSVM`, `EllipticEnvelope` | [Trees and gradient boosting](trees.md) |
| Forecast a series | `ARIMA`, `ETS` | `AutoARIMA`, `Theta`, `ExponentialSmoothing` | [Time series](time_series.md) |
| Split a series into trend and season | `STL` | `ExponentialSmoothing` | [Time series](time_series.md) |
| Scale, encode or impute features | `StandardScaler`, `OneHotEncoder`, `SimpleImputer` | `MinMaxScaler`, `RobustScaler`, `KBinsDiscretizer` | [Preprocessing](preprocessing.md) |
| Keep the most useful features | `SelectKBest` | `VarianceThreshold`, `RFE` | [Preprocessing](preprocessing.md) |
| Score a model with cross-validation | `cross_val_score()` | `mojolearn.metrics` | [API reference](../api/evaluation.rst) |
| Build or run neural network layers | `TransformerBlock`, `SmallMLPTrainer` | `SambaStack`, `Embedding` | [Neural networks](neural_networks.md) |
| Train or run a language model | `LanguageModelTrainer` | `BpeTokenizer` | [Language models](language_models.md) |

## A first pass on tabular data

When you are unsure, fit one tree ensemble and one linear model and compare their scores on held-out
rows:

```python
import numpy as np
import mojolearn

rng = np.random.default_rng(0)
X = rng.random((10_000, 20), dtype=np.float32)
y = (X[:, 0] + X[:, 1] > 1.0).astype(np.int64)
X_train, X_test, y_train, y_test = X[:8_000], X[8_000:], y[:8_000], y[8_000:]

forest = mojolearn.RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
linear = mojolearn.LogisticRegression().fit(X_train, y_train)
print(forest.score(X_test, y_test), linear.score(X_test, y_test))
```

## Devices and numeric modes

In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs. `identical` is the
default, so every example in these guides runs in it without extra settings. On Apple GPUs you can
choose `fast` instead, which is tuned for speed and does not promise identical bits:

```python
mojolearn.set_numeric_mode("fast")
```

See [Numeric modes](numeric_modes.md) and [Devices and the CPU](devices.md).
