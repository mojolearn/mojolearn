# Saving and loading models

A fitted mojolearn model can be written to a file, loaded back on any supported device, and used to
predict on a machine with no GPU. Under `identical`, the loaded model predicts the same bits as the
model that was saved.

## Save and load an estimator

Estimators that support saving have a `save(path)` method and a `load(path)` class method. Files are
NumPy `.npz` archives.

```python
import numpy as np
import mojolearn

rng = np.random.default_rng(0)
X = rng.random((100_000, 20), dtype=np.float32)
y = (X[:, 0] + X[:, 1] > 1.0).astype(np.float32)

model = mojolearn.GradientBoosting(loss="Logloss", n_estimators=200, max_depth=6).fit(X, y)
model.save("gbdt.npz")

loaded = mojolearn.GradientBoosting.load("gbdt.npz")
print(loaded.predict_proba(X[:5]))
```

A loaded model predicts or transforms; it does not refit. Loading a file written by a different
estimator class raises instead of reinterpreting it.

Estimators with `save` and `load` in 0.8.41:

| Family | Estimators |
|---|---|
| Trees and boosting | `GradientBoosting`, `GradientBoostingClassifier`, `GradientBoostingRegressor`, `OrderedRMSE`, `RandomForestClassifier`, `RandomForestRegressor`, `ExtraTreesClassifier`, `ExtraTreesRegressor`, `DecisionTreeClassifier`, `DecisionTreeRegressor`, `IsolationForest` |
| Linear and kernel | `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `LogisticRegression`, `KernelRidge`, `Nystroem`, `RBFSampler`, `SVC`, `SVR` |
| Neighbors and search | `NearestNeighbors`, `KNeighborsClassifier`, `KNeighborsRegressor`, `RadiusNeighbors`, `IVFIndex`, `IVFPQIndex`, `IVFSQIndex`, `IVFRaBitQIndex`, `CagraIndex` |
| Clustering and density | `KMeans`, `DBSCAN`, `HDBSCAN`, `AgglomerativeClustering`, `SpectralClustering`, `GaussianMixture`, `KernelDensity` |
| Decomposition and preprocessing | `PCA`, `TruncatedSVD`, `UMAP`, `StandardScaler`, `MinMaxScaler` |
| Gaussian processes | `GaussianProcessRegressor`, `GaussianProcessClassifier` |
| Time series | `ARIMA`, `ExponentialSmoothing` |
| Other | `Embedding`, `Cholesky` |

## What a saved file contains

- The arrays prediction reads, at their exact dtypes. Floats are never written as decimal text.
  Gradient boosting stores its model text with every float's hex bit pattern beside the decimal, and
  the loader reads the bits.
- For gradient boosting and `parallel_groves` forest files, the numeric mode the model was trained
  under, so a later process default does not change how it predicts.
- Only what prediction needs. For example, a loaded `GradientBoosting` does not carry the training
  configuration or loss curves (`loss_curve_` is `None`).

For forests and k-means, the file's bytes are a pure function of the model, so the same model saved
on two machines gives the same file hash.

## Predict on the CPU

`host_model` opens a saved file on the CPU, choosing the right host model from the file's format tag:

```python
import mojolearn

host = mojolearn.host_model("gbdt.npz")        # a HostGBDT for a gradient boosting file
print(host.predict_proba(X[:5]))

# one-call forms
labels = mojolearn.host_predict("forest.npz", X)
proba = mojolearn.host_predict_proba("forest.npz", X)
```

| Function or class | Use |
|---|---|
| `mojolearn.host_model(path)` | Open any supported saved file on the CPU |
| `mojolearn.host_predict(path, X)` | `host_model(path).predict(X)` |
| `mojolearn.host_predict_proba(path, X)` | `host_model(path).predict_proba(X)` |
| `mojolearn.HostForest.from_file(path)` | A saved random forest or Extra Trees model |
| `mojolearn.HostGBDT.from_file(path)` | A saved `GradientBoosting` or `OrderedRMSE` model |

`HostForest` and `HostGBDT` have `predict`, `predict_proba`, `n_trees`, `inference_engine` and
`model_sha256`, a SHA-256 over the model's arrays that you can record to name the model a prediction
came from. `host_model` also opens linear, ridge, logistic, PCA, truncated SVD, neighbors, scaler,
lasso, elastic net, kernel ridge, Nystroem and random Fourier feature files. A format it does not serve
is refused with the format tag the file carries.

The CPU bindings compute `identical` bits only. They return the same dtypes the GPU estimators return
for the same file. On a CPU-only install, `Estimator.load(path)` followed by `predict` also runs on the
CPU for the families listed under "Prediction from a saved model" in
[Devices and the CPU](devices.md#the-cpu-surface).

## Forest inference engines

Random forests and Extra Trees have two prediction algorithms, chosen with `inference_engine`:

| `inference_engine` | Behavior |
|---|---|
| `auto` (default) | `parallel_groves` in `fast` (Apple GPUs only); `sequential` in `identical` |
| `sequential` | Each row adds trees in increasing tree order, then divides by the tree count |
| `parallel_groves` | 32 fixed tree groups in parallel, combined by a fixed 16/8/4/2/1 reduction |

Under `identical`, each engine gives the same bits across vendors, but the two add in different orders, so they do not give each
other's bits. Probabilities can differ slightly, and near ties a class label can differ. You can switch
the engine on a fitted forest:

```python
clf = mojolearn.RandomForestClassifier(inference_engine="parallel_groves").fit(X, y)
p_groves = clf.predict_proba(X[:5])
clf.inference_engine = "sequential"
p_seq = clf.predict_proba(X[:5])
```

A saved forest keeps its engine. `HostForest` serves both on the CPU; a `parallel_groves` archive saved
under a mode other than `identical` is refused by the CPU binding.

## Language-model checkpoints

The byte-level language model saves and resumes through its own checkpoint methods:
`LanguageModelTrainer.save_checkpoint(path)` and `LanguageModelTrainer.from_checkpoint(path)` for
training, and `LanguageModelInference.from_checkpoint(path)` for CPU inference. See
[Language models](language_models.md).

## See also

- [Devices and the CPU](devices.md): when the CPU path is used.
- [Verify on your hardware](verify.md): the verifier also checks bundled saved models.
