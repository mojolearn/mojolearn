# Quickstart

## Train and predict

```python
import numpy as np
import mojolearn

rng = np.random.default_rng(0)
X = rng.random((100_000, 20), dtype=np.float32)
y = (X[:, 0] + X[:, 1] > 1.0).astype(np.float32)

model = mojolearn.GradientBoosting(loss="Logloss", n_estimators=200, max_depth=6)
model.fit(X, y)
print(model.predict_proba(X[:5]))
print(model.numeric_mode_used(), mojolearn.vendor())
```

`vendor()` names the backend that ran, and `numeric_mode_used()` the numeric mode. Run the same script with
the same release on a supported Apple, NVIDIA or AMD GPU and the model and its predictions are the same bits.

## The estimator interface

Estimators follow scikit-learn's conventions: construct with hyperparameters, then `fit`, `predict`,
`predict_proba`, `transform` or `score` as the estimator supports. Inputs are float32 arrays; NumPy arrays
work directly, and the library itself does not require NumPy.

```python
from mojolearn import KMeans, PCA, RandomForestClassifier

labels = KMeans(n_clusters=8).fit(X).labels_
Z = PCA(n_components=5).fit_transform(X)
clf = RandomForestClassifier(n_estimators=100).fit(X, y)
```

## Choose a numeric mode

`identical` is the default. Select another mode for the process or per call:

```python
previous = mojolearn.set_numeric_mode("deterministic")   # returns the previous mode
mojolearn.set_numeric_mode("identical")                  # back to the default
```

On Apple GPUs, `fast` is also available; it runs on Apple GPUs only.

or set `MOJOLEARN_NUMERIC_MODE` in the environment. Estimators that take a `numeric_mode=` argument, such as
`RandomForestClassifier`, override the process mode per call. Asking a family for a mode it does not ship raises
a named error rather than running something weaker. See [Numeric modes](numeric_modes.md).

## Next

- [Verify on your hardware](verify.md): check the installed release against its recorded bits.
- [Saving and loading models](saving_models.md): predict from a saved model on any supported device or the CPU.
- [API reference](../api/index.rst): every public name, by family.
