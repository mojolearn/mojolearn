# Linear models

Linear models are fast to fit, easy to read through their coefficients, and a good baseline for any
regression or classification task. They work best on scaled features; see [Preprocessing](preprocessing.md).

| Use | Estimators |
|---|---|
| Least squares | `LinearRegression` |
| Regularized regression | `Ridge`, `Lasso`, `ElasticNet`, `Lars`, `LassoLars` |
| Regularization chosen by cross-validation | `RidgeCV`, `LassoCV`, `ElasticNetCV` |
| Classification | `LogisticRegression`, `LogisticRegressionCV`, `RidgeClassifier`, `SGDClassifier`, `Perceptron` |
| Robust and quantile regression | `HuberRegressor`, `QuantileRegressor` |
| Counts, positive targets and other distributions | `PoissonRegressor`, `GammaRegressor`, `TweedieRegressor` |
| Bayesian regression | `BayesianRidge`, `ARDRegression` |
| Monotone fits | `IsotonicRegression` |

## Regression

```python
import numpy as np
import mojolearn

rng = np.random.default_rng(0)
X = rng.random((10_000, 8), dtype=np.float32)
y = X @ np.arange(8, dtype=np.float32) + 1.0

ols = mojolearn.LinearRegression().fit(X, y)
print(ols.coef_, ols.intercept_)

ridge = mojolearn.Ridge(alpha=0.5).fit(X, y)
lasso = mojolearn.Lasso(alpha=0.01).fit(X, y)
print(ridge.score(X, y), lasso.score(X, y))
```

To let the data pick the penalty, use a cross-validated estimator:

```python
model = mojolearn.RidgeCV(alphas=(0.01, 0.1, 1.0)).fit(X, y)
print(model.alpha_)
```

## Classification

`LogisticRegression` uses the logistic loss for two classes and the multinomial loss for more. `C` is the
inverse regularization strength, as in scikit-learn.

```python
labels = (X[:, 0] + X[:, 1] > 1.0).astype(np.int64)
clf = mojolearn.LogisticRegression(C=10.0).fit(X, labels)
proba = clf.predict_proba(X[:5])
print(clf.classes_, clf.coef_)
```

## Saving a model

`LinearRegression`, `Ridge`, `Lasso`, `ElasticNet` and `LogisticRegression` save to a file and load back:

```python
ridge.save("ridge.npz")
loaded = mojolearn.Ridge.load("ridge.npz")
```

## Devices and numeric modes

In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs. `identical` is the
default. On Apple GPUs you can choose `fast` instead, which is tuned for speed and does not promise
identical bits: call `mojolearn.set_numeric_mode()` with `"fast"`. See [Numeric modes](numeric_modes.md).

## See also

- [Choosing an estimator](choosing_an_estimator.md)
- Kernel methods for non-linear fits: `SVC`, `SVR`, `KernelRidge`
- API reference: [Linear models](../api/linear.rst),
  [Support vector machines and kernel methods](../api/svm_kernels.rst)
