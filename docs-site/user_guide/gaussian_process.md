# Gaussian processes

A Gaussian process predicts a value and how sure it is about that value. It suits small and medium data
sets where an uncertainty estimate matters, such as calibration curves, experiment design and smooth
signals. The kernel says which inputs count as similar.

| Use | Estimators |
|---|---|
| Regression with a standard deviation per prediction | `GaussianProcessRegressor` |
| Classification with probabilities | `GaussianProcessClassifier` |
| Kernels | `RBF`, `Matern`, `ConstantKernel`, `WhiteKernel` |

## Regression

Kernels combine with `+` and `*`. A `ConstantKernel` times an `RBF` models the signal, and a
`WhiteKernel` models noise. With `optimizer="fmin_l_bfgs_b"`, `fit` tunes the kernel parameters.

```python
import numpy as np
import mojolearn

rng = np.random.default_rng(0)
X = rng.random((200, 3), dtype=np.float32)
y = np.sin(X.sum(axis=1)).astype(np.float32)

kernel = (mojolearn.ConstantKernel(1.0) * mojolearn.RBF([1.0, 1.0, 1.0])
          + mojolearn.WhiteKernel(0.01))
gp = mojolearn.GaussianProcessRegressor(kernel, optimizer="fmin_l_bfgs_b").fit(X, y)
mean, std = gp.predict(X[:5], return_std=True)
print(gp.kernel_)
```

`kernel_` holds the fitted kernel, and `log_marginal_likelihood()` scores a set of kernel parameters.

## Classification

```python
labels = (X[:, 0] + X[:, 1] > 1.0).astype(np.int32)
gpc = mojolearn.GaussianProcessClassifier(mojolearn.RBF(0.5)).fit(X, labels)
proba = gpc.predict_proba(X[:5])
```

## Saving a model

```python
gp.save("gp.npz")
loaded = mojolearn.GaussianProcessRegressor.load("gp.npz")
```

## Devices and numeric modes

In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs. `identical` is the
default. On Apple GPUs you can choose `fast` instead, which is tuned for speed and does not promise
identical bits: call `mojolearn.set_numeric_mode()` with `"fast"`. See [Numeric modes](numeric_modes.md).

## See also

- [Choosing an estimator](choosing_an_estimator.md)
- `BayesianRidge`, a linear model with uncertainty: [Linear models](linear.md)
- API reference: [Gaussian processes](../api/gaussian_process.rst)
