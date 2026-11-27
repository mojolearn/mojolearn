# Numeric modes

Every mojolearn computation runs under one of two numeric modes. The mode is a promise about the
output bits, not a different algorithm: the estimator, its hyperparameters and its API are the same in
every mode.

## What each mode promises

| Mode | Promise |
|---|---|
| `identical` | The default. In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs. |
| `fast` | Opt-in mode for throughput. Results can vary across runs and devices. |

A release ships only when its recorded Apple, NVIDIA and AMD results agree.

Three things to know about `identical`:

- It holds within one release. Bits may change between releases, for example when a kernel moves to a
  faster or more accurate fold order. Compare runs made with the same release.
- It covers the supported configurations: the families, settings and devices listed in the
  [support matrix](../notes/support_matrix.md).
- It is an FP32 guarantee about mojolearn's own arithmetic. Weights trained in another framework
  reproduce under mojolearn's arithmetic, not that framework's.

## Which kernels run

On Apple GPUs, selecting `fast` requires fast kernels for the operation. On Linux
with NVIDIA, AMD or a CPU-only install, selecting `fast` uses the identical kernels
if the installation has no separate fast kernels. In that case, changing the
setting runs the same arithmetic.

## Selecting a mode

There are three ways to choose a mode. A narrower choice overrides a wider one.

| Scope | How |
|---|---|
| Process, before import | Set `MOJOLEARN_NUMERIC_MODE` to `identical` or `fast` in the environment |
| Process, at runtime | `mojolearn.set_numeric_mode("fast")`, which returns the previous mode |
| One estimator | `numeric_mode=` on estimators whose constructor has it |

```python
import mojolearn

print(mojolearn.numeric_mode())                  # "identical" unless you changed it
previous = mojolearn.set_numeric_mode("fast")
# ... work under fast ...
mojolearn.set_numeric_mode(previous)             # restore
```

`set_numeric_mode` loads the requested mode's binaries immediately, so a mode that cannot be honored
fails on that line, not at a later `fit`.

Estimators that take `numeric_mode=` include `RandomForestClassifier`, `RandomForestRegressor`,
`ExtraTreesClassifier`, `ExtraTreesRegressor`, `DecisionTreeClassifier`, `GradientBoostingClassifier`,
`GradientBoostingRegressor`, `StandardScaler`, `MinMaxScaler`, `MLPClassifier` and the sequence and
time-series estimators except `ExponentialSmoothing` and `AutoARIMA`, which always use the process mode. Check an estimator's signature in the [API reference](../api/index.rst).
The default `numeric_mode=None` means "use the process mode".

```python
import numpy as np
from mojolearn import RandomForestClassifier

rng = np.random.default_rng(0)
X = rng.random((10_000, 20), dtype=np.float32)
y = (X[:, 0] > 0.5).astype(np.float32)

clf = RandomForestClassifier(n_estimators=100, numeric_mode="identical").fit(X, y)
print(clf.numeric_mode_used())
```

`numeric_mode_used()` reports the mode the estimator actually ran on, read back from the binary it
loaded rather than from the string you passed. Record it alongside results.

## Refusals

Asking a family for a mode it does not ship raises a `ValueError` that names the binding and the
missing mode. mojolearn never runs a weaker mode in its place.

An unknown mode name raises `ValueError("numeric_mode must be fast, identical or None")`.
The verifier checks which kernels were loaded and requires identical arithmetic.
See [Verify on your hardware](verify.md).

## Numeric profiles

A numeric profile is a separate setting from the numeric mode. The mode is the promise; the profile is
the number format the matrix products compute in. Transformer language-model inference (`CausalLM`,
`ParallelCausalLM`) uses `fixed15_v1` by default; every other family, Mamba models and all training use
`fp32_v1`. Other profiles are opt-in and apply only to the families they list. See
[Language models](language_models.md#hugging-face-checkpoints) for what `fixed15_v1` computes and how to choose `fp32_v1`
instead.

| Function | Purpose |
|---|---|
| `mojolearn.numeric_profile(use="inference")` | The profile a call that names none runs under |
| `mojolearn.set_numeric_profile(name, use="inference")` | Set the process default; returns the previous value |
| `mojolearn.numeric_profiles()` | Every registered profile, as dicts |
| `mojolearn.numeric_profile_measured(profile, vendor=None)` | Measured rows for a profile |

A profile under `identical` keeps the cross-vendor promise, just as `fp32_v1` does. Choosing a profile
never changes the mode. See [Runtime and numeric modes](../api/runtime.rst).

## What to use when

- **Report and publish in `identical`.** It is the default and the only mode whose results another
  person can reproduce on other supported hardware with the same release.
- **On Apple GPUs, explore in `fast`**, for example during hyperparameter search, then rerun the
  configuration you keep in `identical`.
- Record `mojolearn.__version__`, `numeric_mode_used()`, `mojolearn.vendor()` and
  `mojolearn.gpu_arch()` with any result you intend to compare.
