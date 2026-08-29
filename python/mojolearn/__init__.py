"""mojolearn: GPU machine-learning algorithms in Mojo, running on Apple Silicon.

The algorithms here mirror the designs of the CUDA implementations that
established them -- CatBoost, cuVS, cuML, RAFT -- and run on hardware none of
those can reach. See NOTICE for the attribution each carries.

WHAT IS IN THIS ALPHA, AND WHAT IS NOT
---------------------------------------
Twenty-three estimators, two submodules and two functions: `NearestNeighbors`, `KNeighborsClassifier`,
`KNeighborsRegressor`, `KMeans`, `DBSCAN`, `PCA`, `TruncatedSVD`,
`LinearRegression`, `Ridge`, `LogisticRegression` (binary, L-BFGS),
`GradientBoosting`, `RandomForestClassifier`, `RandomForestRegressor`,
`ExtraTreesClassifier` and `ExtraTreesRegressor`.
All are backed by kernels verified against hand-computed expectations and,
for k-means, k-NN classification and regression, the forests and the
ensemble, against cuVS, cuML, scikit-learn and CatBoost's own output. Each
class docstring carries a WHAT IS HONORED / REFUSED table: a parameter the
kernel does not carry is refused BY NAME with the reason, never accepted
and ignored, and `tools/e2u_matrix_fit.py` measures that every honored
parameter moves the answer on a fixture that should move.

`GradientBoosting` is the CatBoost GPU tree learner -- all three of its
growth policies (`grow_policy`: the oblivious SymmetricTree default,
Depthwise and Lossguide, the last two building non-symmetric trees) -- and
trains thirteen of their loss functions -- RMSE, Logloss, CrossEntropy,
Quantile, MAE, LogLinQuantile, MAPE, Poisson, Lq, Expectile, Tweedie, Huber,
MultiClass -- with the leaf estimator CatBoost itself picks per loss.
`MultiClassOneVsAll` is NOT on this surface and is named rather than left out
silently (see `ensemble.py`'s `_UNREACHABLE_LOSSES` for why). The
non-symmetric policies accept exactly the losses CatBoost's GPU registers a
non-symmetric trainer for, and refuse the rest by name.

**`DBSCAN`, `PCA`, `TruncatedSVD` and `LinearRegression` ARE here** (since
2026-08-23). This docstring used to say they were not, and that sentence
outlived the fact by a day: `density.py`, `decomposition.py` and
`linear_model.py` had been bound through `_mojolearn_estimators` with the
policy each needs stated on the class, and were reachable only as
submodules while `__getattr__` still told a caller they had no surface.
What is STILL absent is named in `_NOT_YET` below, with the line where
the thing that exists stops.

WHAT THIS IS NOT
----------------
Not a drop-in scikit-learn replacement. The estimators take scikit-learn's
shapes and return scikit-learn's layouts, but **the defaults follow the
upstream each algorithm mirrors, not scikit-learn's**, and where those differ
it is documented on the class. `KMeans.n_init` is the one that will surprise
you: cuVS's default is 1 and scikit-learn's is 10.

Not validated outside Apple Silicon. One source targets Metal, CUDA and ROCm,
and this has been measured on exactly one M4. Support is not validation.

**There is no CPU path, and this docstring used to say there was.** The
sentence read "One source targets CPU, Metal, CUDA and ROCm", which was true
of the predecessor library and was carried across the rename. It is false
here: every estimator in this package requires a GPU, `kernel_matrix.mojo`
states "There is no CPU column. This tree has no CPU path," and no CPU kernel
exists to fall back to. A user reading a promise of CPU support in the
package's own docstring would discover otherwise at the first call, which is
the worst place in the tree for that sentence to have been.
"""

from ._version import __version__

# THE NUMERIC MODE IS CHOSEN HERE, BEFORE ANY BINDING IS IMPORTED.
# MOJOLEARN_NUMERIC_MODE picks one of THREE tiers, each keeping the one
# below it: `fast` (the default, no promise), `deterministic` (same bits
# run to run on ONE device) and `identical` (also the same bits across
# Metal, CUDA and HIP). Each upper tier loads its own binary set from
# python/mojolearn/<tier>/*.so under the canonical names; see
# _backend.py, whose allow-list refused `deterministic` outright until
# 2026-08-29 and so made a tier that existed in the compiler
# unreachable from Python.
from . import _backend as _backend

_backend.select()
numeric_mode = _backend.numeric_mode

#: CHOOSE THE TIER IN CODE. `MOJOLEARN_NUMERIC_MODE` still works and still
#: sets the starting value, but it is no longer the only way in: the mode is
#: a runtime choice, and estimators take a per-instance `numeric_mode=`.
#:
#:     mojolearn.set_numeric_mode("deterministic")     # process default
#:     mojolearn.RandomForestClassifier(numeric_mode="identical")
#:
#: All three tiers ship in ONE wheel and can be loaded into ONE process at
#: once -- measured on 2026-08-29 by calling all three interleaved and
#: checking each returned its own arithmetic. See `_backend.load_set`.
set_numeric_mode = _backend.set_default_mode

#: WHICH GPU API THE LOADED BINARIES WERE COMPILED FOR: 'metal', 'cuda' or
#: 'hip', read back out of the binaries (`mojo_only/vendor.mojo`). On Linux
#: one wheel carries a CUDA set and a HIP set and `_backend._layout()` picks
#: one at import; `vendor()` is what it picked, cross-checked against what
#: the binaries answer. There is no CPU path: a Linux box with neither
#: device refuses at import, naming what it looked for.
vendor = _backend.vendor

from .cluster import KMeans
from .decomposition import PCA, TruncatedSVD
from .density import DBSCAN, KernelDensity
from .ensemble import GradientBoosting
from .extratrees import ExtraTreesClassifier, ExtraTreesRegressor
from .linear_model import LinearRegression, LogisticRegression, Ridge
from .neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors,
)
from .randomforest import RandomForestClassifier, RandomForestRegressor

# ---------------------------------------------------------------------------
# THE 2026-08-24 SURFACE. Seven estimators, two submodules and two functions
# over lanes that were finished and CERTIFIED at the kernel level and simply
# unreachable from Python. Each carries its own WHAT IS HONORED / REFUSED
# table and its own cross-vendor status, and those statuses DIFFER: `SVC`,
# `Lasso`, `ElasticNet`, `AgglomerativeClustering` and `mojolearn.metrics`
# stand on lanes with three-vendor identity cards at leg 11; `IsolationForest`,
# `SpectralClustering` and `ExponentialSmoothing` do NOT and say so on the
# class. Read the class, not this list.
#
# `ARIMA` is deliberately absent and is named in `_NOT_YET` below. Its lane
# has no `fit`: the optimizer that would produce the coefficients is unported,
# so the class would have to demand its own answer as an argument.
#
# Imported eagerly, like the block above, which is safe because every impl
# module resolves its binding on FIRST USE rather than at import. A partial
# build therefore still yields an importable package whose missing pieces
# raise BY NAME when touched, which is `_backend.py`'s whole design.
from . import _linalg_impl as linalg
from . import _metrics_impl as metrics
from ._hierarchy_impl import AgglomerativeClustering
from ._iforest_impl import IsolationForest
from ._solver_impl import ElasticNet, Lasso
from ._spectral_impl import SpectralClustering
from ._svm_impl import SVC
from ._tsa_impl import ExponentialSmoothing, kpss_test, select_d

__all__ = [
    "AgglomerativeClustering",
    "DBSCAN",
    "KernelDensity",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "ElasticNet",
    "GradientBoosting",
    "ExponentialSmoothing",
    "IsolationForest",
    "KMeans",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "Lasso",
    "LinearRegression",
    "LogisticRegression",
    "SVC",
    "SpectralClustering",
    "NearestNeighbors",
    "PCA",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "Ridge",
    "TruncatedSVD",
    "kpss_test",
    "linalg",
    "metrics",
    "select_d",
    "__version__",
    "numeric_mode",
    "set_numeric_mode",
    "vendor",
]

# Named absences. Importing one of these raises with a reason rather than an
# AttributeError, because "why is RadiusNeighbors missing" is a question
# the answer to is interesting and short. Each value names the thing that
# EXISTS and where it stops. (`KNeighborsClassifier` / `KNeighborsRegressor`
# were here until 2026-08-23; they are exported above now.)
_NOT_YET = {
    "ARIMA": (
        "arima/ (the batched Kalman filter likelihood, its gradient and "
        "predict all exist and are gated on one Apple M4); NO fit. "
        "`estimate_x0`, the batched L-BFGS driver and the CSS likelihood are "
        "NOT PORTED (arima/UNPORTED.tsv), and those are exactly what produces "
        "the coefficients every existing entry point REQUIRES as input, so an "
        "`ARIMA` class would have to demand its own answer as an argument"
    ),
    "SVR": (
        "svm/ (C-SVC only: svmType != C_SVC raises); regression is rung 2 in "
        "svm/UNPORTED.tsv"
    ),
    "RadiusNeighbors": (
        "neighbors/ported/neighbors/ball_cover/ (radius search exists for "
        "DBSCAN's eps neighbourhood); no caller-facing surface"
    ),
}


def __getattr__(name):
    if name in _NOT_YET:
        raise AttributeError(
            f"mojolearn.{name} is not in this release. What exists stops at "
            f"{_NOT_YET[name]}; binding it directly would put policy where "
            "no check can see it. See the module docstring."
        )
    raise AttributeError(f"module 'mojolearn' has no attribute {name!r}")
