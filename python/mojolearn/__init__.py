"""mojolearn: GPU machine-learning algorithms in Mojo, running on Apple Silicon.

The algorithms here mirror the designs of the CUDA implementations that
established them -- CatBoost, cuVS, cuML, RAFT -- and run on hardware none of
those can reach. See NOTICE for the attribution each carries.

WHAT IS IN THIS ALPHA, AND WHAT IS NOT
---------------------------------------
Eleven estimators: `NearestNeighbors`, `KMeans`, `DBSCAN`, `PCA`,
`TruncatedSVD`, `LinearRegression`, `GradientBoosting`,
`RandomForestClassifier`, `RandomForestRegressor`, `ExtraTreesClassifier`
and `ExtraTreesRegressor`. All are backed by kernels verified against
hand-computed expectations and, for k-means, the forests and the ensemble,
against cuVS, cuML, scikit-learn and CatBoost's own output. Each
class docstring carries a WHAT IS HONORED / REFUSED table: a parameter the
kernel does not carry is refused BY NAME with the reason, never accepted
and ignored, and `tools/e2u_matrix_fit.py` measures that every honored
parameter moves the answer on a fixture that should move.

`GradientBoosting` is the CatBoost oblivious-tree learner and trains twelve
of their loss functions -- RMSE, Logloss, CrossEntropy, Quantile, MAE,
LogLinQuantile, MAPE, Poisson, Lq, Expectile, Tweedie, Huber -- with the leaf
estimator CatBoost itself picks per loss. **MultiClass is not on this
surface**: it is implemented and gated in the Mojo layer, but the wrapper is
one-dimensional and `predict_proba` would need routing through the softmax
over `numClasses - 1` stored approxes. It is named here rather than left out
silently.

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
# MOJOLEARN_NUMERIC_MODE=identical loads python/mojolearn/identical/*.so
# under the canonical names (see _backend.py); default is fast.
from . import _backend as _backend

_backend.select()
numeric_mode = _backend.numeric_mode

from .cluster import KMeans
from .decomposition import PCA, TruncatedSVD
from .density import DBSCAN
from .ensemble import GradientBoosting
from .extratrees import ExtraTreesClassifier, ExtraTreesRegressor
from .linear_model import LinearRegression
from .neighbors import NearestNeighbors
from .randomforest import RandomForestClassifier, RandomForestRegressor

__all__ = [
    "DBSCAN",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "GradientBoosting",
    "KMeans",
    "LinearRegression",
    "NearestNeighbors",
    "PCA",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "TruncatedSVD",
    "__version__",
    "numeric_mode",
]

# Named absences. Importing one of these raises with a reason rather than an
# AttributeError, because "why is KNeighborsClassifier missing" is a
# question the answer to is interesting and short. Each value names the
# thing that EXISTS and where it stops.
_NOT_YET = {
    "KNeighborsClassifier": (
        "neighbors/estimator.mojo (knn_search): a vote over what it "
        "returns, not written"
    ),
    "KNeighborsRegressor": (
        "neighbors/estimator.mojo (knn_search): a mean over what it "
        "returns, not written"
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
