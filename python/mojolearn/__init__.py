"""mojolearn: GPU machine-learning algorithms in Mojo, running on Apple Silicon.

The algorithms here mirror the designs of the CUDA implementations that
established them -- CatBoost, cuVS, cuML, RAFT -- and run on hardware none of
those can reach. See NOTICE for the attribution each carries.

WHAT IS IN THIS ALPHA, AND WHAT IS NOT
---------------------------------------
Three estimators: `NearestNeighbors`, `KMeans` and `GradientBoosting`. All
are backed by kernels verified against hand-computed expectations and, for
k-means and the ensemble, against cuVS and CatBoost's own output.

`GradientBoosting` is the CatBoost oblivious-tree learner and trains twelve
of their loss functions -- RMSE, Logloss, CrossEntropy, Quantile, MAE,
LogLinQuantile, MAPE, Poisson, Lq, Expectile, Tweedie, Huber -- with the leaf
estimator CatBoost itself picks per loss. **MultiClass is not on this
surface**: it is implemented and gated in the Mojo layer, but the wrapper is
one-dimensional and `predict_proba` would need routing through the softmax
over `numClasses - 1` stored approxes. It is named here rather than left out
silently.

**DBSCAN, PCA and OLS are NOT here**, and they are named rather than left
out silently. Their kernels exist in the repository, are verified, and are
benchmarked -- the PCA and OLS numbers quoted in the README come from them --
but they have no caller-facing surface yet, and binding a kernel directly at
this boundary would put policy decisions (workspace budgets, fixed-point
scales, normalization) somewhere no check can see them. That is the mistake
this library is organized to avoid. They land when their surfaces do.

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
from .ensemble import GradientBoosting
from .neighbors import NearestNeighbors

__all__ = [
    "GradientBoosting",
    "KMeans",
    "NearestNeighbors",
    "__version__",
    "numeric_mode",
]

# Named absences. Importing one of these raises with a reason rather than an
# AttributeError, because "why is DBSCAN missing" is a question the answer to
# is interesting and short.
_NOT_YET = {
    "DBSCAN": "dbscan/ported/dbscan/dbscan.mojo:132 (dbscan_fit_impl)",
    "PCA": "decomposition/ported/linalg/detail/pca.mojo:262 (pca_fit)",
    "LinearRegression": "glm/ported/glm/ols.mojo:82 (ols_fit)",
}


def __getattr__(name):
    if name in _NOT_YET:
        raise AttributeError(
            f"mojolearn.{name} is not in this release. Its kernel exists and "
            f"is verified at {_NOT_YET[name]}, but it has no caller-facing "
            "surface yet, and binding a kernel directly would put policy "
            "where no check can see it. See the module docstring."
        )
    raise AttributeError(f"module 'mojolearn' has no attribute {name!r}")
