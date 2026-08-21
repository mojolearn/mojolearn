"""mojolearn: GPU machine-learning algorithms in Mojo, running on Apple Silicon.

The algorithms here mirror the designs of the CUDA implementations that
established them -- CatBoost, cuVS, cuML, RAFT -- and run on hardware none of
those can reach. See NOTICE for the attribution each carries.

WHAT IS IN THIS ALPHA, AND WHAT IS NOT
---------------------------------------
Two estimators: `NearestNeighbors` and `KMeans`. Both are backed by kernels
verified against hand-computed expectations and, for k-means, against
CatBoost and cuVS behaviour.

`GradientBoosting` -- the CatBoost oblivious-tree learner, trained on twelve
of their loss functions -- IS WRITTEN AND IS NOT LOADABLE FROM PYTHON YET.
`mojolearn.ensemble.GradientBoosting` exists and is correct: the identical
entry point runs end to end from a `mojo build` executable. What fails is the
CPython extension, which carries no compiled Metal code (0 AIR blobs against
the executable's 81) and cannot JIT the shared-memory GBDT kernels at load.
PORTING.md 70 has the measurement and the five hypotheses that died on it.
Importing it raises with that reason rather than dying inside Metal.

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

Not validated outside Apple Silicon. One source targets CPU, Metal, CUDA and
ROCm, and this has been measured on exactly one M4. Support is not
validation.
"""

from ._version import __version__
from .cluster import KMeans
from .neighbors import NearestNeighbors

__all__ = ["KMeans", "NearestNeighbors", "__version__"]

# Named absences. Importing one of these raises with a reason rather than an
# AttributeError, because "why is DBSCAN missing" is a question the answer to
# is interesting and short.
_NOT_YET = {
    "GradientBoosting": (
        "gbdt/estimator.mojo:gbdt_fit -- WRITTEN AND WORKING, but the "
        "CPython extension cannot load its Metal kernels: "
        "`mojo build --emit shared-lib` embeds 0 compiled kernels against "
        "an executable's 81, and the runtime JIT fails on the "
        "shared-memory ones. See PORTING.md 70. Use it from Mojo until "
        "the loader is fixed."
    ),
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
