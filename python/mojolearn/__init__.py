# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""mojolearn: GPU machine-learning algorithms in Mojo, running on Apple Silicon.

The algorithms here mirror the designs of the CUDA implementations that
established them -- CatBoost, cuVS, cuML, RAFT -- and run on hardware none of
those can reach. See NOTICE for the attribution each carries.

WHAT IS IN THIS ALPHA, AND WHAT IS NOT
---------------------------------------
Twenty-seven estimators, three submodules, two functions and (since
2026-09-01) the three Mamba block classes: `NearestNeighbors`, `KNeighborsClassifier`,
`KNeighborsRegressor`, `RadiusNeighbors`, `KMeans`, `DBSCAN`, `PCA`, `TruncatedSVD`,
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
**`ARIMA` IS here** (since 2026-09-01), and `_NOT_YET` below, which existed
to explain its absence, is now empty. It is the one estimator in this package
whose `y` is 2-D: the lane is BATCHED, one series per row, each with its own
parameters, and `_arima_impl.py` says on the class what else follows from
that.
**THE MAMBA BLOCKS ARE here** (since 2026-09-01): `Mamba1Block`,
`Mamba2Block` and `Mamba3Block` (also as the `mojolearn.mamba`
submodule), reference-pinned SSM blocks with explicit caller-owned state
for prefill, continuation and single-token decode. They are NOT
estimators -- no `fit` -- and they exist for cross-checking:
`_mamba_impl.py` carries the contracts. The Mamba-1/2 surface gate
printed green in all three tiers on 2026-09-01 (one box, one vendor);
the Mamba-3 arms printed green the same evening at `08a38a13` in all
three tiers, identical bitwise-asserted (`mamba/FEATURE_PARITY.md`'s
RUN LEDGER; still one box, one vendor).
**`GaussianProcessRegressor` IS here** (since 2026-09-01, later the same
day), with its four kernel classes `RBF`, `Matern`, `ConstantKernel` and
`WhiteKernel`. It was the one `_NOT_YET` entry ever held back for a reason
OTHER than a missing surface: its IDENTICAL card was believed to diverge
Apple against AMD on 8 of 3,494 stages. That reading was WITHDRAWN at
`9835094e` -- the eight lines are the sabotaged half of one
`GP_SAB_STD_EXP` clean-then-sabotaged pair, an arm built to make a device
`exp` differ, and the shipped path is byte-identical on the other 3,486
lines -- and with the sole blocker withdrawn the withholding text
contradicted the evidence, so the surface was written (`_gp_impl.py`,
whose header carries the history).

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
#: 'hip', read back out of the binaries (`checks/vendor.mojo`). On Linux
#: one wheel carries a CUDA set and a HIP set and `_backend._layout()` picks
#: one at import; `vendor()` is what it picked, cross-checked against what
#: the binaries answer. There is no CPU path: a Linux box with neither
#: device refuses at import, naming what it looked for.
vendor = _backend.vendor

#: The GPU architecture directory this process loads from ('sm_80',
#: 'gfx942', ...) and how it was chosen -- the architecture axis
#: (2026-08-30, docs/LINUX_WHEEL.md). Both return None-ish answers on the
#: flat and arch-less layouts, where no choice exists to report.
gpu_arch = _backend.gpu_arch
gpu_arch_how = _backend.gpu_arch_how

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
    RadiusNeighbors,
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
# `SVR` joined them on 2026-09-01, on the same lane as `SVC` and through the
# same compiled extension, but WITHOUT `SVC`'s three-vendor card: the
# regression half was gated after leg 11 and has not been in a cross-vendor
# round. Its class says so.
#
# `ARIMA` JOINED THEM ON 2026-09-01 and this block said it could not. It read
# that ARIMA was "deliberately absent" because its lane had no `fit`, so the
# class would have to demand its own answer as an argument. True while it was
# true: `estimate_x0` (over an own-written Householder QR) and an own-written
# batched L-BFGS landed that day, gated in both numeric tiers, and what was
# missing afterwards was only the Python door. `arima/` has a three-vendor
# identity card for its KALMAN FILTER at `221aa141`; the FIT does not, and the
# class says so rather than inheriting the lane's headline.
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
from ._svm_impl import SVC, SVR
from ._arima_impl import ARIMA
from ._tsa_impl import ExponentialSmoothing, kpss_test, select_d

# `GaussianProcessRegressor` JOINED 2026-09-01, the last name ever held in
# `_NOT_YET` below and the only one held for a reason other than a missing
# surface. The blocker -- an IDENTICAL card believed to diverge Apple
# against AMD -- was WITHDRAWN at `9835094e` (the divergent lines were a
# sabotage arm's own block; see `_gp_impl.py`'s header), and the surface
# was written that day. Its binding `_mojolearn_gp` resolves on FIRST USE
# like every other: an unbuilt extension leaves the package importable and
# raises BY NAME with the build command when touched (`_backend.py`'s
# design). ORIGINAL WORK: no upstream GP exists in cuML/cuVS/RAFT at the
# pinned commits; scikit-learn `_gpr.py` is the semantics oracle only.
from ._gp_impl import (
    ConstantKernel,
    GaussianProcessRegressor,
    Matern,
    RBF,
    WhiteKernel,
)

# `Mamba1Block` / `Mamba2Block` JOINED 2026-09-01 (later the same day
# again), closing `mamba/FEATURE_PARITY.md`'s "PyPI surface: NONE
# EXISTS" row -- the single largest parity gap: both blocks were
# certified at the kernel level (Mamba-1 three-vendor, Mamba-2 gated on
# Apple) and exported no Python symbol at all. Not estimators -- no
# `fit` -- so they also live in the `mojolearn.mamba` submodule; their
# binding `_mojolearn_mamba` (the FOURTEENTH) resolves on FIRST USE like
# every other, so an unbuilt extension leaves the package importable and
# raises BY NAME with the build command when touched (`_backend.py`'s
# design). The Mamba-1/2 surface gate printed green in all three tiers
# on 2026-09-01; `Mamba3Block` joined later the same day and its
# `tests/test_mamba_surface.py` arms printed green that evening at
# `08a38a13`, all three tiers, identical bitwise-asserted
# (`mamba/FEATURE_PARITY.md` RUN LEDGER -- one box, one vendor; the
# NVIDIA/AMD columns stay OWED there).
from . import mamba
from ._mamba_impl import (
    Mamba1Block,
    Mamba1State,
    Mamba2Block,
    Mamba2State,
    Mamba3Block,
    Mamba3State,
)

__all__ = [
    "ARIMA",
    "AgglomerativeClustering",
    "ConstantKernel",
    "DBSCAN",
    "GaussianProcessRegressor",
    "KernelDensity",
    "Matern",
    "RBF",
    "WhiteKernel",
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
    "Mamba1Block",
    "Mamba1State",
    "Mamba2Block",
    "Mamba2State",
    "Mamba3Block",
    "Mamba3State",
    "SVC",
    "SVR",
    "SpectralClustering",
    "NearestNeighbors",
    "PCA",
    "RadiusNeighbors",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "Ridge",
    "TruncatedSVD",
    "kpss_test",
    "linalg",
    "mamba",
    "metrics",
    "select_d",
    "__version__",
    "numeric_mode",
    "set_numeric_mode",
    "vendor",
]

# Named absences. Importing one of these raises with a reason rather than an
# AttributeError, because "why is X missing" is a question the answer to is
# interesting and short. Each value names the thing that EXISTS and where it
# stops.
#
# IT IS EMPTY, AND THAT IS THE POINT. Four entries have been deleted from it
# and none was ever reworded: `KNeighborsClassifier` / `KNeighborsRegressor`
# on 2026-08-23, `SVR` on 2026-09-01, `ARIMA` the same day, and
# `GaussianProcessRegressor` the same day again. The first three said the
# kernels, the oracle and the gates were done and only the Python surface was
# missing, each was true when written, and in every case the fix for the
# sentence was to write the surface rather than to soften the sentence. The
# fourth was DIFFERENT and its difference is worth keeping: it was withheld
# ON PURPOSE with the surface unwritten, because its IDENTICAL card was
# believed to diverge Apple against AMD on 8 of 3,494 stages. That reading
# was WITHDRAWN at `9835094e` (2026-09-01): the eight lines were the
# SABOTAGED half of one GP_SAB_STD_EXP clean-then-sabotaged pair -- an arm
# whose whole statement is that a device exp is a vendor choice in its last
# bit -- and the shipped path is byte-identical on the other 3,486 lines
# (bench/results/e1/GP_CROSS_VENDOR_DIVERGENCE.md, corrected in place). The
# entry was deleted when the surface was written, not when the reading was
# withdrawn, because exposing an estimator is a shipping decision and Andrew
# delegated it. Keep the mechanism: the next lane that is finished underneath
# and unreachable from Python belongs in here, by name, not left to an
# AttributeError -- and so does the next lane withheld on purpose, with the
# purpose written out the way the GP's was.
_NOT_YET = {}


def __getattr__(name):
    if name in _NOT_YET:
        raise AttributeError(
            f"mojolearn.{name} is not in this release. What exists stops at "
            f"{_NOT_YET[name]}; binding it directly would put policy where "
            "no check can see it. See the module docstring."
        )
    raise AttributeError(f"module 'mojolearn' has no attribute {name!r}")
