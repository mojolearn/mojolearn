# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GPU machine learning in Mojo with explicit numerical contracts.

The same source tree targets Apple Metal, NVIDIA CUDA, and AMD HIP. Public
estimators offer ``fast``, ``deterministic``, and ``identical`` modes; the
last promises cross-vendor bit identity only for configurations certified in
the project's support matrix. A GPU is required and there is no CPU fallback.

The API uses familiar scikit-learn shapes but is not a drop-in replacement.
Defaults may follow the mirrored GPU implementation, and unsupported behavior
is refused explicitly. See each class docstring and the project README for
the current surface and limitations.
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
from .ensemble import ExperimentalTwoLevelFeatureFreq, GradientBoosting
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
from ._umap_impl import UMAP
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
# again), closing `archive/evidence/mamba/FEATURE_PARITY.md`'s "PyPI surface: NONE
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
# (`archive/evidence/mamba/FEATURE_PARITY.md` RUN LEDGER -- one box, one vendor; the
# NVIDIA/AMD columns stay OWED there).
# `_mojolearn_training.so` SHIPS IN THE WHEEL, so its Python half must be
# reachable from the package. The release workflow's "nothing ships that
# nothing imports" gate caught this on the 0.4.0 cut: `_training_impl.py`
# was in the wheel and no module imported it, which is the shape
# `mojolearn/torch_ops.py` had for two minor releases before it was
# deleted. This module is NOT abandoned -- it drives a shipped binding and
# has its own surface gate (`tests/test_training_surface.py`) -- so the
# fix is the import, not a deletion. It is imported PRIVATELY and adds no
# public name: training stays internal, exactly as the release notes and
# the paper say, and `__all__` below is unchanged. The import is safe at
# package load because the module resolves its binding lazily through
# `_backend`, so an unbuilt training extension still raises BY NAME when
# touched rather than at import.
from . import _training_impl as _training_impl  # noqa: F401  (private)
from . import mamba
from ._mamba_impl import (
    Mamba1Block,
    Mamba1State,
    Mamba2Block,
    Mamba2State,
    Mamba3Block,
    Mamba3State,
)

# `TransformerBlock` JOINED 2026-09-02, giving profile
# `mojolearn.identical.transformer.fp32.v1` its first Python symbol: the
# lane's forward is CERTIFIED at the kernel level with a recorded
# three-column round (2026-08-28, clauses (a) and (d), the same
# 30-record card bytes on Apple, NVIDIA and AMD -- transformer/README.md
# is the authority and its OWED list is real: clauses (b), (c), (e), the
# sabotage ladder and the whole BACKWARD profile are not in that record)
# and exported no Python symbol at all before this. Not an estimator --
# no `fit` -- so it also lives in the `mojolearn.transformer` submodule;
# its binding `_mojolearn_transformer` (the FIFTEENTH) resolves on FIRST
# USE like every other, so an unbuilt extension leaves the package
# importable and raises BY NAME with the build command when touched
# (`_backend.py`'s design). THE PYTHON PATH RAN 2026-09-02, the day the
# binding first compiled -- `tests/test_transformer_surface.py` printed
# green in all three tiers, 44 checks 0 failed each, with the binding
# rebuilt for each tier (`bash bindings/build_transformer.sh` per tier
# first). ONE BOX, ONE VENDOR -- an Apple M4. The NVIDIA and AMD columns
# of this surface are OWED, and so is the corpus cross-check
# (transformer/README.md's "PyPI surface" section is the ledger).
from . import transformer
from ._transformer_impl import TransformerBlock, TransformerState

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
    "ExperimentalTwoLevelFeatureFreq",
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
    "TransformerBlock",
    "TransformerState",
    "NearestNeighbors",
    "PCA",
    "RadiusNeighbors",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "Ridge",
    "TruncatedSVD",
    "UMAP",
    "kpss_test",
    "linalg",
    "mamba",
    "metrics",
    "transformer",
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
