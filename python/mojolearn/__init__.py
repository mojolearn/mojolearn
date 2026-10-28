# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GPU machine learning in Mojo with explicit numerical contracts.

The same source tree targets Apple Metal, NVIDIA CUDA, and AMD HIP.
``identical`` is the default mode. In ``identical`` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.
On a machine with no GPU, ``fit`` trains on the CPU for every estimator with a CPU binding and
refuses by name otherwise, and saved models predict on the CPU: classical
estimators, forests and boosting, neural blocks and loaded language models.
Their supported configurations are documented per API.
``LanguageModelInference`` runs the byte-LM forward pass
(docs/BYTE_LM_CPU_INFERENCE.md). ``LanguageModelHostTrainer`` runs one training
step, forward, backward and the AdamW update
(docs/BYTE_LM_CPU_TRAINING.md), and reproduces the recorded GPU bytes of the
retained three-vendor capture for all 128 of its steps: the gradient, the loss,
and the post-step parameters and both Adam moments. Both are one model profile
at one batch shape; identity is claimed per shape, because the weight gradients
contract over the token count.

The tree and classical estimators additionally offer ``fast`` (Apple GPUs
only). The neural surface (``TransformerBlock``, the
Mamba blocks, the training ops and the byte LM) offers ``identical`` and, on
Apple GPUs, ``fast``; asking it for another mode raises.

The API uses familiar scikit-learn shapes but is not a drop-in replacement.
Defaults may follow the mirrored GPU implementation, and unsupported behavior
is refused explicitly. See each class docstring and the project README for
the current surface and limitations.
"""

from ._version import __version__
from . import _native_loaded_modules  # fixed native module/export connections, before bindings load
from . import _native_array_storage  # fixed native allocation metadata, before binding connections

# THE NUMERIC MODE IS CHOSEN HERE, BEFORE ANY BINDING IS IMPORTED.
# MOJOLEARN_NUMERIC_MODE picks one of TWO tiers: `fast` (explicit opt-in,
# no promise) and `identical` (the default; the same bits across
# Metal, CUDA and HIP). NOT EVERY LANE OFFERS BOTH -- the neural
# bindings are identical-only (`_backend._IDENTICAL_ONLY`), and asking them
# for a lower tier raises rather than quietly resolving to something else.
# The identical tier loads its own binary set from
# python/mojolearn/identical/*.so under the canonical names; see
# _backend.py.
# A release ships one merged native file per directory set; this finder
# serves a binding from it when the binding's own .so is absent
# (`_merged.py`). Installed before any binding is imported.
from . import _merged as _merged
_merged.install()
from . import _backend as _backend

_backend.select()
numeric_mode = _backend.numeric_mode

#: CHOOSE THE TIER IN CODE. `MOJOLEARN_NUMERIC_MODE` still works and still
#: sets the starting value, but it is no longer the only way in: the mode is
#: a runtime choice, and estimators take a per-instance `numeric_mode=`.
#:
#:     mojolearn.set_numeric_mode("fast")     # process default
#:     mojolearn.RandomForestClassifier(numeric_mode="identical")
#:
#: Both tiers ship in ONE wheel and can be loaded into ONE process at
#: once -- measured on 2026-08-29 by calling them interleaved and
#: checking each returned its own arithmetic. See `_backend.load_set`. The
#: process default still applies to the lanes that HAVE the tier; a neural
#: estimator under a `fast` default raises rather than silently upgrading.
set_numeric_mode = _backend.set_default_mode

#: `numeric_profile`: THE NUMBER FORMAT OF THE MATRIX PRODUCTS. Its own
#: parameter, beside `numeric_mode` and independent of it: the mode is the
#: promise (identical, fast), the profile is what the
#: products compute in. Since 2026-09-29 the INFERENCE default is
#: `fixed15_v1` for the transformer models of `mojolearn.models`; every
#: other family (the Mamba models, a bare TransformerBlock) and every
#: trainer compute `fp32_v1` when the caller names nothing, and say so. A
#: profile named where it cannot compute is refused by name, never replaced;
#: one that was measured and failed quality is not offered at all.
#: `weight_format=` is how weights are STORED; this is how the products are
#: COMPUTED. See `_numeric_profile.py`.
#:
#:     mojolearn.numeric_profiles()                   # every registered row
#:     mojolearn.set_numeric_profile("fp32_v1")       # the arithmetic before 2026-09-29
#:     mojolearn.models.CausalLM.load(path, numeric_profile="fp32_v1")
from . import _numeric_profile as _numeric_profile

numeric_profile = _numeric_profile.default_profile
set_numeric_profile = _numeric_profile.set_default_profile
numeric_profiles = _numeric_profile.profiles
numeric_profile_measured = _numeric_profile.measured

#: WHICH GPU API THE LOADED BINARIES WERE COMPILED FOR: 'metal', 'cuda' or
#: 'hip', read back out of the binaries (`checks/vendor.mojo`). On Linux
#: one wheel carries a CUDA set and a HIP set and `_backend._layout()` picks
#: one at import; `vendor()` is what it picked, cross-checked against what
#: the binaries answer. A box with neither device refuses at import, naming
#: what it looked for, UNLESS the CPU inference binding is built. Then the
#: package imports, every GPU binding raises by name on use, and `vendor()`
#: answers 'cpu' (DEVIATION 2615).
vendor = _backend.vendor

#: The GPU architecture directory this process loads from ('sm_80',
#: 'gfx942', ...) and how it was chosen -- the architecture axis
#: (2026-08-30, docs/LINUX_WHEEL.md). Both return None-ish answers on the
#: flat and arch-less layouts, where no choice exists to report.
gpu_arch = _backend.gpu_arch
gpu_arch_how = _backend.gpu_arch_how

#: THE ARRAY TYPE EVERY ESTIMATOR RETURNS (numpy-free-0.7, DEVIATION 2434).
#: NumPy is no longer a dependency of this package: inputs are read
#: through the buffer protocol (a NumPy array, an `array.array`, a
#: `mojolearn.Array`, a nested list where a surface says so) and every
#: array an estimator hands back is a `mojolearn.Array` -- owning its
#: memory, exporting `__array_interface__` and the buffer protocol, so
#: `numpy.asarray(result)` is ZERO-COPY for a caller who has NumPy
#: (`python/mojolearn/NUMPY_FREE_CONTRACT.md`).
from ._array import Array
from . import model_selection
from .model_selection import cross_val_score

from .cluster import KMeans
from .decomposition import PCA, TruncatedSVD
from .density import DBSCAN, KernelDensity
from .ensemble import (ExperimentalTwoLevelFeatureFreq, GradientBoosting, OrderedRMSE,
                       GradientBoostingClassifier, GradientBoostingRegressor)
from .extratrees import ExtraTreesClassifier, ExtraTreesRegressor
from . import preprocessing
from .preprocessing import MinMaxScaler, StandardScaler
from .linear_model import LinearRegression, LogisticRegression, QNRegressor, Ridge
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
from . import linalg
from . import lowbit
from . import models
from .linalg import matmul
import sys as _sys

from . import _metrics_impl as metrics
# `metrics` is an alias of a private module, so `import mojolearn.metrics` and
# `from mojolearn.metrics import accuracy_score` found no `mojolearn/metrics.py`
# and raised ModuleNotFoundError while `mojolearn.metrics.accuracy_score`
# worked (Sep 22 pip smoke). Registering the alias makes both spellings work.
_sys.modules[__name__ + ".metrics"] = metrics
from ._hierarchy_impl import AgglomerativeClustering
from ._iforest_impl import IsolationForest
from ._solver_impl import ElasticNet, Lasso
from ._spectral_impl import SpectralClustering, SpectralEmbedding
from . import manifold
from . import umap
from .umap import UMAP
from .neural_network import SmallMLPTrainer
# The forest host lane (2026-09-13): predict and predict_proba of a SAVED
# RandomForest, ExtraTrees or (the GBDT host lane, the same day)
# GradientBoosting model on a CPU with no GPU, through
# host/_mojolearn_forest_host.so. Resolves its binding on first use, so an
# install without it still imports and raises BY NAME when touched.
# cpu-gpu-cleanup c-core (2026-10-02): the CPU inference modules load on
# first touch (`__getattr__` below), so importing the package on a GPU
# install pulls in no CPU-side module.
_CPU_INFERENCE = {
    "HostForest": "._forest_host", "host_model": "._forest_host", "host_predict": "._forest_host",
    "host_predict_proba": "._forest_host", "HostGBDT": "._gbdt_host",
    # cgr-py (2026-10-03): the language-model names load on first touch
    # too. `language_model` re-exports the CPU trainer and CPU inference
    # from `_byte_lm_host`, which an eager `from .language_model import`
    # pulled in on every import; the GPU trainers come from their own
    # modules, so a GPU install importing the package loads neither.
    "language_model": None,
    "SmallByteLanguageModelTrainer": "._byte_lm_impl",
    "ByteLanguageModelConfig": "._byte_lm_config",
    "LanguageModelTrainer": ".language_model",
    "LanguageModelConfig": ".language_model",
    "LanguageModelInference": "._byte_lm_host",
    "LanguageModelHostTrainer": "._byte_lm_host",
    # The public CPU neural inference (host/_mojolearn_neural_host.so).
    "neural_inference": None,
    # The operation session (`mojolearn.ops`, autograd/ops_api.mojo): loads on
    # first touch, so the package imports without NumPy.
    "ops": None,
    # Model import (docs/plans/MODEL_IMPORT.md, lane ag-import-docs): the
    # package and its entry point load on first touch, as `ops` does.
    "importers": None,
    "import_model": ".importers",
    "MLPInference": ".neural_inference", "TransformerBlockInference": ".neural_inference",
    "Mamba1BlockInference": ".neural_inference", "Mamba2BlockInference": ".neural_inference",
    "Mamba3BlockInference": ".neural_inference", "SambaInference": ".neural_inference",
}
from ._svm_impl import SVC, SVR
# lane/expose-qn-objectives (2026-09-20): the linear machines, the
# quasi-Newton solver on the four hinge-family losses, and the `svm`
# namespace that holds all four.
from . import svm
from .svm import LinearSVC, LinearSVR
from ._arima_impl import ARIMA
from ._tsa_impl import ExponentialSmoothing, kpss_test, select_d
from . import tokenizer
from . import lm_corpus
from .tokenizer import BpeTokenizer
# Public neural inference on the CPU from GPU-trained weights (2026-09-15):
# `neural_inference`'s classes load on first touch (`_CPU_INFERENCE`).

# Workstream D, 2026-09-14: the door-less families of the claim-surface
# census given a binding and a class. `Cholesky` binds `_mojolearn_gp` (the GP
# build already links cholesky/); the other four have their own bindings,
# resolved on FIRST USE like every other, so an unbuilt one leaves the
# package importable and raises BY NAME with the build command when
# touched. Compile-checked on one Apple M4; no box has run them through
# the Python door, and their identity_break lanes and three columns are
# owed.
from ._cholesky_impl import Cholesky
from . import kernel_methods
from .kernel_methods import KernelRidge, Nystroem, RBFSampler
from . import mixture
from .mixture import GaussianMixture
from . import hdbscan
from .hdbscan import HDBSCAN
from . import resample

# `IVFIndex` and `Embedding` JOINED 2026-09-14 (lane/expose-ivf-embedding),
# the two entries `_NOT_YET` held since the claim-surface census. IVFIndex
# is IVF-Flat build plus search under one card (reference: cuVS `ivf_flat`)
# (`bindings/_mojolearn_ivf.mojo`); `pixi run check-ivf` reads ALL OK with
# one IDENTICAL card on Apple, NVIDIA and AMD
# (bench/results/ivf_embed_km_legs_2026-09-14/README.md). Embedding is
# profile mojolearn.identical.embedding.fp32.v1's gather and ascending fold
# with padding_idx and the microbatch carry (`bindings/_mojolearn_embedding.mojo`);
# its clause (a) card is byte-identical on the three columns and its
# sabotage arms are resolved in the same branch. The identity_break lanes
# `ivf` and `embedding` and their three columns are in
# bench/results/identity_break/2026-09-14_ivf-embedding/.
from ._ivf_impl import IVFIndex
from . import embedding
from .embedding import Embedding

# `GaussianProcessRegressor` JOINED 2026-09-01, the last name deleted from
# `_NOT_YET` below and the only one held for a reason other than a missing
# surface. The blocker -- an IDENTICAL card believed to diverge Apple
# against AMD -- was WITHDRAWN at `9835094e` (the divergent lines were a
# sabotage arm's own block; see `_gp_impl.py`'s header), and the surface
# was written that day. Its binding `_mojolearn_gp` resolves on FIRST USE
# like every other: an unbuilt extension leaves the package importable and
# raises BY NAME with the build command when touched (`_backend.py`'s
# design). No reference GP exists in cuML/cuVS/RAFT at the
# pinned commits; scikit-learn `_gpr.py` is the semantics oracle only.
from ._gp_impl import (
    ConstantKernel,
    GaussianProcessRegressor,
    Matern,
    RBF,
    WhiteKernel,
)
# `GaussianProcessClassifier` JOINED 2026-09-15 (lane/gaussian-process-
# classifier): scikit-learn `_gpc.py`'s Laplace approximation on the same
# binding, one-vs-rest past two classes, `optimizer=None` only (DEVIATIONS
# 2830-2833 close DEVIATION 1766).
from ._gpc_impl import GaussianProcessClassifier

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
# public optimizer/loss names historically. The alpha training module below
# now exposes the explicit supported primitives; SmallMLPTrainer also composes
# them internally. The import is safe at
# package load because the module resolves its binding lazily through
# `_backend`, so an unbuilt training extension still raises BY NAME when
# touched rather than at import.
from . import _training_impl as _training_impl  # noqa: F401  (private)
from . import training
from .training import SGD, Adam, AdamW, clip_grad_norm_, cross_entropy
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
# The Samba-shaped training stack (embedding, Mamba-3 / attention blocks,
# final norm, LM head, cross-entropy, AdamW with schedule and clause 9.2
# accumulation, position-keyed RNG, JSON checkpoint) over the training,
# mamba and transformer bindings. `_samba_impl.py` holds no numerics.
from ._samba_impl import SambaConfig, SambaStack
# Existing capacity-oriented trainers are also discoverable at package level.
# These imports resolve no device bindings; GPU work begins on first use.
from .parallel_training import ParallelByteLanguageModelTrainer, ParallelNeuralTrainer
from .model_pool_training import PooledByteLanguageModelTrainer
from .offload_training import OffloadedByteLanguageModelTrainer
from . import parallel_classical
from . import parallel_ensemble
from . import parallel_forecasting
from . import parallel_gaussian_process
from . import parallel_graph
from . import parallel_ivf
from . import parallel_model_selection
from . import parallel_neighbors
from . import parallel_neighbors_reference
from . import parallel_preprocessing
from .parallel_ivf import DistributedIVFIndex

__all__ = [
    "parallel_classical",
    "parallel_ensemble",
    "parallel_forecasting",
    "parallel_gaussian_process",
    "parallel_graph",
    "parallel_ivf",
    "parallel_model_selection",
    "parallel_neighbors",
    "parallel_neighbors_reference",
    "parallel_preprocessing",
    "DistributedIVFIndex",
    "ParallelByteLanguageModelTrainer",
    "ParallelNeuralTrainer",
    "PooledByteLanguageModelTrainer",
    "OffloadedByteLanguageModelTrainer",
    "model_selection",
    "cross_val_score",
    "preprocessing",
    "MinMaxScaler",
    "StandardScaler",
    "Array",
    "SGD",
    "Adam",
    "AdamW",
    "SambaConfig",
    "SambaStack",
    "ARIMA",
    "AgglomerativeClustering",
    "Cholesky",
    "KernelRidge",
    "Nystroem",
    "RBFSampler",
    "GaussianMixture",
    "HDBSCAN",
    "IVFIndex",
    "Embedding",
    "embedding",
    "kernel_methods",
    "mixture",
    "hdbscan",
    "resample",
    "ConstantKernel",
    "DBSCAN",
    "GaussianProcessClassifier",
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
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "OrderedRMSE",
    "ExponentialSmoothing",
    "BpeTokenizer",
    "lm_corpus",
    "GPT2Tokenizer",
    "MLPInference",
    "TransformerBlockInference",
    "Mamba1BlockInference",
    "Mamba2BlockInference",
    "Mamba3BlockInference",
    "SambaInference",
    "IsolationForest",
    "KMeans",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "Lasso",
    "LinearRegression",
    "LinearSVC",
    "LinearSVR",
    "LogisticRegression",
    "QNRegressor",
    "Mamba1Block",
    "Mamba1State",
    "Mamba2Block",
    "Mamba2State",
    "Mamba3Block",
    "Mamba3State",
    "SVC",
    "SVR",
    "svm",
    "SpectralClustering",
    "SpectralEmbedding",
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
    "SmallMLPTrainer",
    "SmallByteLanguageModelTrainer",
    "ByteLanguageModelConfig",
    "LanguageModelConfig",
    "LanguageModelTrainer",
    "LanguageModelInference",
    "LanguageModelHostTrainer",
    "HostForest",
    "HostGBDT",
    "host_model",
    "host_predict",
    "host_predict_proba",
    "kpss_test",
    "linalg",
    "lowbit",
    "models",
    "matmul",
    "mamba",
    "manifold",
    "metrics",
    "neural_network",
    "language_model",
    "ops",
    "importers",
    "import_model",
    "tokenizer",
    "transformer",
    "training",
    "umap",
    "clip_grad_norm_",
    "cross_entropy",
    "select_d",
    "__version__",
    "numeric_mode",
    "set_numeric_mode",
    "numeric_profile",
    "set_numeric_profile",
    "numeric_profiles",
    "numeric_profile_measured",
    "vendor",
    "gpu_arch",
    "gpu_arch_how",
]

# THE ALGORITHM EXPANSION'S DOORS (lane/algos-prep, 2026-09-27). Nine lanes
# add public algorithms in parallel and none of them edits this file: each
# owns `_expansion_<lane>.py`, whose `__all__` is what it makes public. They
# are imported here in lane order (host_surface.EXPANSION_LANES) and their
# names bound and appended to `__all__`. A name that is already public, or
# that two doors both export, is refused at import BY NAME rather than one
# silently shadowing the other, which is what a bare `import *` would do.
from . import (_expansion_linear, _expansion_cluster, _expansion_neighbors, _expansion_decomp,
               _expansion_prep, _expansion_sequence, _expansion_trees, _expansion_cnn, _expansion_ann,
               _expansion_metrics)
_LAZY_EXPANSION_EXPORTS = {}
for _door in (_expansion_linear, _expansion_cluster, _expansion_neighbors, _expansion_decomp,
              _expansion_prep, _expansion_sequence, _expansion_trees, _expansion_cnn, _expansion_ann,
              _expansion_metrics):
    for _name in _door.__all__:
        if _name in __all__ or _name in globals():
            raise ImportError(f"mojolearn: {_door.__name__} exports {_name!r}, which is already public "
                              "or already bound; choose another name")
        if _door is _expansion_sequence:
            # Its optional NumPy-based implementations must not prevent the
            # core library or the verifier's installation guidance importing.
            _LAZY_EXPANSION_EXPORTS[_name] = _door
        else:
            globals()[_name] = getattr(_door, _name)
        __all__.append(_name)
del _door
globals().pop("_name", None)

# Named absences. Importing one of these raises with a reason rather than an
# AttributeError, because "why is X missing" is a question the answer to is
# interesting and short. Each value names the thing that EXISTS and where it
# stops.
#
# It is EMPTY today, and that is the point. IVFIndex and Embedding were
# added by the claim-surface census on 2026-09-14 (both finished underneath
# with no Python door) and deleted the same day when their doors were
# written. Before them four entries had been deleted from it
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
# withdrawn, because exposing an estimator is a shipping decision.
# Keep the mechanism: the next lane that is finished underneath
# and unreachable from Python belongs in here, by name, not left to an
# AttributeError -- and so does the next lane withheld on purpose, with the
# purpose written out the way the GP's was.
_NOT_YET = {}


def __getattr__(name):
    if name in _CPU_INFERENCE:
        import importlib
        where = _CPU_INFERENCE[name]
        if where is None:   # a submodule named in __all__
            value = importlib.import_module("." + name, __name__)
        else:
            value = getattr(importlib.import_module(where, __name__), name)
        globals()[name] = value
        return value
    if name in _LAZY_EXPANSION_EXPORTS:
        value = getattr(_LAZY_EXPANSION_EXPORTS[name], name)
        globals()[name] = value
        return value
    if name == "GPT2Tokenizer":
        # Renamed BpeTokenizer 2026-09-18; kept in __all__ because 0.8.x
        # shipped it. tokenizer.__getattr__ warns and returns the same class.
        return getattr(tokenizer, name)
    if name in _NOT_YET:
        raise AttributeError(
            f"mojolearn.{name} is not in this release. What exists stops at "
            f"{_NOT_YET[name]}; binding it directly would put policy where "
            "no check can see it. See the module docstring."
        )
    raise AttributeError(f"module 'mojolearn' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))  # glue: module attribute names for dir


# CPU categorical GradientBoosting prediction is a native tree operation too;
# connect its fixed host metadata after the public classes and lazy names are
# available. The GPU package keeps explicit host APIs lazy. An unavailable
# native tree owner has no vendor context and remains importable as before.
from ._native_trees_bridge import connect_cpu_prediction as _connect_cpu_tree_prediction

_connect_cpu_tree_prediction()
del _connect_cpu_tree_prediction

# The native Array operations share the base binding's core helpers.
from ._array import connect_helpers as _connect_array_helpers
from ._buffer import _native_optional as _array_helper
_connect_array_helpers(_array_helper)
del _connect_array_helpers, _array_helper

# Field-only constructors must precede every method/storage identity snapshot.
from ._native_linear_constructors import install as _install_linear_constructors
_install_linear_constructors()
del _install_linear_constructors

# Full-reset setters precede every class-layout snapshot; the linear subset keeps its separate contract.
from ._native_full_reset_params import install as _install_full_reset_params
_install_full_reset_params()
del _install_full_reset_params

# Field-only declared prep/decomp constructors precede every class-layout snapshot too.
from ._native_declared_constructors import install as _install_declared_constructors
_install_declared_constructors()
del _install_declared_constructors

# Layouts connected at module import (neighbor queries, host inference, ...)
# predate the installers above: rebind their replaced __init__/set_params.
from ._native_classical_bridge import refresh_installed_members as _refresh_installed_members
_refresh_installed_members()
del _refresh_installed_members

# Connect CV's fixed native estimator ABIs after every public class exists.
from ._native_mode_metadata import connect_classes as _connect_mode_metadata_classes
from . import _mode as _mode_metadata_owner
import sys as _mode_metadata_sys
_connect_mode_metadata_classes(_mode_metadata_owner._MODE_METADATA_CONTEXT, _mode_metadata_sys.modules[__name__])
del _connect_mode_metadata_classes, _mode_metadata_owner, _mode_metadata_sys

# Native attribute slots precede the estimator and parameter layout snapshots.
from ._native_public_attributes import connect_classes as _connect_public_attributes
import sys as _attribute_sys
_connect_public_attributes(_attribute_sys.modules[__name__])
del _connect_public_attributes, _attribute_sys

from ._native_msel_dispatch import connect as _connect_native_cv
import sys as _cv_sys
_connect_native_cv(model_selection._NATIVE_CONTEXT, _cv_sys.modules[__name__])
del _connect_native_cv, _cv_sys
# The prep owners (RFE, IterativeImputer, TargetEncoder's cv splitter) reach a
# wrapped estimator or splitter through the same compiled model-selection entry
# and plan table (bindings/prep_native_estimator.mojo), never its Python methods.
from . import _expansion_prep as _prep_owner
_prep_owner._ENCODE_CONTEXT.update(native_estimator_entry=model_selection._NATIVE,
                                   native_estimator_context=model_selection._NATIVE_CONTEXT)
del _prep_owner
from . import _expansion_trees as _tree_owner
_tree_owner._TREES_CONTEXT.update(native_estimator_entry=model_selection._NATIVE,
                                 native_estimator_context=model_selection._NATIVE_CONTEXT)
del _tree_owner

# Parameter-read schemas capture the final built-in installers' method and
# descriptor identities; the connection changes only this private plan table.
from ._native_params_setter import connect as _connect_params_setter
from . import _mode as _params_setter_owner
_connect_params_setter(model_selection._NATIVE_CONTEXT, _params_setter_owner._PARAMS_METADATA_CONTEXT)
del _connect_params_setter, _params_setter_owner

from ._native_params_metadata import connect_classes as _connect_params_metadata_classes
from . import _mode as _params_metadata_owner
import sys as _params_metadata_sys
_connect_params_metadata_classes(_params_metadata_owner._PARAMS_METADATA_CONTEXT, _params_metadata_sys.modules[__name__])
from ._native_declared_params import connect as _connect_declared_params
_connect_declared_params(_params_metadata_sys.modules[__name__], model_selection._NATIVE_CONTEXT,
                         _params_metadata_owner._PARAMS_METADATA_CONTEXT)
del _connect_declared_params
del _connect_params_metadata_classes, _params_metadata_owner, _params_metadata_sys

# Complete constructor ownership only after the native setter layouts exist.
from ._native_linear_constructors import connect as _connect_linear_constructors
from ._native_params_setter import _CONTEXT as _linear_constructor_setters
_connect_linear_constructors(_linear_constructor_setters)
del _connect_linear_constructors, _linear_constructor_setters

# Connect full-reset ownership and nested routes after all getter/setter plans.
from ._native_full_reset_params import connect as _connect_full_reset_params
from ._native_params_setter import _CONTEXT as _full_reset_linear
from ._native_declared_params import CONTEXT as _full_reset_declared
from . import _mode as _full_reset_mode
_connect_full_reset_params(_full_reset_mode._PARAMS_METADATA_CONTEXT, _full_reset_linear,
                           _full_reset_declared, model_selection._NATIVE_CONTEXT)
del _connect_full_reset_params, _full_reset_linear, _full_reset_declared, _full_reset_mode

# Declared-family constructors release overwritten values under the declared setter's admission.
from ._native_declared_constructors import connect as _connect_declared_constructors
from ._native_declared_params import CONTEXT as _declared_constructor_setters
_connect_declared_constructors(_declared_constructor_setters)
del _connect_declared_constructors, _declared_constructor_setters

# Snapshot family storage after every native constructor/attribute installer.
from ._native_family_layouts import connect as _connect_family_layouts
import sys as _family_layout_sys
_connect_family_layouts(_family_layout_sys.modules[__name__])
del _connect_family_layouts, _family_layout_sys

# Default member estimators are constructed only after every constructor above is connected.
from ._native_msel_prep import connect_estimator_defaults as _connect_estimator_defaults
_connect_estimator_defaults()
del _connect_estimator_defaults

# The public parallel drivers' class/storage snapshots are taken last: every
# import-time class connection above (native attribute slots, constructors)
# precedes them, so a correct call is not refused as a changed hook.
from ._native_parallel_public import refresh as _refresh_parallel_public
_refresh_parallel_public(parallel_ensemble._FAMILY_CONTEXT)
del _refresh_parallel_public

# Complete process backend metadata is native after import-time provider setup.
from ._native_backend_public import connect as _connect_backend_public
import sys as _backend_public_sys
_connect_backend_public(_backend_public_sys.modules[__name__])
del _connect_backend_public, _backend_public_sys
# The same one-call native functions connect() installed, named statically.
from ._native_backend_public import (gpu_arch, gpu_arch_how, numeric_mode, vendor,
                                     set_default_mode as set_numeric_mode)
# Profile bootstrap defaults are already shared with the native providers.
# Replace both public aliases and module functions after import connections.
from ._native_numeric_profile import connect as _connect_numeric_profile_public
import sys as _numeric_profile_public_sys
_connect_numeric_profile_public(_numeric_profile_public_sys.modules[__name__])
del _connect_numeric_profile_public, _numeric_profile_public_sys
# The same one-call native functions connect() installed, named statically.
from ._native_numeric_profile import (_profiles as numeric_profiles, _measured as numeric_profile_measured,
                                      _default_profile as numeric_profile,
                                      _set_default_profile as set_numeric_profile)
