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

# The tokenizer is core: its host unit carries the compiled descriptors every
# family connects through, so it loads with the package.
from . import tokenizer
from .tokenizer import BpeTokenizer
from . import _families as _families
import importlib as _importlib

# EVERY OTHER PUBLIC NAME LOADS ITS FAMILY ON FIRST TOUCH (lane
# lazy-family-import, 2026-11-12; `_families.py`). `import mojolearn` loads
# the core only; `mojolearn.Ridge` (or `from mojolearn import Ridge`, or
# `import mojolearn.linear_model`) imports that family's modules, whose own
# import-time connections run unchanged and load its native units, and then
# the package's class connections run over the newly loaded classes. A value
# is the module that defines the name ('.module'); None is a submodule.
#
# THE ALGORITHM EXPANSION'S DOORS (lane/algos-prep, 2026-09-27) own
# `_expansion_<lane>.py`, whose `__all__` is what each makes public; their
# names are routed here in door order (host_surface.EXPANSION_LANES). When a
# door loads, `_families` refuses BY NAME a door whose `__all__` differs from
# the names routed to it, so no name can belong to two doors or shadow
# another (tests/test_lazy_family_import.py checks this table against the
# doors' source).
_LAZY_NAMES = {
    "model_selection": None, "cross_val_score": ".model_selection",
    "cluster": None, "KMeans": ".cluster",
    "decomposition": None, "PCA": ".decomposition", "TruncatedSVD": ".decomposition",
    "density": None, "DBSCAN": ".density", "KernelDensity": ".density",
    "ensemble": None, "ExperimentalTwoLevelFeatureFreq": ".ensemble", "GradientBoosting": ".ensemble",
    "OrderedRMSE": ".ensemble", "GradientBoostingClassifier": ".ensemble",
    "GradientBoostingRegressor": ".ensemble",
    "extratrees": None, "ExtraTreesClassifier": ".extratrees", "ExtraTreesRegressor": ".extratrees",
    "preprocessing": None, "MinMaxScaler": ".preprocessing", "StandardScaler": ".preprocessing",
    "linear_model": None, "LinearRegression": ".linear_model", "LogisticRegression": ".linear_model",
    "QNRegressor": ".linear_model", "Ridge": ".linear_model",
    "neighbors": None, "KNeighborsClassifier": ".neighbors", "KNeighborsRegressor": ".neighbors",
    "NearestNeighbors": ".neighbors", "RadiusNeighbors": ".neighbors",
    "randomforest": None, "RandomForestClassifier": ".randomforest", "RandomForestRegressor": ".randomforest",
    "linalg": None, "matmul": ".linalg", "lowbit": None, "models": None,
    # `mojolearn/metrics.py` puts the private module itself under this name.
    "metrics": None,
    "AgglomerativeClustering": "._hierarchy_impl", "IsolationForest": "._iforest_impl",
    "ElasticNet": "._solver_impl", "Lasso": "._solver_impl",
    "SpectralClustering": "._spectral_impl", "SpectralEmbedding": "._spectral_impl",
    "manifold": None, "umap": None, "UMAP": ".umap",
    "neural_network": None, "SmallMLPTrainer": ".neural_network",
    # The CPU inference and language-model names (cpu-gpu-cleanup c-core,
    # 2026-10-02; cgr-py 2026-10-03), lazy before this lane too: a GPU install
    # importing the package loads no CPU-side module.
    "HostForest": "._forest_host", "host_model": "._forest_host", "host_predict": "._forest_host",
    "host_predict_proba": "._forest_host", "HostGBDT": "._gbdt_host",
    "language_model": None,
    "SmallByteLanguageModelTrainer": "._byte_lm_impl",
    "ByteLanguageModelConfig": "._byte_lm_config",
    "LanguageModelTrainer": ".language_model",
    "LanguageModelConfig": ".language_model",
    "LanguageModelInference": "._byte_lm_host",
    "LanguageModelHostTrainer": "._byte_lm_host",
    "neural_inference": None,
    # The operation session (`mojolearn.ops`, autograd/ops_api.mojo) and model
    # import (docs/plans/MODEL_IMPORT.md): the package imports without NumPy.
    "ops": None,
    "importers": None,
    "import_model": ".importers",
    "MLPInference": ".neural_inference", "TransformerBlockInference": ".neural_inference",
    "Mamba1BlockInference": ".neural_inference", "Mamba2BlockInference": ".neural_inference",
    "Mamba3BlockInference": ".neural_inference", "SambaInference": ".neural_inference",
    "SVC": "._svm_impl", "SVR": "._svm_impl",
    "svm": None, "LinearSVC": ".svm", "LinearSVR": ".svm",
    "ARIMA": "._arima_impl",
    "ExponentialSmoothing": "._tsa_impl", "kpss_test": "._tsa_impl", "select_d": "._tsa_impl",
    "lm_corpus": None,
    "Cholesky": "._cholesky_impl",
    "kernel_methods": None, "KernelRidge": ".kernel_methods", "Nystroem": ".kernel_methods",
    "RBFSampler": ".kernel_methods",
    "mixture": None, "GaussianMixture": ".mixture",
    "hdbscan": None, "HDBSCAN": ".hdbscan",
    "resample": None,
    "IVFIndex": "._ivf_impl",
    "embedding": None, "Embedding": ".embedding",
    "ConstantKernel": "._gp_impl", "GaussianProcessRegressor": "._gp_impl", "Matern": "._gp_impl",
    "RBF": "._gp_impl", "WhiteKernel": "._gp_impl",
    "GaussianProcessClassifier": "._gpc_impl",
    "training": None, "SGD": ".training", "Adam": ".training", "AdamW": ".training",
    "clip_grad_norm_": ".training", "cross_entropy": ".training",
    "mamba": None, "Mamba1Block": "._mamba_impl", "Mamba1State": "._mamba_impl",
    "Mamba2Block": "._mamba_impl", "Mamba2State": "._mamba_impl", "Mamba3Block": "._mamba_impl",
    "Mamba3State": "._mamba_impl",
    "transformer": None, "TransformerBlock": "._transformer_impl", "TransformerState": "._transformer_impl",
    "SambaConfig": "._samba_impl", "SambaStack": "._samba_impl",
    "parallel_training": None, "ParallelByteLanguageModelTrainer": ".parallel_training",
    "ParallelNeuralTrainer": ".parallel_training",
    "model_pool_training": None, "PooledByteLanguageModelTrainer": ".model_pool_training",
    "offload_training": None, "OffloadedByteLanguageModelTrainer": ".offload_training",
    "parallel_classical": None, "parallel_ensemble": None, "parallel_forecasting": None,
    "parallel_gaussian_process": None, "parallel_graph": None, "parallel_ivf": None,
    "parallel_model_selection": None, "parallel_neighbors": None, "parallel_neighbors_reference": None,
    "parallel_preprocessing": None, "DistributedIVFIndex": ".parallel_ivf",
    # _expansion_linear
    "SGDClassifier": "._expansion_linear", "SGDRegressor": "._expansion_linear",
    "PoissonRegressor": "._expansion_linear", "GammaRegressor": "._expansion_linear",
    "TweedieRegressor": "._expansion_linear", "HuberRegressor": "._expansion_linear",
    "BayesianRidge": "._expansion_linear", "ARDRegression": "._expansion_linear", "Lars": "._expansion_linear",
    "LassoLars": "._expansion_linear", "QuantileRegressor": "._expansion_linear",
    "Perceptron": "._expansion_linear", "PassiveAggressiveClassifier": "._expansion_linear",
    "PassiveAggressiveRegressor": "._expansion_linear", "SGDOneClassSVM": "._expansion_linear",
    "RidgeClassifier": "._expansion_linear", "RidgeCV": "._expansion_linear", "LassoCV": "._expansion_linear",
    "ElasticNetCV": "._expansion_linear", "LogisticRegressionCV": "._expansion_linear",
    "IsotonicRegression": "._expansion_linear",
    # _expansion_cluster
    "MiniBatchKMeans": "._expansion_cluster", "BisectingKMeans": "._expansion_cluster",
    "MeanShift": "._expansion_cluster", "OPTICS": "._expansion_cluster",
    "AffinityPropagation": "._expansion_cluster", "BayesianGaussianMixture": "._expansion_cluster",
    # _expansion_neighbors
    "LocalOutlierFactor": "._expansion_neighbors", "NearestCentroid": "._expansion_neighbors",
    "OneClassSVM": "._expansion_neighbors", "KernelPCA": "._expansion_neighbors",
    "PolynomialCountSketch": "._expansion_neighbors", "AdditiveChi2Sampler": "._expansion_neighbors",
    "SkewedChi2Sampler": "._expansion_neighbors", "LabelPropagation": "._expansion_neighbors",
    "LabelSpreading": "._expansion_neighbors", "KNNImputer": "._expansion_neighbors",
    "PageRank": "._expansion_neighbors", "connected_components": "._expansion_neighbors",
    "Louvain": "._expansion_neighbors", "SVGP": "._expansion_neighbors",
    # _expansion_decomp
    "IncrementalPCA": "._expansion_decomp", "GaussianRandomProjection": "._expansion_decomp",
    "SparseRandomProjection": "._expansion_decomp", "johnson_lindenstrauss_min_dim": "._expansion_decomp",
    "NMF": "._expansion_decomp", "FastICA": "._expansion_decomp", "FactorAnalysis": "._expansion_decomp",
    "lu_factor": "._expansion_decomp", "lu_solve": "._expansion_decomp", "solve": "._expansion_decomp",
    "lstsq": "._expansion_decomp", "randomized_svd": "._expansion_decomp", "PLSRegression": "._expansion_decomp",
    "PLSCanonical": "._expansion_decomp", "CCA": "._expansion_decomp", "DictionaryLearning": "._expansion_decomp",
    "MiniBatchDictionaryLearning": "._expansion_decomp", "SparsePCA": "._expansion_decomp",
    "MiniBatchSparsePCA": "._expansion_decomp", "sparse_encode": "._expansion_decomp",
    "SparseCoder": "._expansion_decomp", "LatentDirichletAllocation": "._expansion_decomp",
    "Isomap": "._expansion_decomp", "MDS": "._expansion_decomp", "ClassicalMDS": "._expansion_decomp",
    "LocallyLinearEmbedding": "._expansion_decomp", "MinCovDet": "._expansion_decomp",
    "EllipticEnvelope": "._expansion_decomp", "AlternatingLeastSquares": "._expansion_decomp",
    # _expansion_prep
    "f_classif": "._expansion_prep", "f_regression": "._expansion_prep", "r_regression": "._expansion_prep",
    "chi2": "._expansion_prep", "mutual_info_classif": "._expansion_prep",
    "mutual_info_regression": "._expansion_prep", "RobustScaler": "._expansion_prep",
    "MaxAbsScaler": "._expansion_prep", "OrdinalEncoder": "._expansion_prep", "OneHotEncoder": "._expansion_prep",
    "TargetEncoder": "._expansion_prep", "SimpleImputer": "._expansion_prep",
    "KBinsDiscretizer": "._expansion_prep", "GaussianNB": "._expansion_prep", "MultinomialNB": "._expansion_prep",
    "BernoulliNB": "._expansion_prep", "LinearDiscriminantAnalysis": "._expansion_prep",
    "QuadraticDiscriminantAnalysis": "._expansion_prep", "QuantileTransformer": "._expansion_prep",
    "PowerTransformer": "._expansion_prep", "Normalizer": "._expansion_prep",
    "PolynomialFeatures": "._expansion_prep", "SplineTransformer": "._expansion_prep",
    "Binarizer": "._expansion_prep", "LabelEncoder": "._expansion_prep", "LabelBinarizer": "._expansion_prep",
    "MultiLabelBinarizer": "._expansion_prep", "IterativeImputer": "._expansion_prep",
    "VarianceThreshold": "._expansion_prep", "SelectKBest": "._expansion_prep", "RFE": "._expansion_prep",
    "ComplementNB": "._expansion_prep", "CategoricalNB": "._expansion_prep",
    # _expansion_sequence
    "LSTMRegressor": "._expansion_sequence", "LSTMClassifier": "._expansion_sequence",
    "GRURegressor": "._expansion_sequence", "GRUClassifier": "._expansion_sequence",
    "RMSprop": "._expansion_sequence", "Adagrad": "._expansion_sequence", "AutoARIMA": "._expansion_sequence",
    "STL": "._expansion_sequence", "VAR": "._expansion_sequence", "MLPClassifier": "._expansion_sequence",
    "MLPRegressor": "._expansion_sequence", "RNNRegressor": "._expansion_sequence",
    "RNNClassifier": "._expansion_sequence", "Lion": "._expansion_sequence", "Adafactor": "._expansion_sequence",
    "LAMB": "._expansion_sequence", "Adamax": "._expansion_sequence", "NAdam": "._expansion_sequence",
    "StepLR": "._expansion_sequence", "ExponentialLR": "._expansion_sequence", "OneCycleLR": "._expansion_sequence",
    "LayerNorm": "._expansion_sequence", "layer_norm_forward": "._expansion_sequence",
    "layer_norm_backward": "._expansion_sequence", "Theta": "._expansion_sequence",
    "OptimizedTheta": "._expansion_sequence", "DynamicTheta": "._expansion_sequence",
    "DynamicOptimizedTheta": "._expansion_sequence", "AutoTheta": "._expansion_sequence",
    "CrostonClassic": "._expansion_sequence", "CrostonOptimized": "._expansion_sequence",
    "CrostonSBA": "._expansion_sequence", "ETS": "._expansion_sequence", "DampedETS": "._expansion_sequence",
    "GARCH": "._expansion_sequence", "ProphetForecaster": "._expansion_sequence",
    "MoEBlock": "._expansion_sequence", "Sequential": "._expansion_sequence",
    # _expansion_trees
    "DecisionTreeClassifier": "._expansion_trees", "DecisionTreeRegressor": "._expansion_trees",
    "BaggingClassifier": "._expansion_trees", "BaggingRegressor": "._expansion_trees",
    "AdaBoostClassifier": "._expansion_trees", "AdaBoostRegressor": "._expansion_trees",
    "DARTRegressor": "._expansion_trees", "DARTClassifier": "._expansion_trees",
    "RandomTreesEmbedding": "._expansion_trees", "VotingClassifier": "._expansion_trees",
    "VotingRegressor": "._expansion_trees", "StackingClassifier": "._expansion_trees",
    "StackingRegressor": "._expansion_trees", "MultiOutputClassifier": "._expansion_trees",
    "MultiOutputRegressor": "._expansion_trees", "OneVsRestClassifier": "._expansion_trees",
    "CalibratedClassifierCV": "._expansion_trees", "TreeExplainer": "._expansion_trees",
    "KernelExplainer": "._expansion_trees", "PermutationExplainer": "._expansion_trees",
    # _expansion_cnn
    "Conv2d": "._expansion_cnn", "Conv1d": "._expansion_cnn", "MaxPool2d": "._expansion_cnn",
    "AvgPool2d": "._expansion_cnn", "MaxPool1d": "._expansion_cnn", "AvgPool1d": "._expansion_cnn",
    "CNNClassifier": "._expansion_cnn", "BatchNorm2d": "._expansion_cnn", "BatchNorm1d": "._expansion_cnn",
    "Dropout2d": "._expansion_cnn", "AdaptiveAvgPool2d": "._expansion_cnn", "AdaptiveMaxPool2d": "._expansion_cnn",
    "BasicBlock": "._expansion_cnn", "GCNConv": "._expansion_cnn", "SAGEConv": "._expansion_cnn",
    # _expansion_ann
    "IVFPQIndex": "._expansion_ann", "TSNE": "._expansion_ann", "CagraIndex": "._expansion_ann",
    "IVFSQIndex": "._expansion_ann", "IVFRaBitQIndex": "._expansion_ann", "refine": "._expansion_ann",
    # _expansion_metrics
}

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

__all__.extend(name for name, where in _LAZY_NAMES.items()  # glue: the doors' names, in door order
               if where is not None and where.startswith("._expansion_"))

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
    if name in _LAZY_NAMES:
        where = _LAZY_NAMES[name]
        if where is None:   # a submodule named in __all__
            value = _importlib.import_module("." + name, __name__)
        else:
            value = getattr(_importlib.import_module(where, __name__), name)
        globals()[name] = value
        return value
    if name in _families.EAGER_BEFORE:
        # A package module bound as an attribute by the eager imports before
        # this lane (`mojolearn._gp_impl`): imported, and connected, on touch.
        return _importlib.import_module("." + name, __name__)
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


# The native Array operations share the base binding's core helpers.
from ._array import connect_helpers as _connect_array_helpers
from ._buffer import _native_optional as _array_helper
_connect_array_helpers(_array_helper)
del _connect_array_helpers, _array_helper

# Complete process backend metadata is native; it reads the loaded units and
# re-reads them when a family loads one (_native_backend_public.refresh).
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

# Last: record the core, install the first-touch finder and connect the
# core's classes. Every later package import connects its own classes.
import sys as _families_sys
_families.install(_families_sys.modules[__name__])
del _families_sys
