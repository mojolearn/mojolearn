# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""scikit-learn's module layout over mojolearn's existing modules.

`from sklearn.linear_model import Lasso` becomes `from mojolearn.linear_model
import Lasso`. These modules predate most of the estimators scikit-learn keeps
in them, which are defined in the expansion modules and bound on the package.
Each module below gets a PEP 562 `__getattr__` that resolves exactly the names
listed for it from the package, at first use, so importing the module does
not import the package's other families early (and cannot form an import
cycle with them). Names a module defines itself never reach `__getattr__`.
The modules scikit-learn has and mojolearn had not (naive_bayes, tree, ...)
are their own files that import their classes directly.
"""

import importlib

#: module -> the public mojolearn names scikit-learn keeps in that module
#: (scikit-learn 1.9), which the module does not define itself.
LAYOUT = {
    "cluster": ("AffinityPropagation", "AgglomerativeClustering", "BisectingKMeans", "DBSCAN", "HDBSCAN",
                "MeanShift", "MiniBatchKMeans", "OPTICS", "SpectralClustering"),
    "decomposition": ("DictionaryLearning", "FactorAnalysis", "FastICA", "IncrementalPCA", "KernelPCA",
                      "LatentDirichletAllocation", "MiniBatchDictionaryLearning", "MiniBatchSparsePCA", "NMF",
                      "SparseCoder", "SparsePCA"),
    "ensemble": ("AdaBoostClassifier", "AdaBoostRegressor", "BaggingClassifier", "BaggingRegressor",
                 "ExtraTreesClassifier", "ExtraTreesRegressor", "IsolationForest", "RandomForestClassifier",
                 "RandomForestRegressor", "RandomTreesEmbedding", "StackingClassifier", "StackingRegressor",
                 "VotingClassifier", "VotingRegressor"),
    "linear_model": ("ARDRegression", "BayesianRidge", "ElasticNet", "ElasticNetCV", "GammaRegressor",
                     "HuberRegressor", "Lars", "Lasso", "LassoCV", "LassoLars", "LogisticRegressionCV",
                     "PassiveAggressiveClassifier", "PassiveAggressiveRegressor", "Perceptron", "PoissonRegressor",
                     "QuantileRegressor", "RidgeCV", "RidgeClassifier", "SGDClassifier", "SGDOneClassSVM",
                     "SGDRegressor", "TweedieRegressor"),
    "manifold": ("ClassicalMDS", "Isomap", "LocallyLinearEmbedding", "MDS", "TSNE"),
    "mixture": ("BayesianGaussianMixture",),
    "neighbors": ("KernelDensity", "LocalOutlierFactor", "NearestCentroid"),
    "preprocessing": ("Binarizer", "KBinsDiscretizer", "LabelBinarizer", "LabelEncoder", "MaxAbsScaler",
                      "MultiLabelBinarizer", "Normalizer", "OneHotEncoder", "OrdinalEncoder", "PolynomialFeatures",
                      "PowerTransformer", "QuantileTransformer", "RobustScaler", "SplineTransformer",
                      "TargetEncoder"),
    "svm": ("OneClassSVM",),
}


def lazy_exports(module):
    """The module `__getattr__` for `module` (a key of `LAYOUT`)."""
    names = frozenset(LAYOUT[module])

    def __getattr__(name):
        if name in names:
            return getattr(importlib.import_module("mojolearn"), name)
        raise AttributeError(f"module 'mojolearn.{module}' has no attribute {name!r}")

    return __getattr__
