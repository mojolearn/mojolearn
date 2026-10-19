"""Navigation between the guide pages and the API reference.

- Guide pages: an inline-code name that is a documented public name (`KMeans`, `mojolearn.vendor()`,
  `GradientBoosting.load`) becomes a link to its API entry. Guides keep writing plain backticks.
- API class pages: a "See also" box after the class description names related estimators (the classifier
  and regressor of one model, an estimator and its CV variant, a few hand-picked groups) and links the
  family's user-guide page.
"""
# cpu-route: documentation build tooling, never imported by the mojolearn runtime.
import pathlib
import re
import sys
import types

from docutils import nodes
from sphinx import addnodes

import mojolearn

_DOCS = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DOCS / "tools"))
from gen_api import family_of  # noqa: E402


def _excluded():
    lines = (_DOCS / "api_exclude.txt").read_text().splitlines()
    return {line.split("\t", 1)[0].strip() for line in lines if line.strip() and not line.startswith("#")}


_EXCLUDED = _excluded()
_PUBLIC = {n for n in mojolearn.__all__ if n != "__version__" and n not in _EXCLUDED}
_CLASSES = {n for n in _PUBLIC if isinstance(getattr(mojolearn, n, None), type)}
_MODULES = {n for n in _PUBLIC if isinstance(getattr(mojolearn, n, None), types.ModuleType)}

# ---- guide pages -> API reference ------------------------------------------------------------------

_NAME = re.compile(r"^(mojolearn\.)?([A-Za-z_]\w*)(?:\.([A-Za-z_]\w*))?(\(\))?$")
_NO_LINK_INSIDE = (nodes.literal_block, nodes.reference, addnodes.pending_xref, nodes.title)


def _target(text):
    m = _NAME.match(text.strip())
    if not m:
        return None
    prefixed, base, member, called = m.groups()
    if base not in _PUBLIC:
        return None
    if base in _MODULES and not prefixed:
        return None  # a bare `metrics` or `models` in prose is a word, not the module
    if base[:1].islower() and not (prefixed or called or member):
        return None  # a bare `numeric_mode` is usually the parameter; `numeric_mode()` is the function
    if member:
        if member.startswith("_") or f"{base}.{member}" in _EXCLUDED:
            return None
        if not hasattr(getattr(mojolearn, base), member):
            return None
        return f"mojolearn.{base}.{member}"
    return f"mojolearn.{base}"


def _inside(node, kinds):
    parent = node.parent
    while parent is not None:
        if isinstance(parent, kinds):
            return True
        parent = parent.parent
    return False


def link_api_names(app, doctree):
    docname = app.env.docname
    if docname.startswith("api/"):
        return
    for literal in list(doctree.findall(nodes.literal)):
        if _inside(literal, _NO_LINK_INSIDE):
            continue
        target = _target(literal.astext())
        if target is None:
            continue
        ref = addnodes.pending_xref("", refdomain="py", reftype="obj", reftarget=target,
                                    refexplicit=True, refwarn=False, refdoc=docname)
        ref["py:module"] = None
        ref["py:class"] = None
        literal.replace_self(ref)
        ref.append(literal)


# ---- API class pages: "See also" ---------------------------------------------------------------------

# Hand-picked groups of estimators that answer the same question in different ways.
_GROUPS = [
    ("RandomForestClassifier", "ExtraTreesClassifier", "GradientBoostingClassifier", "DecisionTreeClassifier"),
    ("RandomForestRegressor", "ExtraTreesRegressor", "GradientBoostingRegressor", "DecisionTreeRegressor"),
    ("KMeans", "MiniBatchKMeans", "BisectingKMeans"),
    ("DBSCAN", "HDBSCAN", "OPTICS"),
    ("GaussianMixture", "BayesianGaussianMixture"),
    ("PCA", "IncrementalPCA", "KernelPCA", "TruncatedSVD", "SparsePCA"),
    ("UMAP", "TSNE", "Isomap", "SpectralEmbedding"),
    ("Lasso", "LassoCV", "LassoLars", "ElasticNet"),
    ("SVC", "SVR", "LinearSVC", "LinearSVR"),
    ("NearestNeighbors", "IVFIndex", "IVFPQIndex", "IVFSQIndex", "IVFRaBitQIndex", "CagraIndex"),
    ("StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler"),
    ("SimpleImputer", "IterativeImputer", "KNNImputer"),
    ("IsolationForest", "LocalOutlierFactor", "OneClassSVM", "EllipticEnvelope"),
    ("GaussianProcessRegressor", "GaussianProcessClassifier"),
    ("ARIMA", "AutoARIMA"),
    ("ETS", "ExponentialSmoothing"),
    ("Theta", "AutoTheta", "OptimizedTheta", "DynamicTheta", "DynamicOptimizedTheta"),
    ("CrostonClassic", "CrostonOptimized", "CrostonSBA"),
]

# API family page -> user-guide page.
_GUIDES = {
    "boosting": "trees", "trees": "trees", "linear": "linear", "clustering": "clustering",
    "decomposition": "decomposition", "neighbors": "neighbors", "gaussian_process": "gaussian_process",
    "time_series": "time_series", "preprocessing": "preprocessing", "neural": "neural_networks",
    "language_models": "language_models", "autograd": "autograd",
}
_ESTIMATOR_FAMILIES = {"boosting", "trees", "linear", "clustering", "decomposition", "neighbors",
                       "gaussian_process", "time_series", "preprocessing", "svm_kernels", "density"}


def _stem(name):
    return re.sub(r"(Classifier|Regressor)$", "", re.sub(r"CV$", "", name))


def _related(name):
    out = [n for n in sorted(_CLASSES) if n != name and _stem(n) == _stem(name)]
    for group in _GROUPS:
        if name in group:
            out += [n for n in group if n != name and n in _CLASSES and n not in out]
    return out[:6]


def see_also(app, what, name, obj, options, lines):
    if what != "class" or not name.startswith("mojolearn."):
        return
    short = name[len("mojolearn."):]
    if short not in _CLASSES:
        return  # module pages re-document classes; only the class's own page gets the box
    related = _related(short)
    family = family_of(short, obj)
    guide = _GUIDES.get(family)
    items = []
    if related:
        items.append(", ".join(f":class:`~mojolearn.{n}`" for n in related))
    if guide:
        items.append(f"User guide: :doc:`/user_guide/{guide}`")
    if family in _ESTIMATOR_FAMILIES:
        items.append(":doc:`/user_guide/choosing_an_estimator`")
    if not items:
        return
    lines += ["", ".. seealso::", ""]
    for item in items:
        lines += [f"   {item}", ""]


def setup(app):
    app.connect("doctree-read", link_api_names)
    app.connect("autodoc-process-docstring", see_also)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
