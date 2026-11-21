# SPDX-License-Identifier: Apache-2.0
"""What `model.digest_` hashes, per class, and its one native connection.

`digest_` is the hash of a fitted model's state (bindings/fitted_digest_door.mojo
has the walk, core/fitted_digest.mojo the format). The compiled walk reads the
tables below; `tools/gen_digest_fields.py` writes docs/api/DIGEST_FIELDS.md
from the same tables and the save schemas, so the table and the code cannot
disagree.

The field source of a class, first match in its ancestry:

1. `REFUSED`: no digest yet; the read raises AttributeError with the reason.
2. A connected save schema (`_native_archive_bridge.register`): the members
   `save` writes.
3. Otherwise: every public fitted attribute (an instance field whose name ends
   in "_" and does not start with "_", value not None, name not in
   `NOT_STATE`), plus the private fields in `EXTRA`.

This module holds declarations and import-time glue only: nothing here runs
when a model is fitted, and the read itself is one native call.
"""
#: class name -> the private instance fields that are fitted state (they
#: decide the model's answers and no public attribute carries them).
#: A name the model does not hold is passed over, so a base class may list
#: the union of its subclasses' fields. Listed here: state `fit` writes that
#: `predict` / `transform` reads. Not listed: device handles, pins, lifecycle
#: flags and caches written by a later call (they are not fitted state, and a
#: handle differs per device).
EXTRA_BY_NAME = {
    # preprocessing: encoders, imputers, discretizers (bindings/prep_encode_native_door.mojo)
    "SimpleImputer": ("_fill", "_keep", "_indicator"),
    "KBinsDiscretizer": ("_edges", "_ne", "_stride"),
    "OrdinalEncoder": ("_missing", "_infrequent", "_grouping"),
    "OneHotEncoder": ("_infrequent", "_grouping", "_drop_after"),
    "SplineTransformer": ("_knots", "_lo", "_hi", "_nk", "_k"),
    "TargetEncoder": ("_T", "_cmax", "_enc", "_meta"),
    "_SelectorMixin": ("_mask",),
    # the imputer's generator word and posterior step advance with each sampled transform:
    # they decide the next transform's answers, so they are state.
    "IterativeImputer": ("_keep", "_indicator", "_bounds_k", "_posterior", "_rng", "_post_step"),
    # stateless-fit transformers (bindings/prep_stateless_native_door.mojo)
    "PolynomialFeatures": ("_poly_index", "_poly_start"),
    "QuantileTransformer": ("_q",),
    "PowerTransformer": ("_pt_anchor", "_pt_anchor_kind", "_mean", "_scale", "_method"),
    # label encoders (bindings/prep_labels_native_door.mojo)
    "LabelEncoder": ("_classes", "_cats"),
    "LabelBinarizer": ("_classes", "_cats"),
    "MultiLabelBinarizer": ("_classes", "_cats", "_given"),
    # naive Bayes and discriminant analysis (bindings/prep_classify_native_door.mojo)
    "GaussianNB": ("_const", "_raw_var"),
    "_DiscreteNB": ("_w", "_bias", "_cc", "_flp", "_cmax"),
    "LinearDiscriminantAnalysis": ("_coef", "_inter", "_rank", "_scal_full", "_max_components"),
    "QuadraticDiscriminantAnalysis": ("_rot", "_logc"),
    # linear models: the centering of the fit (providers/python/linear_fit_boundary_unpack.mojo)
    "LinearRegression": ("_x_mean", "_y_mean"),
    "Ridge": ("_x_mean", "_y_mean"),
    # the stacked weights predict reads (bindings/qn_boundary_pack.mojo, bindings/svc_linear_boundary_pack.mojo)
    "QNRegressor": ("_w",),
    "_LinearSVMBase": ("_w",),
    # the link the GLM predict applies (bindings/x_linear_regressors_pack.mojo)
    "_GLMBase": ("_link",),
    # kernel methods: the kernel's parameters and seed as fitted (bindings/kernel_methods_boundary_pack.mojo)
    "KernelRidge": ("_kernel_params", "_squeeze"),
    "Nystroem": ("_kernel_params",),
    "RBFSampler": ("_params",),
}

#: Public names (ending in "_") that record how a call ran, not what was
#: fitted: whether the input was copied on the way in, which shards or which
#: query tile the last call used. They differ with the input's container or
#: the machine while the fitted model is the same, so no class hashes them.
NOT_STATE = ("input_copied_", "fortran_copied_", "last_shards_", "used_query_tile_", "n_candidate_distances_")

#: class name -> why it has no digest yet, in words a user reads.
REFUSED_BY_NAME = {
    "_BaggingBase": "its fitted members are held as one forest by the trees owner, which the digest does not read yet",
    "_RecurrentBase": "its fitted weights are held by the recurrent owner (state_dict), which the digest does not read yet",
    "Sequential": "its layers hold the fitted weights, and the digest does not walk layers yet",
    "CNNClassifier": "its layers hold the fitted weights, and the digest does not walk layers yet",
}

#: The tables the compiled walk reads, keyed by class (filled by `declare`).
EXTRA = {}
REFUSED = {}

#: The one connection of the `digest_` property: the property object, the
#: compiled entry and its context. Filled in place by `connect`; `_mode` puts
#: this dictionary in its metadata context, where
#: bindings/fitted_digest_property_pack.mojo reads it.
CONNECTION = {"property": None, "entry": None, "context": None}


def declare(cls):
    """Enter `cls` in the class-keyed tables when a table above names it
    (called once per class, at its definition, by the shared estimator
    base)."""
    name = cls.__name__
    if name in EXTRA_BY_NAME:
        EXTRA[cls] = EXTRA_BY_NAME[name]
    if name in REFUSED_BY_NAME:
        REFUSED[cls] = REFUSED_BY_NAME[name]


def connect(base):
    """Connect the compiled digest entry once: `(entry, context)`. `base` is
    the shared estimator base whose `digest_` property this serves. On an
    installation without the saved-model owner the entry raises
    ModuleNotFoundError in words, as `save` does there."""
    if CONNECTION["context"] is not None:
        return CONNECTION["entry"], CONNECTION["context"]
    from ._native_archive_bridge import connect as connect_archive
    entry, context = connect_archive()
    if "archive_unavailable" not in context:
        context.update(digest_base=base, digest_extra=EXTRA, digest_refused=REFUSED, digest_skip=NOT_STATE)
        CONNECTION["entry"] = entry
    CONNECTION["context"] = context
    return entry, context


def read(model):
    """`model.digest_`: one native call."""
    entry = CONNECTION["entry"]
    if entry is None:
        from ._mode import CheckMixin
        entry = connect(CheckMixin)[0]
    return entry(model, "fitted_digest", (), CONNECTION["context"])

