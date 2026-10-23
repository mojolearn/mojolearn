# SPDX-License-Identifier: Apache-2.0
"""Python surface for the supported dense, Euclidean UMAP slice."""

from . import _portable_math as math
import operator

from . import _backend, _native_archive_bridge, _native_family_bridge

from ._mode import NumericModeMixin
from .linear_model import _round_f32

#: `np.finfo(np.float32).max`, as a literal: the largest finite float32.
_F32_MAX = 3.4028234663852886e+38

#: The saved-model format tag (`UMAP.save`, lane/inference-forecast-umap-pca).
_UMAP_FORMAT = "mojolearn-umap-1"

# The compiled saved-model owner, connected once at import: `save` and
# `load` are one native call each (bindings/archive_schema_native.mojo
# writes the schema registered after the class; host_archive_native.mojo
# `host_archive_umap` builds the loaded model).
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()
# The compiled family owner, connected once at import: `fit`,
# `fit_transform` and `transform` are one native call each
# (bindings/mani_umap_native.mojo).
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()


def _integer(value, name, minimum, maximum=(1 << 63) - 1):
    # A bool is an int to `operator.index` and is refused by name here; the
    # NumPy bool scalar is caught by its type name so a caller who still
    # passes one gets the same answer (DEVIATION 2370).
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise ValueError(f"UMAP {name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"UMAP {name} must be an integer") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"UMAP {name} must be in [{minimum}, {maximum}]")
    return result


#: scikit-learn / umap-learn metric names -> the kNN's cuVS DistanceType
#: (-1 is the legacy euclidean sentinel, so a euclidean fit keeps its bits).
_METRICS = {
    "euclidean": -1, "l2": -1,
    "sqeuclidean": 0,
    "cosine": 2,
    "manhattan": 3, "l1": 3, "cityblock": 3, "taxicab": 3,
    "chebyshev": 7, "linf": 7, "linfinity": 7, "infinity": 7,
    "minkowski": 9,
}


def _scalar(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"UMAP {name} must be a finite float32 value") from exc
    if not math.isfinite(result) or abs(result) > _F32_MAX:
        raise ValueError(f"UMAP {name} must be a finite float32 value")
    # `float(np.float32(result))`: one round-to-nearest-even to binary32.
    return _round_f32(result)


# Maintainer notes (moved from the docstring):
# - Option parity (lane/algos-decomp, 2026-09-27): n_components 2 and 3 keep
#   their bits; the others run the run-time-dimension optimizer (DEVIATION
#   5322); local_connectivity uses umap-learn's rho interpolation (DEVIATION
#   5323); supervised fit (DEVIATION 5324).
# - Spectral initialization uses at most 20 Lanczos basis vectors for 2D/3D.
# - transform has its own qualification requirements; fit certificates do not
#   certify transform or reference RNG bits.
# - Batch independence of transform: lane/umap-batch-fix (2026-09-16), see
#   umap/transform.mojo and the note above `save`.
# - embedding_ and returned embeddings are `_array.Array`s (DEVIATION 2370;
#   they were ndarrays). The private training copies are plain copies rather
#   than read-only views: an Array has no `setflags`, and nothing outside this
#   instance holds them.
class UMAP(NumericModeMixin):
    """Uniform Manifold Approximation and Projection (UMAP) embedding.

    Builds a fuzzy nearest-neighbor graph of the training rows with exact
    neighbor search, then optimizes a low-dimensional layout. Input is a dense
    array converted to C-order float32. The neighbor graph is stored in CSR
    form, so memory grows linearly with ``n_samples`` for a bounded
    ``n_neighbors``; the exact neighbor search still compares every pair of
    rows. Parameters follow umap-learn; defaults that differ are
    ``n_epochs=None`` (200 epochs) and ``random_state=0``.

    At least ``2 * n_components + 4`` samples are needed for the spectral
    initialization.

    Parameters
    ----------
    n_neighbors : int, default 15
        Size of the local neighborhood; at least 2 and at most ``n_samples``.
    n_components : int, default 2
        Dimension of the embedding, 1 to 32.
    min_dist : float, default 0.1
        Minimum distance between embedded points; ``0 <= min_dist <= spread``.
    spread : float, default 1.0
        Scale of the embedded points; must be positive.
    n_epochs : int or None, default None
        Number of optimization epochs; None uses 200.
    random_state : int, default 0
        Seed for the optimizer and random initialization; a nonnegative
        integer.
    set_op_mix_ratio : float, default 1.0
        Mix between fuzzy union (1.0) and fuzzy intersection (0.0), in [0, 1].
    local_connectivity : float, default 1.0
        Number of nearest neighbors assumed locally connected; any value
        ``>= 0`` (umap-learn's interpolation for fractional values).
    metric : str, default "euclidean"
        Input distance: ``"euclidean"`` (``"l2"``), ``"sqeuclidean"``,
        ``"cosine"``, ``"manhattan"`` (``"l1"``, ``"cityblock"``,
        ``"taxicab"``), ``"chebyshev"`` (``"linf"``, ``"linfinity"``,
        ``"infinity"``) or ``"minkowski"``. A callable or any other name
        raises ``ValueError``.
    init : {"spectral", "random", "pca"} or array, default "spectral"
        Initial layout. ``"random"`` draws uniform(-10, 10); ``"pca"`` is
        umap-learn's scaled PCA plus N(0, 1e-4) noise; an array must have
        shape ``(n_samples, n_components)`` and be finite.
    learning_rate : float, default 1.0
        Initial learning rate; must be positive.
    repulsion_strength : float, default 1.0
        Weight of negative samples; must be nonnegative.
    negative_sample_rate : int, default 5
        Negative samples per positive sample.
    a, b : float or None, default None
        Curve parameters of the low-dimensional similarity. Give both or
        neither; when None they are derived from ``min_dist`` and ``spread``.
    metric_kwds : dict or None, default None
        Extra metric arguments. Only ``{"p": p}`` for ``"minkowski"`` (``p > 0``,
        default 2.0); any other key raises ``ValueError``.
    target_n_neighbors : int, default -1
        Neighbors used for the target graph in supervised fits; -1 uses
        ``n_neighbors``, otherwise at least 2.
    target_metric : {"categorical", "l2", "euclidean"}, default "categorical"
        Distance on ``y`` for supervised fits. With ``"categorical"``, the
        label -1 means unknown. Other values raise ``NotImplementedError``.
    target_weight : float, default 0.5
        Weight of the target in [0, 1]; 0 ignores the data, 1 ignores ``y``.
    densmap : bool, default False
        densMAP is not supported; True raises ``NotImplementedError``.
    output_metric : str, default "euclidean"
        Only ``"euclidean"`` (or ``"l2"``) is supported; any other value raises
        ``NotImplementedError``.

    Attributes
    ----------
    embedding_ : Array of shape (n_samples, n_components)
        The float32 embedding of the training data.
    n_features_in_ : int
        Number of features seen during ``fit``.
    input_copied_ : bool
        Whether ``X`` had to be copied to C-order float32.

    Notes
    -----
    ``transform`` keeps private copies of the training data and embedding,
    adding ``O(n_samples * (n_features + n_components))`` memory. It finds
    each new row's neighbors among the training rows, initializes it from
    their weighted embedding and refines it against the frozen training
    embedding for 100 epochs (``max(1, n_epochs // 3)`` when ``n_epochs`` is
    set) at ``learning_rate / 4``. The result for a row does not depend on
    the other rows in the call.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.UMAP(n_neighbors=15, n_components=2).fit(X)
    >>> model.embedding_.shape
    (1000, 2)
    >>> Y = model.transform(X[:10])
    """

    _BINDING = "_mojolearn_metrics"

    def __init__(self, n_neighbors=15, n_components=2, *, min_dist=0.1,
                 spread=1.0, n_epochs=None, random_state=0,
                 set_op_mix_ratio=1.0, local_connectivity=1.0,
                 metric="euclidean", init="spectral", learning_rate=1.0,
                 repulsion_strength=1.0, negative_sample_rate=5, a=None, b=None,
                 metric_kwds=None, target_n_neighbors=-1, target_metric="categorical",
                 target_weight=0.5, densmap=False, output_metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.spread = spread
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.metric = metric
        self.init = init
        self.learning_rate = learning_rate
        self.repulsion_strength = repulsion_strength
        self.negative_sample_rate = negative_sample_rate
        self.a = a
        self.b = b
        self.metric_kwds = metric_kwds
        self.target_n_neighbors = target_n_neighbors
        self.target_metric = target_metric
        self.target_weight = target_weight
        self.densmap = densmap
        self.output_metric = output_metric
        self._parameters()

    def _parameters(self):
        neighbors = _integer(self.n_neighbors, "n_neighbors", 2)
        components = _integer(self.n_components, "n_components", 1, 32)
        epochs = 0 if self.n_epochs is None else _integer(
            self.n_epochs, "n_epochs", 1)
        seed = _integer(self.random_state, "random_state", 0)
        min_dist = _scalar(self.min_dist, "min_dist")
        spread = _scalar(self.spread, "spread")
        mix = _scalar(self.set_op_mix_ratio, "set_op_mix_ratio")
        connectivity = _scalar(self.local_connectivity, "local_connectivity")
        rate = _scalar(self.learning_rate, "learning_rate")
        repulsion = _scalar(self.repulsion_strength, "repulsion_strength")
        negatives = _integer(self.negative_sample_rate, "negative_sample_rate",
                             0, (1 << 31) - 1)
        if rate <= 0:
            raise ValueError("UMAP learning_rate must be positive")
        if repulsion < 0:
            raise ValueError("UMAP repulsion_strength must be nonnegative")
        if not 0 <= min_dist <= spread or spread <= 0:
            raise ValueError("UMAP requires 0 <= min_dist <= spread and spread > 0")
        if not 0 <= mix <= 1:
            raise ValueError("UMAP set_op_mix_ratio must be in [0, 1]")
        if connectivity < 0:
            raise ValueError("UMAP local_connectivity must be >= 0")
        self._extras()
        return [neighbors, components, epochs, min_dist, spread, mix,
                connectivity, seed, rate, repulsion, negatives]

    def _extras(self):
        """(metric code, metric_arg, a, b) of the option-parity controls
        (lane/algos-decomp, 2026-09-27), validated; the legacy defaults are
        (-1, 2.0, 0.0, 0.0)."""
        if getattr(self, "densmap", False):
            raise NotImplementedError(
                "UMAP densmap=True is not implemented: the density-augmented "
                "objective (umap-learn's densMAP) is a different optimizer")
        if getattr(self, "output_metric", "euclidean") not in ("euclidean", "l2"):
            raise NotImplementedError(
                f"UMAP output_metric={self.output_metric!r} is not implemented; the layout "
                "optimizer's gradient is the euclidean one")
        if not isinstance(self.metric, str) or self.metric not in _METRICS:
            raise ValueError(
                f"UMAP metric={self.metric!r} is not supported; the kNN computes "
                f"{sorted(_METRICS)} (a callable, and every other name, is refused by name)")
        code = _METRICS[self.metric]
        kw = dict(getattr(self, "metric_kwds", None) or {})
        p = 2.0
        if code == 9:
            p = _scalar(kw.pop("p", 2.0), "metric_kwds['p']")
            if not p > 0:
                raise ValueError("UMAP minkowski p must be positive")
        if kw:
            raise ValueError(f"UMAP metric_kwds {sorted(kw)} are not used by metric={self.metric!r}")
        a, b = getattr(self, "a", None), getattr(self, "b", None)
        if (a is None) != (b is None):
            raise ValueError("UMAP a and b must be given together")
        if a is None:
            a = b = 0.0
        else:
            a, b = _scalar(a, "a"), _scalar(b, "b")
            if not (a > 0 and b > 0):
                raise ValueError("UMAP a and b must be positive")
        init = self.init
        if isinstance(init, str):
            if init not in ("spectral", "random", "pca"):
                raise ValueError(f"UMAP init={init!r} must be 'spectral', 'random', 'pca' or an array")
        return (code, p, a, b)

    def fit(self, X, y=None):
        """Fit the embedding to ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data; converted to C-order float32 and must be finite.
        y : array-like of shape (n_samples,) or (n_samples, n_targets), optional
            Target for supervised UMAP. Categorical labels (one per row, -1
            for unknown) or, with ``target_metric="l2"``, finite continuous
            values.

        Returns
        -------
        self : UMAP
            The fitted estimator.

        Raises
        ------
        ValueError
            If ``X`` is not finite, ``n_neighbors`` exceeds ``n_samples``,
            there are too few samples for spectral initialization, or a
            parameter is out of range.
        NotImplementedError
            For ``densmap=True``, a non-euclidean ``output_metric`` or an
            unsupported ``target_metric``.
        """
        return _NATIVE(self, "family_mani_umap_fit", (self._bind(), X, y), _NATIVE_CONTEXT)

    def fit_transform(self, X, y=None):
        """Fit to ``X`` and return the training embedding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like, optional
            Target for supervised UMAP (see ``fit``).

        Returns
        -------
        embedding : Array of shape (n_samples, n_components)
            The float32 embedding, sharing memory with ``embedding_``.
        """
        return _NATIVE(self, "family_mani_umap_fit_transform", (self._bind(), X, y), _NATIVE_CONTEXT)

    def transform(self, X):
        """Embed new rows in the fitted coordinate system without refitting.

        Each row is embedded independently: a batch of rows returns the same
        values as embedding the rows one at a time. Passing the exact training
        data returns a copy of the training embedding. Later edits of
        ``embedding_`` or of the original ``X`` do not change the model.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            New rows; must be finite and have the training feature count.

        Returns
        -------
        embedding : Array of shape (n_queries, n_components)
            The float32 embedding of the new rows.

        Raises
        ------
        ValueError
            If the model is not fitted, the parameters or numeric mode changed
            since ``fit`` (refit first), the feature count differs, or ``X``
            is not finite.
        """
        return _NATIVE(self, "family_mani_umap_transform", (self._bind(), X), _NATIVE_CONTEXT)

    # Saved layout (lane/inference-forecast-umap-pca, 2026-09-15): training and
    # embedding (float32), `meta` <i8 [n_neighbors, n_components, n_epochs,
    # random_state, negative_sample_rate, n_epochs_is_none, input_copied],
    # `controls` <f4 [min_dist, spread, set_op_mix_ratio, local_connectivity,
    # learning_rate, repulsion_strength], `extras`, `metric`, `init`,
    # `numeric_mode`. On a CPU-only install a loaded model runs through the
    # metrics host binding.
    # Transform does not depend on the query batch. Until lane/umap-batch-fix
    # (2026-09-16) it did, four ways: the sigma floor was a mean over every
    # query's neighbor distances, each edge weight was scaled by the maximum
    # over the batch, the negative-sample draws were keyed by a row's position
    # in the batch, and with n_epochs unset the epoch count fell from 100 to 30
    # above ten thousand queries. All four read one row now (umap/transform.mojo;
    # measured closed on the CPU host route and on Metal), so a loaded model on
    # a CPU answers the GPU's bytes for a row whatever else is asked with it.
    def save(self, path):
        """Save the fitted model to ``path`` as an ``.npz`` file.

        The file holds the training rows, the embedding, the parameters and
        the numeric mode. A model loaded with ``UMAP.load`` (or
        ``mojolearn.host_model``) can ``transform`` new rows, including on a
        CPU-only install, but cannot refit.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Raises
        ------
        ValueError
            If the model is not fitted.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    def load(cls, path):
        """Load a model written by ``save``.

        The loaded model transforms new rows against the saved training rows
        and embedding; it does not refit.

        Parameters
        ----------
        path : str or path-like
            File written by ``UMAP.save``.

        Returns
        -------
        model : UMAP
            The restored model, with ``embedding_``, ``n_features_in_`` and
            ``input_copied_`` set.

        Raises
        ------
        ValueError
            If the file is not a UMAP model or its contents are inconsistent.
        """
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)


def _config(i):
    return ("_transform_config", i)


_native_archive_bridge.register(UMAP, _UMAP_FORMAT, (
    ("need", "_transform_training", "ValueError", "UMAP save requires a successful fit"),
    ("format", "format", _UMAP_FORMAT),
    ("estimator", "estimator"),
    ("str", "numeric_mode", "_transform_mode"),
    ("str", "metric", "metric"),
    ("init_name", "init", "init", "array"),
    # option parity (lane/algos-decomp): metric code, minkowski p, a, b
    ("f8", "extras", (("float", _config(11)), ("float", _config(12)),
                      ("float", _config(13)), ("float", _config(14)))),
    ("array", "training", "_transform_training"),
    ("array", "embedding", "_transform_embedding"),
    ("i8", "meta", (("int", _config(0)), ("int", _config(1)), ("int", _config(2)),
                    ("int", _config(7)), ("int", _config(10)), ("noneflag", "n_epochs"),
                    ("bool_or", "input_copied_", False))),
    ("f4", "controls", (("float", _config(3)), ("float", _config(4)), ("float", _config(5)),
                        ("float", _config(6)), ("float", _config(8)), ("float", _config(9)))),
), host_archive_umap_metrics=dict(_METRICS))

# The owner's fixed tables: the metric codes, and the binding resolver a
# 'random' / 'pca' start uses for the decomp kit (binding acquisition only).
_native_family_bridge.tables(family_mani_umap_metrics=dict(_METRICS), family_mani_binding=_backend.binding)
