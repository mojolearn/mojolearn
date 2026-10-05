# SPDX-License-Identifier: Apache-2.0
"""Python surface for the supported dense, Euclidean UMAP slice."""

from . import _portable_math as math
import operator

from . import _backend
from ._buffer import addr, addr_ro, all_finite, as_f32_c, empty
from ._mode import NumericModeMixin
from .linear_model import _round_f32

#: `np.finfo(np.float32).max`, as a literal: the largest finite float32.
_F32_MAX = 3.4028234663852886e+38

#: The saved-model format tag (`UMAP.save`, lane/inference-forecast-umap-pca).
_UMAP_FORMAT = "mojolearn-umap-1"


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


class UMAP(NumericModeMixin):
    """UMAP embedding with exact Euclidean neighbors and spectral init.

    Supports fit/fit_transform/transform. OPTION PARITY (lane/algos-decomp,
    2026-09-27): n_components 1 to 32 (2 and 3 keep their bits; the others
    run the run-time-dimension optimizer, DEVIATION 5322); any
    local_connectivity >= 0 (umap-learn's rho interpolation, DEVIATION 5323);
    metric euclidean / l2, sqeuclidean, cosine, manhattan (l1, cityblock,
    taxicab), chebyshev (linf, infinity), minkowski (metric_kwds={'p': p});
    init 'spectral', 'random' (uniform(-10, 10)), 'pca' (umap-learn's scaled
    PCA plus N(0, 1e-4) noise, through the decomp cells) or an
    (n_samples, n_components) array; a and b given directly; fit(X, y)
    supervised with target_metric 'categorical' (-1 unknown) or 'l2' /
    'euclidean', target_weight and target_n_neighbors (DEVIATION 5324).
    REFUSED BY NAME: densmap=True, output_metric other than euclidean, a
    callable metric and every other metric name, other target metrics.
    The source fit path stores a CSR graph in O(n_samples*n_neighbors)
    space, linear in samples when n_neighbors is bounded; exact neighbor
    search still performs quadratic pair comparisons. Spectral initialization
    uses at most 20 Lanczos basis vectors for 2D/3D output. At least
    2*n_components+4 samples are required. n_epochs=None uses 200 epochs;
    random_state defaults to 0. Input is converted to C-order float32;
    input_copied_ records whether a copy was needed.

    transform retains private copies of training data and embedding, adding
    O(n_samples*n_features + n_samples*n_components) storage, plus parameters
    and mode from the last successful fit. It computes query-to-training
    neighbors, memberships, weighted initialization and seeded refinement
    against the frozen training embedding. Default refinement is 100 epochs
    at every request size, or max(1, n_epochs//3) when explicitly set;
    learning rate is learning_rate/4; repulsion_strength and
    negative_sample_rate apply to both fit and transform. Query batching
    does NOT change results: a batch of N returns the same bytes as N calls
    of one row (lane/umap-batch-fix, 2026-09-16). This path has its own
    qualification requirements; existing fit certificates do not certify
    transform or reference RNG bits.

    `embedding_` and every returned embedding are `_array.Array`s of float32
    (DEVIATION 2370; they were ndarrays). The private training copies are
    plain copies rather than read-only views: an Array has no
    `setflags`, and nothing outside this instance holds them.
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

    def _initial(self, x, n, nc, seed):
        """The initial layout of a non-spectral init (None for 'spectral'):
        'random' is uniform(-10, 10) on the Philox stream (umap-learn's and
        cuML's range), 'pca' is umap-learn's PCA of the data scaled to max
        |coordinate| 10 plus N(0, 1e-4) noise, an array is used as given. The
        arithmetic is the decomp lane's cells, IDENTICAL on every column."""
        init = self.init
        if isinstance(init, str) and init == "spectral":
            return None
        from ._expansion_decomp import _Kit, _M, _svd_flip_v
        k = _Kit("identical")
        if isinstance(init, str) and init == "random":
            u = k.rand(n, nc, seed, 90, 0)
            return k.ew("adds", k.ew("scale", u, s=20.0), s=-10.0).out()
        if isinstance(init, str):     # 'pca'
            M = _M.from_input(x)
            if nc > min(M.r, M.c):
                raise ValueError("UMAP init='pca' needs n_components <= min(n_samples, n_features)")
            Xc = k.center(M, k.colmean(M))
            if M.r >= M.c:
                _, Vt = k.svd(Xc)
            else:
                # wide data: the SVD of Xc^T gives Xc's left vectors U (its
                # Vt rows); Xc's right vectors are then U^T Xc / S
                S, Ut = k.svd(Xc.T)
                if not all(v > 0 for v in S.s[:nc]):  # glue: zero singular value check over nc components
                    raise ValueError("UMAP init='pca' found a zero singular value")
                Vt = k.mm(Ut.rows(0, nc), Xc)
                Vt = k.ew("div", Vt, S.take_cols(list(range(nc))).T)
            Vt = _svd_flip_v(Vt.rows(0, nc))
            coords = k.mm(Xc, Vt, tb=True)
            # max |coords| in Mojo: the positions of the minimum of -|coords|
            # (x_decomp/moves.mojo argmin_all), an exact negation
            mag = k.ew("abs", coords)
            at = k.argmin_all(k.ew("scale", mag, s=-1.0))
            peak = mag.s[int(at.s[0])] if at.r else 0.0
            if not peak > 0:
                raise ValueError("UMAP init='pca' found a zero spread")
            coords = k.ew("scale", coords, s=10.0 / peak)
            noise = k.ew("scale", k.rand(n, nc, seed, 91, 1), s=1e-4)
            return k.ew("add", coords, noise).out()
        a, _ = as_f32_c(init, ndim=2, name="init")
        if tuple(a.shape) != (n, nc):
            raise ValueError(f"UMAP init array has shape {tuple(a.shape)}; ({n}, {nc}) is needed")
        if not all_finite(a):
            raise ValueError("UMAP init array must be finite")
        return a

    def _target(self, y, n):
        """(kind, dims, codes) of a supervised target: 'categorical' labels
        as float32 codes (sorted unique values, -1 kept as the unknown
        label), or a continuous target ('l2' / 'euclidean') as float32."""
        tm = self.target_metric
        w = _scalar(self.target_weight, "target_weight")
        if not 0 <= w <= 1:
            raise ValueError("UMAP target_weight must be in [0, 1]")
        tk = _integer(self.target_n_neighbors, "target_n_neighbors", -1)
        if tk in (0, 1):
            raise ValueError("UMAP target_n_neighbors must be -1 or >= 2")
        if tm == "categorical":
            vals = list(y) if not hasattr(y, "tolist") else y.tolist()
            if len(vals) != n or any(isinstance(v, (list, tuple)) for v in vals):
                raise ValueError("UMAP categorical y must be one label per row")
            order = sorted({v for v in vals if v != -1}, key=lambda v: (str(type(v)), v))
            code = {v: float(i) for i, v in enumerate(order)}
            codes = [-1.0 if v == -1 else code[v] for v in vals]
            t, _ = as_f32_c(codes, ndim=1, name="y")
            return 1, 1, t, max(tk, 0), w
        if tm in ("l2", "euclidean"):
            t, _ = as_f32_c(y, ndim=None, name="y")
            if t.ndim == 1:
                t = t.reshape((t.shape[0], 1))
            if t.ndim != 2 or t.shape[0] != n or not all_finite(t):
                raise ValueError("UMAP continuous y must be finite with one row per sample")
            return 2, int(t.shape[1]), t, max(tk, 0), w
        raise NotImplementedError(
            f"UMAP target_metric={tm!r} is not implemented; 'categorical' and 'l2' / 'euclidean' are")

    def fit(self, X, y=None):
        config = self._parameters()
        extras = self._extras()
        x, copied = as_f32_c(X, ndim=2, name="X")
        if not all_finite(x):
            raise ValueError("UMAP input coordinates must be finite")
        n, d = x.shape
        if config[0] > n:
            raise ValueError("UMAP n_neighbors exceeds n_samples")
        initial = self._initial(x, n, config[1], config[7])
        if initial is None and n < 2 * config[1] + 4:
            raise ValueError("UMAP has too few samples for spectral initialization")
        target = None if y is None else self._target(y, n)
        embedding = empty((n, config[1]), "<f4")
        # n_samples, n_features, n_neighbors, n_components, n_epochs,
        # min_dist, spread, set_op_mix_ratio, local_connectivity, random_state,
        # learning_rate, repulsion_strength, negative_sample_rate.
        binding = self._bind()
        mode = (self.numeric_mode or _backend.default_mode()).strip().lower()
        if binding.umap_numeric_mode() != {"fast": 0, "identical": 1,
                                           "deterministic": 2}[mode]:
            raise RuntimeError("UMAP binary numeric mode disagrees with requested mode")
        legacy = (initial is None and target is None and extras == (-1, 2.0, 0.0, 0.0)
                  and config[1] in (2, 3))
        if legacy:
            # Preserve the legacy ABI for default controls and existing wheels.
            native_config = config[:8] if config[8:] == [1.0, 1.0, 5] else config
            columns = binding.umap_fit_transform(
                addr_ro(x, name="X"), addr(embedding, name="embedding_"),
                [n, d, *native_config])
        else:
            tkind, tdims, tarr, tk, tw = target if target is not None else (0, 1, None, 0, 0.5)
            columns = binding.umap_fit_transform_ex(
                [addr_ro(x, name="X"), addr(embedding, name="embedding_"),
                 0 if initial is None else addr_ro(initial, name="init"),
                 0 if tarr is None else addr_ro(tarr, name="y")],
                [n, d, *config, *extras, tkind, tdims, tk, tw])
        if columns != config[1] or not all_finite(embedding):
            raise RuntimeError("UMAP returned an invalid embedding")
        # Prepare all retained state before publishing a successful fit. Copies
        # prevent caller edits of X or embedding_ from changing transform's model.
        training = x.copy()
        frozen_embedding = embedding.copy()
        self.embedding_ = embedding
        self.n_features_in_ = d
        self.input_copied_ = copied
        self._transform_training = training
        self._transform_embedding = frozen_embedding
        self._transform_config = tuple(config) + tuple(extras)
        self._transform_mode = mode
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).embedding_

    def transform(self, X):
        """Embed new rows in the last fitted coordinate system without refitting.

        Identical training-input bytes return a copy of the saved embedding.
        Parameter or mode changes require a successful refit. Later edits of
        public embedding_ or the original X do not alter the retained model.

        The answer for a row does NOT depend on the other rows in the same
        call. The sigma floor's mean, the edge-weight scale and the
        negative-sample counter are all per row, and the refinement epoch
        count does not read the request size, so a batch of N returns the
        same bytes as N calls of one row, on every device alike
        (umap/transform.mojo, and lane/umap-batch-fix, 2026-09-16, which
        measured all four couplings closed on the CPU host route and on
        Metal).
        """
        if not hasattr(self, "_transform_training"):
            raise ValueError("UMAP transform requires a successful fit")
        config = self._parameters()
        extras = self._extras()
        mode = (self.numeric_mode or _backend.default_mode()).strip().lower()
        if tuple(config) + tuple(extras) != self._transform_config or mode != self._transform_mode:
            raise ValueError("UMAP parameters or numeric mode changed after fit; refit before transform")
        x, _ = as_f32_c(X, ndim=2, name="X")
        training = self._transform_training
        fitted_embedding = self._transform_embedding
        if x.shape[1] != training.shape[1]:
            raise ValueError("UMAP transform feature count differs from fitted data")
        if not all_finite(x):
            raise ValueError("UMAP transform input coordinates must be finite")
        # `Array.tobytes()` is the C-order bytes, so this byte compare is the
        # one the NumPy-era code made.
        if x.shape == training.shape and x.tobytes() == training.tobytes():
            return fitted_embedding.copy()
        binding = self._bind()
        if binding.umap_numeric_mode() != {"fast": 0, "identical": 1,
                                           "deterministic": 2}[mode]:
            raise RuntimeError("UMAP binary numeric mode disagrees with fitted mode")
        output = empty((x.shape[0], config[1]), "<f4")
        native_config = config[:8] if config[8:] == [1.0, 1.0, 5] else config
        if extras != (-1, 2.0, 0.0, 0.0):
            native_config = [*config, *extras]
        columns = binding.umap_transform(
            [addr_ro(training, name="training"),
             addr_ro(fitted_embedding, name="embedding_"),
             addr_ro(x, name="X"), addr(output, name="output")],
            [training.shape[0], x.shape[0], training.shape[1], *native_config])
        if columns != config[1] or not all_finite(output):
            raise RuntimeError("UMAP transform returned an invalid embedding")
        return output

    def save(self, path):
        """Write the fitted embedding to `path` as an npz: the retained
        training rows and embedding (float32), `meta` `<i8` [n_neighbors,
        n_components, n_epochs, random_state, negative_sample_rate,
        n_epochs_is_none, input_copied], `controls` `<f4` [min_dist, spread,
        set_op_mix_ratio, local_connectivity, learning_rate,
        repulsion_strength], `metric`, `init` and `numeric_mode`
        (lane/inference-forecast-umap-pca, 2026-09-15).

        A loaded model transforms; it does not refit. On a CPU-only install
        that is public inference through the metrics host binding
        (`mojolearn.host_model(path)` or `UMAP.load(path)`).

        THE RESULT DOES NOT DEPEND ON THE QUERY BATCH. Until
        lane/umap-batch-fix (2026-09-16) it did, four ways: the sigma floor
        was a mean over every query's neighbor distances, each edge weight
        was scaled by the maximum over the batch, the negative-sample draws
        were keyed by a row's position in the batch, and with n_epochs unset
        the epoch count fell from 100 to 30 above ten thousand queries. All
        four read one row now, so a loaded model on a CPU answers the GPU's
        bytes for a row whatever else is asked with it, and a row asked
        alone embeds exactly as it does in a batch."""
        if not hasattr(self, "_transform_training"):
            raise ValueError("UMAP save requires a successful fit")
        from . import _serialize
        from ._array import Array
        c = self._transform_config
        arrays = {
            "format": _UMAP_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": self._transform_mode,
            "metric": str(self.metric),
            "init": self.init if isinstance(self.init, str) else "array",
            # option parity (lane/algos-decomp): metric code, minkowski p, a, b
            "extras": Array.from_list([float(v) for v in c[11:15]], "<f8"),  # glue: four saved option scalars
            "training": self._transform_training,
            "embedding": self._transform_embedding,
            "meta": Array.from_list(
                [int(c[0]), int(c[1]), int(c[2]), int(c[7]), int(c[10]),
                 1 if self.n_epochs is None else 0,
                 1 if getattr(self, "input_copied_", False) else 0],
                "<i8"),
            "controls": Array.from_list([c[3], c[4], c[5], c[6], c[8], c[9]], "<f4"),
        }
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load an embedding saved by `save`. The result transforms new
        rows against the saved training rows and embedding; it does not
        refit. Transform results do not depend on the query batch (see
        `save`)."""
        from . import _serialize
        from .decomposition import _check_saved_by, _restore_mode
        arrays = _serialize.read_npz(path, _UMAP_FORMAT)
        _check_saved_by(arrays, path, cls)
        meta = _serialize.exact(arrays, "meta", "<i8")
        controls = _serialize.exact(arrays, "controls", "<f4")
        if meta.size != 7 or controls.size != 6:
            raise ValueError(f"mojolearn: {path!r} holds {meta.size} meta and {controls.size} control "
                             "fields; 7 and 6 are needed")
        m = [int(meta[i]) for i in range(7)]  # glue: unpacks the fixed meta vector
        f = [float(controls[i]) for i in range(6)]  # glue: unpacks the fixed controls vector
        init = _serialize.scalar_str(arrays, "init")
        ex = [-1.0, 2.0, 0.0, 0.0]
        if "extras" in arrays:
            e = _serialize.exact(arrays, "extras", "<f8")
            if e.size != 4:
                raise ValueError(f"mojolearn: {path!r} holds {e.size} extras; 4 are needed")
            ex = [float(e[i]) for i in range(4)]  # glue: unpacks four saved extras
        obj = cls(n_neighbors=m[0], n_components=m[1], n_epochs=None if m[5] else m[2],
                  random_state=m[3], min_dist=f[0], spread=f[1], set_op_mix_ratio=f[2],
                  local_connectivity=f[3], metric=_serialize.scalar_str(arrays, "metric"),
                  init="spectral" if init == "array" else init, learning_rate=f[4],
                  repulsion_strength=f[5], negative_sample_rate=m[4],
                  a=ex[2] if ex[2] > 0 else None, b=ex[3] if ex[3] > 0 else None,
                  metric_kwds={"p": ex[1]} if int(ex[0]) == 9 else None)
        _restore_mode(obj, arrays)
        config = tuple(obj._parameters()) + tuple(obj._extras())
        saved = (m[0], m[1], m[2], f[0], f[1], f[2], f[3], m[3], f[4], f[5], m[4],
                 int(ex[0]), ex[1], ex[2], ex[3])
        if config != saved:
            raise ValueError(f"mojolearn: {path!r} controls {saved} do not survive validation as {config}")
        training = _serialize.exact(arrays, "training", "<f4")
        embedding = _serialize.exact(arrays, "embedding", "<f4")
        if (training.ndim != 2 or embedding.ndim != 2 or embedding.shape[0] != training.shape[0]
                or embedding.shape[1] != m[1]):
            raise ValueError(f"mojolearn: {path!r} training {tuple(training.shape)} and embedding "
                             f"{tuple(embedding.shape)} do not agree with n_components={m[1]}")
        obj._transform_training = training
        obj._transform_embedding = embedding
        obj._transform_config = config
        obj._transform_mode = obj.numeric_mode
        obj.embedding_ = embedding.copy()
        obj.n_features_in_ = int(training.shape[1])
        obj.input_copied_ = bool(m[6])
        return obj
