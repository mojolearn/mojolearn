# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GPU linear models, mirroring cuML's `LinearRegression(algorithm='eig')`,
`Ridge(solver='eig')` and `LogisticRegression(solver='qn')`.

NUMPY-FREE SINCE DEVIATION 2360 (branch numpy-free-0.7). Inputs cross the
boundary through `_buffer.as_f32_c` and friends, outputs are `_array.Array`,
and the two host reductions this file owns -- the column means and the
target mean that center an intercept fit -- run through the native
`column_mean_f64` helper (bindings/_mojolearn.mojo, DEVIATION 2303/2324),
whose accumulation order is written down. What is still Python here is
named at each site: label encoding (permitted, O(rows)), and the
elementwise centering and sqrt-weight rescale, which run through the native
`center_columns_f32` / `scale_rows_f32` helpers when the loaded binary
carries them and through a Python reference spelling of the same
arithmetic otherwise (DEVIATION 2450).
"""

import array
import math

from . import _mojolearn_estimators
from ._array import Array
from ._buffer import (
    addr, addr_ro, all_finite, as_f32_c, as_f64_c, empty, view, zeros,
)
from ._labels import decode_labels, flatten_labels, sorted_classes
from ._mode import NumericModeMixin


# ---------------------------------------------------------------------------
# Host helpers (DEVIATION 2360). Small on purpose; every one says what it
# costs.
# ---------------------------------------------------------------------------


def _round_f32(v):
    """`float(np.float32(v))`: ONE round-to-nearest-even of a Python float
    (binary64) to binary32 and back. `array.array('f')` stores through the C
    `(float)` cast, which is that rounding; an out-of-range value becomes
    +-inf exactly as NumPy's `astype(float32)` makes it."""
    return array.array("f", (float(v),))[0]


def _shape_of(obj):
    """The shape of an array-like WITHOUT converting it: `.shape` when the
    object has one (ndarray, Array, memoryview), the nesting of a list or
    tuple otherwise. Used only to raise the estimator's own message before
    `_buffer` would raise its generic one."""
    shape = getattr(obj, "shape", None)
    if shape is not None:
        return tuple(shape)
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (list, tuple)):
            return (len(obj), len(obj[0]))
        return (len(obj),)
    if hasattr(obj, "__len__"):
        return (len(obj),)
    return ()


def _labels_1d(y):
    """`y` as a Python list of scalars, plus its shape. Labels may be ints,
    floats or strings, so this never goes through a float buffer; `tolist()`
    (ndarray, Array, array.array) or plain iteration. O(rows), which is the
    one Python loop the contract permits on labels."""
    shape = _shape_of(y)
    if len(shape) != 1:
        return None, shape
    return flatten_labels(y), shape


# The `classes_` ORDER RULE and the label <-> code maps live in
# `_labels.py` (DEVIATION 2340): `sorted_classes`, `decode_labels`. Every
# classifier in the package orders `classes_` through that one spelling;
# DEVIATION 2364 is this file's adoption of it (a Python list, int64 /
# float64 Arrays out of `predict` for int / float labels, a list otherwise).


def _flatten(obj):
    """Every leaf of a nested list/tuple (or anything with `tolist()`) as
    one flat Python list; a scalar is a one-element list."""
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    if isinstance(obj, (list, tuple)):
        out = []
        for v in obj:
            out.extend(_flatten(v))
        return out
    return [obj]


#: struct-format codes of the integer buffer types.
_INT_FORMATS = frozenset("bBhHiIlLqQnN")


def _buffer_format(obj):
    """The struct format code of a buffer-protocol object, byte-order prefix
    stripped; None when `obj` exports no buffer (a list, a scalar)."""
    if isinstance(obj, Array):
        return {"<f4": "f", "<f8": "d", "<i4": "i", "<i8": "q", "<u4": "I",
                "<u1": "B", "<f2": "e"}.get(obj.dtype)
    try:
        with view(obj) as buf:
            return buf.format.lstrip("<>=@!|")
    except TypeError:
        return None


def _is_integer_labels(y):
    """`np.issubdtype(y.dtype, np.integer)`: True for a buffer whose format
    is an integer code (bools are NOT integers here, as in NumPy), or a
    list/tuple whose every leaf is a Python int (bools excluded)."""
    fmt = _buffer_format(y)
    if fmt is not None:
        return fmt in _INT_FORMATS
    leaves = _flatten(y)
    return bool(leaves) and all(type(v) is int for v in leaves)


def _dtype_name(y):
    """What to print for `y`'s dtype in a refusal: the buffer format code
    when it has one, the Python type name otherwise."""
    fmt = _buffer_format(y)
    if fmt is not None:
        return f"buffer format {fmt!r}"
    return f"type {type(y).__name__}"


def _accuracy_host(pred, y):
    """The fraction of rows where `pred == y`, a Python count over O(rows)
    labels (DEVIATION 2365, a host reduction). `pred` is an Array or a
    list, as the `predict` methods return."""
    pred = pred.tolist() if isinstance(pred, Array) else list(pred)
    labels, _ = _labels_1d(y)
    if labels is None or len(labels) != len(pred):
        raise ValueError(
            "mojolearn: y must be 1-D with one entry per row of X"
        )
    hits = 0
    for p, t in zip(pred, labels):
        if p == t:
            hits += 1
    return hits / len(pred) if pred else float("nan")


def _target_1d(y, n_rows, ndim_msg, length_msg):
    """A 1-D C-contiguous float32 target of length `n_rows`, with the
    estimator's own two messages raised before `_buffer`'s generic ones."""
    if len(_shape_of(y)) != 1:
        raise ValueError(ndim_msg)
    t, _ = as_f32_c(y, ndim=1, name="y")
    if t.shape[0] != n_rows:
        raise ValueError(length_msg)
    return t


def _r2_sums(pred, y):
    """`(SS_res, SS_tot)` for scikit-learn's R^2, from float32 predictions
    and a float64 target, accumulated SEQUENTIALLY in Python float64: the
    target mean first, then both sums of squares in one pass in row order.

    A HOST REDUCTION, OUTSIDE THE IDENTITY CLAIM (DEVIATION 2365). This
    used to be NumPy's pairwise `np.sum`; a sequential fold rounds
    differently in the last bits, so a fixture that recorded a `score()`
    value under the NumPy-era code is RE-BASELINE OWED. The predictions
    themselves are untouched: this is a summary of the answer, not the
    answer. Callers apply their own convention for `SS_tot == 0`.
    """
    t, _ = as_f64_c(y, ndim=1, name="y")
    tl = t.tolist()
    pl = pred.tolist()
    n = len(tl)
    total = 0.0
    for v in tl:
        total += v
    mean = total / n
    ss_res = 0.0
    ss_tot = 0.0
    for v, p in zip(tl, pl):
        r = v - p
        ss_res += r * r
        d = v - mean
        ss_tot += d * d
    return ss_res, ss_tot


def _r2_host(pred, y):
    """`1 - SS_res / SS_tot`, and 0.0 for a constant target -- the linear
    models' NumPy-era convention (`if denom else 0.0`). See `_r2_sums`."""
    ss_res, ss_tot = _r2_sums(pred, y)
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0


def _native_helper(name):
    """A host helper of the base binding (`column_mean_f64`,
    `center_columns_f32`, `scale_rows_f32`), resolved through
    `_buffer._native`: once per name for the process's tier, raising by
    name when the binary predates it. The helpers are host code with no
    device context and no tier-dependent arithmetic, so any tier's binary
    gives the same bits (DEVIATION 2450). The Python fallbacks of
    DEVIATION 2361 that once shadowed them were removed 2026-09-10."""
    from ._buffer import _native
    return _native(name)


def _column_means_f64(x, rows, cols):
    """Per-column float64 means of a C-contiguous float32 `[rows, cols]`
    buffer, in the ORDER `bindings/_mojolearn.mojo::column_mean_f64_binding`
    defines (DEVIATION 2324): row by row, column by column, one binary64
    round-to-nearest-even addition per element, then one division by
    `rows`. Returns a Python list of `cols` floats.

    A 1-D vector is a `[rows, 1]` matrix here, which is how
    `_vector_mean` uses it.
    """
    fn = _native_helper("column_mean_f64")
    out = empty((cols,), "<f8")
    fn(addr_ro(x, name="X"), int(rows), int(cols),
       addr(out, name="column means"))
    return out.tolist()


def _weight_total(weights):
    """`sum(w)` in float64, SEQUENTIAL (DEVIATION 2366; NumPy's was the
    pairwise 1-D kernel, so the weighted-fit bits move and the weighted
    OLS cards are RE-BASELINE OWED with the rest)."""
    total = 0.0
    for v in weights.tolist():
        total += v
    if total <= 0.0:
        raise ValueError(
            "mojolearn: sample_weight sums to zero, so the weighted mean "
            "cuML's preProcessData forms is a division by zero"
        )
    return total


def _column_means(x, weights):
    """Column means in float64, narrowed to float32 -- weighted when
    `weights` is not None. Returns a Python list of float32-valued floats.

    Unweighted this is `column_mean_f64` over `x`: the sequential
    row-order float64 accumulation the helper's docstring defines
    (DEVIATION 2324). It REPLACES `x.mean(axis=0, dtype=np.float64)`, whose
    blocked reduction had no order a second implementation could
    reproduce; the OLS reference cards are RE-BASELINE OWED on all three
    vendors for that reason (NUMPY_FREE_CONTRACT.md).

    Weighted it is cuML's `raft::stats::weightedMean`, `sum_i w_i x_ij /
    sum_i w_i` (`raft/stats/detail/weighted_mean.cuh:49-64`), in THIS
    ORDER (DEVIATION 2366): (1) a float32 copy `wx_ij = fl32(x_ij * w_i)`
    -- the float64 product of two float32 values is exact, so that is ONE
    rounding per element; (2) `column_mean_f64` over `wx`, which yields
    `sum_i wx_ij / rows` in the defined order; (3) `mean_j * rows / total`
    in float64, `total` the sequential float64 sum of the weights; (4) one
    narrowing to float32. The NumPy-era spelling kept the products in
    float64 and summed with the pairwise kernel, so these bits DIFFER from
    it: the weighted OLS cards are RE-BASELINE OWED. Theirs divides by the
    SUM OF THE WEIGHTS and not by the row count, so a uniform weight of 2
    leaves the mean unchanged, which is the property the gate checks.
    """
    rows, cols = x.shape
    if weights is None:
        mu = _column_means_f64(x, rows, cols)
    else:
        w = weights.tolist()
        total = _weight_total(weights)
        wx = Array.from_list(
            [[v * wr for v in row] for wr, row in zip(w, x.tolist())], "<f4"
        )
        mu = [m * rows / total for m in _column_means_f64(wx, rows, cols)]
    return [_round_f32(m) for m in mu]


def _vector_mean(v, weights):
    """The scalar float64 mean of the target, weighted when `weights` is
    not None. The 1-D half of `_column_means` (a vector is a `[rows, 1]`
    matrix to `column_mean_f64`); see that docstring for the order. This
    REPLACES `v.mean(dtype=np.float64)`, NumPy's pairwise 1-D kernel, so the
    intercept's bits move with the column means (RE-BASELINE OWED)."""
    n = v.shape[0]
    if weights is None:
        return _column_means_f64(v, n, 1)[0]
    total = _weight_total(weights)
    wy = Array.from_list([a * b for a, b in zip(v.tolist(), weights.tolist())],
                         "<f4")
    return _column_means_f64(wy, n, 1)[0] * n / total


def _dims(x):
    """`(rows, cols)` of a 2-D Array, or `(rows, 1)` of a 1-D one: a vector
    is a `[rows, 1]` matrix to every host helper in this file."""
    return (x.shape[0], x.shape[1]) if x.ndim == 2 else (x.shape[0], 1)


def _center(x, mu32):
    """`x - mu` in float32, one rounding per element: `fl32(x_ij - mu_j)`
    with both operands float32 values; `x` is `[rows, cols]` or a vector
    with `cols == 1`, `mu32` a list of `cols` float32-valued floats.

    DEVIATION 2450 -- NATIVE WHEN AVAILABLE; THE PYTHON FALLBACK IS THE
    REFERENCE SPELLING. The binding's `center_columns_f32(x, rows, cols,
    mean, out)` reproduces this loop exactly; it takes the means as
    FLOAT64, and they are handed over already rounded to float32 (exactly
    representable in float64), so whether the helper widens `x` to the
    mean or narrows the mean to `x`, the difference it rounds is the same
    difference. The fallback computes each difference in float64 and
    rounds once, which is the correctly rounded float32 subtraction (double
    rounding is innocuous for +, -, *, / and sqrt when the wide format has
    at least 2p + 2 bits; 53 >= 50): the bits NumPy's float32 subtract
    produced, and the bits the helper produces.
    """
    rows, cols = _dims(x)
    fn = _native_helper("center_columns_f32")
    mean = Array.from_list([float(m) for m in mu32], "<f8")
    out = empty(x.shape, "<f4")
    fn(addr_ro(x, name="X"), int(rows), int(cols),
       addr_ro(mean, name="column means"), addr(out, name="centered X"))
    return out


def _shift(v, mu32):
    """The 1-D `_center`: `fl32(v_i - mu)`, `mu` already a float32 value."""
    return _center(v, [mu32])


def _scale_rows(x, root):
    """Row `i` of `x` times `root[i]`, float32: `fl32(x_ij * r_i)`, one
    rounding (the float64 product of two float32 values is exact); `x` is
    `[rows, cols]` or a vector, `root` a list of `rows` float32-valued
    floats. The device's `ols_fit_weighted` performs the same multiply,
    which is the bit-for-bit claim
    `check_ols_sample_weight_host_rescale_matches_device` gates.

    DEVIATION 2450 -- NATIVE WHEN AVAILABLE (`scale_rows_f32(x, rows, cols,
    w, out)`, float32 weights); THE PYTHON FALLBACK IS THE REFERENCE
    SPELLING of the same arithmetic.
    """
    rows, cols = _dims(x)
    fn = _native_helper("scale_rows_f32")
    w = Array.from_list([float(r) for r in root], "<f4")
    out = empty(x.shape, "<f4")
    fn(addr_ro(x, name="X"), int(rows), int(cols),
       addr_ro(w, name="sqrt weights"), addr(out, name="scaled X"))
    return out


def _check_sample_weight(sample_weight, n_rows, estimator):
    """Validate `sample_weight` and return it as a C-order float32 Array,
    or None when there are no weights.

    cuML validates nothing here: `olsFit` takes a raw pointer and trusts it.
    These three checks are OURS and they are input validation, which is the
    one kind of refusal that is correct rather than a gap -- a length
    mismatch would read past the end of the array on the device, and a
    negative weight makes `sqrt(w)` a NaN that silently poisons the whole
    fit. Non-finite weights are refused for the same reason.
    """
    if sample_weight is None:
        return None
    shape = _shape_of(sample_weight)
    if len(shape) != 1:
        raise ValueError(
            f"mojolearn {estimator}: sample_weight must be 1-D, got "
            f"{len(shape)} dimensions"
        )
    w, _ = as_f32_c(sample_weight, ndim=1, name="sample_weight")
    if w.shape[0] != n_rows:
        raise ValueError(
            f"mojolearn {estimator}: sample_weight has {w.shape[0]} entries "
            f"but X has {n_rows} rows"
        )
    if not all_finite(w):
        raise ValueError(
            f"mojolearn {estimator}: sample_weight contains a non-finite "
            "value; sqrt of it would poison every row of the fit"
        )
    if w.min() < 0.0:
        raise ValueError(
            f"mojolearn {estimator}: sample_weight has a negative entry; "
            "a weighted least squares scales rows by sqrt(w) (ols.cuh:100) "
            "and sqrt of a negative is a NaN"
        )
    return w


class LinearRegression(NumericModeMixin):
    """Ordinary least squares through normal equations on the GPU.

    This is cuML's `algorithm='eig'` arm (`lstsqEig`, RAFT), which forms
    ``X.T @ X`` and so squares the condition number. It is less robust than
    an SVD-based solver and should not be used for badly conditioned
    designs; cuML's SVD and QR solvers (`lstsqSvdJacobi`, `lstsqSvdQR`,
    `lstsqQR`) are not written here (glm/NOT_IMPLEMENTED.tsv). A design with
    more features than samples takes a second Gram route instead of an SVD;
    see the `n_features > n` row below. On the tall route the Gram matrix is
    equilibrated by exact power-of-two column scales and its pseudo-inverse
    keeps eigenvalues above ``n_features * eps32 * max`` (DEVIATIONS 2620,
    2621), so a column's units no longer change the model's rank.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY (measured row by row by
    `tools/e2u_matrix_fit.py`):

        fit_intercept   honored   True (the default) centers X and y ON THE
                                  HOST and the device solves the centered
                                  system; see THE INTERCEPT IS A HOST
                                  REIMPLEMENTATION below. False sends the
                                  raw design to the device, which is the
                                  arm the implemented `ols_fit` carries.
        sample_weight   honored   a weighted least squares IS an unweighted
                                  one on rows rescaled by sqrt(w), which is
                                  exactly what cuML does (ols.cuh:99-110)
                                  before it dispatches; see SAMPLE WEIGHTS
                                  below
        n_features == 1 honored   a single column is scalar least squares
                                  and the Gram is 1 x 1 with condition
                                  number 1; cuML switches to its SVD solver
                                  here because THEIR eigensolver does not
                                  take one column (linear_regression.pyx:
                                  390-394), a limitation the device Jacobi
                                  does not share (DEVIATION 551)
        n_features > n  honored   the minimum-norm solution
                                  w = X.T (X X.T)^+ y, through the Gram of
                                  the ROWS, which is n x n and nonsingular
                                  at full row rank (DEVIATION 550,
                                  glm/impl/linalg/detail/lstsq_min_norm.mojo).
                                  It squares the condition number exactly as
                                  the tall route does, so it is no more
                                  accurate than this class already is and no
                                  less; a true SVD or LQ route would be
                                  better and is not written
        y 2-D           refused   one target only, at this boundary

    SAMPLE WEIGHTS ARE A HOST RESCALE HERE, AND THAT IS A DEVIATION WITH A
    REASON. `olsFit` takes `sqrt` of the weights, multiplies row `i` of X
    and entry `i` of y by it, solves, and undoes the scaling
    (ols.cuh:99-110, 129-141). Both halves of the scaling are implemented and run
    ON THE DEVICE in `glm/impl/ols.mojo::ols_fit_weighted`; what is not
    yet in place is a BINDING that can hand a weight pointer across
    (`bindings/_mojolearn_estimators.mojo::ols_fit_binding` takes a fixed
    `params` of length 2). Until it is, this class applies the same two
    operations on the host and calls the unweighted entry.

    That is defensible where a host reimplementation usually is not, and the
    reason is arithmetic rather than convenience: `math.sqrt` of a float32
    value in float64 followed by ONE float32 rounding IS the IEEE correctly
    rounded float32 square root (double rounding is innocuous for sqrt when
    the wide format carries at least 2p + 2 bits, and 53 >= 50), the row
    multiply is one float32 rounding of an exact float64 product, and both
    are the same operations the device kernels perform. The two routes are
    therefore expected to agree BIT FOR BIT except on denormals, where the
    device flushes and the host does not, and
    `check_ols_sample_weight_host_rescale_matches_device` in
    `glm/checks/ols_check.mojo` is the gate on that rather than this
    paragraph. `sample_weight` with `fit_intercept=True` uses WEIGHTED
    column means, which is what cuML does too (`raft::stats::weightedMean`,
    preprocess.cuh:95-97, 110-112).

    THE INTERCEPT IS A HOST REIMPLEMENTATION, NO REFERENCE FILE. cuML's Python
    default `fit_intercept=True` wraps the solver in `preProcessData`
    (center X and y on the DEVICE) and `postProcessData` (intercept =
    mean(y) - mu_X . coef; preprocess.cuh:98-176). The implemented `ols_fit`
    REFUSES `fit_intercept` by name because those two are not implemented
    (glm/impl/ols.mojo). This class therefore does the centering here,
    on the host: column means and the y mean in float64 through the native
    `column_mean_f64` helper (a sequential row-order accumulation whose
    order is the helper's contract, DEVIATION 2324), subtracted in float32,
    and the intercept as `mean(y) - sum(mu_X * coef)` with `math.fsum`
    (exactly rounded; NO BLAS dot, which would be a platform-dependent host
    reduction -- E2's first finding). The device sees a centered design;
    the arithmetic that reaches it is a function of the inputs alone. The
    mean-centering is mathematically what cuML does, but it runs on the
    host in float64 where theirs runs on the device in float32, so
    `coef_` CAN differ from cuML's in the last bits on ill-conditioned
    data. Named here because a hidden host step is the thing this library
    exists to not have.
    """

    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn_estimators"

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y, sample_weight=None):
        x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
        rows, cols = x.shape
        target = _target_1d(
            y, rows,
            "mojolearn LinearRegression currently requires one target",
            "mojolearn LinearRegression X and y lengths differ",
        )
        weights = _check_sample_weight(sample_weight, rows, "LinearRegression")
        if self.fit_intercept:
            # float64 column means -> float32, then a float32 subtraction.
            # The means come from `column_mean_f64`, a sequential row-order
            # float64 accumulation with a written-down order (DEVIATION
            # 2324); the 1-D target mean is the same helper over a
            # [rows, 1] view. Neither goes through BLAS or libm. The dot
            # below is the one place a BLAS call would have slipped in, so
            # it is an exactly-rounded fsum instead.
            #
            # WITH WEIGHTS THE MEANS ARE WEIGHTED, which is cuML's
            # preProcessData (`raft::stats::weightedMean`, sum(w*x)/sum(w),
            # preprocess.cuh:95-97 and 110-112) and NOT the unweighted mean
            # applied to weighted rows. The two differ, and using the wrong
            # one puts the intercept in the wrong place without moving any
            # coefficient enough to notice.
            mu32 = _column_means(x, weights)
            self._x_mean = Array.from_list(mu32, "<f4")
            self._y_mean = _vector_mean(target, weights)
            # NumPy narrowed the Python-float y mean to float32 BEFORE the
            # float32 subtract (value-based / weak-scalar casting); the
            # same order here so the centered bits are the same bits.
            work_x = _center(x, mu32)
            work_y = _shift(target, _round_f32(self._y_mean))
        else:
            work_x, work_y = x, target
            self._x_mean = zeros((cols,), "<f4")
            self._y_mean = 0.0
        if weights is not None:
            # `olsFit`, ols.cuh:99-110, on the host. See SAMPLE WEIGHTS in
            # the class docstring for why this is here and not in the Mojo
            # layer, and for the bit-for-bit claim the gate checks.
            root = [_round_f32(math.sqrt(v)) for v in weights.tolist()]
            work_x = _scale_rows(work_x, root)
            work_y = _scale_rows(work_y, root)
        self.coef_ = empty((cols,), "<f4")
        self._bind("_mojolearn_estimators").ols_fit(
            addr_ro(work_x, name="X"), addr_ro(work_y, name="y"),
            addr(self.coef_, name="coef_"),
            [rows, cols],
        )
        if self.fit_intercept:
            dot = math.fsum(
                float(a) * float(b)
                for a, b in zip(self._x_mean.tolist(), self.coef_.tolist())
            )
            self.intercept_ = float(self._y_mean - dot)
        else:
            self.intercept_ = 0.0
        self.n_features_in_ = cols
        return self

    def predict(self, X):
        if not hasattr(self, "coef_"):
            raise ValueError("mojolearn LinearRegression: call fit before predict")
        x, _ = as_f32_c(X, ndim=2, name="X")
        if x.shape[1] != self.n_features_in_:
            raise ValueError("mojolearn LinearRegression feature count differs from fit")
        out = empty((x.shape[0],), "<f4")
        self._bind("_mojolearn_estimators").ols_predict(
            addr_ro(x, name="X"), addr_ro(self.coef_, name="coef_"),
            addr(out, name="predictions"),
            [x.shape[0], x.shape[1], float(self.intercept_)],
        )
        return out

    def score(self, X, y):
        """R^2, a sequential float64 host reduction (DEVIATION 2365)."""
        return _r2_host(self.predict(X), y)


class Ridge(NumericModeMixin):
    """l2-regularized least squares on the GPU, cuML's `solver='eig'` arm.

    Mirrors `cuml/python/cuml/linear_model/ridge.pyx` on top of
    `cuml/cpp/src/glm/ridge.cuh::ridgeFit` (DEVIATION 545; the Mojo implementation is
    `glm/impl/ridge.mojo` and the design note there is worth reading:
    their `eig` solver is an SVD through the eigendecomposition of `X.T @ X`
    followed by `ridgeSolve`, NOT "OLS with alpha added", and so is ours).
    Solves `min ||y - Xw||^2 + alpha ||w||^2`; the same objective as
    scikit-learn's `Ridge`, and the same minimizer. Forms `X.T @ X`, so the
    condition number is squared (see `LinearRegression`).

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY (measured row by row by
    `tools/e2u_matrix_fit.py`):

        alpha           honored   a non-negative float (ridge.pyx:296);
                                  0.0 is allowed and is least squares
                                  through the ridge program
        fit_intercept   honored   True centers X and y ON THE HOST exactly
                                  as `LinearRegression` does (that class's
                                  docstring: THE INTERCEPT IS A HOST
                                  REIMPLEMENTATION, DEVIATION 517); the
                                  implemented `ridge_fit` sees a centered
                                  design with fit_intercept=False, which
                                  is `ridge.cuh:247`'s `intercept = 0` arm
        solver          'eig' only  cuML's 'auto' maps to 'eig'
                                  (ridge.pyx:304); 'svd' (ridgeSVD ->
                                  cuSOLVER gesvd, glm/NOT_IMPLEMENTED.tsv) and
                                  'cd' (cuml/solvers/cd.pyx, a different
                                  solver) are REFUSED by name
        normalize       refused   only reachable with fit_intercept on
                                  their side and is preProcessData's
                                  meanvar arm (preprocess.cuh:76-108),
                                  not implemented
        sample_weight   refused   ridge.cuh:197-208 / 220-231, a sqrt-
                                  scaling of both operands and its exact
                                  inverse, not implemented
        n_features == 1 refused   ridge.cuh:210 forces ridgeSVD for one
                                  column and the Python layer warns and
                                  switches (ridge.pyx:355); raised BY NAME
                                  by the Mojo layer
        y 2-D           refused   one target only, at this boundary
    """

    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn_estimators"

    def __init__(self, *, alpha=1.0, solver="auto", fit_intercept=True,
                 normalize=False):
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if solver not in ("auto", "eig", "svd", "cd"):
            raise TypeError(f"solver {solver!r} is not supported")
        if solver in ("svd", "cd"):
            raise NotImplementedError(
                f"mojolearn Ridge: solver={solver!r} is not implemented "
                "(ridgeSVD is raft::linalg::svdQR -> cuSOLVER gesvd; 'cd' "
                "is cuml/solvers/cd.pyx); solver='eig' (cuML's 'auto') is "
                "the implemented arm. See glm/NOT_IMPLEMENTED.tsv"
            )
        if normalize:
            raise NotImplementedError(
                "mojolearn Ridge: normalize is not implemented (preprocess.cuh:"
                "76-108, the meanvar arm of preProcessData; glm/NOT_IMPLEMENTED.tsv)"
            )
        self.alpha = alpha
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.normalize = normalize

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError(
                "mojolearn Ridge: sample_weight is not implemented "
                "(ridge.cuh:197-208; glm/NOT_IMPLEMENTED.tsv)"
            )
        self.solver_ = "eig"
        x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
        rows, cols = x.shape
        target = _target_1d(
            y, rows,
            "mojolearn Ridge currently requires one target",
            "mojolearn Ridge X and y lengths differ",
        )
        if self.fit_intercept:
            # The same host centering as LinearRegression, for the same
            # reasons; read that class's fit. `column_mean_f64` replaces
            # `x.mean(axis=0, dtype=np.float64)` (DEVIATION 2361); the
            # ridge cards are RE-BASELINE OWED with the OLS ones.
            mu32 = _column_means(x, None)
            self._x_mean = Array.from_list(mu32, "<f4")
            self._y_mean = _vector_mean(target, None)
            work_x = _center(x, mu32)
            work_y = _shift(target, _round_f32(self._y_mean))
        else:
            work_x, work_y = x, target
            self._x_mean = zeros((cols,), "<f4")
            self._y_mean = 0.0
        self.coef_ = empty((cols,), "<f4")
        self._bind("_mojolearn_estimators").ridge_fit(
            addr_ro(work_x, name="X"), addr_ro(work_y, name="y"),
            addr(self.coef_, name="coef_"),
            [rows, cols, float(self.alpha)],
        )
        if self.fit_intercept:
            dot = math.fsum(
                float(a) * float(b)
                for a, b in zip(self._x_mean.tolist(), self.coef_.tolist())
            )
            self.intercept_ = float(self._y_mean - dot)
        else:
            self.intercept_ = 0.0
        self.n_features_in_ = cols
        return self

    def predict(self, X):
        if not hasattr(self, "coef_"):
            raise ValueError("mojolearn Ridge: call fit before predict")
        x, _ = as_f32_c(X, ndim=2, name="X")
        if x.shape[1] != self.n_features_in_:
            raise ValueError("mojolearn Ridge feature count differs from fit")
        out = empty((x.shape[0],), "<f4")
        # The same gemv + intercept epilogue OLS predicts with
        # (`gemmPredict` upstream serves both, `base.pyx:134`).
        self._bind("_mojolearn_estimators").ols_predict(
            addr_ro(x, name="X"), addr_ro(self.coef_, name="coef_"),
            addr(out, name="predictions"),
            [x.shape[0], x.shape[1], float(self.intercept_)],
        )
        return out

    def score(self, X, y):
        """R^2, a sequential float64 host reduction (DEVIATION 2365)."""
        return _r2_host(self.predict(X), y)


# cuML's `qn_params.loss` ids (cuml/linear_model/qn.h); the Python door maps
# 'sigmoid' to QN_LOSS_LOGISTIC and 'softmax' to QN_LOSS_SOFTMAX.
_QN_OPT_RETCODE = {0: "OPT_SUCCESS", 1: "OPT_NUMERIC_ERROR",
                   2: "OPT_LS_FAILED", 3: "OPT_MAX_ITERS_REACHED",
                   4: "OPT_INVALID_ARGS"}


def _log_or_inf(p):
    """`np.log` on one probability: `log(p)` for `p > 0`, `-inf` at exactly
    zero (NumPy's answer, minus its warning), NaN for a negative."""
    if p > 0.0:
        return math.log(p)
    if p == 0.0:
        return float("-inf")
    return float("nan")


class LogisticRegression(NumericModeMixin):
    """Binary logistic regression on the GPU, cuML's quasi-Newton solver.

    Mirrors `cuml/python/cuml/linear_model/logistic_regression.py` on top of
    `cuml/python/cuml/solvers/qn.pyx` and `cuml/cpp/src/glm/qn/` (DEVIATIONS
    546-549; the Mojo implementation is `glm/impl/qn/*.mojo`, one file per
    theirs). The objective is `mean_i logloss_i + (1/(2 C n)) ||w||^2`
    (`penalty_normalized=True`: cuML divides the penalty by n so that its
    minimizer is scikit-learn's `LogisticRegression(C)` minimizer), the
    solver is L-BFGS (`lbfgs_memory=5`) with a backtracking Armijo line
    search -- or OWL-QN with a PROJECTED Armijo line search when the penalty
    has an l1 part (`qn_solvers.cuh:420`, DEVIATION 552) -- and convergence
    is `max|grad| <= tol * max(loss, tol)` or an objective change below
    `tol * 0.01 * max(loss, tol)` over 10 iterations
    (`qn_util.cuh::check_convergence`).

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY (measured row by row by
    `tools/e2u_matrix_fit.py`):

        penalty         all four honored   'l2' is Tikhonov with
                                  l2 = 1/C (logistic_regression.py:640);
                                  None is the unregularized arm (qn.cuh:61);
                                  'l1' and 'elasticnet' set l1 != 0, which
                                  selects OWL-QN (qn_solvers.cuh:420-445),
                                  implemented 2026-09-01 as
                                  glm/impl/qn/qn_solvers.mojo::min_owlqn
                                  (DEVIATION 552)
        C               honored   inverse regularization strength, > 0
        tol             honored   grad_tol = tol, change_tol = tol * 0.01,
                                  ftol = change_tol * 0.1 (qn.pyx:504-506,
                                  qn_util.cuh:88-98)
        fit_intercept   honored   the bias is a parameter of the solver
                                  (GLMDims, glm_base.cuh:96); it is NOT
                                  penalized (glm_regularizer.cuh:45); no
                                  host centering here, unlike the linear
                                  models
        max_iter        honored   the L-BFGS iteration cap; reaching it is
                                  a WARNING upstream and `n_iter_ ==
                                  max_iter` with `retcode_ == 3` here
        linesearch_max_iter honored  default 50 as theirs
        class_weight    refused   becomes a sample_weight upstream
                                  (logistic_regression.py:400-436), and
                                  sample_weight is not implemented
        sample_weight   refused   GLMBase::add_sample_weights and the
                                  weighted getLossAndDZ arm, not implemented
        l1_ratio        honored, and REQUIRED with penalty='elasticnet'
                                  (logistic_regression.py:310-316): the
                                  split is l1 = l1_ratio / C,
                                  l2 = (1 - l1_ratio) / C
        solver          'qn' only the only value cuML accepts either
        > 2 classes     refused   softmax (glm_softmax.cuh) is not implemented;
                                  the Mojo layer raises by name
        warm_start      absent    cuML's QN has it, LogisticRegression
                                  does not expose it; w0 = 0 always

    THE l1 ARM IS A DIFFERENT SOLVER, NOT A DIFFERENT PENALTY. `|w|` has no
    gradient at zero, which is where an l1 solution sits, so cuML switches
    from L-BFGS to OWL-QN whenever `l1 != 0` (`qn_solvers.cuh:420`) and so
    does this implementation. OWL-QN keeps the L-BFGS history and replaces three
    things: the objective carries `l1 * ||w||_1` in its VALUE, the direction
    is built from a PSEUDO-gradient, and every step is projected back into
    the orthant it started in. That projection is what produces coefficients
    that are EXACTLY zero, which is the thing an l1 fit is asked for -- and
    it means the identity claim on this arm is about the SPARSITY PATTERN as
    well as about bits, because every branch that zeroes a coefficient is a
    float comparison with a discrete output. The intercept is not
    l1-penalized, exactly as it is not l2-penalized (`pg_limit = D * C`,
    `qn_solvers.cuh:447`).

    OUTPUTS: `coef_` (1, n_features) float32, `intercept_` (1,) float32,
    `classes_` (the two labels, sorted, a Python LIST -- DEVIATION 2364),
    `n_iter_` an int64 Array `[k]`, plus `objective_` (the final value of
    the objective the solver minimized) and `retcode_` (cuML's OPT_RETCODE,
    0 = converged). `predict` is `classes_[score > 0]`, `predict_proba` is
    float64 (n, 2) through `identical_exp64` on the host (DEVIATION 549;
    cuML computes it in float32 on the device and stores float64).

    IDENTITY: under MOJOLEARN_NUMERIC_MODE=identical every reduction in
    the objective, the gradient and the solver's dot products is a pinned
    fold, `exp`/`log` are the portable pair, and the line search's one
    host multiply-add is an fma -- so the accepted steps, the L-BFGS
    history, the ITERATION COUNT and the coefficients are a function of
    the inputs alone; the card (`qn.iterNNNN.*`, `qn.n_iter`, `qn.coef`)
    records all of them. Under the default FAST mode the reductions are
    the vendor's and the count may differ across GPUs.
    """

    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn_estimators"

    def __init__(self, *, penalty="l2", tol=1e-4, C=1.0, fit_intercept=True,
                 class_weight=None, max_iter=1000, linesearch_max_iter=50,
                 l1_ratio=None, solver="qn"):
        if penalty not in ("l1", "l2", "elasticnet", None):
            raise ValueError(f"`penalty` {penalty!r} not supported.")
        if solver != "qn":
            raise ValueError(
                "Only quasi-newton `qn` solver is supported, not %s" % solver)
        if class_weight is not None:
            raise NotImplementedError(
                "mojolearn LogisticRegression: class_weight is not implemented "
                "(it becomes a sample_weight upstream, logistic_regression.py"
                ":400-436, and sample_weight is not implemented; glm/NOT_IMPLEMENTED.tsv)"
            )
        if C <= 0:
            raise ValueError(f"C must be positive, got {C}")
        self.penalty = penalty
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.class_weight = None
        self.max_iter = max_iter
        self.linesearch_max_iter = linesearch_max_iter
        # `logistic_regression.py:310-316`, copied including the two
        # messages: `l1_ratio` is REQUIRED for elasticnet and IGNORED
        # (set to None) for every other penalty.
        self.l1_ratio = None
        if penalty == "elasticnet":
            if l1_ratio is None:
                raise ValueError(
                    "l1_ratio has to be specified for loss='elasticnet'"
                )
            if l1_ratio < 0.0 or l1_ratio > 1.0:
                raise ValueError(
                    "l1_ratio value has to be between 0.0 and 1.0"
                )
            self.l1_ratio = l1_ratio
        self.solver = solver
        # QN(...) defaults the Python door does not expose
        self.lbfgs_memory = 5
        self.penalty_normalized = True

    def _get_qn_params(self):
        """`_get_qn_params`, logistic_regression.py:632-649, all four arms.

        Returns `(l1_strength, l2_strength)`. `qn_fit` then divides both by
        `n` when `penalty_normalized` (qn.cuh:54-59), so what the solver sees
        is `(1/C)/n`, and `l1 != 0` is what selects OWL-QN there.
        """
        if self.penalty is None:
            return 0.0, 0.0
        if self.penalty == "l1":
            return 1.0 / self.C, 0.0
        if self.penalty == "l2":
            return 0.0, 1.0 / self.C
        strength = 1.0 / self.C
        return self.l1_ratio * strength, (1.0 - self.l1_ratio) * strength

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError(
                "mojolearn LogisticRegression: sample_weight is not implemented "
                "(GLMBase::add_sample_weights, glm_base.cuh:115; "
                "glm/NOT_IMPLEMENTED.tsv)"
            )
        x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
        rows, cols = x.shape
        labels, shape = _labels_1d(y)
        if labels is None:
            raise ValueError("mojolearn LogisticRegression requires a 1-D y")
        if len(labels) != rows:
            raise ValueError("mojolearn LogisticRegression X and y lengths differ")
        # LabelEncoder: sorted unique classes -> 0..k-1
        # (logistic_regression.py:383), under the package-wide ORDER RULE
        # (`_labels.sorted_classes`, DEVIATION 2340): a Python list.
        self.classes_, codes = sorted_classes(labels)
        n_classes = len(self.classes_)
        if n_classes > 2:
            raise NotImplementedError(
                f"mojolearn LogisticRegression: {n_classes} classes need the "
                "softmax loss (glm_softmax.cuh, QN_LOSS_SOFTMAX), which is NOT "
                "IMPLEMENTED; binary only. See glm/NOT_IMPLEMENTED.tsv"
            )
        if n_classes < 2:
            raise ValueError("mojolearn LogisticRegression: y has one class")
        # Label encoding, the permitted O(rows) Python loop: code 1 is
        # `classes_[1]`, the class the solver maps to +1.
        y_enc = Array.from_list([float(c) for c in codes], "<f4")
        l1, l2 = self._get_qn_params()
        n_param = cols + (1 if self.fit_intercept else 0)
        w = zeros((n_param,), "<f4")
        info = zeros((2,), "<f4")
        n_iter = self._bind("_mojolearn_estimators").qn_fit(
            addr_ro(x, name="X"), addr_ro(y_enc, name="y"),
            addr(w, name="coef_"), addr(info, name="info"),
            [rows, cols, n_classes,
             float(l1), float(l2), float(self.tol), float(self.tol * 0.01),
             int(self.max_iter), int(self.linesearch_max_iter),
             int(self.lbfgs_memory), 1 if self.fit_intercept else 0,
             1 if self.penalty_normalized else 0, 0],
        )
        self._w = w
        # A slice of an Array COPIES (the _array contract), so these are
        # detached from `_w` exactly as `.copy()` detached them before.
        self.coef_ = w[:cols].reshape((1, cols))
        self.intercept_ = (w[cols:cols + 1] if self.fit_intercept
                           else zeros((1,), "<f4"))
        self.n_iter_ = Array.from_list([int(n_iter)], "<i8")
        self.objective_ = float(info[0])
        self.retcode_ = int(info[1])
        self.n_features_in_ = cols
        return self

    def decision_function(self, X):
        if not hasattr(self, "_w"):
            raise ValueError("mojolearn LogisticRegression: call fit first")
        x, _ = as_f32_c(X, ndim=2, name="X")
        if x.shape[1] != self.n_features_in_:
            raise ValueError("mojolearn LogisticRegression feature count differs from fit")
        out = empty((x.shape[0],), "<f4")
        self._bind("_mojolearn_estimators").qn_decision_function(
            addr_ro(x, name="X"), addr_ro(self._w, name="coef_"),
            addr(out, name="scores"),
            [x.shape[0], x.shape[1], 1 if self.fit_intercept else 0],
        )
        return out

    def predict(self, X):
        """`qn_predict`: `z > 0 ? 1 : 0` (qn.cuh:276), mapped to classes_.
        An int64 / float64 Array for int / float labels, a Python list for
        anything else (DEVIATION 2364)."""
        scores = self.decision_function(X)
        return decode_labels(self.classes_,
                             [1 if s > 0.0 else 0 for s in scores.tolist()])

    def predict_proba(self, X):
        scores = self.decision_function(X)
        out = empty((scores.shape[0], 2), "<f8")
        self._bind("_mojolearn_estimators").qn_sigmoid(
            addr_ro(scores, name="scores"), addr(out, name="proba"),
            [scores.shape[0]])
        return out

    def predict_log_proba(self, X):
        """`log(predict_proba(X))`, float64 (n, 2), via `math.log` per
        element, `-inf` at an exact zero as NumPy gave.

        DEVIATION 2363 -- DEFECT, FLAGGED: this is an O(rows * 2) Python
        loop over the probability matrix. Two columns keep it proportionate
        to the O(rows) label loops the contract permits, but it is still a
        host `log` in a predict path and belongs behind a binding
        (`qn_sigmoid` could return the log form directly). Routed there
        later; recorded here so it is not mistaken for a design.
        """
        return Array.from_list(
            [[_log_or_inf(a), _log_or_inf(b)]
             for a, b in self.predict_proba(X).tolist()],
            "<f8",
        )

    def score(self, X, y):
        """Accuracy: the fraction of rows where `predict(X) == y`, a Python
        count over O(rows) labels (DEVIATION 2365, a host reduction)."""
        return _accuracy_host(self.predict(X), y)
