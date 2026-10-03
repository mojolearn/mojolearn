# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GPU linear models. Reference: cuML's `LinearRegression(algorithm='eig')`,
`Ridge(solver='eig')` and `LogisticRegression(solver='qn')`.

NUMPY-FREE SINCE DEVIATION 2360 (branch numpy-free-0.7). Inputs cross the
boundary through `_buffer.as_f32_c` and friends, outputs are `_array.Array`,
and the centering of an intercept fit -- the column sums, the subtraction
and the sqrt-weight rescale -- runs in the estimators binding
(`lm_col_sums`, `lm_center`, `lm_scale_rows`; on the device on a GPU
install, lane hr-small-passes 2026-10-02). The column sums are exact,
rounded once to float64, so they do not depend on any summation order.
What is still Python here is named at each site: label encoding
(permitted, O(rows)).
"""

import array
from . import _portable_math as math

from . import _backend, _mojolearn_estimators, _serialize
from ._array import Array
from ._buffer import (
    addr, addr_ro, all_finite, as_f32_c, as_f64_c, empty, view, zeros,
)
from ._labels import (
    argmax_rows, classes_from_member, classes_member, decode_labels,
    encode_labels, flatten_labels, sorted_classes, threshold_codes,
)
from ._mode import NumericModeMixin

#: The model file formats (the classical host inference lane, 2026-09-13).
#: `save` writes exactly what `predict` reads, raw bytes and exact dtypes,
#: through `_serialize.write_npz`, so equal models give equal file hashes
#: on every machine; `load` refuses a cast (`_serialize.exact`). The
#: intercept travels as `<f8` because it is a Python float computed with
#: `math.fsum` at fit time and `predict` hands `float(self.intercept_)` to
#: the binding, so the file carries the exact value predict uses.
_LINEAR_FORMAT = "mojolearn-linear-1"
_LOGISTIC_FORMAT = "mojolearn-logistic-1"


def _saved_mode(est):
    """The tier `predict` would run on now, persisted as GradientBoosting
    persists it, so a loaded model never silently changes tier."""
    mode = getattr(est, "numeric_mode", None) or _backend.default_mode()
    if not isinstance(mode, str) or mode.strip().lower() not in (
        "fast", "deterministic", "identical"
    ):
        raise ValueError(f"mojolearn: cannot save invalid numeric_mode {mode!r}")
    return mode.strip().lower()


def _restore_mode(obj, arrays):
    mode = _serialize.scalar_str(arrays, "numeric_mode")
    if mode not in ("fast", "deterministic", "identical"):
        raise ValueError(f"mojolearn: invalid saved numeric_mode {mode!r}")
    obj.numeric_mode = mode


def _check_saved_by(arrays, path, cls):
    """The file's `estimator` must be `cls` or a base of it, so the host
    subclasses of `_classical_host.py` load the plain class's file."""
    saved_as = _serialize.scalar_str(arrays, "estimator")
    if saved_as not in (c.__name__ for c in cls.__mro__):
        raise ValueError(
            f"mojolearn: {path!r} was saved by {saved_as}, not {cls.__name__}"
        )


def _save_linear(est, path, extra=None):
    """`LinearRegression.save` and `Ridge.save`: `coef_` `<f4`, the
    intercept `<f8`, `meta` `<i8` [n_features_in_, fit_intercept]."""
    if not hasattr(est, "coef_"):
        raise RuntimeError("this estimator is not fitted yet")
    arrays = {
        "format": _LINEAR_FORMAT,
        "estimator": type(est).__name__,
        "numeric_mode": _saved_mode(est),
        "coef": est.coef_,
        "intercept": Array.from_list([float(est.intercept_)], "<f8"),
        "meta": Array.from_list(
            [int(est.n_features_in_), 1 if est.fit_intercept else 0], "<i8"
        ),
    }
    if extra:
        arrays.update(extra)
    return _serialize.write_npz(path, arrays)


def _load_linear(cls, path, arrays=None, **kwargs):
    # Ridge has already read the archive to validate its alpha member. Reuse
    # that exact decoded mapping instead of reopening and materializing every
    # member a second time. LinearRegression still enters through the normal
    # one-read path.
    if arrays is None:
        arrays = _serialize.read_npz(path, _LINEAR_FORMAT)
    _check_saved_by(arrays, path, cls)
    meta = _serialize.exact(arrays, "meta", "<i8")
    if meta.size != 2:
        raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, 2 are needed")
    obj = cls(fit_intercept=bool(int(meta[1])), **kwargs)
    _restore_mode(obj, arrays)
    obj.coef_ = _serialize.exact(arrays, "coef", "<f4")
    obj.n_features_in_ = int(meta[0])
    if obj.coef_.ndim != 1 or obj.coef_.size != obj.n_features_in_:
        raise ValueError(f"mojolearn: {path!r} coef does not match n_features_in_")
    intercept = _serialize.exact(arrays, "intercept", "<f8")
    if intercept.size != 1:
        raise ValueError(f"mojolearn: {path!r} intercept must hold one value")
    obj.intercept_ = float(intercept[0])
    return obj, arrays


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


# The centering of LinearRegression and Ridge (lane hr-small-passes,
# 2026-10-02) runs in the estimators binding: `lm_col_sums`, `lm_center`
# and `lm_scale_rows`, on the device on a GPU install
# (glm/impl/center_device.mojo) and over the same items on the CPU column
# (glm/host/center_host.mojo), so every tier writes the same words. `b` is
# the estimator's `_bind("_mojolearn_estimators")`.


def _col_sums(b, x, rows, cols):
    """The column sums of a C-contiguous float32 `[rows, cols]` buffer: each
    the EXACT sum rounded once to float64 (round to nearest even), so the
    blocked device fold and the host pass agree on every vendor
    (glm/impl/center_items.mojo). Returns a Python list of `cols` floats."""
    out = empty((cols,), "<f8")
    b.lm_col_sums(addr_ro(x, name="X"), addr(out, name="column sums"),
                  [int(rows), int(cols)])
    return out.tolist()


def _column_means_f64(b, x, rows, cols):
    """Per-column float64 means: `_col_sums` divided by `rows` (one binary64
    division). A 1-D vector is a `[rows, 1]` matrix here."""
    return [v / rows for v in _col_sums(b, x, rows, cols)]


def _weight_total(b, weights):
    """`sum(w)`: the exact sum rounded once to float64 (`_col_sums` over a
    `[rows, 1]` view)."""
    total = _col_sums(b, weights, weights.shape[0], 1)[0]
    if total <= 0.0:
        raise ValueError(
            "mojolearn: sample_weight sums to zero, so the weighted mean "
            "cuML's preProcessData forms is a division by zero"
        )
    return total


def _column_means(b, x, weights):
    """Column means in float64, narrowed to float32 -- weighted when
    `weights` is not None. Returns a Python list of float32-valued floats.

    Unweighted: `_column_means_f64`. Weighted it is cuML's
    `raft::stats::weightedMean`, `sum_i w_i x_ij / sum_i w_i`, in THIS
    ORDER: (1) `wx_ij = fl32(x_ij * w_i)` (`lm_scale_rows`); (2) the
    column means of `wx`; (3) `mean_j * rows / total` in float64, `total`
    the weights' `_weight_total`; (4) one narrowing to float32. Theirs
    divides by the SUM OF THE WEIGHTS and not by the row count, so a
    uniform weight of 2 leaves the mean unchanged.
    """
    rows, cols = x.shape
    if weights is None:
        mu = _column_means_f64(b, x, rows, cols)
    else:
        total = _weight_total(b, weights)
        wx = _helper_output(x.shape, rows * cols)
        b.lm_scale_rows(addr_ro(x, name="X"), addr_ro(weights, name="sample_weight"),
                        addr(wx, name="weighted X"), [int(rows), int(cols)])
        mu = [m * rows / total for m in _column_means_f64(b, wx, rows, cols)]
    return [_round_f32(m) for m in mu]


def _vector_mean(b, v, weights):
    """The scalar float64 mean of the target, weighted when `weights` is
    not None: the 1-D half of `_column_means`."""
    n = v.shape[0]
    if weights is None:
        return _column_means_f64(b, v, n, 1)[0]
    total = _weight_total(b, weights)
    wy = _helper_output(v.shape, n)
    b.lm_scale_rows(addr_ro(v, name="y"), addr_ro(weights, name="sample_weight"),
                    addr(wy, name="weighted y"), [int(n), 1])
    return _column_means_f64(b, wy, n, 1)[0] * n / total


def _dims(x):
    """`(rows, cols)` of a 2-D Array, or `(rows, 1)` of a 1-D one: a vector
    is a `[rows, 1]` matrix to every helper in this file."""
    return (x.shape[0], x.shape[1]) if x.ndim == 2 else (x.shape[0], 1)


def _helper_output(shape, size):
    """DEVIATION 2632: the float32 destination of an op that writes EVERY
    element (`lm_center`, `lm_scale_rows`), taken from
    `_buffer._output_store`'s uninitialized raw allocation instead of
    `empty`, whose zero fill cost 105 to 121 ms of a 4,000,000 x 11
    LinearRegression fit on the H100 pod. No byte of the result comes from
    the allocation, so no bit moves."""
    from ._buffer import _output_store
    return Array._owned(_output_store("f", size), tuple(shape), "<f4", "C")


def _center(b, x, mu32):
    """`x - mu` in float32 per column (`lm_center`): one binary32
    subtraction per cell, operands and result flushed to signed zero when
    subnormal; `x` is `[rows, cols]` or a vector with `cols == 1`, `mu32` a
    list of `cols` float32-valued floats."""
    rows, cols = _dims(x)
    mean = Array.from_list([float(m) for m in mu32], "<f4")
    out = _helper_output(x.shape, rows * cols)
    b.lm_center(addr_ro(x, name="X"), addr_ro(mean, name="column means"),
                addr(out, name="centered X"), [int(rows), int(cols)])
    return out


def _shift(b, v, mu32):
    """The 1-D `_center`: `fl32(v_i - mu)`, `mu` already a float32 value."""
    return _center(b, v, [mu32])


def _scale_rows(b, x, root):
    """Row `i` of `x` times `root[i]` in float32 (`lm_scale_rows`): one
    binary32 multiplication per cell, operands and result flushed when
    subnormal; `root` a list of `rows` float32-valued floats."""
    rows, cols = _dims(x)
    w = Array.from_list([float(r) for r in root], "<f4")
    out = _helper_output(x.shape, rows * cols)
    b.lm_scale_rows(addr_ro(x, name="X"), addr_ro(w, name="sqrt weights"),
                    addr(out, name="scaled X"), [int(rows), int(cols)])
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


def _ols_tsqr_on(rows, cols):
    """Whether `LinearRegression.fit` takes the blocked TSQR: a tall design
    (rows >= cols + 1, cols + 1 <= 512) and MOJOLEARN_LINALG_TSQR not 0."""
    from ._expansion_decomp import _tsqr_lstsq_on
    return _tsqr_lstsq_on(rows, cols, 1)


def _ols_normal_eq_default(b):
    """Whether this binding's build routes LinearRegression.fit to the
    equilibrated normal equations instead of the TSQR (lane
    apple-fast-olsne: FAST on Apple, the comptime OLS_FAST_NORMAL_EQ read
    back through `ols_normal_eq_default`; False on a binding without it)."""
    q = _optional_export(b, "ols_normal_eq_default")
    return bool(q()) if q is not None else False


def _optional_export(b, name):
    """`name` from binding `b`, or None when it does not export it. A host
    binding's stand-in raises ImportError (by name) for a missing export
    rather than AttributeError, so a plain getattr default never applies on a
    CPU-only install, and LinearRegression.fit raised there."""
    try:
        return getattr(b, name, None)
    except ImportError:
        return None


def _ols_tsqr(x, y, rows, cols, mode):
    """coef_ (float32, cols) of min ||x w - y|| (x and y already centered and
    weighted as `fit` prepares them) through the blocked TSQR of [x | y]
    (x_decomp/tsqr_core.mojo, lane neural-pass140): R_aug = [[R, Q^T y],
    [0, rho]] in one pass over the rows, then the minimum-norm solution from
    the SVD of the small R (`_expansion_decomp._tsqr_lstsq_core`). No Gram
    matrix, so the condition number is not squared. A singular value at or
    below cols * eps32 * s_max is dropped (the dependent directions of a
    rank-deficient design, a constant column centered to zero among them,
    get no weight: the minimum-norm solution). The columns are equilibrated
    first as the normal equations route does (DEVIATION 2620, lane
    apple-fast-tsqr): R's column j is scaled by the exact power of two that
    rule picks for ||x e_j||^2 before the SVD and coef_ is scaled by it
    after, so a column's units do not move the cutoff (istella's columns
    span seven orders of magnitude; unequilibrated, real directions fell
    under it). DEVIATION 2621 still differs: the cutoff is on singular
    values, not on squared ones."""
    from ._expansion_decomp import _F32_EPS, _Kit, _mode, _tsqr_lstsq_core
    k = _Kit(_mode(mode))
    X, _, _, _ = _tsqr_lstsq_core(k, x, y, rows, cols, 1, _F32_EPS * cols, equilibrate=True)
    return X.out((cols,))


def _ols_fast_tsqr_r(r_aug, cols, mode):
    """coef_ (float32, cols) from the TSQR's R_aug ((cols + 1) square) that
    `ols_center_tsqr_r` formed on the device from the raw X and y
    (-D MOJOLEARN_OLS_FAST_DEVICE_CENTER, lane/apple-fast-core; FAST + Apple
    only). The small-R solve is `_expansion_decomp._tsqr_lstsq_core`'s after
    its `x_decomp_tsqr_r` call, line for line, with nrhs = 1 and
    `_ols_tsqr`'s cutoff `_F32_EPS * cols`: repeated here rather than split
    out of that shared function so the switch touches only this FAST-only
    branch; when it wins, `_tsqr_lstsq_core` gains the R entry and this goes."""
    from ._expansion_decomp import _F32_EPS, _Kit, _M, _f32, _mode
    from ._linalg_impl import _svd_tall
    k = _Kit(_mode(mode))
    n = cols + 1
    top = r_aug.rows(0, cols)
    R, C = top.cols(0, cols), top.cols(cols, n)
    Ur, S, Vt = _svd_tall(k, R, False)
    cut = _f32(S.s[0] * (_F32_EPS * cols))
    inv = k.ew("recip", k.ew("select", S, S, _M.zeros(1, 1), s=cut))
    X = k.mm(Vt, k.ew("mul", k.mm(Ur, C, ta=True), inv.T), ta=True)
    return X.out((cols,))


class LinearRegression(NumericModeMixin):
    """Ordinary least squares on the GPU.

    THE DEFAULT ROUTE (lane neural-pass140, 2026-10-02) is the blocked TSQR
    of [X | y] and the SVD of its small R (`_ols_tsqr`: no Gram matrix, the
    condition number not squared) for every tall design with at most 511
    features; the centering, weights and intercept below are unchanged.
    MOJOLEARN_LINALG_TSQR=0 (and any design the TSQR does not take) keeps
    the normal-equations solver this docstring describes from here on.

    This is the eigendecomposition solver (reference: cuML's `algorithm='eig'`, `lstsqEig`, RAFT), which forms
    ``X.T @ X`` and so squares the condition number. It is less robust than
    an SVD-based solver and should not be used for badly conditioned
    designs; cuML's SVD and QR solvers (`lstsqSvdJacobi`, `lstsqSvdQR`,
    `lstsqQR`) are not written here (glm/NOT_IMPLEMENTED.tsv). A design with
    more features than samples takes a second Gram route instead of an SVD;
    see the `n_features > n` row below. On the tall route the Gram matrix is
    equilibrated by exact power-of-two column scales and its pseudo-inverse
    keeps eigenvalues above ``n_features * eps32 * max`` (DEVIATIONS 2620,
    2621), so a column's units no longer change the model's rank. The wide
    route does the same to ``X @ X.T`` with row scales and an
    ``n * eps32 * max`` cutoff (DEVIATION 2622).

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
    (glm/impl/ols.mojo). This class therefore does the centering itself:
    column means and the y mean in float64 from exact column sums
    (`lm_col_sums`, rounded once; lane hr-small-passes), subtracted in
    float32 (`lm_center`), both on the device on a GPU install,
    and the intercept as `mean(y) - sum(mu_X * coef)` with `math.fsum`
    (exactly rounded; NO BLAS dot, which would be a platform-dependent host
    reduction -- E2's first finding). The device sees a centered design;
    the arithmetic that reaches it is a function of the inputs alone. The
    mean-centering is mathematically what cuML does, but its sums are
    exact where theirs accumulate in float32, so `coef_` CAN differ from
    cuML's in the last bits on ill-conditioned data.
    """

    #: scikit-learn's estimator kind: `cross_val_score` stratifies a
    #: classifier's default folds, as scikit-learn's does.
    _estimator_type = "regressor"

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
        b = self._bind("_mojolearn_estimators")
        fast_ne = _ols_normal_eq_default(b)
        normal_eq = not _ols_tsqr_on(rows, cols) or fast_ne
        resident = _optional_export(b, "ols_fit_resident")
        if fast_ne and weights is None and resident is not None:
            # lane apple-fast-olsne: FAST Apple builds only (the binding's
            # compiled OLS_FAST_NORMAL_EQ): the normal equations with X and y
            # uploaded once. Every other build keeps main's route below.
            self.coef_ = empty((cols,), "<f4")
            mu = empty((cols,), "<f4")
            ymean = empty((1,), "<f8")
            resident(addr_ro(x, name="X"), addr_ro(target, name="y"),
                     addr(self.coef_, name="coef_"), addr(mu, name="column means"),
                     addr(ymean, name="y mean"),
                     [rows, cols, 1 if self.fit_intercept else 0])
            if self.fit_intercept:
                self._x_mean = mu
                self._y_mean = float(ymean.tolist()[0])
            else:
                self._x_mean = zeros((cols,), "<f4")
                self._y_mean = 0.0
            self._set_intercept(cols)
            return self
        if self.fit_intercept and weights is None and self._fast_device_center():
            # -D MOJOLEARN_OLS_FAST_DEVICE_CENTER (lane/apple-fast-core,
            # 2026-10-02, FAST + Apple only, default off): the three device
            # trips below (`_column_means` -> `lm_col_sums`, `_center` ->
            # `lm_center`, then the solve's own upload of the centered copy)
            # become ONE upload of the raw X and y with the centering on the
            # device (`glm/estimator.mojo`). The solve is the same one this
            # layer would pick: the TSQR's R comes back and `_ols_fast_tsqr_r`
            # finishes it exactly as `_ols_tsqr` does; off the TSQR it is
            # `ols_fit` on the device-centered data. The intercept is formed
            # here as below from the means the device hands back.
            b = self._bind("_mojolearn_estimators")
            means = empty((cols,), "<f4")
            if _ols_tsqr_on(rows, cols):
                from ._expansion_decomp import _M
                n = cols + 1
                r_aug = _M.zeros(n, n)
                y_mean = b.ols_center_tsqr_r(
                    addr_ro(x, name="X"), addr_ro(target, name="y"),
                    r_aug.addr, addr(means, name="means"), [rows, cols],
                )
                self.coef_ = _ols_fast_tsqr_r(r_aug, cols, getattr(self, "numeric_mode", None))
            else:
                self.coef_ = empty((cols,), "<f4")
                y_mean = b.ols_fit_centered(
                    addr_ro(x, name="X"), addr_ro(target, name="y"),
                    addr(self.coef_, name="coef_"), addr(means, name="means"),
                    [rows, cols],
                )
            self._x_mean = means
            self._y_mean = float(y_mean)
            dot = math.fsum(
                float(a) * float(b)
                for a, b in zip(self._x_mean.tolist(), self.coef_.tolist())
            )
            self.intercept_ = float(self._y_mean - dot)
            self.n_features_in_ = cols
            return self
        if self.fit_intercept:
            # float64 column means -> float32, then a float32 subtraction.
            # The means come from exact column sums rounded once to float64
            # (`lm_col_sums`, lane hr-small-passes); the 1-D target mean is
            # the same op over a [rows, 1] view. Neither goes through BLAS or libm. The dot
            # below is the one place a BLAS call would have slipped in, so
            # it is an exactly-rounded fsum instead.
            #
            # WITH WEIGHTS THE MEANS ARE WEIGHTED, which is cuML's
            # preProcessData (`raft::stats::weightedMean`, sum(w*x)/sum(w),
            # preprocess.cuh:95-97 and 110-112) and NOT the unweighted mean
            # applied to weighted rows. The two differ, and using the wrong
            # one puts the intercept in the wrong place without moving any
            # coefficient enough to notice.
            b = self._bind("_mojolearn_estimators")
            mu32 = _column_means(b, x, weights)
            self._x_mean = Array.from_list(mu32, "<f4")
            self._y_mean = _vector_mean(b, target, weights)
            # NumPy narrowed the Python-float y mean to float32 BEFORE the
            # float32 subtract (value-based / weak-scalar casting); the
            # same order here so the centered bits are the same bits.
            work_x = _center(b, x, mu32)
            work_y = _shift(b, target, _round_f32(self._y_mean))
        else:
            work_x, work_y = x, target
            self._x_mean = zeros((cols,), "<f4")
            self._y_mean = 0.0
        if weights is not None:
            # `olsFit`, ols.cuh:99-110, on the host. See SAMPLE WEIGHTS in
            # the class docstring for why this is here and not in the Mojo
            # layer, and for the bit-for-bit claim the gate checks.
            root = [_round_f32(math.sqrt(v)) for v in weights.tolist()]
            b = self._bind("_mojolearn_estimators")
            work_x = _scale_rows(b, work_x, root)
            work_y = _scale_rows(b, work_y, root)
        if not normal_eq:
            # lane neural-pass140: the blocked TSQR of [X | y] and the SVD of
            # its small R (_ols_tsqr); MOJOLEARN_LINALG_TSQR=0 (and FAST on
            # Apple, `_ols_normal_eq_default`) keeps the normal equations below
            self.coef_ = _ols_tsqr(work_x, work_y, rows, cols, getattr(self, "numeric_mode", None))
        else:
            self.coef_ = empty((cols,), "<f4")
            self._bind("_mojolearn_estimators").ols_fit(
                addr_ro(work_x, name="X"), addr_ro(work_y, name="y"),
                addr(self.coef_, name="coef_"),
                [rows, cols],
            )
        self._set_intercept(cols)
        return self

    def _set_intercept(self, cols):
        if self.fit_intercept:
            dot = math.fsum(
                float(a) * float(b)
                for a, b in zip(self._x_mean.tolist(), self.coef_.tolist())
            )
            self.intercept_ = float(self._y_mean - dot)
        else:
            self.intercept_ = 0.0
        self.n_features_in_ = cols

    def _fast_device_center(self):
        """Whether `fit` takes the device-centering entries: the loaded
        binding registers `ols_center_tsqr_r` only when it was built FAST
        on Apple with -D MOJOLEARN_OLS_FAST_DEVICE_CENTER
        (`bindings/_mojolearn_estimators.mojo`), so the name's presence is
        the whole switch: no env read, and the host binding has not got it
        (its proxy raises ImportError for a missing name)."""
        try:
            return getattr(self._bind("_mojolearn_estimators"), "ols_center_tsqr_r", None) is not None
        except (AttributeError, ImportError):
            return False

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

    def save(self, path):
        """Write the fitted model to `path` as an npz: `coef_` as fitted,
        the intercept as float64, the feature count and `fit_intercept`.
        What `predict` reads and nothing else (the classical host inference
        lane, 2026-09-13); `mojolearn.host_model(path)` predicts from it on
        a CPU with no GPU."""
        return _save_linear(self, path)

    @classmethod
    def load(cls, path):
        """Load a model saved by `save`. The result predicts; it does not
        refit and carries no training-time means."""
        obj, _ = _load_linear(cls, path)
        return obj


class Ridge(NumericModeMixin):
    """l2-regularized least squares on the GPU, cuML's `solver='eig'` arm.

    Reference: `cuml/python/cuml/linear_model/ridge.pyx` and
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

    #: scikit-learn's estimator kind: `cross_val_score` stratifies a
    #: classifier's default folds, as scikit-learn's does.
    _estimator_type = "regressor"

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
        b = self._bind("_mojolearn_estimators")
        q = getattr(b, "ridge_resident_default", None)
        resident = getattr(b, "ridge_fit_resident", None)
        use_resident = q is not None and resident is not None and bool(q())
        if use_resident:
            # lane apple-fast-ridgespeed: FAST Apple builds (default unless
            # -D MOJOLEARN_RIDGE_RESIDENT_OFF): X and y uploaded once, the same
            # column sums, center and ridgeEig on the resident buffers (the
            # same words as the route below).
            self.coef_ = empty((cols,), "<f4")
            mu = empty((cols,), "<f4")
            ymean = empty((1,), "<f8")
            resident(addr_ro(x, name="X"), addr_ro(target, name="y"),
                     addr(self.coef_, name="coef_"), addr(mu, name="column means"),
                     addr(ymean, name="y mean"),
                     [rows, cols, float(self.alpha), 1 if self.fit_intercept else 0])
            if self.fit_intercept:
                self._x_mean = mu
                self._y_mean = float(ymean.tolist()[0])
            else:
                self._x_mean = zeros((cols,), "<f4")
                self._y_mean = 0.0
        elif self.fit_intercept:
            # The same centering as LinearRegression, for the same
            # reasons; read that class's fit.
            mu32 = _column_means(b, x, None)
            self._x_mean = Array.from_list(mu32, "<f4")
            self._y_mean = _vector_mean(b, target, None)
            work_x = _center(b, x, mu32)
            work_y = _shift(b, target, _round_f32(self._y_mean))
        else:
            work_x, work_y = x, target
            self._x_mean = zeros((cols,), "<f4")
            self._y_mean = 0.0
        if not use_resident:
            self.coef_ = empty((cols,), "<f4")
            b.ridge_fit(
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
        # (`gemmPredict` in the reference serves both, `base.pyx:134`).
        self._bind("_mojolearn_estimators").ols_predict(
            addr_ro(x, name="X"), addr_ro(self.coef_, name="coef_"),
            addr(out, name="predictions"),
            [x.shape[0], x.shape[1], float(self.intercept_)],
        )
        return out

    def score(self, X, y):
        """R^2, a sequential float64 host reduction (DEVIATION 2365)."""
        return _r2_host(self.predict(X), y)

    def save(self, path):
        """As `LinearRegression.save`, plus `alpha` as float64; the two
        classes share one file format because they share one predict."""
        return _save_linear(
            self, path, {"alpha": Array.from_list([float(self.alpha)], "<f8")}
        )

    @classmethod
    def load(cls, path):
        """Load a model saved by `save`. The result predicts; it does not
        refit. `solver_` is the eig arm, the only one `fit` runs."""
        arrays = _serialize.read_npz(path, _LINEAR_FORMAT)
        alpha = _serialize.exact(arrays, "alpha", "<f8")
        if alpha.size != 1:
            raise ValueError(f"mojolearn: {path!r} alpha must hold one value")
        obj, _ = _load_linear(cls, path, arrays=arrays, alpha=float(alpha[0]))
        obj.solver_ = "eig"
        return obj


# cuML's `qn_params.loss` ids (cuml/linear_model/qn.h); the Python door maps
# 'sigmoid' to QN_LOSS_LOGISTIC and 'softmax' to QN_LOSS_SOFTMAX.
_QN_OPT_RETCODE = {0: "OPT_SUCCESS", 1: "OPT_NUMERIC_ERROR",
                   2: "OPT_LS_FAILED", 3: "OPT_MAX_ITERS_REACHED",
                   4: "OPT_INVALID_ARGS"}


#: `glm/impl/linear_model/qn.mojo`'s ids, the 14th `qn_fit` field.
_QN_LOSS_LOGISTIC = 0
_QN_LOSS_SQUARED = 1
_QN_LOSS_SOFTMAX = 2
_QN_LOSS_SVC_L1 = 3
_QN_LOSS_SVC_L2 = 4
_QN_LOSS_SVR_L1 = 5
_QN_LOSS_SVR_L2 = 6
_QN_LOSS_ABS = 7


def _coef_from_w(w, cols, n_targets, fit_intercept):
    """`coef_` and `intercept_` from the fitted `W` block. One target: the
    first `cols` entries and the last (a slice of an Array COPIES, the
    _array contract, so both are detached from `_w`). C targets: cuML's
    column-major `w[c + C*j]`, the bias column at `j == cols`, unpacked by
    an O(C * n_features) Python loop over the parameters, not the rows."""
    if n_targets == 1:
        coef = w[:cols].reshape((1, cols))
        intercept = w[cols:cols + 1] if fit_intercept else zeros((1,), "<f4")
        return coef, intercept
    values = w.tolist()
    coef = Array.from_list(
        [[values[c + n_targets * j] for j in range(cols)] for c in range(n_targets)],
        "<f4",
    )
    intercept = (Array.from_list([values[c + n_targets * cols] for c in range(n_targets)], "<f4")
                 if fit_intercept else zeros((n_targets,), "<f4"))
    return coef, intercept


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

    Reference: `cuml/python/cuml/linear_model/logistic_regression.py`,
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
                                  a WARNING in the reference and `n_iter_ ==
                                  max_iter` with `retcode_ == 3` here
        linesearch_max_iter honored  default 50 as theirs
        class_weight    refused   becomes a sample_weight in the reference
                                  (logistic_regression.py:400-436), and
                                  sample_weight is not implemented
        sample_weight   refused   GLMBase::add_sample_weights and the
                                  weighted getLossAndDZ arm, not implemented
        l1_ratio        honored, and REQUIRED with penalty='elasticnet'
                                  (logistic_regression.py:310-316): the
                                  split is l1 = l1_ratio / C,
                                  l2 = (1 - l1_ratio) / C
        solver          'qn' only the only value cuML accepts either
        > 2 classes     honored   since lane/logistic-multiclass
                                  (2026-09-14): the softmax loss
                                  (QN_LOSS_SOFTMAX, glm/impl/qn/
                                  glm_softmax.mojo, DEVIATIONS 705-711,
                                  gated by `pixi run check-glm-multinomial`)
                                  through the same L-BFGS; `coef_` is
                                  (C, n_features), `intercept_` (C,),
                                  `decision_function` (n, C) float32,
                                  `predict_proba` the softmax of each row
                                  in float64 on the host (`qn_softmax`,
                                  the rule of DEVIATION 549), `predict`
                                  the row argmax with the FIRST maximum
                                  winning a tie (the lowest class index,
                                  `_labels.argmax_rows`, the positional
                                  rule of `softmax_row_max`). REFUSED by
                                  name with more than two classes: an l1
                                  or elasticnet penalty (OWL-QN on the
                                  softmax objective runs but has no
                                  identity gate; CONTRIBUTING.md (Non-default paths))
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

    OUTPUTS: `coef_` (1, n_features) float32, `intercept_` (1,) float32
    (with C > 2 classes: (C, n_features) and (C,), row c the class
    `classes_[c]`, from cuML's column-major `W` block `w[c + C*j]`),
    `classes_` (the labels, sorted, a Python LIST -- DEVIATION 2364),
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

    #: scikit-learn's estimator kind: `cross_val_score` stratifies a
    #: classifier's default folds, as scikit-learn's does.
    _estimator_type = "classifier"

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
        if len(_shape_of(y)) != 1:
            raise ValueError("mojolearn LogisticRegression requires a 1-D y")
        # LabelEncoder: sorted unique classes -> 0..k-1
        # (logistic_regression.py:383), under the package-wide ORDER RULE
        # (DEVIATION 2340) through `encode_labels`: the native encoder for a
        # numeric buffer (DEVIATION 2500, as the forests take it), the
        # `sorted_classes` routine for everything else. Same classes, same
        # codes (`tests/test_labels_native.py`).
        self.classes_, codes = encode_labels(y)
        if len(codes) != rows:
            raise ValueError("mojolearn LogisticRegression X and y lengths differ")
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("mojolearn LogisticRegression: y has one class")
        # `qn.cuh:106`: one target for two classes (the logistic loss, code
        # 1 is `classes_[1]`, the class the solver maps to +1), C targets
        # for C > 2 (the softmax loss, code c is `classes_[c]`).
        n_targets = 1 if n_classes == 2 else n_classes
        loss = _QN_LOSS_LOGISTIC if n_targets == 1 else _QN_LOSS_SOFTMAX
        # Label encoding, the permitted O(rows) Python loop.
        y_enc = codes.astype("<f4")  # dense codes < 2**24: exact
        l1, l2 = self._get_qn_params()
        if n_targets > 1 and l1 != 0.0:
            raise NotImplementedError(
                f"mojolearn LogisticRegression: penalty={self.penalty!r} with "
                f"{n_classes} classes selects OWL-QN on the softmax objective, "
                "which runs but has no identity gate (glm/checks/"
                "multinomial_check.mojo gates l2 and no penalty); refused by "
                "name until it is gated (CONTRIBUTING.md (Non-default paths))"
            )
        n_param = cols + (1 if self.fit_intercept else 0)
        w = zeros((n_param * n_targets,), "<f4")
        info = zeros((2,), "<f4")
        n_iter = self._bind("_mojolearn_estimators").qn_fit(
            addr_ro(x, name="X"), addr_ro(y_enc, name="y"),
            addr(w, name="coef_"), addr(info, name="info"),
            [rows, cols, n_classes,
             float(l1), float(l2), float(self.tol), float(self.tol * 0.01),
             int(self.max_iter), int(self.linesearch_max_iter),
             int(self.lbfgs_memory), 1 if self.fit_intercept else 0,
             1 if self.penalty_normalized else 0, 0, loss],
        )
        self._w = w
        self.coef_, self.intercept_ = _coef_from_w(w, cols, n_targets, self.fit_intercept)
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
        n_targets = self._n_targets()
        if n_targets == 1:
            out = empty((x.shape[0],), "<f4")
            self._bind("_mojolearn_estimators").qn_decision_function(
                addr_ro(x, name="X"), addr_ro(self._w, name="coef_"),
                addr(out, name="scores"),
                [x.shape[0], x.shape[1], 1 if self.fit_intercept else 0],
            )
            return out
        # C > 2: `(n, C)` float32, `scores[i, c]` the class-c logit, the
        # 4-field call of the same entry (lane/logistic-multiclass).
        out = empty((x.shape[0], n_targets), "<f4")
        self._bind("_mojolearn_estimators").qn_decision_function(
            addr_ro(x, name="X"), addr_ro(self._w, name="coef_"),
            addr(out, name="scores"),
            [x.shape[0], x.shape[1], 1 if self.fit_intercept else 0, n_targets],
        )
        return out

    def _n_targets(self):
        """`qn.cuh:106`: 1 for two classes, C for C > 2."""
        n_classes = len(self.classes_)
        return 1 if n_classes == 2 else n_classes

    def predict(self, X):
        """`qn_predict`: `z > 0 ? 1 : 0` (qn.cuh:276), mapped to classes_.
        An int64 / float64 Array for int / float labels, a Python list for
        anything else (DEVIATION 2364). With C > 2 classes the row argmax
        of the decision function, the FIRST maximum winning a tie (the
        lowest class index, `_labels.argmax_rows`; the same rule the
        device's `softmax_row_max` applies to a tie and to `+0.0` against
        `-0.0`, and cuML's `qn_predict` argmax over `C` scores)."""
        if self._n_targets() == 1:
            x, _ = as_f32_c(X, ndim=2, name="X")
            codes = empty((x.shape[0],), "<i8")
            binding = self._bind("_mojolearn_estimators")
            native = getattr(binding, "qn_predict_binary", None)
            if native is not None:
                native(
                    addr_ro(x, name="X"), addr_ro(self._w, name="coef_"),
                    addr(codes, name="codes"),
                    [x.shape[0], x.shape[1], 1 if self.fit_intercept else 0],
                )
                return decode_labels(self.classes_, codes)
            scores = self.decision_function(x)
            return decode_labels(self.classes_, threshold_codes(scores))
        scores = self.decision_function(X)
        return decode_labels(self.classes_, argmax_rows(scores))

    def predict_proba(self, X):
        scores = self.decision_function(X)
        n_targets = self._n_targets()
        if n_targets == 1:
            out = empty((scores.shape[0], 2), "<f8")
            self._bind("_mojolearn_estimators").qn_sigmoid(
                addr_ro(scores, name="scores"), addr(out, name="proba"),
                [scores.shape[0]])
            return out
        # C > 2: the softmax of each row in float64 on the host through
        # `identical_exp64` (`qn_softmax_host`, the rule of DEVIATION 549).
        out = empty((scores.shape[0], n_targets), "<f8")
        self._bind("_mojolearn_estimators").qn_softmax(
            addr_ro(scores, name="scores"), addr(out, name="proba"),
            [scores.shape[0], n_targets])
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
            [[_log_or_inf(p) for p in row]
             for row in self.predict_proba(X).tolist()],
            "<f8",
        )

    def score(self, X, y):
        """Accuracy: the fraction of rows where `predict(X) == y`, a Python
        count over O(rows) labels (DEVIATION 2365, a host reduction)."""
        return _accuracy_host(self.predict(X), y)

    def save(self, path):
        """Write the fitted model to `path` as an npz: `_w` as fitted (the
        `n_targets * (n_features + fit_intercept)` float32 block inference
        reads; `coef_` and `intercept_` are its copies and are rebuilt by
        `load`), `classes_` as `_labels.classes_member` (its length is the
        class count, so a C > 2 model needs no extra field), the feature
        count and `fit_intercept` (the classical host inference lane,
        2026-09-13; C > 2 since lane/logistic-multiclass, 2026-09-14)."""
        if not hasattr(self, "_w"):
            raise RuntimeError("this estimator is not fitted yet")
        arrays = {
            "format": _LOGISTIC_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "w": self._w,
            "classes": classes_member(self.classes_),
            "meta": Array.from_list(
                [int(self.n_features_in_), 1 if self.fit_intercept else 0], "<i8"
            ),
        }
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a model saved by `save`. The result predicts; it does not
        refit and carries no `n_iter_`, `objective_` or `retcode_`."""
        arrays = _serialize.read_npz(path, _LOGISTIC_FORMAT)
        _check_saved_by(arrays, path, cls)
        meta = _serialize.exact(arrays, "meta", "<i8")
        if meta.size != 2:
            raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, 2 are needed")
        obj = cls(fit_intercept=bool(int(meta[1])))
        _restore_mode(obj, arrays)
        cols = int(meta[0])
        obj.classes_ = classes_from_member(arrays["classes"])
        if len(obj.classes_) < 2:
            raise ValueError(f"mojolearn: {path!r} must carry at least two classes")
        n_targets = obj._n_targets()
        w = _serialize.exact(arrays, "w", "<f4")
        if w.ndim != 1 or w.size != (cols + (1 if obj.fit_intercept else 0)) * n_targets:
            raise ValueError(f"mojolearn: {path!r} w does not match n_features_in_, fit_intercept and the class count")
        obj._w = w
        obj.coef_, obj.intercept_ = _coef_from_w(w, cols, n_targets, obj.fit_intercept)
        obj.n_features_in_ = cols
        return obj


def _qn_fit_one_target(est, x, y_enc, n_classes, loss, l1, l2, grad_tol,
                       change_tol, svr_eps=0.0):
    """One `qn_fit` call on a one-target loss (the 15-field form): the
    fitted `(n_features + fit_intercept,)` float32 block, `num_iters`, the
    objective and the OPT_RETCODE. `est` supplies the solver fields."""
    rows, cols = x.shape
    w = zeros((cols + (1 if est.fit_intercept else 0),), "<f4")
    info = zeros((2,), "<f4")
    n_iter = est._bind("_mojolearn_estimators").qn_fit(
        addr_ro(x, name="X"), addr_ro(y_enc, name="y"),
        addr(w, name="coef_"), addr(info, name="info"),
        [rows, cols, n_classes,
         float(l1), float(l2), float(grad_tol), float(change_tol),
         int(est.max_iter), int(est.linesearch_max_iter),
         int(est.lbfgs_memory), 1 if est.fit_intercept else 0,
         1 if est.penalty_normalized else 0, 0, int(loss), float(svr_eps)],
    )
    return w, int(n_iter), float(info[0]), int(info[1])


def _qn_scores(est, X, w, n_targets=1):
    """`qn_decision_function` on a fitted block: `(n,)` float32 for one
    target, `(n, n_targets)` for the column-major one-vs-rest block."""
    x, _ = as_f32_c(X, ndim=2, name="X")
    if x.shape[1] != est.n_features_in_:
        raise ValueError(f"mojolearn {type(est).__name__} feature count differs from fit")
    params = [x.shape[0], x.shape[1], 1 if est.fit_intercept else 0]
    shape = (x.shape[0],)
    if n_targets > 1:
        params.append(n_targets)
        shape = (x.shape[0], n_targets)
    out = empty(shape, "<f4")
    est._bind("_mojolearn_estimators").qn_decision_function(
        addr_ro(x, name="X"), addr_ro(w, name="coef_"),
        addr(out, name="scores"), params,
    )
    return out


def _check_qn_solver_fields(name, tol, max_iter, linesearch_max_iter, lbfgs_memory):
    if not tol > 0:
        raise ValueError(f"mojolearn {name}: tol must be positive, got {tol}")
    for field, value in (("max_iter", max_iter),
                         ("linesearch_max_iter", linesearch_max_iter),
                         ("lbfgs_memory", lbfgs_memory)):
        if int(value) != value or value < 1:
            raise ValueError(f"mojolearn {name}: {field} must be a positive integer, got {value}")


class QNRegressor(NumericModeMixin):
    """Linear regression on the squared or the absolute loss, solved by the
    quasi-Newton solver `LogisticRegression` uses.

    Reference: `cuml.solvers.QN` with `loss='l2'` (squared) and `loss='l1'`
    (absolute), `cuml/cpp/src/glm/qn/glm_linear.cuh`; the Mojo
    implementation is `glm/impl/qn/glm_linear.mojo` (DEVIATION 707). The
    objective is `mean_i lz(y_i, x_i w + b) + l1 ||w||_1 + (l2 / 2) ||w||^2`
    with `lz = (z - y)^2 / 2` or `|z - y|`, both strengths divided by `n`
    when `penalty_normalized`. `l1_strength != 0` selects OWL-QN, as it
    does for `LogisticRegression`. The intercept is a solver parameter and
    is not penalized.

        loss            'squared_error' or 'absolute_error'
        l1_strength, l2_strength   honored, non-negative
        fit_intercept, max_iter, tol, delta, linesearch_max_iter,
        lbfgs_memory, penalty_normalized   honored; grad_tol = tol,
                        change_tol = delta if given else tol * 0.01
                        (qn.pyx:504-506)
        warm_start      refused   w0 = 0 always
        sample_weight   refused   not implemented (glm/NOT_IMPLEMENTED.tsv)

    OUTPUTS: `coef_` (n_features,) float32, `intercept_` a float,
    `n_iter_`, `objective_`, `retcode_` as on `LogisticRegression`.
    """

    #: scikit-learn's estimator kind: `cross_val_score` stratifies a
    #: classifier's default folds, as scikit-learn's does.
    _estimator_type = "regressor"

    _BINDING = "_mojolearn_estimators"
    _LOSSES = {"squared_error": _QN_LOSS_SQUARED, "absolute_error": _QN_LOSS_ABS}

    def __init__(self, *, loss="squared_error", fit_intercept=True,
                 l1_strength=0.0, l2_strength=0.0, max_iter=1000, tol=1e-4,
                 delta=None, linesearch_max_iter=50, lbfgs_memory=5,
                 warm_start=False, penalty_normalized=True):
        if loss not in self._LOSSES:
            raise ValueError(
                f"Expected loss to be one of {list(self._LOSSES)}, got {loss!r}")
        if warm_start:
            raise NotImplementedError(
                "mojolearn QNRegressor: warm_start is not implemented (the "
                "solver starts from w0 = 0)")
        if l1_strength < 0 or l2_strength < 0:
            raise ValueError(
                "mojolearn QNRegressor: l1_strength and l2_strength must be "
                f"non-negative, got {l1_strength} and {l2_strength}")
        if delta is not None and delta < 0:
            raise ValueError(f"mojolearn QNRegressor: delta must be non-negative, got {delta}")
        _check_qn_solver_fields("QNRegressor", tol, max_iter, linesearch_max_iter, lbfgs_memory)
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength
        self.max_iter = max_iter
        self.tol = tol
        self.delta = delta
        self.linesearch_max_iter = linesearch_max_iter
        self.lbfgs_memory = lbfgs_memory
        self.warm_start = False
        self.penalty_normalized = penalty_normalized

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError(
                "mojolearn QNRegressor: sample_weight is not implemented "
                "(GLMBase::add_sample_weights, glm_base.cuh:115; "
                "glm/NOT_IMPLEMENTED.tsv)")
        x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
        rows, cols = x.shape
        t = _target_1d(y, rows, "mojolearn QNRegressor requires a 1-D y",
                       "mojolearn QNRegressor X and y lengths differ")
        change_tol = self.delta if self.delta is not None else self.tol * 0.01
        w, n_iter, self.objective_, self.retcode_ = _qn_fit_one_target(
            self, x, t, 1, self._LOSSES[self.loss], self.l1_strength,
            self.l2_strength, self.tol, change_tol)
        self._w = w
        self.coef_ = w[:cols]
        self.intercept_ = float(w[cols]) if self.fit_intercept else 0.0
        self.n_iter_ = Array.from_list([n_iter], "<i8")
        self.n_features_in_ = cols
        return self

    def predict(self, X):
        if not hasattr(self, "_w"):
            raise ValueError("mojolearn QNRegressor: call fit before predict")
        return _qn_scores(self, X, self._w)

    def score(self, X, y):
        """R^2 on the host, as `LinearRegression.score`."""
        return _r2_host(self.predict(X), y)
