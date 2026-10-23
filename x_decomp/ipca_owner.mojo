# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`IncrementalPCA`'s fit, partial_fit, transform and inverse_transform as
ONE native operation each (lane mca2-decomp-a, 2026-10-22): the steps
`_expansion_decomp.IncrementalPCA._partial`, `_gram_svd`, `transform` and
`inverse_transform` sequenced from Python with the decomp kit (`_Kit`), one
x_decomp call each.

SAME CALLS, SAME ORDER, SAME OPERANDS. Every step runs on the randomized PCA
kit (`RKit`, x_decomp/pca_randomized.mojo), which keeps each matrix where
the Python kit kept it and reaches every step through the typed body of the
entry the Python kit called, with the same broadcast modes and float32
scalars; the binary64 scalars (1 / total, seen / n, the sqrt of the mean
correction) are the same IEEE operations CPython performed. The Gram's
eigendecomposition is resident where `_Kit.eigh` was (an IDENTICAL GPU build
with `x_decomp_dev_eigh`, `IpcaDev.has_eigh`), else the host-address entry.

The running state between batches and calls is the public arrays (the
exact words the Python kit's matrices held); each step re-reads them, so a
partial_fit after a fit, or a transform after either, sees the same values
the resident matrices held."""
from std.math import sqrt
from std.memory import memcpy

from x_decomp.api import eigh_run
from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.pca_randomized import RDev, RKit

# x_decomp/cells.mojo op codes (`_OP` in _expansion_decomp.py)
comptime _OP_ADD = 0
comptime _OP_SUB = 1
comptime _OP_MUL = 2
comptime _OP_DIV = 3
comptime _OP_SQRT = 7
comptime _OP_SQ = 8
comptime _OP_SCALE = 14
comptime _OP_SQDIFF = 21
# x_decomp/moves.mojo PLACE_COLS (`_MV_PLACE_COLS`)
comptime _MV_PLACE_COLS = 2


trait IpcaDev:
    """The resident eigendecomposition the Python kit reached on a GPU
    binding (`x_decomp_dev_eigh`, registered only with IDN_EIGH_RESIDENT)."""

    @staticmethod
    def has_eigh() -> Bool:
        ...

    @staticmethod
    def eigh(a: Int, w: F32Ptr, v: F32Ptr, n: Int, uplo: Int) raises:
        ...


struct NoIpcaDev(IpcaDev):
    """The host binding: no resident eigendecomposition."""

    @staticmethod
    def has_eigh() -> Bool:
        return False

    @staticmethod
    def eigh(a: Int, w: F32Ptr, v: F32Ptr, n: Int, uplo: Int) raises:
        raise Error("x_decomp: no resident matrices on the host binding")


def _eigh[E: Exec, D: RDev, RES: Bool, X: IpcaDev](mut kit: RKit[E, D, RES], a: Int) raises -> Tuple[Int, Int]:
    """`_Kit.eigh(A)`: (w 1 x n ascending, v n x n eigenvectors in columns),
    both host words; a resident operand is solved on a device copy where
    the build exports `x_decomp_dev_eigh`."""
    var n = kit.r(a)
    var w = kit.zeros(1, n)
    var v = kit.zeros(n, n)
    var pw = kit.ptr(w)
    var pv = kit.ptr(v)
    var resident = False
    comptime if RES:
        resident = X.has_eigh() and n >= 1 and kit.c(a) == n and kit.mats[a].dev >= 0
    if resident:
        X.eigh(kit.mats[a].dev, pw, pv, n, 0)
    else:
        eigh_run[E](kit.ptr(a), pw, pv, n, 0)
    return (w, v)


def _reversed_columns[E: Exec, D: RDev, RES: Bool](mut kit: RKit[E, D, RES], a: Int) raises -> Int:
    """`A.take_cols(range(c - 1, -1, -1))` of a host matrix: one strided
    copy per column (data movement)."""
    var rows = kit.r(a)
    var cols = kit.c(a)
    var out = kit.zeros(rows, cols)
    var src = kit.ptr(a)
    var dst = kit.ptr(out)
    for t in range(cols):  # driver-loop: over the Gram's eigenpairs (model structure: n_features)
        var j = cols - 1 - t
        for i in range(rows):  # small-loop(rows of a features x features eigenvector matrix): one word each
            dst[i * cols + t] = src[i * cols + j]
    return out


def _stack_rows[E: Exec, D: RDev, RES: Bool](mut kit: RKit[E, D, RES], a: Int, b: Int, c: Int) raises -> Int:
    """`_Kit.vstack([a, b, c])`: the rows of each placed after the previous
    ones (PLACE_COLS moves on a GPU binding, copies on the host)."""
    var cols = kit.c(a)
    var ra = kit.r(a)
    var rb = kit.r(b)
    var rc = kit.r(c)
    var total = ra + rb + rc
    comptime if RES:
        if total * cols:
            var out = kit.device_output(total, cols)
            var one = kit.zeros(1, 1)
            var done = 0
            var parts = [a, b, c]
            for p in range(3):  # driver-loop: over the three stacked blocks
                var part = parts[p]
                var count = kit.r(part) * kit.c(part)
                if count:
                    var source = kit.device(part)
                    var index = kit.device(one)
                    D.move(source, index, kit.mats[out].dev, _MV_PLACE_COLS, count, cols, cols, done * cols,
                           0, 0, count, total * cols, 1)
                done += kit.r(part)
            return out
    var out = kit.zeros(total, cols)
    var dst = kit.ptr(out)
    var offset = 0
    var parts = [a, b, c]
    for p in range(3):  # driver-loop: over the three stacked blocks
        var part = parts[p]
        var count = kit.r(part) * kit.c(part)
        if count:
            memcpy(dest=dst.unsafe_offset(offset), src=kit.ptr(part), count=count)
        offset += count
    return out


def _gram_svd[E: Exec, D: RDev, RES: Bool, X: IpcaDev](mut kit: RKit[E, D, RES], z: Int) raises -> Tuple[Int, Int]:
    """`_gram_svd(k, Z)`: S descending (1 x d) and Vt (d x d) through the
    eigendecomposition of Z^T Z, then svd_flip(u_based_decision=False)."""
    var gram = kit.mm(z, z, ta=True)
    var eig = _eigh[E, D, RES, X](kit, gram)
    var w = _reversed_columns(kit, eig[0])
    var s = kit.ew(_OP_SQRT, w)
    var v = _reversed_columns(kit, eig[1])
    var vt = kit.transpose(v)
    var signs = kit.absmax_signs(vt, False)
    vt = kit.ew(_OP_MUL, vt, signs)
    return (s, vt)


def _words(ref words: List[Float32]) -> F32Ptr:
    return F32Ptr(unsafe_from_address=Int(words.unsafe_ptr()))


def ipca_partial_step[E: Exec, D: RDev, RES: Bool, X: IpcaDev](
    x: F32Ptr, n: Int, d: Int, requested: Int, seen: Int, prev_count: Int, prev_d: Int,
    prev_components: F32Ptr, prev_singular: F32Ptr, prev_mean: F32Ptr, prev_var: F32Ptr,
    components_out: F32Ptr, singular_out: F32Ptr, mean_out: F32Ptr, var_out: F32Ptr,
    variance_out: F32Ptr, ratio_out: F32Ptr,
) raises -> Tuple[Int, Float64]:
    """One `_partial(Xb)` of the n x d batch at x. `requested` is
    n_components (-1: None); `seen` the samples already fitted, with the
    previous state (prev_count components) read when seen > 0. Writes the
    published arrays (components first, at the kept count) and returns
    (n_components_, noise_variance_). `prev_d` is the state's feature count.
    The outputs may not alias the state."""
    var has_prev = seen > 0
    var nc: Int
    if requested < 0:
        nc = min(n, d) if not has_prev else prev_count
    else:
        nc = requested
        if nc < 1 or nc > d:
            raise Error("n_components=" + String(nc) + " must be between 1 and n_features=" + String(d))
        if nc > n and not has_prev:
            raise Error("n_components=" + String(nc) + " must be less or equal to the batch number of samples "
                        + String(n) + " for the first partial_fit call.")
    if has_prev and prev_d != d:
        raise Error("Number of input features has changed from the previous batch")
    if seen + n < 2:
        # the explained variance divides by n_samples_seen - 1
        raise Error("Found array with " + String(n) + " sample(s) (shape=(" + String(n) + ", " + String(d)
                    + ")) while a minimum of 2 is required.")
    var kit = RKit[E, D, RES]()
    try:
        var xb = kit.add(Int(x), -1, n, d)
        var last_mean: Int
        var last_var: Int
        if not has_prev:
            last_mean = kit.zeros(1, d)
            last_var = kit.zeros(1, d)
        else:
            last_mean = kit.add(Int(prev_mean), -1, 1, d)
            last_var = kit.add(Int(prev_var), -1, 1, d)
        # _incremental_mean_and_var
        var new_sum = kit.sum(xb, True)
        var total = seen + n
        var last_sum = kit.ew(_OP_SCALE, last_mean, scalar=Float64(seen))
        var summed = kit.ew(_OP_ADD, last_sum, new_sum)
        var upd_mean = kit.ew(_OP_SCALE, summed, scalar=1.0 / Float64(total))
        var bmean = kit.ew(_OP_SCALE, new_sum, scalar=1.0 / Float64(n))
        var spread = kit.ew(_OP_SQDIFF, xb, bmean)
        var new_unnorm = kit.sum(spread, True)
        var upd_unnorm = new_unnorm
        if has_prev:
            var last_over_new = Float64(seen) / Float64(n)
            var last_unnorm = kit.ew(_OP_SCALE, last_var, scalar=Float64(seen))
            var rescaled = kit.ew(_OP_SCALE, last_sum, scalar=1.0 / last_over_new)
            var t = kit.ew(_OP_SUB, rescaled, new_sum)
            var t2 = kit.ew(_OP_SQ, t)
            var corr = kit.ew(_OP_SCALE, t2, scalar=last_over_new / Float64(total))
            var partial = kit.ew(_OP_ADD, last_unnorm, new_unnorm)
            upd_unnorm = kit.ew(_OP_ADD, partial, corr)
        var upd_var = kit.ew(_OP_SCALE, upd_unnorm, scalar=1.0 / Float64(total))
        # the stacked matrix
        var z: Int
        if not has_prev:
            z = kit.ew(_OP_SUB, xb, upd_mean)
        else:
            var xc = kit.ew(_OP_SUB, xb, bmean)
            var shift = kit.ew(_OP_SUB, last_mean, bmean)
            var mc = kit.ew(_OP_SCALE, shift, scalar=sqrt((Float64(seen) / Float64(total)) * Float64(n)))
            var components = kit.add(Int(prev_components), -1, prev_count, d)
            var singular = kit.add(Int(prev_singular), -1, 1, prev_count)
            var singular_t = kit.transpose(singular)
            var prev = kit.ew(_OP_MUL, components, singular_t)
            z = _stack_rows(kit, prev, xc, mc)
        var decomposed = _gram_svd[E, D, RES, X](kit, z)
        var s = decomposed[0]
        var vt = decomposed[1]
        var s2 = kit.ew(_OP_SQ, s)
        var ev = kit.ew(_OP_SCALE, s2, scalar=1.0 / Float64(total - 1))
        var scaled_var = kit.ew(_OP_SCALE, upd_var, scalar=Float64(total))
        var tot_value = kit.total(scaled_var)
        # the float32 total read back is exact: the 1 x 1 operand holds it
        var tot_var = kit.constant(Float32(tot_value))
        var s2b = kit.ew(_OP_SQ, s)
        var evr = kit.ew(_OP_DIV, s2b, tot_var)
        var kept_rows = kit.indices(0, nc)
        var kept_components = kit.take_rows(vt, kept_rows)
        var kept_singular = kit.columns(s, 0, nc)
        var kept_ev = kit.columns(ev, 0, nc)
        var noise = Float64(0.0)
        if nc < min(d, kit.r(z)):
            var rest = kit.columns(ev, nc, kit.c(ev))
            var rest_value = kit.total(rest)
            var rest_total = kit.constant(Float32(rest_value))
            var nv = kit.ew(_OP_SCALE, rest_total, scalar=1.0 / Float64(kit.c(rest)))
            noise = kit.word(nv)
        var kept_ratio = kit.columns(evr, 0, nc)
        kit.copy_out(kept_components, components_out, nc * d)
        kit.copy_out(kept_singular, singular_out, nc)
        kit.copy_out(upd_mean, mean_out, d)
        kit.copy_out(upd_var, var_out, d)
        kit.copy_out(kept_ev, variance_out, nc)
        kit.copy_out(kept_ratio, ratio_out, nc)
        kit.close()
        return (nc, noise)
    except error:
        kit.close()
        raise error^


def ipca_fit_run[E: Exec, D: RDev, RES: Bool, X: IpcaDev](
    x: F32Ptr, n: Int, d: Int, requested: Int, batch_size: Int,
    components_out: F32Ptr, singular_out: F32Ptr, mean_out: F32Ptr, var_out: F32Ptr,
    variance_out: F32Ptr, ratio_out: F32Ptr,
) raises -> Tuple[Int, Float64]:
    """`IncrementalPCA.fit`'s batch loop: batches of batch_size rows (a batch
    whose end leaves fewer than n_components rows is skipped, as the Python
    loop skipped it), then the remainder; each batch one `ipca_partial_step`
    on the state the previous one wrote. The outputs are sized for at most
    min(d, batch) components (the caller allocates d x d, d each); returns
    (n_components_, noise_variance_)."""
    if batch_size == 0:
        raise Error("integer division or modulo by zero")
    var mb = max(requested, 0)
    # two state sets alternate: a step never reads what it writes
    var comps = [List[Float32](length=d * d, fill=Float32(0)), List[Float32](length=d * d, fill=Float32(0))]
    var sing = [List[Float32](length=d, fill=Float32(0)), List[Float32](length=d, fill=Float32(0))]
    var mean = [List[Float32](length=d, fill=Float32(0)), List[Float32](length=d, fill=Float32(0))]
    var var_ = [List[Float32](length=d, fill=Float32(0)), List[Float32](length=d, fill=Float32(0))]
    var ev = [List[Float32](length=d, fill=Float32(0)), List[Float32](length=d, fill=Float32(0))]
    var evr = [List[Float32](length=d, fill=Float32(0)), List[Float32](length=d, fill=Float32(0))]
    var seen = 0
    var count = 0
    var noise = Float64(0.0)
    var cur = 0
    var start = 0
    # `range(n // batch_size)`: no full batch for a negative batch size
    var batches = n // batch_size if batch_size > 0 else 0
    for _ in range(batches):  # driver-loop: over the fit's minibatches
        var end = start + batch_size
        if end + mb > n:
            continue
        var nxt = 1 - cur
        var got = ipca_partial_step[E, D, RES, X](
            x.unsafe_offset(start * d), end - start, d, requested, seen, count, d,
            _words(comps[cur]), _words(sing[cur]), _words(mean[cur]), _words(var_[cur]),
            _words(comps[nxt]), _words(sing[nxt]), _words(mean[nxt]), _words(var_[nxt]),
            _words(ev[nxt]), _words(evr[nxt]),
        )
        count = got[0]
        noise = got[1]
        seen += end - start
        cur = nxt
        start = end
    if start < n:
        var nxt = 1 - cur
        var got = ipca_partial_step[E, D, RES, X](
            x.unsafe_offset(start * d), n - start, d, requested, seen, count, d,
            _words(comps[cur]), _words(sing[cur]), _words(mean[cur]), _words(var_[cur]),
            _words(comps[nxt]), _words(sing[nxt]), _words(mean[nxt]), _words(var_[nxt]),
            _words(ev[nxt]), _words(evr[nxt]),
        )
        count = got[0]
        noise = got[1]
        seen = n
        cur = nxt
    if seen == 0:
        raise Error("x_decomp: IncrementalPCA fit saw no batch")
    memcpy(dest=components_out, src=_words(comps[cur]), count=count * d)
    memcpy(dest=singular_out, src=_words(sing[cur]), count=count)
    memcpy(dest=mean_out, src=_words(mean[cur]), count=d)
    memcpy(dest=var_out, src=_words(var_[cur]), count=d)
    memcpy(dest=variance_out, src=_words(ev[cur]), count=count)
    memcpy(dest=ratio_out, src=_words(evr[cur]), count=count)
    # the state words stay alive past the copies that read their addresses
    _ = comps^
    _ = sing^
    _ = mean^
    _ = var_^
    _ = ev^
    _ = evr^
    return (count, noise)


def ipca_transform_run[E: Exec, D: RDev, RES: Bool](
    x: F32Ptr, n: Int, d: Int, nc: Int, mean: F32Ptr, components: F32Ptr, variance: F32Ptr, whiten: Bool,
    out: F32Ptr,
) raises:
    """`transform`: (X - mean_) components_^T, divided by sqrt(explained
    variance) per column when whitening; n x nc written to out."""
    var kit = RKit[E, D, RES]()
    try:
        var m = kit.add(Int(x), -1, n, d)
        var mu = kit.add(Int(mean), -1, 1, d)
        var c = kit.add(Int(components), -1, nc, d)
        var centered = kit.ew(_OP_SUB, m, mu)
        var y = kit.mm(centered, c, tb=True)
        if whiten:
            var ev = kit.add(Int(variance), -1, 1, nc)
            var root = kit.ew(_OP_SQRT, ev)
            y = kit.ew(_OP_DIV, y, root)
        kit.copy_out(y, out, n * nc)
        kit.close()
    except error:
        kit.close()
        raise error^


def ipca_inverse_run[E: Exec, D: RDev, RES: Bool](
    y: F32Ptr, n: Int, d: Int, nc: Int, mean: F32Ptr, components: F32Ptr, variance: F32Ptr, whiten: Bool,
    out: F32Ptr,
) raises:
    """`inverse_transform`: Y C + mean_, C the components scaled by
    sqrt(explained variance) per row when whitening; n x d written to out."""
    var kit = RKit[E, D, RES]()
    try:
        var ym = kit.add(Int(y), -1, n, nc)
        var c = kit.add(Int(components), -1, nc, d)
        if whiten:
            var ev = kit.add(Int(variance), -1, 1, nc)
            var root = kit.ew(_OP_SQRT, ev)
            var root_t = kit.transpose(root)
            c = kit.ew(_OP_MUL, c, root_t)
        var mu = kit.add(Int(mean), -1, 1, d)
        var product = kit.mm(ym, c)
        var result = kit.ew(_OP_ADD, product, mu)
        kit.copy_out(result, out, n * d)
        kit.close()
    except error:
        kit.close()
        raise error^
