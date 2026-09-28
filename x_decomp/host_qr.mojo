# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU column's Householder QR, `host_qr_factor`'s words at CPU speed
(lane decomp-cpu, 2026-09-28). Host only: compiled into the CPU host binding.

`decomposition/host/pca_full_oracle.mojo::host_qr_factor` replays the GPU's
`qr_panel_kernel` (the tall-skinny slices, then one QR of the stacked R's)
one column pair at a time with strided reads. This file computes the same
words in the same order:
  * slice b's factorization is independent of every other slice (a task);
  * column j's QR_TPB norm partials, each lane t folding rows j + t,
    j + t + QR_TPB, ... ascending, then the halving tree (plain adds, as
    `host_halving_sum`), then the reflector (r_jj, u1, tau) and the
    division of the column below the diagonal: unchanged;
  * the dot partials of EVERY trailing column c > j are formed in one
    pass over the rows (SIMD across c: each (t, c) lane is its own chain
    in the same row order), then each column's tail, total and `td` as
    before, then every trailing column updated row by row (SIMD across c).
    Doing all the dots before all the updates is the same computation:
    column c's dot reads column j (not updated in this step) and column c
    below row j (updated only by column c's own update), and its update
    writes only column c.
Under MOJOLEARN_HOST_SABOTAGE the dot fold is the oracle's sabotage arm
(lanes added serially descending), as `_qr_fold` does.

Proof: x_decomp/checks/dense_check.mojo holds `fast_qr_factor` to
`host_qr_factor` bit for bit (one slice and many, square and tall, tails
of the SIMD width) and arm host_qr_dot_fold.patch must make it fail.
"""
from std.math import ceildiv, fma
from std.memory import bitcast
from std.sys.info import simd_width_of

from checks.numerics import ftz, identical_div, identical_mul_add, identical_sqrt
from decomposition.host.pca_full_oracle import QR_TPB, host_qr_slice_count
from decomposition.host.pca_oracle import PCA_ORACLE_HOST_SABOTAGE
from x_decomp.cells import F32Ptr
from x_decomp.host_simd import ftz_v, mul_add_v

comptime W = simd_width_of[DType.float32]()
comptime V = SIMD[DType.float32, W]


@always_inline
def _halving(p: F32Ptr) -> Float32:
    """`host_halving_sum` over QR_TPB partials at p (clobbers them)."""
    var step = QR_TPB // 2
    while step > 0:
        for t in range(step):
            p.unsafe_store(t, p.unsafe_load(t) + p.unsafe_load(t + step))
        step //= 2
    return p.unsafe_load(0)


@always_inline
def _dot_fold(p: F32Ptr) -> Float32:
    comptime if PCA_ORACLE_HOST_SABOTAGE:
        var acc = Float32(0.0)
        for tt in range(QR_TPB):
            acc = acc + p.unsafe_load(QR_TPB - 1 - tt)
        return acc
    return _halving(p)


def qr_slice(a: F32Ptr, r_out: F32Ptr, m: Int, n: Int, n_slices: Int, b: Int):
    """`host_qr_panel_slice(a, r_out, m, n, n_slices, b)`: slice b of the
    m x n row-major `a` factored in place; its R at r_out + b * n * n."""
    var rb = (b * m) // n_slices
    var re = ((b + 1) * m) // n_slices
    var ms = re - rb
    var rbase = b * n * n
    var np = ceildiv(n, W) * W
    # dot partials: QR_TPB lanes x np columns; one column's lanes, contiguous
    var dp = List[Float32](length=QR_TPB * np, fill=Float32(0))
    var pd = F32Ptr(unsafe_from_address=Int(dp.unsafe_ptr()))
    var lane = List[Float32](length=QR_TPB, fill=Float32(0))
    var pl = F32Ptr(unsafe_from_address=Int(lane.unsafe_ptr()))
    var tdv = List[Float32](length=np, fill=Float32(0))
    var ptd = F32Ptr(unsafe_from_address=Int(tdv.unsafe_ptr()))
    var row0 = a.unsafe_offset(rb * n)

    for j in range(n):
        for t in range(QR_TPB):
            var acc = Float32(0.0)
            var i = j + t
            while i < ms:
                var v = ftz(row0.unsafe_load(i * n + j))
                acc = ftz(identical_mul_add(v, v, acc))
                i += QR_TPB
            pl.unsafe_store(t, acc)
        var sigma = _halving(pl)
        var normx = ftz(identical_sqrt(sigma))
        var ajj = ftz(row0.unsafe_load(j * n + j))

        if normx == Float32(0.0):
            r_out.unsafe_store(rbase + j * n + j, Float32(0.0))
            continue
        var s = Float32(-1.0) if ajj >= Float32(0.0) else Float32(1.0)
        var r_jj = ftz(s * normx)
        var u1 = ftz(ajj - r_jj)
        var tau = ftz(identical_div(ftz(ftz(-s) * u1), normx))
        r_out.unsafe_store(rbase + j * n + j, r_jj)
        for i2 in range(j + 1, ms):
            var cur = ftz(row0.unsafe_load(i2 * n + j))
            row0.unsafe_store(i2 * n + j, ftz(identical_div(cur, u1)))
        var c0 = j + 1
        var nc = n - c0
        if nc == 0:
            continue
        # every trailing column's QR_TPB dot partials in one pass over the rows
        for t in range(QR_TPB):
            var dst = pd.unsafe_offset(t * np)
            for c in range(nc):
                dst.unsafe_store(c, Float32(0))
            var i3 = j + 1 + t
            while i3 < ms:
                var w = V(ftz(row0.unsafe_load(i3 * n + j)))
                var src = row0.unsafe_offset(i3 * n + c0)
                var c = 0
                while c + W <= nc:
                    var x = ftz_v[W](src.unsafe_load[width=W](c))
                    dst.unsafe_store(c, ftz_v[W](mul_add_v[W](w, x, dst.unsafe_load[width=W](c))))
                    c += W
                while c < nc:
                    var x1 = ftz(src.unsafe_load(c))
                    dst.unsafe_store(c, ftz(identical_mul_add(w[0], x1, dst.unsafe_load(c))))
                    c += 1
                i3 += QR_TPB
        # each column's fold, total and td; row j updated
        for c in range(nc):
            for t in range(QR_TPB):
                pl.unsafe_store(t, pd.unsafe_load(t * np + c))
            var tail = _dot_fold(pl)
            var ajc = ftz(row0.unsafe_load(j * n + c0 + c))
            var total = ftz(ajc + tail)
            var td = ftz(tau * total)
            row0.unsafe_store(j * n + c0 + c, ftz(ajc - td))
            ptd.unsafe_store(c, -td)
        # every trailing column updated, row by row
        for i4 in range(j + 1, ms):
            var w2 = V(ftz(row0.unsafe_load(i4 * n + j)))
            var rowp = row0.unsafe_offset(i4 * n + c0)
            var c = 0
            while c + W <= nc:
                var cur2 = ftz_v[W](rowp.unsafe_load[width=W](c))
                rowp.unsafe_store(c, ftz_v[W](mul_add_v[W](ptd.unsafe_load[width=W](c), w2, cur2)))
                c += W
            while c < nc:
                var cur1 = ftz(rowp.unsafe_load(c))
                rowp.unsafe_store(c, ftz(identical_mul_add(ptd.unsafe_load(c), w2[0], cur1)))
                c += 1

    for t in range(n * n):
        var rr = t // n
        var cc = t - rr * n
        if cc > rr:
            if rr < ms:
                r_out.unsafe_store(rbase + t, ftz(row0.unsafe_load(rr * n + cc)))
            else:
                r_out.unsafe_store(rbase + t, Float32(0.0))
        elif cc < rr:
            r_out.unsafe_store(rbase + t, Float32(0.0))
    _ = dp^
    _ = lane^
    _ = tdv^


def qr_slices(m: Int, n: Int) -> Int:
    return host_qr_slice_count(m, n)


def fast_qr_finish(scratch: F32Ptr, r: F32Ptr, ns: Int, n: Int):
    """The stacked slice R's (ns * n x n) factored once more into R."""
    qr_slice(scratch, r, ns * n, n, 1, 0)


def fast_qr_factor(a: F32Ptr, m: Int, n: Int) raises -> List[Float32]:
    """`host_qr_factor` (a is destroyed): R, n x n row major. Serial driver;
    HostExec runs the slices as tasks."""
    if m < n:
        raise Error("qr_factor needs at least as many rows as columns")
    var ns = qr_slices(m, n)
    var r = List[Float32](length=n * n, fill=Float32(0.0))
    var pr = F32Ptr(unsafe_from_address=Int(r.unsafe_ptr()))
    if ns == 1:
        qr_slice(a, pr, m, n, 1, 0)
        return r^
    var scratch = List[Float32](length=ns * n * n, fill=Float32(0.0))
    var ps = F32Ptr(unsafe_from_address=Int(scratch.unsafe_ptr()))
    for b in range(ns):
        qr_slice(a, ps, m, n, ns, b)
    fast_qr_finish(ps, pr, ns, n)
    _ = scratch^
    return r^
