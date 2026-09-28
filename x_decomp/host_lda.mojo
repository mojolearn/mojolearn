# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU column's spelling of `lda_doc_row` (x_decomp/cells.mojo,
DEVIATION 5313), sklearn `_update_doc_distribution` for one document (lane
decomp-cpu, 2026-09-28). Host only: compiled into the CPU host binding.

THE SAME ARITHMETIC IN THE SAME ORDER. Per iteration the cell forms
  norm_phi_w = ftz(fma(ftz(E_t), ftz(EW_tw), acc)), t ascending, + EPS,
  r_w = div0(X_w, norm_phi_w + EPS) for each nonzero word w, and
  D_t = E_t * sum_w ftz(fma(ftz(r_w), ftz(EW_tw), acc)), w ascending over
  the nonzero words,
then the scalar Dirichlet expectation and the stop test. What changed is
only WHICH outputs advance together:
  * the first fold's vector lanes are different WORDS (each lane is one
    word's t-ascending chain), read from a per-document copy of EW's
    nonzero-word columns (ftz applied at the copy; ftz is idempotent);
  * the second fold's vector lanes are different TOPICS (each lane is one
    topic's w-ascending chain), read from EW^T packed once per call;
  * the zero-count words are listed once per document (the cell skips them
    in both folds on every iteration; the list is the same test).
No lane is ever a piece of another lane's sum, so the vector width is not a
pin. Every document is its own task (xd_parallel), so the result is the
same at every thread count.

Proof: x_decomp/checks/rows_check.mojo holds HostExec.lda_rows to the
oracle (one iteration) and to the device column (several iterations:
D, E and the iteration counts, zero and subnormal counts in the fixture);
x_decomp/checks/sabotage/host_lda_topic_fold.patch must make it fail.
"""
from std.math import ceildiv

from checks.numerics import ftz
from x_decomp.cells import F32Ptr, add, digamma, div0, exp_c, mul, sub
from x_decomp.host_simd import W, V, _ftz1, ftz_v, mul_add_v

comptime TB = 4  # topic vectors advanced together in the second fold


def lda_pack_t(ew: F32Ptr, k: Int, v: Int) -> List[Float32]:
    """EW (k x v) transposed to v x kp (kp = k rounded up to W), ftz
    applied, the padding topics 0."""
    var kp = ceildiv(k, W) * W
    var out = List[Float32](length=max(1, v * kp), fill=Float32(0))
    for w in range(v):
        for t in range(k):
            out[w * kp + t] = _ftz1(ew.unsafe_load(t * v + w))
    return out^


def lda_doc_row_host(
    X: F32Ptr, EW: F32Ptr, EWt: F32Ptr, D: F32Ptr, E: F32Ptr, i: Int, k: Int, v: Int, prior: Float32,
    max_iter: Int, tol: Float32,
) -> Float32:
    """`lda_doc_row` for document i (see the module docstring): D and E rows
    i are the start and the answer. Returns the iterations run."""
    var base = i * k
    var kp = ceildiv(k, W) * W
    var eps = Float32(2.220446049250313e-16)
    var nz = List[Int](capacity=v)
    for w in range(v):
        if ftz(X.unsafe_load(i * v + w)) != Float32(0):
            nz.append(w)
    var nnz = len(nz)
    var npad = max(W, ceildiv(nnz, W) * W)
    var ewc_buf = List[Float32](length=k * npad, fill=Float32(0))
    var ewc = ewc_buf.unsafe_ptr()
    for t in range(k):
        for j in range(nnz):
            ewc[t * npad + j] = _ftz1(EW.unsafe_load(t * v + nz[j]))
    var xs = List[Float32](length=npad, fill=Float32(0))
    for j in range(nnz):
        xs[j] = ftz(X.unsafe_load(i * v + nz[j]))
    var r = List[Float32](length=npad, fill=Float32(0))
    var last = List[Float32](length=k, fill=Float32(0))
    var ef = List[Float32](length=k, fill=Float32(0))
    var it = 0
    for _ in range(max_iter):
        it += 1
        for t in range(k):
            last[t] = D.unsafe_load(base + t)
            ef[t] = ftz(E.unsafe_load(base + t))
        # first fold: lanes are words, each a t-ascending chain
        var j0 = 0
        while j0 < nnz:
            var acc = V(0)
            for t in range(k):
                acc = ftz_v[W](mul_add_v[W](V(ef[t]), ewc.load[width=W](t * npad + j0), acc))
            for l in range(min(W, nnz - j0)):
                r[j0 + l] = div0(xs[j0 + l], add(acc[l], eps))
            j0 += W
        # second fold: lanes are topics, each a w-ascending chain
        var total = Float32(0)
        var tb0 = 0
        while tb0 < kp:
            var nb = min(TB, (kp - tb0) // W)
            var acc = InlineArray[V, TB](fill=V(0))
            for j in range(nnz):
                var rv = V(_ftz1(r[j]))
                var row = EWt + nz[j] * kp + tb0
                comptime for b in range(TB):
                    if b < nb:
                        acc[b] = ftz_v[W](mul_add_v[W](rv, row.load[width=W](b * W), acc[b]))
            for b in range(nb):
                for l in range(W):
                    var t = tb0 + b * W + l
                    if t < k:
                        var dt = add(mul(E.unsafe_load(base + t), acc[b][l]), prior)
                        D.unsafe_store(base + t, dt)
                        total = add(total, dt)
            tb0 += nb * W
        var psi_total = digamma(total)
        var change = Float32(0)
        for t in range(k):
            var dt = D.unsafe_load(base + t)
            E.unsafe_store(base + t, exp_c(sub(digamma(dt), psi_total)))
            change = add(change, abs(sub(last[t], dt)))
        if div0(change, Float32(k)) < tol:
            break
    _ = ewc_buf^
    return Float32(it)
