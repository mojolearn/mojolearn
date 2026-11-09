# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host column of x_decomp/eigh_ops.mojo (lane n7-device-linalg-ops,
2026-11-08): the same algorithm, the same words. No GPU import.

  `eigh_op_dense_host`    `host_eigh_rr_sorted` (x_decomp/rr_solve.mojo,
                          the words of `rr_batch_kernel` for each problem),
                          the problems in parallel over the host pool (each
                          writes only its own w, v rows).
  `eigh_op_sym_host`      `eigh_op_sym`: from n = TD_MIN_N (IDENTICAL) the
                          host tridiagonal solve (`host_eigh_td`) on a
                          copy, the Jacobi of the input on a refusal; the
                          Jacobi below (lane n9-decomp-eigh-families).
  `eigh_op_tridiag_host`  the device's items one call per index
                          (`td_bisect_item`, `td_gap_item`, `td_vec_item`, in
                          parallel over the index, as x_decomp/eigh_tridiag_host.mojo
                          runs them), then `host_sign_flip` and `eigh_ascending`;
                          on a refusal, the dense Jacobi of T.

Both return the device op's status word: the number of problems whose
answer does not hold (an unconverged Jacobi). The outputs of a failed
problem are unspecified, as on the device.
"""
from checks.numerics import ftz
from cluster.host.host_cells import host_cells
from decomposition.host.linalg_public import eigh_ascending
from decomposition.host.pca_oracle import host_sign_flip
from x_decomp.cells import F32Ptr
from x_decomp.eigh_tridiag_cells import EIGH_TD_IDN, TD_MIN_N
from x_decomp.eigh_tridiag_host import host_eigh_td, host_td_solve
from x_decomp.rr_block import IDN_EIGH_BLOCK, rb_use
from x_decomp.rr_solve import host_eigh_rr_sorted


@always_inline
def _lp(l: List[Float32]) -> F32Ptr:
    return F32Ptr(unsafe_from_address=Int(l.unsafe_ptr()))


def _eo_host_one(a: F32Ptr, w: F32Ptr, v: F32Ptr, n: Int) -> Int32:
    """One dense problem (n x n at a, read only) into w, v: 0 when it
    converged, 1 otherwise."""
    var m = List[Float32](capacity=n * n)
    for i in range(n * n):
        m.append(a.unsafe_load(i))
    var wl = List[Float32]()
    var vl = List[Float32]()
    try:
        _ = host_eigh_rr_sorted(m, n, wl, vl)
    except:
        return Int32(1)
    for i in range(n):
        w.unsafe_store(i, wl[i])
    for i in range(n * n):
        v.unsafe_store(i, vl[i])
    return Int32(0)


def eigh_op_dense_host(a: F32Ptr, w: F32Ptr, v: F32Ptr, batch: Int, n: Int) raises -> Int:
    """`eigh_op_dense` on host memory: `batch` problems of n x n at `a`
    (read only here; the device op consumes its copy) into w (batch x n)
    and v (batch x n x n). Returns the count of unconverged problems."""
    if n < 1 or batch < 1:
        raise Error("eigh_op_dense_host: n and batch must be positive, got n=" + String(n) + " batch=" + String(batch))
    var bad = List[Int32](length=batch, fill=Int32(0))
    var bp = bad.unsafe_ptr()

    def one(b: Int) {imm a, imm w, imm v, imm bp, imm n}:
        bp.unsafe_store(b, _eo_host_one(a + b * n * n, w + b * n, v + b * n * n, n))

    # every problem writes only its own rows; a Jacobi sweep is ~n^3 work
    host_cells(one, batch, 40 * n * n * n)
    var count = 0
    for b in range(batch):  # small-loop(batch: one status word per problem)
        count += Int(bad[b])
    _ = bad^
    return count


def eigh_op_sym_host(a: F32Ptr, w: F32Ptr, v: F32Ptr, n: Int) raises -> Int:
    """`eigh_op_sym` on host memory: one n x n problem at `a` (both
    triangles, read only) into w (n ascending) and v (n x n, eigenvector c
    in column c). Returns 1 when no route holds, else 0. `HostExec.eigh`
    is this after its triangle mirror."""
    if n < 1:
        raise Error("eigh_op_sym_host: n must be positive, got " + String(n))
    comptime if IDN_EIGH_BLOCK:
        if rb_use(n):
            raise Error(
                "eigh_op_sym: MOJOLEARN_IDN_EIGH_BLOCK_ON is a quarantined candidate with no device op;"
                " build without it"
            )
    comptime if EIGH_TD_IDN:
        if n >= TD_MIN_N:
            var work = List[Float32](capacity=n * n)
            for i in range(n * n):
                work.append(a.unsafe_load(i))
            if host_eigh_td(work, n, w, v):
                return 0
    return Int(_eo_host_one(a, w, v, n))


def eigh_op_tridiag_host(d: F32Ptr, e: F32Ptr, w: F32Ptr, z: F32Ptr, n: Int) raises -> Int:
    """`eigh_op_tridiag` on host memory: T (diagonal d, off-diagonal e) into
    w (n ascending) and z (n x n, eigenvector c in column c). Returns 1 when
    neither the tridiagonal solve nor its Jacobi fallback holds, else 0."""
    if n < 1:
        raise Error("eigh_op_tridiag_host: n must be positive, got " + String(n))
    # eo_ftz_copy_kernel
    var ddl = List[Float32](length=n, fill=Float32(0.0))
    var eel = List[Float32](length=n, fill=Float32(0.0))
    for i in range(n):
        ddl[i] = ftz(d.unsafe_load(i))
        if i < n - 1:
            eel[i] = ftz(e.unsafe_load(i))
    var dd = _lp(ddl)
    var ee = _lp(eel)
    var infol = List[Float32](length=4, fill=Float32(0.0))
    var whl = List[Float32](length=n, fill=Float32(0.0))
    var wll = List[Float32](length=n, fill=Float32(0.0))
    var wtl = List[Float32](length=n, fill=Float32(0.0))
    var info = _lp(infol)
    var wh = _lp(whl)
    var wlo = _lp(wll)
    var wt = _lp(wtl)

    var zl = List[Float32](length=n * n, fill=Float32(0.0))
    host_td_solve(dd, ee, info, wh, wlo, wt, _lp(zl), n)
    if infol[2] == Float32(0.0):
        host_sign_flip(zl, n)
        var got = eigh_ascending(wtl, zl, n, True, 0)
        for i in range(n):
            w.unsafe_store(i, got.w[i])
        for i in range(n * n):
            z.unsafe_store(i, got.v[i])
        _ = ddl^
        _ = eel^
        return 0
    # the refusal: the dense Jacobi of T (eo_tridiag_dense_kernel)
    var tl = List[Float32](length=n * n, fill=Float32(0.0))
    for i in range(n):
        tl[i * n + i] = ddl[i]
        if i < n - 1:
            tl[i * n + i + 1] = eel[i]
            tl[(i + 1) * n + i] = eel[i]
    var bad = Int(_eo_host_one(_lp(tl), w, z, n))
    _ = tl^
    _ = ddl^
    _ = eel^
    return bad
