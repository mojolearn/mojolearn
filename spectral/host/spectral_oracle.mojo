# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""The host oracle for the spectral lane: the SAME Lanczos, serial, on the
host, in Float32 through the IDENTICAL helpers (the bit-for-bit reference
the device arm is gated against) and in Float64 (the tolerance reference).

NO REFERENCE FILE. cuVS ships one backend and needs no second opinion. This file
is `spectral/impl/sparse/solver/detail/lanczos.mojo` re-spelled with
every device launch replaced by a host loop that performs the SAME
ARITHMETIC IN THE SAME ORDER:

  device `spmv_kernel`          -> `host_spmv`: per row, ascending over the
                                   sorted entries, `fma` from `+0.0`, flushed
  `identical_gemm` (the dots,   -> `host_dot`: the contract's leaf partition
  the gemvs, the Ritz product)     (`contract_leaf_size(k)`) with the
                                   serial ascending leaf and the fixed
                                   balanced tree across leaves -- written
                                   here generically over dtype, and
                                   `check_oracle_dot_is_the_gemm_contract`
                                   asserts the Float32 arm equals
                                   `gemm_oracle` bit for bit
  `axpy_kernel`/`sub_kernel`/   -> one `hfma`/subtraction/division per
  `scale_vector_kernel`/           element, flushed
  `kernel_normalize`/
  `clamp_down_vector_kernel`    -> the same select
  the host scalars              -> the same host code, through the same
                                   helpers
  the projected solve           -> `symmetric_eig_host` -- SHARED with the
                                   device arm (the host is part of the plan)

The Float64 arm is the same function instantiated at `DType.float64` with
plain operations (see `symmetric_eig_host.mojo`'s `hfma`/`hflush`/`hsqrt`):
it says how far the Float32 answer is from the converged eigenpairs, which
a bit-compare cannot. `dense_laplacian_eigenvalues_f64` is the third
opinion: `decomposition/checks/jacobi_eigh.mojo` (Float64 cyclic Jacobi,
read-only import) on the DENSE Laplacian for `n <= 64`, so the Lanczos
Ritz values have a closed, independent reference on a graph with no closed
form.
"""

from std.math import isfinite
from std.sys.compile import is_defined

from checks.fixed_point import choose_scale
from checks.numerics import ftz
from cluster.host.kmeans_oracle import (
    INIT_KMEANS_PLUS_PLUS,
    KMeansHostTrace,
    METRIC_L2_EXPANDED,
    host_assign,
    host_fit_main,
    host_plan_sum_scale,
    host_row_norms,
)
from core.knn_host_predict import KNN_HOST_METRIC_FROM_IS_SQRT, host_knn_search
from gemm.host.identical_gemm import contract_leaf_size
from spectral.checks.symmetric_eig_host import (
    hflush,
    hfma,
    hsqrt,
    symmetric_eig_host,
)
from spectral.host.spectral_predict_host import (
    SpectralPredictionState,
    spectral_keep_embedding_order,
)
from spectral.impl.sparse.coo import CooGraph
from spectral.impl.sparse.op.coo_ops import (
    coo_remove_diagonal,
    coo_remove_scalar,
    coo_sort,
    refuse_repeated_keys,
    sorted_coo_to_csr,
)
from spectral.impl.sparse.solver.lanczos_types import LANCZOS_LA, LANCZOS_SA


# THE HOST FILE (workstream E batch 2, lane/cpu-training-e2, 2026-09-14).
# This oracle moved here from spectral/checks/spectral_oracle.mojo, which
# re-exports it, because it SHIPS in the metrics CPU host binding
# (bindings/_mojolearn_metrics_host.mojo) as the spectral lane's training
# path. The four constants below and `lanczos_v0` are spelled here rather
# than imported from spectral/impl/sparse/solver/detail/lanczos.mojo, which
# imports max.gpu; `check_spectral_host_constants_match` in
# spectral/checks/spectral_check.mojo would be the place to hold them to
# the device file's values. `dense_laplacian_eigenvalues_f64` stays in the
# checks file (it imports decomposition/checks/jacobi_eigh.mojo).

#: `LANCZOS_ALPHA_CLAMP`, `LANCZOS_U_CLAMP`, `LANCZOS_BETA_CLAMP`,
#: `lanczos.mojo:259-261`.
comptime LANCZOS_ALPHA_CLAMP = Float32(1e-9)
comptime LANCZOS_U_CLAMP = Float32(1e-7)
comptime LANCZOS_BETA_CLAMP = Float32(1e-6)

#: The gate's negative control for the spectral fit (see
#: `host_spectral_fit_predict_dataset`).
comptime SPECTRAL_ORACLE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def lanczos_v0(seed: UInt64, n: Int) -> List[Float32]:
    """`lanczos_v0`, `lanczos.mojo:920` (DEVIATION 772's start vector):
    hashed uniform `[0, 1)`, 24 bits, exact."""
    var out = List[Float32]()
    for i in range(n):
        var z = seed * UInt64(0x9E3779B97F4A7C15) + UInt64(i) + UInt64(1)
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        z = z ^ (z >> 31)
        var top = UInt32((z >> 40) & UInt64(0xFFFFFF))
        out.append(Float32(top) * Float32(5.9604644775390625e-08))
    return out^


# ---------------------------------------------------------------------------
# The contract's dot, generic over dtype
# ---------------------------------------------------------------------------


def host_leaf_partial[
    dt: DType
](a: List[Scalar[dt]], b: List[Scalar[dt]], p_begin: Int, p_end: Int) -> Scalar[dt]:
    """Contract 7.1: serial ascending, seeded `+0.0`, one fma per step,
    flushed at every seam."""
    var acc = Scalar[dt](0)
    for p in range(p_begin, p_end):
        acc = hflush[dt](hfma[dt](hflush[dt](a[p]), hflush[dt](b[p]), acc))
    return hflush[dt](acc)


def host_fold_tree[dt: DType](partials: List[Scalar[dt]]) -> Scalar[dt]:
    """Contract 7.2: adjacent pairing, ascending, odd tail carried, no
    padding, every node flushed. Character for character
    `gemm_oracle.mojo::fold_balanced_tree`, generic."""
    var p = len(partials)
    if p == 0:
        return Scalar[dt](0)
    var current = partials.copy()
    while len(current) > 1:
        var width = len(current)
        var pairs = width // 2
        var nxt = List[Scalar[dt]]()
        for q in range(pairs):
            nxt.append(hflush[dt](hflush[dt](current[2 * q]) + hflush[dt](current[2 * q + 1])))
        if width % 2 != 0:
            nxt.append(current[width - 1])
        current = nxt^
    return hflush[dt](current[0])


def host_dot[dt: DType](a: List[Scalar[dt]], b: List[Scalar[dt]], k: Int) -> Scalar[dt]:
    """`identical_gemm` at one cell, `k` terms: leaves of
    `contract_leaf_size(k)`, the tree across them."""
    var leaf = contract_leaf_size(k)
    var partials = List[Scalar[dt]]()
    var p0 = 0
    while p0 < k:
        var p1 = p0 + leaf
        if p1 > k:
            p1 = k
        partials.append(host_leaf_partial[dt](a, b, p0, p1))
        p0 = p1
    return host_fold_tree[dt](partials)


# ---------------------------------------------------------------------------
# The Laplacian on the host
# ---------------------------------------------------------------------------


struct HostLaplacian[dt: DType](Movable):
    """The negated (normalized or plain) Laplacian as a sorted COO with
    row offsets, plus `diag` (sqrt degree, zeros to one; empty when not
    normalized). Mirrors `DeviceCoo` + `diagonal`."""

    var n: Int
    var rows: List[Int32]
    var cols: List[Int32]
    var vals: List[Scalar[Self.dt]]
    var indptr: List[Int32]
    var diag: List[Scalar[Self.dt]]

    def __init__(out self, n: Int):
        self.n = n
        self.rows = List[Int32]()
        self.cols = List[Int32]()
        self.vals = List[Scalar[Self.dt]]()
        self.indptr = List[Int32]()
        self.diag = List[Scalar[Self.dt]]()


def host_laplacian[dt: DType](g: CooGraph, norm_laplacian: Bool) raises -> HostLaplacian[dt]:
    """`create_laplacian` on the host: mark/insert the diagonal, sort, the
    per-row ascending degree fold (DEVIATION 776), `D - A`, then (normalized)
    `diagonal`, `sqrt`, zero-to-one, the symmetric scale, the diagonal set to
    one; then negate. Same order of operations as the device kernels."""
    var n = g.n
    var marked = List[Bool]()
    for _ in range(n):
        marked.append(True)
    for i in range(g.nnz()):
        if g.rows[i] == g.cols[i]:
            marked[Int(g.rows[i])] = False
    var rows = g.rows.copy()
    var cols = g.cols.copy()
    var vals32 = g.vals.copy()
    for idx in range(n):
        if marked[idx]:
            rows.append(Int32(idx))
            cols.append(Int32(idx))
            vals32.append(Float32(0.0))
    var sorted_g = coo_sort(CooGraph(n, rows^, cols^, vals32^))
    refuse_repeated_keys(sorted_g)
    var out = HostLaplacian[dt](n)
    out.rows = sorted_g.rows.copy()
    out.cols = sorted_g.cols.copy()
    out.indptr = sorted_coo_to_csr(sorted_g)
    var nnz = sorted_g.nnz()
    for i in range(nnz):
        out.vals.append(Scalar[dt](sorted_g.vals[i]))
    # degrees: per row, ascending, flushed, seeded +0.0
    var degrees = List[Scalar[dt]]()
    for r in range(n):
        var acc = Scalar[dt](0)
        for j in range(Int(out.indptr[r]), Int(out.indptr[r + 1])):
            acc = hflush[dt](acc + out.vals[j])
        degrees.append(acc)
    # D - A
    for i in range(nnz):
        var r = out.rows[i]
        var v = out.vals[i]
        if r == out.cols[i]:
            out.vals[i] = hflush[dt](degrees[Int(r)] - v)
        else:
            out.vals[i] = -v
    if norm_laplacian:
        for _ in range(n):
            out.diag.append(Scalar[dt](0))
        for i in range(nnz):
            if out.rows[i] == out.cols[i]:
                out.diag[Int(out.rows[i])] = out.vals[i]
        for r in range(n):
            var s = hflush[dt](hsqrt[dt](out.diag[r]))
            if s == Scalar[dt](0):
                s = Scalar[dt](1)
            out.diag[r] = s
        for i in range(nnz):
            var dr = out.diag[Int(out.rows[i])]
            var dc = out.diag[Int(out.cols[i])]
            var row_scale = Scalar[dt](0)
            if dr != Scalar[dt](0):
                row_scale = hflush[dt](Scalar[dt](1) / dr)
            var col_scale = Scalar[dt](0)
            if dc != Scalar[dt](0):
                col_scale = hflush[dt](Scalar[dt](1) / dc)
            var t = hflush[dt](row_scale * out.vals[i])
            out.vals[i] = hflush[dt](t * col_scale)
        for i in range(nnz):
            if out.rows[i] == out.cols[i]:
                out.vals[i] = Scalar[dt](1)
    for i in range(nnz):
        out.vals[i] = -out.vals[i]
    return out^


def host_spmv[dt: DType](L: HostLaplacian[dt], x: List[Scalar[dt]]) -> List[Scalar[dt]]:
    """`spmv_kernel`: per row ascending, `fma` from `+0.0`, flushed."""
    var out = List[Scalar[dt]]()
    for r in range(L.n):
        var acc = Scalar[dt](0)
        for j in range(Int(L.indptr[r]), Int(L.indptr[r + 1])):
            acc = hflush[dt](hfma[dt](L.vals[j], x[Int(L.cols[j])], acc))
        out.append(acc)
    return out^


# ---------------------------------------------------------------------------
# The Lanczos on the host
# ---------------------------------------------------------------------------


struct OracleResult[dt: DType](Movable):
    """Every recorded stage of one host run, in the order the device arm
    records them, so a gate can compare stage by stage."""

    var step_alpha: List[Scalar[Self.dt]]
    var step_beta: List[Scalar[Self.dt]]
    var restart_res: List[Scalar[Self.dt]]
    var restart_ritz: List[Scalar[Self.dt]]
    """`k` Ritz values per restart (restart 0 = the first pass), flat."""
    var restart_sweeps: List[Int32]
    """Jacobi sweeps the projected solve took, one per restart."""
    var ritz: List[Scalar[Self.dt]]
    var ritz_vectors: List[Scalar[Self.dt]]
    var embedding: List[Scalar[Self.dt]]
    var n_out: Int
    var restarts: Int
    var converged: Bool
    var v0: List[Scalar[Self.dt]]
    var diag: List[Scalar[Self.dt]]
    """The Laplacian's `diag` (sqrt degree, zeros to one) when normalized,
    kept for `SpectralClustering(prediction_data=True)`."""

    def __init__(out self):
        self.step_alpha = List[Scalar[Self.dt]]()
        self.step_beta = List[Scalar[Self.dt]]()
        self.restart_res = List[Scalar[Self.dt]]()
        self.restart_ritz = List[Scalar[Self.dt]]()
        self.restart_sweeps = List[Int32]()
        self.ritz = List[Scalar[Self.dt]]()
        self.ritz_vectors = List[Scalar[Self.dt]]()
        self.embedding = List[Scalar[Self.dt]]()
        self.n_out = 0
        self.restarts = 0
        self.converged = False
        self.v0 = List[Scalar[Self.dt]]()
        self.diag = List[Scalar[Self.dt]]()


def _clamp[dt: DType](value: Scalar[dt], thr: Scalar[dt]) -> Scalar[dt]:
    if abs(value) < thr:
        return Scalar[dt](0)
    return value


def _norm2[dt: DType](x: List[Scalar[dt]], n: Int) -> Scalar[dt]:
    return hflush[dt](hsqrt[dt](host_dot[dt](x, x, n)))


def _row[dt: DType](V: List[Scalar[dt]], j: Int, n: Int) -> List[Scalar[dt]]:
    var out = List[Scalar[dt]]()
    for p in range(n):
        out.append(V[j * n + p])
    return out^


def _host_lanczos_aux[
    dt: DType
](
    L: HostLaplacian[dt],
    mut V: List[Scalar[dt]],
    mut u: List[Scalar[dt]],
    mut alpha: List[Scalar[dt]],
    mut beta: List[Scalar[dt]],
    start_idx: Int,
    end_idx: Int,
    ncv: Int,
    mut res: OracleResult[dt],
) raises:
    var n = L.n
    var v = _row[dt](V, start_idx, n)
    for i in range(start_idx, end_idx):
        u = host_spmv[dt](L, v)
        var alpha_i = host_dot[dt](v, u, n)
        var prev = (i - 1 + ncv) % ncv
        var b = beta[prev]
        # vv = 0; vv += alpha_i * v; vv += b * V[prev]; u += -1 * vv
        var vv = List[Scalar[dt]]()
        for p in range(n):
            var t = hflush[dt](hfma[dt](alpha_i, v[p], Scalar[dt](0)))
            t = hflush[dt](hfma[dt](b, V[prev * n + p], t))
            vv.append(t)
        for p in range(n):
            u[p] = hflush[dt](hfma[dt](Scalar[dt](-1), vv[p], u[p]))
        # uu[0..i] = V[0..i] u
        var uu = List[Scalar[dt]]()
        for j in range(i + 1):
            var vj = _row[dt](V, j, n)
            uu.append(host_dot[dt](vj, u, n))
        # tmp = V^T uu (k = i + 1 terms per coordinate); u = u - tmp
        for p in range(n):
            var col = List[Scalar[dt]]()
            for j in range(i + 1):
                col.append(V[j * n + p])
            var t = host_dot[dt](col, uu, i + 1)
            u[p] = hflush[dt](u[p] - t)
        alpha_i = hflush[dt](alpha_i + uu[i])
        alpha_i = _clamp[dt](alpha_i, Scalar[dt](LANCZOS_ALPHA_CLAMP))
        alpha[i] = alpha_i
        var beta_i = _norm2[dt](u, n)
        for p in range(n):
            u[p] = _clamp[dt](u[p], Scalar[dt](LANCZOS_U_CLAMP))
        beta_i = _clamp[dt](beta_i, Scalar[dt](LANCZOS_BETA_CLAMP))
        beta[i] = beta_i
        res.step_alpha.append(alpha_i)
        res.step_beta.append(beta_i)
        if i >= end_idx - 1:
            break
        for p in range(n):
            var val: Scalar[dt]
            if beta_i == Scalar[dt](0):
                val = hflush[dt](u[p] / Scalar[dt](1))
            else:
                val = hflush[dt](u[p] / beta_i)
            v[p] = val
            V[(i + 1) * n + p] = val


def _host_solve_ritz[
    dt: DType
](
    alpha: List[Scalar[dt]],
    beta: List[Scalar[dt]],
    beta_k: List[Scalar[dt]],
    has_beta_k: Bool,
    k: Int,
    which: Int,
    ncv: Int,
    mut eigenvalues_k: List[Scalar[dt]],
    mut eigenvectors_k: List[Scalar[dt]],
) raises -> Int:
    var t = List[Scalar[dt]]()
    for _ in range(ncv * ncv):
        t.append(Scalar[dt](0))
    for i in range(ncv):
        t[i * ncv + i] = alpha[i]
    for row in range(ncv):
        if row < ncv - 1:
            t[row * ncv + (row + 1)] = beta[row]
        if row > 0:
            t[row * ncv + (row - 1)] = beta[row - 1]
    if has_beta_k:
        for tid in range(k):
            t[k * ncv + tid] = beta_k[tid]
            t[tid * ncv + k] = beta_k[tid]
    var evals = List[Scalar[dt]]()
    var evecs = List[Scalar[dt]]()
    var sweeps = symmetric_eig_host[dt](t, ncv, evals, evecs)
    var first: Int
    if which == LANCZOS_SA:
        first = 0
    elif which == LANCZOS_LA:
        first = ncv - k
    else:
        raise Error("oracle: which not implemented")
    eigenvalues_k.clear()
    eigenvectors_k.clear()
    for c in range(k):
        eigenvalues_k.append(evals[first + c])
    for j in range(ncv):
        for c in range(k):
            eigenvectors_k.append(evecs[j * ncv + (first + c)])
    return sweeps


def _host_ritz_vectors[
    dt: DType
](E: List[Scalar[dt]], V: List[Scalar[dt]], k: Int, n: Int, ncv: Int) -> List[Scalar[dt]]:
    """`ritz = E^T V`, `k x n`, each cell a `ncv`-term contract dot."""
    var out = List[Scalar[dt]]()
    for c in range(k):
        var ecol = List[Scalar[dt]]()
        for j in range(ncv):
            ecol.append(E[j * k + c])
        for p in range(n):
            var vcol = List[Scalar[dt]]()
            for j in range(ncv):
                vcol.append(V[j * n + p])
            out.append(host_dot[dt](ecol, vcol, ncv))
    return out^


def _host_residual[
    dt: DType
](beta_last: Scalar[dt], E: List[Scalar[dt]], k: Int, ncv: Int, mut beta_k: List[Scalar[dt]]) -> Scalar[dt]:
    beta_k.clear()
    for c in range(k):
        var s = E[(ncv - 1) * k + c]
        beta_k.append(hflush[dt](hfma[dt](beta_last, s, Scalar[dt](0))))
    return hflush[dt](hsqrt[dt](host_dot[dt](beta_k, beta_k, k)))


def host_lanczos_smallest[
    dt: DType
](
    L: HostLaplacian[dt],
    k: Int,
    maxIter: Int,
    ncv: Int,
    tol: Scalar[dt],
    which: Int,
    v0: List[Scalar[dt]],
    mut res: OracleResult[dt],
) raises:
    """`lanczos_smallest` on the host; fills `res.ritz`, `res.ritz_vectors`
    (`k x n`), the per-step scalars, the restart residuals."""
    var n = L.n
    var V = List[Scalar[dt]]()
    for _ in range(ncv * n):
        V.append(Scalar[dt](0))
    var u = v0.copy()
    var v0nrm = _norm2[dt](u, n)
    for p in range(n):
        V[p] = hflush[dt](u[p] / v0nrm)
    var alpha = List[Scalar[dt]]()
    var beta = List[Scalar[dt]]()
    for _ in range(ncv):
        alpha.append(Scalar[dt](0))
        beta.append(Scalar[dt](0))
    _host_lanczos_aux[dt](L, V, u, alpha, beta, 0, ncv, ncv, res)
    var eigenvalues_k = List[Scalar[dt]]()
    var E = List[Scalar[dt]]()
    var beta_k = List[Scalar[dt]]()
    for _ in range(k):
        beta_k.append(Scalar[dt](0))
    res.restart_sweeps.append(
        Int32(_host_solve_ritz[dt](alpha, beta, beta_k, False, k, which, ncv, eigenvalues_k, E))
    )
    var ritz = _host_ritz_vectors[dt](E, V, k, n, ncv)
    var r = _host_residual[dt](beta[ncv - 1], E, k, ncv, beta_k)
    res.restart_res.append(r)
    for c in range(k):
        res.restart_ritz.append(eigenvalues_k[c])
    var restarts = 0
    var iter = ncv
    while r > tol and iter < maxIter:
        restarts += 1
        for c in range(k):
            beta[c] = Scalar[dt](0)
            alpha[c] = eigenvalues_k[c]
        for i in range(k * n):
            V[i] = ritz[i]
        # uu = V[0..k) u; u -= V^T uu
        var uu = List[Scalar[dt]]()
        for j in range(k):
            var vj = _row[dt](V, j, n)
            uu.append(host_dot[dt](vj, u, n))
        for p in range(n):
            var col = List[Scalar[dt]]()
            for j in range(k):
                col.append(V[j * n + p])
            var t = host_dot[dt](col, uu, k)
            u[p] = hflush[dt](u[p] - t)
        var unrm = _norm2[dt](u, n)
        for p in range(n):
            V[k * n + p] = hflush[dt](u[p] / unrm)
        var vk = _row[dt](V, k, n)
        u = host_spmv[dt](L, vk)
        var alpha_k = host_dot[dt](vk, u, n)
        alpha[k] = alpha_k
        for p in range(n):
            u[p] = hflush[dt](hfma[dt](-alpha_k, V[k * n + p], u[p]))
        for p in range(n):
            var col = List[Scalar[dt]]()
            for j in range(k):
                col.append(V[j * n + p])
            var t = host_dot[dt](col, beta_k, k)
            u[p] = hflush[dt](u[p] - t)
        var beta_kk = _norm2[dt](u, n)
        beta[k] = beta_kk
        res.step_alpha.append(alpha_k)
        res.step_beta.append(beta_kk)
        if beta_kk == Scalar[dt](0):
            raise Error("oracle: restart breakdown, beta[k] == 0 (DEVIATION 774)")
        for p in range(n):
            V[(k + 1) * n + p] = hflush[dt](u[p] / beta_kk)
        _host_lanczos_aux[dt](L, V, u, alpha, beta, k + 1, ncv, ncv, res)
        iter += ncv - k
        res.restart_sweeps.append(
            Int32(_host_solve_ritz[dt](alpha, beta, beta_k, True, k, which, ncv, eigenvalues_k, E))
        )
        ritz = _host_ritz_vectors[dt](E, V, k, n, ncv)
        r = _host_residual[dt](beta[ncv - 1], E, k, ncv, beta_k)
        res.restart_res.append(r)
        for c in range(k):
            res.restart_ritz.append(eigenvalues_k[c])
    res.ritz = eigenvalues_k.copy()
    res.ritz_vectors = ritz^
    res.restarts = restarts
    res.converged = r <= tol


def oracle_embedding[
    dt: DType
](
    g: CooGraph,
    n_components: Int,
    norm_laplacian: Bool,
    drop_first: Bool,
    tolerance: Scalar[dt],
    seed: UInt64,
) raises -> OracleResult[dt]:
    """`transform_graph` on the host: the Laplacian, the Lanczos with cuVS's
    config (`max_iterations = 10n`, `ncv = min(n - k, max(2k + 1, 20))`,
    `LA`), the division by `diag`, the reversed gather."""
    var n = g.n
    var k = n_components
    if n - k <= 0:
        raise Error("Please set `ncv` to a value in (0, n_samples)")
    var ncv_hi = 2 * k + 1
    if ncv_hi < 20:
        ncv_hi = 20
    var ncv = n - k
    if ncv_hi < ncv:
        ncv = ncv_hi
    var L = host_laplacian[dt](g, norm_laplacian)
    var res = OracleResult[dt]()
    res.diag = L.diag.copy()
    var v0_32 = lanczos_v0(seed, n)
    var v0 = List[Scalar[dt]]()
    for i in range(n):
        v0.append(Scalar[dt](v0_32[i]))
    res.v0 = v0.copy()
    host_lanczos_smallest[dt](L, k, 10 * n, ncv, tolerance, LANCZOS_LA, v0, res)
    var vecs = res.ritz_vectors.copy()
    if norm_laplacian:
        for c in range(k):
            for p in range(n):
                vecs[c * n + p] = hflush[dt](vecs[c * n + p] / L.diag[p])
    var n_out = k - 1 if drop_first else k
    res.n_out = n_out
    res.embedding.clear()
    for p in range(n):
        for c_out in range(n_out):
            var src = n_out - 1 - c_out
            res.embedding.append(vecs[src * n + p])
    return res^


# ---------------------------------------------------------------------------
# The spectral clustering FIT on the host (workstream E batch 2)
# ---------------------------------------------------------------------------
#
# `spectral/impl/cluster/detail/spectral.mojo::fit_predict_dataset` then
# `fit_predict_graph`, restated: `create_connectivity_graph`
# (`spectral/impl/preprocessing/detail/spectral_embedding.mojo:130`) is
# the host k-NN self-join (`core/knn_host_predict.mojo::host_knn_search`
# at L2SqrtExpanded, the row itself its first neighbor), the `(i,
# neighbor, 1.0)` COO, `coo_symmetrize_kernel` (`spectral/impl/sparse/
# linalg/detail/symmetrize.mojo:44`, one row segment at a time over the
# row-sorted input, the transposed lookup, `ftz(0.5 * ftz(a + b))`, the
# two writes into a zero-filled `2 * nnz` output), `coo_sort` and
# `coo_remove_scalar(0)`; then `oracle_embedding` (the host oracle above,
# `norm_laplacian` true, `drop_first` false, the seed); then k-means on
# the row-major embedding exactly as `fit_predict_graph` sets it up:
# `plan_sum_scale` over the embedding, `choose_scale(n, n)` for the
# weights, unit weights, `KMeansParams.default()` with `n_clusters`,
# `INIT_KMEANS_PLUS_PLUS`, L2 expanded, the seed, `n_init` and
# `oversampling_factor = 0.0` (the classic sequential k-means++), through
# `cluster/host/kmeans_oracle.mojo::host_fit_main`, then `predict`'s fresh
# assignment with the row norms `spectral.mojo` computes (`take_sqrt` 0).
#
# THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` hands the k-means
# `seed + 1`, a different restart draw on every fixture, on top of the
# core family's extra unit per quantized cell that `host_fit_main`
# carries. Read back by `metrics_host_sabotage`.


def host_coo_symmetrize(g_in: CooGraph) -> CooGraph:
    """`coo_symmetrize_kernel` over every row of the ROW-SORTED `g_in`
    (module comment above): the RAW `2 * nnz` output with its zero slots,
    as theirs returns it."""
    var n = g_in.n
    var nnz = g_in.nnz()
    var row_ind = sorted_coo_to_csr(g_in)
    var out_n = 2 * nnz if nnz > 0 else 1
    var orows = List[Int32](length=out_n, fill=Int32(0))
    var ocols = List[Int32](length=out_n, fill=Int32(0))
    var ovals = List[Float32](length=out_n, fill=Float32(0.0))
    for row in range(n):
        var start_idx = Int(row_ind[row])
        var stop_idx = Int(row_ind[row + 1])
        var row_nnz = 0
        var out_start_idx = start_idx * 2
        for idx in range(0, stop_idx - start_idx):
            var cur_row = g_in.rows[start_idx + idx]
            var cur_col = g_in.cols[start_idx + idx]
            var cur_val = g_in.vals[start_idx + idx]
            var lookup_row = Int(cur_col)
            var t_start = Int(row_ind[lookup_row])
            var t_stop = Int(row_ind[lookup_row + 1])
            var transpose = Float32(0.0)
            var found_match = False
            for t_idx in range(t_start, t_stop):
                if g_in.cols[t_idx] == cur_row and g_in.rows[t_idx] == cur_col:
                    transpose = g_in.vals[t_idx]
                    found_match = True
                    break
            var res = ftz(Float32(0.5) * ftz(cur_val + transpose))
            if (not found_match) and cur_val != Float32(0.0):
                orows[out_start_idx + row_nnz] = cur_col
                ocols[out_start_idx + row_nnz] = cur_row
                ovals[out_start_idx + row_nnz] = res
                row_nnz += 1
            if res != Float32(0.0):
                orows[out_start_idx + row_nnz] = cur_row
                ocols[out_start_idx + row_nnz] = cur_col
                ovals[out_start_idx + row_nnz] = res
                row_nnz += 1
    # the device reads back exactly 2 * nnz slots
    var rows2 = List[Int32]()
    var cols2 = List[Int32]()
    var vals2 = List[Float32]()
    for i in range(2 * nnz):
        rows2.append(orows[i])
        cols2.append(ocols[i])
        vals2.append(ovals[i])
    return CooGraph(n, rows2^, cols2^, vals2^)


def host_create_connectivity_graph(
    dataset: List[Float32], n_samples: Int, n_features: Int, k_search: Int
) raises -> CooGraph:
    """`create_connectivity_graph` (module comment above): the refusals in
    its words, the k-NN self-join, the COO, symmetrize, sort, drop zeros."""
    if n_samples <= 0 or n_features <= 0:
        raise Error("spectral: dataset must be n_samples x n_features with both positive")
    if len(dataset) != n_samples * n_features:
        raise Error("spectral: dataset length does not match n_samples x n_features")
    if k_search < 1 or k_search > n_samples:
        raise Error(
            "spectral: n_neighbors=" + String(k_search)
            + " must satisfy 1 <= n_neighbors <= n_samples (" + String(n_samples) + ")"
        )
    for i in range(len(dataset)):
        var x = dataset[i]
        if not isfinite(x):
            raise Error(
                "spectral: dataset has a non-finite value at index " + String(i)
                + " -- refused by name (a NaN may not reach a card)"
            )
    var nnz = n_samples * k_search
    var dist = List[Float32](length=nnz, fill=Float32(0.0))
    var idx = List[UInt32](length=nnz, fill=UInt32(0))
    # brute_force (L2SqrtExpanded), dataset against itself
    host_knn_search(
        dataset, n_samples, dataset, n_samples, n_features, k_search,
        KNN_HOST_METRIC_FROM_IS_SQRT, True, dist, idx,
    )
    var rows = List[Int32]()
    var cols = List[Int32]()
    var vals = List[Float32]()
    for e in range(nnz):
        rows.append(Int32(e // k_search))
        cols.append(Int32(idx[e]))
        vals.append(Float32(1.0))
    var knn = CooGraph(n_samples, rows^, cols^, vals^)
    var sym_raw = host_coo_symmetrize(knn)
    var sym_sorted = coo_sort(sym_raw)
    return coo_remove_scalar(sym_sorted, Float32(0.0))


def host_spectral_config_validate(
    n_init: Int, n_neighbors: Int, eigen_tol: Float32
) raises:
    """`_config`'s refusals, `spectral/estimator.mojo`, in its words."""
    if n_init < 1:
        raise Error(
            "spectral clustering: n_init must be at least 1, got "
            + String(n_init)
        )
    if n_neighbors < 1:
        raise Error(
            "spectral clustering: n_neighbors must be at least 1, got "
            + String(n_neighbors)
        )
    if not (eigen_tol > Float32(0.0)):
        raise Error(
            "spectral clustering: eigen_tol must be positive, got "
            + String(eigen_tol)
        )


def host_spectral_fit_predict_graph(
    graph: CooGraph,
    n_clusters: Int,
    n_components: Int,
    n_init: Int,
    eigen_tol: Float32,
    seed: UInt64,
    mut labels: List[Int32],
    mut embedding_out: List[Float32],
) raises -> Int:
    """`host_spectral_fit_predict_graph_keep` keeping nothing."""
    var state = SpectralPredictionState()
    return host_spectral_fit_predict_graph_keep(
        graph, n_clusters, n_components, n_init, eigen_tol, seed, labels,
        embedding_out, state, False,
    )


def host_spectral_fit_predict_graph_keep(
    graph: CooGraph,
    n_clusters: Int,
    n_components: Int,
    n_init: Int,
    eigen_tol: Float32,
    seed: UInt64,
    mut labels: List[Int32],
    mut embedding_out: List[Float32],
    mut state: SpectralPredictionState,
    keep: Bool,
) raises -> Int:
    """`fit_predict_graph_keep` (module comment above). Returns `n_out`.
    With `keep`, `state` receives copies of the Ritz values, the undivided
    Ritz vectors, `diag` and the final centroids (lane/spectral-predict)."""
    var n_samples = graph.n
    if n_clusters < 1 or n_clusters > n_samples:
        raise Error(
            "spectral clustering: n_clusters=" + String(n_clusters)
            + " must satisfy 1 <= n_clusters <= n_samples"
        )
    var res = oracle_embedding[DType.float32](
        graph, n_components, True, False, eigen_tol, seed
    )
    var n_out = res.n_out
    if keep:
        spectral_keep_embedding_order(res.ritz, res.ritz_vectors, n_components, n_samples, state)
        state.diag = res.diag.copy()
    embedding_out.clear()
    for i in range(len(res.embedding)):
        embedding_out.append(res.embedding[i])
    var n_features = n_out
    var sum_scale = host_plan_sum_scale(embedding_out, n_samples, n_features)
    var weight_scale = choose_scale(Float64(n_samples), n_samples)
    var ones = List[Float32](length=n_samples, fill=Float32(1.0))
    var centroids = List[Float32](length=n_clusters * n_features, fill=Float32(0.0))
    var u_labels = List[UInt32](length=n_samples, fill=UInt32(0))
    var kmeans_seed = seed
    comptime if SPECTRAL_ORACLE_HOST_SABOTAGE:
        # THE SABOTAGE ARM: a different restart draw. Wrong on purpose; see
        # SPECTRAL_ORACLE_HOST_SABOTAGE.
        kmeans_seed = seed + UInt64(1)
    var trace = KMeansHostTrace()
    _ = host_fit_main(
        embedding_out, n_samples, n_features, ones, n_clusters, centroids,
        u_labels, INIT_KMEANS_PLUS_PLUS, kmeans_seed, n_init, 300, Float64(1e-4),
        METRIC_L2_EXPANDED, Float64(0.0), Float32(sum_scale),
        Float32(weight_scale), trace, String("spectral."),
    )
    if keep:
        state.centroids = centroids.copy()
    # kmeans::fit_predict's fresh assignment, with the row norms
    # spectral.mojo computes (take_sqrt 0, the L2 expanded metric)
    var x_norm = host_row_norms(embedding_out, n_samples, n_features, False)
    var c_norm = host_row_norms(centroids, n_clusters, n_features, False)
    var min_dist = List[Float32](length=n_samples, fill=Float32(0.0))
    host_assign(
        embedding_out, n_samples, x_norm, centroids, n_clusters, c_norm,
        n_features, False, u_labels, min_dist,
    )
    labels.clear()
    for i in range(n_samples):
        labels.append(Int32(u_labels[i]))
    return n_out


def host_spectral_fit_predict_dataset(
    dataset: List[Float32],
    n_samples: Int,
    n_features: Int,
    n_clusters: Int,
    n_components: Int,
    n_init: Int,
    n_neighbors: Int,
    eigen_tol: Float32,
    seed: UInt64,
    mut labels: List[Int32],
    mut embedding_out: List[Float32],
) raises -> Int:
    """`host_spectral_fit_predict_dataset_keep` keeping nothing."""
    var state = SpectralPredictionState()
    return host_spectral_fit_predict_dataset_keep(
        dataset, n_samples, n_features, n_clusters, n_components, n_init,
        n_neighbors, eigen_tol, seed, labels, embedding_out, state, False,
    )


def host_spectral_fit_predict_dataset_keep(
    dataset: List[Float32],
    n_samples: Int,
    n_features: Int,
    n_clusters: Int,
    n_components: Int,
    n_init: Int,
    n_neighbors: Int,
    eigen_tol: Float32,
    seed: UInt64,
    mut labels: List[Int32],
    mut embedding_out: List[Float32],
    mut state: SpectralPredictionState,
    keep: Bool,
) raises -> Int:
    """`spectral_fit_predict_dataset_host`, `spectral/estimator.mojo`, then
    `fit_predict_dataset` (module comment above). Returns `n_out`."""
    if n_samples <= 0 or n_features <= 0:
        raise Error(
            "spectral clustering: X must be n_samples x n_features with both"
            " positive, got " + String(n_samples) + " x " + String(n_features)
        )
    if len(dataset) < n_samples * n_features:
        raise Error(
            "spectral clustering: X holds " + String(len(dataset))
            + " floats, needs " + String(n_samples * n_features)
        )
    host_spectral_config_validate(n_init, n_neighbors, eigen_tol)
    var graph = host_create_connectivity_graph(
        dataset, n_samples, n_features, n_neighbors
    )
    return host_spectral_fit_predict_graph_keep(
        graph, n_clusters, n_components, n_init, eigen_tol, seed, labels,
        embedding_out, state, keep,
    )


def host_validate_connectivity_coo(
    rows: List[Int32],
    cols: List[Int32],
    vals: List[Float32],
    n_samples: Int,
    n_components: Int,
) raises:
    """The device path's refusals on a given COO, in its words:
    `transform_graph`'s non-finite and negative values,
    `compute_graph_laplacian`'s index range and the Lanczos entry's shape
    (`host_spectral_fit_predict_coo_keep` names the files)."""
    var nnz = len(vals)
    for i in range(nnz):
        var v = vals[i]
        if not isfinite(v):
            raise Error(
                "spectral: connectivity_graph has a non-finite value at entry "
                + String(i) + " -- refused by name"
            )
        if v < Float32(0.0):
            raise Error(
                "spectral: connectivity_graph has a negative value at entry "
                + String(i) + " -- refused by name (sqrt of a negative degree is NaN in theirs)"
            )
    for i in range(nnz):
        var r = Int(rows[i])
        var c = Int(cols[i])
        if r < 0 or r >= n_samples or c < 0 or c >= n_samples:
            raise Error(
                "connectivity_graph: entry " + String(i) + " has (row, col) = ("
                + String(r) + ", " + String(c) + ") outside [0, "
                + String(n_samples) + ")"
            )
    var k = n_components
    if n_samples - k > 0:
        var ncv_hi = 2 * k + 1
        if ncv_hi < 20:
            ncv_hi = 20
        var ncv = n_samples - k
        if ncv_hi < ncv:
            ncv = ncv_hi
        if k < 1:
            raise Error(
                "lanczos: need 1 <= n_components < n, got " + String(k)
                + " for n=" + String(n_samples)
            )
        if ncv <= k + 1 or ncv > n_samples:
            raise Error(
                "lanczos: need n_components + 1 < ncv <= n, got ncv=" + String(ncv)
                + " n_components=" + String(k) + " n=" + String(n_samples)
            )


def host_spectral_fit_predict_coo(
    rows: List[Int32],
    cols: List[Int32],
    vals: List[Float32],
    n_samples: Int,
    n_clusters: Int,
    n_components: Int,
    n_init: Int,
    n_neighbors: Int,
    eigen_tol: Float32,
    seed: UInt64,
    mut labels: List[Int32],
    mut embedding_out: List[Float32],
) raises -> Int:
    """`host_spectral_fit_predict_coo_keep` keeping nothing."""
    var state = SpectralPredictionState()
    return host_spectral_fit_predict_coo_keep(
        rows, cols, vals, n_samples, n_clusters, n_components, n_init,
        n_neighbors, eigen_tol, seed, labels, embedding_out, state, False,
    )


def host_spectral_fit_predict_coo_keep(
    rows: List[Int32],
    cols: List[Int32],
    vals: List[Float32],
    n_samples: Int,
    n_clusters: Int,
    n_components: Int,
    n_init: Int,
    n_neighbors: Int,
    eigen_tol: Float32,
    seed: UInt64,
    mut labels: List[Int32],
    mut embedding_out: List[Float32],
    mut state: SpectralPredictionState,
    keep: Bool,
) raises -> Int:
    """`spectral_fit_predict_graph_host`, `spectral/estimator.mojo:189`, then
    `fit_predict_graph` (the spectral-precomputed lane, 2026-09-14): the
    affinity graph is GIVEN as COO triples, so no k-NN runs and
    `n_neighbors` is validated and read by nobody. Returns `n_out`.

    The refusals are the device path's, in its words: the estimator's
    three and `_config`'s, `fit_predict_graph`'s `n_clusters` (inside
    `host_spectral_fit_predict_graph`), `transform_graph`'s non-finite and
    negative value refusals (`spectral/impl/preprocessing/detail/
    spectral_embedding.mojo:328-339`), `compute_graph_laplacian`'s index
    range (`spectral/impl/sparse/linalg/detail/laplacian.mojo:183-191`),
    and the Lanczos entry's shape (`spectral/impl/sparse/solver/detail/
    lanczos.mojo:689-694`), which `oracle_embedding` does not restate and
    without which a host Lanczos with `ncv <= k + 1` would index past its
    basis. The repeated-key refusal is `host_laplacian`'s. The one order
    difference: the device reaches the repeated-key refusal before the
    Lanczos shape refusal, the host after, so an input wrong in both ways
    names the other one first. No value moves.

    The arithmetic is `host_spectral_fit_predict_graph` on a `CooGraph` of
    copies, exactly as the device entry builds its graph; the sabotage
    arms (`seed + 1` for the recluster, the extra quantized unit in
    `host_fit_main`) are reached on this path as on the dataset one."""
    if n_samples <= 0:
        raise Error(
            "spectral clustering: n_samples must be positive, got "
            + String(n_samples)
        )
    var nnz = len(vals)
    if nnz <= 0:
        raise Error(
            "spectral clustering: the connectivity graph has no entries"
        )
    if len(rows) != nnz or len(cols) != nnz:
        raise Error(
            "spectral clustering: rows, cols and vals must be the same"
            " length, got " + String(len(rows)) + ", " + String(len(cols))
            + ", " + String(nnz)
        )
    host_spectral_config_validate(n_init, n_neighbors, eigen_tol)
    if n_clusters < 1 or n_clusters > n_samples:
        raise Error(
            "spectral clustering: n_clusters=" + String(n_clusters)
            + " must satisfy 1 <= n_clusters <= n_samples"
        )
    host_validate_connectivity_coo(rows, cols, vals, n_samples, n_components)
    var r_copy = rows.copy()
    var c_copy = cols.copy()
    var v_copy = vals.copy()
    var graph = CooGraph(n_samples, r_copy^, c_copy^, v_copy^)
    return host_spectral_fit_predict_graph_keep(
        graph, n_clusters, n_components, n_init, eigen_tol, seed, labels,
        embedding_out, state, keep,
    )


# ---------------------------------------------------------------------------
# SpectralEmbedding on the host (lane/expose-spectral-embedding, 2026-09-20)
# ---------------------------------------------------------------------------
#
# `spectral/impl/spectral_embedding.mojo::transform` and
# `transform_connectivity` restated: the connectivity graph above (dataset
# arm only), then `oracle_embedding` at cuVS's default tolerance `1e-5`,
# with `norm_laplacian` and `drop_first` the caller's. No k-means runs, so
# the negative control is its own: `-D MOJOLEARN_HOST_SABOTAGE=1` negates
# embedding column 0.

comptime SPECTRAL_EMBEDDING_TOLERANCE = Float32(1e-5)


def _host_embedding_from_graph(
    graph: CooGraph,
    n_components: Int,
    norm_laplacian: Bool,
    drop_first: Bool,
    seed: UInt64,
    mut embedding_out: List[Float32],
) raises -> Int:
    var res = oracle_embedding[DType.float32](
        graph, n_components, norm_laplacian, drop_first,
        SPECTRAL_EMBEDDING_TOLERANCE, seed,
    )
    var n_out = res.n_out
    embedding_out.clear()
    for i in range(len(res.embedding)):
        embedding_out.append(res.embedding[i])
    comptime if SPECTRAL_ORACLE_HOST_SABOTAGE:
        # THE SABOTAGE ARM: column 0 negated. Wrong on purpose.
        for p in range(graph.n):
            embedding_out[p * n_out] = -embedding_out[p * n_out]
    return n_out


def host_spectral_embedding_dataset(
    dataset: List[Float32],
    n_samples: Int,
    n_features: Int,
    n_components: Int,
    n_neighbors: Int,
    norm_laplacian: Bool,
    drop_first: Bool,
    seed: UInt64,
    mut embedding_out: List[Float32],
) raises -> Int:
    """`spectral_embedding_dataset_host`, `spectral/estimator.mojo`, on the
    host. `n_components` is the Lanczos count (the caller's plus one when
    `drop_first`). Returns `n_out`."""
    var graph = host_create_connectivity_graph(
        dataset, n_samples, n_features, n_neighbors
    )
    host_validate_connectivity_coo(
        graph.rows, graph.cols, graph.vals, n_samples, n_components
    )
    return _host_embedding_from_graph(
        graph, n_components, norm_laplacian, drop_first, seed, embedding_out
    )


def host_spectral_embedding_coo(
    rows: List[Int32],
    cols: List[Int32],
    vals: List[Float32],
    n_samples: Int,
    n_components: Int,
    norm_laplacian: Bool,
    drop_first: Bool,
    seed: UInt64,
    mut embedding_out: List[Float32],
) raises -> Int:
    """`spectral_embedding_graph_host`, `spectral/estimator.mojo`, on the
    host: the affinity graph is GIVEN as COO triples, its diagonal entries
    dropped first. Returns `n_out`."""
    if n_samples <= 0:
        raise Error(
            "spectral embedding: n_samples must be positive, got "
            + String(n_samples)
        )
    var nnz = len(vals)
    if nnz <= 0:
        raise Error("spectral embedding: the connectivity graph has no entries")
    if len(rows) != nnz or len(cols) != nnz:
        raise Error(
            "spectral embedding: rows, cols and vals must be the same"
            " length, got " + String(len(rows)) + ", " + String(len(cols))
            + ", " + String(nnz)
        )
    host_validate_connectivity_coo(rows, cols, vals, n_samples, n_components)
    var r_copy = rows.copy()
    var c_copy = cols.copy()
    var v_copy = vals.copy()
    var graph = coo_remove_diagonal(CooGraph(n_samples, r_copy^, c_copy^, v_copy^))
    return _host_embedding_from_graph(
        graph, n_components, norm_laplacian, drop_first, seed, embedding_out
    )
