# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host oracle for the spectral lane: the SAME Lanczos, serial, on the
host, in Float32 through the IDENTICAL helpers (the bit-for-bit reference
the device arm is gated against) and in Float64 (the tolerance reference).

NOT A PORT. cuVS ships one backend and needs no second opinion. This file
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

from decomposition.checks.jacobi_eigh import jacobi_eigh
from gemm.checks.gemm_oracle import contract_leaf_size
from spectral.checks.symmetric_eig_host import (
    hflush,
    hfma,
    hsqrt,
    symmetric_eig_host,
)
from spectral.impl.sparse.coo import CooGraph
from spectral.impl.sparse.op.coo_ops import (
    coo_sort,
    refuse_repeated_keys,
    sorted_coo_to_csr,
)
from spectral.impl.sparse.solver.detail.lanczos import (
    LANCZOS_ALPHA_CLAMP,
    LANCZOS_BETA_CLAMP,
    LANCZOS_U_CLAMP,
    lanczos_v0,
)
from spectral.impl.sparse.solver.lanczos_types import LANCZOS_LA, LANCZOS_SA


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
        raise Error("oracle: which not ported")
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
# The dense Float64 cross-check
# ---------------------------------------------------------------------------


def dense_laplacian_eigenvalues_f64(g: CooGraph, norm_laplacian: Bool) raises -> List[Float64]:
    """The Float64 Laplacian of `g` (NOT negated: these are the eigenvalues
    of `L`, ascending) densified and handed to `decomposition`'s Float64
    Jacobi. `n <= 64` only; a third, independent opinion on the spectrum."""
    var n = g.n
    if n > 64:
        raise Error("dense_laplacian_eigenvalues_f64: n <= 64 only")
    var L = host_laplacian[DType.float64](g, norm_laplacian)
    var a = List[Float64]()
    var vecs = List[Float64]()
    for _ in range(n * n):
        a.append(0.0)
        vecs.append(0.0)
    for i in range(len(L.vals)):
        a[Int(L.rows[i]) * n + Int(L.cols[i])] = -L.vals[i]
    jacobi_eigh(a, vecs, n)
    var d = List[Float64]()
    for i in range(n):
        d.append(a[i * n + i])
    for i in range(1, n):
        var key = d[i]
        var j = i - 1
        while j >= 0 and d[j] > key:
            d[j + 1] = d[j]
            j -= 1
        d[j + 1] = key
    return d^
