# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The decomposition owners' kit steps beyond the randomized PCA kit
(lane mca2-decomp-a, 2026-10-22): `_Kit.eigh`, `_Kit.lu` / `lu_solve`,
`_Kit.rand` on any stream, `_M.rows` / `take_cols` and `_pinv_rows`, on
`RKit` matrices (x_decomp/pca_randomized.mojo), each the typed body of the
entry the Python kit called, resident where the Python kit was.

`DecompDev` is the binding's half: the resident eigh / LU the IDENTICAL GPU
build registers (`x_decomp_dev_eigh`, `x_decomp_dev_lu`) and the solver
drivers the binding registers under one name with a host and a GPU body
(`x_decomp_ica_solve`, `x_decomp_nmf_solve`, ...). x_decomp/decomp_owner_host.mojo
and decomp_owner_dev.mojo implement it for the two bindings."""
from std.memory import memcpy

from x_decomp.api import eigh_run, lu_solve_run, move_run, rand_run
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.pca_randomized import RDev, RKit
from x_decomp.nmf import NmfArgs
from x_decomp.fa_em import FaArgs
from x_decomp.lda_fit import LdaArgs

# x_decomp/cells.mojo op codes (`_OP` in _expansion_decomp.py)
comptime OP_ADD = 0
comptime OP_SUB = 1
comptime OP_MUL = 2
comptime OP_DIV = 3
comptime OP_MAXS = 5
comptime OP_SQRT = 7
comptime OP_SQ = 8
comptime OP_EXP = 9
comptime OP_LOGS = 10
comptime OP_ABS = 13
comptime OP_SCALE = 14
comptime OP_RECIP = 16
comptime OP_MINS = 19
comptime OP_SQDIFF = 21
comptime OP_ADDS = 22
comptime OP_GTS = 23
comptime OP_DIGAMMA = 24
comptime OP_SELECT = 35
#: x_decomp/moves.mojo PLACE_COLS (`_MV_PLACE_COLS`)
comptime MV_PLACE_COLS = 2
#: `_Kit.rand`'s smallest resident draw
comptime RAND_RESIDENT_MIN = 1024


trait DecompDev:
    """The binding's resident and solver entries the Python kit reached."""

    @staticmethod
    def has_eigh() -> Bool:
        ...

    @staticmethod
    def eigh(a: Int, w: F32Ptr, v: F32Ptr, n: Int, uplo: Int) raises:
        ...

    @staticmethod
    def has_lu() -> Bool:
        ...

    @staticmethod
    def lu(a: Int, lu: Int, piv: I32Ptr, info: F32Ptr, n: Int) raises:
        ...

    @staticmethod
    def ica_solve(
        px: F32Ptr, pw: F32Ptr, nc: Int, xr: Int, xc: Int, algo: Int, fun: Int, max_iter: Int, alpha: Float64,
        tol: Float64,
    ) raises -> Int:
        ...

    @staticmethod
    def nmf_solve(px: F32Ptr, pw: F32Ptr, ph: F32Ptr, n: Int, d: Int, nc: Int, solver: Int, a: NmfArgs) raises -> Int:
        ...

    @staticmethod
    def nmf_nndsvd(pu: F32Ptr, ps: F32Ptr, pv: F32Ptr, pw: F32Ptr, ph: F32Ptr, n: Int, d: Int, r: Int, nc: Int) raises -> Int:
        ...

    @staticmethod
    def has_qr_r() -> Bool:
        ...

    @staticmethod
    def qr_r(a: Int, r: F32Ptr, m: Int, n: Int, consume: Bool) raises -> Int:
        ...

    @staticmethod
    def fa_em_main(pa: F32Ptr, pp: F32Ptr, pw: F32Ptr, pl: MutPointer[Float64, MutUntrackedOrigin], ar: Int,
                   args: FaArgs) raises -> Int:
        ...

    @staticmethod
    def ortho_rotation(pc: F32Ptr, po: F32Ptr, nrow: Int, ncol: Int, vm: Bool, max_iter: Int, tol: Float64) raises:
        ...

    @staticmethod
    def fa_flags() -> Int:
        """FactorAnalysis's FAST Apple switches (`x_decomp_fa_defines`): bit 1
        GRAM_ONCE, 2 ITER_DEVICE, 4 TRANSFORM_FUSED, 8 GRAM_DF; 0 elsewhere."""
        ...

    @staticmethod
    def fa_gram(x: Int, mean: Int, g: Int, var_: Int, n: Int, d: Int) raises:
        ...

    @staticmethod
    def fa_em(g: Int, psi_src: F32Ptr, w_dst: F32Ptr, psi_dst: F32Ptr, ll_dst: MutPointer[Float64, MutAnyOrigin],
              d: Int, nc: Int, n: Int, max_iter: Int, tol: Float64) raises -> Int:
        ...

    @staticmethod
    def fa_transform(x: Int, mean: Int, pm: Int, dst: Int, n: Int, d: Int, nc: Int) raises:
        ...

    @staticmethod
    def lda_rows(x: Int, ew: Int, d: Int, e: Int, n: Int, k: Int, v: Int, max_iter: Int, prior: Float64,
                 tol: Float64) raises:
        """`x_decomp_dev_lda_rows` on device ids (the GPU binding only)."""
        ...

    @staticmethod
    def lda_bound(x: Int, ddt: Int, dcomp: Int, dst: Int, n: Int, k: Int, v: Int, floor: Float64) raises:
        """`x_decomp_dev_lda_bound` on device ids (the GPU binding only)."""
        ...

    @staticmethod
    def lda_fit(px: F32Ptr, pc: F32Ptr, pe: F32Ptr, n: Int, v: Int, nc: Int, a: LdaArgs, mut draw: Int,
                mut nbi: Int) raises -> Int:
        ...

    @staticmethod
    def lda_online(px: F32Ptr, pc: F32Ptr, pe: F32Ptr, n: Int, v: Int, nc: Int, bs: Int, mdi: Int, seed: Int,
                   mut draw: Int, mut nbi: Int, fv: List[Float64]) raises:
        ...

    @staticmethod
    def sqdist(a: Int, b: Int, dst: Int, na: Int, nb: Int, d: Int) raises:
        """`x_decomp_dev_sqdist` on device ids (the GPU binding only)."""
        ...

    @staticmethod
    def lanczos_dev(ia: Int, iq: Int, iab: Int, n: Int, j0: Int, j1: Int, cap: Int) raises:
        """`x_decomp_dev_lanczos` on device ids (the GPU binding only)."""
        ...

    @staticmethod
    def lanczos_host(pa: F32Ptr, pq: F32Ptr, pab: F32Ptr, n: Int, j0: Int, j1: Int, cap: Int) raises:
        """`x_decomp_lanczos` on host words (the host binding only)."""
        ...

    @staticmethod
    def grp_flags() -> Int:
        """The random projections' switches (`x_decomp_grp_cls2`): bit 1
        NOSCAN, 2 DEVSCAN, 4 LAZY, 8 FUSED, 16 SRP_STRAT; 0 elsewhere."""
        ...

    @staticmethod
    def grp_fit_fused(src: F32Ptr, cnt: Int, dst: Int, host_out: F32Ptr, count: Int, seed: Int, mode: Int, sd: Int,
                      sgrid: Int, sc: Float64, thr: Float64) raises -> Int:
        """`x_decomp_grp_fit_fused` (FUSED builds only)."""
        ...

    @staticmethod
    def dev_project(a: Int, b: Int, c: Int, flag: Int, m: Int, k: Int, n: Int) raises -> Int:
        """`x_decomp_dev_project` on device ids (the GPU binding only)."""
        ...


def words_of(ref words: List[Float32]) -> F32Ptr:
    return F32Ptr(unsafe_from_address=Int(words.unsafe_ptr()))


def kit_eigh[E: Exec, D: RDev, RES: Bool, X: DecompDev](mut kit: RKit[E, D, RES], a: Int) raises -> Tuple[Int, Int]:
    """`_Kit.eigh(A)`: (w 1 x n ascending, v n x n eigenvectors in columns),
    host words; an operand already on the device is solved on a device copy
    where the build exports `x_decomp_dev_eigh`, else the host-address entry."""
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


def kit_reversed_columns[E: Exec, D: RDev, RES: Bool](mut kit: RKit[E, D, RES], a: Int) raises -> Int:
    """`A.take_cols(range(c - 1, -1, -1))` of a host matrix: a strided copy
    per column (data movement)."""
    var rows = kit.r(a)
    var cols = kit.c(a)
    var out = kit.zeros(rows, cols)
    var src = kit.ptr(a)
    var dst = kit.ptr(out)
    for t in range(cols):  # driver-loop: over the eigenpairs (model structure: n_features)
        var j = cols - 1 - t
        for i in range(rows):  # small-loop(rows of a features x features eigenvector matrix): one word each
            dst[i * cols + t] = src[i * cols + j]
    return out


def kit_rows[E: Exec, D: RDev, RES: Bool](mut kit: RKit[E, D, RES], a: Int, start: Int, stop: Int) raises -> Int:
    """`_M.rows(start, stop)`: the device gather for a resident matrix, else
    the host slice."""
    var cols = kit.c(a)
    if kit.mats[a].dev >= 0 and 0 <= start and start < stop and stop <= kit.r(a):
        var idx = kit.indices(start, stop)
        return kit.take_rows(a, idx)
    var out = kit.zeros(stop - start, cols)
    if (stop - start) * cols:
        memcpy(dest=kit.ptr(out), src=kit.ptr(a).unsafe_offset(start * cols), count=(stop - start) * cols)
    return out


def kit_rand[E: Exec, D: RDev, RES: Bool](
    mut kit: RKit[E, D, RES], rows: Int, cols: Int, seed: Int, stream: Int, kind: Int
) raises -> Int:
    """`_Kit.rand(rows, cols, seed, stream, kind)`: drawn on the device from
    RAND_RESIDENT_MIN values on a GPU binding, else on the host; seed and
    stream mod 2^32."""
    var count = rows * cols
    var s = seed & 0xFFFFFFFF
    var st = stream & 0xFFFFFFFF
    comptime if RES:
        if count >= RAND_RESIDENT_MIN:
            var out = kit.device_output(rows, cols)
            D.rand(kit.mats[out].dev, count, s, st, kind)
            return out
    var out = kit.zeros(rows, cols)
    if count:
        rand_run[E](kit.ptr(out), count, s, st, kind)
    return out


def kit_lu[E: Exec, D: RDev, RES: Bool, X: DecompDev](
    mut kit: RKit[E, D, RES], a: Int, mut piv: List[Int32]
) raises -> Tuple[Int, Int]:
    """`_Kit.lu(A)`: (the factor, info); the pivots into `piv`. The factor
    is made in a device matrix from a device copy where the build exports
    `x_decomp_dev_lu`, else in a host copy."""
    var n = kit.r(a)
    piv = List[Int32](length=max(n, 1), fill=Int32(0))
    var info = kit.zeros(1, 1)
    var pp = I32Ptr(unsafe_from_address=Int(piv.unsafe_ptr()))
    var resident = False
    comptime if RES:
        resident = X.has_lu() and n >= 1 and kit.c(a) == n
    if resident:
        var lu = kit.device_output(n, n)
        var da = kit.device(a)
        var info_ptr = kit.ptr(info)
        _ = info  # keep the Python object alive through the native call above
        var lu_dev = kit.mats[lu].dev
        X.lu(da, lu_dev, pp, info_ptr, n)
        _ = info  # keep the Python object alive through the native call above
        var _keep_ret_info = (lu, Int(kit.ptr(info)[0]))
        _ = info  # keep the Python object alive through the native call above
        var _keep_ret_info = _keep_ret_info^
        _ = info  # keep the Python object alive through the native call above
        return _keep_ret_info^
    var lu = kit.copy_host(a)
    _exec_lu[E](kit.ptr(lu), pp, kit.ptr(info), n)
    _ = info  # keep the Python object alive through the native call above
    var _keep_ret_info = (lu, Int(kit.ptr(info)[0]))
    _ = info  # keep the Python object alive through the native call above
    var _keep_ret_info = _keep_ret_info^
    _ = info  # keep the Python object alive through the native call above
    return _keep_ret_info^


def _exec_lu[E: Exec](pa: F32Ptr, pp: I32Ptr, pi: F32Ptr, n: Int) raises:
    # `lu_py[E]`'s body
    E.lu(pa, pp, pi, n)


def kit_lu_solve[E: Exec, D: RDev, RES: Bool](
    mut kit: RKit[E, D, RES], lu: Int, piv: List[Int32], b: Int
) raises -> Int:
    """`_Kit.lu_solve(lu, piv, B)`: a host copy of B solved in place by the
    host-address entry (the factor's words read where they are)."""
    var out = kit.copy_host(b)
    var pp = I32Ptr(unsafe_from_address=Int(piv.unsafe_ptr()))
    lu_solve_run[E](kit.ptr(lu), pp, kit.ptr(out), kit.r(lu), kit.c(b), 0)
    return out


def kit_pinv_rows[E: Exec, D: RDev, RES: Bool, X: DecompDev](mut kit: RKit[E, D, RES], c: Int) raises -> Int:
    """`_pinv_rows(k, C)`: C^T (C C^T)^-1 through the LU solve of the Gram."""
    var gram = kit.mm(c, c, tb=True)
    var piv = List[Int32]()
    var factored = kit_lu[E, D, RES, X](kit, gram, piv)
    if factored[1] != 0:
        raise Error("compute_inverse_components: the components are rank deficient")
    var solved = kit_lu_solve(kit, factored[0], piv, c)
    _ = piv^
    return kit.transpose(solved)


def kit_colmean[E: Exec, D: RDev, RES: Bool](mut kit: RKit[E, D, RES], a: Int) raises -> Int:
    """`_Kit.colmean(A)`: the column sums scaled by 1 / rows."""
    var sums = kit.sum(a, True)
    var rows = kit.r(a)
    return kit.ew(OP_SCALE, sums, scalar=1.0 / Float64(rows))


comptime _MV_TAKE_COLS = 4


def kit_strided[E: Exec, D: RDev, RES: Bool](
    mut kit: RKit[E, D, RES], a: Int, count: Int, stride_in: Int, off: Int
) raises -> Int:
    """`_Kit.strided(A, count, stride, off)`: A's values at off + t stride
    as 1 x count (the TAKE_COLS move with one column index)."""
    var stride = off + 1 if count == 1 else stride_in
    var idx = kit.constant(Float32(off))
    var n = kit.r(a) * kit.c(a)
    return kit.move(a, idx, 1, count, _MV_TAKE_COLS, count, 1, stride, 0, 0, 0, n, count, 1, RES)


def kit_diag[E: Exec, D: RDev, RES: Bool](mut kit: RKit[E, D, RES], a: Int) raises -> Int:
    """`_Kit.diag(A)`: the diagonal of square A as 1 x n."""
    var n = kit.r(a)
    return kit_strided(kit, a, n, n + 1, 0)


def kit_copy[E: Exec, D: RDev, RES: Bool](mut kit: RKit[E, D, RES], a: Int) raises -> Int:
    """`_Kit.copy(A)`: on the device (one PLACE_COLS move) when A is there,
    else a host copy."""
    var rows = kit.r(a)
    var cols = kit.c(a)
    var n = rows * cols
    comptime if RES:
        if n and kit.mats[a].dev >= 0:
            var out = kit.device_output(rows, cols)
            var one = kit.zeros(1, 1)
            var source = kit.device(a)
            var index = kit.device(one)
            D.move(source, index, kit.mats[out].dev, MV_PLACE_COLS, n, cols, cols, 0, 0, 0, n, n, 1)
            return out
    return kit.copy_host(a)


def kit_place_strided[E: Exec, D: RDev, RES: Bool](
    mut kit: RKit[E, D, RES], dst: Int, v: Int, stride: Int, off: Int
) raises -> Int:
    """`_Kit.place_strided(D, V, stride, off)`: D[off + t stride] = V[t] in
    place (the PLACE_COLS move with width 1); returns D."""
    var count = kit.r(v) * kit.c(v)
    if count == 0:
        return dst
    var total = kit.r(dst) * kit.c(dst)
    var one = kit.zeros(1, 1)
    comptime if RES:
        var source = kit.device(v)
        var index = kit.device(one)
        var target = kit.device(dst)
        D.move(source, index, target, MV_PLACE_COLS, count, 1, stride, off, 0, 0, count, total, 1)
        return dst
    move_run(MV_PLACE_COLS, kit.ptr(v), kit.ptr(one), kit.ptr(dst), count, 1, stride, off, 0, 0, count, total, 1)
    return dst


def kit_diag_add[E: Exec, D: RDev, RES: Bool](mut kit: RKit[E, D, RES], a: Int, v: Int) raises -> Int:
    """`_Kit.diag_add(A, v)`: a copy of square A with v added to its diagonal."""
    var copied = kit_copy(kit, a)
    var diagonal = kit_diag(kit, a)
    var added = kit.ew(OP_ADD, diagonal, v)
    var stride = kit.r(a) + 1
    return kit_place_strided(kit, copied, added, stride, 0)


def kit_eye[E: Exec, D: RDev, RES: Bool](mut kit: RKit[E, D, RES], n: Int) raises -> Int:
    """`_eye(n)`: the host n x n identity."""
    var out = kit.zeros(n, n)
    var p = kit.ptr(out)
    for i in range(n):  # driver-loop: over the identity's diagonal (model structure: components)
        p[i * n + i] = 1.0
    return out


def kit_inv[E: Exec, D: RDev, RES: Bool, X: DecompDev](mut kit: RKit[E, D, RES], a: Int) raises -> Int:
    """`_inv(k, A)`: A^-1 through the LU solve of the identity."""
    var piv = List[Int32]()
    var factored = kit_lu[E, D, RES, X](kit, a, piv)
    if factored[1] != 0:
        raise Error("singular matrix")
    var n = kit.r(a)
    var eye = kit_eye(kit, n)
    var solved = kit_lu_solve(kit, factored[0], piv, eye)
    _ = piv^
    return solved


def kit_logdet[E: Exec, D: RDev, RES: Bool, X: DecompDev](mut kit: RKit[E, D, RES], a: Int) raises -> Float64:
    """`_logdet(k, A)`: the sum of log|diag(LU)| (the LU's info not read)."""
    var piv = List[Int32]()
    var factored = kit_lu[E, D, RES, X](kit, a, piv)
    var diagonal = kit_diag(kit, factored[0])
    var absolute = kit.ew(OP_ABS, diagonal)
    var logs = kit.ew(OP_LOGS, absolute, scalar=1.1754943508222875e-38)
    var value = kit.total(logs)
    _ = piv^
    return value
