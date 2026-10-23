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

from x_decomp.api import eigh_run, lu_solve_run, rand_run
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.pca_randomized import RDev, RKit
from x_decomp.nmf import NmfArgs

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
        X.lu(da, kit.mats[lu].dev, pp, kit.ptr(info), n)
        return (lu, Int(kit.ptr(info)[0]))
    var lu = kit.copy_host(a)
    _exec_lu[E](kit.ptr(lu), pp, kit.ptr(info), n)
    return (lu, Int(kit.ptr(info)[0]))


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
