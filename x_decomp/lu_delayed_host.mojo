# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/lu-delayed-update (2026-10-11): MOJOLEARN_IDN_LU_DELAYED_UPDATE's switch,
widths and HOST COLUMN (`lu_delayed_host`, run by x_decomp/host.mojo
`HostExec.lu`). THE ORDER is stated once, in x_decomp/lu_delayed.mojo's
module docstring (the device half). No GPU import: this file links into the
host binding.
"""

from std.sys.compile import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_chunk, host_predict_task_count
from gemm.contract import CONTRACT_K_LEAF_MIN, contract_leaf_size, leaf_begin, leaf_count, leaf_end
from x_decomp.cells import F32Ptr, I32Ptr, div0, lu_finish_host, lu_swap_elem

#: The switch: IDENTICAL builds with `-D MOJOLEARN_IDN_LU_DELAYED_UPDATE`.
#: Needs LU_FAST_STEP1 (the one-launch panel; IDENTICAL default) on the
#: device; the host twin keys on the same two so the columns agree.
#: (LU_FAST_STEP1's IDENTICAL condition is spelled here, not imported: this
#: file is the host column's and imports nothing GPU; x_decomp/lu_delayed.mojo
#: asserts the two agree.)
comptime IDN_LU_DELAYED_UPDATE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (is_defined["MOJOLEARN_IDN_GATES_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
    and not is_defined["MOJOLEARN_LU_FAST_STEP1_OFF"]()
    and is_defined["MOJOLEARN_IDN_LU_DELAYED_UPDATE"]()
)
#: The inner panel width (main's LU_PANEL_NB; NB128's loss was widening it).
comptime LDP = 32
#: The outer block, the big update's k. 256 = two contract leaves: wide enough
#: that the trailing square is touched 8x less often than at 32, narrow enough
#: that the in-block updates (n x <= 224 at k = 32) stay a small share. A
#: cost rule on the GEMM's leaf size, not on any matrix dimension.
comptime LDU_NB = 256
#: The host fold buffer: leaves of one BU call (k <= LDU_NB).
comptime _LDU_MAX_LEAVES = 8



def _ldu_par[FuncType: def(Int) -> None](ref func: FuncType, n: Int):
    """Tasks 0 .. n - 1 in contiguous groups on the host pool (x_decomp/host.mojo
    `xd_parallel`'s split). Every task writes only its own row, so the words
    are the serial loop's."""
    var groups = host_predict_task_count(n)
    var chunk = host_predict_chunk(n, groups)

    def group(g: Int) {imm func, imm n, imm chunk}:
        for i in range(g * chunk, min(n, (g + 1) * chunk)):
            func(i)

    if groups <= 1:
        group(0)
        return
    host_parallelize(group, groups)


def _ldu_bu_host(a: F32Ptr, act: F32Ptr, n: Int, r0: Int, c0: Int, p0: Int, m: Int, nc: Int, k: Int):
    """BU(p0, k) on the host: the contract's cell (gemm/host/gemm_oracle.mojo
    `gemm_oracle_cell` at contract_leaf_size(k)), then one subtraction.
    Rows are tasks; each row reads only L columns [p0, p0 + k) and U rows
    [p0, p0 + k), which no task writes."""
    if m <= 0 or nc <= 0 or k <= 0:
        return
    var leaf = contract_leaf_size(k)
    var pc = leaf_count(k, leaf)
    comptime assert LDU_NB <= _LDU_MAX_LEAVES * CONTRACT_K_LEAF_MIN, "lu_delayed: a BU call (k <= LDU_NB) has at most this many leaves"

    def row(t: Int) {imm a, imm act, imm n, imm r0, imm c0, imm p0, imm nc, imm k, imm leaf, imm pc}:
        var i = r0 + t
        var buf = InlineArray[Float32, _LDU_MAX_LEAVES](fill=Float32(0))
        for jj in range(nc):
            var j = c0 + jj
            for q in range(pc):
                var g = Float32(0)
                for p in range(leaf_begin(q, leaf), leaf_end(q, leaf, k)):
                    var lv = Float32(0)
                    if act.unsafe_load(p0 + p) != Float32(0):
                        lv = ftz(a.unsafe_load(i * n + p0 + p))
                    g = ftz(identical_mul_add(lv, ftz(a.unsafe_load((p0 + p) * n + j)), g))
                buf[q] = ftz(g)
            # the contract's balanced tree: pairs (2q, 2q + 1), an odd last
            # node carried unchanged, then the output seam
            var w = pc
            while w > 1:
                var pairs = w // 2
                for q in range(pairs):
                    buf[q] = ftz(ftz(buf[2 * q]) + ftz(buf[2 * q + 1]))
                if w % 2 != 0:
                    buf[pairs] = buf[w - 1]
                w = pairs + w % 2
            var gg = ftz(buf[0])
            a.unsafe_store(i * n + j, ftz(ftz(a.unsafe_load(i * n + j)) - ftz(gg)))

    _ldu_par(row, m)


def _ldu_cols_host(a: F32Ptr, piv: I32Ptr, act: F32Ptr, k0: Int, k1: Int, jlo: Int, jtri: Int, n: Int, swaps: Bool):
    """`lfm_swaps_trsm_kernel`'s statements, columns [jlo, n) as tasks."""
    if n <= jlo:
        return

    def col(t: Int) {imm a, imm piv, imm act, imm k0, imm k1, imm jlo, imm jtri, imm n, imm swaps}:
        var j = jlo + t
        if swaps:
            for k in range(k0, k1):
                lu_swap_elem(a, piv, k, j, n)
        if j < jtri:
            for k in range(k0 + 1, k1):
                var acc = ftz(a.unsafe_load(k * n + j))
                for kp in range(k0, k):
                    if act.unsafe_load(kp) != Float32(0):
                        var l = a.unsafe_load(k * n + kp)
                        acc = ftz(identical_mul_add(-l, ftz(a.unsafe_load(kp * n + j)), ftz(acc)))
                a.unsafe_store(k * n + j, acc)

    _ldu_par(col, n - jlo)


def lu_delayed_host(a: F32Ptr, piv: I32Ptr, n: Int, info: F32Ptr):
    """THE ORDER above on the host (IDN_LU_DELAYED_UPDATE's host column)."""
    info.unsafe_store(0, Float32(0))
    var actl = List[Float32](length=max(n, 1), fill=Float32(0))
    var act = F32Ptr(unsafe_from_address=Int(actl.unsafe_ptr()))
    var K0 = 0
    while K0 < n:
        var K1 = min(K0 + LDU_NB, n)
        var k0 = K0
        while k0 < K1:
            var k1 = min(k0 + LDP, K1)
            # 1. the panel (`lu_fast_panel`'s statements; NB128's host panel at 32)
            for k in range(k0, k1):
                var p = k
                var best = abs(ftz(a.unsafe_load(k * n + k)))
                for i in range(k + 1, n):
                    var v = abs(ftz(a.unsafe_load(i * n + k)))
                    if v > best:
                        best = v
                        p = i
                piv.unsafe_store(k, Int32(p))
                if p != k:
                    for j in range(k1):
                        var t = a.unsafe_load(k * n + j)
                        a.unsafe_store(k * n + j, a.unsafe_load(p * n + j))
                        a.unsafe_store(p * n + j, t)
                var d = ftz(a.unsafe_load(k * n + k))
                if d == Float32(0):
                    if info.unsafe_load(0) == Float32(0):
                        info.unsafe_store(0, Float32(k + 1))
                    act.unsafe_store(k, Float32(0))
                    continue
                act.unsafe_store(k, Float32(1))
                for i in range(k + 1, n):
                    var l = div0(a.unsafe_load(i * n + k), d)
                    a.unsafe_store(i * n + k, l)
                    for j in range(k + 1, k1):
                        a.unsafe_store(
                            i * n + j,
                            ftz(identical_mul_add(-l, ftz(a.unsafe_load(k * n + j)), ftz(a.unsafe_load(i * n + j)))),
                        )
            # 2. every later column's swaps; the block's U rows
            _ldu_cols_host(a, piv, act, k0, k1, k1, K1, n, True)
            # 3. the block's remaining columns
            _ldu_bu_host(a, act, n, k1, k1, k0, n - k1, K1 - k1, k1 - k0)
            k0 = k1
        if K1 < n:
            var c = K0
            while c < K1:
                var ce = min(c + LDP, K1)
                _ldu_cols_host(a, piv, act, c, ce, K1, n, n, False)
                _ldu_bu_host(a, act, n, ce, K1, c, K1 - ce, n - K1, ce - c)
                c = ce
            _ldu_bu_host(a, act, n, K1, K1, K0, n - K1, n - K1, K1 - K0)
        K0 = K1
    lu_finish_host(a, n * n)
    _ = actl^
