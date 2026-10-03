# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SVC's binary64 epilogues and row glue, THE CPU HOST COLUMN (the CPU
binding `_mojolearn_svm_host` only; cgfin-c-svm, 2026-10-02).

The GPU binding runs these on the device (`svm/impl/svc_epilogue.mojo`).
This module runs the SAME per-row functions (`svm/impl/svc_rows.mojo`:
software binary64 over UInt64 words, integer instructions only) in host
loops, so the CPU column computes the device's words:

  * the epilogues of `decision_function`, `predict`, `predict_proba` and
    `predict_log_proba` (`epilogue_row`), rows split across host tasks by
    shape only (no row's bits depend on the split);
  * Platt's `sigmoid_train` (`platt_solve`) over `tree_sum_sf64`, the
    device fold's chunk-and-level order;
  * the probability shuffle (the Feistel bijection keyed by
    `shuffle_seed32`);
  * the row glue (gather, pair selection, per-row bounds) as plain loops.

`python/mojolearn/_svm_impl.py` keeps its Python reference of the libsvm
arithmetic (`_sigmoid_train`, `_splitmix_perm`, ...) in the same orders.
"""

from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_chunk, host_predict_task_count
from core.shuffle_iterator import FeistelBijection
from checks.soft_f64 import sf64_log
from svm.impl.svc_rows import (
    EPI_BINARY_CODES,
    EPI_LOG_PROBA,
    EPI_OVO,
    EPI_OVR,
    EPI_PROBA,
    GLUE_C_ROWS,
    GLUE_FOLD,
    GLUE_FOLD_FINISH,
    GLUE_GATHER,
    GLUE_SELECT,
    I32P,
    PlattSums,
    ST_OK,
    U32P,
    U64P,
    _F32_ONE,
    c_row,
    epilogue_row,
    epilogue_scratch,
    fold_finish_cell,
    fold_split_cell,
    gather_cell,
    platt_channels,
    platt_solve,
    platt_terms,
    shuffle_seed32,
    sp_exp,
    tree_sum_sf64,
)


def pair_epilogue(
    mode: Int, dec: U32P, n: Int, n_pairs: Int, k: Int, pi: I32P, ab: U64P,
    label1: UInt64, out_addr: Int,
) raises:
    """`epilogue_row` over the n rows, split across host tasks."""
    var tasks = host_predict_task_count(n)
    if tasks < 1:
        tasks = 1
    if tasks > n:
        tasks = n
    var chunk = host_predict_chunk(n, tasks)
    var failed = List[Int](length=tasks, fill=0)
    var fp = failed.unsafe_ptr()
    var per_row = epilogue_scratch(mode, k)
    var out32 = U32P(unsafe_from_address=out_addr)
    var out64 = U64P(unsafe_from_address=out_addr)

    def _rows(c: Int) {imm mode, imm dec, imm n, imm n_pairs, imm k, imm pi, imm ab, imm label1, imm chunk, imm fp, imm per_row, imm out32, imm out64}:
        var scr = List[UInt64](length=max(1, per_row), fill=UInt64(0))
        var sp = rebind[U64P](scr.unsafe_ptr())
        var lo = c * chunk
        var hi = min(lo + chunk, n)
        for r in range(lo, hi):
            var st = epilogue_row(mode, dec, n, n_pairs, k, pi, ab, label1, r, sp, out32, out64)
            if st != ST_OK:
                fp[c] = 1
                break
        _ = scr^

    if tasks == 1:
        _rows(0)
    else:
        host_parallelize(_rows, tasks)
    for c in range(tasks):
        if failed[c] != 0:
            raise Error("svc epilogue: a row raised (float division by zero or math domain error)")


struct HostPlatt(PlattSums, Movable):
    """The Platt sums on the host: `platt_terms` per row, then the device
    fold's order (`tree_sum_sf64`) per channel."""

    var dec: MutPointer[UInt64, MutUntrackedOrigin]
    var lab: MutPointer[UInt64, MutUntrackedOrigin]
    var n: Int

    def __init__(out self, dec_addr: Int, lab_addr: Int, n: Int):
        self.dec = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=dec_addr)
        self.lab = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=lab_addr)
        self.n = n

    def platt_sums(
        mut self, mode: Int, a: UInt64, b: UInt64, hi_t: UInt64, lo_t: UInt64
    ) raises -> SIMD[DType.uint64, 8]:
        var nch = platt_channels(mode)
        var cols = List[List[UInt64]]()
        for _ in range(nch):
            cols.append(List[UInt64](length=self.n, fill=UInt64(0)))
        for i in range(self.n):
            var v = platt_terms(mode, self.dec[i], self.lab[i], a, b, hi_t, lo_t)
            for ch in range(nch):
                cols[ch][i] = v[ch]
        var out = SIMD[DType.uint64, 8](0)
        for ch in range(nch):
            out[ch] = tree_sum_sf64(cols[ch])
        return out


def shuffle_perm(n: Int, seed: UInt64, dst: I32P):
    """The probability shuffle: `dst[i]` = the Feistel bijection at i."""
    var f = FeistelBijection(n, shuffle_seed32(seed))
    for i in range(n):
        dst[i] = Int32(f(i))


# ------------------------------------------------------------ Python doors
def _ix(v: PythonObject) raises -> Int:
    var x = Int(py=v)
    if x < 0:
        raise Error("svc epilogue: negative size")
    return x


def _f64_bits(v: PythonObject) raises -> UInt64:
    return bitcast[DType.uint64](Float64(py=v))


def _glue_select(codes: I32P, n: Int, ci: Int, cj: Int, cvec: U32P, has_c: Bool,
                 idx: I32P, lab: U32P, cout: U32P) -> Int:
    var m = 0
    for i in range(n):
        var c = Int(codes[i])
        if c == ci or c == cj:
            idx[m] = Int32(i)
            lab[m] = _F32_ONE if c == cj else UInt32(0)
            if has_c:
                cout[m] = cvec[i]
            m += 1
    return m


def svc_pair_epilogue_binding(
    dec_addr: PythonObject, pairs_addr: PythonObject, ab_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """The host twin of `svc_pair_epilogue_device_binding`
    (svm/impl/svc_epilogue.mojo), same modes and params."""
    if len(params) != 5:
        raise Error("svc_pair_epilogue: params must contain 5 values")
    var mode = _ix(params[0])
    var dst = Int(py=out_addr)
    if dst == 0:
        raise Error("svc_pair_epilogue: null output")
    if mode == GLUE_GATHER:
        var nsr = _ix(params[1])
        var nsc = _ix(params[2])
        var nor = _ix(params[3])
        var noc = _ix(params[4])
        var ra = Int(py=pairs_addr)
        var ca = Int(py=ab_addr)
        if (ra == 0 and nor != nsr) or (ca == 0 and noc != nsc):
            raise Error("svc gather: an identity axis must keep its length")
        var src = U32P(unsafe_from_address=Int(py=dec_addr)) if nsr * nsc > 0 else U32P(unsafe_from_address=dst)
        var rows = I32P(unsafe_from_address=ra if ra != 0 else dst)
        var cols = I32P(unsafe_from_address=ca if ca != 0 else dst)
        var out = U32P(unsafe_from_address=dst)
        for cell in range(nor * noc):
            if gather_cell(cell, src, nsr, nsc, rows, ra != 0, cols, ca != 0, noc, out) != ST_OK:
                raise Error("svc gather: an index is outside the source matrix")
        return PythonObject(nor)
    if mode == GLUE_SELECT:
        var n = _ix(params[1])
        var ci = Int(py=params[2])
        var cj = Int(py=params[3])
        var cout = Int(py=params[4])
        var cb = Int(py=pairs_addr)
        var has_c = cb != 0 and cout != 0
        if n == 0:
            return PythonObject(0)
        var count = _glue_select(
            I32P(unsafe_from_address=Int(py=dec_addr)), n, ci, cj,
            U32P(unsafe_from_address=cb if has_c else dst), has_c,
            I32P(unsafe_from_address=dst), U32P(unsafe_from_address=Int(py=ab_addr)),
            U32P(unsafe_from_address=cout if has_c else dst),
        )
        return PythonObject(count)
    if mode == GLUE_FOLD:
        var m = _ix(params[1])
        var begin = _ix(params[2])
        var end = _ix(params[3])
        var lo = Int(py=params[4])
        if begin > end or end > m:
            raise Error("svc fold: bad fold range")
        var npos = 0
        if m > 0:
            var idx = I32P(unsafe_from_address=Int(py=dec_addr))
            var perm = I32P(unsafe_from_address=Int(py=pairs_addr))
            var lab = U32P(unsafe_from_address=Int(py=ab_addr))
            var lout = U32P(unsafe_from_address=lo if lo != 0 else dst)
            for k in range(m):
                npos += fold_split_cell(k, idx, perm, lab, m, begin, end, I32P(unsafe_from_address=dst), lout)
        return PythonObject(npos)
    if mode == GLUE_FOLD_FINISH:
        var m = _ix(params[1])
        var dp = U32P(unsafe_from_address=Int(py=dec_addr))
        var perm = I32P(unsafe_from_address=Int(py=pairs_addr))
        var lab = U32P(unsafe_from_address=Int(py=ab_addr))
        var consts = U32P(unsafe_from_address=Int(py=params[2]))
        var lbo = U64P(unsafe_from_address=Int(py=params[3]))
        for k in range(m):
            fold_finish_cell(k, dp, perm, lab, consts, m, U64P(unsafe_from_address=dst), lbo)
        return PythonObject(m)
    if mode == GLUE_C_ROWS:
        var n = _ix(params[1])
        var k = _ix(params[2])
        var c = _f64_bits(params[3])
        var swa = Int(py=dec_addr)
        var coa = Int(py=pairs_addr)
        var cwa = Int(py=ab_addr)
        var out = U32P(unsafe_from_address=dst)
        var sw = U64P(unsafe_from_address=swa if swa != 0 else dst)
        var codes = I32P(unsafe_from_address=coa if coa != 0 else dst)
        var cw = U64P(unsafe_from_address=cwa if cwa != 0 else dst)
        for i in range(n):
            if c_row(i, sw, swa != 0, codes, cw, cwa != 0, k, c, out) != ST_OK:
                raise Error("svc C rows: a class code is outside class_weight")
        return PythonObject(n)
    var n = _ix(params[1])
    var n_pairs = _ix(params[2])
    var k = _ix(params[3])
    var label1 = _f64_bits(params[4])
    if mode > EPI_BINARY_CODES:
        raise Error("svc_pair_epilogue: unknown mode")
    if n == 0:
        return PythonObject(0)
    if n_pairs < 1 or k < 2:
        raise Error("svc_pair_epilogue: needs a pair and two classes")
    var dec = U32P(unsafe_from_address=Int(py=dec_addr))
    var pa = Int(py=pairs_addr)
    var pi = I32P(unsafe_from_address=pa if mode != EPI_BINARY_CODES else dst)
    var aa = Int(py=ab_addr)
    var ab = U64P(unsafe_from_address=aa if mode == EPI_PROBA or mode == EPI_LOG_PROBA else dst)
    if mode != EPI_BINARY_CODES:
        for pr in range(n_pairs):
            var i = Int(pi[2 * pr])
            var j = Int(pi[2 * pr + 1])
            if i < 0 or j < 0 or i >= k or j >= k or i == j:
                raise Error("svc_pair_epilogue: invalid class pair")
    with GILReleased(Python()):
        pair_epilogue(mode, dec, n, n_pairs, k, pi, ab, label1, dst)
    return PythonObject(n)


def svc_platt_train_binding(
    dec_addr: PythonObject, labels_addr: PythonObject, out_addr: PythonObject, n: PythonObject
) raises -> PythonObject:
    """libsvm's `sigmoid_train` over n float64 decision values and +1/-1
    float64 labels; writes (A, B) as two float64 at out_addr. Returns n."""
    var count = _ix(n)
    if count == 0:
        raise Error("svc_platt_train: no decision values")
    var da = Int(py=dec_addr)
    var la = Int(py=labels_addr)
    var op = U64P(unsafe_from_address=Int(py=out_addr))
    with GILReleased(Python()):
        var src = HostPlatt(da, la, count)
        var r = platt_solve(src, count)
        op[0] = r[0]
        op[1] = r[1]
    return PythonObject(count)


def svc_splitmix_perm_binding(
    out_addr: PythonObject, n: PythonObject, seed_lo: PythonObject, seed_hi: PythonObject
) raises -> PythonObject:
    """The probability shuffle of `[0, n)` into n int32 at out_addr, the
    64-bit seed handed in as two 32-bit halves. Returns n."""
    var count = _ix(n)
    if count == 0:
        return PythonObject(0)
    if count > 2147483647:
        raise Error("svc_splitmix_perm: n exceeds int32")
    var lo = UInt64(_ix(seed_lo)) & UInt64(0xFFFFFFFF)
    var hi = UInt64(_ix(seed_hi)) & UInt64(0xFFFFFFFF)
    var op = I32P(unsafe_from_address=Int(py=out_addr))
    with GILReleased(Python()):
        shuffle_perm(count, (hi << 32) | lo, op)
    return PythonObject(count)


def svc_portable_math_binding(in_addr: PythonObject, out_addr: PythonObject, n: PythonObject, which: PythonObject) raises -> PythonObject:
    """THE TWIN'S OWN CHECK DOOR: `sp_exp` (which 0) or `sf64_log` (which 1)
    over n float64, so a job can hold them to `_portable_math.exp/log` (the
    C library) bit for bit on any sweep. Returns n."""
    var count = _ix(n)
    if count == 0:
        return PythonObject(0)
    var w = Int(py=which)
    var ip = U64P(unsafe_from_address=Int(py=in_addr))
    var op = U64P(unsafe_from_address=Int(py=out_addr))
    with GILReleased(Python()):
        for i in range(count):
            op[i] = sp_exp(ip[i]) if w == 0 else sf64_log(ip[i])
    return PythonObject(count)
