# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Python entry points for `svm/host/svc_proba.mojo` (the shell, CPU binding
`_mojolearn_svm_host`): they unpack PythonObject arguments, check sizes, call
the typed host-column functions and box the result."""

from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from svm.impl.svc_rows import (
    EPI_BINARY_CODES,
    EPI_LOG_PROBA,
    EPI_PROBA,
    GLUE_C_ROWS,
    GLUE_FOLD,
    GLUE_FOLD_FINISH,
    GLUE_GATHER,
    GLUE_SELECT,
)
from svm.host.svc_proba import (
    host_c_rows,
    host_check_pairs,
    host_fold_finish,
    host_fold_split,
    host_gather,
    host_pair_epilogue,
    host_platt_train,
    host_select,
    host_splitmix_perm,
)


def _ix(v: PythonObject) raises -> Int:
    var x = Int(py=v)
    if x < 0:
        raise Error("svc epilogue: negative size")
    return x


def _f64_bits(v: PythonObject) raises -> UInt64:
    return bitcast[DType.uint64](Float64(py=v))


def svc_pair_epilogue_binding(
    dec_addr: PythonObject, pairs_addr: PythonObject, ab_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """The host twin of `svc_pair_epilogue_device_binding`
    (svm/impl/svc_epilogue_pyapi.mojo), same modes and params."""
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
        var sa = Int(py=dec_addr) if nsr * nsc > 0 else dst
        host_gather(sa, nsr, nsc, ra, nor, ca, noc, dst)
        return PythonObject(nor)
    if mode == GLUE_SELECT:
        var n = _ix(params[1])
        var ci = Int(py=params[2])
        var cj = Int(py=params[3])
        var cout = Int(py=params[4])
        var cb = Int(py=pairs_addr)
        if n == 0:
            return PythonObject(0)
        var count = host_select(Int(py=dec_addr), n, ci, cj, cb, dst, Int(py=ab_addr), cout)
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
            npos = host_fold_split(
                Int(py=dec_addr), Int(py=pairs_addr), Int(py=ab_addr), m, begin, end, dst, lo
            )
        return PythonObject(npos)
    if mode == GLUE_FOLD_FINISH:
        var m = _ix(params[1])
        host_fold_finish(
            Int(py=dec_addr), Int(py=pairs_addr), Int(py=ab_addr), Int(py=params[2]), m, dst,
            Int(py=params[3]),
        )
        return PythonObject(m)
    if mode == GLUE_C_ROWS:
        var n = _ix(params[1])
        var k = _ix(params[2])
        var c = _f64_bits(params[3])
        var swa = Int(py=dec_addr)
        var coa = Int(py=pairs_addr)
        var cwa = Int(py=ab_addr)
        host_c_rows(swa, coa, cwa, k, c, n, dst)
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
    var da = Int(py=dec_addr)
    var pa = Int(py=pairs_addr) if mode != EPI_BINARY_CODES else dst
    var aa = Int(py=ab_addr) if mode == EPI_PROBA or mode == EPI_LOG_PROBA else dst
    if mode != EPI_BINARY_CODES:
        host_check_pairs(pa, n_pairs, k)
    with GILReleased(Python()):
        host_pair_epilogue(mode, da, n, n_pairs, k, pa, aa, label1, dst)
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
    var oa = Int(py=out_addr)
    with GILReleased(Python()):
        host_platt_train(da, la, count, oa)
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
    var lo = _ix(seed_lo)
    var hi = _ix(seed_hi)
    var oa = Int(py=out_addr)
    with GILReleased(Python()):
        host_splitmix_perm(oa, count, lo, hi)
    return PythonObject(count)


# TOMBSTONE: svc_portable_math (dead native entry: only the one-off py-consolidation check tools/py_consolidated/svc_equal.py called it) deleted 2026-10-26 by @@SHA@@; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_svm.patch

