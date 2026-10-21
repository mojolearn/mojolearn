# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Python entry points for `svm/impl/svc_epilogue.mojo` (the shell): they
unpack PythonObject arguments, check sizes, call the typed device launchers
and box the result."""

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
from svm.impl.svc_epilogue import (
    check_pairs,
    device_c_rows,
    device_fold_finish,
    device_fold_split,
    device_gather,
    device_pair_epilogue,
    device_platt_train,
    device_portable_math,
    device_select,
    device_splitmix_perm,
)


def _ix(v: PythonObject) raises -> Int:
    var x = Int(py=v)
    if x < 0:
        raise Error("svc epilogue: negative size")
    return x


def _f64_bits(v: PythonObject) raises -> UInt64:
    return bitcast[DType.uint64](Float64(py=v))


def svc_pair_epilogue_device_binding(
    dec_addr: PythonObject, pairs_addr: PythonObject, ab_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params[0] is the mode. Modes 0..5 (params [mode, n_rows, n_pairs,
    n_classes, label1]): 0 ovo (float32 n x P), 1 ovr (float64 n x K), 2
    vote codes (int64 n), 3 probabilities (float64 n x K; ab = 2P float64
    (A, B) per pair), 4 their logs, 5 binary codes from the device's labels
    (int64 n); returns n.
    6 GATHER: dec=src float32, pairs=row indices int32 or 0, ab=column
      indices int32 or 0, out float32; params [6, n_src_rows, n_src_cols,
      n_out_rows, n_out_cols]; returns n_out_rows.
    7 SELECT: dec=class codes int32 (n), pairs=per-row bounds float32 or 0,
      ab=labels out float32 (n), out=indices int32 (n); params [7, n, ci, cj,
      bounds-out address or 0]; returns the count.
    8 C_ROWS: dec=sample weights float64 or 0, pairs=class codes int32 or 0,
      ab=class weights float64 (K) or 0, out float32 (n); params [8, n, K,
      C, 0]; returns n.
    9 FOLD (lane/cgr-kernel): dec=pair rows int32 (m), pairs=shuffle int32
      (m), ab=pair labels float32 (m), out=training rows then held rows
      int32 (m); params [9, m, begin, end, training-labels-out address
      (float32, m - (end - begin))]; returns the class-i training count.
    10 FOLD_FINISH: dec=decisions float32 in shuffle order (m), pairs=shuffle
      int32, ab=pair labels float32, out=decisions float64 in pair row order
      (m); params [10, m, fold constants address (10 float32: flag, value per
      fold), labels-out address (float64 +1/-1, m), 0]; returns m."""
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
        var sa = Int(py=dec_addr)
        with GILReleased(Python()):
            device_gather(sa, nsr, nsc, ra, nor, ca, noc, dst)
        return PythonObject(nor)
    if mode == GLUE_SELECT:
        var n = _ix(params[1])
        var ci = Int(py=params[2])
        var cj = Int(py=params[3])
        var cout = Int(py=params[4])
        var ca = Int(py=dec_addr)
        var cb = Int(py=pairs_addr)
        var la = Int(py=ab_addr)
        var count = 0
        with GILReleased(Python()):
            count = device_select(ca, n, ci, cj, cb, dst, la, cout)
        return PythonObject(count)
    if mode == GLUE_FOLD:
        var m = _ix(params[1])
        var begin = _ix(params[2])
        var end = _ix(params[3])
        var lo = Int(py=params[4])
        if begin > end or end > m:
            raise Error("svc fold: bad fold range")
        var ia = Int(py=dec_addr)
        var pa2 = Int(py=pairs_addr)
        var la = Int(py=ab_addr)
        var npos = 0
        with GILReleased(Python()):
            npos = device_fold_split(ia, pa2, la, m, begin, end, dst, lo)
        return PythonObject(npos)
    if mode == GLUE_FOLD_FINISH:
        var m = _ix(params[1])
        var ca = Int(py=params[2])
        var lba = Int(py=params[3])
        var da2 = Int(py=dec_addr)
        var pa2 = Int(py=pairs_addr)
        var la = Int(py=ab_addr)
        with GILReleased(Python()):
            device_fold_finish(da2, pa2, la, ca, m, dst, lba)
        return PythonObject(m)
    if mode == GLUE_C_ROWS:
        var n = _ix(params[1])
        var k = _ix(params[2])
        var c = _f64_bits(params[3])
        var swa = Int(py=dec_addr)
        var coa = Int(py=pairs_addr)
        var cwa = Int(py=ab_addr)
        with GILReleased(Python()):
            device_c_rows(swa, coa, cwa, k, c, n, dst)
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
    var pa = Int(py=pairs_addr)
    if mode != EPI_BINARY_CODES:
        check_pairs(pa, n_pairs, k)
    var da = Int(py=dec_addr)
    var aa = Int(py=ab_addr) if mode == EPI_PROBA or mode == EPI_LOG_PROBA else 0
    with GILReleased(Python()):
        device_pair_epilogue(mode, da, n, n_pairs, k, pa, aa, label1, dst)
    return PythonObject(n)


def svc_platt_train_device_binding(
    dec_addr: PythonObject, labels_addr: PythonObject, out_addr: PythonObject, n: PythonObject
) raises -> PythonObject:
    """libsvm's `sigmoid_train` over n float64 decision values and +1/-1
    float64 labels, the row sums on the device; writes (A, B) as two float64
    at out_addr. Returns n."""
    var count = _ix(n)
    var oa = Int(py=out_addr)
    var da = Int(py=dec_addr)
    var la = Int(py=labels_addr)
    if count == 0:
        raise Error("svc_platt_train: no decision values")
    with GILReleased(Python()):
        device_platt_train(da, la, count, oa)
    return PythonObject(count)


def svc_splitmix_perm_device_binding(
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
    var dst = Int(py=out_addr)
    with GILReleased(Python()):
        device_splitmix_perm(dst, count, lo, hi)
    return PythonObject(count)


def svc_portable_math_device_binding(
    in_addr: PythonObject, out_addr: PythonObject, n: PythonObject, which: PythonObject
) raises -> PythonObject:
    """THE TWIN'S OWN CHECK DOOR: `sp_exp` (which 0) or `sf64_log` (which 1)
    over n float64 on the device, so a job can hold them to
    `_portable_math.exp/log` bit for bit. Returns n."""
    var count = _ix(n)
    if count == 0:
        return PythonObject(0)
    var w = Int(py=which)
    var ia = Int(py=in_addr)
    var oa = Int(py=out_addr)
    with GILReleased(Python()):
        device_portable_math(ia, oa, count, w)
    return PythonObject(count)
