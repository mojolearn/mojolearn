# SPDX-License-Identifier: Apache-2.0
# Typed descriptor bridge to the existing SVC epilogue owners. No numerical changes.
from std.memory import bitcast
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
    device_select,
    device_splitmix_perm,
)


def _ix(value: Int) raises -> Int:
    if value < 0:
        raise Error("svc epilogue: negative size")
    return value


def _f64_bits(value: Float64) -> UInt64:
    return bitcast[DType.uint64](value)

def svc_native_epilogue(dec_addr: Int, pairs_addr: Int, ab_addr: Int, out_addr: Int,
                        params: List[Int], floats: List[Float64]) raises -> Int:
    if len(params) != 5:
        raise Error("svc_pair_epilogue: params must contain 5 values")
    var mode = _ix(params[0])
    var dst = out_addr
    if dst == 0:
        raise Error("svc_pair_epilogue: null output")
    if mode == GLUE_GATHER:
        var nsr = _ix(params[1])
        var nsc = _ix(params[2])
        var nor = _ix(params[3])
        var noc = _ix(params[4])
        var ra = pairs_addr
        var ca = ab_addr
        if (ra == 0 and nor != nsr) or (ca == 0 and noc != nsc):
            raise Error("svc gather: an identity axis must keep its length")
        var sa = dec_addr
        device_gather(sa, nsr, nsc, ra, nor, ca, noc, dst)
        return nor
    if mode == GLUE_SELECT:
        var n = _ix(params[1])
        var ci = params[2]
        var cj = params[3]
        var cout = params[4]
        var ca = dec_addr
        var cb = pairs_addr
        var la = ab_addr
        var count = 0
        count = device_select(ca, n, ci, cj, cb, dst, la, cout)
        return count
    if mode == GLUE_FOLD:
        var m = _ix(params[1])
        var begin = _ix(params[2])
        var end = _ix(params[3])
        var lo = params[4]
        if begin > end or end > m:
            raise Error("svc fold: bad fold range")
        var ia = dec_addr
        var pa2 = pairs_addr
        var la = ab_addr
        var npos = 0
        npos = device_fold_split(ia, pa2, la, m, begin, end, dst, lo)
        return npos
    if mode == GLUE_FOLD_FINISH:
        var m = _ix(params[1])
        var ca = params[2]
        var lba = params[3]
        var da2 = dec_addr
        var pa2 = pairs_addr
        var la = ab_addr
        device_fold_finish(da2, pa2, la, ca, m, dst, lba)
        return m
    if mode == GLUE_C_ROWS:
        var n = _ix(params[1])
        var k = _ix(params[2])
        var c = _f64_bits(floats[3])
        var swa = dec_addr
        var coa = pairs_addr
        var cwa = ab_addr
        device_c_rows(swa, coa, cwa, k, c, n, dst)
        return n
    var n = _ix(params[1])
    var n_pairs = _ix(params[2])
    var k = _ix(params[3])
    var label1 = _f64_bits(floats[4])
    if mode > EPI_BINARY_CODES:
        raise Error("svc_pair_epilogue: unknown mode")
    if n == 0:
        return 0
    if n_pairs < 1 or k < 2:
        raise Error("svc_pair_epilogue: needs a pair and two classes")
    var pa = pairs_addr
    if mode != EPI_BINARY_CODES:
        check_pairs(pa, n_pairs, k)
    var da = dec_addr
    var aa = ab_addr if mode == EPI_PROBA or mode == EPI_LOG_PROBA else 0
    device_pair_epilogue(mode, da, n, n_pairs, k, pa, aa, label1, dst)
    return n

def svc_native_platt(dec_addr: Int, labels_addr: Int, out_addr: Int, n: Int) raises -> Int:
    var count = _ix(n)
    var oa = out_addr
    var da = dec_addr
    var la = labels_addr
    if count == 0:
        raise Error("svc_platt_train: no decision values")
    device_platt_train(da, la, count, oa)
    return count

def svc_native_splitmix(out_addr: Int, n: Int, seed_lo: Int, seed_hi: Int) raises -> Int:
    var count = _ix(n)
    if count == 0:
        return 0
    if count > 2147483647:
        raise Error("svc_splitmix_perm: n exceeds int32")
    var lo = _ix(seed_lo)
    var hi = _ix(seed_hi)
    var dst = out_addr
    device_splitmix_perm(dst, count, lo, hi)
    return count
