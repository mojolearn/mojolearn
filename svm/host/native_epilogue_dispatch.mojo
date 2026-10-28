# SPDX-License-Identifier: Apache-2.0
# Existing CPU epilogues reached by the shared typed descriptor transport.
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


def _ix(v: Int) raises -> Int:
    var x = v
    if x < 0:
        raise Error("svc epilogue: negative size")
    return x


def _f64_bits(v: Float64) raises -> UInt64:
    return bitcast[DType.uint64](v)


def svc_host_epilogue(
    dec_addr: Int, pairs_addr: Int, ab_addr: Int,
    out_addr: Int, params: List[Int], floats: List[Float64],
) raises -> Int:
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
        var sa = dec_addr if nsr * nsc > 0 else dst
        host_gather(sa, nsr, nsc, ra, nor, ca, noc, dst)
        return nor
    if mode == GLUE_SELECT:
        var n = _ix(params[1])
        var ci = params[2]
        var cj = params[3]
        var cout = params[4]
        var cb = pairs_addr
        if n == 0:
            return 0
        var count = host_select(dec_addr, n, ci, cj, cb, dst, ab_addr, cout)
        return count
    if mode == GLUE_FOLD:
        var m = _ix(params[1])
        var begin = _ix(params[2])
        var end = _ix(params[3])
        var lo = params[4]
        if begin > end or end > m:
            raise Error("svc fold: bad fold range")
        var npos = 0
        if m > 0:
            npos = host_fold_split(
                dec_addr, pairs_addr, ab_addr, m, begin, end, dst, lo
            )
        return npos
    if mode == GLUE_FOLD_FINISH:
        var m = _ix(params[1])
        host_fold_finish(
            dec_addr, pairs_addr, ab_addr, params[2], m, dst,
            params[3],
        )
        return m
    if mode == GLUE_C_ROWS:
        var n = _ix(params[1])
        var k = _ix(params[2])
        var c = _f64_bits(floats[3])
        var swa = dec_addr
        var coa = pairs_addr
        var cwa = ab_addr
        host_c_rows(swa, coa, cwa, k, c, n, dst)
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
    var da = dec_addr
    var pa = pairs_addr if mode != EPI_BINARY_CODES else dst
    var aa = ab_addr if mode == EPI_PROBA or mode == EPI_LOG_PROBA else dst
    if mode != EPI_BINARY_CODES:
        host_check_pairs(pa, n_pairs, k)
    host_pair_epilogue(mode, da, n, n_pairs, k, pa, aa, label1, dst)
    return n


def svc_host_platt(
    dec_addr: Int, labels_addr: Int, out_addr: Int, n: Int
) raises -> Int:
    var count = _ix(n)
    if count == 0:
        raise Error("svc_platt_train: no decision values")
    var da = dec_addr
    var la = labels_addr
    var oa = out_addr
    host_platt_train(da, la, count, oa)
    return count


def svc_host_splitmix(
    out_addr: Int, n: Int, seed_lo: Int, seed_hi: Int
) raises -> Int:
    var count = _ix(n)
    if count == 0:
        return 0
    if count > 2147483647:
        raise Error("svc_splitmix_perm: n exceeds int32")
    var lo = _ix(seed_lo)
    var hi = _ix(seed_hi)
    var oa = out_addr
    host_splitmix_perm(oa, count, lo, hi)
    return count
