# SPDX-License-Identifier: Apache-2.0
"""RUN piece of the native MinCovDet / EllipticEnvelope owner
(bindings/mcd_native.mojo is UNPACK and PACK): fast_mcd's one-feature
shortcut or the one XD_MCD call, correct_covariance, reweight_covariance,
`_pinvh`, `_mahal`, `_slogdet` and the offset's order statistics, over the
typed kit (bindings/decomp_kit_run.mojo). Same x_decomp operations, same
words, same order: the bits do not move. Supports are host '<u1' words in a
native list (were Python `array('B')` stores), copied once by PACK. No Python
(lane entangled-decompkit, 2026-11-12)."""
from std.math import ceil, floor, inf
from std.memory import bitcast
from bindings.decomp_kit_run import (
    DKit, DK_SUB, DK_ADD, DK_MUL, DK_SCALE, DK_ABS, DK_RECIP, DK_SELECT, DK_LOGS, DK_ADDS, DK_LE,
)
from core.decomp_native_impl import decomp_f32_to_u8, decomp_i32_to_u8, decomp_u8_to_f32, decomp_count_u8
from x_decomp.descriptor_schema import XD_MCD, XD_CONSISTENCY_FACTOR, XD_CHI2_QUANTILE

comptime MCD_F32_EPS = 1.1920928955078125e-07
comptime MCD_LOG_2PI = 1.8378770664093453
comptime MCD_TINY = 1.1754943508222875e-38
comptime McdU8 = MutPointer[UInt8, MutUntrackedOrigin]
comptime McdI32 = MutPointer[Int32, MutUntrackedOrigin]


def mcd_scalar(kit: DKit, op: Int, floats: List[Float64]) raises -> Float64:
    # One float64 result of the x_decomp table (XD_CONSISTENCY_FACTOR or
    # XD_CHI2_QUANTILE), written to a host word.
    var word = List[Float64](length=1, fill=0.0)
    var addresses: List[Int] = [Int(word.unsafe_ptr())]
    _ = kit.xd(op, addresses, List[Int](), floats)
    var value = word[0]
    _ = word^
    return value


def mcd_f32_below(x: Float64) -> Float32:
    # `_f32_below`: the largest float32 strictly below x.
    var f = Float32(x)
    if Float64(f) < x:
        return f
    var u = bitcast[DType.uint32](f)
    if f > 0.0:
        u -= 1
    elif f == 0.0:
        u = UInt32(0x80000001)
    else:
        u += 1
    return bitcast[DType.float32](u)


def mcd_pinvh(mut kit: DKit, a: Int) raises -> Int:
    # `_pinvh`: V diag(1/w) V^T over |w| > max|w| * n * float32 eps.
    var wv = kit.eigh(a)
    var w = wv[0]
    var wmax = Float64(0.0)
    if kit.ncols(w):
        var top = kit.reduce(w, 0)
        wmax = kit.word(top)
    var cut = Float64(Float32(wmax * Float64(kit.nrows(a)) * MCD_F32_EPS))
    var absolute = kit.ew(DK_ABS, w)
    var inverse = kit.ew(DK_RECIP, w)
    var zero = kit.zeros(1, 1)
    var kept = kit.ew(DK_SELECT, absolute, inverse, zero, cut)
    var scaled = kit.ew(DK_MUL, wv[1], kept)
    return kit.mm(scaled, wv[1], tb=True)


def mcd_emp_cov(mut kit: DKit, xs: Int, assume_centered: Bool) raises -> Int:
    # `_emp_cov`: (X - mean)^T (X - mean) / n.
    var rows = kit.nrows(xs)
    if assume_centered:
        var gram = kit.mm(xs, xs, ta=True)
        return kit.ew(DK_SCALE, gram, scalar=1.0 / Float64(rows))
    var mean = kit.colmean(xs)
    var centered = kit.ew(DK_SUB, xs, mean)
    var gram = kit.mm(centered, centered, ta=True)
    return kit.ew(DK_SCALE, gram, scalar=1.0 / Float64(rows))


def mcd_mahal(mut kit: DKit, x: Int, loc: Int, precision: Int) raises -> Int:
    var centered = kit.ew(DK_SUB, x, loc)
    var product = kit.mm(centered, precision)
    var terms = kit.ew(DK_MUL, product, centered)
    return kit.rowsum(terms)


def mcd_masked_cov(mut kit: DKit, m: Int, mask: Int, assume_centered: Bool) raises -> Tuple[Int, Int]:
    # `_masked_cov`: the location and covariance of the masked rows.
    var count = kit.total_word(mask)
    var masked = kit.ew(DK_MUL, m, mask)
    var loc: Int
    var centered: Int
    if assume_centered:
        loc = kit.zeros(1, kit.ncols(m))
        centered = masked
    else:
        var sums = kit.colsum(masked)
        loc = kit.ew(DK_SCALE, sums, scalar=1.0 / count)
        var shifted = kit.ew(DK_SUB, m, loc)
        centered = kit.ew(DK_MUL, shifted, mask)
    # TOMBSTONE: MOJOLEARN_MCD_ORDERED_COV (the x_decomp_dev_mcd_cov final covariance arm) deleted 2026-11-08 by 65b756872d; code recoverable at 41ffcec665.
    # Restore: git apply experiments/removed/MOJOLEARN_MCD_ORDERED_COV.patch
    var gram = kit.mm(centered, centered, ta=True)
    return (loc, kit.ew(DK_SCALE, gram, scalar=1.0 / count))


def mcd_slogdet(mut kit: DKit, a: Int) raises -> Tuple[Float64, Float64]:
    # `_slogdet`: (sign, log|det|) from the LU factor (sums ascending).
    var factor = kit.lu(a)
    var aux = kit.lu_aux(factor[0], factor[1])
    # The small aux vector comes down once; the three words below are then
    # host reads.
    kit.host(aux[0])
    if kit.word(aux[0], 1) > 0:
        return (Float64(0.0), -inf[DType.float64]())
    var negative = Int(kit.word(aux[0], 2)) + Int(kit.word(aux[0], 3))
    var absolute = kit.ew(DK_ABS, aux[1])
    var logs = kit.ew(DK_LOGS, absolute, scalar=MCD_TINY)
    var ld = kit.total_word(logs)
    return (Float64(-1.0) if negative % 2 != 0 else Float64(1.0), ld)


def mcd_mask_support(mut kit: DKit, mask: Int, n: Int) raises -> List[UInt8]:
    # An n x 1 0/1 float32 matrix as the '<u1' (n,) support words.
    var support = List[UInt8](length=max(n, 1), fill=0)
    decomp_f32_to_u8(kit.ptr(mask), McdU8(unsafe_from_address=Int(support.unsafe_ptr())), n)
    return support^


def mcd_int_support(kit: DKit, store: Int, n: Int) -> List[UInt8]:
    # `_support_of`: a 0/1 int32 membership store as the '<u1' (n,) support words.
    var support = List[UInt8](length=max(n, 1), fill=0)
    decomp_i32_to_u8(McdI32(unsafe_from_address=kit.iaddr(store)), McdU8(unsafe_from_address=Int(support.unsafe_ptr())),
                     n)
    return support^


struct McdOneFeature(Movable):
    var loc: Int
    var cov: Int
    var support: List[UInt8]
    var dist: Int

    def __init__(out self, loc: Int, cov: Int, var support: List[UInt8], dist: Int):
        self.loc = loc
        self.cov = cov
        self.support = support^
        self.dist = dist


def mcd_1d(mut kit: DKit, x: Int, h: Int) raises -> McdOneFeature:
    # `MinCovDet._mcd_1d_native`: (loc, cov, support, dist) for one feature.
    var n = kit.nrows(x)
    var order = kit.order(x)
    var xs = kit.take_rows(x, order)
    var loc: Int
    var support: List[UInt8]
    var chosen: Int
    if h < n:
        var upper = kit.rows_of(xs, h, n)
        var lower = kit.rows_of(xs, 0, n - h)
        var diff = kit.ew(DK_SUB, upper, lower)
        var starts = kit.argmin_all(diff)
        var ends = kit.take_rows(xs, starts, h)
        var begins = kit.take_rows(xs, starts)
        var sums = kit.ew(DK_ADD, ends, begins)
        var mids = kit.ew(DK_SCALE, sums, scalar=0.5)
        loc = kit.colmean(mids)
        var shifted = kit.ew(DK_SUB, x, loc)
        var centered = kit.ew(DK_ABS, shifted)
        var selected = kit.select_smallest(centered, h)
        chosen = kit.take_rows(x, selected[0])
        if selected[1] >= 0:
            support = mcd_mask_support(kit, selected[1], n)
        else:
            support = mcd_int_support(kit, selected[2], n)
    else:
        loc = kit.colmean(x)
        chosen = x
        # every row is in the support: `_support_of(array("i", [1]) * n, n)`
        support = List[UInt8](length=max(n, 1), fill=1)
    var cov = mcd_emp_cov(kit, chosen, False)
    var precision = mcd_pinvh(kit, cov)
    var dist = mcd_mahal(kit, x, loc, precision)
    return McdOneFeature(loc, cov, support^, dist)


struct McdFit(Movable):
    """The fit's results for PACK: kit matrices and the support words."""
    var raw_loc: Int
    var raw_cov: Int
    var raw_support: List[UInt8]
    var loc: Int
    var cov: Int
    var precision: Int
    var support: List[UInt8]
    var dist: Int
    var offset: Float64

    def __init__(out self):
        self.raw_loc = -1
        self.raw_cov = -1
        self.raw_support = List[UInt8]()
        self.loc = -1
        self.cov = -1
        self.precision = -1
        self.support = List[UInt8]()
        self.dist = -1
        self.offset = 0.0


def mcd_fit_run(mut kit: DKit, x: Int, h: Int, seed_word: Int, assume_centered: Bool, envelope: Bool,
                contamination: Float64) raises -> McdFit:
    """`MinCovDet.fit` after admission (and EllipticEnvelope's offset_ when
    `envelope`)."""
    var n = kit.nrows(x)
    var p = kit.ncols(x)
    var result = McdFit()
    var loc: Int
    var cov: Int
    var support: List[UInt8]
    var dist: Int
    if p == 1:
        var one = mcd_1d(kit, x, h)
        loc = one.loc
        cov = one.cov
        support = List[UInt8]()
        swap(support, one.support)  # take the words, leave `one` whole
        dist = one.dist
    else:
        # fast_mcd in ONE call; only the O(1) plan integers here, with the
        # float expressions they always used.
        var params: List[Int] = [n, p, h, seed_word, 0, 0, 0, 0, 0, 0, 0]
        if n > 500:
            var n_sub = n // 300
            var n_ss = n // n_sub
            var n_m = min(1500, n)
            var share = Float64(h) / Float64(n)
            params = [n, p, h, seed_word, n_sub, n_ss, Int(ceil(Float64(n_ss) * share)),
                      max(10, 500 // n_sub), n_m, Int(ceil(Float64(n_m) * share)), 10 if n > 1500 else 1]
        loc = kit.zeros(1, p)
        cov = kit.zeros(p, p)
        dist = kit.zeros(n, 1)
        var store = kit.istore(n)
        var addresses: List[Int] = [kit.at(x), kit.at(loc), kit.at(cov), kit.iaddr(store), kit.at(dist)]
        _ = kit.xd(XD_MCD, addresses, params, List[Float64]())
        kit.wrote_host(loc)
        kit.wrote_host(cov)
        kit.wrote_host(dist)
        support = mcd_int_support(kit, store, n)
    if assume_centered:
        loc = kit.zeros(1, p)
        var weights = kit.zeros(n, 1)
        decomp_u8_to_f32(McdU8(unsafe_from_address=Int(support.unsafe_ptr())), kit.ptr(weights), n)
        cov = mcd_masked_cov(kit, x, weights, True)[1]
        var inverse = mcd_pinvh(kit, cov)
        var product = kit.mm(x, inverse)
        var terms = kit.ew(DK_MUL, product, x)
        dist = kit.rowsum(terms)
    result.raw_loc = loc
    result.raw_cov = cov
    # correct_covariance: consistency at the normal model (dist_ is rescaled)
    var n_support = decomp_count_u8(McdU8(unsafe_from_address=Int(support.unsafe_ptr())), n)
    result.raw_support = support^
    var corr = mcd_scalar(kit, XD_CONSISTENCY_FACTOR, [Float64(p), Float64(n_support), Float64(n)])
    dist = kit.ew(DK_SCALE, dist, scalar=1.0 / corr)
    # reweight_covariance: the mask dist < thr (`le` against the float32 just below thr)
    var thr = mcd_scalar(kit, XD_CHI2_QUANTILE, [Float64(p), 0.025])
    var below = kit.const(mcd_f32_below(thr))
    var mask = kit.ew(DK_LE, dist, below)
    var moments = mcd_masked_cov(kit, x, mask, assume_centered)
    var factor = mcd_scalar(kit, XD_CONSISTENCY_FACTOR, [Float64(p), 0.975, 1.0])
    var covr = kit.ew(DK_SCALE, moments[1], scalar=factor)
    result.support = mcd_mask_support(kit, mask, n)
    var precision = mcd_pinvh(kit, covr)
    result.loc = moments[0]
    result.cov = covr
    result.precision = precision
    result.dist = mcd_mahal(kit, x, moments[0], precision)
    if envelope:
        result.offset = mcd_offset_run(kit, result.dist, contamination)
    return result^


def mcd_offset_run(mut kit: DKit, dist: Int, contamination: Float64) raises -> Float64:
    # EllipticEnvelope's offset_: the contamination percentile of -dist_.
    var nv = kit.size(dist)
    var q = 100.0 * contamination / 100.0 * Float64(nv - 1)
    var lo = Int(floor(q))
    var hi = min(lo + 1, nv - 1)
    var order = kit.order(dist)
    var vlo = -kit.word(dist, Int(kit.word(order, nv - 1 - lo)))
    var vhi = -kit.word(dist, Int(kit.word(order, nv - 1 - hi)))
    return Float64(Float32(vlo + (vhi - vlo) * (q - Float64(lo))))


def mcd_labels_run(mut kit: DKit, d: Int, offset: Float64) raises -> Int:
    # EllipticEnvelope.predict: 1 where -dist - offset_ >= 0, else -1, as a
    # float32 matrix (PACK casts it to int32); -1 for no rows.
    var rows = kit.nrows(d)
    var negated = kit.ew(DK_SCALE, d, scalar=-1.0)
    var decision = kit.ew(DK_ADDS, negated, scalar=-offset)
    if rows == 0:
        return -1
    var flipped = kit.ew(DK_SCALE, decision, scalar=-1.0)
    var zero = kit.const(0.0)
    var ge = kit.ew(DK_LE, flipped, zero)
    var plus = kit.const(1.0)
    var minus = kit.const(-1.0)
    var chosen = kit.ew(DK_SELECT, ge, plus, minus, 0.5)
    kit.host(chosen)
    return chosen


def mcd_score_run(mut kit: DKit, x: Int, loc: Int, precision: Int) raises -> Float64:
    # EmpiricalCovariance.score: X's covariance around location_ under the precision.
    var p = kit.ncols(x)
    var shifted = kit.ew(DK_SUB, x, loc)
    var cov = mcd_emp_cov(kit, shifted, True)
    var product = kit.ew(DK_MUL, cov, precision)
    var trace = kit.total_word(product)
    var logdet = mcd_slogdet(kit, precision)
    return -(Float64(p) * MCD_LOG_2PI) / 2.0 - 0.5 * trace + 0.5 * logdet[1]
