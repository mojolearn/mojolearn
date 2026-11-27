# SPDX-License-Identifier: Apache-2.0
"""LightGBM v4.6.0 rank_objective.hpp, in Mojolearn's portable FP32 arithmetic.

The tree builder consumes negative gradients and positive diagonal Hessians.
Both executors run the same row bodies and adjacent-pair reduction tree.
LambdaRank normalizes each query by its ideal top-30 DCG and total pair mass.
A stripe evaluates every eligible pair once, then reduces its contributions
without atomics or a quadratic pair list.

For XE_NDCG let p be softmax and t the randomized normalized gain. Start
with g=p-t and apply the off-diagonal matrix M[i,j]=p[i]/(1-p[j]) twice.
The objective gradient is g+Mg+M(Mg). Computing the dominant document's
column as a softmax of the remaining documents avoids division by a rounded
zero on saturated scores. Each query's RNG skips directly to this round.
For CatBoost's QuerySoftMax, Mojolearn accumulates weighted terms in portable
binary64 because their sums can overflow FP32 even when the final derivative
and mean loss fit. This arithmetic also runs on Apple GPUs.
Upstream LightGBM equations and RNG: src/objective/rank_objective.hpp and
include/LightGBM/utils/random.h at github.com/lightgbm-org/LightGBM.
"""
from gbdt.targets.ranking_options import DEFAULT_NDCG_CUTOFF
from std.memory import bitcast
from core.float_sort_key import sort_key_bits
from checks.soft_f64 import (
    SF64_ZERO, SF64_INF, sf64_add, sf64_sub, sf64_mul, sf64_div, sf64_exp, sf64_log,
    sf64_from_f32, sf64_to_f32,
)
from checks.numerics import ftz, identical_div, identical_log1p, identical_mul
from gbdt.targets.pointwise_bodies import routed_exp, routed_log

comptime RF = MutPointer[Float32, MutAnyOrigin]
comptime RU = MutPointer[UInt32, MutAnyOrigin]
comptime LR_CUTOFF = DEFAULT_NDCG_CUTOFF
# A small stripe bounds pair workspace independently of query length and cutoff.
comptime LR_STRIPE = 4
comptime LR_PLANES = 2 * LR_STRIPE + 4
comptime LR_DISCOUNT = 2 * LR_STRIPE
comptime LR_INV_IDCG = LR_DISCOUNT + 1
comptime LR_MASS = LR_DISCOUNT + 2
comptime LR_IDCG = LR_DISCOUNT + 3
comptime XE_PROB = 3
comptime XE_TAIL = 4
comptime XE_TERM_A = 5
comptime XE_TERM_B = 6
comptime XE_FOLD = 7

comptime RR_IDCG = 1
comptime RR_SAVE_IDCG = 2
comptime RR_CLEAR = 3
comptime RR_LAMBDA_PAIRS = 4
comptime RR_LAMBDA_TOP = 5
comptime RR_LAMBDA_FINAL = 6
comptime RR_TOP_INIT = 7
comptime RR_TOP_MERGE = 8
comptime RR_XE_MASSES = 9
comptime RR_XE_FIRST = 10
comptime RR_XE_SECOND = 11
comptime RR_XE_THIRD = 12
comptime RR_DCG = 13
comptime RR_QUERY_NDCG = 14
comptime RR_METRIC_FOLD = 15
comptime RR_METRIC_OUT = 16
comptime RR_SUM = 17
comptime QS_SCORE = 30
comptime QS_MAX = 31
comptime QS_MASSES = 32
comptime QS_DERIVATIVES = 33
comptime QS_LOSS = 34
comptime QS_TOTAL = 35
comptime QS_OUT = 36
comptime QS_FOLD = 37
# Even planes keep all binary64 accumulators aligned inside the existing workspace.
comptime QS_DEN = 2
comptime QS_REL = 4
comptime QS_ROW_EXP = 6
comptime QS_LOSS_SUM = 8
comptime QS_WEIGHT_SUM = 10

@fieldwise_init
struct RankView(ImplicitlyCopyable, Movable):
    var labels: RF
    var scores: RF
    var weights: RF
    var offsets: RU
    var sizes: RU
    var qids: RU
    var order: RU
    var second: RU
    var scratch: RF
    var direction: RF
    var hessian: RF
    var n: Int
    var groups: Int
    var seed: Int
    var iteration: Int
    var weighted: Bool

    @always_inline
    def plane(self, col: Int) -> RF:
        return self.scratch + col * self.n

    @always_inline
    def wide_plane(self, col: Int) -> MutPointer[UInt64, MutAnyOrigin]:
        return rebind[MutPointer[UInt64, MutAnyOrigin]](self.plane(col))


@always_inline
def rank_sort_key(v: Float32) -> UInt32:
    # Stable descending numeric order. Signed zeros have the same rank key.
    return sort_key_bits(v if v != 0 else Float32(0), True)


@always_inline
def _gain(label: Float32) -> Float32:
    return Float32((UInt32(1) << UInt32(label)) - UInt32(1))


@always_inline
def xe_draw(seed: UInt32, draws: UInt64) -> Float32:
    # Jump the MSVC LCG modulo 2^32 to the given 1-based draw. Independent
    # queries start at objective_seed + query id; each round consumes size draws.
    var m = UInt32(214013)
    var c = UInt32(2531011)
    var state = seed
    var left = draws
    while left != 0:
        if (left & UInt64(1)) != 0:
            state = m * state + c
        c = (m + UInt32(1)) * c
        m *= m
        left >>= 1
    return Float32((state >> 16) & UInt32(32767)) * Float32(0.000030517578125)


@always_inline
def _lambda_p(delta: Float32) -> Float32:
    # LightGBM uses 2^20 sigmoid bins on [-25, 25) with sigmoid=1.
    var k = Int(min(Float32(1048575), max(Float32(0), identical_mul(delta + Float32(25), Float32(20971.52)))))
    var x = identical_mul(Float32(k), Float32(50.0 / 1048576.0)) - Float32(25)
    return identical_div(Float32(1), Float32(1) + routed_exp(x))


@always_inline
def _better(a: RankView, i: UInt32, j: UInt32) -> Bool:
    if i == UInt32.MAX:
        return False
    if j == UInt32.MAX:
        return True
    var left = a.scores[Int(i)]
    var right = a.scores[Int(j)]
    return left > right or (left == right and i < j)


@always_inline
def _top_two(a: RankView, row: Int, other: Int):
    var l = a.order[row]
    var r = a.order[other]
    if _better(a, l, r):
        var runner = a.second[row]
        a.second[row] = runner if _better(a, runner, r) else r
    else:
        a.order[row] = r
        var runner = a.second[other]
        a.second[row] = runner if _better(a, runner, l) else l


@always_inline
def _lambda_stripe(a: RankView, row: Int, start: Int, size: Int, pos: Int, stripe: Int):
    var doc = Int(a.order[row])
    var y = a.labels[doc]
    var score = a.scores[doc]
    var direction = a.direction[doc]
    var hessian = a.hessian[doc]
    var mass = a.plane(LR_MASS)[row]
    var inv = a.plane(LR_INV_IDCG)[Int(a.qids[row])]
    var discount = a.plane(LR_DISCOUNT)
    var missing = bitcast[DType.float32](UInt32(0xff800000))
    var last = start + size - 1
    if size > 1 and a.scores[Int(a.order[last])] == missing:
        last -= 1
    var rescale = a.scores[Int(a.order[start])] != a.scores[Int(a.order[last])]
    for k in range(LR_STRIPE):
        var rank = stripe + k
        var top_direction = Float32(0)
        var curvature = Float32(0)
        if rank < min(pos, LR_CUTOFF):
            var top = Int(a.order[start + rank])
            var yt = a.labels[top]
            if yt != y and score != missing and a.scores[top] != missing:
                var diff = a.scores[top] - score
                var gap = identical_mul(abs(_gain(yt) - _gain(y)), abs(discount[start + rank] - discount[row]))
                gap = identical_mul(gap, inv)
                if rescale:
                    gap = identical_div(gap, Float32(0.01) + abs(diff))
                var p = _lambda_p(diff if yt > y else -diff)
                var magnitude = identical_mul(p, gap)
                top_direction = magnitude if yt > y else -magnitude
                curvature = identical_mul(identical_mul(p, Float32(1) - p), gap)
                direction = ftz(direction - top_direction)
                hessian = ftz(hessian + curvature)
                mass = ftz(mass + Float32(2) * magnitude)
        a.plane(2 * k)[row] = top_direction
        a.plane(2 * k + 1)[row] = curvature
    a.direction[doc] = direction
    a.hessian[doc] = hessian
    a.plane(LR_MASS)[row] = mass


@always_inline
def _xe_correction[third: Bool](a: RankView, row: Int, first: Int, winner: Int):
    var p = a.plane(XE_PROB)[row]
    var prev = a.plane(XE_TERM_B if third else XE_TERM_A)
    var z = Float32(0)
    if row != winner:
        z = identical_div(prev[row], Float32(1) - p)
    var term = identical_mul(p, a.plane(1 if third else XE_FOLD)[first] - z)
    if row != winner:
        # p_i/(1-p_winner) is a softmax over the remaining documents.
        # This stays finite even when the winner's rounded probability is 1.
        term = ftz(term + identical_mul(a.plane(XE_TAIL)[row], prev[winner]))
    var grad = ftz(a.direction[row] + term)
    comptime if third:
        var w = a.weights[row] if a.weighted else Float32(1)
        a.direction[row] = -identical_mul(grad, w)
        a.hessian[row] = identical_mul(a.hessian[row], w)
    else:
        a.direction[row] = grad
        a.plane(XE_TERM_B)[row] = term
        a.plane(1)[row] = identical_div(term, Float32(1) - p) if row != winner else Float32(0)


@always_inline
def rank_row[phase: Int](a: RankView, row: Int, arg: Int, width: Int, col: Int):
    comptime if phase == QS_TOTAL:
        if row % (2 * arg) == 0 and row + arg < a.n:
            for i in range(2):
                var plane = a.wide_plane(QS_LOSS_SUM + 2 * i)
                plane[row] = sf64_add(plane[row], plane[row + arg])
        return
    elif phase == QS_OUT:
        var mass = a.wide_plane(QS_WEIGHT_SUM)[0]
        a.direction[0] = sf64_to_f32(sf64_div(a.wide_plane(QS_LOSS_SUM)[0], mass)) if mass != SF64_ZERO else Float32(0)
        return
    elif phase == RR_METRIC_FOLD:
        if row % (2 * arg) == 0 and row + arg < a.groups:
            a.plane(1)[row] = ftz(a.plane(1)[row] + a.plane(1)[row + arg])
        return
    elif phase == RR_METRIC_OUT:
        a.direction[0] = identical_div(a.plane(1)[0], Float32(a.groups))
        return
    elif phase == RR_QUERY_NDCG or phase == RR_SAVE_IDCG:
        var first = Int(a.offsets[row])
        var value = a.plane(0)[first]
        var ideal = a.plane(LR_IDCG)
        comptime if phase == RR_SAVE_IDCG:
            ideal[row] = value
            a.plane(LR_INV_IDCG)[row] = identical_div(Float32(1), value) if value > 0 else Float32(0)
        else:
            # A perfect ordering has exactly the same two sums. Dividing
            # once here makes its NDCG exactly one, including in FP32.
            a.plane(1)[row] = identical_div(value, ideal[row]) if ideal[row] > 0 else Float32(1)
        return
    var q = Int(a.qids[row])
    var first = Int(a.offsets[q])
    var size = Int(a.sizes[q])
    var pos = row - first
    comptime if phase == QS_SCORE:
        a.plane(0)[row] = a.scores[row] if not a.weighted or a.weights[row] > 0 else -Float32.MAX_FINITE
    elif phase == QS_MAX:
        if pos % (2 * arg) == 0 and pos + arg < size:
            a.plane(0)[row] = max(a.plane(0)[row], a.plane(0)[row + arg])
    elif phase == QS_MASSES:
        var w = a.weights[row] if a.weighted else Float32(1)
        var e = SF64_ZERO
        if w > 0:
            var shift = ftz(a.scores[row] - a.plane(0)[first])
            # A large weight can make an exp below the FP32 range relevant again.
            var ex = sf64_from_f32(routed_exp(shift)) if shift >= -80 else sf64_exp(sf64_from_f32(shift))
            e = sf64_mul(sf64_from_f32(w), ex)
        a.wide_plane(QS_DEN)[row] = e
        a.wide_plane(QS_ROW_EXP)[row] = e
        a.wide_plane(QS_REL)[row] = sf64_mul(sf64_from_f32(w), sf64_from_f32(a.labels[row]))
    elif phase == QS_FOLD:
        if pos % (2 * arg) == 0 and pos + arg < size:
            for i in range(2):
                var plane = a.wide_plane(QS_DEN + 2 * i)
                plane[row] = sf64_add(plane[row], plane[row + arg])
    elif phase == QS_DERIVATIVES or phase == QS_LOSS:
        var w = a.weights[row] if a.weighted else Float32(1)
        var target = sf64_mul(sf64_from_f32(w), sf64_from_f32(a.labels[row]))
        var mass = a.wide_plane(QS_REL)[first]
        var denom = a.wide_plane(QS_DEN)[first]
        comptime if phase == QS_DERIVATIVES:
            var e = a.wide_plane(QS_ROW_EXP)[row]
            var p = sf64_div(e, denom) if w > 0 and mass != SF64_ZERO else SF64_ZERO
            # Subtract before dividing so equal weighted masses cancel exactly even for unequal weights.
            var numerator = sf64_sub(sf64_mul(target, denom), sf64_mul(mass, e))
            a.direction[row] = ftz(sf64_to_f32(sf64_div(numerator, denom))) if w > 0 else Float32(0)
            # CatBoost's beta=1, lambda=0.01. The final direction and curvature are FP32.
            var prob = sf64_to_f32(p)
            var curvature = identical_mul(prob, 1 - prob) + Float32(0.01)
            a.hessian[row] = ftz(sf64_to_f32(sf64_mul(mass, sf64_from_f32(curvature)))) if w > 0 else Float32(0)
        else:
            var loss = SF64_ZERO
            if target != SF64_ZERO:
                var e = a.wide_plane(QS_ROW_EXP)[row]
                var ratio = sf64_div(denom, e) if e != SF64_ZERO else SF64_INF
                var recover_from_score = ratio == SF64_INF
                if recover_from_score:
                    ratio = sf64_div(denom, sf64_from_f32(w))
                var ratio32 = sf64_to_f32(ratio)
                var nll = sf64_from_f32(routed_log(ratio32)) if (
                    ratio32 >= Float32(1.17549435e-38) and ratio32 <= Float32.MAX_FINITE
                ) else sf64_log(ratio)
                if not recover_from_score and ratio32 == Float32(1) and denom != e:
                    # A nearly certain document can still contribute a representable nonzero loss.
                    var tail = sf64_to_f32(sf64_div(sf64_sub(denom, e), e))
                    nll = sf64_from_f32(identical_log1p(tail))
                if recover_from_score:
                    # The probability or its reciprocal exceeded binary64's range.
                    var gap = sf64_sub(sf64_from_f32(a.plane(0)[first]), sf64_from_f32(a.scores[row]))
                    nll = sf64_add(gap, nll)
                loss = sf64_mul(target, nll)
            a.wide_plane(QS_LOSS_SUM)[row] = loss
            a.wide_plane(QS_WEIGHT_SUM)[row] = target
    elif phase == RR_SUM or phase == RR_TOP_MERGE:
        if pos % (2 * arg) != 0 or pos + arg >= size:
            return
        comptime if phase == RR_TOP_MERGE:
            _top_two(a, row, row + arg)
        else:
            for i in range(width):
                var plane = a.plane(col + i)
                plane[row] = ftz(plane[row] + plane[row + arg])
    elif phase == RR_IDCG or phase == RR_DCG:
        var discounts = a.plane(LR_DISCOUNT)
        comptime if phase == RR_IDCG:
            discounts[row] = identical_div(Float32(0.6931471805599453), routed_log(Float32(pos + 2)))
        a.plane(0)[row] = identical_mul(_gain(a.labels[Int(a.order[row])]), discounts[row]) if pos < arg else Float32(0)
    elif phase == RR_CLEAR:
        a.direction[row] = 0
        a.hessian[row] = 0
        a.plane(LR_MASS)[row] = 0
    elif phase == RR_LAMBDA_PAIRS:
        _lambda_stripe(a, row, first, size, pos, arg)
    elif phase == RR_LAMBDA_TOP:
        var k = pos - arg
        if k >= 0 and k < LR_STRIPE and pos < LR_CUTOFF:
            var doc = Int(a.order[row])
            a.direction[doc] = ftz(a.direction[doc] + a.plane(2 * k)[first])
            a.hessian[doc] = ftz(a.hessian[doc] + a.plane(2 * k + 1)[first])
    elif phase == RR_LAMBDA_FINAL:
        var mass = a.plane(LR_MASS)[first]
        var scale = Float32(1)
        if mass > 0:
            scale = identical_div(identical_log1p(mass), identical_mul(Float32(0.6931471805599453), mass))
        if a.weighted:
            scale = identical_mul(scale, a.weights[row])
        a.direction[row] = identical_mul(a.direction[row], scale)
        a.hessian[row] = identical_mul(a.hessian[row], scale)
    elif phase == RR_TOP_INIT:
        a.order[row] = UInt32(row)
        a.second[row] = UInt32.MAX
    elif phase == RR_XE_MASSES:
        var best = Int(a.order[first])
        var exp_score = routed_exp(a.scores[row] - a.scores[best])
        var tail = Float32(0)
        if row != best:
            tail = routed_exp(a.scores[row] - a.scores[Int(a.second[first])])
        var draw = UInt64(a.iteration) * UInt64(size) + UInt64(pos) + UInt64(1)
        var phi = Float32(UInt32(1) << UInt32(a.labels[row])) - xe_draw(UInt32(a.seed) + UInt32(q), draw)
        a.plane(0)[row] = exp_score
        a.plane(1)[row] = tail
        a.plane(2)[row] = phi
        a.plane(XE_PROB)[row] = exp_score
        a.plane(XE_TAIL)[row] = tail
        a.plane(XE_TERM_A)[row] = phi
    elif phase == RR_XE_FIRST:
        var winner = Int(a.order[first])
        var p = identical_div(a.plane(XE_PROB)[row], a.plane(0)[first])
        var tail = Float32(0)
        if row != winner:
            tail = identical_div(a.plane(XE_TAIL)[row], a.plane(1)[first])
        a.plane(XE_PROB)[row] = p
        a.plane(XE_TAIL)[row] = tail
        var grad = p - identical_div(a.plane(XE_TERM_A)[row], a.plane(2)[first])
        var complement = Float32(1) - p
        if row == winner and size > 1:
            var second = Int(a.second[first])
            var r = routed_exp(a.scores[second] - a.scores[winner])
            complement = identical_div(identical_mul(r, a.plane(1)[first]), a.plane(0)[first])
        a.direction[row] = grad
        a.hessian[row] = identical_mul(p, complement)
        a.plane(XE_TERM_A)[row] = grad
        a.plane(XE_FOLD)[row] = identical_div(grad, Float32(1) - p) if row != winner else Float32(0)
    elif phase == RR_XE_SECOND or phase == RR_XE_THIRD:
        var winner = Int(a.order[first])
        if size == 1:
            a.direction[row] = Float32(0)
            a.hessian[row] = Float32(0)
            a.plane(1)[row] = Float32(0)
            return
        _xe_correction[phase == RR_XE_THIRD](a, row, first, winner)


trait RankStages:
    def rows[phase: Int](mut self, a: RankView, count: Int, arg: Int, width: Int, col: Int) raises: ...
    def order_by[labels: Bool](mut self, a: RankView) raises: ...


def rank_fold[R: RankStages, phase: Int](mut r: R, a: RankView, largest: Int, width: Int, col: Int) raises:
    var step = 1
    while step < largest:  # driver-loop: over levels of the segmented reduction
        r.rows[phase](a, a.n, step, width, col)
        step *= 2


def rank_ideal[R: RankStages](mut r: R, a: RankView, largest: Int, cutoff: Int = LR_CUTOFF) raises:
    r.order_by[True](a)
    r.rows[RR_IDCG](a, a.n, cutoff, 0, 0)
    rank_fold[R, RR_SUM](r, a, largest, 1, 0)
    r.rows[RR_SAVE_IDCG](a, a.groups, 0, 0, 0)


def lambda_derivatives[R: RankStages](mut r: R, a: RankView, largest: Int) raises:
    r.order_by[False](a)
    r.rows[RR_CLEAR](a, a.n, 0, 0, 0)
    for k in range(0, min(LR_CUTOFF, largest - 1), LR_STRIPE):  # driver-loop: over stripes of ranks above the cutoff
        r.rows[RR_LAMBDA_PAIRS](a, a.n, k, 0, 0)
        rank_fold[R, RR_SUM](r, a, largest, 2 * LR_STRIPE, 0)
        r.rows[RR_LAMBDA_TOP](a, a.n, k, 0, 0)
    rank_fold[R, RR_SUM](r, a, largest, 1, LR_MASS)
    r.rows[RR_LAMBDA_FINAL](a, a.n, 0, 0, 0)


def xe_derivatives[R: RankStages](mut r: R, a: RankView, largest: Int) raises:
    r.rows[RR_TOP_INIT](a, a.n, 0, 0, 0)
    rank_fold[R, RR_TOP_MERGE](r, a, largest, 0, 0)
    r.rows[RR_XE_MASSES](a, a.n, 0, 0, 0)
    rank_fold[R, RR_SUM](r, a, largest, 3, 0)
    r.rows[RR_XE_FIRST](a, a.n, 0, 0, 0)
    rank_fold[R, RR_SUM](r, a, largest, 1, XE_FOLD)
    r.rows[RR_XE_SECOND](a, a.n, 0, 0, XE_TERM_B)
    rank_fold[R, RR_SUM](r, a, largest, 1, 1)
    r.rows[RR_XE_THIRD](a, a.n, 0, 0, 1)


def rank_ndcg[R: RankStages](mut r: R, a: RankView, largest: Int, cutoff: Int = LR_CUTOFF) raises:
    r.order_by[False](a)
    r.rows[RR_DCG](a, a.n, cutoff, 0, 0)
    rank_fold[R, RR_SUM](r, a, largest, 1, 0)
    r.rows[RR_QUERY_NDCG](a, a.groups, 0, 0, 0)
    var step = 1
    while step < a.groups:  # driver-loop: over levels of the metric reduction
        r.rows[RR_METRIC_FOLD](a, a.groups, step, 0, 0)
        step *= 2
    r.rows[RR_METRIC_OUT](a, 1, 0, 0, 0)


def query_softmax[R: RankStages, metric: Bool, reuse_terms: Bool = False](mut r: R, a: RankView, largest: Int) raises:
    # Only the metric immediately following derivatives on the same scores may reuse these planes.
    comptime if not reuse_terms:
        r.rows[QS_SCORE](a, a.n, 0, 0, 0)
        rank_fold[R, QS_MAX](r, a, largest, 0, 0)
        r.rows[QS_MASSES](a, a.n, 0, 0, 0)
        rank_fold[R, QS_FOLD](r, a, largest, 0, 0)
    comptime if metric:
        r.rows[QS_LOSS](a, a.n, 0, 0, 0)
        var step = 1
        while step < a.n:  # driver-loop: over levels of the global loss reduction
            r.rows[QS_TOTAL](a, a.n, step, 0, 0)
            step *= 2
        r.rows[QS_OUT](a, 1, 0, 0, 0)
    else:
        r.rows[QS_DERIVATIVES](a, a.n, 0, 0, 0)
