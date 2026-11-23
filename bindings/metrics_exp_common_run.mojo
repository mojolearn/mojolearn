# SPDX-License-Identifier: Apache-2.0
# RUN pieces shared by the classification, ranking and clustering metric
# doors: typed only, no Python import (tools/python_free_dirs.sh). Lane
# metric-programs-2 (2026-11-12): split from metrics_exp_common.mojo.
from std.memory import bitcast
from std.math import sqrt
from core.metrics_program_plan import MetricProgramPlan
from bindings.arena_prog_run import MetricRoute, metric_inputs_run

comptime MX_NONE = -1


def mx_nan() -> Float64:
    return bitcast[DType.float64](UInt64(0x7FF8000000000000))


def mx_inf() -> Float64:
    return bitcast[DType.float64](UInt64(0x7FF0000000000000))


def mx_isnan(x: Float64) -> Bool:
    return x != x


def mx_isinf(x: Float64) -> Bool:
    return x == x and x - x != 0.0


def mx_class_code(kind: Int, item: Int) -> String:
    """The `array.array` code holding one class item of a label buffer."""
    if kind == 1:  # signed
        if item == 1:
            return "b"
        if item == 2:
            return "h"
        if item == 4:
            return "i"
        return "q"
    if item == 1:
        return "B"
    if item == 2:
        return "H"
    if item == 4:
        return "I"
    return "Q"


# Moved verbatim from bindings/metrics_exp_cluster_run.mojo (lane bindings-inline, 2026-11-22).
# RUN part of bindings/metrics_exp_cluster.mojo: typed Mojo only (no Python
# imports). The binary64 epilogue of NMI and AMI.

def _generalized_average(U: Float64, V: Float64, method: Int) raises -> Float64:
    if method == 0:
        return V if V < U else U
    if method == 1:
        var x = U * V
        if x < 0.0:
            raise Error("math domain error")
        return sqrt(x)
    if method == 2:
        return (U + V) / 2
    return V if V > U else U


def nmi_finish(mi: Float64, entropy_true: Float64, entropy_pred: Float64, method: Int) raises -> Float64:
    """RUN: NMI from the program's MI and the two entropies (lane
    entangled-programs: moved out of the door, same order of operations)."""
    if mi == 0:
        return 0.0
    return mi / _generalized_average(entropy_true, entropy_pred, method)


def ami_finish(mi: Float64, emi: Float64, entropy_true: Float64, entropy_pred: Float64, method: Int) raises -> Float64:
    """RUN: AMI's binary64 epilogue (the eps guards on numerator and
    denominator), moved out of the door unchanged."""
    var norm = _generalized_average(entropy_true, entropy_pred, method)
    var eps = 2.220446049250313e-16
    var den = norm - emi
    if den < 0:
        den = -eps if -eps < den else den
    else:
        den = eps if eps > den else den
    var num = mi - emi
    if num < 0:
        num = -eps if -eps < num else num
    else:
        num = eps if eps > num else num
    return num / den


def contingency_dtype_code(name: String) -> Int:
    """RUN: the requested dtype's cast code for `mexp_cast_i64` (0 int32,
    1 float64, 2 float32), -1 to keep the int64 counts."""
    if name == "int32":
        return 0
    if name == "float64":
        return 1
    if name == "float32":
        return 2
    return -1


# ---------------------------------------------------------------- the contingency program, typed
# Lane mf-score5 (2026-11-23): `_contingency` (bindings/metrics_exp_cluster_door.mojo)
# assembled its program over pins; it is one builder here (the door pins its
# inputs in the builder's order; the search scorers' RUN calls it over typed
# codes), and `cluster_mi_run` is normalized / adjusted mutual information
# over typed codes. The same stages in the same order: the bits do not move.


@fieldwise_init
struct ContingencySlots(ImplicitlyCopyable, Movable):
    """`_contingency`'s program offsets: the counts (and the eps matrix),
    the row and column sums, the pair confusion, the entropies, MI, EMI."""
    var C: Int
    var F: Int
    var R: Int
    var K: Int
    var P: Int
    var E: Int
    var MI: Int
    var EMI: Int


def contingency_stages(mut plan: MetricProgramPlan, n: Int, ka: Int, kb: Int, has_eps: Bool, eps: Float64,
                       mi: Bool, emi: Bool) raises -> ContingencySlots:
    """`_contingency`'s program over its inputs A and B (n int32 codes of
    each clustering into its own sorted classes, ka and kb of them), in that
    order."""
    if ka * kb > 16777216:
        raise Error("mojolearn metrics: the contingency matrix exceeds 2^24 cells")
    var A = plan.input(n)
    var B = plan.input(n)
    var key = plan.scratch(n)
    var kk = max(ka, kb)
    plan.stage_op("pair_key", n, [A, B, key, kk, 0])
    var m = kk * kk
    var off = plan.scratch(m + 1)
    var order = plan.scratch(n)
    plan.stage_op("group_sort", 1, [key, n, m, off, order])
    var cells = ka * kb
    var EPS = MX_NONE
    if has_eps:
        EPS = plan.input_f64(eps)
    var C = plan.want(plan.alloc(2 * cells), 2 * cells, -1, 1)
    var F = MX_NONE
    if has_eps:
        F = plan.want(plan.alloc(2 * cells), 2 * cells, -1, 1)
    var R = plan.want(plan.alloc(2 * ka), 2 * ka, -1, 1)
    var K = plan.want(plan.alloc(2 * kb), 2 * kb, -1, 1)
    var P = plan.want(plan.alloc(8), 8, -1, 1)
    var E = plan.want(plan.alloc(4), 4, -1, 1)
    var MI = MX_NONE
    if mi:
        MI = plan.want(plan.alloc(2), 2, -1, 1)
    var EMI = MX_NONE
    if emi:
        EMI = plan.want(plan.alloc(2), 2, -1, 1)
    plan.stage_op("cont_stats", 1, [off, ka, kb, kk, EPS, C, F, R, K, P, E, MI, EMI, n])
    return ContingencySlots(C, F, R, K, P, E, MI, EMI)


def _own_codes(src: Int, n: Int, k: Int, mut codes: List[Int32]) raises -> Int:
    """The codes of one clustering into its OWN sorted classes, from class
    codes in 0 .. k-1 (a code's rank among the codes present: the order of
    the labels it stands for); the count of its classes."""
    var s = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=src)
    var rank = List[Int32](length=max(k, 1), fill=Int32(-1))
    for i in range(n):  # driver-loop: over the rows (the classes present)
        var c = Int(s[i])
        if c < 0 or c >= k:
            raise Error("ValueError: y codes outside [0, n_classes)")
        rank[c] = 0
    var count = 0
    for c in range(k):  # small-loop(the class codes: their ranks)
        if rank[c] == 0:
            rank[c] = Int32(count)
            count += 1
    codes = List[Int32](length=max(n, 1), fill=Int32(0))
    for i in range(n):  # driver-loop: over the rows (the own codes)
        codes[i] = rank[Int(s[i])]
    return count


def cluster_mi_run(route: MetricRoute, adjusted: Bool, truth: Int, pred: Int, n: Int, k: Int,
                   method: Int) raises -> Float64:
    """THE TYPED RUN of `normalized_mutual_info_score` (adjusted False) or
    `adjusted_mutual_info_score` (adjusted True) over two clusterings given
    as n int32 class codes in 0 .. k-1 (each mapped to its own sorted
    classes, as the door's encoder does; codes keep their labels' order),
    `method` the average method word (2: 'arithmetic'). The door's order
    (bindings/metrics_exp_cluster_door.mojo `_nmi`, `_ami`). No Python."""
    var a = List[Int32]()
    var b = List[Int32]()
    var ka = _own_codes(truth, n, k, a)
    var kb = _own_codes(pred, n, k, b)
    if (ka == kb and kb == 1) or (ka == kb and kb == 0):
        return 1.0
    if adjusted and (ka == 1 or kb == 1):
        return 0.0
    var plan = MetricProgramPlan()
    var r = contingency_stages(plan, n, ka, kb, False, 0.0, True, adjusted)
    var addresses: List[Int] = [Int(a.unsafe_ptr()), Int(b.unsafe_ptr())]
    var sizes: List[Int] = [n, n]
    var arena = metric_inputs_run(route, plan, addresses, sizes)
    _ = len(a)  # copied into the program's arena above
    _ = len(b)
    if adjusted:
        return ami_finish(arena.f64(r.MI), arena.f64(r.EMI), arena.f64(r.E), arena.f64(r.E + 2), method)
    return nmi_finish(arena.f64(r.MI), arena.f64(r.E), arena.f64(r.E + 2), method)
