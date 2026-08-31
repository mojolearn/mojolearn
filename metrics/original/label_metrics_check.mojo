# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Group A gates: accuracy, rand index, adjusted rand index, entropy, mutual
information, homogeneity / completeness / v-measure.

    pixi run mojo run -I . metrics/original/label_metrics_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . metrics/original/label_metrics_check.mojo

THE SHAPE OF EVERY GATE HERE. The device's product is an INTEGER (a count,
a contingency matrix, a pair of pair-counts) and is compared EXACTLY, per
cell, against a host tally built by a different loop. The float epilogue
(DEVIATIONS 650, 651) is compared BITWISE under IDENTICAL against an oracle
written in this file -- serial, ascending, Float32 through the same
`identical_log` / `identical_mul_add` / `ftz` -- and in both modes against
a Float64 reference that spells the sklearn formula (not RAFT's grouping)
to 1e-6 relative. Under FAST the bitwise line is a REPORT.

The checks, in order:

    check_contingency_matrix_exact      both arms (SMEM at k=7 and k=31,
                                        GLOBAL at k=40), offset labels
                                        (minLabel = 3), every cell
    check_accuracy_exact                count EXACT, value bitwise
    check_rand_index_exact              (a, b) EXACT against an O(n^2)
                                        host loop AND against sklearn's
                                        pair-confusion formula from the
                                        host contingency matrix; value
                                        bitwise
    check_adjusted_rand_index           bitwise against the host formula
                                        from the host matrix; the four
                                        early-return branches
    check_entropy_epilogue              IDENTICAL bitwise vs oracle; both
                                        modes vs Float64 reference; the
                                        constant-labels (H = 0) branch
    check_mutual_info_epilogue          the same, plus MI == 0 EXACTLY on a
                                        product-structured fixture
    check_homogeneity_completeness_v    the defined-as-one and defined-as-
                                        zero branches exactly as sklearn
                                        defines them: constant truth (h =
                                        1), constant pred (c = 1, h = 0,
                                        v = 0), perfect (1, 1, 1),
                                        independent (0, 0, v = 0 by the
                                        h + c == 0 branch), singleton
    check_label_epilogue_order_is_visible
                                        the fixture SEPARATES a serial
                                        ascending fold from a descending
                                        one and from the Float64 one, so
                                        the bitwise gates above have teeth

SABOTAGES PERFORMED (2026-08-23), each reverted; outputs in the README:
    (a) contingency index written `pd * width + gt` (transposed):
        check_contingency_matrix_exact fails per cell
    (b) the MI oracle's (i, j) loop reversed: under IDENTICAL
        check_mutual_info_epilogue fails bitwise
    (c) `ftz` dropped from the entropy epilogue's stored term: no Apple
        bit moves (Apple flushes in hardware; the pin is inert here, as
        numerics.mojo says of every pin on this column) -- RECORDED as
        the expected null, not as evidence the seam is unreached
"""

from std.math import log
from max.gpu.host import DeviceBuffer, DeviceContext

from metrics.original.device_io import upload_i32
from metrics.original.fixtures import (
    bits32,
    bits64,
    constant_labels,
    labels_offset,
    labels_true_pred,
    plant_singleton,
)
from metrics.derived.metrics.accuracy_score import accuracy_score_py
from metrics.derived.metrics.adjusted_rand_index import adjusted_rand_index
from metrics.derived.metrics.completeness_score import completeness_score
from metrics.derived.metrics.entropy import entropy
from metrics.derived.metrics.homogeneity_score import homogeneity_score
from metrics.derived.metrics.mutual_info_score import mutual_info_score
from metrics.derived.metrics.rand_index import rand_index
from metrics.derived.metrics.v_measure import v_measure
from metrics.derived.stats.detail.contingency_matrix import (
    CMAT_SMEM_MAX_DIM,
    contingency_matrix,
    get_input_class_cardinality,
)
from metrics.derived.stats.detail.rand_index import rand_index_counts
from metrics.derived.stats.detail.scores import count_correct
from original.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_log,
    identical_mul_add,
    numeric_mode_name,
)


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime N = 4099


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29; see the note
    on that function. A local two-way IDENTICAL-or-FAST answers "FAST"
    for a DETERMINISTIC build, which mislabels every line the driver
    prints.
    """
    return numeric_mode_name()


# ---------------------------------------------------------------- host side


def _host_contingency(
    yt: List[Int32], yp: List[Int32], lo: Int32, k: Int
) -> List[Int32]:
    var c = List[Int32]()
    for _ in range(k * k):
        c.append(Int32(0))
    for i in range(len(yt)):
        var r = Int(yt[i] - lo)
        var q = Int(yp[i] - lo)
        c[r * k + q] += 1
    return c^


def _host_min_max(labels: List[Int32]) -> Tuple[Int32, Int32]:
    var lo = labels[0]
    var hi = labels[0]
    for i in range(len(labels)):
        if labels[i] < lo:
            lo = labels[i]
        if labels[i] > hi:
            hi = labels[i]
    return (lo, hi)


def _oracle_entropy_f32(counts: List[Int64], size: Int) -> Float32:
    """Serial ascending, Float32, the same helpers. INDEPENDENT of
    entropy.mojo's function: written from RAFT's `entropyOp`, not copied."""
    var acc = Float32(0.0)
    for i in range(len(counts)):
        if counts[i] != Int64(0):
            var p = ftz(Float32(counts[i]) / Float32(size))
            acc = ftz(identical_mul_add(-p, ftz(identical_log(p)), acc))
    return acc


def _oracle_mi_f32(c: List[Int32], k: Int, size: Int, reverse: Bool) -> Float32:
    var a = List[Int64]()
    var b = List[Int64]()
    for i in range(k):
        var sa = Int64(0)
        var sb = Int64(0)
        for j in range(k):
            sa += Int64(c[i * k + j])
            sb += Int64(c[j * k + i])
        a.append(sa)
        b.append(sb)
    var acc = Float32(0.0)
    var cells = k * k
    for step in range(cells):
        var idx = cells - 1 - step if reverse else step
        var i = idx // k
        var j = idx % k
        var cij = c[idx]
        if cij != Int32(0) and a[i] * b[j] != Int64(0):
            var fc = Float32(cij)
            var l1 = ftz(identical_log(ftz(Float32(size) * fc)))
            var l2 = ftz(identical_log(ftz(Float32(a[i] * b[j]))))
            acc = ftz(identical_mul_add(fc, ftz(l1 - l2), acc))
    return ftz(acc / Float32(size))


def _ref_entropy_f64(counts: List[Int64], size: Int) -> Float64:
    """sklearn's spelling: `-sum(pi * (log(pi) - log(pi_sum)))`."""
    var acc = 0.0
    var lsum = log(Float64(size))
    for i in range(len(counts)):
        if counts[i] != Int64(0):
            var pi = Float64(counts[i])
            acc += pi * (log(pi) - lsum)
    return -acc / Float64(size)


def _ref_mi_f64(c: List[Int32], k: Int, size: Int) -> Float64:
    """sklearn's spelling: `sum c/n * (log c - log a - log b + log n)`."""
    var a = List[Int64]()
    var b = List[Int64]()
    for i in range(k):
        var sa = Int64(0)
        var sb = Int64(0)
        for j in range(k):
            sa += Int64(c[i * k + j])
            sb += Int64(c[j * k + i])
        a.append(sa)
        b.append(sb)
    var acc = 0.0
    var ln = log(Float64(size))
    for i in range(k):
        for j in range(k):
            var cij = c[i * k + j]
            if cij != Int32(0):
                var nm = Float64(cij)
                acc += (nm / Float64(size)) * (
                    log(nm) - log(Float64(a[i])) - log(Float64(b[j])) + ln
                )
    return acc


def _counts_from(c: List[Int32], k: Int, rows: Bool) -> List[Int64]:
    var out = List[Int64]()
    for i in range(k):
        var s = Int64(0)
        for j in range(k):
            s += Int64(c[i * k + j]) if rows else Int64(c[j * k + i])
        out.append(s)
    return out^


def _transpose(c: List[Int32], k: Int) -> List[Int32]:
    var t = List[Int32]()
    for j in range(k):
        for i in range(k):
            t.append(c[i * k + j])
    return t^


def _rel(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    var m = abs(a) if abs(a) > abs(b) else abs(b)
    if m == 0.0:
        return d
    return d / m


def _assert_bits64(
    tag: String, got: Float64, want: Float64, assert_identical: Bool
) raises:
    var same = bits64(got) == bits64(want)
    var line = (
        tag
        + " got "
        + String(got)
        + " ("
        + bits64(got)
        + ") oracle "
        + String(want)
        + " ("
        + bits64(want)
        + ")"
    )
    if same:
        print("    bitwise OK  " + line)
    elif assert_identical:
        raise Error("BITWISE MISMATCH " + line)
    else:
        print("    bitwise REPORT (FAST, not asserted) " + line)


# ----------------------------------------------------------------- checks


def check_contingency_matrix_exact(ctx: DeviceContext) raises:
    print("check_contingency_matrix_exact [" + _mode_name() + "]")
    var ks = List[Int]()
    ks.append(7)
    ks.append(31)
    ks.append(40)
    for t in range(len(ks)):
        var k = ks[t]
        var arm = String("SMEM") if k <= CMAT_SMEM_MAX_DIM else String(
            "GLOBAL"
        )
        var lp = labels_true_pred(N, k, k, 0.6, 11 + k)
        var yt0 = lp[0].copy()
        var yp0 = lp[1].copy()
        # Plant the full range on both sides so minLabel/maxLabel are the
        # same for truth and prediction, then OFFSET by 3.
        yt0[0] = Int32(k - 1)
        yp0[1] = Int32(k - 1)
        var yt = labels_offset(yt0, 3)
        var yp = labels_offset(yp0, 3)
        var dt = upload_i32(ctx, yt)
        var dp = upload_i32(ctx, yp)
        var mm = get_input_class_cardinality(ctx, dt, N)
        var hmm = _host_min_max(yt)
        if mm[0] != hmm[0] or mm[1] != hmm[1]:
            raise Error(
                "min/max labels: device ("
                + String(mm[0])
                + ", "
                + String(mm[1])
                + ") host ("
                + String(hmm[0])
                + ", "
                + String(hmm[1])
                + ")"
            )
        if Int(mm[1] - mm[0] + 1) != k:
            raise Error("fixture did not plant the full label range")
        var dmat = ctx.enqueue_create_buffer[DType.int32](k * k)
        contingency_matrix(ctx, dt, dp, N, dmat, mm[0], mm[1])
        var h = ctx.enqueue_create_host_buffer[DType.int32](k * k)
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=dmat)
        ctx.synchronize()
        var want = _host_contingency(yt, yp, mm[0], k)
        var bad = 0
        var total = Int64(0)
        for idx in range(k * k):
            var got = h.unsafe_ptr().unsafe_load(idx)
            total += Int64(got)
            if got != want[idx]:
                if bad < 5:
                    print(
                        "    cell "
                        + String(idx // k)
                        + ","
                        + String(idx % k)
                        + " device "
                        + String(got)
                        + " host "
                        + String(want[idx])
                    )
                bad += 1
        _ = h^
        if bad != 0:
            raise Error(
                "contingency matrix k="
                + String(k)
                + " ("
                + arm
                + " arm): "
                + String(bad)
                + " cells differ"
            )
        if total != Int64(N):
            raise Error("contingency matrix does not sum to n")
        print(
            "    k="
            + String(k)
            + " "
            + arm
            + " arm, minLabel="
            + String(mm[0])
            + ": "
            + String(k * k)
            + " cells EXACT, sum "
            + String(total)
        )
    print("  OK")


def check_accuracy_exact(ctx: DeviceContext) raises:
    print("check_accuracy_exact [" + _mode_name() + "]")
    var lp = labels_true_pred(N, 6, 6, 0.73, 5)
    var yt = lp[0].copy()
    var yp = lp[1].copy()
    var want = 0
    for i in range(N):
        if yt[i] == yp[i]:
            want += 1
    var dt = upload_i32(ctx, yt)
    var dp = upload_i32(ctx, yp)
    var got = count_correct(ctx, dt, dp, N)
    if got != want:
        raise Error(
            "accuracy count: device " + String(got) + " host " + String(want)
        )
    var acc = accuracy_score_py(ctx, dt, dp, N)
    var want_acc = Float32(want) / Float32(N)
    if bits32(acc) != bits32(want_acc):
        raise Error(
            "accuracy value: device "
            + bits32(acc)
            + " host "
            + bits32(want_acc)
        )
    # row 39: n = 0 would be 0 / 0 (a NaN in a recorded scalar); refused
    var refused_n0 = False
    try:
        _ = accuracy_score_py(ctx, dt, dp, 0)
    except e:
        print("    n=0: " + String(e))
        refused_n0 = True
    if not refused_n0:
        raise Error("accuracy_score(n=0) must be refused by name (0 / 0)")
    print(
        "    count "
        + String(got)
        + " of "
        + String(N)
        + " EXACT; value "
        + String(acc)
        + " "
        + bits32(acc)
        + " bitwise"
    )
    print("  OK")


def _pair_confusion_rand(c: List[Int32], k: Int, n: Int) -> Tuple[Int64, Int64]:
    """sklearn `pair_confusion_matrix`: tp = sum C^2 - n; the `b` count is
    n^2 - sum_i a_i^2 - sum_j b_j^2 + sum C^2 (tn). Returned as ordered
    pairs / 2, i.e. RAFT's (a, b)."""
    var sum_sq = Int64(0)
    var sum_a2 = Int64(0)
    var sum_b2 = Int64(0)
    for i in range(k):
        var a = Int64(0)
        var b = Int64(0)
        for j in range(k):
            var v = Int64(c[i * k + j])
            sum_sq += v * v
            a += v
            b += Int64(c[j * k + i])
        sum_a2 += a * a
        sum_b2 += b * b
    var nn = Int64(n)
    var tp = sum_sq - nn
    var tn = nn * nn - sum_a2 - sum_b2 + sum_sq
    return (tp // 2, tn // 2)


def check_rand_index_exact(ctx: DeviceContext) raises:
    print("check_rand_index_exact [" + _mode_name() + "]")
    var lp = labels_true_pred(N, 5, 4, 0.7, 21)
    var yt = lp[0].copy()
    var yp = lp[1].copy()
    var a = Int64(0)
    var b = Int64(0)
    for i in range(N):
        for j in range(i):
            var same_t = yt[i] == yt[j]
            var same_p = yp[i] == yp[j]
            if same_t and same_p:
                a += 1
            elif (not same_t) and (not same_p):
                b += 1
    var dt = upload_i32(ctx, yt)
    var dp = upload_i32(ctx, yp)
    var got = rand_index_counts(ctx, dt, dp, N)
    if got[0] != a or got[1] != b:
        raise Error(
            "rand counts: device ("
            + String(got[0])
            + ", "
            + String(got[1])
            + ") host ("
            + String(a)
            + ", "
            + String(b)
            + ")"
        )
    var c = _host_contingency(yt, yp, 0, 5)
    var pc = _pair_confusion_rand(c, 5, N)
    if pc[0] != a or pc[1] != b:
        raise Error(
            "rand counts: O(n^2) loop ("
            + String(a)
            + ", "
            + String(b)
            + ") disagrees with sklearn's pair-confusion ("
            + String(pc[0])
            + ", "
            + String(pc[1])
            + ")"
        )
    var ri = rand_index(ctx, dt, dp, N)
    var n = Int64(N)
    var want = Float64(a + b) / Float64(n * (n - 1) // 2)
    _assert_bits64("rand_index", ri, want, True)
    # size < 2 returns 1.0
    var one = List[Int32]()
    one.append(Int32(0))
    var d1 = upload_i32(ctx, one)
    var d1b = upload_i32(ctx, one)
    if rand_index(ctx, d1, d1b, 1) != 1.0:
        raise Error("rand_index(size=1) must be 1.0")
    print(
        "    a="
        + String(a)
        + " b="
        + String(b)
        + " EXACT (two host tallies agree); size<2 -> 1.0"
    )
    print("  OK")


def _host_ari(c: List[Int32], k: Int, n: Int) -> Float64:
    var sum_c2 = Int64(0)
    var sum_a2 = Int64(0)
    var sum_b2 = Int64(0)
    for i in range(k):
        var a = Int64(0)
        var b = Int64(0)
        for j in range(k):
            var v = Int64(c[i * k + j])
            sum_c2 += v * (v - 1) // 2
            a += v
            b += Int64(c[j * k + i])
        sum_a2 += a * (a - 1) // 2
        sum_b2 += b * (b - 1) // 2
    var n_c2 = Float64(n) * Float64(n - 1) / 2.0
    var expected = Float64(sum_a2) * Float64(sum_b2) / n_c2
    var mx = (Float64(sum_b2) + Float64(sum_a2)) / 2.0
    if mx - expected != 0.0:
        return (Float64(sum_c2) - expected) / (mx - expected)
    return 0.0


def check_adjusted_rand_index(ctx: DeviceContext) raises:
    print("check_adjusted_rand_index [" + _mode_name() + "]")
    var lp = labels_true_pred(N, 5, 4, 0.7, 21)
    var yt = lp[0].copy()
    var yp = lp[1].copy()
    var dt = upload_i32(ctx, yt)
    var dp = upload_i32(ctx, yp)
    var got = adjusted_rand_index(ctx, dt, dp, N)
    var c = _host_contingency(yt, yp, 0, 5)
    _assert_bits64("adjusted_rand_index", got, _host_ari(c, 5, N), True)
    # Branches: size < 2; both constant; perfect; pred constant.
    var one = List[Int32]()
    one.append(Int32(0))
    var d1 = upload_i32(ctx, one)
    var d1b = upload_i32(ctx, one)
    if adjusted_rand_index(ctx, d1, d1b, 1) != 1.0:
        raise Error("ARI(size=1) must be 1.0")
    var k0 = upload_i32(ctx, constant_labels(N, 2))
    var k1 = upload_i32(ctx, constant_labels(N, 5))
    if adjusted_rand_index(ctx, k0, k1, N) != 1.0:
        raise Error("ARI(both constant) must be 1.0 (nUniq == 1 branch)")
    var dt2 = upload_i32(ctx, yt)
    if adjusted_rand_index(ctx, dt, dt2, N) != 1.0:
        raise Error("ARI(perfect) must be 1.0")
    var ari_pc = adjusted_rand_index(ctx, dt, k0, N)
    if ari_pc != 0.0:
        raise Error("ARI(pred constant) must be 0.0, got " + String(ari_pc))
    # singleton cluster in pred
    var yps = yp.copy()
    plant_singleton(yps, 17, Int32(9))
    var dps = upload_i32(ctx, yps)
    var cs = _host_contingency(yt, yps, 0, 10)
    _assert_bits64(
        "adjusted_rand_index(singleton)",
        adjusted_rand_index(ctx, dt, dps, N),
        _host_ari(cs, 10, N),
        True,
    )
    print("    size<2, both-constant, perfect -> 1.0; pred constant -> 0.0")
    print("  OK")


def check_entropy_epilogue(ctx: DeviceContext) raises:
    print("check_entropy_epilogue [" + _mode_name() + "]")
    var lp = labels_true_pred(N, 9, 9, 0.5, 31)
    var yt = lp[0].copy()
    var dt = upload_i32(ctx, yt)
    var got = entropy(ctx, dt, N, 0, 8)
    var c = _host_contingency(yt, yt, 0, 9)
    var counts = _counts_from(c, 9, True)
    var oracle = Float64(_oracle_entropy_f32(counts, N))
    var reference = _ref_entropy_f64(counts, N)
    _assert_bits64("entropy", got, oracle, IDENTICAL)
    var rel = _rel(got, reference)
    print("    vs Float64 sklearn spelling: rel " + String(rel))
    if rel > 1e-6:
        raise Error("entropy off the Float64 reference by " + String(rel))
    # constant labels: H = 0 exactly (one p = 1, log 1 = 0)
    var k0 = upload_i32(ctx, constant_labels(N, 4))
    var h0 = entropy(ctx, k0, N, 4, 4)
    if h0 != 0.0:
        raise Error("entropy(constant) must be 0.0, got " + String(h0))
    if entropy(ctx, k0, 0, 4, 4) != 1.0:
        raise Error("entropy(size=0) must be 1.0 (RAFT :112, sklearn)")
    print("    constant -> 0.0; size 0 -> 1.0")
    print("  OK")


def check_mutual_info_epilogue(ctx: DeviceContext) raises:
    print("check_mutual_info_epilogue [" + _mode_name() + "]")
    var lp = labels_true_pred(N, 9, 7, 0.55, 41)
    var yt = lp[0].copy()
    var yp = lp[1].copy()
    var dt = upload_i32(ctx, yt)
    var dp = upload_i32(ctx, yp)
    var got = mutual_info_score(ctx, dt, dp, N, 0, 8)
    var c = _host_contingency(yt, yp, 0, 9)
    var oracle = Float64(_oracle_mi_f32(c, 9, N, False))
    var reference = _ref_mi_f64(c, 9, N)
    _assert_bits64("mutual_info_score", got, oracle, IDENTICAL)
    var rel = _rel(got, reference)
    print("    vs Float64 sklearn spelling: rel " + String(rel))
    if rel > 1e-6:
        raise Error("MI off the Float64 reference by " + String(rel))
    # MI == 0 EXACTLY on a product-structured labeling (n = 4096).
    var a = List[Int32]()
    var b = List[Int32]()
    for i in range(4096):
        a.append(Int32((i % 4) // 2))
        b.append(Int32(i % 2))
    var da = upload_i32(ctx, a)
    var db = upload_i32(ctx, b)
    var mi0 = mutual_info_score(ctx, da, db, 4096, 0, 1)
    if mi0 != 0.0:
        raise Error(
            "MI of a product-structured labeling must be 0.0 exactly, got "
            + String(mi0)
        )
    print("    product-structured 4096: MI == 0.0 exactly")
    # row 39: size = 0 would be `h_MI / size` = 0 / 0 (RAFT :161); refused
    var refused_s0 = False
    try:
        _ = mutual_info_score(ctx, da, db, 0, 0, 1)
    except e:
        print("    size=0: " + String(e))
        refused_s0 = True
    if not refused_s0:
        raise Error("mutual_info_score(size=0) must be refused by name (0 / 0)")
    print("  OK")


def check_homogeneity_completeness_v(ctx: DeviceContext) raises:
    print("check_homogeneity_completeness_v [" + _mode_name() + "]")
    var lp = labels_true_pred(N, 6, 5, 0.65, 51)
    var yt = lp[0].copy()
    var yp = lp[1].copy()
    var dt = upload_i32(ctx, yt)
    var dp = upload_i32(ctx, yp)
    var h = homogeneity_score(ctx, dt, dp, N, 0, 5)
    var cpl = completeness_score(ctx, dt, dp, N, 0, 5)
    var v = v_measure(ctx, dt, dp, N, 0, 5)
    var c = _host_contingency(yt, yp, 0, 6)
    var mi = Float64(_oracle_mi_f32(c, 6, N, False))
    # COMPLETENESS IS homogeneity(pred, truth): RAFT recomputes MI over the
    # TRANSPOSED matrix (completeness_score.cu swaps the arrays), and a
    # transposed fold is a different Float32 sum in the last bit. sklearn
    # computes one MI and uses it for both. The oracle mirrors RAFT.
    var mi_t = Float64(_oracle_mi_f32(_transpose(c, 6), 6, N, False))
    var ht = Float64(_oracle_entropy_f32(_counts_from(c, 6, True), N))
    var hp = Float64(_oracle_entropy_f32(_counts_from(c, 6, False), N))
    var oh = mi / ht if ht != 0.0 else 1.0
    var oc = mi_t / hp if hp != 0.0 else 1.0
    var ov = 0.0 if oh + oc == 0.0 else 2.0 * oh * oc / (oh + oc)
    _assert_bits64("homogeneity", h, oh, IDENTICAL)
    _assert_bits64("completeness", cpl, oc, IDENTICAL)
    _assert_bits64("v_measure", v, ov, IDENTICAL)
    var rmi = _ref_mi_f64(c, 6, N)
    var rht = _ref_entropy_f64(_counts_from(c, 6, True), N)
    var rhp = _ref_entropy_f64(_counts_from(c, 6, False), N)
    var rh = rmi / rht
    var rc = rmi / rhp
    var rv = 2.0 * rh * rc / (rh + rc)
    print(
        "    vs Float64 sklearn spelling: h rel "
        + String(_rel(h, rh))
        + " c rel "
        + String(_rel(cpl, rc))
        + " v rel "
        + String(_rel(v, rv))
    )
    if _rel(h, rh) > 1e-6 or _rel(cpl, rc) > 1e-6 or _rel(v, rv) > 1e-6:
        raise Error("h/c/v off the Float64 reference")
    # THE DEFINED BRANCHES, as sklearn defines them.
    var k0 = upload_i32(ctx, constant_labels(N, 0))
    # constant TRUTH: entropy_C = 0 -> homogeneity = 1.0; completeness =
    # MI / entropy_K = 0 / H_pred = 0 -> v = 2*1*0/(1+0) = 0
    var h1 = homogeneity_score(ctx, k0, dp, N, 0, 5)
    var c1 = completeness_score(ctx, k0, dp, N, 0, 5)
    var v1 = v_measure(ctx, k0, dp, N, 0, 5)
    if h1 != 1.0 or c1 != 0.0 or v1 != 0.0:
        raise Error(
            "constant truth: want (1, 0, 0) got ("
            + String(h1)
            + ", "
            + String(c1)
            + ", "
            + String(v1)
            + ")"
        )
    # constant PRED: h = 0, c = 1, v = 0
    var h2 = homogeneity_score(ctx, dt, k0, N, 0, 5)
    var c2 = completeness_score(ctx, dt, k0, N, 0, 5)
    var v2 = v_measure(ctx, dt, k0, N, 0, 5)
    if h2 != 0.0 or c2 != 1.0 or v2 != 0.0:
        raise Error(
            "constant pred: want (0, 1, 0) got ("
            + String(h2)
            + ", "
            + String(c2)
            + ", "
            + String(v2)
            + ")"
        )
    # both constant: (1, 1, 1)
    var k0b = upload_i32(ctx, constant_labels(N, 0))
    var v3 = v_measure(ctx, k0, k0b, N, 0, 5)
    if v3 != 1.0:
        raise Error("both constant: v must be 1.0, got " + String(v3))
    # perfect: (1, 1, 1) -- MI == H exactly? MI and H are computed by
    # different spellings (log(n c) - log(a b) vs -p log p), so equality
    # to the last bit is not promised by RAFT either; sklearn's test uses
    # assert_almost_equal. Gate to 1e-6 and report the bits.
    var dt2 = upload_i32(ctx, yt)
    var hp4 = homogeneity_score(ctx, dt, dt2, N, 0, 5)
    var v4 = v_measure(ctx, dt, dt2, N, 0, 5)
    print(
        "    perfect: h "
        + String(hp4)
        + " ("
        + bits64(hp4)
        + ") v "
        + String(v4)
    )
    if _rel(hp4, 1.0) > 1e-6 or _rel(v4, 1.0) > 1e-6:
        raise Error("perfect labeling must score ~1")
    # independent (product-structured): h = c = 0 -> v = 0 by the h+c==0 branch
    var a = List[Int32]()
    var b = List[Int32]()
    for i in range(4096):
        a.append(Int32((i % 4) // 2))
        b.append(Int32(i % 2))
    var da = upload_i32(ctx, a)
    var db = upload_i32(ctx, b)
    var v5 = v_measure(ctx, da, db, 4096, 0, 1)
    var h5 = homogeneity_score(ctx, da, db, 4096, 0, 1)
    if v5 != 0.0 or h5 != 0.0:
        raise Error("independent labelings: want h = 0, v = 0")
    # singleton cluster in pred: no branch, but the cell (t, 9) has count 1
    # and log(n * 1) - log(a_t * 1) is reached
    var yps = yp.copy()
    plant_singleton(yps, 23, Int32(9))
    var dps = upload_i32(ctx, yps)
    var vs = v_measure(ctx, dt, dps, N, 0, 9)
    var cs = _host_contingency(yt, yps, 0, 10)
    var mis = Float64(_oracle_mi_f32(cs, 10, N, False))
    var mis_t = Float64(_oracle_mi_f32(_transpose(cs, 10), 10, N, False))
    var hts = Float64(_oracle_entropy_f32(_counts_from(cs, 10, True), N))
    var hps = Float64(_oracle_entropy_f32(_counts_from(cs, 10, False), N))
    var ohs = mis / hts
    var ocs = mis_t / hps
    print(
        "    MI(truth, pred) "
        + bits64(mis)
        + " MI(pred, truth) "
        + bits64(mis_t)
        + (" (transposed fold differs)" if bits64(mis) != bits64(mis_t) else " (same bits here)")
    )
    _assert_bits64("v_measure(singleton)", vs, 2.0 * ohs * ocs / (ohs + ocs), IDENTICAL)
    print(
        "    constant truth (1,0,0), constant pred (0,1,0), both constant"
        " (1,1,1), independent (0,0,0), singleton: all as sklearn defines"
    )
    print("  OK")


def check_label_epilogue_order_is_visible(ctx: DeviceContext) raises:
    """The teeth of the bitwise gates: the SAME matrix folded descending
    gives different Float32 bits, and the Float64 reference differs from
    the Float32 oracle. If either were equal the gates above could not see
    a reordered or re-typed epilogue."""
    print("check_label_epilogue_order_is_visible [" + _mode_name() + "]")
    var lp = labels_true_pred(N, 9, 7, 0.55, 41)
    var c = _host_contingency(lp[0], lp[1], 0, 9)
    var asc = _oracle_mi_f32(c, 9, N, False)
    var desc = _oracle_mi_f32(c, 9, N, True)
    print(
        "    MI ascending "
        + bits32(asc)
        + " descending "
        + bits32(desc)
        + " float64 "
        + bits64(_ref_mi_f64(c, 9, N))
    )
    if bits32(asc) == bits32(desc):
        raise Error(
            "fixture cannot separate a reversed fold; the bitwise gate has no"
            " teeth"
        )
    if Float64(asc) == _ref_mi_f64(c, 9, N):
        raise Error("Float32 oracle equals the Float64 reference; unexpected")
    print("  OK")


def main() raises:
    print("== metrics/original/label_metrics_check.mojo [" + _mode_name() + "] ==")
    var ctx = DeviceContext()
    check_contingency_matrix_exact(ctx)
    check_accuracy_exact(ctx)
    check_rand_index_exact(ctx)
    check_adjusted_rand_index(ctx)
    check_entropy_epilogue(ctx)
    check_mutual_info_epilogue(ctx)
    check_homogeneity_completeness_v(ctx)
    check_label_epilogue_order_is_visible(ctx)
    print("ALL GROUP A CHECKS PASSED [" + _mode_name() + "]")
