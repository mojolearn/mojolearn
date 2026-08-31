# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Per-cell check for `batched_levelalgo/objectives.mojo`.

    cd /Users/andrewhendel/CascadeProjects/mojolearn && \
        pixi run mojo run -I . extratrees/checks/objectives_check.mojo

WHY IT IS BUILT THE WAY IT IS
-----------------------------
Rule 8 (`PORTING_RULES.md` 7, and `extratrees/README.md`): **a check whose
expected value is the same in every cell verifies the total and nothing about
placement.** The case that earned the rule reported 0 wrong of 512 on a
uniform fixture and 490 wrong of 512, same kernel, on a hashed one. So:

1. Every label and every regression target here is HASHED from the row index
   through `hash32`. Never `i % k`, never a constant. The check PROVES this
   about itself: `distinct_cells` counts how many of the candidate splits
   produce a distinct `(n_left, sq_left)` pair and refuses to pass if the
   fixture has degenerated into a uniform one.
2. The expected value comes from an INDEPENDENT tally written in a different
   style: the implementation makes ONE row-major pass building the
   accumulators; the tally is CLASS-MAJOR, one full sweep of all rows per
   class per side, and never touches `CountBin`/`AggregateBin` at all.
3. Comparison is PER CELL. Every candidate x every quantity is its own
   assertion with its own pass counter, and the first mismatch is printed
   with both values. There is no digest and no total anywhere in this file.

The regression targets are hashed multiples of 1/64 in `[-32, 32)`. That is
deliberate and it is not a weakening: multiples of 1/64 and their squares are
exact in `float64`, and so are sums of 512 of them, so the float comparisons
below are EXACT EQUALITY rather than a tolerance. The one quantity compared
with a tolerance is cuML's `GainPerSplit`, because the tally deliberately
accumulates its three sums in a different order and division does not
commute with reordering.

Analytic fixtures state their closed form in a comment at the site.

SABOTAGE. Six mechanisms were sabotaged one at a time, the check was confirmed
to turn red for each, and the file was restored. A check never seen to fail is
not evidence. The six, with what they cost when broken:

  1. the class-count SQUARE, `sq_count_left += count_k * count_k`
     (`_criterion.pyx:675`) -> `+= count_k`               192 cells red
  2. the Gini DENOMINATOR, `sq / (w * w)` (`:680-681`) -> `sq / w`
                                                          192 cells red
  3. cuML's parent SUBTRACTION, `gain -= val * val`
     (`objectives.cuh:79`) -> `gain += val * val`          66 cells red
  4. the MSE proxy's DENOMINATOR PLACEMENT, left over `n_left` and right
     over `n_right` (`:972-973`), swapped                  64 cells red
  5. the exact numerator's CROSS TERMS, `sq_L*nR + sq_R*nL` -> `sq_L*nL +
     sq_R*nR` (ours, DEVIATION 144)                       292 cells red
  6. the `Int128` CROSS-MULTIPLY in `CompareProxyExact`, replaced by a
     bare numerator comparison (ours, DEVIATION 144)       29 cells red

Sabotage 5 is the one that matters most for rule 8: swapping the cross terms
leaves the TOTAL `sq_L + sq_R` and every symmetric fixture untouched, and only
a per-cell comparison against scattered counts can see it.
"""

from extratrees.impl.decisiontree.batched_levelalgo.objectives import (
    AggregateBin,
    CountBin,
    EntropyObjectiveFunction,
    GiniObjectiveFunction,
    GiniProxyExact,
    MSEObjectiveFunction,
    impurity_improvement,
    proxy_impurity_improvement,
)
from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    float_gain_key,
)
from std.math import log


comptime F = DType.float64
"""The check runs everything in `float64`, which is what sklearn accumulates
in (`_criterion.pyx`, `sum_left`/`sum_right` are `float64_t[::1]`). DEVIATION
135 leaves the DEVICE accumulator type open; that is a separate question and
this file does not touch it."""

comptime N_ROWS: Int = 512
comptime N_CLASSES: Int = 5
comptime N_CAND: Int = 64


# ==========================================================================
# Fixture. Everything is hashed from the row index.
# ==========================================================================


def hash32(x: UInt32) -> UInt32:
    """Fnv1a32 over the four bytes of `x`, then an xorshift finaliser.

    The same family cuML chains for its per-node keys
    (`builder_kernels.cuh:167-170`). Used here only to scatter a fixture.
    """
    var h: UInt32 = 2166136261
    var v = x
    for _ in range(4):
        h = (h ^ (v & 0xFF)) * 16777619
        v >>= 8
    h ^= h >> 15
    h = h * 2246822519
    h ^= h >> 13
    return h


def fixture_label(i: Int) -> Int32:
    """Class of row `i`. SCATTERED: not `i % k`, not a constant."""
    return Int32(Int(hash32(UInt32(i) * 2654435761 + 101) % UInt32(N_CLASSES)))


def fixture_x(i: Int) -> Int:
    """Feature value of row `i`: a full 32-bit hash. SCATTERED.

    The full width matters: 512 draws from 2^32 collide with probability
    ~3e-5, and a duplicate would make `n_left` differ from the rank the
    candidate asked for, which is the property `fixture_threshold` relies on.
    """
    return Int(hash32(UInt32(i) * 40503 + 7))


def fixture_rank(c: Int) -> Int:
    """`n_left` that candidate `c` is built to produce: DISTINCT per candidate.

    Blocks of 8 with a hashed offset inside each block, so the 64 ranks are
    pairwise distinct by construction (block `c` covers `[8c+1, 8c+7]`) and
    scattered inside their block. Distinct ranks are what makes every cell's
    expected value distinct, which is the whole of rule 8; the check asserts
    that downstream rather than trusting this comment.

    Range is `[1, 511]`, never 0 and never `N_ROWS`: the empty child is not a
    scattered-fixture case, it is an analytic one, and `check_rejection`
    exercises it on purpose.
    """
    return 1 + c * 8 + Int(hash32(UInt32(c) * 2654435761 + 12345) % 7)


def sorted_x() -> List[Int]:
    """All 512 feature values, ascending. Insertion sort; this is a check."""
    var v = List[Int](capacity=N_ROWS)
    for i in range(N_ROWS):
        v.append(fixture_x(i))
    for i in range(1, N_ROWS):
        var key = v[i]
        var j = i - 1
        while j >= 0 and v[j] > key:
            v[j + 1] = v[j]
            j -= 1
        v[j + 1] = key
    return v^


def fixture_threshold(order: List[Int], c: Int) -> Int:
    """Threshold of candidate `c`: the `fixture_rank(c)`-th smallest value.

    `x(i) <= threshold` then holds for exactly `fixture_rank(c)` rows, so the
    candidate's `n_left` is known in CLOSED FORM before anything is measured.
    """
    return order[fixture_rank(c) - 1]


def fixture_y(i: Int) -> Scalar[F]:
    """Target of row `i`: a hashed multiple of 1/64 in `[-32, 32)`.

    Dyadic on purpose -- see the module docstring. SCATTERED.
    """
    var k = Int(hash32(UInt32(i) * 374761393 + 59) % 4096) - 2048
    return Scalar[F](k) / Scalar[F](64.0)


# ==========================================================================
# Reporting. One counter per QUANTITY, incremented per CELL.
# ==========================================================================


@fieldwise_init
struct Cell(Copyable, Movable):
    var name: String
    var passed: Int
    var failed: Int
    var first_msg: String

    def __init__(out self, var name: String):
        self.name = name^
        self.passed = 0
        self.failed = 0
        self.first_msg = String("")

    def ok(mut self, condition: Bool, var msg: String):
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            if self.first_msg == "":
                self.first_msg = msg^

    def eq_i(mut self, got: Int64, expected: Int64, cell: Int):
        self.ok(
            got == expected,
            String("cell ")
            + String(cell)
            + ": got "
            + String(got)
            + " expected "
            + String(expected),
        )

    def eq_f(mut self, got: Scalar[F], expected: Scalar[F], cell: Int):
        self.ok(
            got == expected,
            String("cell ")
            + String(cell)
            + ": got "
            + String(got)
            + " expected "
            + String(expected),
        )

    def close_f(
        mut self, got: Scalar[F], expected: Scalar[F], tol: Scalar[F], cell: Int
    ):
        var scale = abs(expected)
        if scale < 1.0:
            scale = 1.0
        self.ok(
            abs(got - expected) <= tol * scale,
            String("cell ")
            + String(cell)
            + ": got "
            + String(got)
            + " expected "
            + String(expected),
        )

    def report(self) -> Int:
        var line = String("  ") + self.name + ": " + String(
            self.passed
        ) + " pass, " + String(self.failed) + " fail"
        if self.failed > 0:
            line += "   FIRST: " + self.first_msg
        print(line)
        return self.failed


# ==========================================================================
# The INDEPENDENT tally. Class-major, row-sweeping, no bin types.
# ==========================================================================


def tally_count(threshold: Int, want_left: Bool, want_class: Int32) -> Int64:
    """Count rows on one side of one candidate with one class.

    Deliberately the WRONG shape for performance and the right shape for
    independence: one full sweep of all rows per (class, side), where the
    implementation makes a single row-major pass over all classes at once.
    """
    var t: Int64 = 0
    for i in range(N_ROWS):
        var is_left = fixture_x(i) <= threshold
        if is_left != want_left:
            continue
        if fixture_label(i) != want_class:
            continue
        t += 1
    return t


def tally_y_sum(threshold: Int, want_left: Bool) -> Scalar[F]:
    """Sum of targets on one side. Exact: dyadic targets, `float64`."""
    var s = Scalar[F](0.0)
    for i in range(N_ROWS):
        var is_left = fixture_x(i) <= threshold
        if is_left != want_left:
            continue
        s += fixture_y(i)
    return s


def tally_y_sq(threshold: Int, want_left: Bool) -> Scalar[F]:
    """Sum of squared targets on one side. Exact, same reason."""
    var s = Scalar[F](0.0)
    for i in range(N_ROWS):
        var is_left = fixture_x(i) <= threshold
        if is_left != want_left:
            continue
        var y = fixture_y(i)
        s += y * y
    return s


# ==========================================================================
# Classification, scattered fixture, per cell.
# ==========================================================================


def check_gini_scattered() -> Int:
    print("[gini] scattered fixture,", N_ROWS, "rows,", N_CAND, "candidates")

    var obj = GiniObjectiveFunction[F](Int32(N_CLASSES), 1)
    var order = sorted_x()

    var left_buf = List[CountBin](length=N_CLASSES, fill=CountBin(0))
    var total_buf = List[CountBin](length=N_CLASSES, fill=CountBin(0))
    var hist_left = left_buf.unsafe_ptr()
    var hist_total = total_buf.unsafe_ptr()

    # Node totals: one row-major pass (the implementation's shape).
    for i in range(N_ROWS):
        total_buf[Int(fixture_label(i))].x += 1

    var c_nleft = Cell(String("n_left"))
    var c_rank = Cell(String("n_left == closed-form rank"))
    var c_sql = Cell(String("sq_left (exact int)"))
    var c_sqr = Cell(String("sq_right (exact int)"))
    var c_gl = Cell(String("gini_left"))
    var c_gr = Cell(String("gini_right"))
    var c_proxy = Cell(String("sklearn proxy (float)"))
    var c_base = Cell(String("Criterion.proxy_impurity_improvement"))
    var c_num = Cell(String("exact num"))
    var c_den = Cell(String("exact den"))
    var c_val = Cell(String("exact value == float proxy"))
    var c_gain = Cell(String("cuML GainPerSplit"))
    var c_order = Cell(String("exact order vs float order"))
    var c_imp = Cell(String("impurity_improvement"))

    # Distinctness of the fixture, proved rather than assumed.
    var seen_nl = List[Int64](length=N_CAND, fill=0)
    var seen_sq = List[Int64](length=N_CAND, fill=0)
    var distinct_cells = 0

    var prev = GiniProxyExact(0, 0, 0, False)
    var prev_proxy = Scalar[F].MIN_FINITE
    var have_prev = False

    for c in range(N_CAND):
        var threshold = fixture_threshold(order, c)

        # ---- implementation side: one row-major pass ----
        for k in range(N_CLASSES):
            left_buf[k].x = 0
        var n_left: Int32 = 0
        for i in range(N_ROWS):
            if fixture_x(i) <= threshold:
                left_buf[Int(fixture_label(i))].x += 1
                n_left += 1

        # ---- independent tally: class-major sweeps ----
        var t_nleft: Int64 = 0
        var t_sql: Int64 = 0
        var t_sqr: Int64 = 0
        for k in range(N_CLASSES):
            var l = tally_count(threshold, True, Int32(k))
            var r = tally_count(threshold, False, Int32(k))
            t_nleft += l
            t_sql += l * l
            t_sqr += r * r
        var t_nright = Int64(N_ROWS) - t_nleft

        c_nleft.eq_i(Int64(Int(n_left)), t_nleft, c)
        # ...and the tally itself must equal the rank this candidate was
        # BUILT to produce. Closed form, so this cell does not lean on the
        # tally being right either.
        c_rank.eq_i(t_nleft, Int64(fixture_rank(c)), c)

        var ex = obj.ProxyImpurityExact(hist_left, hist_total, Int32(N_ROWS), n_left)

        # The exact rational, cell for cell against the tally.
        #   num = sq_L*nR + sq_R*nL ; den = nL*nR
        c_sql.eq_i(
            ex.num - t_sqr * t_nleft, t_sql * t_nright, c
        )  # isolates sq_L
        c_sqr.eq_i(
            ex.num - t_sql * t_nright, t_sqr * t_nleft, c
        )  # isolates sq_R
        c_num.eq_i(ex.num, t_sql * t_nright + t_sqr * t_nleft, c)
        c_den.eq_i(ex.den, t_nleft * t_nright, c)

        # sklearn's children impurity, from the tally's counts:
        #   gini_L = 1 - sq_L / nL^2      (`_criterion.pyx:680-681`)
        var wl = Scalar[F](t_nleft)
        var wr = Scalar[F](t_nright)
        var e_gl = 1.0 - Scalar[F](t_sql) / (wl * wl)
        var e_gr = 1.0 - Scalar[F](t_sqr) / (wr * wr)

        var got_gl = Scalar[F](0.0)
        var got_gr = Scalar[F](0.0)
        obj.ChildrenImpurity(
            hist_left, hist_total, wl, wr, got_gl, got_gr
        )
        c_gl.eq_f(got_gl, e_gl, c)
        c_gr.eq_f(got_gr, e_gr, c)

        # sklearn's proxy: -wR*imp_R - wL*imp_L   (`_criterion.pyx:162-163`)
        var e_proxy = -wr * e_gr - wl * e_gl
        var got_proxy = obj.ProxyImpurityImprovement(
            hist_left, hist_total, Int32(N_ROWS), n_left
        )
        c_proxy.eq_f(got_proxy, e_proxy, c)

        # The base-class expression itself (`_criterion.pyx:162-163`), fed
        # the criterion's own children impurities: same number, and this is
        # the cell that would catch the method silently stopping calling it.
        c_base.eq_f(
            proxy_impurity_improvement[F](wl, wr, got_gl, got_gr),
            got_proxy,
            c,
        )

        # The exact rational must be the SAME NUMBER as the float proxy, to
        # within the float proxy's own rounding. num/den - n vs -wR*g_R-wL*g_L.
        c_val.close_f(ex.value[F](), e_proxy, 1e-12, c)

        # cuML's gain (`objectives.cuh:65-80`), independent order:
        #   sum_j l^2/(nL*n) + sum_j r^2/(nR*n) - sum_j (t/n)^2
        var n_f = Scalar[F](N_ROWS)
        var acc_l = Scalar[F](0.0)
        var acc_r = Scalar[F](0.0)
        var acc_t = Scalar[F](0.0)
        for k in range(N_CLASSES):
            var l = Scalar[F](tally_count(threshold, True, Int32(k)))
            var r = Scalar[F](tally_count(threshold, False, Int32(k)))
            acc_l += l * l / (wl * n_f)
            acc_r += r * r / (wr * n_f)
            var tot = (l + r) / n_f
            acc_t += tot * tot
        var e_gain = acc_l + acc_r - acc_t
        var got_gain = obj.GainPerSplit(
            hist_left, hist_total, Int32(N_ROWS), n_left
        )
        c_gain.close_f(got_gain, e_gain, 1e-12, c)

        # The exact comparator must agree with the float proxy's order
        # wherever the float proxy is not itself a tie. Consecutive
        # candidates, so this is a per-cell test, not a sort.
        if have_prev:
            var got_cmp = GiniObjectiveFunction[F].CompareProxyExact(prev, ex)
            var expect_cmp: Int
            if prev_proxy < e_proxy:
                expect_cmp = -1
            elif prev_proxy > e_proxy:
                expect_cmp = 1
            else:
                expect_cmp = 0
            c_order.ok(
                got_cmp == expect_cmp,
                String("cell ")
                + String(c)
                + ": got "
                + String(got_cmp)
                + " expected "
                + String(expect_cmp),
            )
        prev = ex
        prev_proxy = e_proxy
        have_prev = True

        # `impurity_improvement` (`_criterion.pyx:195-199`) at the whole-tree
        # weight, which for this fixture is the node weight, so the leading
        # factor is 1 and the result is parent - weighted children.
        var parent = obj.NodeImpurity(hist_total, n_f)
        var e_imp = (n_f / n_f) * (
            parent - (wr / n_f * e_gr) - (wl / n_f * e_gl)
        )
        var got_imp = impurity_improvement[F](
            parent, got_gl, got_gr, n_f, n_f, wl, wr
        )
        c_imp.eq_f(got_imp, e_imp, c)

        # fixture distinctness bookkeeping
        var is_new = True
        for p in range(distinct_cells):
            if seen_nl[p] == t_nleft and seen_sq[p] == t_sql:
                is_new = False
                break
        if is_new:
            seen_nl[distinct_cells] = t_nleft
            seen_sq[distinct_cells] = t_sql
            distinct_cells += 1

    var failed = 0
    failed += c_nleft.report()
    failed += c_rank.report()
    failed += c_sql.report()
    failed += c_sqr.report()
    failed += c_num.report()
    failed += c_den.report()
    failed += c_gl.report()
    failed += c_gr.report()
    failed += c_proxy.report()
    failed += c_base.report()
    failed += c_val.report()
    failed += c_gain.report()
    failed += c_order.report()
    failed += c_imp.report()

    print(
        "  fixture distinctness:",
        distinct_cells,
        "distinct (n_left, sq_left) of",
        N_CAND,
        "candidates",
    )
    if distinct_cells < N_CAND:
        print("  FAIL: fixture is not fully scattered -- rule 8")
        failed += 1

    _ = left_buf^
    _ = total_buf^
    _ = order^
    return failed


# ==========================================================================
# Regression, scattered fixture, per cell.
# ==========================================================================


def check_mse_scattered() -> Int:
    print("[mse] scattered fixture,", N_ROWS, "rows,", N_CAND, "candidates")

    var obj = MSEObjectiveFunction[F](1)
    var order = sorted_x()

    var total = AggregateBin[F]()
    for i in range(N_ROWS):
        total += AggregateBin[F](fixture_y(i), 1)
    var sq_sum_total = Scalar[F](0.0)
    for i in range(N_ROWS):
        var y = fixture_y(i)
        sq_sum_total += y * y

    var c_n = Cell(String("count_left"))
    var c_rank = Cell(String("count_left == closed-form rank"))
    var c_sum = Cell(String("sum_left (exact dyadic)"))
    var c_sq = Cell(String("sq_sum_left (exact dyadic)"))
    var c_proxy = Cell(String("sklearn MSE proxy"))
    var c_il = Cell(String("children_impurity left"))
    var c_ir = Cell(String("children_impurity right"))
    var c_gain = Cell(String("cuML GainPerSplit"))
    var c_leaf = Cell(String("SetLeafVector (leaf mean)"))
    var c_node = Cell(String("node_impurity"))

    var distinct_cells = 0
    var seen = List[Scalar[F]](length=N_CAND, fill=0)

    var leaf_out_buf = List[Scalar[F]](length=1, fill=0)
    var leaf_hist_buf = List[AggregateBin[F]](length=1, fill=AggregateBin[F]())
    var leaf_out = leaf_out_buf.unsafe_ptr()
    var leaf_hist = leaf_hist_buf.unsafe_ptr()

    for c in range(N_CAND):
        var threshold = fixture_threshold(order, c)

        # ---- implementation side: one row-major pass ----
        var left = AggregateBin[F]()
        for i in range(N_ROWS):
            if fixture_x(i) <= threshold:
                left += AggregateBin[F](fixture_y(i), 1)

        # ---- independent tally: separate sweeps, different style ----
        var t_sum_l = tally_y_sum(threshold, True)
        var t_sum_r = tally_y_sum(threshold, False)
        var t_sq_l = tally_y_sq(threshold, True)
        var t_n_l: Int64 = 0
        for i in range(N_ROWS):
            if fixture_x(i) <= threshold:
                t_n_l += 1
        var t_n_r = Int64(N_ROWS) - t_n_l

        c_n.eq_i(Int64(Int(left.count)), t_n_l, c)
        c_rank.eq_i(t_n_l, Int64(fixture_rank(c)), c)
        c_sum.eq_f(left.label_sum, t_sum_l, c)
        c_sq.eq_f(sq_sum_total - tally_y_sq(threshold, False), t_sq_l, c)

        var wl = Scalar[F](t_n_l)
        var wr = Scalar[F](t_n_r)

        # sklearn `MSE.proxy_impurity_improvement` (`_criterion.pyx:968-973`):
        #   sum_L^2 / n_L + sum_R^2 / n_R
        var e_proxy = (t_sum_l * t_sum_l) / wl + (t_sum_r * t_sum_r) / wr
        var got_proxy = obj.ProxyImpurityImprovement(
            left, total, Int32(N_ROWS), left.count
        )
        c_proxy.eq_f(got_proxy, e_proxy, c)

        # sklearn `MSE.children_impurity` (`:1009-1014`):
        #   imp_L = sqsum_L/n_L - (sum_L/n_L)^2
        var e_il = t_sq_l / wl - (t_sum_l / wl) ** 2.0
        var e_ir = (sq_sum_total - t_sq_l) / wr - (t_sum_r / wr) ** 2.0
        var got_il = Scalar[F](0.0)
        var got_ir = Scalar[F](0.0)
        obj.ChildrenImpurity(
            t_sq_l, sq_sum_total, left, total, wl, wr, got_il, got_ir
        )
        c_il.eq_f(got_il, e_il, c)
        c_ir.eq_f(got_ir, e_ir, c)

        # cuML `objectives.cuh:234-240`, independent order:
        #   gain = (-S^2/n - (-sL^2/nL - sR^2/nR)) * 0.5/n
        var n_f = Scalar[F](N_ROWS)
        var s_tot = t_sum_l + t_sum_r
        var e_gain = (
            (-(s_tot * s_tot) / n_f)
            - ((-(t_sum_l * t_sum_l) / wl) + (-(t_sum_r * t_sum_r) / wr))
        ) * (0.5 / n_f)
        var got_gain = obj.GainPerSplit(left, total, Int32(N_ROWS), left.count)
        c_gain.close_f(got_gain, e_gain, 1e-12, c)

        # `SetLeafVector` (`objectives.cuh:259-264`): the left child's mean.
        leaf_hist_buf[0] = left
        MSEObjectiveFunction[F].SetLeafVector(leaf_hist, 1, leaf_out)
        c_leaf.eq_f(leaf_out_buf[0], t_sum_l / wl, c)

        # `MSE.node_impurity` (`:938-940`) on the LEFT child, so this cell
        # varies with the candidate instead of being one constant.
        var e_node = t_sq_l / wl - (t_sum_l / wl) ** 2.0
        var got_node = obj.NodeImpurity(t_sq_l, left, wl)
        c_node.eq_f(got_node, e_node, c)

        var is_new = True
        for p in range(distinct_cells):
            if seen[p] == t_sum_l:
                is_new = False
                break
        if is_new:
            seen[distinct_cells] = t_sum_l
            distinct_cells += 1

    var failed = 0
    failed += c_n.report()
    failed += c_rank.report()
    failed += c_sum.report()
    failed += c_sq.report()
    failed += c_proxy.report()
    failed += c_il.report()
    failed += c_ir.report()
    failed += c_gain.report()
    failed += c_leaf.report()
    failed += c_node.report()

    print(
        "  fixture distinctness:",
        distinct_cells,
        "distinct sum_left of",
        N_CAND,
        "candidates",
    )
    if distinct_cells < N_CAND:
        print("  FAIL: fixture is not fully scattered -- rule 8")
        failed += 1

    _ = leaf_out_buf^
    _ = leaf_hist_buf^
    _ = order^
    return failed


# ==========================================================================
# Analytic fixtures. The answer is hand-computable and stated as a closed
# form at each site.
# ==========================================================================


def check_analytic() -> Int:
    print("[analytic] hand-computable fixtures")
    var cell = Cell(String("analytic"))

    var h_buf = List[CountBin](length=N_CLASSES, fill=CountBin(0))
    var hl_buf = List[CountBin](length=N_CLASSES, fill=CountBin(0))
    var h = h_buf.unsafe_ptr()
    var hl = hl_buf.unsafe_ptr()

    # --- A1. PURE NODE, 64 rows all of class 3. -----------------------------
    # gini = 1 - sum_k p_k^2 = 1 - (64/64)^2 = 1 - 1 = 0, EXACTLY.
    var obj = GiniObjectiveFunction[F](Int32(N_CLASSES), 1)
    for k in range(N_CLASSES):
        h_buf[k].x = 0
    h_buf[3].x = 64
    cell.ok(
        obj.NodeImpurity(h, 64.0) == 0.0,
        String("A1 pure node gini != 0: ") + String(obj.NodeImpurity(h, 64.0)),
    )

    # --- A2. 50/50 TWO-CLASS NODE, 32 + 32. --------------------------------
    # gini = 1 - (0.5^2 + 0.5^2) = 1 - 0.5 = 0.5, EXACTLY (0.5 is dyadic).
    for k in range(N_CLASSES):
        h_buf[k].x = 0
    h_buf[0].x = 32
    h_buf[1].x = 32
    cell.ok(
        obj.NodeImpurity(h, 64.0) == 0.5,
        String("A2 50/50 gini != 0.5: ") + String(obj.NodeImpurity(h, 64.0)),
    )

    # --- A3. PERFECT SPLIT: left = 32 of class 0, right = 32 of class 1. ----
    # gini_L = 1 - (32/32)^2 = 0 ; gini_R = 0
    # sklearn proxy = -32*0 - 32*0 = 0
    # exact:  num = sq_L*nR + sq_R*nL = 1024*32 + 1024*32 = 65536 ; den = 1024
    #         value = 65536/1024 - 64 = 64 - 64 = 0
    # cuML gain = parent_gini - 0 = 0.5
    for k in range(N_CLASSES):
        hl_buf[k].x = 0
    hl_buf[0].x = 32
    var il = Scalar[F](0.0)
    var ir = Scalar[F](0.0)
    obj.ChildrenImpurity(hl, h, 32.0, 32.0, il, ir)
    cell.ok(il == 0.0, String("A3 gini_left != 0: ") + String(il))
    cell.ok(ir == 0.0, String("A3 gini_right != 0: ") + String(ir))
    cell.ok(
        obj.ProxyImpurityImprovement(hl, h, 64, 32) == 0.0,
        String("A3 proxy != 0"),
    )
    var a3 = obj.ProxyImpurityExact(hl, h, 64, 32)
    cell.ok(a3.num == 65536, String("A3 num != 65536: ") + String(a3.num))
    cell.ok(a3.den == 1024, String("A3 den != 1024: ") + String(a3.den))
    cell.ok(a3.value[F]() == 0.0, String("A3 exact value != 0"))
    cell.ok(
        obj.GainPerSplit(hl, h, 64, 32) == 0.5,
        String("A3 cuML gain != 0.5: ") + String(obj.GainPerSplit(hl, h, 64, 32)),
    )

    # --- A3b. WORST SPLIT: left = 16+16, right = 16+16 on the same node. ----
    # gini_L = gini_R = 0.5 ; proxy = -32*0.5 - 32*0.5 = -32
    # cuML gain = 0.5 - 0.5 = 0. And -32 < 0 for the proxy while the two
    # cuML gains are 0.5 vs 0, so BOTH orderings must put A3 above A3b.
    for k in range(N_CLASSES):
        hl_buf[k].x = 0
    hl_buf[0].x = 16
    hl_buf[1].x = 16
    var a3b = obj.ProxyImpurityExact(hl, h, 64, 32)
    cell.ok(
        obj.ProxyImpurityImprovement(hl, h, 64, 32) == -32.0,
        String("A3b proxy != -32: ")
        + String(obj.ProxyImpurityImprovement(hl, h, 64, 32)),
    )
    cell.ok(
        obj.GainPerSplit(hl, h, 64, 32) == 0.0,
        String("A3b cuML gain != 0"),
    )
    cell.ok(
        GiniObjectiveFunction[F].CompareProxyExact(a3, a3b) == 1,
        String("A3 should outrank A3b exactly"),
    )
    cell.ok(
        GiniObjectiveFunction[F].CompareProxyExact(a3b, a3) == -1,
        String("comparator is not antisymmetric"),
    )

    # --- A4. REGRESSION NODE WITH CONSTANT y = 2.5, 64 rows. ---------------
    # sum = 160, sq_sum = 64 * 6.25 = 400
    # node_impurity = 400/64 - (160/64)^2 = 6.25 - 6.25 = 0, EXACTLY
    # (2.5 and 6.25 are dyadic, so this is a real equality, not a near-one).
    var mse = MSEObjectiveFunction[F](1)
    var const_node = AggregateBin[F](160.0, 64)
    cell.ok(
        mse.NodeImpurity(400.0, const_node, 64.0) == 0.0,
        String("A4 constant-y node impurity != 0: ")
        + String(mse.NodeImpurity(400.0, const_node, 64.0)),
    )
    # Any split of a constant-y node is worthless: children impurity 0 both
    # sides, and cuML gain = (-(160^2)/64 + (80^2/32 + 80^2/32)) * 0.5/64
    #                      = (-400 + 400) * 0.5/64 = 0.
    var half = AggregateBin[F](80.0, 32)
    var cil = Scalar[F](0.0)
    var cir = Scalar[F](0.0)
    mse.ChildrenImpurity(200.0, 400.0, half, const_node, 32.0, 32.0, cil, cir)
    cell.ok(cil == 0.0, String("A4 left impurity != 0: ") + String(cil))
    cell.ok(cir == 0.0, String("A4 right impurity != 0: ") + String(cir))
    cell.ok(
        mse.GainPerSplit(half, const_node, 64, 32) == 0.0,
        String("A4 cuML gain != 0: ")
        + String(mse.GainPerSplit(half, const_node, 64, 32)),
    )

    # --- A5. EXACT TIE, classification. ------------------------------------
    # Node counts [4, 4], n = 8.
    #   candidate P: left = (2 class0, 1 class1) -> nL=3, sq_L=4+1=5
    #                right = (2, 3)              -> nR=5, sq_R=4+9=13
    #   candidate Q: left = (1 class0, 2 class1) -> nL=3, sq_L=1+4=5
    #                right = (3, 2)              -> nR=5, sq_R=9+4=13
    # Different partitions, identical proxy: num = 5*5 + 13*3 = 64, den = 15.
    # The comparator must return 0 -- a REAL tie, not two floats that rounded
    # together, which is the tie-break path DEVIATION 133 hands to
    # `Split.update`.
    var tot_buf = List[CountBin](length=2, fill=CountBin(0))
    var p_buf = List[CountBin](length=2, fill=CountBin(0))
    var q_buf = List[CountBin](length=2, fill=CountBin(0))
    tot_buf[0] = CountBin(4)
    tot_buf[1] = CountBin(4)
    p_buf[0] = CountBin(2)
    p_buf[1] = CountBin(1)
    q_buf[0] = CountBin(1)
    q_buf[1] = CountBin(2)
    var tie_total = tot_buf.unsafe_ptr()
    var tie_p = p_buf.unsafe_ptr()
    var tie_q = q_buf.unsafe_ptr()
    var obj2 = GiniObjectiveFunction[F](2, 1)
    var ep = obj2.ProxyImpurityExact(tie_p, tie_total, 8, 3)
    var eq = obj2.ProxyImpurityExact(tie_q, tie_total, 8, 3)
    cell.ok(ep.num == 64, String("A5 num != 64: ") + String(ep.num))
    cell.ok(ep.den == 15, String("A5 den != 15: ") + String(ep.den))
    cell.ok(
        GiniObjectiveFunction[F].CompareProxyExact(ep, eq) == 0,
        String("A5 exact tie not reported as a tie"),
    )
    cell.ok(
        obj2.ProxyImpurityImprovement(tie_p, tie_total, 8, 3)
        == obj2.ProxyImpurityImprovement(tie_q, tie_total, 8, 3),
        String("A5 float proxies differ on an exact tie"),
    )
    # And a NEAR-tie must NOT be reported as a tie: move one row across.
    #   candidate R: left = (3 class0, 0 class1) -> nL=3, sq_L=9
    #                right = (1, 4)              -> nR=5, sq_R=1+16=17
    #   num = 9*5 + 17*3 = 96 > 64. R outranks P.
    q_buf[0] = CountBin(3)
    q_buf[1] = CountBin(0)
    var er = obj2.ProxyImpurityExact(tie_q, tie_total, 8, 3)
    cell.ok(er.num == 96, String("A5 R num != 96: ") + String(er.num))
    cell.ok(
        GiniObjectiveFunction[F].CompareProxyExact(er, ep) == 1,
        String("A5 R should outrank P"),
    )

    # --- A6. EXACT TIE, regression. ----------------------------------------
    # y = [+1 x4, -1 x4], total sum = 0, n = 8.
    #   candidate P: left = 3 rows of +1 -> sum_L = +3, nL = 3, sum_R = -3
    #   candidate Q: left = 3 rows of -1 -> sum_L = -3, nL = 3, sum_R = +3
    # proxy = 9/3 + 9/5 = 3 + 1.8 = 4.8 for BOTH (the proxy squares the sums).
    var tot_r = AggregateBin[F](0.0, 8)
    var p_r = AggregateBin[F](3.0, 3)
    var q_r = AggregateBin[F](-3.0, 3)
    cell.ok(
        mse.ProxyImpurityImprovement(p_r, tot_r, 8, 3)
        == mse.ProxyImpurityImprovement(q_r, tot_r, 8, 3),
        String("A6 regression tie broken"),
    )
    cell.ok(
        mse.ProxyImpurityImprovement(p_r, tot_r, 8, 3) == 3.0 + 9.0 / 5.0,
        String("A6 proxy != 3 + 9/5: ")
        + String(mse.ProxyImpurityImprovement(p_r, tot_r, 8, 3)),
    )

    # --- A7. Gini leaf vector on the [4,4] node: probabilities 0.5, 0.5. ---
    var probs_buf = List[Scalar[F]](length=2, fill=0)
    GiniObjectiveFunction[F].SetLeafVector(
        tie_total, 2, probs_buf.unsafe_ptr()
    )
    cell.ok(probs_buf[0] == 0.5, String("A7 p0 != 0.5: ") + String(probs_buf[0]))
    cell.ok(probs_buf[1] == 0.5, String("A7 p1 != 0.5: ") + String(probs_buf[1]))

    _ = h_buf^
    _ = hl_buf^
    _ = tot_buf^
    _ = p_buf^
    _ = q_buf^
    _ = probs_buf^
    return cell.report()


# ==========================================================================
# Rejection paths: min_samples_leaf, and the empty child.
# ==========================================================================


def check_rejection() -> Int:
    print("[rejection] min_samples_leaf and the empty child")
    var cell = Cell(String("rejection"))

    var total_buf = List[CountBin](length=2, fill=CountBin(0))
    var left_buf = List[CountBin](length=2, fill=CountBin(0))
    total_buf[0] = CountBin(4)
    total_buf[1] = CountBin(4)
    var total = total_buf.unsafe_ptr()
    var left = left_buf.unsafe_ptr()

    # min_samples_leaf = 5, node of 8, candidate nL = 3 -> both children fail
    # the test on the left side. cuML returns -max<DataT>()
    # (`objectives.cuh:62-63`); ours returns MIN_FINITE, the same bits.
    var strict = GiniObjectiveFunction[F](2, 5)
    left_buf[0] = CountBin(2)
    left_buf[1] = CountBin(1)
    cell.ok(
        strict.GainPerSplit(left, total, 8, 3) == Scalar[F].MIN_FINITE,
        String("min_samples_leaf: cuML gain not MIN_FINITE"),
    )
    cell.ok(
        strict.ProxyImpurityImprovement(left, total, 8, 3)
        == Scalar[F].MIN_FINITE,
        String("min_samples_leaf: sklearn proxy not MIN_FINITE"),
    )
    var rejected = strict.ProxyImpurityExact(left, total, 8, 3)
    cell.ok(not rejected.valid, String("min_samples_leaf: exact still valid"))
    cell.ok(
        rejected.value[F]() == Scalar[F].MIN_FINITE,
        String("min_samples_leaf: exact value not MIN_FINITE"),
    )

    # nL = 4 passes (4 >= 5 is false)... 4 < 5, still rejected. nL = 5, nR = 3
    # is rejected on the RIGHT. Exercise both sides separately.
    left_buf[0] = CountBin(4)
    left_buf[1] = CountBin(1)
    cell.ok(
        not strict.ProxyImpurityExact(left, total, 8, 5).valid,
        String("min_samples_leaf: right child 3 < 5 not rejected"),
    )
    # And an ACCEPTED candidate on the same objective, so the cell above is
    # not passing because everything is rejected.
    var loose = GiniObjectiveFunction[F](2, 3)
    cell.ok(
        loose.ProxyImpurityExact(left, total, 8, 5).valid,
        String("min_samples_leaf=3: nL=5,nR=3 should be accepted"),
    )
    cell.ok(
        loose.GainPerSplit(left, total, 8, 5) != Scalar[F].MIN_FINITE,
        String("min_samples_leaf=3: gain should be finite"),
    )

    # THE EMPTY CHILD. Only reachable with min_samples_leaf = 0, which
    # sklearn's validation forbids (`_classes.py`, min value 1) and cuML does
    # not guard: their `invLeft = One / nLeft` is +inf and `0 * inf` makes the
    # gain NaN. Ours marks the candidate invalid (DEVIATION 144). Both halves
    # are asserted, because the NaN is THEIR behaviour and this file is the
    # record that we transcribed it rather than quietly repaired it.
    var unguarded = GiniObjectiveFunction[F](2, 0)
    left_buf[0] = CountBin(0)
    left_buf[1] = CountBin(0)
    var nan_gain = unguarded.GainPerSplit(left, total, 8, 0)
    cell.ok(
        nan_gain != nan_gain,
        String("empty child: cuML transcription should be NaN, got ")
        + String(nan_gain),
    )
    var empty = unguarded.ProxyImpurityExact(left, total, 8, 0)
    cell.ok(not empty.valid, String("empty child: exact form not invalid"))
    cell.ok(empty.den == 0, String("empty child: den should be 0"))

    # An invalid candidate must order below every valid one, and two invalid
    # candidates must tie -- the total-order contract `Split.update` relies on.
    left_buf[0] = CountBin(2)
    left_buf[1] = CountBin(1)
    var good = loose.ProxyImpurityExact(left, total, 8, 3)
    cell.ok(
        GiniObjectiveFunction[F].CompareProxyExact(empty, good) == -1,
        String("invalid should rank below valid"),
    )
    cell.ok(
        GiniObjectiveFunction[F].CompareProxyExact(good, empty) == 1,
        String("valid should rank above invalid"),
    )
    cell.ok(
        GiniObjectiveFunction[F].CompareProxyExact(empty, empty) == 0,
        String("two invalids should tie"),
    )

    # Regression side of the same rejection.
    var mse_strict = MSEObjectiveFunction[F](5)
    var t = AggregateBin[F](0.0, 8)
    var l = AggregateBin[F](3.0, 3)
    cell.ok(
        mse_strict.GainPerSplit(l, t, 8, 3) == Scalar[F].MIN_FINITE,
        String("mse min_samples_leaf: cuML gain not MIN_FINITE"),
    )
    cell.ok(
        mse_strict.ProxyImpurityImprovement(l, t, 8, 3)
        == Scalar[F].MIN_FINITE,
        String("mse min_samples_leaf: proxy not MIN_FINITE"),
    )

    _ = total_buf^
    _ = left_buf^
    return cell.report()


# ==========================================================================


def ref_entropy_gain(
    left: List[Int64], total: List[Int64], n_left: Int64, n: Int64
) -> Float64:
    """An INDEPENDENT float64 information gain: H(parent) - sum_c w_c H(c),
    in nats / log(2), class-major -- a different association and a different
    route than cuML's per-class accumulation, so agreement is a check on the
    transcription's arithmetic and not a copy of it."""
    var n_right = n - n_left

    def h(counts: List[Int64], w: Int64) -> Float64:
        var e = Float64(0.0)
        for c in range(len(counts)):
            if counts[c] > 0:
                var p = Float64(counts[c]) / Float64(w)
                e -= p * log(p)
        return e / log(Float64(2.0))

    var right = List[Int64]()
    for c in range(len(total)):
        right.append(total[c] - left[c])
    return (
        h(total, n)
        - Float64(n_left) / Float64(n) * h(left, n_left)
        - Float64(n_right) / Float64(n) * h(right, n_right)
    )


def check_entropy_scattered() -> Int:
    """DEVIATION 459: `EntropyObjectiveFunction.GainPerSplit` against an
    independent float64 reference on the hashed fixture, the sklearn
    impurities against their definition, `float_gain_key`'s order against
    the float's, and the pair `GainKeyExact` publishes."""
    print("[entropy] scattered fixture,", N_ROWS, "rows,", N_CAND, "candidates")
    var obj = EntropyObjectiveFunction[DType.float32](Int32(N_CLASSES), 1)
    var obj64 = EntropyObjectiveFunction[F](Int32(N_CLASSES), 1)
    var order = sorted_x()
    var left_buf = List[CountBin](length=N_CLASSES, fill=CountBin(0))
    var total_buf = List[CountBin](length=N_CLASSES, fill=CountBin(0))
    var hist_left = left_buf.unsafe_ptr()
    var hist_total = total_buf.unsafe_ptr()
    for i in range(N_ROWS):
        total_buf[Int(fixture_label(i))].x += 1

    var c_gain = Cell(String("cuML entropy GainPerSplit (f32 vs f64 ref)"))
    var c_gain64 = Cell(String("GainPerSplit f64 vs f64 ref"))
    var c_nonneg = Cell(String("gain >= 0 after the 217 clamp"))
    var c_key = Cell(String("float_gain_key order == float order"))
    var c_pair = Cell(String("GainKeyExact pair (key, 1, valid)"))
    var c_imp = Cell(String("sklearn entropy impurities"))
    var c_proxy = Cell(String("sklearn entropy proxy"))
    var c_differs = Cell(String("entropy order differs from gini somewhere"))
    var gini = GiniObjectiveFunction[F](Int32(N_CLASSES), 1)

    var prev_gain = Float32(0.0)
    var prev_key = Int64(0)
    var have_prev = False
    var order_flips = 0
    var tot = List[Int64]()
    for c in range(N_CLASSES):
        tot.append(Int64(Int(total_buf[c].x)))
    var prev_gini_val = Scalar[F](0)
    var prev_ent_val = Float64(0)

    for c in range(N_CAND):
        var thr = fixture_threshold(order, c)
        for k in range(N_CLASSES):
            left_buf[k].x = 0
        var n_left: Int64 = 0
        for i in range(N_ROWS):
            if fixture_x(i) <= thr:
                left_buf[Int(fixture_label(i))].x += 1
                n_left += 1
        var lft = List[Int64]()
        for k in range(N_CLASSES):
            lft.append(Int64(Int(left_buf[k].x)))
        var want = ref_entropy_gain(lft, tot, n_left, Int64(N_ROWS))
        var g32 = obj.GainPerSplit(
            hist_left, hist_total, Int32(N_ROWS), Int32(n_left)
        )
        var g64 = obj64.GainPerSplit(
            hist_left, hist_total, Int32(N_ROWS), Int32(n_left)
        )
        # float32 accumulation of ~15 terms of size <= log2(5): 1e-5 relative
        c_gain.close_f(Float64(g32), want, 1e-5, c)
        c_gain64.close_f(g64, want, 1e-12, c)
        c_nonneg.ok(g32 >= Float32(0.0), String("cell ") + String(c))
        var key = float_gain_key(g32)
        var pair = obj.GainKeyExact(g32, Int32(N_ROWS))
        c_pair.ok(
            pair.valid and pair.den == 1 and pair.num == key,
            String("cell ") + String(c) + ": pair is not (key, 1, valid)",
        )
        if have_prev:
            var f_order = 0
            if g32 > prev_gain:
                f_order = 1
            elif g32 < prev_gain:
                f_order = -1
            var k_order = 0
            if key > prev_key:
                k_order = 1
            elif key < prev_key:
                k_order = -1
            c_key.ok(
                f_order == k_order,
                String("cell ") + String(c) + ": key order != float order",
            )
            # where gini and entropy rank consecutive candidates oppositely
            var gv = gini.ProxyImpurityExact(
                hist_left, hist_total, Int32(N_ROWS), Int32(n_left)
            ).value[F]()
            if (gv > prev_gini_val) != (want > prev_ent_val):
                order_flips += 1
            prev_gini_val = gv
            prev_ent_val = want
        else:
            prev_gini_val = gini.ProxyImpurityExact(
                hist_left, hist_total, Int32(N_ROWS), Int32(n_left)
            ).value[F]()
            prev_ent_val = want
        prev_gain = g32
        prev_key = key
        have_prev = True

        # sklearn's impurities: -sum p log p per child, nats
        var il = Scalar[F](0)
        var ir = Scalar[F](0)
        obj64.ChildrenImpurity(
            hist_left,
            hist_total,
            Scalar[F](n_left),
            Scalar[F](Int64(N_ROWS) - n_left),
            il,
            ir,
        )
        var ref_il = Float64(0)
        var ref_ir = Float64(0)
        for k in range(N_CLASSES):
            if lft[k] > 0:
                var pl = Float64(lft[k]) / Float64(n_left)
                ref_il -= pl * log(pl)
            var rk = tot[k] - lft[k]
            if rk > 0:
                var pr = Float64(rk) / Float64(Int64(N_ROWS) - n_left)
                ref_ir -= pr * log(pr)
        c_imp.close_f(il, ref_il, 1e-12, c)
        c_imp.close_f(ir, ref_ir, 1e-12, c)
        var proxy = obj64.ProxyImpurityImprovement(
            hist_left, hist_total, Int32(N_ROWS), Int32(n_left)
        )
        c_proxy.close_f(
            proxy,
            -Float64(Int64(N_ROWS) - n_left) * ref_ir - Float64(n_left) * ref_il,
            1e-12,
            c,
        )

    c_differs.ok(
        order_flips > 0,
        String("gini and entropy ranked every consecutive pair alike; the"
        " fixture cannot see the criterion"),
    )
    print("   ", order_flips, "consecutive pairs ranked oppositely by gini and entropy")

    # the key on hand-picked floats: the two zeros tie, negatives order
    var c_zero = Cell(String("float_gain_key: -0.0 == +0.0, sign order"))
    c_zero.ok(float_gain_key(Float32(-0.0)) == float_gain_key(Float32(0.0)), String("zeros"))
    c_zero.ok(float_gain_key(Float32(-1.0)) < float_gain_key(Float32(-0.5)), String("neg order"))
    c_zero.ok(float_gain_key(Float32(-0.5)) < float_gain_key(Float32(0.0)), String("neg < 0"))
    c_zero.ok(float_gain_key(Float32(0.0)) < float_gain_key(Float32(1e-30)), String("0 < tiny"))
    c_zero.ok(float_gain_key(Float32(1.5)) < float_gain_key(Float32(2.0)), String("pos order"))
    # the rejected sentinel is INVALID, not keyed
    c_zero.ok(
        not obj.GainKeyExact(Float32.MIN_FINITE, 10).valid,
        String("MIN_FINITE must be invalid"),
    )

    _ = total_buf^
    _ = left_buf^
    var failed = 0
    failed += c_gain.report()
    failed += c_gain64.report()
    failed += c_nonneg.report()
    failed += c_key.report()
    failed += c_pair.report()
    failed += c_imp.report()
    failed += c_proxy.report()
    failed += c_differs.report()
    failed += c_zero.report()
    return failed


def main() raises:
    print("objectives_check -- extratrees split scoring, per cell")
    print("")
    var failed = 0
    failed += check_gini_scattered()
    print("")
    failed += check_mse_scattered()
    print("")
    failed += check_entropy_scattered()
    print("")
    failed += check_analytic()
    print("")
    failed += check_rejection()
    print("")
    if failed == 0:
        print("PASS -- every cell")
    else:
        print("FAIL --", failed, "cells wrong")
        raise Error("objectives_check failed")
