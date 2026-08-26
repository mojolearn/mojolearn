"""The check for the exact host oracle, `host_splitter.mojo`.

    Run:  pixi run mojo run -I . extratrees/mojo_only/host_splitter_check.mojo

WHAT AUTHORITY EACH SECTION USES, because "it agrees with itself" is not one
----------------------------------------------------------------------------
1. **Analytic.** `fixtures.analytic_*` build datasets with an EMPTY GAP, so
   the partition -- and therefore the counts, the sums and the score -- are
   the SAME for every threshold inside the gap. The draw is random, so the
   check watches which side of the gap it lands on and asserts the CLOSED
   FORM whenever it lands inside; outside, it falls back to the independent
   tally. The all-constant fixture's closed form is NO SPLIT.
2. **Hashed, per cell.** Every per-candidate quantity the oracle produces is
   compared against a tally written in a deliberately different style: a
   plain loop over a materialised `List[Int]` of node rows, reading the
   fixture's ROW-MAJOR accessor (the oracle reads cuML's COLUMN-MAJOR
   `Dataset`), accumulating in `Float64` with no objective object involved.
   Class counts are compared PER CLASS, not as a total -- `PORTING_RULES.md`
   rule 8, earned here: a uniform fixture once reported 0 wrong of 512 on a
   kernel a hashed fixture showed to be 490 wrong of 512.
3. **Order independence.** The same node, the same key, the columns supplied
   in a different order, must give a BITWISE identical `Split` and identical
   per-candidate records. This is DEVIATION 130's entire point and the
   property a device implementation breaks first.
4. **The Float32-vs-exact comparator**, DEVIATION 145, gets a fixture built
   for it: two two-valued features whose partitions are fixed regardless of
   what threshold is drawn, chosen by search so that their cuML gains are
   BIT-IDENTICAL as `Float32` while their exact rational proxies differ. On
   that fixture `Split.update` and `CompareProxyExact` disagree, so the
   sabotage that swaps one for the other is visible.

SABOTAGE, one per mechanism (rule 8). Each was applied to
`host_splitter.mojo`, run, seen RED, and reverted; the results are in the
lane report. The mechanisms: the constant skip, the `min_samples_leaf`
rejection, the tie-break direction, the exact-vs-float comparator, and the
left/right accumulator assignment.
"""

from std.testing import assert_equal, assert_true

from extratrees.mojo_only.fixtures import (
    AnalyticFixture,
    Dataset as FixtureDataset,
    all_shapes,
    analytic_all_constant,
    analytic_regression_step,
    analytic_separable_gap,
    analytic_tie_pair,
    hashed_classification,
    hashed_regression,
    shape_name,
    shaped_dataset,
)
from extratrees.mojo_only.host_splitter import (
    CandidateRecord,
    HostSplitResult,
    node_split_random_gini,
    node_split_random_mse,
)
from extratrees.mojo_only.pcg_rng import key_for
from extratrees.ported.decisiontree.batched_levelalgo.dataset import Dataset
from extratrees.ported.decisiontree.batched_levelalgo.objectives import (
    CountBin,
    GiniObjectiveFunction,
    GiniProxyExact,
    MSEObjectiveFunction,
)
from extratrees.ported.decisiontree.batched_levelalgo.split import (
    ET_TIE_BREAK_KEYED,
    Split,
)
from extratrees.ported.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
)
from extratrees.ported.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    FeatureRange,
    draw_threshold,
    node_feature_is_constant,
)


comptime F = DType.float64
"""The oracle's accumulator type. `float64` here on purpose: DEVIATION 135 is
OPEN about what the DEVICE accumulates in, and an oracle that pre-committed to
`float32` would settle it by accident."""


comptime _TIE_SALT: UInt32 = 0x7B31C853
"""DEVIATION 463's `SPLIT_TIE_SALT`, spelled locally so this file's tie
re-derivation shares no line with the shipping `split_tie_rank`."""


def _tie_rank_ref(tree_id: UInt32, node_id: UInt32, colid: UInt32) -> UInt32:
    """DEVIATION 463's rank, INDEPENDENTLY transcribed: fnv1a32 from the
    basis over (salt, tree, node, colid), then murmur3's fmix32. The oracle
    computes the same value through `split_tie_salt_for` + `split_tie_rank`;
    two expressions of one arithmetic, the way `split_reduce_check` holds
    `oracle_rank` against the kernel."""
    var h = UInt32(2166136261)
    var words = [_TIE_SALT, tree_id, node_id, colid]
    for w in range(len(words)):
        for b in range(4):
            h = (h ^ ((words[w] >> UInt32(8 * b)) & 0xFF)) * UInt32(16777619)
    h = (h ^ (h >> 16)) * 0x85EBCA6B
    h = (h ^ (h >> 13)) * 0xC2B2AE35
    return h ^ (h >> 16)


# ==========================================================================
# Plumbing: a fixture (row major) behind cuML's Dataset view (column major)
# ==========================================================================


struct Box(Movable):
    """Owns the buffers cuML's `Dataset` points into.

    `Dataset`'s pointers are `MutUntrackedOrigin`, so the compiler tracks no
    relationship between them and the `List`s they address; a bare local list
    is freed after its last SYNTACTIC use and the view reads freed memory.
    `range_draw_check.mojo` records that happening for real. Holding the
    lists in a struct that outlives the view removes the trap instead of
    documenting it.
    """

    var flat: List[Float32]
    var labels: List[Float32]
    var row_ids: List[Int32]
    var n_rows: Int
    var n_cols: Int
    var n_classes: Int

    def __init__(
        out self,
        fixture: FixtureDataset,
        use_class_labels: Bool,
        shuffle: Bool,
    ):
        self.n_rows = fixture.n_rows
        self.n_cols = fixture.n_cols
        self.n_classes = fixture.n_classes
        self.flat = List[Float32](
            length=fixture.n_rows * fixture.n_cols, fill=Float32(0.0)
        )
        for r in range(fixture.n_rows):
            for c in range(fixture.n_cols):
                self.flat[c * fixture.n_rows + r] = fixture.value(r, c)
        self.labels = List[Float32]()
        for r in range(fixture.n_rows):
            if use_class_labels:
                self.labels.append(Float32(Int(fixture.label[r])))
            else:
                self.labels.append(fixture.y[r])
        self.row_ids = List[Int32]()
        for r in range(fixture.n_rows):
            self.row_ids.append(Int32(r))
        if shuffle:
            # A fixed permutation, so a row-index confusion cannot hide
            # behind the identity. Not the RNG under test.
            for i in range(fixture.n_rows - 1, 0, -1):
                var h = UInt32(i) * 2654435761
                h ^= h >> 15
                var j = Int(h % UInt32(i + 1))
                var t = self.row_ids[i]
                self.row_ids[i] = self.row_ids[j]
                self.row_ids[j] = t

    def view(mut self) -> Dataset:
        return Dataset(
            rebind[MutPointer[Float32, MutUntrackedOrigin]](
                self.flat.unsafe_ptr()
            ),
            rebind[MutPointer[Float32, MutUntrackedOrigin]](
                self.labels.unsafe_ptr()
            ),
            Int32(self.n_rows),
            Int32(self.n_cols),
            Int32(self.n_rows),
            Int32(self.n_cols),
            rebind[MutPointer[Int32, MutUntrackedOrigin]](
                self.row_ids.unsafe_ptr()
            ),
            Int32(self.n_classes),
        )

    def node_rows(self, begin: Int, count: Int) -> List[Int]:
        """The node's rows, materialised as plain `Int`s. The tally below
        walks THIS, not `row_ids`, so an off-by-one in the oracle's pointer
        arithmetic has somewhere to show."""
        var out = List[Int]()
        for p in range(begin, begin + count):
            out.append(Int(self.row_ids[p]))
        return out^


# ==========================================================================
# The independent tally. Different data path, different layout, different
# accumulator, no objective object.
# ==========================================================================


@fieldwise_init
struct Tally(Copyable, Movable):
    var n_left: Int
    var n_right: Int
    var cl: List[Int]
    var cr: List[Int]
    var sum_left: Float64
    var sum_right: Float64
    var sq_left: Float64
    var sq_right: Float64

    def gini_proxy(self) -> Float64:
        """`_criterion.pyx:147-163` over `:647-687`: `-nR*gini_R - nL*gini_L`.
        Gini does NOT override the base proxy, so this is the shape, not the
        sum-of-squares one a port usually assumes."""
        var gl = Float64(0.0)
        var gr = Float64(0.0)
        if self.n_left > 0:
            var s = Float64(0.0)
            for i in range(len(self.cl)):
                s += Float64(self.cl[i]) * Float64(self.cl[i])
            gl = 1.0 - s / (Float64(self.n_left) * Float64(self.n_left))
        if self.n_right > 0:
            var s = Float64(0.0)
            for i in range(len(self.cr)):
                s += Float64(self.cr[i]) * Float64(self.cr[i])
            gr = 1.0 - s / (Float64(self.n_right) * Float64(self.n_right))
        return -Float64(self.n_right) * gr - Float64(self.n_left) * gl

    def mse_proxy(self) -> Float64:
        """`_criterion.pyx:944-973`: `sum_L^2/n_L + sum_R^2/n_R`."""
        var out = Float64(0.0)
        if self.n_left > 0:
            out += self.sum_left * self.sum_left / Float64(self.n_left)
        if self.n_right > 0:
            out += self.sum_right * self.sum_right / Float64(self.n_right)
        return out


def tally_for(
    fixture: FixtureDataset,
    rows: List[Int],
    col: Int,
    threshold: Float32,
    n_classes: Int,
) -> Tally:
    """`_partitioner.pyx:236-238`: `value <= threshold` goes LEFT."""
    var cl = List[Int](length=n_classes, fill=0)
    var cr = List[Int](length=n_classes, fill=0)
    var nl = 0
    var nr = 0
    var sl = Float64(0.0)
    var sr = Float64(0.0)
    var ql = Float64(0.0)
    var qr = Float64(0.0)
    for i in range(len(rows)):
        var r = rows[i]
        var v = fixture.value(r, col)
        var yv = Float64(fixture.y[r])
        var k = Int(fixture.label[r])
        if v <= threshold:
            nl += 1
            sl += yv
            ql += yv * yv
            if n_classes > 0:
                cl[k] += 1
        else:
            nr += 1
            sr += yv
            qr += yv * yv
            if n_classes > 0:
                cr[k] += 1
    return Tally(nl, nr, cl^, cr^, sl, sr, ql, qr)


def extremes_for(
    fixture: FixtureDataset, rows: List[Int], col: Int
) -> FeatureRange:
    """`_partitioner.pyx:150-163`, written as two independent `if`s rather
    than their `elif` chain. Equivalent for NaN-free input, which is the only
    input this port accepts (DEVIATION 136), and a different shape of loop --
    which is the point."""
    var lo = fixture.value(rows[0], col)
    var hi = lo
    for i in range(1, len(rows)):
        var v = fixture.value(rows[i], col)
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    return FeatureRange(lo, hi, Int32(0))


def close_rel(got: Float64, expected: Float64, tol: Float64) -> Bool:
    var scale = abs(expected)
    if scale < 1.0:
        scale = 1.0
    return abs(got - expected) <= tol * scale


# ==========================================================================
# 1. Hashed classification, every per-candidate quantity, per cell
# ==========================================================================


def check_hashed_classification() raises -> Int:
    print("[hashed/gini] every per-candidate quantity against a plain tally")
    var cells = 0
    var n_rows = 4096
    var n_cols = 8
    var n_classes = 3
    var fixture = hashed_classification(0xC0FFEE, n_rows, n_cols, n_classes)
    var box = Box(fixture, True, True)
    var dataset = box.view()
    var obj = GiniObjectiveFunction[F](Int32(n_classes), 1)

    var colids = List[Int32]()
    for c in range(n_cols):
        colids.append(Int32(c))

    var nodes = [
        (0, n_rows),
        (0, 1000),
        (13, 7),
        (13, 999),
        (2048, 2048),
        (4095, 1),
    ]
    var seed = UInt64(0x5EED)
    for ni in range(len(nodes)):
        var begin = nodes[ni][0]
        var count = nodes[ni][1]
        var node_id = Int32(ni * 7 + 3)
        var item = NodeWorkItem(
            node_id, 0, InstanceRange(Int32(begin), Int32(count))
        )
        var rows = box.node_rows(begin, count)
        var res = node_split_random_gini[F](
            dataset, item, colids, obj, seed, Int32(11)
        )

        # DEVIATION 216's companion: a PURE node (one class holds every row
        # -- a one-row node trivially so) leafs BEFORE any candidate is
        # visited, sklearn `_tree.pyx:240`, so it reports n_visited == 0.
        var node_pure = False
        for kcls in range(len(res.hist_total)):
            if Int(res.hist_total[kcls].x) == count and count > 0:
                node_pure = True
        if node_pure:
            assert_equal(
                Int(res.n_visited), 0, "a pure node visits no candidates"
            )
            assert_true(
                not res.found, "and cannot have found a split"
            )
        else:
            assert_equal(Int(res.n_visited), n_cols, "n_visited == len(colids)")
        assert_equal(Int(res.node_rows), count, "node_rows == count")
        cells += 2

        # Node totals, per class.
        var totals = List[Int](length=n_classes, fill=0)
        for i in range(len(rows)):
            totals[Int(fixture.label[rows[i]])] += 1
        for k in range(n_classes):
            assert_equal(
                Int(res.hist_total[k].x),
                totals[k],
                "node class total mismatch",
            )
            cells += 1
        if node_pure:
            # A pure node's early leaf (216's companion) carries no
            # candidate records; everything below reads them.
            continue

        var best_ref = -1
        var best_num = Int64(0)
        var best_den = Int64(0)
        for ci in range(n_cols):
            ref rec = res.candidates[ci]
            assert_equal(Int(rec.colid), ci, "records follow the supplied order")
            cells += 1

            var ext = extremes_for(fixture, rows, ci)
            assert_equal(
                rec.extent.min_value.to_bits(),
                ext.min_value.to_bits(),
                "min",
            )
            assert_equal(
                rec.extent.max_value.to_bits(),
                ext.max_value.to_bits(),
                "max",
            )
            assert_equal(Int(rec.extent.n_missing), 0, "no NaNs in a fixture")
            cells += 3

            var want_const = node_feature_is_constant(ext, Int32(count))
            assert_equal(rec.is_constant, want_const, "constant verdict")
            cells += 1
            if want_const:
                assert_true(
                    not rec.drew_threshold and not rec.scored,
                    "a constant feature is skipped before the draw",
                )
                cells += 1
                continue

            # The key, rebuilt from THIS check's own variables: the oracle
            # must key on (seed, tree, node.idx, colid) and nothing else.
            var want_t = draw_threshold(
                key_for(seed, UInt32(11), UInt32(Int(node_id)), UInt32(ci)),
                ext,
            )
            assert_equal(
                rec.threshold.to_bits(), want_t.to_bits(), "threshold bits"
            )
            cells += 1

            var t = tally_for(fixture, rows, ci, rec.threshold, n_classes)
            assert_equal(Int(rec.n_left), t.n_left, "n_left")
            assert_equal(Int(rec.n_right), t.n_right, "n_right")
            cells += 2
            for k in range(n_classes):
                assert_equal(
                    Int(rec.hist_left[k].x),
                    t.cl[k],
                    "left class count, per class",
                )
                cells += 1

            var reject = t.n_left < 1 or t.n_right < 1
            assert_equal(
                rec.rejected_min_samples_leaf,
                reject,
                "min_samples_leaf rejection",
            )
            assert_equal(rec.scored, not reject, "scored")
            cells += 2
            if reject:
                continue

            assert_true(
                close_rel(rec.proxy_float, t.gini_proxy(), 1e-12),
                "float proxy vs the tally: "
                + String(rec.proxy_float)
                + " vs "
                + String(t.gini_proxy()),
            )
            cells += 1

            # The exact rational, rebuilt from the tally's own counts.
            var sq_l = Int64(0)
            var sq_r = Int64(0)
            for k in range(n_classes):
                sq_l += Int64(t.cl[k]) * Int64(t.cl[k])
                sq_r += Int64(t.cr[k]) * Int64(t.cr[k])
            var num = sq_l * Int64(t.n_right) + sq_r * Int64(t.n_left)
            var den = Int64(t.n_left) * Int64(t.n_right)
            assert_true(rec.proxy_exact.valid, "scored implies a valid proxy")
            assert_equal(rec.proxy_exact.num, num, "exact numerator")
            assert_equal(rec.proxy_exact.den, den, "exact denominator")
            cells += 3

            # The argmax, computed here by cross-multiplication, independent
            # of the oracle's reduction.
            if best_ref < 0:
                best_ref = ci
                best_num = num
                best_den = den
            else:
                var lhs = Int128(num) * Int128(best_den)
                var rhs = Int128(best_num) * Int128(den)
                var wins = lhs > rhs
                if lhs == rhs:
                    # An EXACT rational tie between two features. This branch
                    # used to be dead by luck: the check kept the FIRST tied
                    # feature (strict `>` only) and no draw before DEVIATION
                    # 465 produced a tie on this fixture. 465's salted keys
                    # re-rolled every threshold, the hashed/gini fixture then
                    # tied features 0 and 4 exactly at one node, and the
                    # first-wins shortcut disagreed with the SPEC -- which is
                    # DEVIATION 463's keyed rank: the GREATER rank over
                    # (tree=11, this node, colid) wins, re-derived here by
                    # `_tie_rank_ref`, not read back from the oracle. On a
                    # rank COLLISION (a 2^-32 event between distinct colids)
                    # the spec falls to the colid arm, where the ascending
                    # `ci` scan's newcomer is always the greater -- hence
                    # `>=`. Under the `MOJOLEARN_ET_TIE_MAX_COLID` build the
                    # higher colid wins outright, again the newcomer.
                    comptime if ET_TIE_BREAK_KEYED:
                        wins = _tie_rank_ref(
                            UInt32(11), UInt32(Int(node_id)), UInt32(ci)
                        ) >= _tie_rank_ref(
                            UInt32(11), UInt32(Int(node_id)), UInt32(best_ref)
                        )
                    else:
                        wins = True
                if wins:
                    best_ref = ci
                    best_num = num
                    best_den = den

        assert_equal(res.found, best_ref >= 0, "found")
        cells += 1
        if best_ref >= 0:
            assert_equal(
                Int(res.split.colid), best_ref, "the chosen feature"
            )
            assert_equal(
                Int(res.split.n_left),
                Int(res.candidates[best_ref].n_left),
                "the chosen n_left",
            )
            assert_equal(
                res.split.quesval.to_bits(),
                res.candidates[best_ref].threshold.to_bits(),
                "the chosen threshold",
            )
            cells += 3

    _ = box^
    print("   ", cells, "cells over", len(nodes), "nodes x", n_cols, "columns")
    return cells


# ==========================================================================
# 2. Hashed regression
# ==========================================================================


def check_hashed_regression() raises -> Int:
    print("[hashed/mse] every per-candidate quantity against a plain tally")
    var cells = 0
    var n_rows = 2048
    var n_cols = 6
    var fixture = hashed_regression(0xBEEF, n_rows, n_cols)
    var box = Box(fixture, False, True)
    var dataset = box.view()
    var obj = MSEObjectiveFunction[F](1)

    var colids = List[Int32]()
    for c in range(n_cols):
        colids.append(Int32(c))

    var nodes = [(0, n_rows), (5, 3), (5, 501), (1024, 1024)]
    var seed = UInt64(0xD00D)
    for ni in range(len(nodes)):
        var begin = nodes[ni][0]
        var count = nodes[ni][1]
        var node_id = Int32(ni * 13 + 1)
        var item = NodeWorkItem(
            node_id, 0, InstanceRange(Int32(begin), Int32(count))
        )
        var rows = box.node_rows(begin, count)
        var res = node_split_random_mse[F](
            dataset, item, colids, obj, seed, Int32(4)
        )

        var st = Float64(0.0)
        var sq = Float64(0.0)
        for i in range(len(rows)):
            var yv = Float64(fixture.y[rows[i]])
            st += yv
            sq += yv * yv
        assert_true(
            close_rel(res.agg_total.label_sum, st, 1e-12), "node label sum"
        )
        assert_equal(Int(res.agg_total.count), count, "node count")
        assert_true(close_rel(res.sq_sum_total, sq, 1e-12), "node sq sum")
        cells += 3

        var best_ref = -1
        var best_proxy = Float64(0.0)
        for ci in range(n_cols):
            ref rec = res.candidates[ci]
            var ext = extremes_for(fixture, rows, ci)
            assert_equal(
                rec.extent.min_value.to_bits(), ext.min_value.to_bits(), "min"
            )
            assert_equal(
                rec.extent.max_value.to_bits(), ext.max_value.to_bits(), "max"
            )
            cells += 2

            var want_const = node_feature_is_constant(ext, Int32(count))
            assert_equal(rec.is_constant, want_const, "constant verdict")
            cells += 1
            if want_const:
                continue

            var want_t = draw_threshold(
                key_for(seed, UInt32(4), UInt32(Int(node_id)), UInt32(ci)), ext
            )
            assert_equal(
                rec.threshold.to_bits(), want_t.to_bits(), "threshold bits"
            )
            cells += 1

            var t = tally_for(fixture, rows, ci, rec.threshold, 0)
            assert_equal(Int(rec.n_left), t.n_left, "n_left")
            assert_equal(Int(rec.n_right), t.n_right, "n_right")
            assert_equal(Int(rec.agg_left.count), t.n_left, "agg count")
            assert_true(
                close_rel(rec.agg_left.label_sum, t.sum_left, 1e-12),
                "left label sum",
            )
            assert_true(
                close_rel(rec.sq_sum_left, t.sq_left, 1e-12), "left sq sum"
            )
            cells += 5

            var reject = t.n_left < 1 or t.n_right < 1
            assert_equal(
                rec.rejected_min_samples_leaf, reject, "min_samples_leaf"
            )
            cells += 1
            if reject:
                continue

            assert_true(
                close_rel(rec.proxy_float, t.mse_proxy(), 1e-12),
                "mse proxy vs the tally",
            )
            cells += 1

            var take_mse = best_ref < 0 or t.mse_proxy() > best_proxy
            if best_ref >= 0 and t.mse_proxy() == best_proxy:
                # The same tie rule as the gini section above (DEVIATION
                # 463), tree id 4 here. A float MSE-proxy tie on hashed data
                # is a measure-zero event, but the check's independent argmax
                # must still carry the spec's tie order, not first-wins.
                comptime if ET_TIE_BREAK_KEYED:
                    # `>=` for the same rank-collision fallback as the gini
                    # section: the ascending scan's newcomer holds the
                    # greater colid.
                    take_mse = _tie_rank_ref(
                        UInt32(4), UInt32(Int(node_id)), UInt32(ci)
                    ) >= _tie_rank_ref(
                        UInt32(4), UInt32(Int(node_id)), UInt32(best_ref)
                    )
                else:
                    take_mse = True
            if take_mse:
                best_ref = ci
                best_proxy = t.mse_proxy()

        assert_equal(res.found, best_ref >= 0, "found")
        cells += 1
        if best_ref >= 0:
            assert_equal(Int(res.split.colid), best_ref, "the chosen feature")
            cells += 1

    _ = box^
    print("   ", cells, "cells over", len(nodes), "nodes x", n_cols, "columns")
    return cells


# ==========================================================================
# 3. Analytic: the closed form, whenever the draw lands in the gap
# ==========================================================================


def check_analytic_classification() raises -> Int:
    print("[analytic/gini] separable_gap: closed form inside the gap")
    var cells = 0
    var fix = analytic_separable_gap(0x1234)
    var box = Box(fix.data, True, False)
    var dataset = box.view()
    var obj = GiniObjectiveFunction[F](Int32(fix.data.n_classes), 1)
    var colids = List[Int32]()
    colids.append(Int32(fix.feature))
    colids.append(Int32(fix.noise_feature))

    var n = fix.data.n_rows
    var rows = box.node_rows(0, n)
    var item = NodeWorkItem(0, 0, InstanceRange(0, Int32(n)))

    var inside = 0
    var outside = 0
    for tree in range(64):
        var res = node_split_random_gini[F](
            dataset, item, colids, obj, UInt64(0xA5A5), Int32(tree)
        )
        ref rec = res.candidates[0]
        assert_true(rec.scored, "the gap feature is never constant")
        cells += 1
        var t = Float64(rec.threshold)
        if t >= fix.t_lo and t <= fix.t_hi:
            inside += 1
            # THE CLOSED FORM. Invariant for every threshold in [t_lo, t_hi].
            assert_equal(Int(rec.n_left), fix.n_left, "closed-form n_left")
            assert_equal(Int(rec.n_right), fix.n_right, "closed-form n_right")
            for k in range(fix.data.n_classes):
                assert_equal(
                    Int(rec.hist_left[k].x),
                    fix.left_class_counts[k],
                    "closed-form left class count",
                )
                assert_equal(
                    Int(res.hist_total[k].x) - Int(rec.hist_left[k].x),
                    fix.right_class_counts[k],
                    "closed-form right class count",
                )
                cells += 2
            assert_equal(
                rec.proxy_float, Float64(fix.proxy), "closed-form proxy"
            )
            # proxy = num/den - n, and a perfectly pure split has proxy 0,
            # so the exact rational must satisfy num == den * n EXACTLY.
            assert_equal(
                rec.proxy_exact.num,
                rec.proxy_exact.den * Int64(fix.data.n_rows),
                "exact proxy of a pure split is exactly zero: num == den*n",
            )
            cells += 4
            # A perfectly pure split is the maximum of the proxy, so the gap
            # feature must win outright over the noise feature.
            assert_equal(
                Int(res.split.colid), fix.feature, "the gap feature wins"
            )
            assert_true(
                res.candidates[1].proxy_float < Float64(fix.proxy),
                "the noise feature scores strictly worse than pure",
            )
            cells += 2
        else:
            outside += 1
            # Outside the gap the closed form does not apply; the independent
            # tally does, and it must still match cell for cell.
            var tl = tally_for(fix.data, rows, fix.feature, rec.threshold, 2)
            assert_equal(Int(rec.n_left), tl.n_left, "outside-gap n_left")
            for k in range(2):
                assert_equal(
                    Int(rec.hist_left[k].x), tl.cl[k], "outside-gap counts"
                )
                cells += 1
            cells += 1
    assert_true(inside > 0, "no draw ever landed in the gap")
    assert_true(outside > 0, "every draw landed in the gap: no control")
    print("   ", cells, "cells;", inside, "draws inside the gap,", outside, "outside")
    _ = box^
    return cells


def check_analytic_regression() raises -> Int:
    print("[analytic/mse] regression_step: closed form inside the gap")
    var cells = 0
    var fix = analytic_regression_step(0x77)
    var box = Box(fix.data, False, False)
    var dataset = box.view()
    var obj = MSEObjectiveFunction[F](1)
    var colids = List[Int32]()
    colids.append(Int32(fix.feature))
    colids.append(Int32(fix.noise_feature))

    var n = fix.data.n_rows
    var rows = box.node_rows(0, n)
    var item = NodeWorkItem(0, 0, InstanceRange(0, Int32(n)))

    var inside = 0
    var outside = 0
    for tree in range(64):
        var res = node_split_random_mse[F](
            dataset, item, colids, obj, UInt64(0x1010), Int32(tree)
        )
        ref rec = res.candidates[0]
        var t = Float64(rec.threshold)
        if t >= fix.t_lo and t <= fix.t_hi:
            inside += 1
            assert_equal(Int(rec.n_left), fix.n_left, "closed-form n_left")
            assert_true(
                close_rel(rec.agg_left.label_sum, fix.left_sum, 1e-12),
                "closed-form left sum",
            )
            assert_true(
                close_rel(rec.sq_sum_left, fix.left_sum_sq, 1e-12),
                "closed-form left sum of squares",
            )
            assert_true(
                close_rel(rec.proxy_float, fix.proxy, 1e-12),
                "closed-form proxy (1664.0): " + String(rec.proxy_float),
            )
            assert_equal(
                Int(res.split.colid), fix.feature, "the step feature wins"
            )
            # Both children are pure, so their impurities are exactly zero.
            assert_true(
                close_rel(res.impurity_left, fix.impurity_left, 1e-9)
                and close_rel(res.impurity_right, fix.impurity_right, 1e-9),
                "closed-form child impurities",
            )
            cells += 6
        else:
            outside += 1
            var tl = tally_for(fix.data, rows, fix.feature, rec.threshold, 0)
            assert_equal(Int(rec.n_left), tl.n_left, "outside-gap n_left")
            assert_true(
                close_rel(rec.agg_left.label_sum, tl.sum_left, 1e-12),
                "outside-gap left sum",
            )
            cells += 2
    assert_true(inside > 0, "no draw ever landed in the gap")
    assert_true(outside > 0, "every draw landed in the gap: no control")
    print("   ", cells, "cells;", inside, "inside the gap,", outside, "outside")
    _ = box^
    return cells


# ==========================================================================
# 4. All constant: the correct answer is NO SPLIT
# ==========================================================================


def check_all_constant() raises -> Int:
    print("[analytic] all_constant: no split, and every column skipped")
    var cells = 0
    var fix = analytic_all_constant()
    var box = Box(fix.data, True, False)
    var dataset = box.view()
    var obj = GiniObjectiveFunction[F](Int32(fix.data.n_classes), 1)
    var colids = List[Int32]()
    for c in range(fix.data.n_cols):
        colids.append(Int32(c))

    var item = NodeWorkItem(0, 0, InstanceRange(0, Int32(fix.data.n_rows)))
    for tree in range(8):
        var res = node_split_random_gini[F](
            dataset, item, colids, obj, UInt64(3), Int32(tree)
        )
        assert_true(not res.found, "an all-constant node has no valid split")
        assert_equal(Int(res.split.colid), -1, "Split stays at its default")
        assert_true(not res.split.is_valid(), "split.cuh:145 agrees")
        assert_equal(
            Int(res.n_constant), fix.data.n_cols, "every column is constant"
        )
        assert_equal(res.best_index, -1, "no winner")
        cells += 5
        for c in range(fix.data.n_cols):
            assert_true(
                res.candidates[c].is_constant,
                "column " + String(c) + " must be reported constant",
            )
            assert_true(
                not res.candidates[c].drew_threshold,
                "and must never reach the draw",
            )
            cells += 2
    # The labels really are mixed, so "no split" is not "already pure".
    assert_true(
        Int(box.n_classes) == 2, "the fixture is two-class",
    )
    cells += 1
    print("   ", cells, "cells")
    _ = box^
    return cells


# ==========================================================================
# 5. SOME constant: skip exactly those, split on one of the rest
# ==========================================================================


def check_mixed_constants() raises -> Int:
    print("[shaped] some columns constant: skip exactly those")
    var cells = 0
    var n_rows = 512
    var shapes = all_shapes()
    var fixture = shaped_dataset(0xBEAD, n_rows, shapes)
    var box = Box(fixture, True, True)
    var dataset = box.view()
    var obj = GiniObjectiveFunction[F](Int32(fixture.n_classes), 1)
    var colids = List[Int32]()
    for c in range(fixture.n_cols):
        colids.append(Int32(c))

    var rows = box.node_rows(0, n_rows)
    var item = NodeWorkItem(9, 0, InstanceRange(0, Int32(n_rows)))
    var res = node_split_random_gini[F](
        dataset, item, colids, obj, UInt64(0xFACE), Int32(2)
    )

    var n_const_ref = 0
    for c in range(fixture.n_cols):
        var ext = extremes_for(fixture, rows, c)
        var want = node_feature_is_constant(ext, Int32(n_rows))
        assert_equal(
            res.candidates[c].is_constant,
            want,
            "shape " + shape_name(shapes[c]) + " constant verdict",
        )
        cells += 1
        if want:
            n_const_ref += 1
            assert_true(
                not res.candidates[c].scored,
                "a constant column is never scored",
            )
            cells += 1
    assert_equal(Int(res.n_constant), n_const_ref, "n_constant")
    assert_true(n_const_ref > 0, "the shaped fixture must contain constants")
    assert_true(
        n_const_ref < fixture.n_cols, "and must contain non-constants too"
    )
    assert_true(res.found, "a split must be found among the rest")
    assert_true(
        not res.candidates[res.best_index].is_constant,
        "the winner is not a constant column",
    )
    cells += 5
    print("   ", cells, "cells;", n_const_ref, "of", fixture.n_cols, "constant")
    _ = box^
    return cells


# ==========================================================================
# 6. The tie-break: two bit-identical features, both drawn into the gap.
#    The expected winner is DEVIATION 463's keyed rank (the max-colid rule
#    only under the MOJOLEARN_ET_TIE_MAX_COLID build); this assertion read
#    `colid == 1` verbatim until the 2026-08-26 check round.
# ==========================================================================


def check_tie_break() raises -> Int:
    print("[tie] duplicate features, exact tie, the keyed rank picks")
    var cells = 0
    var wins0 = 0
    var wins1 = 0
    var fix = analytic_tie_pair(0x99)
    var box = Box(fix.data, True, False)
    var dataset = box.view()
    var obj = GiniObjectiveFunction[F](Int32(fix.data.n_classes), 1)
    var colids = List[Int32]()
    colids.append(0)
    colids.append(1)
    colids.append(2)

    var item = NodeWorkItem(0, 0, InstanceRange(0, Int32(fix.data.n_rows)))
    var ties = 0
    for tree in range(96):
        var res = node_split_random_gini[F](
            dataset, item, colids, obj, UInt64(0x2222), Int32(tree)
        )
        ref a = res.candidates[0]
        ref b = res.candidates[1]
        var ta = Float64(a.threshold)
        var tb = Float64(b.threshold)
        if (
            ta >= fix.t_lo
            and ta <= fix.t_hi
            and tb >= fix.t_lo
            and tb <= fix.t_hi
        ):
            ties += 1
            # Columns 0 and 1 are bit-identical, and both thresholds are in
            # the empty gap, so the two partitions are IDENTICAL and the
            # exact proxies tie exactly.
            assert_equal(a.proxy_exact.num, b.proxy_exact.num, "tie: num")
            assert_equal(a.proxy_exact.den, b.proxy_exact.den, "tie: den")
            assert_equal(
                GiniObjectiveFunction[F].CompareProxyExact(
                    a.proxy_exact, b.proxy_exact
                ),
                0,
                "the exact comparator must call it a tie",
            )
            # DEVIATION 463: on an exact tie the GREATER keyed rank over
            # (tree, node 0, colid) wins -- re-derived here by
            # `_tie_rank_ref`, NOT read back from the oracle. The winner
            # flips with the tree id, which is the uniform-among-ties
            # property the rule exists for, made visible; `wins0`/`wins1`
            # report the split. `split.cuh:80-89`'s max-colid arm (this
            # assertion's pre-463 expectation, colid 1 always) is the
            # expectation only under the `MOJOLEARN_ET_TIE_MAX_COLID`
            # build. A rank collision (a 2^-32 event) falls to the colid
            # arm, where 1 wins -- hence `>=` on rank 1's side.
            var want_col = 1
            comptime if ET_TIE_BREAK_KEYED:
                var r0 = _tie_rank_ref(UInt32(tree), UInt32(0), UInt32(0))
                var r1 = _tie_rank_ref(UInt32(tree), UInt32(0), UInt32(1))
                if r0 > r1:
                    want_col = 0
            assert_equal(
                Int(res.split.colid),
                want_col,
                "the tie's winner, re-derived from the keyed rank",
            )
            if want_col == 1:
                wins1 += 1
            else:
                wins0 += 1
            var want_thr = b.threshold if want_col == 1 else a.threshold
            assert_equal(
                res.split.quesval.to_bits(),
                want_thr.to_bits(),
                "and carries ITS threshold, not the other's",
            )
            cells += 5
    assert_true(ties >= 8, "not enough double-in-gap draws to test the tie")
    print(
        "    keyed tie winners across the double-in-gap trees: colid 0 won",
        wins0,
        ", colid 1 won",
        wins1,
    )

    # The `quesval` arm of the total order (`split.cuh:86-88`) can only fire
    # on EQUAL colids, which a sampler never produces. Supplying the same
    # column twice is the only way to reach it, and it is a tie in every
    # field, so the incumbent survives: recorded as behaviour, not left to
    # chance.
    var dup = List[Int32]()
    dup.append(0)
    dup.append(0)
    var res2 = node_split_random_gini[F](
        dataset, item, dup, obj, UInt64(0x2222), Int32(0)
    )
    assert_equal(res2.best_index, 0, "an exact duplicate cannot displace")
    cells += 1
    print("   ", cells, "cells;", ties, "double-in-gap draws")
    _ = box^
    return cells


# ==========================================================================
# 7. Order independence: the whole point of DEVIATION 130
# ==========================================================================


def check_order_independence() raises -> Int:
    print("[order] shuffled colids must give a bit-identical Split")
    var cells = 0
    var n_rows = 1024
    var n_cols = 8
    var fixture = hashed_classification(0x0DD, n_rows, n_cols, 4)
    var box = Box(fixture, True, True)
    var dataset = box.view()
    var obj = GiniObjectiveFunction[F](Int32(4), 1)

    var forward = List[Int32]()
    for c in range(n_cols):
        forward.append(Int32(c))

    var item = NodeWorkItem(21, 0, InstanceRange(0, Int32(n_rows)))
    var base = node_split_random_gini[F](
        dataset, item, forward, obj, UInt64(0xABC), Int32(6)
    )
    assert_true(base.found, "the reference run must find a split")
    cells += 1

    # Every rotation, plus a reversal, plus a fixed scramble.
    var orders = List[List[Int32]]()
    for k in range(n_cols):
        var o = List[Int32]()
        for c in range(n_cols):
            o.append(Int32((c + k) % n_cols))
        orders.append(o^)
    var rev = List[Int32]()
    for c in range(n_cols - 1, -1, -1):
        rev.append(Int32(c))
    orders.append(rev^)
    var scram = List[Int32]()
    for c in range(n_cols):
        scram.append(Int32((c * 5 + 3) % n_cols))
    orders.append(scram^)

    for oi in range(len(orders)):
        var res = node_split_random_gini[F](
            dataset, item, orders[oi], obj, UInt64(0xABC), Int32(6)
        )
        assert_equal(
            Int(res.split.colid), Int(base.split.colid), "colid is order-free"
        )
        assert_equal(
            res.split.quesval.to_bits(),
            base.split.quesval.to_bits(),
            "quesval is order-free, bitwise",
        )
        assert_equal(
            res.split.best_metric_val.to_bits(),
            base.split.best_metric_val.to_bits(),
            "best_metric_val is order-free, bitwise",
        )
        assert_equal(
            Int(res.split.n_left), Int(base.split.n_left), "n_left"
        )
        assert_equal(Int(res.n_constant), Int(base.n_constant), "n_constant")
        cells += 5
        # And every per-candidate record, matched by colid rather than by
        # position -- which is what `candidate_for` exists for.
        for c in range(n_cols):
            var i = res.candidate_for(Int32(c))
            var j = base.candidate_for(Int32(c))
            assert_true(i >= 0 and j >= 0, "every colid has a record")
            assert_equal(
                res.candidates[i].threshold.to_bits(),
                base.candidates[j].threshold.to_bits(),
                "per-candidate threshold is order-free",
            )
            assert_equal(
                Int(res.candidates[i].n_left),
                Int(base.candidates[j].n_left),
                "per-candidate n_left is order-free",
            )
            assert_equal(
                res.candidates[i].proxy_exact.num,
                base.candidates[j].proxy_exact.num,
                "per-candidate exact proxy is order-free",
            )
            cells += 4
    print("   ", cells, "cells over", len(orders), "orderings")
    _ = box^
    return cells


# ==========================================================================
# 8. min_samples_leaf: a rejection that actually fires
# ==========================================================================


def check_min_samples_leaf() raises -> Int:
    print("[min_samples_leaf] a rejection that fires, and is a skip not a redraw")
    var cells = 0
    var n_rows = 512
    var shapes = all_shapes()
    var fixture = shaped_dataset(0x5151, n_rows, shapes)
    var box = Box(fixture, True, True)
    var dataset = box.view()
    var rows = box.node_rows(0, n_rows)
    var colids = List[Int32]()
    for c in range(fixture.n_cols):
        colids.append(Int32(c))
    var item = NodeWorkItem(5, 0, InstanceRange(0, Int32(n_rows)))

    var fired = 0
    for msl in [1, 8, 64, 200]:
        var obj = GiniObjectiveFunction[F](
            Int32(fixture.n_classes), Int32(msl)
        )
        for tree in range(8):
            var res = node_split_random_gini[F](
                dataset, item, colids, obj, UInt64(0x606), Int32(tree)
            )
            for c in range(fixture.n_cols):
                ref rec = res.candidates[c]
                if rec.is_constant:
                    continue
                var t = tally_for(
                    fixture, rows, c, rec.threshold, fixture.n_classes
                )
                var want = t.n_left < msl or t.n_right < msl
                assert_equal(
                    rec.rejected_min_samples_leaf,
                    want,
                    "rejection verdict at min_samples_leaf=" + String(msl),
                )
                cells += 1
                if want:
                    fired += 1
                    # A `continue`, NOT a redraw: the threshold recorded is
                    # still the FIRST draw for this key.
                    var again = draw_threshold(
                        key_for(
                            UInt64(0x606), UInt32(tree), UInt32(5), UInt32(c)
                        ),
                        rec.extent,
                    )
                    assert_equal(
                        rec.threshold.to_bits(),
                        again.to_bits(),
                        "a rejected candidate is not re-drawn",
                    )
                    assert_true(
                        not rec.scored, "and it is not scored"
                    )
                    assert_true(
                        res.best_index < 0
                        or Int(res.split.colid) != c
                        or Int(res.candidates[res.best_index].colid) != c,
                        "and it cannot win",
                    )
                    cells += 3
    assert_true(
        fired > 0,
        "no min_samples_leaf rejection ever fired -- the fixture is the"
        " defect, not the splitter",
    )
    print("   ", cells, "cells;", fired, "rejections fired")
    _ = box^
    return cells


# ==========================================================================
# 9. DEVIATION 145: Float32 `best_metric_val` ties where the exact proxy does
#    not, built on purpose so the wrong comparator is visible
# ==========================================================================


def _two_valued_fixture(
    n_rows: Int, left_a: List[Int], left_b: List[Int], labels: List[Int32]
) -> FixtureDataset:
    """Two features, each taking only the values 0.0 and 1.0.

    A two-valued feature has a FIXED partition no matter what threshold is
    drawn: `min == 0`, `max == 1`, the draw is in `[0, 1)`, and every value
    `0.0` satisfies `<= t` while every `1.0` does not. sklearn's `== max`
    guard (`_splitter.pyx:653-654`) maps a draw of exactly `1.0` back to
    `0.0`, which is still the same partition. So this fixture pins the
    partition without pinning the RNG.
    """
    var x = List[Float32]()
    var y = List[Float32]()
    var lab = List[Int32]()
    for r in range(n_rows):
        x.append(Float32(1.0) if left_a[r] == 0 else Float32(0.0))
        x.append(Float32(1.0) if left_b[r] == 0 else Float32(0.0))
        lab.append(labels[r])
        y.append(Float32(Int(labels[r])))
    return FixtureDataset(n_rows, 2, 2, x^, y^, lab^)


def check_exact_beats_float() raises -> Int:
    print("[145] Float32 gains that tie while the exact proxies do not")
    var cells = 0
    var n = 512
    var t0 = 256  # class-0 rows: 0..255, class-1 rows: 256..511

    var obj = GiniObjectiveFunction[F](Int32(2), 1)

    # --- the search. Enumerate (n_left, l0) partitions, score each the way
    # the oracle would, and look for two whose Float32 gains are BIT
    # IDENTICAL while `CompareProxyExact` separates them. Pure arithmetic:
    # no data is built until a pair is found.
    var cand_nl = List[Int]()
    var cand_l0 = List[Int]()
    var cand_bits = List[UInt32]()
    var cand_num = List[Int64]()
    var cand_den = List[Int64]()

    var total_buf = List[CountBin](length=2, fill=CountBin(0))
    total_buf[0].x = Int32(t0)
    total_buf[1].x = Int32(n - t0)
    var left_buf = List[CountBin](length=2, fill=CountBin(0))
    var ht = total_buf.unsafe_ptr()
    var hl = left_buf.unsafe_ptr()

    for nl in range(1, n):
        var lo = 0
        if nl > t0:
            lo = nl - t0
        var hi = nl
        if hi > t0:
            hi = t0
        for l0 in range(lo, hi + 1):
            var l1 = nl - l0
            if l1 > n - t0:
                continue
            left_buf[0].x = Int32(l0)
            left_buf[1].x = Int32(l1)
            var g = obj.GainPerSplit(hl, ht, Int32(n), Int32(nl))
            var ex = obj.ProxyImpurityExact(hl, ht, Int32(n), Int32(nl))
            if not ex.valid:
                continue
            cand_nl.append(nl)
            cand_l0.append(l0)
            cand_bits.append(g.cast[DType.float32]().to_bits[DType.uint32]())
            cand_num.append(ex.num)
            cand_den.append(ex.den)
    _ = total_buf.unsafe_ptr()
    _ = left_buf.unsafe_ptr()

    var seen = Dict[UInt32, Int]()
    var hit_a = -1
    var hit_b = -1
    for i in range(len(cand_bits)):
        var b = cand_bits[i]
        if b in seen:
            var j = seen[b]
            var lhs = Int128(cand_num[i]) * Int128(cand_den[j])
            var rhs = Int128(cand_num[j]) * Int128(cand_den[i])
            if lhs != rhs:
                # `hit_a` is the one with the LARGER exact proxy; it is given
                # the SMALLER colid below, so a float tie-break -- which
                # falls through to "greater colid wins" -- picks the wrong
                # one and the check goes red.
                if lhs > rhs:
                    hit_a = i
                    hit_b = j
                else:
                    hit_a = j
                    hit_b = i
                break
        else:
            seen[b] = i
    assert_true(
        hit_a >= 0,
        "no Float32 gain collision with distinct exact proxies was found at"
        " n=512; DEVIATION 145 has no fixture and the search is the defect",
    )
    print(
        "    collision: (n_left,l0) =",
        cand_nl[hit_a],
        cand_l0[hit_a],
        "vs",
        cand_nl[hit_b],
        cand_l0[hit_b],
        " gain bits",
        cand_bits[hit_a],
    )

    # --- build the data that realises those two partitions.
    var labels = List[Int32]()
    for r in range(n):
        labels.append(Int32(0) if r < t0 else Int32(1))

    var left_a = List[Int](length=n, fill=0)
    var left_b = List[Int](length=n, fill=0)
    # feature 0 (colid 0) gets the LARGER exact proxy; feature 1 the smaller.
    for i in range(cand_l0[hit_a]):
        left_a[i] = 1
    for i in range(cand_nl[hit_a] - cand_l0[hit_a]):
        left_a[t0 + i] = 1
    for i in range(cand_l0[hit_b]):
        left_b[i] = 1
    for i in range(cand_nl[hit_b] - cand_l0[hit_b]):
        left_b[t0 + i] = 1

    var fixture = _two_valued_fixture(n, left_a, left_b, labels)
    var box = Box(fixture, True, False)
    var dataset = box.view()
    var colids = List[Int32]()
    colids.append(0)
    colids.append(1)
    var item = NodeWorkItem(0, 0, InstanceRange(0, Int32(n)))

    for tree in range(16):
        var res = node_split_random_gini[F](
            dataset, item, colids, obj, UInt64(0x145), Int32(tree)
        )
        ref a = res.candidates[0]
        ref b = res.candidates[1]
        # The partition really is fixed regardless of the draw.
        assert_equal(Int(a.n_left), cand_nl[hit_a], "feature 0 n_left is fixed")
        assert_equal(Int(b.n_left), cand_nl[hit_b], "feature 1 n_left is fixed")
        assert_equal(
            Int(a.hist_left[0].x), cand_l0[hit_a], "feature 0 left class 0"
        )
        assert_equal(
            Int(b.hist_left[0].x), cand_l0[hit_b], "feature 1 left class 0"
        )
        # The float reduction key really is a tie...
        assert_equal(
            a.gain.cast[DType.float32]().to_bits(),
            b.gain.cast[DType.float32]().to_bits(),
            "the Float32 gains must tie -- otherwise this fixture proves"
            " nothing about DEVIATION 145",
        )
        # ...while the exact comparator really does separate them...
        assert_equal(
            GiniObjectiveFunction[F].CompareProxyExact(
                a.proxy_exact, b.proxy_exact
            ),
            1,
            "the exact comparator must rank feature 0 above feature 1",
        )
        # ...and `Split.update`, the WRONG comparator, would pick colid 1.
        var wrong = Split()
        _ = wrong.update(Split(a.threshold, 0, a.gain.cast[DType.float32](), a.n_left))
        _ = wrong.update(Split(b.threshold, 1, b.gain.cast[DType.float32](), b.n_left))
        assert_equal(
            Int(wrong.colid),
            1,
            "Split.update must pick the WRONG feature here, or the fixture"
            " is not adversarial",
        )
        # The oracle must pick colid 0.
        assert_equal(
            Int(res.split.colid),
            0,
            "the exact comparator is the authority (DEVIATION 145)",
        )
        cells += 8
    print("   ", cells, "cells")
    _ = box^
    return cells


# ==========================================================================
# 10. Missing values are refused, not randomized (DEVIATION 136)
# ==========================================================================


def check_nan_refused() raises -> Int:
    print("[136] a NaN in the input is an error")
    var cells = 0
    var n = 64
    var x = List[Float32]()
    var y = List[Float32]()
    var lab = List[Int32]()
    for r in range(n):
        x.append(Float32(r))
        x.append(Float32(r) * 0.5)
        lab.append(Int32(r % 2))
        y.append(Float32(r % 2))
    x[2 * 7] = Float32(0.0) / Float32(0.0)  # a NaN in feature 0, row 7
    var fixture = FixtureDataset(n, 2, 2, x^, y^, lab^)
    var box = Box(fixture, True, False)
    var dataset = box.view()
    var obj = GiniObjectiveFunction[F](Int32(2), 1)
    var colids = List[Int32]()
    colids.append(0)
    colids.append(1)
    var item = NodeWorkItem(0, 0, InstanceRange(0, Int32(n)))

    var raised = False
    try:
        var res = node_split_random_gini[F](
            dataset, item, colids, obj, UInt64(1), Int32(0)
        )
        _ = res.found
    except e:
        raised = True
    assert_true(raised, "a NaN feature value must raise, not coin-flip")
    cells += 1

    # And the clean column alone is fine, so the refusal is about the NaN and
    # not about the fixture.
    var only1 = List[Int32]()
    only1.append(1)
    var ok = node_split_random_gini[F](
        dataset, item, only1, obj, UInt64(1), Int32(0)
    )
    assert_true(ok.found, "the NaN-free column still splits")
    cells += 1
    print("   ", cells, "cells")
    _ = box^
    return cells


def main() raises:
    var cells = 0
    cells += check_hashed_classification()
    cells += check_hashed_regression()
    cells += check_analytic_classification()
    cells += check_analytic_regression()
    cells += check_all_constant()
    cells += check_mixed_constants()
    cells += check_tie_break()
    cells += check_order_independence()
    cells += check_min_samples_leaf()
    cells += check_exact_beats_float()
    cells += check_nan_refused()
    print("host_splitter: ", cells, "cells")
    print("host_splitter_check: PASS")
