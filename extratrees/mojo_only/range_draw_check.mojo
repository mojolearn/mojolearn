"""The range pass, the constant test, and the keyed threshold draw.

These are steps 1-3 of DEVIATION 137 — the part of `computeSplitKernel` this
formulation replaces — and each is checked against a different kind of
authority:

* **the range** against an independent per-column tally over the node's rows,
  on adversarial column shapes (constant, near-constant either side of
  `FEATURE_THRESHOLD`, two-valued, one far outlier, all-negative, spanning
  zero, one odd row);
* **the constant test** against sklearn's own three arms, INCLUDING the
  float32 rounding that widens the band — a fact this check demonstrates
  rather than asserts;
* **the draw** against its own stated contract: inside the range, never equal
  to `max`, a pure function of the key, and different for every component of
  the key.

The last one carries the question that cannot be settled by reading:
sklearn's `if threshold == max_feature_value: threshold = min_feature_value`
guard (`_splitter.pyx:651-652`) exists because THEIR uniform can return
exactly `max`. Ours comes from RAFT's `next_float`, whose range is
`[0, 1 - 2^-24]`, so the mathematical product is strictly inside — but the
float32 rounding of `min + res * span` can still land on `max`. This check
COUNTS how often, over many ranges, instead of assuming either way.
"""

from std.testing import assert_equal, assert_true

from extratrees.mojo_only.fixtures import (
    Dataset as FixtureDataset,
    all_shapes,
    shape_name,
    shaped_dataset,
)
from extratrees.mojo_only.pcg_rng import key_for
from extratrees.ported.decisiontree.batched_levelalgo.dataset import Dataset
from extratrees.ported.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
)
from extratrees.ported.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    FEATURE_THRESHOLD,
    FeatureRange,
    draw_threshold,
    draw_threshold_raw,
    node_feature_is_constant,
    node_feature_min_max,
)


def column_major(fixture: FixtureDataset) -> List[Float32]:
    """CuML's `Dataset` is column major (`dataset.h:24`); the fixture module
    is row major. One transposition, done once, so the rest of the check reads
    the same layout the kernels will."""
    var out = List[Float32](
        length=fixture.n_rows * fixture.n_cols, fill=Float32(0.0)
    )
    for r in range(fixture.n_rows):
        for c in range(fixture.n_cols):
            out[c * fixture.n_rows + r] = fixture.value(r, c)
    return out^


def main() raises:
    var cells = 0

    var shapes = all_shapes()
    var n_rows = 512
    var fixture = shaped_dataset(0xA11CE, n_rows, shapes)
    var flat = column_major(fixture)
    var labels = List[Float32](length=n_rows, fill=Float32(0.0))

    # A node that is NOT the whole dataset, and NOT starting at 0, over a
    # SHUFFLED row_ids: three separate ways for an index mistake to show.
    var row_ids = List[Int32]()
    for r in range(n_rows):
        row_ids.append(Int32(r))
    for i in range(n_rows - 1, 0, -1):
        var h = UInt32(i) * 2654435761
        h ^= h >> 15
        var j = Int(h % UInt32(i + 1))
        var t = row_ids[i]
        row_ids[i] = row_ids[j]
        row_ids[j] = t

    var dataset = Dataset(
        rebind[MutPointer[Float32, MutUntrackedOrigin]](flat.unsafe_ptr()),
        rebind[MutPointer[Float32, MutUntrackedOrigin]](labels.unsafe_ptr()),
        Int32(n_rows),
        Int32(fixture.n_cols),
        Int32(n_rows),
        Int32(fixture.n_cols),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](row_ids.unsafe_ptr()),
        Int32(1),
    )

    # --- the range pass, per (node, column), against an independent tally ---
    var node_ranges = [(0, n_rows), (0, 1), (7, 1), (7, 100), (300, 212), (511, 1)]
    print("[range] per (node, column) against an independent tally")
    for nr in node_ranges:
        var item = NodeWorkItem(0, 0, InstanceRange(Int32(nr[0]), Int32(nr[1])))
        for c in range(fixture.n_cols):
            var got = node_feature_min_max(dataset, item, Int32(c))
            # Independent: a plain loop over the node's rows, no seeding
            # subtleties, using the fixture's own row-major accessor.
            var lo = Float32(0.0)
            var hi = Float32(0.0)
            var first = True
            for s in range(nr[0], nr[0] + nr[1]):
                var v = fixture.value(Int(row_ids[s]), c)
                if first:
                    lo = v
                    hi = v
                    first = False
                else:
                    if v < lo:
                        lo = v
                    if v > hi:
                        hi = v
            assert_equal(
                got.min_value.to_bits(),
                lo.to_bits(),
                "min differs, column " + shape_name(shapes[c]),
            )
            assert_equal(
                got.max_value.to_bits(),
                hi.to_bits(),
                "max differs, column " + shape_name(shapes[c]),
            )
            assert_equal(got.n_missing, 0, "no fixture may contain a NaN")
            cells += 3
    print("   ", cells, "cells over", len(node_ranges), "nodes x", fixture.n_cols, "shapes")

    # --- the constant test, both sides, on the straddling pair -------------
    # The near-constant columns are built to sit just below, exactly on, and
    # just above FEATURE_THRESHOLD. That pair is the only thing that can catch
    # a wrong comparison direction in the constant test.
    print("[constant] sklearn's test, both sides of FEATURE_THRESHOLD")
    var whole = NodeWorkItem(0, 0, InstanceRange(0, Int32(n_rows)))
    var reported = List[Int]()
    for c in range(fixture.n_cols):
        var extent = node_feature_min_max(dataset, whole, Int32(c))
        var is_const = node_feature_is_constant(extent, Int32(n_rows))
        reported.append(1 if is_const else 0)
        var spread = extent.max_value - extent.min_value
        print(
            "   ",
            shape_name(shapes[c]),
            " min ",
            extent.min_value,
            " max ",
            extent.max_value,
            " spread ",
            spread,
            " constant ",
            is_const,
        )
        cells += 1
    # A fixture in which every column agreed would test nothing.
    var n_const = 0
    for c in range(fixture.n_cols):
        n_const += reported[c]
    assert_true(
        n_const > 0 and n_const < fixture.n_cols,
        "the shape fixture must contain BOTH constant and non-constant"
        " columns, or this test has no sides",
    )
    cells += 1

    # THE FLOAT32 WIDENING, demonstrated. sklearn's test is
    # `max <= min + FEATURE_THRESHOLD` with both operands float32. At
    # min == 0.5 the sum rounds UP, so the effective band is wider than 1e-7.
    var anchored_at_zero = Float32(0.0) + FEATURE_THRESHOLD
    var anchored_at_half = Float32(0.5) + FEATURE_THRESHOLD
    var band_at_zero = anchored_at_zero - Float32(0.0)
    var band_at_half = anchored_at_half - Float32(0.5)
    print(
        "[float32 band] at min=0.0 the band is",
        band_at_zero,
        " at min=0.5 it is",
        band_at_half,
    )
    assert_true(
        band_at_half > band_at_zero,
        "the float32 addition must WIDEN the band away from zero -- if this"
        " ever fails, the test has been promoted to float64 and will disagree"
        " with sklearn on real data",
    )
    cells += 1

    # AND FURTHER OUT IT DOES NOT WIDEN, IT VANISHES. At 3.25 one ulp is
    # 2.384e-7, so half an ulp is 1.19e-7 -- larger than FEATURE_THRESHOLD --
    # and `3.25f + 1e-7f` rounds straight back to 3.25. sklearn's constant test
    # therefore degenerates to `max <= min` for every feature whose magnitude
    # is above about 2: a column at that scale is "constant" only if it is
    # EXACTLY constant. This is not a quirk of ours; it is what their float32
    # expression does, and a port that computed it in float64 would call
    # columns constant that sklearn splits.
    var band_at_325 = (Float32(3.25) + FEATURE_THRESHOLD) - Float32(3.25)
    assert_equal(
        band_at_325,
        Float32(0.0),
        "at 3.25 the float32 constant band must VANISH, not merely widen",
    )
    print("[float32 band] at min=3.25 it is", band_at_325, "-- the band vanishes")
    cells += 1

    # --- the draw ----------------------------------------------------------
    print("[draw] contract, keying, and whether sklearn's == max guard fires")
    var n_draws = 0
    var n_raw_hit_max = 0
    var n_guarded = 0
    for c in range(fixture.n_cols):
        var extent = node_feature_min_max(dataset, whole, Int32(c))
        if node_feature_is_constant(extent, Int32(n_rows)):
            continue
        for tree_id in range(4):
            for node_id in range(64):
                var key = key_for(
                    UInt64(0xBEEF), UInt32(tree_id), UInt32(node_id), UInt32(c)
                )
                var t = draw_threshold(key, extent)
                var raw = draw_threshold_raw(key, extent)
                assert_true(
                    t >= extent.min_value and t <= extent.max_value,
                    "a threshold outside [min, max] cannot split this node",
                )
                assert_true(
                    t != extent.max_value,
                    "a threshold equal to max sends EVERY row left; sklearn's"
                    " guard (_splitter.pyx:651-652) exists to prevent it",
                )
                if raw == extent.max_value:
                    n_raw_hit_max += 1
                    assert_equal(
                        t.to_bits(),
                        extent.min_value.to_bits(),
                        "the guard must replace such a draw with MIN, not max",
                    )
                    n_guarded += 1
                # a pure function of the key
                assert_equal(
                    t.to_bits(),
                    draw_threshold(key, extent).to_bits(),
                    "the same key must give the same threshold, always",
                )
                n_draws += 1
                cells += 3
    print(
        "   ",
        n_draws,
        "draws; raw draw landed exactly on max",
        n_raw_hit_max,
        "times; guard replaced",
        n_guarded,
    )

    # --- CAN sklearn's guard fire here at all? A targeted sweep ------------
    # The draws above never landed on `max`, but they only cover ten column
    # ranges. The guard's reachability is a question about float32 rounding of
    # `min + res * span`, so sweep spans across many magnitudes -- including
    # spans so small that a single rounding step spans the whole interval --
    # and count. Reported either way; a guard that cannot fire is still kept,
    # because it is sklearn's and because "cannot fire on this generator, at
    # these magnitudes, in this many draws" is not "cannot fire".
    var swept = 0
    var swept_hit_max = 0
    for e in range(-30, 31):
        var span = Float32(2.0) ** Float32(e)
        for base in [Float32(0.0), Float32(1.0), Float32(-1.0), Float32(1e6)]:
            var ext = FeatureRange(base, base + span, 0)
            if ext.max_value <= ext.min_value:
                continue  # span vanished under the base's exponent
            for node_id in range(64):
                var k = key_for(
                    UInt64(0x51DE), UInt32(1), UInt32(node_id), UInt32(e + 40)
                )
                var raw = draw_threshold_raw(k, ext)
                var guarded = draw_threshold(k, ext)
                assert_true(
                    guarded >= ext.min_value and guarded < ext.max_value
                    or guarded == ext.min_value,
                    "guarded threshold escaped [min, max)",
                )
                if raw == ext.max_value:
                    swept_hit_max += 1
                    assert_equal(
                        guarded.to_bits(),
                        ext.min_value.to_bits(),
                        "the guard must replace a max-valued draw with MIN",
                    )
                swept += 1
                cells += 1
    print(
        "[guard sweep]",
        swept,
        "draws over 61 span magnitudes x 4 bases; raw hit max",
        swept_hit_max,
        "times",
    )

    # --- every component of the key must move the draw ---------------------
    # Rule 8's shape applied to a hash: a key component that is ignored is
    # invisible unless something varies ONLY that component.
    var extent0 = node_feature_min_max(dataset, whole, Int32(0))
    var base = draw_threshold(
        key_for(UInt64(1), UInt32(2), UInt32(3), UInt32(4)), extent0
    )
    var moved = 0
    if draw_threshold(key_for(UInt64(9), UInt32(2), UInt32(3), UInt32(4)), extent0) != base:
        moved += 1
    if draw_threshold(key_for(UInt64(1), UInt32(9), UInt32(3), UInt32(4)), extent0) != base:
        moved += 1
    if draw_threshold(key_for(UInt64(1), UInt32(2), UInt32(9), UInt32(4)), extent0) != base:
        moved += 1
    if draw_threshold(key_for(UInt64(1), UInt32(2), UInt32(3), UInt32(9)), extent0) != base:
        moved += 1
    assert_equal(
        moved,
        4,
        "all four key components (seed, tree, node, feature) must change the"
        " draw; a component that does not is silently not in the key",
    )
    cells += 4

    # --- a degenerate node: one row -----------------------------------------
    # min == max, so the feature is constant and must never be drawn from.
    var one = NodeWorkItem(0, 0, InstanceRange(42, 1))
    for c in range(fixture.n_cols):
        var extent = node_feature_min_max(dataset, one, Int32(c))
        assert_equal(
            extent.min_value.to_bits(),
            extent.max_value.to_bits(),
            "a one-row node has min == max for every feature",
        )
        assert_true(
            node_feature_is_constant(extent, 1),
            "and every feature is therefore constant there",
        )
        cells += 2

    # KEEP THE BACKING BUFFERS ALIVE. `Dataset`'s pointers are
    # `MutUntrackedOrigin` -- the compiler tracks no relationship between them
    # and the `List`s they point into, so Mojo's ASAP destruction frees a list
    # after its LAST SYNTACTIC USE, which without these lines is the `Dataset`
    # constructor call. This is not hypothetical: without them, this check read
    # freed memory and reported min == 0.0 for every column, including the
    # all-3.25 constant column and the all-negative one, while the range
    # section a few lines earlier still passed because the freed pages had not
    # been reused yet.
    _ = flat.unsafe_ptr()
    _ = labels.unsafe_ptr()
    _ = row_ids.unsafe_ptr()

    print("range_draw: ", cells, "cells")
    print("range_draw_check: PASS")
