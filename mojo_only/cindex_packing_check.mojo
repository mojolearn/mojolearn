"""Does this tree's compressed index feed CatBoost's POINTWISE accumulators?

The pointwise family reads a feature's bits from a FIXED POSITION. It never
consults `TCFeature::Mask` or `::Shift` -- it hardcodes the packing:

    one-byte     (ci >> (24 - (f << 2))) & 255      f in {0,2,4,6}
    half-byte    (bins >> (28 - 4*i)) & 15          i in 0..7
    binary       nibble fid/4, bit (3 - (fid & 3))  fid in 0..31

The greedy-subsets family, which this repository ported first, reads `Mask`
and `Shift` per feature and so works with ANY packing. That difference is
why this check exists: the pointwise family will read whatever bits sit at
those positions and produce a perfectly well-formed histogram of the wrong
features. Nothing downstream can detect it -- every bin is populated, every
total is right, and the tree just splits on the wrong columns.

WHAT IS GATED. That `build_layout`'s `(shift, mask)` for every feature is
EXACTLY what the pointwise decode computes, for all three policies and at
every position in a group:

  K1  one-byte, 4 per column: shift 24, 16, 8, 0 -- CatBoost's
      `24 - (f << 2)` with its `f` stepping 0, 2, 4, 6 (its `f` is twice the
      feature index, because it indexes (feature, parity) slots).
  K2  half-byte, 8 per column: shift 28 - 4*i.
  K3  binary, 32 per column: shift 31 - fid, and the IDENTITY that makes
      CatBoost's two-step decode agree with it --

          31 - fid  ==  (28 - 4 * (fid / 4)) + (3 - (fid % 4))

      which is the whole reason `pw_hb_binary_sum` can recover a one-bit
      feature from a 16-bin histogram over its nibble. Checked at all 32
      positions, not derived.
  K4  masks are 255 / 15 / 1, and a feature's group is CONTIGUOUS in the
      compressed index -- 4, 8 and 32 features per column respectively,
      which is what `(blockIdx.x / M) * 4` (and `* 8`, `* 32`) assumes.

This is a HOST check. It compares two independently-written formulas -- the
builder's and the kernels' -- rather than running either, because if they
agree here they agree everywhere and if they disagree no fixture is needed
to say so.
"""

from gbdt.gpu_data.compressed_index_builder import build_layout


def main() raises:
    var failures = 0

    # ---------------------------------------------------------------- K1
    var ob: List[Int] = [20, 40, 100, 200, 33, 60, 70, 90]
    var lay1 = build_layout(ob)
    for f in range(8):
        ref cf = lay1.features[f]
        var k = f % 4
        # CatBoost's `f` steps 0,2,4,6 for feature index 0..3, so its shift
        # is `24 - (2k << 2)` ... which is `24 - 8k`.
        var want_shift = UInt32(24 - 8 * k)
        if cf.shift != want_shift:
            print(
                "FAIL K1: one-byte feature", f, "at group position", k,
                "has shift", cf.shift, "but the accumulator reads",
                want_shift,
            )
            failures += 1
        if cf.mask != UInt32(255):
            print("FAIL K1: one-byte mask is", cf.mask, "not 255")
            failures += 1
    if failures == 0:
        print("  ok   K1 -- one-byte: shifts 24/16/8/0, mask 255")

    # ---------------------------------------------------------------- K2
    var hb: List[Int] = [5, 12, 9, 3, 15, 7, 2, 11, 6, 4]
    var lay2 = build_layout(hb)
    var bad2 = 0
    for f in range(10):
        ref cf = lay2.features[f]
        var i = f % 8
        var want_shift = UInt32(28 - 4 * i)
        if cf.shift != want_shift or cf.mask != UInt32(15):
            print(
                "FAIL K2: half-byte feature", f, "position", i, "shift",
                cf.shift, "mask", cf.mask, "want shift", want_shift,
                "mask 15",
            )
            bad2 += 1
    if bad2 != 0:
        failures += bad2
    else:
        print("  ok   K2 -- half-byte: shifts 28..0 step 4, mask 15")

    # ---------------------------------------------------------------- K3
    var bins: List[Int] = List[Int]()
    for _ in range(40):
        bins.append(1)
    var lay3 = build_layout(bins)
    var bad3 = 0
    for f in range(40):
        ref cf = lay3.features[f]
        var fid = f % 32
        var want_shift = UInt32(31 - fid)
        if cf.shift != want_shift or cf.mask != UInt32(1):
            print(
                "FAIL K3: binary feature", f, "position", fid, "shift",
                cf.shift, "mask", cf.mask, "want", want_shift, "mask 1",
            )
            bad3 += 1
        # THE IDENTITY. CatBoost never computes `31 - fid`; it computes a
        # nibble and a bit within it, and the histogram is over the NIBBLE.
        # If these two disagree at any position, `pw_hb_binary_sum` sums the
        # wrong eight nibble values and the feature is silently swapped with
        # one of its three neighbours -- a permutation, so the group total
        # is unchanged.
        var nibble = fid // 4
        var bit_in_nibble = 3 - (fid % 4)
        var catboost_shift = (28 - 4 * nibble) + bit_in_nibble
        if catboost_shift != 31 - fid:
            print(
                "FAIL K3: at fid", fid, "CatBoost's nibble/bit decode gives",
                catboost_shift, "but the flat position is", 31 - fid,
            )
            bad3 += 1
    if bad3 != 0:
        failures += bad3
    else:
        print(
            "  ok   K3 -- binary: shift 31-fid at all 32 positions, and"
            " CatBoost's nibble/bit decode agrees at every one"
        )

    # ---------------------------------------------------------------- K4
    # a group is contiguous: 4 / 8 / 32 features share one column
    var bad4 = 0
    for f in range(8):
        var want_col = f // 4
        if Int(lay1.features[f].offset) != want_col:
            print(
                "FAIL K4: one-byte feature", f, "is in column",
                lay1.features[f].offset, "but `(blockIdx.x / M) * 4`"
                " assumes column", want_col,
            )
            bad4 += 1
    for f in range(10):
        if Int(lay2.features[f].offset) != f // 8:
            print("FAIL K4: half-byte feature", f, "column mismatch")
            bad4 += 1
    for f in range(40):
        if Int(lay3.features[f].offset) != f // 32:
            print("FAIL K4: binary feature", f, "column mismatch")
            bad4 += 1
    if bad4 != 0:
        failures += bad4
    else:
        print(
            "  ok   K4 -- groups are contiguous: 4 one-byte, 8 half-byte,"
            " 32 binary features per column"
        )

    if failures != 0:
        raise Error(
            String(failures)
            + " packing mismatch(es): this tree's compressed index does NOT"
            " feed the pointwise accumulators, and they will histogram the"
            " wrong features without any downstream symptom"
        )
    print(
        "compressed index packing: byte-identical to what CatBoost's"
        " pointwise family decodes (K1-K4)"
    )
