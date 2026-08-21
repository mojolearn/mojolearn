"""Gate for the one-hot flags the pointwise scorer reads.

**THIS CHECK EXISTS BECAUSE A GATED MECHANISM STAYED BROKEN.**

`scan_pointwise_histograms_kernel` skips the bin prefix scan for a one-hot
feature, which is theirs at `split_properties_helpers.cuh:126`
(`if (!feature->OneHotFeature)`). A one-hot bin is an EQUALITY test, so a
running prefix across its bins is not a quantity that means anything.

That skip IS gated, twice: `pointwise_offsets_check` plants
`[0, 0, 0, 1, 0, 0]` and `pointwise_dispatch_check` plants a one-hot feature
per policy. Both **construct the flag array BY HAND** and hand it straight to
the kernel.

So when `PolicyScoreHelper.__init__` filled that array with a hardcoded
`UInt8(0)` -- on the line after it read the same layout for the offset --
nothing saw it. The kernel was correct, the checks on the kernel were
correct, and the array the product actually passes was a constant. Every
one-hot feature got prefix-summed and its equality candidates scored as
thresholds. `bench/oracle_cat.txt` found it: 0 of 18 one-hot splits matching
CatBoost, train mse 2.30 against their 0.147.

`PORTING_RULES` 3 and 8 at once: the file had a caller, was not in
`UNWIRED.md`, the suite was green, and the branch the suite ran was not the
branch the product ran.

So this check does the one thing neither of those did: it hands
`PolicyScoreHelper` a LAYOUT with one-hot features in it and reads back what
the constructor actually built.

GATES:
  H1  a layout with one-hot features produces a flag array with those exact
      features flagged -- per feature, not a count.
  H2  a layout with NONE produces all zeros, so the gate is not passed by a
      constructor that flags everything.
  H3  the flags survive the POLICY SPLIT. A one-hot feature is flagged in
      its own policy's helper at its own within-policy index, which is not
      its global feature id. That indexing is where a per-policy table goes
      wrong, and it is invisible to a fixture with one policy.
"""

from max.gpu.host import DeviceContext

from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.methods.pointwise_scores_calcer import PolicyScoreHelper

comptime N_ROWS = 512


def flags_of(
    ctx: DeviceContext,
    folds: List[Int],
    one_hot: List[Bool],
) raises -> List[List[Int]]:
    """Build the helpers the searcher would build, and read their flags
    back off the device -- through the constructor, not around it."""
    var lay = build_layout(folds, one_hot)
    var blocks = blocks_for(lay, N_ROWS)
    var gids = List[Int]()
    for f in range(len(folds)):
        gids.append(f)
    var out = List[List[Int]]()
    for b in range(len(blocks)):
        var h = PolicyScoreHelper(ctx, blocks[b], lay, N_ROWS, 4, gids)
        var host = ctx.enqueue_create_host_buffer[DType.uint8](
            h.feature_count
        )
        ctx.enqueue_copy(dst_buf=host, src_buf=h.d_one_hot)
        ctx.synchronize()
        var row = List[Int]()
        for i in range(h.feature_count):
            row.append(Int(host[i]))
        out.append(row^)
    return out^


def main() raises:
    var ctx = DeviceContext()
    var failures = 0

    # three policies at once, one-hot features in TWO of them, and none of
    # them first in its policy -- a flag array indexed by global id instead
    # of by within-policy position lands on the wrong feature only when
    # those two indices differ
    var folds: List[Int] = [1, 1, 12, 9, 6, 20, 32, 48, 100]
    var one_hot: List[Bool] = [
        False, False, False, True, True, False, True, False, False
    ]

    var lay = build_layout(folds, one_hot)
    var blocks = blocks_for(lay, N_ROWS)
    var got = flags_of(ctx, folds, one_hot)

    # ---------------------------------------------------------------- H1
    var bad1 = 0
    var flagged = 0
    for b in range(len(blocks)):
        ref blk = blocks[b]
        for i in range(blk.count()):
            var gid = blk.feature_ids[i]
            var want = 1 if one_hot[gid] else 0
            if want == 1:
                flagged += 1
            if got[b][i] != want:
                print(
                    "FAIL H1: policy", blk.policy, "position", i,
                    "(global feature", gid, ") has flag", got[b][i],
                    "expected", want,
                )
                bad1 += 1
    if flagged == 0:
        print(
            "FAIL H1: the fixture flagged NO features, so this gate would"
            " pass against a constructor that returns all zeros",
        )
        bad1 += 1
    failures += bad1
    if bad1 == 0:
        print(
            "  ok   H1 --", flagged,
            "one-hot features flagged at their own within-policy positions,"
            " read back through the constructor",
        )

    # ---------------------------------------------------------------- H2
    var none: List[Bool] = [
        False, False, False, False, False, False, False, False, False
    ]
    var got2 = flags_of(ctx, folds, none)
    var stray = 0
    for b in range(len(got2)):
        for i in range(len(got2[b])):
            if got2[b][i] != 0:
                stray += 1
    if stray != 0:
        print(
            "FAIL H2:", stray,
            "features flagged one-hot in a layout that declares none",
        )
        failures += 1
    else:
        print("  ok   H2 -- a layout with no one-hot features flags none")

    # ---------------------------------------------------------------- H3
    # the within-policy index differs from the global id for every feature
    # after the first policy; assert the fixture actually has such a case
    var differs = 0
    for b in range(len(blocks)):
        ref blk = blocks[b]
        for i in range(blk.count()):
            if blk.feature_ids[i] != i and one_hot[blk.feature_ids[i]]:
                differs += 1
    if differs == 0:
        print(
            "FAIL H3: no flagged feature has a within-policy index that"
            " differs from its global id, so this fixture cannot tell the"
            " two indexings apart",
        )
        failures += 1
    else:
        print(
            "  ok   H3 --", differs,
            "flagged features sit at a within-policy index that differs"
            " from their global id, so the two indexings are separable",
        )

    if failures != 0:
        raise Error(String(failures) + " gate(s) failed")
    print("pointwise one-hot flags: H1-H3 pass")
