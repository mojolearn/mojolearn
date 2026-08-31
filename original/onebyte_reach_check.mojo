# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""WHICH of the four one-byte accumulators a fixture actually enters.

WHY THIS EXISTS
---------------
`PORTING.md` 108 recorded that `bench/oracle254.txt` "is the only one that
reaches the 8-bit kernel at all", and that is how the `fixed_scale` defect
was found: three fixtures, three kernels, and a whole fourth kernel that no
differential had ever run. The same sentence is a warning about the other
three. A fixture built for the 6-bit accumulator that silently lands on the
5-bit one still matches CatBoost split for split, because both accumulators
compute the same histogram -- so a green differential is NOT evidence that
the kernel it was built for ran.

`NEXT_TWO.md` rung 5 asks for a fixture per bit width. This file is the part
that makes that claim checkable rather than asserted.

THEIR DISPATCH, AND IT IS IN TWO PLACES
---------------------------------------
HOST (`pointwise_kernels.cpp:57-60`). All four widths are launched
unconditionally, each given its own feature count:

    DISPATCH_ONE_BYTE(ComputeHist2NonBinary, 4, 5)
    DISPATCH_ONE_BYTE(ComputeHist2NonBinary, 6, 6)
    DISPATCH_ONE_BYTE(ComputeHist2NonBinary, 7, 7)
    DISPATCH_ONE_BYTE(ComputeHist2NonBinary, 8, 8)

`FoldsHistogram::FeatureCountForBits(from, to)` (`folds_histogram.h:16-24`)
sums `Counts[bit]` over the range, and `Counts` is filled by
`TCpuGrid::ComputeFoldsHistogram` (`feature_layout.cpp:23-32`) at
`Counts[IntLog2(foldCount)]++`, where `IntLog2` is CEIL
(`libs/helpers/math_utils.h:14-16`). The launcher's only host-side gate is
`if (featureCountForBits)` (`pointwise_hist2_one_byte_templ.cuh:226`), so a
width no feature needs is skipped and every other width is launched over
EVERY one-byte feature.

DEVICE (`pointwise_hist2_one_byte_templ.cuh:179-183`). Each launched kernel
refuses the blocks that are not its own:

    constexpr ui32 upperBound = (1 << BITS);
    constexpr ui32 lowerBound = BITS > 5 ? upperBound / 2 : 15;
    if (maxBinCount <= lowerBound || maxBinCount > upperBound) return;

`maxBinCount` is the MAXIMUM `TCFeature::Folds` over the block's four
features (`GetMaxBinCount`, `split_properties_helpers.cuh:25-45`). So the
four ranges are

    5 bit    16 <= folds <=  32      (note the 15, not 16, in lowerBound)
    6 bit    33 <= folds <=  64
    7 bit    65 <= folds <= 128
    8 bit   129 <= folds <= 256

which are disjoint and agree with the host bucketing: `ceil(log2(folds))` is
4 or 5 exactly on 9..32, 6 on 33..64, 7 on 65..128, 8 on 129..256. Folds at
or below 15 never reach this family at all -- `SplitByPolicy`
(`compressed_index_builder.h:66-70`) sends them to HalfByte, whose
`MaxFolds()` is 15 (`grid_policy.h:62-64`).

WHAT IS OBSERVED HERE, AND WHAT IS PLANTED
------------------------------------------
DEVIATION 115's rule applies: a check that builds the kernel's inputs by
hand checks the kernel and not the caller. So everything that DECIDES the
dispatch is taken from the product:

  * `build_layout` assigns the policy, from the fixture's real fold counts.
  * `PolicyScoreHelper.__init__` builds `d_folds` -- the array
    `GetMaxBinCount` reduces -- and `folds_hist`, whose
    `feature_count_for_bits` is the host gate. Both come out of the
    constructor the searcher calls, not out of this file.

What this file plants is only what is INERT to the dispatch: the target, the
weight, the document indices and a single whole-dataset partition. None of
them is read by either gate.

The observation itself is that the four widths are then run ONE AT A TIME
into a zeroed histogram. Exactly one may come back non-empty, and it must be
the one their ranges name. Three empty and one full is the reach statement;
two full would mean our ranges overlap, and four empty would mean the
fixture reaches no accumulator at all and its differential result is about a
histogram of zeros.

R1  each committed one-byte fixture is claimed by exactly one width, and it
    is the width their bounds name for its fold count.
R2  THE SABOTAGE. The same fixture's data, re-declared at four fold counts
    that span the four ranges, walks the claim 5 -> 6 -> 7 -> 8. Without it
    R1 is a constant that would report "5-bit" for everything. The declared
    count is only ever raised, never lowered, so every planted bin stays
    inside the declared range and no write leaves its slot.
"""

from max.gpu.host import DeviceContext
from max.gpu.host.device_attribute import DeviceAttribute

from gbdt.grid_creator.binarization import binarize
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.gpu_data.grid_policy import POLICY_ONE_BYTE
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.pointwise_kernels import compute_hist2_non_binary
from gbdt.methods.pointwise_scores_calcer import PolicyScoreHelper

from original.oracle_check import Oracle, load_oracle


def expected_width(folds: Int) raises -> Int:
    """Their bounds, on the host, so the expectation is not our dispatch.

    `pointwise_hist2_one_byte_templ.cuh:180-181` with `maxBinCount = folds`.
    Returns 0 for a fold count no one-byte accumulator claims.
    """
    if folds > 15 and folds <= 32:
        return 5
    if folds > 32 and folds <= 64:
        return 6
    if folds > 64 and folds <= 128:
        return 7
    if folds > 128 and folds <= 256:
        return 8
    return 0


def claim_counts(
    ctx: DeviceContext,
    o: Oracle,
    declared_folds: List[Int],
) raises -> List[Int]:
    """Run each bit width alone and report how many histogram cells it wrote.

    `declared_folds[f]` is what the layout is told feature `f` has. For R1 it
    is the fixture's real border count; for R2 it is deliberately larger, to
    move the claim without moving a single bin.
    """
    var n_rows = o.rows
    var n_features = o.feats
    var one_hot = List[Bool]()
    for _ in range(n_features):
        one_hot.append(False)

    var lay = build_layout(declared_folds, one_hot)
    var blocks = blocks_for(lay, n_rows)

    # THE COMPRESSED INDEX, built the way `check_tree_structure` builds it:
    # their borders, our `binarize`, one launch of the product's writer per
    # feature.
    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows * lay.columns)
    var z = ctx.enqueue_create_host_buffer[DType.uint32](n_rows * lay.columns)
    for i in range(n_rows * lay.columns):
        z.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=z.unsafe_ptr())
    ctx.synchronize()

    var hb = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var bins = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    for f in range(n_features):
        ref cf = lay.features[f]
        for r in range(n_rows):
            hb.unsafe_ptr().unsafe_store(
                r, UInt8(binarize(o.x[f][r], o.borders[f]))
            )
        ctx.enqueue_copy(dst_buf=bins, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(Int(cf.offset) * n_rows), cf.mask, cf.shift,
            bins.unsafe_ptr(), Int32(n_rows), cindex.unsafe_ptr(),
            grid_dim=(n_rows + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=(WRITE_BLOCK_SIZE, 1, 1),
        )
        ctx.synchronize()

    # PLANTED, and inert to both gates: the target, the weight, the document
    # order and one whole-dataset partition.
    var targets = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var weights = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var docs = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    var hd = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for r in range(n_rows):
        ht.unsafe_ptr().unsafe_store(r, o.y[r])
        hw.unsafe_ptr().unsafe_store(r, Float32(1.0))
        hd.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.enqueue_copy(dst_buf=targets, src_ptr=ht.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=docs, src_ptr=hd.unsafe_ptr())

    # `TDataPartition{Offset, Size}` pairs. Sixteen slots so no shift the
    # offsets helper computes can run off the end; only slot 0 is non-empty.
    var parts = ctx.enqueue_create_buffer[DType.uint32](32)
    var hp = ctx.enqueue_create_host_buffer[DType.uint32](32)
    for i in range(32):
        hp.unsafe_ptr().unsafe_store(i, UInt32(0))
    hp.unsafe_ptr().unsafe_store(1, UInt32(n_rows))
    ctx.enqueue_copy(dst_buf=parts, src_ptr=hp.unsafe_ptr())
    ctx.synchronize()

    var gids = List[Int]()
    for f in range(n_features):
        gids.append(f)

    var out = List[Int]()
    for _ in range(4):
        out.append(-1)

    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)

    for b in range(len(blocks)):
        if blocks[b].policy != POLICY_ONE_BYTE:
            continue
        # THE PRODUCT'S OWN TABLES. `d_folds` is what `GetMaxBinCount`
        # reduces and `folds_hist` is the host gate; both are built by the
        # constructor the searcher calls.
        var h = PolicyScoreHelper(ctx, blocks[b], lay, n_rows, 4, gids)
        var cells = 2 * h.bin_feature_count
        var hh = ctx.enqueue_create_host_buffer[DType.float32](cells)

        # The 8-bit accumulator is Int32 fixed point (DEVIATION 93), so a
        # scale of 1.0 would quantize this fixture's gradients to zero and
        # make an ENTERED kernel look like a skipped one -- which is
        # `PORTING.md` 108's defect wearing a different hat. 1024 is well
        # inside Int32 for 4096 rows of this target.
        var scale = Float32(1024.0)

        for w in range(4):
            var bits = 5 + w
            ctx.enqueue_memset(h.d_hist, Float32(0.0))
            ctx.synchronize()
            var fcb: Int
            if bits == 5:
                fcb = h.folds_hist.feature_count_for_bits(4, 5)
            elif bits == 6:
                fcb = h.folds_hist.feature_count_for_bits(6, 6)
            elif bits == 7:
                fcb = h.folds_hist.feature_count_for_bits(7, 7)
            else:
                fcb = h.folds_hist.feature_count_for_bits(8, 8)

            if bits == 5:
                compute_hist2_non_binary[5](
                    ctx, h.d_offset.unsafe_ptr(),
                    h.d_first_fold.unsafe_ptr(), h.d_folds.unsafe_ptr(),
                    h.feature_count, cindex.unsafe_ptr(),
                    targets.unsafe_ptr(), weights.unsafe_ptr(),
                    docs.unsafe_ptr(), n_rows, parts.unsafe_ptr(), 1, 1,
                    True, h.bin_feature_count, h.d_hist.unsafe_ptr(),
                    fcb, sm_count, scale,
                )
            elif bits == 6:
                compute_hist2_non_binary[6](
                    ctx, h.d_offset.unsafe_ptr(),
                    h.d_first_fold.unsafe_ptr(), h.d_folds.unsafe_ptr(),
                    h.feature_count, cindex.unsafe_ptr(),
                    targets.unsafe_ptr(), weights.unsafe_ptr(),
                    docs.unsafe_ptr(), n_rows, parts.unsafe_ptr(), 1, 1,
                    True, h.bin_feature_count, h.d_hist.unsafe_ptr(),
                    fcb, sm_count, scale,
                )
            elif bits == 7:
                compute_hist2_non_binary[7](
                    ctx, h.d_offset.unsafe_ptr(),
                    h.d_first_fold.unsafe_ptr(), h.d_folds.unsafe_ptr(),
                    h.feature_count, cindex.unsafe_ptr(),
                    targets.unsafe_ptr(), weights.unsafe_ptr(),
                    docs.unsafe_ptr(), n_rows, parts.unsafe_ptr(), 1, 1,
                    True, h.bin_feature_count, h.d_hist.unsafe_ptr(),
                    fcb, sm_count, scale,
                )
            else:
                compute_hist2_non_binary[8](
                    ctx, h.d_offset.unsafe_ptr(),
                    h.d_first_fold.unsafe_ptr(), h.d_folds.unsafe_ptr(),
                    h.feature_count, cindex.unsafe_ptr(),
                    targets.unsafe_ptr(), weights.unsafe_ptr(),
                    docs.unsafe_ptr(), n_rows, parts.unsafe_ptr(), 1, 1,
                    True, h.bin_feature_count, h.d_hist.unsafe_ptr(),
                    fcb, sm_count, scale,
                )
            ctx.synchronize()
            ctx.enqueue_copy(dst_buf=hh, src_buf=h.d_hist)
            ctx.synchronize()
            var nz = 0
            for i in range(cells):
                if hh[i] != Float32(0.0):
                    nz += 1
            out[w] = nz

    # KEEP THE BUFFERS ALIVE PAST THE LAUNCHES. A DeviceBuffer handed to a
    # kernel as a raw pointer is dead at `.unsafe_ptr()` and the next
    # allocation lands on it, intermittently.
    ctx.synchronize()
    _ = docs
    _ = parts
    _ = cindex
    _ = bins
    _ = targets
    _ = weights
    return out^


def report(name: String, folds: Int, counts: List[Int]) raises -> Int:
    """Print the four cell counts and return 1 if the claim is not their one.
    """
    var want = expected_width(folds)
    var claimed = 0
    var claimed_by = -1
    var line = String("    folds ") + String(folds) + " -> "
    for w in range(4):
        line += String(5 + w) + "bit:" + String(counts[w]) + " "
        if counts[w] > 0:
            claimed += 1
            claimed_by = 5 + w
    print(line, " (their bounds name", want, "bit )")
    if claimed != 1:
        print(
            "FAIL", name, ":", claimed,
            "widths wrote a non-empty histogram; exactly one may, or the"
            " four ranges are not disjoint (all four empty means the"
            " fixture reaches no accumulator at all)",
        )
        return 1
    if claimed_by != want:
        print(
            "FAIL", name, ": claimed by the", claimed_by,
            "bit accumulator where their bounds name", want,
        )
        return 1
    return 0


def main() raises:
    var ctx = DeviceContext()
    var failures = 0

    # ------------------------------------------------------------- R1
    print("R1 -- each committed one-byte fixture, at its real fold count:")
    var paths = List[String]()
    paths.append(String("bench/oracle24.txt"))
    paths.append(String("bench/oracle48.txt"))
    paths.append(String("bench/oracle100.txt"))
    paths.append(String("bench/oracle254.txt"))
    var seen = List[Int]()
    for i in range(len(paths)):
        var p = paths[i].copy()
        var o = load_oracle(p)
        var folds = List[Int]()
        for f in range(o.feats):
            folds.append(len(o.borders[f]))
        # every feature the same width here, which is what makes a fixture
        # a statement about ONE accumulator
        var widest = 0
        for f in range(o.feats):
            if folds[f] > widest:
                widest = folds[f]
        print("  ", p)
        var c = claim_counts(ctx, o, folds)
        failures += report(p, widest, c)
        seen.append(expected_width(widest))

    var covered = 0
    for w in range(4):
        for i in range(len(seen)):
            if seen[i] == 5 + w:
                covered += 1
                break
    if covered != 4:
        print(
            "FAIL R1: the four fixtures cover", covered,
            "of the four one-byte widths, so at least one accumulator has"
            " no fixture",
        )
        failures += 1
    else:
        print(
            "  ok   R1 -- all four one-byte accumulators have a fixture, and"
            " each fixture is claimed by exactly the one their bounds name"
        )

    # ------------------------------------------------------------- R2
    # THE SABOTAGE. One fixture's data, four declared fold counts. Its real
    # bins run 0..24, so every declared count below is at least that and no
    # planted bin leaves its slot; only the DISPATCH moves.
    print(
        "R2 -- one fixture, four declared fold counts, the claim must walk:"
    )
    var o24 = load_oracle(String("bench/oracle24.txt"))
    var ladder = List[Int]()
    ladder.append(24)
    ladder.append(40)
    ladder.append(80)
    ladder.append(200)
    var walked = List[Int]()
    for i in range(len(ladder)):
        var declared = List[Int]()
        for _ in range(o24.feats):
            declared.append(ladder[i])
        var c = claim_counts(ctx, o24, declared)
        failures += report(String("R2"), ladder[i], c)
        var by = -1
        for w in range(4):
            if c[w] > 0:
                by = 5 + w
        walked.append(by)
    var distinct = True
    for i in range(len(walked)):
        for j in range(i + 1, len(walked)):
            if walked[i] == walked[j]:
                distinct = False
    if not distinct:
        print(
            "FAIL R2: the four declared fold counts did not move the claim"
            " to four different accumulators, so R1's answer does not"
            " depend on the fold count and is not an observation",
        )
        failures += 1
    else:
        print(
            "  ok   R2 -- the claim walked 5 -> 6 -> 7 -> 8 on ONE fixture's"
            " data as only its declared fold count moved"
        )

    if failures != 0:
        raise Error(String(failures) + " reach gate(s) failed")
    print("one-byte reach: R1-R2 pass")
