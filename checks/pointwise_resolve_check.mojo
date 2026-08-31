# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device winner fold against the host fold, record for record.

DEVIATION 207 moved the pointwise level winner's resolution --
`PolicyScoreHelper.read_optimal_split`'s per-block fold, the calcer's
per-helper fold, and the searcher's consumption of the record -- onto the
device (`gbdt/methods/kernel/pointwise_split_resolve.mojo`). The fit
gates hold the whole path end to end, but they hold it at fixtures whose
ties never fire and whose `binFeaturesWeights` are ones, so two things
would slip through them:

* a WRONG TIE RULE (signed sentinel comparison, reversed fold order,
  keep-first-instead-of-second) -- invisible until a real tie, which is
  exactly when it matters;
* a DEAD `score_before` WRITE in the pack kernel -- ranking-invisible at
  uniform feature weights (PREP_BILL step 27), so no fit gate can see it.

So this check plants HASHED records ([[uniform-test-data-hides-
permutation]]: every cell distinct) plus every tie case by hand, folds
them with the DEVICE kernels in the calcer's exact launch order, folds
the same records with the HOST `take_best` in the host code's exact
nesting, and compares record for record -- to the BIT for the floats.
The pack half is read back whole: winner slot, `score_before`, and the
five-word descriptor against a planted feature table.

R1  hashed records, no ties: device == host over every (level, field).
R2  planted ties: equal gain across helpers (earlier policy wins), equal
    gain within a helper (earlier block wins), gain tie broken by fid as
    UNSIGNED (the sentinel loses), fid tie broken by bin.
R3  all-sentinel level: the fold returns the default record and the pack
    clamps the table index to feature 0 with the record's bin.
R4  the pack's plumbing: `score_before` holds the winner's Score and the
    descriptor holds the winner's feature row + bin, per planted cell.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from gbdt.methods.helpers import TBestSplitProperties, take_best
from gbdt.methods.kernel.pointwise_split_resolve import (
    PW_SENTINEL_ID,
    launch_pw_fold_winner,
    launch_pw_pack_winner,
    launch_pw_seed_sentinel,
)


def _mix(x: UInt64) -> UInt64:
    """splitmix64's finalizer -- distinct planted values everywhere."""
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D4914D82CE2B49)
    return z ^ (z >> 31)


struct HelperRecords(Copyable, Movable):
    """One fake helper's per-block records, host side."""

    var ids: List[UInt32]
    var scores: List[Float32]

    def __init__(out self):
        self.ids = List[UInt32]()
        self.scores = List[Float32]()

    def append(mut self, fid: UInt32, bin: UInt32, score: Float32, gain: Float32):
        self.ids.append(fid)
        self.ids.append(bin)
        self.scores.append(score)
        self.scores.append(gain)

    def blocks(self) -> Int:
        return len(self.ids) // 2


def _host_fold(helpers: List[HelperRecords]) -> TBestSplitProperties:
    """The host nesting, verbatim: per-block `take_best(record, best)`
    inside a helper (`PolicyScoreHelper.read_optimal_split`), then
    `take_best(folded, best)` across helpers (`ReadOptimalSplit`,
    `pointwise_scores_calcer.h:94-105`). Uses the REAL `take_best`, not a
    reimplementation, so this side cannot drift from the searcher."""
    var best = TBestSplitProperties()
    for h in range(len(helpers)):
        if helpers[h].blocks() == 0:
            continue
        var local = TBestSplitProperties()
        for b in range(helpers[h].blocks()):
            local = take_best(
                TBestSplitProperties(
                    Int32(helpers[h].ids[2 * b]),
                    Int32(helpers[h].ids[2 * b + 1]),
                    helpers[h].scores[2 * b],
                    helpers[h].scores[2 * b + 1],
                ),
                local,
            )
        best = take_best(local, best)
    return best


def _device_fold_and_pack(
    ctx: DeviceContext,
    helpers: List[HelperRecords],
    depth: Int,
    feat_table: List[UInt32],
    n_features: Int,
    mut out_winner: List[UInt32],
    mut out_wscores: List[Float32],
    mut out_sb: List[Float32],
    mut out_desc: List[UInt32],
) raises:
    """The device side, launched in `resolve_optimal_split`'s exact order,
    then packed and read back whole."""
    var d_best_ids = ctx.enqueue_create_buffer[DType.uint32](2)
    var d_best_scores = ctx.enqueue_create_buffer[DType.float32](2)
    var d_winners_ids = ctx.enqueue_create_buffer[DType.uint32](2 * (depth + 1))
    var d_winners_scores = ctx.enqueue_create_buffer[DType.float32](
        2 * (depth + 1)
    )
    var d_sb = ctx.enqueue_create_buffer[DType.float32](1)
    var d_desc = ctx.enqueue_create_buffer[DType.uint32](5)
    var d_table = ctx.enqueue_create_buffer[DType.uint32](len(feat_table))
    ctx.enqueue_copy(dst_buf=d_table, src_ptr=feat_table.unsafe_ptr())
    # a planted poison so a dead write is a FAIL, not a lucky zero
    d_sb.enqueue_fill(Float32(-777.25))

    var d_res_ids = List[DeviceBuffer[DType.uint32]]()
    var d_res_scores = List[DeviceBuffer[DType.float32]]()
    for h in range(len(helpers)):
        var n = helpers[h].blocks()
        var di = ctx.enqueue_create_buffer[DType.uint32](2 * n if n > 0 else 2)
        var ds = ctx.enqueue_create_buffer[DType.float32](
            2 * n if n > 0 else 2
        )
        if n > 0:
            ctx.enqueue_copy(dst_buf=di, src_ptr=helpers[h].ids.unsafe_ptr())
            ctx.enqueue_copy(
                dst_buf=ds, src_ptr=helpers[h].scores.unsafe_ptr()
            )
        d_res_ids.append(di^)
        d_res_scores.append(ds^)

    # `resolve_optimal_split`'s order: skip empty, is_first on the first
    # live helper, sentinel if none
    var first = True
    for h in range(len(helpers)):
        if helpers[h].blocks() == 0:
            continue
        launch_pw_fold_winner(
            ctx,
            d_res_ids[h],
            d_res_scores[h],
            helpers[h].blocks(),
            first,
            d_best_ids,
            d_best_scores,
        )
        first = False
    if first:
        launch_pw_seed_sentinel(ctx, d_best_ids, d_best_scores)

    launch_pw_pack_winner(
        ctx,
        d_best_ids,
        d_best_scores,
        depth,
        d_winners_ids,
        d_winners_scores,
        d_sb,
        d_table,
        n_features,
        d_desc,
    )

    var h_w = ctx.enqueue_create_host_buffer[DType.uint32](2 * (depth + 1))
    var h_ws = ctx.enqueue_create_host_buffer[DType.float32](2 * (depth + 1))
    var h_sb = ctx.enqueue_create_host_buffer[DType.float32](1)
    var h_desc = ctx.enqueue_create_host_buffer[DType.uint32](5)
    ctx.enqueue_copy(dst_buf=h_w, src_buf=d_winners_ids)
    ctx.enqueue_copy(dst_buf=h_ws, src_buf=d_winners_scores)
    ctx.enqueue_copy(dst_buf=h_sb, src_buf=d_sb)
    ctx.enqueue_copy(dst_buf=h_desc, src_buf=d_desc)
    ctx.synchronize()

    out_winner.clear()
    out_wscores.clear()
    out_sb.clear()
    out_desc.clear()
    out_winner.append(h_w[2 * depth])
    out_winner.append(h_w[2 * depth + 1])
    out_wscores.append(h_ws[2 * depth])
    out_wscores.append(h_ws[2 * depth + 1])
    out_sb.append(h_sb[0])
    for k in range(5):
        out_desc.append(h_desc[k])

    # ([[mojo-buffer-freed-at-last-use]])
    _ = d_best_ids^
    _ = d_best_scores^
    _ = d_winners_ids^
    _ = d_winners_scores^
    _ = d_sb^
    _ = d_desc^
    _ = d_table^
    _ = d_res_ids^
    _ = d_res_scores^
    _ = h_w^
    _ = h_ws^
    _ = h_sb^
    _ = h_desc^
    _ = feat_table[0]


def _check_case(
    ctx: DeviceContext,
    helpers: List[HelperRecords],
    feat_table: List[UInt32],
    n_features: Int,
    label: String,
    mut failures: Int,
) raises:
    """One fold: device vs host, plus the pack's three outputs."""
    var want = _host_fold(helpers)

    var got_w = List[UInt32]()
    var got_ws = List[Float32]()
    var got_sb = List[Float32]()
    var got_desc = List[UInt32]()
    var depth = 3
    _device_fold_and_pack(
        ctx, helpers, depth, feat_table, n_features,
        got_w, got_ws, got_sb, got_desc,
    )

    var bad = 0
    if got_w[0] != UInt32(want.feature_id):
        print("     ", label, "feature: got", got_w[0], "want", want.feature_id)
        bad += 1
    if got_w[1] != UInt32(want.bin_id):
        print("     ", label, "bin: got", got_w[1], "want", want.bin_id)
        bad += 1
    if got_ws[0].to_bits() != want.score.to_bits():
        print("     ", label, "score: got", got_ws[0], "want", want.score)
        bad += 1
    if got_ws[1].to_bits() != want.gain.to_bits():
        print("     ", label, "gain: got", got_ws[1], "want", want.gain)
        bad += 1
    if got_sb[0].to_bits() != want.score.to_bits():
        print(
            "     ", label, "score_before: got", got_sb[0], "want",
            want.score, "(a dead pack write leaves the -777.25 poison)",
        )
        bad += 1
    # the descriptor: the winner's table row (clamped for the sentinel)
    # plus the winner's bin
    var fid_c = 0
    if UInt32(want.feature_id) < UInt32(n_features):
        fid_c = Int(UInt32(want.feature_id))
    for k in range(4):
        if got_desc[k] != feat_table[4 * fid_c + k]:
            print(
                "     ", label, "desc word", k, ": got", got_desc[k],
                "want", feat_table[4 * fid_c + k],
            )
            bad += 1
    if got_desc[4] != UInt32(want.bin_id):
        print("     ", label, "desc bin: got", got_desc[4], "want", want.bin_id)
        bad += 1

    if bad != 0:
        print("FAIL", label, "--", bad, "fields differ from the host fold")
        failures += 1


def main() raises:
    var ctx = DeviceContext()
    var failures = 0

    comptime N_FEATURES = 24
    var table = List[UInt32]()
    for f in range(N_FEATURES):
        # distinct hashed words per cell so a transposed or off-by-one
        # table read cannot match
        table.append(UInt32(_mix(UInt64(4 * f)) & 0xFFFF))
        table.append(UInt32(_mix(UInt64(4 * f + 1)) & 0xFF))
        table.append(UInt32(_mix(UInt64(4 * f + 2)) & 0x1F))
        table.append(UInt32(f % 2))

    # ---------------------------------------------------------- R1
    # three helpers, hashed scores, no ties anywhere
    var r1 = List[HelperRecords]()
    for h in range(3):
        var hr = HelperRecords()
        for b in range(5 + h):
            var u = _mix(UInt64(1000 * h + b))
            var fid = UInt32(8 * h + b % 8)
            var bin = UInt32(u & 0x3F)
            var gain = Float32(Int(u & 0xFFFFF)) / Float32(-1024.0)
            var score = gain * Float32(0.5) + Float32(Int(u >> 40)) / (
                Float32(65536.0)
            )
            hr.append(fid, bin, score, gain)
        r1.append(hr^)
    _check_case(ctx, r1, table, N_FEATURES, String("R1"), failures)
    if failures == 0:
        print(
            "  ok   R1 -- 3 helpers x 5-7 hashed blocks: device fold =="
            " host fold on every field, pack read back whole"
        )

    # ---------------------------------------------------------- R2
    # the tie table, one case per launch so each rule is the decider
    var before = failures

    # equal gain ACROSS helpers: the EARLIER policy's record must win
    var t1 = List[HelperRecords]()
    var t1a = HelperRecords()
    t1a.append(UInt32(3), UInt32(7), Float32(1.5), Float32(-2.0))
    var t1b = HelperRecords()
    # same gain, LARGER fid: loses the gain tie on fid -- and a reversed
    # helper order would pick it
    t1b.append(UInt32(9), UInt32(2), Float32(1.25), Float32(-2.0))
    t1.append(t1a^)
    t1.append(t1b^)
    _check_case(ctx, t1, table, N_FEATURES, String("R2/helpers"), failures)

    # equal gain, equal fid, equal bin WITHIN a helper: the earlier block
    # (the incumbent) must survive -- distinguishable by SCORE, which the
    # comparator never reads
    var t2 = List[HelperRecords]()
    var t2a = HelperRecords()
    t2a.append(UInt32(5), UInt32(4), Float32(11.0), Float32(-3.0))
    t2a.append(UInt32(5), UInt32(4), Float32(22.0), Float32(-3.0))
    t2.append(t2a^)
    _check_case(ctx, t2, table, N_FEATURES, String("R2/blocks"), failures)

    # gain tie against the SENTINEL: (ui32)-1 must LOSE -- a signed
    # comparison picks it
    var t3 = List[HelperRecords]()
    var t3a = HelperRecords()
    t3a.append(PW_SENTINEL_ID, UInt32(0), Float32(3.4028234663852886e38),
               Float32(3.4028234663852886e38))
    t3a.append(UInt32(17), UInt32(9), Float32(3.4028234663852886e38),
               Float32(3.4028234663852886e38))
    t3.append(t3a^)
    _check_case(ctx, t3, table, N_FEATURES, String("R2/sentinel"), failures)

    # gain and fid equal, bin decides
    var t4 = List[HelperRecords]()
    var t4a = HelperRecords()
    t4a.append(UInt32(6), UInt32(31), Float32(0.5), Float32(-1.0))
    t4a.append(UInt32(6), UInt32(2), Float32(0.75), Float32(-1.0))
    t4.append(t4a^)
    _check_case(ctx, t4, table, N_FEATURES, String("R2/bin"), failures)

    if failures == before:
        print(
            "  ok   R2 -- planted ties: earlier helper, earlier block,"
            " unsigned sentinel, bin tie-break -- all four decided as the"
            " host fold decides them"
        )

    # ---------------------------------------------------------- R3
    before = failures
    var r3 = List[HelperRecords]()
    var r3a = HelperRecords()
    r3a.append(PW_SENTINEL_ID, UInt32(0), Float32(3.4028234663852886e38),
               Float32(3.4028234663852886e38))
    r3.append(r3a^)
    var empty = HelperRecords()
    r3.append(empty^)
    _check_case(ctx, r3, table, N_FEATURES, String("R3"), failures)
    if failures == before:
        print(
            "  ok   R3 -- all-sentinel level: the default record survives"
            " the fold and the pack clamps its table row to feature 0"
        )

    if failures != 0:
        raise Error(String(failures) + " resolve cases FAILED")
    print("pointwise winner resolve: R1-R3 pass")
