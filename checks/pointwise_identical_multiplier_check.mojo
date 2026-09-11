# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gate for DEVIATION 2624: an ordered-tier POINTWISE histogram depends on
neither `sm_count` nor the order in which the device finishes its blocks.

    pixi run check-pointwise-identical-multiplier

WHAT BROKE. `EstimateBlockPerFeatureMultiplier` splits the document axis of
every pointwise histogram launch `M` ways, and at `M > 1` each document
block float-`atomicAdd`s its partial into the same `binSums` cell. Float
addition is not associative, so the cell followed the completion order, and
`M` itself follows `sm_count` (132 on an H100). On the H100, 2026-09-11,
three in-process repeats of one IDENTICAL pointwise fit gave three different
`tree001.depth00.hist.*` hashes on all three policies, and Istella-S 1M
moved its model hash between rounds.

WHY THE EXISTING GATES WERE GREEN. `pointwise_dispatch_check.mojo` plants
INTEGER stats under 2^24, so its partial sums add exactly in any order, and
a fit's tree 0 sees dyadic Logloss gradients for the same reason. The planes
here are real-valued on purpose.

GATES
  G1  host: under the ordered tiers `pw_block_multiplier` is 1 at every
      probed grid, row count and SM count.
  G2  device: `compute_hist2` for the binary, half-byte and one-byte
      policies, a full pass at depth 0 then a partial pass at depth 1, at
      SM counts 1, 16, 132 and 4096 (the last three times), every histogram
      bit-equal to the SM count 1 run.

SABOTAGE (rule 7): `pointwise_doc_split_for` returning True under the
ordered tiers restores the pre-2624 launch and must fail G1 and G2.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, numeric_mode_name
from gbdt.gpu_data.grid_policy import (
    POLICY_BINARY,
    POLICY_HALF_BYTE,
    POLICY_ONE_BYTE,
)
from gbdt.methods.pointwise_kernels import (
    FoldsHistogram,
    compute_hist2,
    folds_histogram_from_folds,
    pw_block_multiplier,
)

#: 400,000 rows, so the unguarded ladder reaches M = 32 at SM count 4096
#: (400000 / 32 > 10000) and M = 1 at SM count 1 on the binary grid.
comptime N_ROWS = 400000

#: the left child of the depth-1 partial pass; smaller than the right, so
#: the drivers walk the left and subtract for the right
comptime N_LEFT = 150000

comptime N_OB = 16   # one-byte features, 4 blocks of 4
comptime N_HB = 24   # half-byte features, 3 blocks of 8
comptime N_B = 64    # binary features, 2 blocks of 32
comptime N_HB_BLOCKS = N_HB // 8
comptime N_B_BLOCKS = N_B // 32
comptime N_COLUMNS = 4 + N_HB_BLOCKS + N_B_BLOCKS
comptime HB_COL0 = 4
comptime B_COL0 = 4 + N_HB_BLOCKS

#: a power of two, with every Int32 partial well inside range:
#: 400000 rows x 1.25 x 1024 < 2^31
comptime FIXED_SCALE = Float32(1024.0)


struct Policy(Movable):
    var policy: Int
    var name: String
    var n_features: Int
    var folds: List[UInt32]
    var offset: List[UInt32]
    var first: List[UInt32]
    var one_hot: List[UInt8]
    var line: Int

    def __init__(
        out self,
        policy: Int,
        name: String,
        var folds: List[UInt32],
        var offset: List[UInt32],
    ):
        self.policy = policy
        self.name = name
        self.n_features = len(folds)
        self.first = List[UInt32]()
        self.one_hot = List[UInt8]()
        var c = UInt32(0)
        for f in range(len(folds)):
            self.first.append(c)
            self.one_hot.append(UInt8(0))
            c += folds[f]
        self.line = Int(c)
        self.folds = folds^
        self.offset = offset^


def build_policies() -> List[Policy]:
    var out = List[Policy]()

    var ob_folds: List[UInt32] = [
        20, 25, 30, 32, 40, 50, 60, 64, 90, 100, 120, 128, 200, 150, 256, 180,
    ]
    var ob_off = List[UInt32]()
    for f in range(N_OB):
        ob_off.append(UInt32((f // 4) * N_ROWS))
    out.append(Policy(POLICY_ONE_BYTE, "one-byte", ob_folds^, ob_off^))

    var hb_folds: List[UInt32] = [
        16, 5, 12, 3, 9, 16, 7, 11, 4, 16, 13, 6, 2, 15, 8, 10,
        16, 9, 3, 14, 7, 5, 11, 16,
    ]
    var hb_off = List[UInt32]()
    for j in range(N_HB):
        hb_off.append(UInt32((HB_COL0 + j // 8) * N_ROWS))
    out.append(Policy(POLICY_HALF_BYTE, "half-byte", hb_folds^, hb_off^))

    var b_folds = List[UInt32]()
    var b_off = List[UInt32]()
    for j in range(N_B):
        b_folds.append(UInt32(1))
        b_off.append(UInt32((B_COL0 + j // 32) * N_ROWS))
    out.append(Policy(POLICY_BINARY, "binary", b_folds^, b_off^))
    return out^


def build_cindex(policies: List[Policy]) -> List[UInt32]:
    var ci = List[UInt32](capacity=N_COLUMNS * N_ROWS)
    for _ in range(N_COLUMNS * N_ROWS):
        ci.append(UInt32(0))
    ref ob = policies[0]
    for g in range(4):
        for r in range(N_ROWS):
            var word = UInt32(0)
            for k in range(4):
                var f = 4 * g + k
                var b = (r * (7 + 3 * f) + 5 * k) % Int(ob.folds[f])
                word |= UInt32(b) << UInt32(24 - 8 * k)
            ci[g * N_ROWS + r] = word
    ref hb = policies[1]
    for g in range(N_HB_BLOCKS):
        for r in range(N_ROWS):
            var word = UInt32(0)
            for k in range(8):
                var j = 8 * g + k
                var b = (r * (11 + 5 * j) + 3 * j) % Int(hb.folds[j])
                word |= UInt32(b) << UInt32(28 - 4 * k)
            ci[(HB_COL0 + g) * N_ROWS + r] = word
    for g in range(N_B_BLOCKS):
        for r in range(N_ROWS):
            var word = UInt32(0)
            for k in range(32):
                var j = 32 * g + k
                var bit = (r * (13 + 2 * j) + j) % 2
                word |= UInt32(bit) << UInt32(31 - k)
            ci[(B_COL0 + g) * N_ROWS + r] = word
    return ci^


def run_policy(
    ctx: DeviceContext,
    p: Policy,
    mut d_ci: DeviceBuffer[DType.uint32],
    mut d_tgt: DeviceBuffer[DType.float32],
    mut d_wt: DeviceBuffer[DType.float32],
    mut d_idx: DeviceBuffer[DType.uint32],
    mut d_parts1: DeviceBuffer[DType.uint32],
    mut d_parts2: DeviceBuffer[DType.uint32],
    sm_count: Int,
) raises -> List[UInt32]:
    """Full pass at depth 0, then the partial pass at depth 1 on the same
    buffer, exactly the searcher's order. Returns the bit patterns of the
    full-pass part and of both depth-1 parts, concatenated."""
    var n = p.n_features
    var d_off = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_first = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_folds = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_oh = ctx.enqueue_create_buffer[DType.uint8](n)
    ctx.enqueue_copy(dst_buf=d_off, src_ptr=p.offset.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_first, src_ptr=p.first.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_folds, src_ptr=p.folds.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_oh, src_ptr=p.one_hot.unsafe_ptr())

    var line2 = p.line * 2
    var fh = FoldsHistogram()
    if p.policy == POLICY_ONE_BYTE:
        fh = folds_histogram_from_folds(p.folds)

    var d_hist = ctx.enqueue_create_buffer[DType.float32](2 * line2)
    var h_hist = ctx.enqueue_create_host_buffer[DType.float32](2 * line2)
    ctx.enqueue_memset(d_hist, Float32(0.0))

    compute_hist2(
        ctx, p.policy,
        d_off.unsafe_ptr(), d_first.unsafe_ptr(), d_folds.unsafe_ptr(),
        d_oh.unsafe_ptr(), n, 0, p.line,
        d_ci.unsafe_ptr(), d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
        d_idx.unsafe_ptr(), N_ROWS, d_parts1.unsafe_ptr(), 1, 1,
        d_hist.unsafe_ptr(), p.line, True, fh.copy(), sm_count, FIXED_SCALE,
    )
    ctx.enqueue_copy(dst_buf=h_hist, src_buf=d_hist)
    ctx.synchronize()
    var bits = List[UInt32]()
    var hp = h_hist.unsafe_ptr().unsafe_bitcast[UInt32]()
    for k in range(line2):
        bits.append(hp.unsafe_load(k))

    compute_hist2(
        ctx, p.policy,
        d_off.unsafe_ptr(), d_first.unsafe_ptr(), d_folds.unsafe_ptr(),
        d_oh.unsafe_ptr(), n, 0, p.line,
        d_ci.unsafe_ptr(), d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
        d_idx.unsafe_ptr(), N_ROWS, d_parts2.unsafe_ptr(), 2, 1,
        d_hist.unsafe_ptr(), p.line, False, fh.copy(), sm_count, FIXED_SCALE,
    )
    ctx.enqueue_copy(dst_buf=h_hist, src_buf=d_hist)
    ctx.synchronize()
    hp = h_hist.unsafe_ptr().unsafe_bitcast[UInt32]()
    for k in range(2 * line2):
        bits.append(hp.unsafe_load(k))
    _ = d_off^
    _ = d_first^
    _ = d_folds^
    _ = d_oh^
    return bits^


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST:
        print(
            "REFUSED: DEVIATION 2624 gates the ordered tiers; this build is",
            numeric_mode_name(),
            "(run with -D MOJOLEARN_NUMERIC_IDENTICAL=1)",
        )
        raise Error("check-pointwise-identical-multiplier needs an ordered tier")

    print("check-pointwise-identical-multiplier, tier", numeric_mode_name())
    var failures = 0

    # ============================================================ G1
    var sms: List[Int] = [1, 16, 132, 4096]
    var sizes: List[Int] = [24000, 400000, 1000000, 2000000]
    var bad1 = 0
    var probes = 0
    for nx in range(1, 9):
        for d in range(7):
            for si in range(len(sizes)):
                for mi in range(len(sms)):
                    var m = pw_block_multiplier(
                        nx, 1 << d, 1, sizes[si], sms[mi]
                    )
                    probes += 1
                    if m != 1:
                        if bad1 < 4:
                            print(
                                "     G1 nx", nx, "ny", 1 << d, "rows",
                                sizes[si], "sm", sms[mi], "-> multiplier", m,
                            )
                        bad1 += 1
    if bad1 != 0:
        print("FAIL G1: --", bad1, "of", probes, "grids split the document axis")
        failures += 1
    else:
        print("  ok   G1 --", probes, "grids, multiplier 1 at every one")

    # ============================================================ G2
    var ctx = DeviceContext()
    var policies = build_policies()
    var cindex = build_cindex(policies)

    var indices = List[UInt32](capacity=N_ROWS)
    var target = List[Float32](capacity=N_ROWS)
    var weight = List[Float32](capacity=N_ROWS)
    for r in range(N_ROWS):
        indices.append(UInt32((r * 2654435761) % N_ROWS))
        # REAL-VALUED planes: integers would add exactly in any order
        var u = Float32((r * 2654435761) % 1000003) / Float32(1000003.0)
        target.append(u - Float32(0.5))
        weight.append(
            Float32(1.0) + Float32((r * 40503) % 997) / Float32(3988.0)
        )

    var d_idx = ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
    var d_tgt = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_wt = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_ci = ctx.enqueue_create_buffer[DType.uint32](N_COLUMNS * N_ROWS)
    ctx.enqueue_copy(dst_buf=d_idx, src_ptr=indices.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_tgt, src_ptr=target.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_wt, src_ptr=weight.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ci, src_ptr=cindex.unsafe_ptr())

    var parts1: List[UInt32] = [UInt32(0), UInt32(N_ROWS)]
    var parts2: List[UInt32] = [
        UInt32(0), UInt32(N_LEFT), UInt32(N_LEFT), UInt32(N_ROWS - N_LEFT),
    ]
    var d_parts1 = ctx.enqueue_create_buffer[DType.uint32](2)
    var d_parts2 = ctx.enqueue_create_buffer[DType.uint32](4)
    ctx.enqueue_copy(dst_buf=d_parts1, src_ptr=parts1.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_parts2, src_ptr=parts2.unsafe_ptr())
    ctx.synchronize()
    _ = indices[0]
    _ = target[0]
    _ = weight[0]
    _ = cindex[0]
    _ = parts1[0]
    _ = parts2[0]

    var runs: List[Int] = [1, 16, 132, 4096, 4096, 4096]
    var bad2 = 0
    var cells = 0
    for pi in range(len(policies)):
        var reference = run_policy(
            ctx, policies[pi], d_ci, d_tgt, d_wt, d_idx, d_parts1, d_parts2,
            runs[0],
        )
        for ri in range(1, len(runs)):
            var got = run_policy(
                ctx, policies[pi], d_ci, d_tgt, d_wt, d_idx, d_parts1,
                d_parts2, runs[ri],
            )
            var wrong = 0
            for k in range(len(reference)):
                if got[k] != reference[k]:
                    wrong += 1
            cells += len(reference)
            if wrong != 0:
                print(
                    "     G2", policies[pi].name, "run", ri, "sm", runs[ri],
                    ":", wrong, "of", len(reference),
                    "cells differ in bits from sm 1",
                )
                bad2 += wrong
    if bad2 != 0:
        print("FAIL G2: --", bad2, "of", cells, "cells moved with sm_count or repeat")
        failures += 1
    else:
        print(
            "  ok   G2 --", cells,
            "cells bit-equal across sm 1/16/132/4096 and three repeats,"
            " three policies, full and partial pass",
        )

    if failures != 0:
        raise Error(
            "check-pointwise-identical-multiplier: " + String(failures)
            + " gate(s) failed"
        )
    print("check-pointwise-identical-multiplier PASS")
