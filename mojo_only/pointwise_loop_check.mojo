"""Gate for `gbdt/methods/kernel/compute_point_hist2_loop.mojo`.

CatBoost's one pointwise histogram loop, and the only thing it actually
promises: **every point in `[offset, offset + dsSize)` reaches `AddPoint`
exactly once, with the bin its index gathers and the target and weight at
its row.**

That promise is carried entirely by the head/tail alignment peel and the
strided body -- three separate index regimes with different quanta per entry
point (32/32, 128/64, 128/128), a `blockIdx.x % BLOCKS_PER_FEATURE` gate on
the peels, and an `(i | 31)` in the blocked-iteration count. Every one of
those is an off-by-one waiting to happen, and none of them is visible in a
fit: a dropped or doubled point moves a leaf value by a fraction and the
model still trains.

WHY THE PLANTS ARE HASHED. A loop that delivered every point to the WRONG
bin would pass any check that compares totals, and this repository has twice
shipped a check that did exactly that ([[uniform-test-data-hides-permutation]]).
So `cindex` scatters rows across bins by a hash, targets and weights are
distinct per row, and the comparison is PER BIN.

GATES:
  L1  `compute_histogram` at n = 1, 2, 4, over a sweep of (offset, dsSize)
      chosen to hit every branch: dsSize shorter than the head, dsSize that
      leaves no tail, offsets at 0 and at every residue class that matters.
  L2  `compute_histogram_2` over the same sweep. Its head quantum is 128 and
      its tail quantum is 64 -- they are NOT the same number, which is the
      kind of asymmetry a port silently symmetrises.
  L3  `compute_histogram_4` over the same sweep, head and tail both 128.
  L4  the three entry points agree with each other per bin. They load 1, 2
      and 4 points per iteration and must sum identically.
  L5  `blocks_per_feature = 4`: the four blocks must PARTITION the points,
      not each take all of them. Summed across blocks, the histogram must
      equal the single-block one exactly.
  L6  the same sweep through an accumulator that takes a `barrier()`
      INSIDE `add_point`, which is what every real CatBoost accumulator
      does (`pointwise_hist2_one_byte_5bit.cu:79`, `:108`, `:147` sync a
      `tiled_partition<8>` there).

      **READ THIS BEFORE TRUSTING L6.** It does NOT gate the
      uniform-iteration requirement, and it was written believing it would.
      Reverting `compute_histogram_2` to CatBoost's per-thread counts --
      which genuinely diverges, 8 iterations on one thread against 7 on
      another -- leaves all 160 cases exact. `mojo_only/
      divergent_barrier_probe.mojo` then showed why: a divergent
      threadgroup barrier does not misbehave on this device at all, not
      even at the 512-thread / 64-row shape `PORTING.md` 11 names as its
      evidence.

      So L1-L6 ALL pass whether the body loop converges the block or not.
      The uniform path is kept because it is correct by specification, not
      because anything here can see it -- and that is stated rather than
      left for a reader to assume otherwise
      ([[mojotrees-verify-reach-not-output]]: a check that cannot tell a
      working change from a no-op is not evidence, and saying so is part of
      the result). `PORTING.md` 92 carries the full account.

SABOTAGES RUN, and what each one moved. One per mechanism, verified by
breaking the loop and watching the gate go red:

  tail peel removed (scalar)      L1 all three n, 835 bins; and L5
  head peel not gated on          L5 only, 12 bins -- the single-block
    `blockIdx.x % BLOCKS_PER_FEATURE`   cases cannot see it, which is
                                        exactly why L5 exists
  `_2`'s tail quantum symmetrised L2 only, 394 bins. Its head is 128 and
    from 64 to 128                      its tail is 64; a port that made
                                        them agree would look tidier and
                                        be wrong

A MECHANISM THIS CHECK DELIBERATELY DOES NOT GATE. Replacing `(i | 31)` with
`i` in the blocked-iteration count changes nothing: all 160 cases stay
bit-identical for every `n`. That is not a hole in the check, it is a fact
about the code -- `blocked * n` is `(iterCount // n) * n <= iterCount` either
way, so both spellings deliver the same points in the same per-lane order.
Their `(i | 31)` is divergence control, not a bounds guard. Recorded here
rather than left silent, because a reader who tries that sabotage and sees
green should find out why before concluding the check is weak.

A CHECK BUG THIS FOUND IN ITSELF, worth keeping. The first version folded
the per-thread tallies onto 32 lane slots. At a 128-thread block that put
four threads on one address doing non-atomic read-modify-write, and every
gate went red against a loop that was correct. The tally is now one per
thread. A racing check does not report no result -- it reports the wrong
culprit.
"""

from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext

from gbdt.methods.kernel.compute_point_hist2_loop import (
    PointHist2,
    compute_histogram,
    compute_histogram_2,
    compute_histogram_4,
)

comptime NBINS = 16
comptime BLOCK = 128
comptime N_ROWS = 5000


struct TallyHist[origin: MutOrigin](PointHist2):
    """The smallest accumulator that can catch a misdelivered point.

    Two planes: the summed target and the summed weight, per bin. Nothing
    about bin packing, shared-memory slicing or writeback -- which is the
    point, because the loop under test must not know about those either.

    `add_point_2` and `add_point_4` are `add_point` twice and four times in
    (x, y, z, w) order, which is the contract `PointHist2` states and which
    is what makes L4 a real gate rather than a tautology.
    """

    var buf: MutPointer[
        Float32, Self.origin, address_space = AddressSpace.SHARED
    ]

    def __init__(
        out self,
        buf: MutPointer[
            Float32, Self.origin, address_space = AddressSpace.SHARED
        ],
    ):
        self.buf = buf

    def add_point(mut self, ci: UInt32, t: Float32, w: Float32):
        var b = Int(ci) % NBINS
        # ONE TALLY PER THREAD, and that is not a detail. The first version
        # of this check folded threads onto 32 lane slots, so with a
        # 128-thread block four threads did a non-atomic read-modify-write
        # on the same address and lost updates -- and every gate went red
        # against a loop that was correct. A check that races is a check
        # that reports the wrong culprit.
        var slot = Int(thread_idx.x) * NBINS * 2 + b * 2
        self.buf.unsafe_store(slot, self.buf.unsafe_load(slot) + t)
        self.buf.unsafe_store(
            slot + 1, self.buf.unsafe_load(slot + 1) + w
        )

    def add_point_2(
        mut self,
        ci: SIMD[DType.uint32, 2],
        t: SIMD[DType.float32, 2],
        w: SIMD[DType.float32, 2],
    ):
        self.add_point(ci[0], t[0], w[0])
        self.add_point(ci[1], t[1], w[1])

    def add_point_4(
        mut self,
        ci: SIMD[DType.uint32, 4],
        t: SIMD[DType.float32, 4],
        w: SIMD[DType.float32, 4],
    ):
        self.add_point(ci[0], t[0], w[0])
        self.add_point(ci[1], t[1], w[1])
        self.add_point(ci[2], t[2], w[2])
        self.add_point(ci[3], t[3], w[3])

    def reduce(mut self):
        barrier()


struct BarrierTallyHist[origin: MutOrigin](PointHist2):
    """`TallyHist` with a `barrier()` inside `add_point`.

    Numerically identical to `TallyHist` -- the barrier changes no sum. Its
    whole job is to make the loop's iteration counts OBSERVABLE: a
    threadgroup barrier reached by different numbers of threads is
    undefined, and this is the only accumulator in the check that can tell
    whether the body loop converged the block or not.

    The barrier goes in `add_point` and NOT in `add_point_2` / `add_point_4`
    around the whole group, because that is where theirs is: one sync
    between the two half-writes of every point.
    """

    var buf: MutPointer[
        Float32, Self.origin, address_space = AddressSpace.SHARED
    ]

    def __init__(
        out self,
        buf: MutPointer[
            Float32, Self.origin, address_space = AddressSpace.SHARED
        ],
    ):
        self.buf = buf

    def add_point(mut self, ci: UInt32, t: Float32, w: Float32):
        var b = Int(ci) % NBINS
        var slot = Int(thread_idx.x) * NBINS * 2 + b * 2
        barrier()
        self.buf.unsafe_store(slot, self.buf.unsafe_load(slot) + t)
        barrier()
        self.buf.unsafe_store(slot + 1, self.buf.unsafe_load(slot + 1) + w)

    def add_point_2(
        mut self,
        ci: SIMD[DType.uint32, 2],
        t: SIMD[DType.float32, 2],
        w: SIMD[DType.float32, 2],
    ):
        self.add_point(ci[0], t[0], w[0])
        self.add_point(ci[1], t[1], w[1])

    def add_point_4(
        mut self,
        ci: SIMD[DType.uint32, 4],
        t: SIMD[DType.float32, 4],
        w: SIMD[DType.float32, 4],
    ):
        self.add_point(ci[0], t[0], w[0])
        self.add_point(ci[1], t[1], w[1])
        self.add_point(ci[2], t[2], w[2])
        self.add_point(ci[3], t[3], w[3])

    def reduce(mut self):
        barrier()


def _barrier_tally_kernel[variant: Int](
    indices: MutPointer[UInt32, MutAnyOrigin],
    offset: Int32,
    ds_size: Int32,
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    cindex: MutPointer[UInt32, MutAnyOrigin],
    out_buf: MutPointer[Float32, MutAnyOrigin],
):
    var smem = stack_allocation[
        BLOCK * NBINS * 2, Float32, address_space = AddressSpace.SHARED
    ]()
    var t = Int(thread_idx.x)
    for k in range(NBINS * 2):
        smem.unsafe_store(t * NBINS * 2 + k, 0.0)
    barrier()

    var hist = BarrierTallyHist(smem)

    comptime if variant == 1:
        compute_histogram[BLOCK, 1, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    elif variant == 4:
        compute_histogram[BLOCK, 1, 4, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    elif variant == 12:
        compute_histogram_2[BLOCK, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    else:
        compute_histogram_4[BLOCK, 1, 1, 1](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )

    barrier()
    if t < NBINS:
        var st = Float32(0.0)
        var sw = Float32(0.0)
        for lane in range(BLOCK):
            st += smem.unsafe_load(lane * NBINS * 2 + t * 2)
            sw += smem.unsafe_load(lane * NBINS * 2 + t * 2 + 1)
        var at = t * 2
        out_buf.unsafe_store(at, st)
        out_buf.unsafe_store(at + 1, sw)


def _tally_kernel[variant: Int, blocks_per_feature: Int](
    indices: MutPointer[UInt32, MutAnyOrigin],
    offset: Int32,
    ds_size: Int32,
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    cindex: MutPointer[UInt32, MutAnyOrigin],
    out_buf: MutPointer[Float32, MutAnyOrigin],
):
    var smem = stack_allocation[
        BLOCK * NBINS * 2, Float32, address_space = AddressSpace.SHARED
    ]()
    var t = Int(thread_idx.x)
    for k in range(NBINS * 2):
        smem.unsafe_store(t * NBINS * 2 + k, 0.0)
    barrier()

    var hist = TallyHist(smem)

    comptime if variant == 1:
        compute_histogram[BLOCK, 1, 1, 1, blocks_per_feature](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    elif variant == 2:
        compute_histogram[BLOCK, 1, 2, 1, blocks_per_feature](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    elif variant == 4:
        compute_histogram[BLOCK, 1, 4, 1, blocks_per_feature](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    elif variant == 12:
        compute_histogram_2[BLOCK, 1, 1, blocks_per_feature](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )
    else:
        compute_histogram_4[BLOCK, 1, 1, blocks_per_feature](
            hist, indices, UInt32(offset), UInt32(ds_size), target, weight,
            cindex,
        )

    barrier()
    # fold the 32 per-thread tallies into the block's output slice
    if t < NBINS:
        var st = Float32(0.0)
        var sw = Float32(0.0)
        for lane in range(BLOCK):
            st += smem.unsafe_load(lane * NBINS * 2 + t * 2)
            sw += smem.unsafe_load(lane * NBINS * 2 + t * 2 + 1)
        var at = Int(block_idx.x) * NBINS * 2 + t * 2
        out_buf.unsafe_store(at, st)
        out_buf.unsafe_store(at + 1, sw)


def main() raises:
    var ctx = DeviceContext()
    var failures = 0

    # ---- the pool: hashed, distinct per row, scattered across bins ----
    var indices_h = List[UInt32]()
    var target_h = List[Float32]()
    var weight_h = List[Float32]()
    var cindex_h = List[UInt32]()
    for r in range(N_ROWS):
        # a non-identity gather, so reading the row instead of the index fails
        indices_h.append(UInt32((r * 2654435761) % N_ROWS))
        target_h.append(Float32((r * 37) % 101 + 1))
        weight_h.append(Float32((r * 53) % 97 + 1))
        cindex_h.append(UInt32((r * 7919) % NBINS))

    var d_idx = ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
    var d_tgt = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_wt = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_ci = ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
    ctx.enqueue_copy(dst_buf=d_idx, src_ptr=indices_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_tgt, src_ptr=target_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_wt, src_ptr=weight_h.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ci, src_ptr=cindex_h.unsafe_ptr())

    # branch-hitting sweep: unaligned offsets, sizes below the head, sizes
    # with an empty tail, sizes that are exactly one stripe
    var offsets: List[Int] = [0, 1, 7, 31, 32, 63, 100, 127, 128, 129]
    var sizes: List[Int] = [0, 1, 5, 31, 32, 33, 63, 64, 127, 128, 129, 255,
                            256, 500, 1024, 2000]

    var variants: List[Int] = [1, 2, 4, 12, 14]
    var names: List[String] = [
        String("compute_histogram n=1"),
        String("compute_histogram n=2"),
        String("compute_histogram n=4"),
        String("compute_histogram_2"),
        String("compute_histogram_4"),
    ]

    var d_out = ctx.enqueue_create_buffer[DType.float32](4 * NBINS * 2)
    var h_out = ctx.enqueue_create_host_buffer[DType.float32](4 * NBINS * 2)

    for vi in range(len(variants)):
        var v = variants[vi]
        var bad = 0
        var cases = 0
        for oi in range(len(offsets)):
            for si in range(len(sizes)):
                var off = offsets[oi]
                var n = sizes[si]
                if off + n > N_ROWS:
                    continue
                cases += 1

                # host answer, per bin
                var want_t = List[Float32]()
                var want_w = List[Float32]()
                for _ in range(NBINS):
                    want_t.append(0.0)
                    want_w.append(0.0)
                for r in range(off, off + n):
                    var b = Int(cindex_h[Int(indices_h[r])]) % NBINS
                    want_t[b] += target_h[r]
                    want_w[b] += weight_h[r]

                ctx.enqueue_memset(d_out, Float32(0.0))
                if v == 1:
                    ctx.enqueue_function[_tally_kernel[1, 1]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1), block_dim=(BLOCK, 1, 1),
                    )
                elif v == 2:
                    ctx.enqueue_function[_tally_kernel[2, 1]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1), block_dim=(BLOCK, 1, 1),
                    )
                elif v == 4:
                    ctx.enqueue_function[_tally_kernel[4, 1]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1), block_dim=(BLOCK, 1, 1),
                    )
                elif v == 12:
                    ctx.enqueue_function[_tally_kernel[12, 1]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1), block_dim=(BLOCK, 1, 1),
                    )
                else:
                    ctx.enqueue_function[_tally_kernel[14, 1]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1), block_dim=(BLOCK, 1, 1),
                    )
                ctx.enqueue_copy(dst_buf=h_out, src_buf=d_out)
                ctx.synchronize()

                for b in range(NBINS):
                    if h_out[b * 2] != want_t[b] or h_out[b * 2 + 1] != want_w[
                        b
                    ]:
                        if bad < 4:
                            print(
                                "  ",
                                names[vi],
                                "offset",
                                off,
                                "size",
                                n,
                                "bin",
                                b,
                                ": got",
                                h_out[b * 2],
                                h_out[b * 2 + 1],
                                "want",
                                want_t[b],
                                want_w[b],
                            )
                        bad += 1
        if bad != 0:
            print("FAIL:", names[vi], "--", bad, "wrong bins")
            failures += 1
        else:
            print("  ok  ", names[vi], "--", cases, "cases, all bins exact")

    # ---------------------------------------------------------------- L5
    # blocks_per_feature = 4 must PARTITION the points
    var off5 = 3
    var n5 = 4000
    var want_t5 = List[Float32]()
    var want_w5 = List[Float32]()
    for _ in range(NBINS):
        want_t5.append(0.0)
        want_w5.append(0.0)
    for r in range(off5, off5 + n5):
        var b = Int(cindex_h[Int(indices_h[r])]) % NBINS
        want_t5[b] += target_h[r]
        want_w5[b] += weight_h[r]

    ctx.enqueue_memset(d_out, Float32(0.0))
    ctx.enqueue_function[_tally_kernel[1, 4]](
        d_idx.unsafe_ptr(), Int32(off5), Int32(n5),
        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
        grid_dim=(4, 1, 1), block_dim=(BLOCK, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=h_out, src_buf=d_out)
    ctx.synchronize()

    var bad5 = 0
    for b in range(NBINS):
        var st = Float32(0.0)
        var sw = Float32(0.0)
        for blk in range(4):
            st += h_out[blk * NBINS * 2 + b * 2]
            sw += h_out[blk * NBINS * 2 + b * 2 + 1]
        if st != want_t5[b] or sw != want_w5[b]:
            if bad5 < 4:
                print(
                    "   L5 bin", b, ": got", st, sw, "want", want_t5[b],
                    want_w5[b],
                )
            bad5 += 1
    if bad5 != 0:
        print(
            "FAIL: L5 -- the 4 blocks do not partition the points (",
            bad5,
            "bins wrong). 4x the weight means BLOCKS_PER_FEATURE is ignored;"
            " a quadrupled head and tail means the peel is not gated on"
            " blockIdx.x % BLOCKS_PER_FEATURE.",
        )
        failures += 1
    else:
        print("  ok   L5 -- 4 blocks partition the points exactly")

    # ---------------------------------------------------------------- L6
    # the same sweep through an accumulator that barriers inside add_point
    var bvariants: List[Int] = [1, 4, 12, 14]
    var bnames: List[String] = [
        String("compute_histogram n=1"),
        String("compute_histogram n=4"),
        String("compute_histogram_2"),
        String("compute_histogram_4"),
    ]
    for vi in range(len(bvariants)):
        var v = bvariants[vi]
        var bad = 0
        var cases = 0
        for oi in range(len(offsets)):
            for si in range(len(sizes)):
                var off = offsets[oi]
                var n = sizes[si]
                if off + n > N_ROWS:
                    continue
                cases += 1
                var want_t = List[Float32]()
                var want_w = List[Float32]()
                for _ in range(NBINS):
                    want_t.append(0.0)
                    want_w.append(0.0)
                for r in range(off, off + n):
                    var b = Int(cindex_h[Int(indices_h[r])]) % NBINS
                    want_t[b] += target_h[r]
                    want_w[b] += weight_h[r]

                ctx.enqueue_memset(d_out, Float32(0.0))
                if v == 1:
                    ctx.enqueue_function[_barrier_tally_kernel[1]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1), block_dim=(BLOCK, 1, 1),
                    )
                elif v == 4:
                    ctx.enqueue_function[_barrier_tally_kernel[4]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1), block_dim=(BLOCK, 1, 1),
                    )
                elif v == 12:
                    ctx.enqueue_function[_barrier_tally_kernel[12]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1), block_dim=(BLOCK, 1, 1),
                    )
                else:
                    ctx.enqueue_function[_barrier_tally_kernel[14]](
                        d_idx.unsafe_ptr(), Int32(off), Int32(n),
                        d_tgt.unsafe_ptr(), d_wt.unsafe_ptr(),
                        d_ci.unsafe_ptr(), d_out.unsafe_ptr(),
                        grid_dim=(1, 1, 1), block_dim=(BLOCK, 1, 1),
                    )
                ctx.enqueue_copy(dst_buf=h_out, src_buf=d_out)
                ctx.synchronize()
                for b in range(NBINS):
                    if h_out[b * 2] != want_t[b] or h_out[
                        b * 2 + 1
                    ] != want_w[b]:
                        if bad < 3:
                            print(
                                "   L6", bnames[vi], "offset", off, "size",
                                n, "bin", b, ": got", h_out[b * 2],
                                "want", want_t[b],
                            )
                        bad += 1
        if bad != 0:
            print(
                "FAIL: L6", bnames[vi], "--", bad,
                "wrong bins with a barrier inside add_point. The body loop"
                " is not converging the block, so the barrier is divergent"
                " (PORTING.md 11).",
            )
            failures += 1
        else:
            print(
                "  ok   L6", bnames[vi], "--", cases,
                "cases exact with a barrier inside add_point",
            )

    _ = d_idx^
    _ = d_tgt^
    _ = d_wt^
    _ = d_ci^
    _ = d_out^

    if failures != 0:
        raise Error(String(failures) + " gate(s) failed")
    print("pointwise loop: L1-L5 pass")
