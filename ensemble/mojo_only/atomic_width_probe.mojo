"""Does this device have a 64-bit INTEGER atomicAdd? Ten lines, then a fact.

    pixi run mojo run -I . ensemble/mojo_only/atomic_width_probe.mojo

STANDING_ORDERS rule 9: every capability denial gets a probe before it
constrains a design. This one is load-bearing rather than cautionary.

cuML's `ClassificationBin` (`bins.cuh:14, 17-18`) is

    using BinCountT = unsigned long long int;
    static_assert(sizeof(BinCountT) == 8, "BinCountT must be 64 bits");
    struct ClassificationBin { BinCountT count; ... }

and its `AtomicAdd` (`bins.cuh:29-32`) is a single `atomicAdd` on that
64-bit unsigned. That integer atomic is the entire reason the unweighted
classification path can be bit-identical across Metal, CUDA and HIP:
integer addition is associative, so the histogram does not depend on the
order threads arrive, and `Split::update` (`split.cuh:142-191`) then
breaks every tie on a total order (gain, then colid, then quesval). No
float64, no dither, no fixed point -- IF the width exists.

So there are exactly three outcomes and each one writes a different
`bins.mojo`:

  1. 64-bit integer atomicAdd works in GLOBAL and SHARED memory.
     Transcribe `bins.cuh` verbatim; no deviation to record.
  2. It works in global but not shared. Their kernel needs both: the
     shared-histogram arm is the default and only falls back to global
     when the histogram will not fit
     (`builder_kernels_impl.cuh:322-333`, `builder.cuh:545`).
  3. Neither. Fall back to 32-bit counts.

Outcome 3 is EXACT, not approximate, and that is worth stating before
the numbers arrive: a bin count is bounded by `n_sampled_rows`, which is
`IdxT` = `int` in every cuML RF instantiation, so it cannot exceed
2^31-1 and cannot overflow a UInt32. Their 64-bit width buys headroom
their own index type never lets them use.

WHAT THIS PROBE HAS TO GET RIGHT. A wrong answer here is worse than no
answer, and the failure mode is silence: an unsupported atomic that
compiles and drops writes reports a small number, not an error. So the
probe plants CONTENTION deliberately -- every thread in the grid adds to
the SAME cell -- and compares against an exact analytic total. A
mechanism that half-works fails this; a `+=` that the compiler turned
into a non-atomic read-modify-write fails this. It also runs a
`values-are-distinct` arm where each thread adds a hashed value, so a
kernel that adds the right COUNT of wrong things is separated from one
that adds the right things (STANDING_ORDERS rule 8: uniform addends
verify the total and nothing else).
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

comptime BLOCK = 256
comptime BLOCKS = 64
comptime THREADS = BLOCK * BLOCKS


def _global_u64_kernel(
    out_buf: MutPointer[UInt64, MutAnyOrigin],
):
    """Every thread in the grid adds to cell 0 (uniform) and cell 1
    (hashed). Maximum contention on purpose."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    _ = Atomic.fetch_add(out_buf.unsafe_offset(0), UInt64(1))
    # A hashed addend, so a kernel that adds the right NUMBER of wrong
    # values is distinguishable from one that adds the right values.
    var h = (UInt64(tid) * 0x9E3779B97F4A7C15) >> 40
    _ = Atomic.fetch_add(out_buf.unsafe_offset(1), h)


def _shared_u64_kernel(
    out_buf: MutPointer[UInt64, MutAnyOrigin],
):
    """The shape their default histogram arm uses: accumulate into
    shared memory under contention, then flush to global with a second
    atomic (`builder_kernels_impl.cuh:345-350`)."""
    var scratch = stack_allocation[
        2, Scalar[DType.uint64], address_space = AddressSpace.SHARED
    ]()
    if thread_idx.x == 0:
        scratch[unsafe_offset=0] = UInt64(0)
        scratch[unsafe_offset=1] = UInt64(0)
    barrier()
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    _ = Atomic.fetch_add(scratch.unsafe_offset(0), UInt64(1))
    var h = (UInt64(tid) * 0x9E3779B97F4A7C15) >> 40
    _ = Atomic.fetch_add(scratch.unsafe_offset(1), h)
    barrier()
    if thread_idx.x == 0:
        _ = Atomic.fetch_add(
            out_buf.unsafe_offset(0), scratch[unsafe_offset=0]
        )
        _ = Atomic.fetch_add(
            out_buf.unsafe_offset(1), scratch[unsafe_offset=1]
        )


def _global_u32_kernel(
    out_buf: MutPointer[UInt32, MutAnyOrigin],
):
    """Outcome 3's arm, measured rather than assumed to work."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    _ = Atomic.fetch_add(out_buf.unsafe_offset(0), UInt32(1))
    var h = UInt32(((UInt64(tid) * 0x9E3779B97F4A7C15) >> 40) & 0xFFFFFFFF)
    _ = Atomic.fetch_add(out_buf.unsafe_offset(1), h)


def main() raises:
    var ctx = DeviceContext()

    # The analytic expectation, computed on the host by the same rule.
    var want_count = UInt64(THREADS)
    var want_hash = UInt64(0)
    for tid in range(THREADS):
        want_hash += (UInt64(tid) * 0x9E3779B97F4A7C15) >> 40

    print("atomic width probe:", THREADS, "threads all hitting one cell")
    print("  analytic count =", want_count, " analytic hashed sum =", want_hash)

    var failures = 0

    # --- arm 1: 64-bit integer atomicAdd in GLOBAL memory ---------------
    var g64 = ctx.enqueue_create_buffer[DType.uint64](2)
    ctx.enqueue_memset(g64, UInt64(0))
    ctx.enqueue_function[_global_u64_kernel](
        g64.unsafe_ptr(), grid_dim=BLOCKS, block_dim=BLOCK
    )
    var hg64 = ctx.enqueue_create_host_buffer[DType.uint64](2)
    ctx.enqueue_copy(dst_buf=hg64, src_buf=g64)
    ctx.synchronize()
    var got_c = hg64.unsafe_ptr().unsafe_load(0)
    var got_h = hg64.unsafe_ptr().unsafe_load(1)
    if got_c == want_count and got_h == want_hash:
        print("  GLOBAL  uint64 atomicAdd: OK  (count", got_c, "hashed", got_h, ")")
    else:
        failures += 1
        print(
            "  GLOBAL  uint64 atomicAdd: FAIL  count", got_c, "want", want_count,
            " hashed", got_h, "want", want_hash,
        )

    # --- arm 2: 64-bit integer atomicAdd in SHARED memory ---------------
    var s64 = ctx.enqueue_create_buffer[DType.uint64](2)
    ctx.enqueue_memset(s64, UInt64(0))
    ctx.enqueue_function[_shared_u64_kernel](
        s64.unsafe_ptr(), grid_dim=BLOCKS, block_dim=BLOCK
    )
    var hs64 = ctx.enqueue_create_host_buffer[DType.uint64](2)
    ctx.enqueue_copy(dst_buf=hs64, src_buf=s64)
    ctx.synchronize()
    var got_sc = hs64.unsafe_ptr().unsafe_load(0)
    var got_sh = hs64.unsafe_ptr().unsafe_load(1)
    if got_sc == want_count and got_sh == want_hash:
        print("  SHARED  uint64 atomicAdd: OK  (count", got_sc, "hashed", got_sh, ")")
    else:
        failures += 1
        print(
            "  SHARED  uint64 atomicAdd: FAIL  count", got_sc, "want", want_count,
            " hashed", got_sh, "want", want_hash,
        )

    # --- arm 3: 32-bit integer atomicAdd in GLOBAL memory, the fallback -
    var g32 = ctx.enqueue_create_buffer[DType.uint32](2)
    ctx.enqueue_memset(g32, UInt32(0))
    ctx.enqueue_function[_global_u32_kernel](
        g32.unsafe_ptr(), grid_dim=BLOCKS, block_dim=BLOCK
    )
    var hg32 = ctx.enqueue_create_host_buffer[DType.uint32](2)
    ctx.enqueue_copy(dst_buf=hg32, src_buf=g32)
    ctx.synchronize()
    var got_c32 = UInt64(Int(hg32.unsafe_ptr().unsafe_load(0)))
    var got_h32 = hg32.unsafe_ptr().unsafe_load(1)
    var want_h32 = UInt32(want_hash & 0xFFFFFFFF)
    if got_c32 == want_count and got_h32 == want_h32:
        print("  GLOBAL  uint32 atomicAdd: OK  (count", got_c32, ")")
    else:
        failures += 1
        print(
            "  GLOBAL  uint32 atomicAdd: FAIL  count", got_c32, "want",
            want_count, " hashed", got_h32, "want", want_h32,
        )

    # --- the sabotage: prove the comparison above can FAIL --------------
    # A probe whose check cannot fail reports "OK" for a kernel that does
    # nothing. Re-run arm 1 WITHOUT zeroing, so the buffer starts at the
    # previous total; the analytic expectation must now be wrong.
    ctx.enqueue_function[_global_u64_kernel](
        g64.unsafe_ptr(), grid_dim=BLOCKS, block_dim=BLOCK
    )
    ctx.enqueue_copy(dst_buf=hg64, src_buf=g64)
    ctx.synchronize()
    var sab_c = hg64.unsafe_ptr().unsafe_load(0)
    if sab_c == want_count:
        failures += 1
        print(
            "  SABOTAGE: FAIL -- a second un-zeroed launch still read",
            sab_c,
            "so this check cannot see a kernel that did nothing",
        )
    else:
        print(
            "  SABOTAGE: OK  -- second un-zeroed launch read", sab_c,
            "not", want_count, ", so the comparison has teeth",
        )

    if failures == 0:
        print("atomic width probe: ALL OK")
    else:
        raise Error(
            "atomic width probe: " + String(failures) + " arm(s) failed"
        )
