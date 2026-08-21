"""What atomic widths does this device actually have? Probe first, design after.

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
order threads arrive, and `Split::update` (`split.cuh:142-191`) then breaks
every tie on a total order. No float64, no dither, no fixed point -- IF the
width exists.

WHAT THIS PROBE FOUND, 2026-08-21, on the M4 (Apple column)
------------------------------------------------------------
**A 64-bit integer `atomicAdd` does not exist, and the toolchain says so at
COMPILE time rather than at run time:**

    error: Atomic operation is not supported for this type on Apple GPU
    error: failed to legalize operation 'pop.atomic.rmw' that was
           explicitly marked illegal

That is the good failure mode, and it is worth naming as such. The feared
mode was a silent drop -- an unsupported atomic that compiles and loses
writes reports a small, plausible histogram, and no oracle catches a
histogram that is merely too small in a way that looks like a sampling
difference. Instead the build refuses.

Because it refuses at COMPILE time, the 64-bit arms below cannot simply be
run and reported as failures: their mere presence would stop this file from
building on Apple. They are therefore behind
`ensemble/mojo_only/atomic_matrix.column_has_64bit_int_atomics`, which is a
kernel-matrix ROW and not an inline `if apple` -- the same discipline every
other vendor divergence in this repository follows. On a column whose row
says True the arms compile and run; on Apple they are elided and the probe
prints why.

The resolution is EXACT rather than approximate, and the argument is in
`atomic_matrix.bin_counter_is_exact_at_32_bits`: a bin count is bounded by
`n_sampled_rows`, which is `IdxT` = `int` in every cuML RF instantiation,
so it cannot exceed 2^31-1 and cannot overflow a UInt32. Their 64-bit width
buys headroom their own index type never lets them use.

WHAT THIS PROBE HAS TO GET RIGHT
---------------------------------
A wrong answer here is worse than no answer. So every arm plants CONTENTION
deliberately -- every thread in the grid adds to the SAME cell -- and
compares against an exact analytic total. A mechanism that half-works fails
this; a `+=` the compiler turned into a non-atomic read-modify-write fails
this. Each arm also runs a `values-are-distinct` cell where every thread
adds a HASHED value, so a kernel that adds the right COUNT of wrong things
is separated from one that adds the right things -- per STANDING_ORDERS
rule 8, uniform addends verify the total and nothing about placement.

And the probe sabotages its own comparison at the end, because a check that
cannot fail reports OK for a kernel that did nothing.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from max.gpu.host.device_attribute import DeviceAttribute
from mojo_only.kernel_matrix import (
    TARGET_COLUMN,
    column_name,
    column_shared_limit,
)
from ensemble.mojo_only.atomic_matrix import (
    column_has_64bit_int_atomics,
    column_has_float32_atomics,
)

comptime BLOCK = 256
comptime BLOCKS = 64
comptime THREADS = BLOCK * BLOCKS
comptime HAS_U64 = column_has_64bit_int_atomics(TARGET_COLUMN)
comptime HAS_F32 = column_has_float32_atomics(TARGET_COLUMN)


@always_inline
def _hashed(tid: Int) -> UInt64:
    """A scattered addend, so an arm that adds the right number of wrong
    values is distinguishable from one that adds the right values."""
    return (UInt64(tid) * 0x9E3779B97F4A7C15) >> 40


def _global_u32_kernel(out_buf: MutPointer[UInt32, MutAnyOrigin]):
    """32-bit integer atomicAdd in GLOBAL memory: the width `ensemble/`
    actually ships, mirroring `bins.cuh:31` narrowed per DEVIATION 101."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    _ = Atomic.fetch_add(out_buf.unsafe_offset(0), UInt32(1))
    _ = Atomic.fetch_add(
        out_buf.unsafe_offset(1), UInt32(Int(_hashed(tid) & 0xFFFFFFFF))
    )


def _shared_u32_kernel(out_buf: MutPointer[UInt32, MutAnyOrigin]):
    """The shape their DEFAULT histogram arm uses: accumulate into shared
    memory under contention, then flush to global with a second atomic
    (`builder_kernels_impl.cuh:326-333` zeroes it, `:345-350` flushes it).

    This is the arm that matters most. Their fast path is the shared one;
    global is the fallback taken only when the histogram will not fit
    (`builder.cuh:545`). A probe that only tested global memory would be
    testing the branch cuML tries hardest not to take.
    """
    var scratch = stack_allocation[
        2, Scalar[DType.uint32], address_space = AddressSpace.SHARED
    ]()
    if thread_idx.x == 0:
        scratch[unsafe_offset=0] = UInt32(0)
        scratch[unsafe_offset=1] = UInt32(0)
    barrier()
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    _ = Atomic.fetch_add(scratch.unsafe_offset(0), UInt32(1))
    _ = Atomic.fetch_add(
        scratch.unsafe_offset(1), UInt32(Int(_hashed(tid) & 0xFFFFFFFF))
    )
    barrier()
    if thread_idx.x == 0:
        _ = Atomic.fetch_add(out_buf.unsafe_offset(0), scratch[unsafe_offset=0])
        _ = Atomic.fetch_add(out_buf.unsafe_offset(1), scratch[unsafe_offset=1])


def _global_f32_kernel(out_buf: MutPointer[Float32, MutAnyOrigin]):
    """Float32 atomicAdd, which the regression bins would need if they
    were not going to fixed point. Recorded as AVAILABILITY only: float
    addition is not associative, so this arm existing does not make a
    reproducible histogram out of it."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    _ = Atomic.fetch_add(out_buf.unsafe_offset(0), Float32(1.0))
    _ = Atomic.fetch_add(
        out_buf.unsafe_offset(1), Float32(Int(_hashed(tid) & 0xFFFF))
    )


def _global_u64_kernel(out_buf: MutPointer[UInt64, MutAnyOrigin]):
    """cuML's own width, transcribed (`bins.cuh:31`).

    The BODY is behind the `atomic_matrix` row, not the definition:
    Mojo 1.0 rejects `comptime if` at module scope ("'comptime if' must be
    contained in a function"), so a column that lacks the width cannot
    simply elide the whole `def`. Guarding the body instead leaves an empty
    kernel on those columns -- which never runs, because `main`'s enqueue is
    behind the same row -- and keeps the source single and GPU-agnostic.
    """
    comptime if HAS_U64:
        var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
        _ = Atomic.fetch_add(out_buf.unsafe_offset(0), UInt64(1))
        _ = Atomic.fetch_add(out_buf.unsafe_offset(1), _hashed(tid))
    else:
        _ = out_buf


def probe_launch_bounds(ctx: DeviceContext) raises -> Int:
    """Do the block sizes this lane launches actually fit the device, and
    is the shared-memory limit knowable at all?

    TWO SEPARATE QUESTIONS, and they get opposite answers here.

    THREADS PER BLOCK IS QUERYABLE, so it is queried. This lane launches
    at three widths -- 1024 for `computeQuantilesBatchedKernel`
    (`quantiles.cuh:271`, `min(1024, max_n_bins)`), 512 for the segmented
    sort's blocks, and TPB_DEFAULT 128 everywhere else -- and none of them
    was ever checked against the device. cuML gets this for free because
    1024 is CUDA's architectural maximum; on Metal it is a per-device
    limit and a per-KERNEL one, and a kernel that exceeds it fails at
    dispatch rather than at compile.

    SHARED MEMORY PER BLOCK IS NOT QUERYABLE. `MAX_SHARED_MEMORY_PER_
    MULTIPROCESSOR` raises "not supported for Apple GPU" on this device,
    measured here. That is WHY `shared_memory_config` reads a kernel
    matrix row instead of `handle.get_device_properties().sharedMemPerBlock`
    the way `builder.cuh:539` does -- not a shortcut past a device query,
    but the absence of one. This arm records the absence, so the next
    reader does not have to rediscover it before deciding the matrix row
    is a cop-out.
    """
    print()
    print("LAUNCH BOUNDS --")
    var wrong = 0

    var max_threads = Int(
        ctx.get_attribute(DeviceAttribute.MAX_THREADS_PER_BLOCK)
    )
    print("    MAX_THREADS_PER_BLOCK =", max_threads)
    var widths = [1024, 512, 128, 256]
    var names = [
        String("computeQuantilesBatchedKernel cap"),
        String("segmented sort SORT_BLOCK"),
        String("TPB_DEFAULT"),
        String("sample_features / mask kernels"),
    ]
    for i in range(len(widths)):
        var ok = widths[i] <= max_threads
        print("      ", names[i], widths[i], "fits" if ok else "*** EXCEEDS ***")
        if not ok:
            wrong += 1

    var shared_queryable = True
    try:
        _ = ctx.get_attribute(
            DeviceAttribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        )
    except:
        shared_queryable = False
    print(
        "    shared memory per block queryable:",
        shared_queryable,
        "-- kernel matrix row says",
        column_shared_limit(TARGET_COLUMN),
        "bytes",
    )
    if shared_queryable:
        print(
            "      NOTE: it is queryable now. `shared_memory_config`'s"
            " docstring says it is not, and that claim must be re-read."
        )

    if wrong == 0:
        print("  launch bounds OK: every width this lane launches fits")
    return wrong


def main() raises:
    var ctx = DeviceContext()

    # The analytic expectation, computed on the host by the same rule.
    var want_count = UInt64(THREADS)
    var want_hash = UInt64(0)
    var want_hash16 = Float32(0.0)
    for tid in range(THREADS):
        want_hash += _hashed(tid)
        want_hash16 += Float32(Int(_hashed(tid) & 0xFFFF))

    var lb_wrong = probe_launch_bounds(ctx)
    print()
    print("atomic width probe -- column:", column_name(TARGET_COLUMN))
    print(" ", THREADS, "threads all hitting one cell, maximum contention")
    print("  analytic count =", want_count, " analytic hashed sum =", want_hash)

    var failures = 0

    # --- arm 1: 32-bit integer atomicAdd, GLOBAL. The shipped width. -----
    var g32 = ctx.enqueue_create_buffer[DType.uint32](2)
    ctx.enqueue_memset(g32, UInt32(0))
    ctx.enqueue_function[_global_u32_kernel](
        g32.unsafe_ptr(), grid_dim=BLOCKS, block_dim=BLOCK
    )
    var hg32 = ctx.enqueue_create_host_buffer[DType.uint32](2)
    ctx.enqueue_copy(dst_buf=hg32, src_buf=g32)
    ctx.synchronize()
    var want_h32 = UInt32(Int(want_hash & 0xFFFFFFFF))
    if (
        UInt64(Int(hg32.unsafe_ptr().unsafe_load(0))) == want_count
        and hg32.unsafe_ptr().unsafe_load(1) == want_h32
    ):
        print("  GLOBAL uint32 atomicAdd: OK  (count and hashed sum exact)")
    else:
        failures += 1
        print(
            "  GLOBAL uint32 atomicAdd: FAIL  count",
            hg32.unsafe_ptr().unsafe_load(0), "want", want_count,
            " hashed", hg32.unsafe_ptr().unsafe_load(1), "want", want_h32,
        )

    # --- arm 2: 32-bit integer atomicAdd, SHARED. Their FAST path. -------
    var s32 = ctx.enqueue_create_buffer[DType.uint32](2)
    ctx.enqueue_memset(s32, UInt32(0))
    ctx.enqueue_function[_shared_u32_kernel](
        s32.unsafe_ptr(), grid_dim=BLOCKS, block_dim=BLOCK
    )
    var hs32 = ctx.enqueue_create_host_buffer[DType.uint32](2)
    ctx.enqueue_copy(dst_buf=hs32, src_buf=s32)
    ctx.synchronize()
    if (
        UInt64(Int(hs32.unsafe_ptr().unsafe_load(0))) == want_count
        and hs32.unsafe_ptr().unsafe_load(1) == want_h32
    ):
        print(
            "  SHARED uint32 atomicAdd: OK  -- their default histogram arm"
            " is expressible"
        )
    else:
        failures += 1
        print(
            "  SHARED uint32 atomicAdd: FAIL  count",
            hs32.unsafe_ptr().unsafe_load(0), "want", want_count,
            " hashed", hs32.unsafe_ptr().unsafe_load(1), "want", want_h32,
        )

    # --- arm 3: float32 atomicAdd. Availability only, never permission. --
    comptime if HAS_F32:
        var gf = ctx.enqueue_create_buffer[DType.float32](2)
        ctx.enqueue_memset(gf, Float32(0.0))
        ctx.enqueue_function[_global_f32_kernel](
            gf.unsafe_ptr(), grid_dim=BLOCKS, block_dim=BLOCK
        )
        var hgf = ctx.enqueue_create_host_buffer[DType.float32](2)
        ctx.enqueue_copy(dst_buf=hgf, src_buf=gf)
        ctx.synchronize()
        var got_fc = hgf.unsafe_ptr().unsafe_load(0)
        if got_fc == Float32(Int(THREADS)):
            print(
                "  GLOBAL float32 atomicAdd: OK  (count exact at",
                got_fc,
                "). AVAILABILITY ONLY: float addition is not associative,"
                " so this does not make a reproducible histogram.",
            )
        else:
            failures += 1
            print(
                "  GLOBAL float32 atomicAdd: FAIL  count", got_fc,
                "want", Float32(Int(THREADS)),
            )
        _ = want_hash16
    else:
        print(
            "  GLOBAL float32 atomicAdd: not compiled -- atomic_matrix row"
            " says this column lacks it"
        )

    # --- arm 4: cuML's own 64-bit width, where the row says it exists ----
    comptime if HAS_U64:
        var g64 = ctx.enqueue_create_buffer[DType.uint64](2)
        ctx.enqueue_memset(g64, UInt64(0))
        ctx.enqueue_function[_global_u64_kernel](
            g64.unsafe_ptr(), grid_dim=BLOCKS, block_dim=BLOCK
        )
        var hg64 = ctx.enqueue_create_host_buffer[DType.uint64](2)
        ctx.enqueue_copy(dst_buf=hg64, src_buf=g64)
        ctx.synchronize()
        if (
            hg64.unsafe_ptr().unsafe_load(0) == want_count
            and hg64.unsafe_ptr().unsafe_load(1) == want_hash
        ):
            print("  GLOBAL uint64 atomicAdd: OK  (cuML's own bins.cuh:31 width)")
        else:
            failures += 1
            print("  GLOBAL uint64 atomicAdd: FAIL")
    else:
        print(
            "  GLOBAL uint64 atomicAdd: NOT AVAILABLE on this column, and"
            " it is a COMPILE error rather than a silent drop --"
        )
        print(
            "      'Atomic operation is not supported for this type on"
            " Apple GPU'. This is why ensemble/ narrows cuML's BinCountT"
            " to 32 bits (DEVIATION 101), which is EXACT because a bin"
            " count is bounded by n_sampled_rows and IdxT is int."
        )

    # --- the sabotage: prove the comparison above can FAIL ---------------
    # A probe whose check cannot fail reports OK for a kernel that did
    # nothing. Re-run arm 1 WITHOUT zeroing: the buffer starts at the
    # previous total, so the analytic expectation must now be wrong.
    ctx.enqueue_function[_global_u32_kernel](
        g32.unsafe_ptr(), grid_dim=BLOCKS, block_dim=BLOCK
    )
    ctx.enqueue_copy(dst_buf=hg32, src_buf=g32)
    ctx.synchronize()
    var sab_c = UInt64(Int(hg32.unsafe_ptr().unsafe_load(0)))
    if sab_c == want_count:
        failures += 1
        print(
            "  SABOTAGE: FAIL -- a second un-zeroed launch still read",
            sab_c, "so this comparison cannot see a kernel that did nothing",
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
