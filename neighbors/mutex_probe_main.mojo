"""Soundness probe: cuVS's cross-block mutex handoff, on Metal, through Mojo.

NO CUVS COUNTERPART. This is the gate in front of porting the
`gridDim.x > 1` arm of `fusedL2kNN` (`fused_l2_knn.cuh:226-338`), whose
producer/consumer protocol per row-tile is:

    producer (blockIdx.x != 0), `:313-338`:
        tid 0:  while (atomicCAS(mutex, 0, 1) != 0) ;   // acquire
        __threadfence(); __syncthreads();
        write numOfNN (value, key) words to global      // PLAIN stores
        __threadfence(); __syncthreads();
        tid 0:  atomicExch(mutex, -2);                  // release
        __threadfence();

    consumer (blockIdx.x == 0), `:241-303`, gridDim.x - 1 times:
        tid 0:  while (atomicCAS(mutex, -2, -1) != -2) ;
        __threadfence(); __syncthreads();
        read the words                                  // PLAIN loads
        __threadfence(); __syncthreads();
        tid 0:  atomicExch(mutex, 0);
        __threadfence();

WHAT MOJO EXPOSES ON APPLE, ESTABLISHED BY COMPILATION NOT DOCS
----------------------------------------------------------------
- `std.gpu.intrinsics.threadfence` is comptime-asserted
  `"threadfence is only implemented on NVIDIA GPUs"`
  (`stdlib/std/gpu/intrinsics.mojo:790-792` at Mojo 1.0). There is NO
  standalone device-scope fence for Metal.
- `pop.atomic.cmpxchg` is legalized for Apple ONLY as `weak` and ONLY
  relaxed: the backend rejects, by name, a strong exchange ("Apple GPU only
  supports `weak` compare-exchange; AIR exposes no strong compare-exchange
  primitive") and any acquire/acq_rel success ordering ("Apple GPU does not
  support `acquire` atomic ordering"). Both messages are from the Mojo 1.0
  compiler itself, reproduced by this file's history.
- `Atomic.load[Ordering.ACQUIRE]` and `Atomic.store[Ordering.RELEASE]` (and
  SEQUENTIAL for both) DO legalize and run on the Apple target. Verified by
  enqueue, not by host compile.

So the only expressible translation folds their `__threadfence` into the
mutex accesses as a TEST-AND-TEST-AND-SET: spin on an ACQUIRE load until
the mutex shows the awaited state, claim it with a weak RELAXED
compare-exchange (a failed claim just re-enters the spin), and hand it off
with a RELEASE store. The synchronizes-with edge is load-acquire observing
the value that store-release published, which is the C++11 statement of
what their `atomicCAS`-spin plus `__threadfence` accomplishes -- CUDA
defines `__threadfence()` as `cuda::atomic_thread_fence(seq_cst,
thread_scope_device)`. Their `atomicExch` release becomes the RELEASE
store, which is legal because only the mutex HOLDER writes the release
value and theirs discards the returned old value too (`:277`, `:337`).
No ABA hides in the relaxed claim: each state value has exactly one writer
role, and the single consumer is the only block that ever consumes a `-2`.

WHAT THIS PROBE ESTABLISHES OR REFUTES
---------------------------------------
Whether that acquire/release mutex makes one block's PLAIN device-memory
stores visible and ordered to another block's PLAIN loads on the M4, under
real contention, deterministically. Payloads are HASHED per (row, block,
word) -- never uniform, a uniform expected value verified a wrong-reduction
bug here twice -- and the exchange buffer is POISONED before every launch so
a stale or early read cannot masquerade as a pass.

Two sabotage arms prove the probe has teeth:
- SABOTAGE_SKIP: the last producer skips the top half of its words. The
  consumer must see poison. Every iteration must FAIL.
- SABOTAGE_EARLY_RELEASE: the producer releases the mutex BEFORE writing,
  then dawdles. If plain loads can overtake the handoff, this shows it.

CO-RESIDENCY IS ASSUMED, AS THEIRS ASSUMES IT. A spinning producer only
terminates if the consumer runs. cuVS caps the whole grid at
`numSMs * blocksPerSM` (`pairwise_distance_base.cuh:296-322`) so every block
is resident; the port does the same with M4 inputs. The last config here
deliberately OVERSUBSCRIBES 2x to see whether Metal's scheduler still makes
progress; a hang there is a finding about the envelope, not about the
in-envelope protocol.
"""

from std.atomic import Atomic, Ordering
from std.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.sync import barrier

comptime MTX_TPB = 256  # = FKNN_THREADS, the block shape the port launches
comptime MTX_WMAX = 64  # exchange words per (row, producer), = max numOfNN
comptime MTX_YMAX = 16
comptime MTX_POISON_VAL = Float32(-7777.0)
comptime MTX_POISON_KEY = UInt32(0xDEADBEEF)

comptime SABOTAGE_NONE = 0
comptime SABOTAGE_EARLY_RELEASE = 1
comptime SABOTAGE_SKIP = 2


@always_inline
def _mix(u: UInt32) -> UInt32:
    """splitmix32 finalizer; same function runs on host and device."""
    var z = u
    z = (z ^ (z >> 16)) * UInt32(0x7FEB352D)
    z = (z ^ (z >> 15)) * UInt32(0x846CA68B)
    return z ^ (z >> 16)


@always_inline
def _payload(row: Int, bx: Int, w: Int, salt: Int) -> UInt32:
    return _mix(
        UInt32(row) * UInt32(0x9E3779B9)
        ^ UInt32(bx) * UInt32(0x85EBCA6B)
        ^ UInt32(w) * UInt32(0xC2B2AE35)
        ^ UInt32(salt)
    )


def mutex_probe_kernel(
    mutexes: MutPointer[Int32, MutAnyOrigin],
    ex_val: MutPointer[Float32, MutAnyOrigin],
    ex_key: MutPointer[UInt32, MutAnyOrigin],
    out_val: MutPointer[Float32, MutAnyOrigin],
    out_key: MutPointer[UInt32, MutAnyOrigin],
    x_blocks_in: Int32,
    w_in: Int32,
    salt_in: Int32,
    sabotage_in: Int32,
):
    """One row per `block_idx.y`; `block_idx.x == 0` consumes, the rest
    produce, one at a time, through a single exchange buffer -- the exact
    state machine of `fused_l2_knn.cuh:241-338` with `__threadfence` folded
    into the mutex ops as ACQUIRE/RELEASE (see module docstring)."""
    var row = Int(block_idx.y)
    var bx = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var x_blocks = Int(x_blocks_in)
    var w_count = Int(w_in)
    var salt = Int(salt_in)
    var sabotage = Int(sabotage_in)
    var mtx = mutexes.unsafe_offset(row)

    if bx == 0:
        # --- consumer, their `:241-303` ---
        var accf = Float32(0.0)
        var acck = UInt32(0)
        var processed = 0
        while processed < x_blocks - 1:
            if tid == 0:
                # `while (atomicCAS(mutex, -2, -1) != -2);` + the
                # `__threadfence()` acquire that follows, `:251-255`, as
                # acquire-load spin + weak relaxed claim (module docstring).
                while True:
                    if Atomic.load[ordering = Ordering.ACQUIRE](
                        mtx
                    ) != Int32(-2):
                        continue
                    var expected = Int32(-2)
                    if Atomic.compare_exchange[
                        success_ordering = Ordering.RELAXED,
                        failure_ordering = Ordering.RELAXED,
                        weak=True,
                    ](mtx, expected, Int32(-1)):
                        break
            barrier()  # their `__syncthreads()`, `:256`
            var w = tid
            while w < w_count:
                accf += ex_val.unsafe_load(row * MTX_WMAX + w)
                acck += ex_key.unsafe_load(row * MTX_WMAX + w)
                w += MTX_TPB
            barrier()  # their `:279`
            if tid == 0:
                # their `atomicExch(mutex, 0)` + trailing `__threadfence()`,
                # `:281`; RELEASE orders the reads above before the handback.
                Atomic.store[ordering = Ordering.RELEASE](mtx, Int32(0))
            processed += 1
        _ = Atomic.fetch_add(out_val.unsafe_offset(row), accf)
        _ = Atomic.fetch_add(out_key.unsafe_offset(row), acck)
    else:
        # --- producer, their `:313-338` ---
        if tid == 0:
            # `while (atomicCAS(mutex, 0, 1) != 0);` + acquire, `:314-318`,
            # same acquire-load spin + weak relaxed claim.
            while True:
                if Atomic.load[ordering = Ordering.ACQUIRE](mtx) != Int32(0):
                    continue
                var expected = Int32(0)
                if Atomic.compare_exchange[
                    success_ordering = Ordering.RELAXED,
                    failure_ordering = Ordering.RELAXED,
                    weak=True,
                ](mtx, expected, Int32(1)):
                    break
        barrier()
        if sabotage == SABOTAGE_EARLY_RELEASE and tid == 0:
            # The bug arm: hand the buffer over BEFORE filling it, then
            # dawdle so the consumer has every chance to read early.
            Atomic.store[ordering = Ordering.RELEASE](mtx, Int32(-2))
            var dawdle = UInt32(0)
            for i in range(20000):
                dawdle = _mix(dawdle ^ UInt32(i))
            if dawdle == UInt32(0x12345678):  # keep the loop observable
                ex_key.unsafe_store(row * MTX_WMAX, dawdle)
        var w = tid
        while w < w_count:
            var write_it = True
            if sabotage == SABOTAGE_SKIP and bx == x_blocks - 1:
                if w >= w_count // 2:
                    write_it = False
            if write_it:
                var h = _payload(row, bx, w, salt)
                ex_val.unsafe_store(
                    row * MTX_WMAX + w, Float32(Int(h & UInt32(0x3FF)))
                )
                ex_key.unsafe_store(row * MTX_WMAX + w, h)
            w += MTX_TPB
        barrier()  # their `:332`
        if tid == 0 and sabotage != SABOTAGE_EARLY_RELEASE:
            # their `atomicExch(mutex, -2)` + `__threadfence()`, `:336-337`.
            Atomic.store[ordering = Ordering.RELEASE](mtx, Int32(-2))


def _run_config(
    ctx: DeviceContext,
    x_blocks: Int,
    y_blocks: Int,
    w_count: Int,
    iters: Int,
    sabotage: Int,
) raises -> Int:
    """Returns the number of iterations whose result mismatched the host
    oracle. Poisons the exchange buffer before every launch."""
    var mutexes = ctx.enqueue_create_buffer[DType.int32](MTX_YMAX)
    var ex_val = ctx.enqueue_create_buffer[DType.float32](
        MTX_YMAX * MTX_WMAX
    )
    var ex_key = ctx.enqueue_create_buffer[DType.uint32](MTX_YMAX * MTX_WMAX)
    var out_val = ctx.enqueue_create_buffer[DType.float32](MTX_YMAX)
    var out_key = ctx.enqueue_create_buffer[DType.uint32](MTX_YMAX)

    var h_poison_val = ctx.enqueue_create_host_buffer[DType.float32](
        MTX_YMAX * MTX_WMAX
    )
    var h_poison_key = ctx.enqueue_create_host_buffer[DType.uint32](
        MTX_YMAX * MTX_WMAX
    )
    var h_out_val = ctx.enqueue_create_host_buffer[DType.float32](MTX_YMAX)
    var h_out_key = ctx.enqueue_create_host_buffer[DType.uint32](MTX_YMAX)
    ctx.synchronize()
    for i in range(MTX_YMAX * MTX_WMAX):
        h_poison_val.unsafe_ptr().unsafe_store(i, MTX_POISON_VAL)
        h_poison_key.unsafe_ptr().unsafe_store(i, MTX_POISON_KEY)

    var bad_iters = 0
    for it in range(iters):
        var salt = it * 7919 + x_blocks * 131 + w_count
        ctx.enqueue_copy(dst_buf=ex_val, src_ptr=h_poison_val.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=ex_key, src_ptr=h_poison_key.unsafe_ptr())
        ctx.enqueue_memset(mutexes, Int32(0))
        ctx.enqueue_memset(out_val, Float32(0.0))
        ctx.enqueue_memset(out_key, UInt32(0))
        ctx.enqueue_function[mutex_probe_kernel](
            mutexes.unsafe_ptr(),
            ex_val.unsafe_ptr(),
            ex_key.unsafe_ptr(),
            out_val.unsafe_ptr(),
            out_key.unsafe_ptr(),
            Int32(x_blocks),
            Int32(w_count),
            Int32(salt),
            Int32(sabotage),
            grid_dim=(x_blocks, y_blocks, 1),
            block_dim=(MTX_TPB, 1, 1),
        )
        ctx.enqueue_copy(dst_ptr=h_out_val.unsafe_ptr(), src_buf=out_val)
        ctx.enqueue_copy(dst_ptr=h_out_key.unsafe_ptr(), src_buf=out_key)
        ctx.synchronize()

        var iter_ok = True
        for row in range(y_blocks):
            var want_f = Float32(0.0)
            var want_k = UInt32(0)
            for bx in range(1, x_blocks):
                for w in range(w_count):
                    var h = _payload(row, bx, w, salt)
                    want_f += Float32(Int(h & UInt32(0x3FF)))
                    want_k += h
            var got_f = h_out_val.unsafe_ptr().unsafe_load(row)
            var got_k = h_out_key.unsafe_ptr().unsafe_load(row)
            if got_f != want_f or got_k != want_k:
                iter_ok = False
        if not iter_ok:
            bad_iters += 1
    return bad_iters


def main() raises:
    var ctx = DeviceContext()

    # In-envelope configurations. 120 blocks is the computed M4 capacity for
    # this block shape (see launch_config_generator); every config at or
    # under it must be exact on EVERY iteration.
    var total_bad = 0
    var configs_x = [2, 8, 16, 12]
    var configs_y = [2, 8, 4, 10]
    var configs_w = [64, 64, 37, 64]
    var configs_i = [50, 200, 200, 200]
    for c in range(4):
        var bad = _run_config(
            ctx,
            configs_x[c],
            configs_y[c],
            configs_w[c],
            configs_i[c],
            SABOTAGE_NONE,
        )
        print(
            "config grid=(",
            configs_x[c],
            ",",
            configs_y[c],
            ") W=",
            configs_w[c],
            "iters=",
            configs_i[c],
            "bad_iters=",
            bad,
        )
        total_bad += bad

    # Teeth check 1: a producer that skips half its words must be caught
    # every single time, or the verification itself is blind.
    var skip_bad = _run_config(ctx, 8, 8, 64, 25, SABOTAGE_SKIP)
    print("sabotage SKIP: bad_iters=", skip_bad, "of 25 (want 25)")

    # Teeth check 2: release-before-write. Every miss this arm scores is a
    # reordering the real protocol would have prevented.
    var early_bad = _run_config(ctx, 8, 8, 64, 200, SABOTAGE_EARLY_RELEASE)
    print(
        "sabotage EARLY_RELEASE: bad_iters=",
        early_bad,
        "of 200 (>0 shows the probe can see a handoff violation)",
    )

    # Oversubscription arm LAST: 256 blocks vs ~120 resident. A hang here is
    # an envelope finding; the launcher never produces such a grid.
    print("oversubscribed arm: grid=(16,16), 256 blocks, 100 iters...")
    var over_bad = _run_config(ctx, 16, 16, 64, 100, SABOTAGE_NONE)
    print("oversubscribed: bad_iters=", over_bad)
    total_bad += over_bad

    if total_bad != 0:
        raise Error("MUTEX PROBE FAIL: in-envelope mismatches")
    if skip_bad != 25:
        raise Error("MUTEX PROBE FAIL: SKIP sabotage was not fully detected")
    print("MUTEX PROBE PASS")
