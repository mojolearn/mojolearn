"""Does a DIVERGENT threadgroup barrier actually fail on this device? NO.

`PORTING.md` 11 is the reason two histogram families in this repository run
one iteration count for the whole block instead of CatBoost's per-thread
counts. Its stated evidence is that a divergent barrier was OBSERVED failing:

    "It is not an edge case: a 64-row partition over a 512-thread block gives
     warp 0 one iteration and warps 1 to 15 zero. The measured symptom was
     every feature's histogram reading 0.0."

**This probe runs exactly that shape and it does not fail.** 512-thread
block, a barrier inside a loop whose count is `(n - tid + block - 1) / block`
so warp 0 runs one iteration and warps 1-15 run zero, and every one of the
512 slots comes back with the value it should, at n = 64, 100, 511, 512, 513,
1000 and 2000. No hang, no zeros.

WHAT THIS DOES AND DOES NOT ESTABLISH -- and the distinction is the point.

  IT DOES establish that on this M4 under this Mojo, a divergent
  `barrier()` in a simple accumulate loop returns correct results at the
  shape item 11 names. So item 11's symptom is NOT reproducible in
  isolation, and a reader should not treat "divergent barrier" as a
  sufficient explanation for a histogram of zeros without re-deriving it.

  IT DOES NOT establish that divergent barriers are safe. They are
  undefined in every relevant programming model -- CUDA's `__syncthreads`
  and Metal's `threadgroup_barrier` both require uniform execution -- and
  "undefined happens to work today" is not a property a port can build on.

  IT DOES NOT establish that item 11's original observation was wrong.
  Twice while writing the pointwise checks, a RACING per-thread tally
  produced exactly the "everything reads zero or a fraction of what it
  should" symptom with nothing wrong in the loop. That is a strictly better
  fit for the reported evidence than a divergent barrier that this probe
  cannot make fail. It is a hypothesis, not a finding, and it is written
  here so someone can test it rather than inherit it.

SO WHY DOES THE UNIFORM PATH STAY? Because it is correct by specification
and costs one predicate per point, and because the alternative is relying on
undefined behaviour whose current benign result this probe would be the
first to notice changing. That is the whole reason this is a permanent check
and not a scratch file: it is the tripwire under a decision that is
currently justified by the spec alone.
"""
from std.gpu import thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext


def divergent_kernel[block: Int](
    n_rows: Int32, out_buf: MutPointer[Float32, MutAnyOrigin]
):
    var smem = stack_allocation[
        block, Float32, address_space = AddressSpace.SHARED
    ]()
    var t = Int(thread_idx.x)
    smem.unsafe_store(t, 0.0)
    barrier()

    # thread t owns rows t, t+block, t+2*block, ...  -> divergent counts
    var iters = 0
    if Int(n_rows) > t:
        iters = (Int(n_rows) - t + block - 1) // block

    for _ in range(iters):
        barrier()          # <-- reached a different number of times per thread
        smem.unsafe_store(t, smem.unsafe_load(t) + 1.0)
        barrier()

    barrier()
    out_buf.unsafe_store(t, smem.unsafe_load(t))


def main() raises:
    var ctx = DeviceContext()
    comptime BLOCK = 512
    var d = ctx.enqueue_create_buffer[DType.float32](BLOCK)
    var h = ctx.enqueue_create_host_buffer[DType.float32](BLOCK)
    var sizes: List[Int] = [64, 100, 511, 512, 513, 1000, 2000]
    var total_wrong = 0
    for si in range(len(sizes)):
        var n = sizes[si]
        ctx.enqueue_memset(d, Float32(0.0))
        ctx.enqueue_function[divergent_kernel[BLOCK]](
            Int32(n), d.unsafe_ptr(),
            grid_dim=(1, 1, 1), block_dim=(BLOCK, 1, 1),
        )
        ctx.enqueue_copy(dst_buf=h, src_buf=d)
        ctx.synchronize()
        # host answer: thread t contributes iters(t) to slot t%64
        var bad = 0
        var want0 = Float32(0.0)
        for t in range(BLOCK):
            var it = 0
            if n > t:
                it = (n - t + BLOCK - 1) // BLOCK
            if t == 0:
                want0 = Float32(it)
            if h[t] != Float32(it):
                bad += 1
        total_wrong += bad
        print(
            "  n =", n, "-> wrong slots:", bad, "of", BLOCK,
            " (slot 0 got", h[0], "want", want0, ")",
        )
    _ = d^
    if total_wrong != 0:
        raise Error(
            "a divergent threadgroup barrier NOW misbehaves on this device"
            " (" + String(total_wrong) + " wrong slots). That is a change"
            " from 2026-08-21, when it did not, and it means PORTING.md 11"
            " and 92 both need rewriting -- and that every kernel relying"
            " on uniform iteration is now relying on it for a real reason"
            " rather than a specification one."
        )
    print(
        "divergent threadgroup barrier: correct in all", BLOCK,
        "slots at every size -- NOT reproducible on this device"
    )
