"""Can a shared-memory pointer cross a function boundary? YES.

This probe exists because `PORTING.md` 13 asserted for months that it could
not, and that assertion is why every histogram kernel in
`gbdt/methods/greedy_subsets_searcher/kernel/` carries its own copy of
CatBoost's ONE shared loop -- a duplication that has already produced one
silently-wrong-histogram bug (`PORTING.md` 13's original entry).

The claim was never a language limit. `stack_allocation` yields
`MutUntrackedOrigin`; a callee annotated `MutAnyOrigin` rejects it because
that is a DIFFERENT origin, not a wildcard. Parameterize the origin and it
works.

This is a check rather than a scratch probe because the claim it falsifies
was load-bearing for a design decision, and a claim about the toolchain rots
silently. If Mojo ever does take this away, this goes red and the entry in
`PORTING.md` 13 gets rewritten again -- rather than the tree quietly keeping
a duplication whose justification expired.

The gate is per-slot, not a total: each of the 64 threads adds a DISTINCT
value through the callee, so a callee that wrote to the wrong slot, or wrote
nothing and let the caller's initialisation stand, fails. A gate that summed
the buffer would pass on both.
"""

from std.gpu import thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext


@always_inline
def accumulate_into_shared[
    origin: MutOrigin, //
](
    smem: MutPointer[Float32, origin, address_space = AddressSpace.SHARED],
    slot: Int,
    v: Float32,
):
    """The callee `PORTING.md` 13 said could not be written.

    Note the `[origin: MutOrigin, //]`: that, and nothing else, is the
    difference between this compiling and the error that was read as a
    missing language feature.
    """
    smem.unsafe_store(slot, smem.unsafe_load(slot) + v)


def shared_pointer_kernel(out_buf: MutPointer[Float32, MutAnyOrigin]):
    var smem = stack_allocation[
        64,
        Float32,
        address_space = AddressSpace.SHARED,
    ]()
    var t = Int(thread_idx.x)
    if t < 64:
        # a NON-ZERO initialisation, so a callee that never ran is visible
        smem.unsafe_store(t, -1.0)
    barrier()
    accumulate_into_shared(smem, t % 64, Float32(t + 2))
    barrier()
    if t < 64:
        out_buf.unsafe_store(t, smem.unsafe_load(t))


def main() raises:
    var ctx = DeviceContext()
    var d = ctx.enqueue_create_buffer[DType.float32](64)
    ctx.enqueue_function[shared_pointer_kernel](
        d.unsafe_ptr(), grid_dim=(1, 1, 1), block_dim=(64, 1, 1)
    )
    var h = ctx.enqueue_create_host_buffer[DType.float32](64)
    ctx.enqueue_copy(dst_buf=h, src_buf=d)
    ctx.synchronize()

    var wrong = 0
    for i in range(64):
        # -1 initial, plus (i + 2) added through the callee
        if h[i] != Float32(i + 1):
            wrong += 1
    _ = d^
    if wrong != 0:
        raise Error(
            String(wrong)
            + " of 64 slots wrong: a shared-memory pointer no longer"
            " survives a function boundary, and PORTING.md 13 needs"
            " rewriting again"
        )
    print(
        "shared pointer across a function boundary: WORKS (64/64 slots),"
        " so PORTING.md 13's original blocker does not exist"
    )
