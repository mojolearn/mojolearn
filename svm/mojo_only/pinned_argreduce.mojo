"""Block-wide argmin / argmax with an EXPLICIT tie-break, no lane primitive.

NOT A PORT. `smoblocksolve.cuh` reduces `KVPair<math_t, int>` with
`cub::BlockReduce<Pair, WSIZE>::Reduce(pair, cuda::minimum{} / maximum{},
n_ws)`, and `KVPair::operator<` / `operator>` compare `val` ONLY
(`cpp/src_prims/selection/kselection.cuh:66-79`, both carrying the comment
"///@todo: should we also consider the key when values are the same?").
On equal `val` the surviving key is whichever pair CUB's warp-then-block
fold happens to keep, which is a function of the fold topology and therefore
of the lane width and the vendor (IDENTITY_PATHS rows 8 and 20).

DEVIATION 633 (svm/README.md, identity content section 3): the tie resolves
to the SMALLER KEY, and the key the block solve passes is the TRAINING INDEX
`ws_idx[tid]`, not `tid`, so the winner is a pure function of the working
SET, not of the order the kernel cache permuted it into.

The shape is `core/pinned_reduce.mojo`'s halving tree through threadgroup
memory with no warp primitive, so no lane width can reach it. A selection
over a TOTAL order (value, then key) is exactly commutative and associative
away from NaN, so the result is independent of `block_size` as long as the
padding threads pass the identity (`+inf` for argmin, `-inf` for argmax,
with any key). NaN never enters: the block solve masks with `+-INFINITY`
and their `CheckStoppingCondition` THROWS on a NaN `diff` before the next
launch; ours raises at the same place.

CONTRACT (the same as `pinned_block_sum`'s): every thread of the block
calls it; `block_size` is a power of two; the result is returned to EVERY
thread (theirs broadcasts `u` and `l` through `__shared__`); the trailing
`barrier()` protects the slab for back-to-back calls. Mode-free: there is
no FAST arm to preserve, because the library call's tie order is not a
spelling anyone wrote down.
"""

from std.gpu import thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier


@always_inline
def pinned_block_argmin[
    block_size: Int
](value: Float32, key: Int32) -> Tuple[Float32, Int32]:
    """`min` over `(value, key)` lexicographically; returned to all threads."""
    var tid = Int(thread_idx.x)
    var vals = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var keys = stack_allocation[
        block_size,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    vals[tid] = value
    keys[tid] = key
    barrier()
    var step = block_size // 2
    while step > 0:
        if tid < step:
            var ov = vals[tid + step]
            var ok = keys[tid + step]
            var mv = vals[tid]
            var mk = keys[tid]
            if ov < mv or (ov == mv and ok < mk):
                vals[tid] = ov
                keys[tid] = ok
        barrier()
        step //= 2
    var rv = vals[0]
    var rk = keys[0]
    barrier()
    return (rv, rk)


@always_inline
def pinned_block_argmax[
    block_size: Int
](value: Float32, key: Int32) -> Tuple[Float32, Int32]:
    """`max` over value, ties to the SMALLER key; returned to all threads."""
    var tid = Int(thread_idx.x)
    var vals = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var keys = stack_allocation[
        block_size,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    vals[tid] = value
    keys[tid] = key
    barrier()
    var step = block_size // 2
    while step > 0:
        if tid < step:
            var ov = vals[tid + step]
            var ok = keys[tid + step]
            var mv = vals[tid]
            var mk = keys[tid]
            if ov > mv or (ov == mv and ok < mk):
                vals[tid] = ov
                keys[tid] = ok
        barrier()
        step //= 2
    var rv = vals[0]
    var rk = keys[0]
    barrier()
    return (rv, rk)


@always_inline
def pinned_block_max_all[block_size: Int](value: Float32) -> Float32:
    """`cub::BlockReduce<math_t>::Reduce(f_tmp, cuda::maximum{})` for
    `f_max`; a pure max, returned to every thread (theirs reads it on thread
    0 and broadcasts `diff` through `__shared__`)."""
    var tid = Int(thread_idx.x)
    var vals = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    vals[tid] = value
    barrier()
    var step = block_size // 2
    while step > 0:
        if tid < step:
            var ov = vals[tid + step]
            if ov > vals[tid]:
                vals[tid] = ov
        barrier()
        step //= 2
    var rv = vals[0]
    barrier()
    return rv
