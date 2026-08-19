"""FAISS `WarpSelect`: the register-resident top-k queue.

PORT OF `raft/neighbors/detail/faiss_select/Select.cuh:340-500` at RAFT
`661a3b8`, the general `WarpSelect<K, V, Dir, Comp, NumWarpQ, NumThreadQ,
ThreadsPerBlock>`. Partial. Do not improve.

WHY THIS AND NOT `matrix/detail/select_warpsort.cuh`
-----------------------------------------------------
Read the header of `merge_network_warp.mojo` first: the two are different
files implementing the same FAISS design, and neither subsumes the other.

The point of THIS one is that it has no shared memory and no block phase.
It is a struct a distance kernel keeps in registers and feeds one value at a
time, which is the only reason
`cuvs/src/neighbors/detail/knn_brute_force.cuh:443` can dispatch
`k <= 64 && row-major && L2` to `fusedL2Knn` and never write a distance
matrix. A device-wide top-k -- ours or a vendor's -- cannot do this job at
all: it reads its input from memory by construction, which forces the matrix
to be materialized.

WHAT THE FUSED KERNEL ACTUALLY INSTANTIATES
--------------------------------------------
`fused_l2_knn.cuh:743-771` has exactly TWO instantiations, and they are the
whole reachable parameter set:

    numOfNN <= 32  ->  NumWarpQ = 32, NumThreadQ = 2
    numOfNN <= 64  ->  NumWarpQ = 64, NumThreadQ = 3
    otherwise      ->  ASSERT("num of nearest neighbors must be <= 64")

with `ThreadsPerBlock = 32`, `Dir = false` (smallest), `K = float`,
`V = uint32_t` (`fused_l2_knn.cuh:220-222`). So `kNumWarpQRegisters` is 1 or
2, and `NumThreadQ` is 2 or 3.

NOT PORTED, and each is a row in `UNPORTED.tsv`
------------------------------------------------
  * `WarpSelect<..., NumWarpQ = 1, ...>`, the k == 1 specialization
    (`Select.cuh:503-548`). It is a different algorithm -- a plain warp
    min-reduction over a `KeyValuePair` -- and `fused_l2_knn.cuh` never
    reaches it, because `NumWarpQ` there is 32 or 64 whatever `numOfNN` is.
  * `BlockSelect` and `FinalBlockMerge` (`Select.cuh:20-338`) and all of
    `MergeNetworkBlock.cuh`. The fused kernel uses the WARP queue only; its
    cross-block merge is its own code (`fused_l2_knn.cuh:146-186`,
    `updateSortedWarpQ`), not `BlockSelect`.
  * `key_value_block_select.cuh`. Not on this path.

DEVIATIONS
----------
1. **`__any_sync(0xffffffff, needSort)` becomes a warp maximum.**
   `Select.cuh:405-409`. Mojo has no warp vote/ballot primitive; it has the
   warp reductions. `warp.max(1 if needSort else 0) != 0` is a warp-wide OR
   of a boolean and computes the identical value for every lane. This is a
   LANGUAGE-LEVEL counterpart to a CUDA intrinsic -- `gpu.primitives.warp` is
   how Mojo spells `__shfl_xor_sync` and friends -- not a device-wide library
   call standing in for an algorithm, and nothing leaves registers.
   The cost is that the reduction is a shuffle tree where theirs is one
   instruction. Not measured.

2. **Register arrays are padded `SIMD`** -- see `merge_network_warp.mojo`
   DEVIATION 1. `NumThreadQ = 3` becomes a `SIMD` of width 4 whose element 3
   is never read or written.

3. **`initK` / `initV` are constructor arguments, as theirs are, but they
   are also the padding sentinel.** Theirs leaves `threadK[NumThreadQ]`
   entirely filled by the constructor loop; ours fills the padding lane too,
   so a stray read is inert.

4. **`@always_inline` everywhere.** Theirs is `__device__ inline` on every
   method. See `merge_network_warp.mojo` DEVIATION 3 for why dropping it is
   not cosmetic on Metal.
"""

from std.gpu.primitives.warp import lane_id, max as warp_max, shuffle_idx

from neighbors.ported.neighbors.detail.faiss_select.merge_network_warp import (
    WARP_LANES,
    comp_gt,
    comp_lt,
    is_pow2,
    pow2_ceil,
    warp_merge_any_registers,
    warp_sort_any_registers,
)


struct WarpSelect[num_warp_q: Int, num_thread_q: Int, dir: Bool](
    Copyable, Movable
):
    """`WarpSelect<K, V, Dir, Comp, NumWarpQ, NumThreadQ, ThreadsPerBlock>`,
    `Select.cuh:346-500`, at `K = float`, `V = uint32_t`.

    `dir` TRUE produces the largest values, FALSE the smallest. k-NN wants
    FALSE (`fused_l2_knn.cuh:221`).

    `ThreadsPerBlock` is not a parameter here: the general specialization
    uses it for nothing but a `static_assert` that it is a power of two
    (`Select.cuh:362`); only the `NumWarpQ == 1` specialization, which is not
    ported, actually reads it.
    """

    #: `kNumWarpQRegisters = NumWarpQ / WarpSize`, `:347`.
    comptime num_warp_q_registers = Self.num_warp_q // WARP_LANES
    #: Padded SIMD widths. See DEVIATION 2.
    comptime warp_width = pow2_ceil(Self.num_warp_q // WARP_LANES)
    comptime thread_width = pow2_ceil(Self.num_thread_q)

    #: `const K initK`, `:483`.
    var init_k: Float32
    #: `const V initV`, `:486`.
    var init_v: UInt32
    #: `int numVals`, `:489`. Number of valid elements in the thread queue.
    var num_vals: Int
    #: `K warpKTop`, `:492`. The k-th lowest (`!Dir`) element so far.
    var warp_k_top: Float32
    #: `K threadK[NumThreadQ]` / `V threadV[NumThreadQ]`, `:495-496`.
    var thread_k: SIMD[DType.float32, Self.thread_width]
    var thread_v: SIMD[DType.uint32, Self.thread_width]
    #: `K warpK[kNumWarpQRegisters]` / `V warpV[...]`, `:499-500`.
    #: `warpK[0]` is highest (`Dir`) or lowest (`!Dir`).
    var warp_k: SIMD[DType.float32, Self.warp_width]
    var warp_v: SIMD[DType.uint32, Self.warp_width]
    #: `int kLane`, `:505`. Which lane holds an approximation (>= k) to the
    #: k-th element in the LAST warp-queue register.
    var k_lane: Int

    @always_inline
    def __init__(out self, init_k_val: Float32, init_v_val: UInt32, k: Int):
        """`WarpSelect(K initKVal, V initVVal, int k)`, `:349-374`."""
        comptime assert is_pow2(
            Self.num_warp_q
        ), "warp queue must be power-of-2"

        self.init_k = init_k_val
        self.init_v = init_v_val
        self.num_vals = 0
        self.warp_k_top = init_k_val
        # `kLane((k - 1) % WarpSize)`, `:351`. ALREADY reduced modulo the
        # warp size by them, which is why `shuffle_idx` is enough here and
        # `select_warpsort.cuh`'s `set_k_th_` is not expressible.
        self.k_lane = (k - 1) % WARP_LANES

        # `:365-369` and `:371-374`, plus the padding lanes (DEVIATION 3).
        self.thread_k = SIMD[DType.float32, Self.thread_width](init_k_val)
        self.thread_v = SIMD[DType.uint32, Self.thread_width](init_v_val)
        self.warp_k = SIMD[DType.float32, Self.warp_width](init_k_val)
        self.warp_v = SIMD[DType.uint32, Self.warp_width](init_v_val)

    @always_inline
    def add_thread_q(mut self, key: Float32, val: UInt32):
        """`addThreadQ`, `:376-391`.

        Rotate right and insert at 0, but only if the value beats the
        current k-th. No warp communication happens here, which is what lets
        a caller with divergent rows call it alone.
        """
        var better: Bool

        @parameter
        if Self.dir:
            better = comp_gt(key, self.warp_k_top)
        else:
            better = comp_lt(key, self.warp_k_top)

        if better:
            @parameter
            for step in range(Self.num_thread_q - 1):
                comptime I = Self.num_thread_q - 1 - step
                self.thread_k[I] = self.thread_k[I - 1]
                self.thread_v[I] = self.thread_v[I - 1]
            self.thread_k[0] = key
            self.thread_v[0] = val
            self.num_vals += 1

    @always_inline
    def check_thread_q(mut self):
        """`checkThreadQ`, `:393-419`.

        WARNING, theirs: all threads in a warp must participate. The vote is
        what makes the flush warp-uniform, so the merge below -- which
        shuffles -- is reached by every lane together.
        """
        var need_sort = self.num_vals == Self.num_thread_q
        # `__any_sync(0xffffffff, needSort)`, `:398`. See DEVIATION 1.
        var flag = Int32(0)
        if need_sort:
            flag = Int32(1)
        if warp_max(flag) == Int32(0):
            # no lanes have triggered a sort
            return

        self.merge_warp_q()

        # Any top-k elements have been merged into the warp queue; the
        # thread queues are free to reset. Theirs, `:409-416`.
        self.num_vals = 0

        @parameter
        for i in range(Self.num_thread_q):
            self.thread_k[i] = self.init_k
            self.thread_v[i] = self.init_v

        # We have to beat at least this element. `:418`.
        self.warp_k_top = shuffle_idx(
            self.warp_k[Self.num_warp_q_registers - 1], UInt32(self.k_lane)
        )

    @always_inline
    def merge_warp_q(mut self):
        """`mergeWarpQ`, `:421-432`.

        Sorts the per-thread queues, then merges the two already-sorted
        lists into one. `FullMerge = false`: the thread queue is about to be
        reset, so there is no reason to finish sorting it.
        """
        warp_sort_any_registers[
            Self.num_thread_q, Self.thread_width, not Self.dir
        ](self.thread_k, self.thread_v)

        warp_merge_any_registers[
            Self.num_warp_q_registers,
            Self.num_thread_q,
            Self.warp_width,
            Self.thread_width,
            not Self.dir,
            False,
        ](self.warp_k, self.warp_v, self.thread_k, self.thread_v)

    @always_inline
    def add(mut self, key: Float32, val: UInt32):
        """`add`, `:436-440`. WARNING, theirs: all threads in a warp must
        participate; otherwise call `add_thread_q` and `check_thread_q`
        separately."""
        self.add_thread_q(key, val)
        self.check_thread_q()

    @always_inline
    def reduce(mut self):
        """`reduce`, `:442-447`. Dump and merge the queues, producing the
        final per-warp result."""
        self.merge_warp_q()

    @always_inline
    def write_out(
        self,
        out_k: MutPointer[Float32, MutAnyOrigin],
        out_v: MutPointer[UInt32, MutAnyOrigin],
        k: Int,
    ):
        """`writeOut`, `:450-463`. The answer is strided across the warp:
        register `i` of lane `l` is output slot `i * WarpSize + l`."""
        var lane = Int(lane_id())

        @parameter
        for i in range(Self.num_warp_q_registers):
            var idx = i * WARP_LANES + lane
            if idx < k:
                out_k.unsafe_store(idx, self.warp_k[i])
                out_v.unsafe_store(idx, self.warp_v[i])
