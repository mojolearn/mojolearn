# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""Warp-sort top-k, the bitonic WarpSelect family.

PORT OF `raft/matrix/detail/select_warpsort.cuh` at RAFT `9aa17e5`, together
with the whole of `raft/util/bitonic_sort.cuh` which it is built on. Partial:
ONE of the four warp queues is ported, `warp_sort_immediate`. Do not improve.

Same rule as its sibling `select_radix.mojo`: this is a RAFT file we READ AND
TRANSLITERATE, not a RAFT call we stand in for, so it lives in `gbdt/` with
raft as its upstream and carries the attribution duty that follows.

WHY THIS FILE EXISTS AT ALL, AND WHY IT DID NOT UNTIL NOW
---------------------------------------------------------
`NOT_IMPLEMENTED.tsv` carried this file as `UNPORTABLE`, on the ground that it has
14 warp intrinsics (`__shfl_xor_sync`, `laneId`) and Mojo 1.0 had none.
**That ground was false and has been retracted** (`PORTING.md 2`,
`VENDOR_LIBRARIES.md`). The primitives are under `std.gpu.primitives.warp`
and `max.gpu.sync`; the earlier searches looked one namespace level too high
in four places and missed all four.

It matters because RAFT does not consider radix its first choice.
`select_k-inl.cuh:38`, `choose_select_k_algorithm`, opens at `:40` with

    if (k > 256) { ...radix... } else { if (k > 2) { ...warp family... } }

so every k a k-NN user actually asks for -- 10, 50, 100 -- goes to the warp
family in RAFT's own learned dispatch, and radix is their fallback for the
large-k tail. Until this file, this tree ran RAFT's second choice across the
entire practical range for a reason that turned out to be wrong.

WHAT IS PORTED AND WHAT IS NOT
-------------------------------
PORTED, transliterated:
  * `raft/util/bitonic_sort.cuh` in full: `bitonic<Size>::merge_impl` and
    `::sort_impl`, as `bitonic_merge` / `bitonic_sort`.
  * `warp_sort` (the base queue): the constructor, `load_sorted`, `store`,
    `merge_in`.
  * `warp_sort_immediate`: `add`, `done`.
  * `block_sort::done` and `::store`, the shared-memory tree merge, as free
    functions `block_sort_done` / `block_sort_store`.
  * `block_kernel`, dense layout, as `warpsort_topk_block_kernel`.
  * `cub::Traits<float>::TwiddleIn` / `TwiddleOut`, which `block_kernel`
    applies on the way in and out.

NOT PORTED, and each one is a row in `NOT_IMPLEMENTED.tsv`:
  * `warp_sort_filtered`, `warp_sort_distributed`, `warp_sort_distributed_ext`.
    See "THE ONE CONSTRUCT THAT DOES NOT PORT" below; all three need the same
    missing thing.
  * `calc_launch_parameter`, `launch_setup`, `calc_optimal_params`,
    `warpsort_params_cache`, `LaunchThreshold`. These are the HOST-side
    occupancy search, and they are built on
    `cudaOccupancyMaxPotentialBlockSizeVariableSMem`
    (`select_warpsort.cuh:846`), which has no counterpart in this toolchain.
    The launch geometry is pinned instead; see LAUNCH GEOMETRY below.
  * `select_k` / `select_k_impl` / `select_k_`, the host entry points,
    including the `len_per_thread <= 4` choice between immediate and
    filtered. With only one queue ported there is nothing to choose.
  * The CSR `RowLayout` (`select::csr_layout`) and the `in_indptr` argument.
    Dense only.
  * The `kMaxGridDimY = 32768` batch chunking loop. It exists because CUDA's
    grid Y dimension is capped at 65535; it is a host-side loop over kernel
    launches and the caller can run it if a batch ever exceeds the cap.

THE ONE CONSTRUCT THAT DOES NOT PORT, EXACTLY
----------------------------------------------
`select_warpsort.cuh:327` (and `:452`, `:560`), `set_k_th_`:

    k_th_ = shfl(val_arr_[kMaxArrLen - 1], k - 1, kWarpWidth);

This is `__shfl_sync` with an explicit `width`, and the source lane `k - 1`
is DELIBERATELY allowed to exceed that width -- their own comment at `:329`
says so: "it's ok if it is outside the warp size / width; the modulo op will
be done inside the __shfl_sync". `k` runs up to 256 while `kWarpWidth` is at
most 32, so `k - 1` is out of range for most k and the `% width` inside the
intrinsic is load-bearing arithmetic, not a guard.

Mojo's `shuffle_idx(value, offset)` takes no `width` and this file has no
evidence about what it does with an out-of-range lane. Reconstructing the
modulo by hand (`shuffle_idx(v, (k - 1) % warp_width)`) would be a GUESS at
what CUDA computes, not a transliteration, so it is not done here. That is
the single reason all three filtered/distributed queues are left out: each
one calls `set_k_th_` on every buffer flush and none of them is meaningful
without it.

`warp_sort_immediate` never calls `set_k_th_`. It is the only member of the
family that needs nothing but `shuffle_xor` and `lane_id`, which is what
makes it the natural first target and not merely the easiest.

`shuffle_xor` DOES port with no such doubt, and the reason is worth writing
down because it looks like the same problem and is not. `bitonic_sort.cuh`
calls `shfl_xor(key, stride, warp_width)` only with `stride < warp_width`
(`bitonic_sort.cuh:206`, whose loop starts at `warp_width >> 1`), and the
subwarps are power-of-two sized and aligned, so `lane ^ stride` never leaves
the subwarp and the `width` argument cannot change the answer. Mojo's
two-argument `shuffle_xor(value, offset)` is therefore EXACTLY their call
on every use site in this file, not an approximation of it.

THE COMPARATOR, AND HOW ITS TIES DIFFER FROM `select_radix.mojo`
----------------------------------------------------------------
`select_radix.mojo` records that RAFT's radix select has NO index tie-break
and places tied outputs with `atomicAdd`, so the returned indices are not
reproducible run to run. The question for this file is whether warpsort
agrees. **It agrees about the comparator and disagrees about the
consequence**, and both halves are findings.

Their comparator, `select_warpsort.cuh:103-109`:

    template <bool Ascending, typename T>
    auto is_ordered(T left, T right) -> bool
    {
      if constexpr (Ascending) { return left < right; }
      if constexpr (!Ascending) { return left > right; }
    }

STRICT. The payload is not consulted. So warpsort has no index tie-break
either, and neither does the sort network: `bitonic_sort.cuh:195` swaps on
`ascending ? key > other : key < other` and `:209` assigns on
`(ascending != is_second) ? key > other : key < other`, both strict, so an
element never displaces an equal one. Ties keep the incumbent.

**This is the OPPOSITE of `select_radix.mojo`'s tie-break note in its
effect.** Radix resolves ties through `atomicAdd`, whose order is a race.
Warpsort resolves them through a bitonic network, whose comparison schedule
is fixed and data-independent, so for a FIXED launch geometry the same input
gives the same indices every run. That is a strictly stronger determinism
property than the file this tree currently ships, and it was not the reason
warpsort was ported.

Two things it is NOT. It is not geometry-independent: which of several tied
elements survives depends on which lane loaded it, so changing
`num_blocks` or `block_dim` can change which tied index comes back. And it
is not an `IDENTICAL` column: that still needs an explicit index tie-break
RAFT does not have in either family. Recorded, not done -- adding one here
would be an improvement on RAFT.

DEVIATIONS, each one deliberate
--------------------------------
1. **Register arrays are `SIMD` values, not arrays.** RAFT holds
   `T val_arr_[kMaxArrLen]` and nvcc keeps it in registers. The obvious Mojo
   transliteration, `stack_allocation` with no address space, is thread-local
   MEMORY and would spill every element of the queue (`PORTING.md 26`, which
   cost this tree a whole slower-than-naive GEMM). `kMaxArrLen` is a power of
   two by construction, so `SIMD[DType.uint32, arr_len]` is exact.

2. **`bitonic::merge_impl`'s triple loop is written as a recursion over
   comptime `SIZE`/`BASE`.** Their loop indexes `keys[i]` with `i` derived
   from a `#pragma unroll` induction variable; a `SIMD` element index must be
   comptime, and Mojo's comptime loop is `@parameter for` over a `range`,
   which cannot express `for (size = Size; size > 1; size >>= 1)`. The
   recursion emits the SAME comparator network: their triple loop is
   level-by-level over independent compare-exchange pairs, and the recursion
   is the same levels visited depth-first, so every element sees all of its
   levels in the same relative order. The per-level inner order
   (`i` descending from `offset + stride - 1`) is preserved literally even
   though the pairs are disjoint and it could not matter.

3. **Shared memory is two statically sized arrays, not one `extern
   __shared__` byte blob.** `calc_smem_size_for_block_wide`
   (`select_warpsort.cuh:661`) computes ONE dynamic allocation and carves the
   index half out of it at a `Pow2<256>::roundUp` boundary. Mojo's
   `stack_allocation` in `AddressSpace.SHARED` is static, so the size is a
   comptime function of `capacity` and `block_warps`, and the value and index
   halves are two allocations. The 256-byte round-up existed only to align
   the second half of one blob and has no arithmetic effect; with two
   allocations there is no second half to align. Slot count per half is
   `ceildiv(block_warps, 2) * capacity`, which is `>= ceildiv(nwarps, 2) * k`
   for every legal call. At the maximum instantiation (capacity 256,
   block_warps 8) that is 4 KB per half, 8 KB total, against Metal's 32 KB
   threadgroup budget (`PORTING.md 1`). A consequence: the `store` and
   `load_sorted` overloads that face shared memory are typed
   `UnsafePointer[..., address_space = AddressSpace.SHARED,
   origin=MutUntrackedOrigin]` rather than `MutPointer`, because theirs is
   one untyped `uint8_t*` and Mojo carries the address space in the type.

4. **`block_sort` is two free functions, not a class.** RAFT's `block_sort`
   is a template-template wrapper whose entire job is to pick a queue type
   and forward `add`. With one queue ported there is nothing to pick, and its
   `init_blockwide(k, nullptr)` for `warp_sort_immediate`
   (`select_warpsort.cuh:613`) is just `{k}`. `done` and `store` are the only
   code in it and they are here verbatim.

5. **`__ldcs` becomes a plain load.** `block_kernel:796,798` uses the
   non-temporal load hint. It is a cache-policy hint with no semantic
   content and no counterpart here.

6. **`__launch_bounds__(256)` is dropped.** No counterpart. `block_warps`
   defaults to 8, which with a 32-lane warp is the 256 threads their
   attribute names.

7. **`in_idx == nullptr` becomes an explicit `has_in_idx` flag.** RAFT
   branches on a null pointer (`:798`). A Mojo kernel argument cannot be
   usefully null, and `PORTING.md 19` forbids selecting a pointer with a
   conditional expression in this tree, so the caller passes a flag and, when
   it is 0, any distinct valid buffer that is never read. When the flag is 0
   the payload is `i`, the column index within the row, which is what k-NN
   wants.

8. **`select_min` is the comptime `ascending` parameter, and the twiddle is
   pure.** `select_radix.mojo`'s `twiddle_in` folds `select_min` into the key
   by inverting all bits. RAFT does not do that here: `block_kernel` twiddles
   with the unmodified `cub::Traits<T>::TwiddleIn` and instantiates the
   kernel with `Ascending = select_min` (`:900`, `:913`). Copied their way,
   which is why `twiddle_in` in this file takes no `select_min` and is NOT
   the same function as the one in `select_radix.mojo` despite the name.

9. **`WARP_LANES` is a pinned 32.** This is not a Mojo limitation and not a
   choice: RAFT pins it too, `static const int WarpSize = 32;` at
   `util/cuda_dev_essentials.cuh:74`, and the whole family's array sizes are
   `Capacity / min(Capacity, WarpSize)`. It is correct on Apple and NVIDIA
   and WRONG on AMD, whose wavefront is 64 -- the same cross-vendor hazard
   `original/kernel_matrix.mojo` names for CatBoost's hardcoded 32. On AMD
   this file would build subwarps of the wrong width and the shuffles would
   cross them. Pinned, declared, and left for a measurement on that hardware.

LAUNCH GEOMETRY the caller must use
------------------------------------
`capacity` is `bound_by_power_of_two(k)` (`integer_utils.hpp:144`), i.e. the
smallest power of two `>= k`, and must satisfy `k <= capacity <= 256`
(`kMaxCapacity`, `:99`).

    warp_width = min(capacity, 32)
    block_dim  = (block_warps * warp_width, 1, 1)   # MUST be a multiple of 32
    grid_dim   = (num_blocks, batch_size, 1)
    shared     = none passed; it is static in the kernel

`block_dim.x` must be a multiple of 32 because the main loop's uniformity
argument below depends on `threadIdx.x % 32 == laneId()` holding for the
whole grid stride. With the default `block_warps = 8` and any
`capacity >= 32`, `block_dim.x` is 256 and this is automatic. For
`capacity < 32` the caller must scale `block_warps` the way
`calc_launch_parameter:1058` does, by `capacity_per_full_warp / capacity`.

`num_blocks == 1` is the single-pass form and writes the final answer:
`out_val`/`out_idx` are `batch_size * k` long. With `num_blocks > 1` the
kernel writes `batch_size * num_blocks * k` partial results laid out as
`[batch][block][k]`, and a SECOND launch of this same kernel over that
buffer with `len = k * num_blocks`, `num_blocks = 1` merges them --
`select_k_:1106-1120` does exactly that and it is the caller's loop, not ported
here.

`out_val` is FLOAT and already twiddled back; `out_idx` is UInt32.

WHY THE MAIN LOOP BOUND IS `len + laneId()` AND MUST STAY THAT WAY
-------------------------------------------------------------------
`block_kernel:793` reads `per_thread_lim = len + laneId()`, which looks like
an off-by-lane bug and is the opposite. With `i = base + lane + m * stride`,
the condition `i < len + lane` is `base + m * stride < len`, which does not
mention `lane`: **every lane of a warp runs the same number of iterations.**
That is what makes `buf_len_` warp-uniform, which is what makes the
`if (buf_len_ == kMaxArrLen)` flush inside `add` -- which contains shuffles
-- reach all 32 lanes together. Lanes past the end still call `add`, with
`kDummy`. Change this bound and the kernel hangs.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.primitives.warp import lane_id, shuffle_xor
from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier


#: `static const int WarpSize = 32;`, `util/cuda_dev_essentials.cuh:74`.
#: Theirs, pinned, and wrong on AMD. See DEVIATION 9.
comptime WARP_LANES = 32

#: `static constexpr int kMaxCapacity = 256;`, `select_warpsort.cuh:99`.
comptime MAX_CAPACITY = 256

#: `__launch_bounds__(256)` on `block_kernel`, expressed as warps.
comptime DEFAULT_BLOCK_WARPS = 8


@always_inline
def is_ordered[ascending: Bool](left: UInt32, right: UInt32) -> Bool:
    """`is_ordered`, `select_warpsort.cuh:103-109`.

    Whether `left` should indeed be on the left w.r.t. `right`. STRICT, and
    the payload is not consulted: see the comparator section of the module
    docstring.
    """

    @parameter
    if ascending:
        return left < right
    else:
        return left > right


@always_inline
def dummy_key[ascending: Bool]() -> UInt32:
    """`kDummy`, `select_warpsort.cuh:139`.

    `Ascending ? upper_bound<T>() : lower_bound<T>()`. `block_kernel`
    instantiates the queue on `cub::Traits<T>::UnsignedBits`, which has no
    infinity, so `upper_bound` is `numeric_limits<uint32_t>::max()` and
    `lower_bound` is `numeric_limits<uint32_t>::lowest()`, i.e. zero
    (`util/cudart_utils.hpp:404-418`).
    """

    @parameter
    if ascending:
        return UInt32(0xFFFFFFFF)
    else:
        return UInt32(0)


@always_inline
def twiddle_in(value: Float32) -> UInt32:
    """`cub::Traits<float>::TwiddleIn`, as `block_kernel:796` applies it.

    NOT the same function as `select_radix.mojo`'s `twiddle_in`: that one
    folds `select_min` in by inverting, this one does not, because warpsort
    carries the direction in the comptime `ascending` parameter instead. See
    DEVIATION 8.
    """
    var bits = bitcast[DType.uint32](value)
    if (bits & UInt32(0x80000000)) != 0:
        return bits ^ UInt32(0xFFFFFFFF)
    return bits ^ UInt32(0x80000000)


@always_inline
def twiddle_out(bits: UInt32) -> Float32:
    """`cub::Traits<float>::TwiddleOut`, as `block_kernel:805` applies it.

    Restores the float representation the radix-comparable key was made from.
    """
    var b = bits
    if (b & UInt32(0x80000000)) != 0:
        b = b ^ UInt32(0x80000000)
    else:
        b = b ^ UInt32(0xFFFFFFFF)
    return bitcast[DType.float32](b)


# =========================================================================
# `raft/util/bitonic_sort.cuh`
#
# Warp-wide bitonic merge and sort over data STRIDED among `warp_width`
# threads: for a fixed `i`, `keys[i]` is sorted across the threads of a
# subwarp, and for `i < j` no `keys[j]` in any thread is smaller than any
# `keys[i]` in any other thread.
#
# `BASE` is this port's addition and carries no meaning of its own: it is
# where their `keys + kSize2` pointer arithmetic went, because a `SIMD`
# element index has to be comptime and a `SIMD` cannot be offset by a
# pointer. See DEVIATION 2.
# =========================================================================


@always_inline
def bitonic_merge[
    SIZE: Int, BASE: Int, ARR: Int
](
    mut keys: SIMD[DType.uint32, ARR],
    mut vals: SIMD[DType.uint32, ARR],
    ascending: Bool,
    warp_width: Int,
):
    """`bitonic<Size>::merge_impl`, `bitonic_sort.cuh:180-217`.

    Sorts any bitonic sequence; equivalently, merges two already-sorted
    halves of opposite order.
    """

    @parameter
    if SIZE > 1:
        # Their in-register phase, `:185-201`. One level of compare-exchange
        # at stride `SIZE/2` over this block, then the two half-blocks. Inner
        # order kept descending from `offset + stride - 1` exactly as theirs,
        # though the pairs are disjoint and cannot see each other.
        comptime STRIDE = SIZE // 2

        @parameter
        for t in range(STRIDE):
            var key = keys[BASE + STRIDE - 1 - t]
            var other = keys[BASE + SIZE - 1 - t]
            var swaps: Bool
            if ascending:
                swaps = key > other
            else:
                swaps = key < other
            if swaps:
                keys[BASE + STRIDE - 1 - t] = other
                keys[BASE + SIZE - 1 - t] = key
                var payload = vals[BASE + STRIDE - 1 - t]
                vals[BASE + STRIDE - 1 - t] = vals[BASE + SIZE - 1 - t]
                vals[BASE + SIZE - 1 - t] = payload

        bitonic_merge[STRIDE, BASE, ARR](keys, vals, ascending, warp_width)
        bitonic_merge[STRIDE, BASE + STRIDE, ARR](
            keys, vals, ascending, warp_width
        )
    else:
        # Their cross-lane phase, `:202-216`. One element per thread, a
        # butterfly over the subwarp.
        var lane = Int(lane_id())
        var key = keys[BASE]
        var payload = vals[BASE]
        var stride = warp_width >> 1
        while stride > 0:
            var is_second = (lane & stride) != 0
            # NB: don't put the shuffles in a conditional; they must be
            # called by all threads in a warp. Theirs, `:212`.
            var other = shuffle_xor(key, UInt32(stride))
            var other_payload = shuffle_xor(payload, UInt32(stride))
            var do_assign: Bool
            if ascending != is_second:
                do_assign = key > other
            else:
                do_assign = key < other
            if do_assign:
                key = other
                payload = other_payload
            stride >>= 1
        keys[BASE] = key
        vals[BASE] = payload


@always_inline
def bitonic_sort[
    SIZE: Int, BASE: Int, ARR: Int
](
    mut keys: SIMD[DType.uint32, ARR],
    mut vals: SIMD[DType.uint32, ARR],
    ascending: Bool,
    warp_width: Int,
):
    """`bitonic<Size>::sort_impl`, `bitonic_sort.cuh:220-236`."""

    @parameter
    if SIZE == 1:
        # `:225-229`. Note `lane & width` is the ASCENDING flag and is
        # per-lane, while `width` -- which is both the loop variable and the
        # subwarp width handed to the merge -- is uniform, so the trip count
        # is uniform and no lane misses a shuffle.
        var lane = Int(lane_id())
        var width = 2
        while width < warp_width:
            bitonic_merge[1, BASE, ARR](
                keys, vals, (lane & width) != 0, width
            )
            width <<= 1
    else:
        comptime HALF = SIZE // 2
        bitonic_sort[HALF, BASE, ARR](keys, vals, False, warp_width)
        bitonic_sort[HALF, BASE + HALF, ARR](keys, vals, True, warp_width)

    bitonic_merge[SIZE, BASE, ARR](keys, vals, ascending, warp_width)


# =========================================================================
# `warp_sort` + `warp_sort_immediate`
# =========================================================================


struct WarpSortImmediate[capacity: Int, ascending: Bool](Copyable, Movable):
    """`warp_sort_immediate<Capacity, Ascending, T, IdxT>`, `:595-658`,
    flattened together with its base `warp_sort<...>`, `:129-268`.

    A fixed-size warp-level priority queue: feed data through it and get the
    `k <= capacity` smallest (ascending) or greatest (descending) keys. This
    variant adds EVERY input to the intermediate buffer and therefore sorts
    once per `capacity` inputs, which is why RAFT prefers it for short rows
    and prefers `warp_sort_filtered` for long ones (`:1220`).

    Keys are `cub::Traits<T>::UnsignedBits`, i.e. already twiddled, exactly
    as `block_kernel:786-787` instantiates it. Payloads are UInt32 indices.
    """

    #: `kWarpWidth = std::min<int>(Capacity, WarpSize)`, `:141`.
    comptime warp_width = Self.capacity if Self.capacity < WARP_LANES else WARP_LANES
    #: `kMaxArrLen = Capacity / kWarpWidth`, `:232`.
    comptime arr_len = Self.capacity // (
        Self.capacity if Self.capacity < WARP_LANES else WARP_LANES
    )

    #: `const int k`, `:143`. The number of elements to select.
    var k: Int
    #: `T val_arr_[kMaxArrLen]`, `:234`.
    var val_arr: SIMD[DType.uint32, Self.arr_len]
    #: `IdxT idx_arr_[kMaxArrLen]`, `:235`.
    var idx_arr: SIMD[DType.uint32, Self.arr_len]
    #: `T val_buf_[kMaxArrLen]`, `:655`.
    var val_buf: SIMD[DType.uint32, Self.arr_len]
    #: `IdxT idx_buf_[kMaxArrLen]`, `:656`.
    var idx_buf: SIMD[DType.uint32, Self.arr_len]
    #: `int buf_len_`, `:657`. WARP-UNIFORM by construction; see the module
    #: docstring on the `len + laneId()` loop bound.
    var buf_len: Int

    @always_inline
    def __init__(out self, k: Int):
        """`warp_sort(int k)` `:154-161` and `warp_sort_immediate(int k)`
        `:603-612`, merged."""
        self.k = k
        self.val_arr = SIMD[DType.uint32, Self.arr_len](
            dummy_key[Self.ascending]()
        )
        self.idx_arr = SIMD[DType.uint32, Self.arr_len](UInt32(0))
        self.val_buf = SIMD[DType.uint32, Self.arr_len](
            dummy_key[Self.ascending]()
        )
        self.idx_buf = SIMD[DType.uint32, Self.arr_len](UInt32(0))
        self.buf_len = 0

    @always_inline
    def merge_in(mut self):
        """`warp_sort::merge_in<PerThreadSizeIn>`, `:254-268`, at
        `PerThreadSizeIn == kMaxArrLen` which is the only instantiation
        `warp_sort_immediate` makes (`:632-633`, `:645-646`).

        The other array is sorted in the OPPOSITE direction, which is what
        makes a single `bitonic.merge` enough to restore the queue.
        """

        @parameter
        for j in range(Self.arr_len):
            var key = self.val_arr[j]
            var other = self.val_buf[j]
            if is_ordered[Self.ascending](other, key):
                self.val_arr[j] = other
                self.idx_arr[j] = self.idx_buf[j]

        bitonic_merge[Self.arr_len, 0, Self.arr_len](
            self.val_arr, self.idx_arr, Self.ascending, Self.warp_width
        )

    @always_inline
    def add(mut self, val: UInt32, idx: UInt32):
        """`warp_sort_immediate::add`, `:618-640`."""
        # NB: the loop is used here to ensure the constant indexing, to not
        # force the buffers spill into the local memory. Theirs, `:620-621`,
        # and it is doubly true here: a runtime `SIMD` element index is not
        # a register access at all.
        @parameter
        for i in range(Self.arr_len):
            if i == self.buf_len:
                self.val_buf[i] = val
                self.idx_buf[i] = idx

        self.buf_len += 1
        if self.buf_len == Self.arr_len:
            bitonic_sort[Self.arr_len, 0, Self.arr_len](
                self.val_buf, self.idx_buf, not Self.ascending, Self.warp_width
            )
            self.merge_in()

            @parameter
            for i in range(Self.arr_len):
                self.val_buf[i] = dummy_key[Self.ascending]()
            self.buf_len = 0

    @always_inline
    def done(mut self):
        """`warp_sort_immediate::done`, `:642-648`.

        `buf_len` is warp-uniform, so this branch does not diverge and the
        shuffles inside `bitonic_sort` are reached by every lane.
        """
        if self.buf_len != 0:
            bitonic_sort[Self.arr_len, 0, Self.arr_len](
                self.val_buf, self.idx_buf, not Self.ascending, Self.warp_width
            )
            self.merge_in()

    @always_inline
    def load_sorted(
        mut self,
        in_val: UnsafePointer[
            Scalar[DType.uint32],
            address_space = AddressSpace.SHARED,
            origin=MutUntrackedOrigin,
        ],
        in_idx: UnsafePointer[
            Scalar[DType.uint32],
            address_space = AddressSpace.SHARED,
            origin=MutUntrackedOrigin,
        ],
        do_merge: Bool,
    ):
        """`warp_sort::load_sorted`, `:182-198`.

        Loads k values from a contiguous per-subwarp array and merges them
        in. `do_merge` must be the same for all threads within a subwarp; it
        exists so that threads of a FULL warp do not diverge on the merge.

        Performs collective warp operations at the end, so it is safe to
        call `store` with the same arguments right after -- but not the
        reverse, because the access patterns differ.
        """
        if do_merge:
            # `Pow2<kWarpWidth>::mod(laneId()) ^ Pow2<kWarpWidth>::Mask`,
            # `:184`. The XOR reverses the lane order, which is what puts
            # the incoming array in the opposite direction the merge wants.
            var idx = (Int(lane_id()) & (Self.warp_width - 1)) ^ (
                Self.warp_width - 1
            )

            @parameter
            for t in range(Self.arr_len):
                if idx < self.k:
                    var value = in_val.unsafe_load(idx)
                    if is_ordered[Self.ascending](
                        value, self.val_arr[Self.arr_len - 1 - t]
                    ):
                        self.val_arr[Self.arr_len - 1 - t] = value
                        self.idx_arr[
                            Self.arr_len - 1 - t
                        ] = in_idx.unsafe_load(idx)
                idx += Self.warp_width

        # `if (kWarpWidth < WarpSize || do_merge)`, `:196`. The left half is
        # a comptime constant here.
        if Self.warp_width < WARP_LANES or do_merge:
            bitonic_merge[Self.arr_len, 0, Self.arr_len](
                self.val_arr, self.idx_arr, Self.ascending, Self.warp_width
            )

    @always_inline
    def store(
        self,
        out_val: UnsafePointer[
            Scalar[DType.uint32],
            address_space = AddressSpace.SHARED,
            origin=MutUntrackedOrigin,
        ],
        out_idx: UnsafePointer[
            Scalar[DType.uint32],
            address_space = AddressSpace.SHARED,
            origin=MutUntrackedOrigin,
        ],
    ):
        """`warp_sort::store` with the identity postprocessors, `:218-230`.

        The two pointers are SHARED-address-space, because the only caller of
        this overload is the block tree-merge. `store_twiddled` below is the
        global-memory one.

        Their loop condition is `i < kMaxArrLen && idx < k`, an early exit.
        Written as a guard inside the comptime loop, which is the same set of
        stores because `idx` only grows.
        """
        var idx = Int(lane_id()) & (Self.warp_width - 1)

        @parameter
        for i in range(Self.arr_len):
            if idx < self.k:
                out_val.unsafe_store(idx, self.val_arr[i])
                out_idx.unsafe_store(idx, self.idx_arr[i])
            idx += Self.warp_width

    @always_inline
    def store_twiddled(
        self,
        out_val: MutPointer[Float32, MutAnyOrigin],
        out_idx: MutPointer[UInt32, MutAnyOrigin],
    ):
        """`warp_sort::store` with `ValF = cub::Traits<T>::TwiddleOut`, which
        is how `block_kernel:803-805` calls it. Restores the float
        representation on the way out.
        """
        var idx = Int(lane_id()) & (Self.warp_width - 1)

        @parameter
        for i in range(Self.arr_len):
            if idx < self.k:
                out_val.unsafe_store(idx, twiddle_out(self.val_arr[i]))
                out_idx.unsafe_store(idx, self.idx_arr[i])
            idx += Self.warp_width


# =========================================================================
# `block_sort`, `:672-739`. See DEVIATION 4 for why it is functions.
# =========================================================================


@always_inline
def block_sort_done[
    capacity: Int, ascending: Bool
](
    mut queue: WarpSortImmediate[capacity, ascending],
    val_smem: UnsafePointer[
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
    idx_smem: UnsafePointer[
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """`block_sort::done`, `:689-721`.

    At the point of calling this, the warp-level queues have consumed all
    input independently. What remains is to tree-merge them through shared
    memory, halving the number of live warps each round.

    NB: there is no need for a second `barrier()` between `store` and
    `load_sorted`. The pointers shift every iteration, so individual warps
    either access the same locations or do not overlap with any other warp;
    the access patterns within a warp differ between the two functions, but
    `load_sorted` implies a warp sync at the end so no `syncwarp` is needed
    either. Theirs, `:699-703`, and it is the reason this loop has one
    barrier and not two.
    """
    queue.done()

    comptime WW = WarpSortImmediate[capacity, ascending].warp_width
    var nwarps = Int(block_dim.x) // WW
    var warp_id = Int(thread_idx.x) // WW
    var k = queue.k

    # THE LOOP UPDATE IS RESTATED, AND THE REASON IS A COMPILER BUG.
    #
    # Theirs, `:709-710`, is a C for-increment with a comma operator:
    #
    #     for (int shift_mask = ~0, split = (nwarps + 1) >> 1; nwarps > 1;
    #          nwarps = split, split = (nwarps + 1) >> 1)
    #
    # i.e. `nwarps = split` and then `split` recomputed from the NEW `nwarps`.
    # Written that way literally, `mojo build` CRASHES -- not a diagnostic, a
    # segfault in the compiler. Bisected to a 7-line program with no GPU, no
    # SIMD and no warp primitive in it:
    #
    #     def main():
    #         var a = 8
    #         var b = 4
    #         while a > 1:
    #             a = b          # bare copy INTO the condition variable
    #             b = b - 1
    #         print(a)
    #
    # The trigger is exactly that: the while-condition variable's last write in
    # the body is a bare copy of another local. Any arithmetic on top of the
    # copy (`a = b - 1`, `a = b; a = a - 1`) compiles. See `PORTING.md`.
    #
    # `split` is a pure function of `nwarps` -- `split == (nwarps + 1) >> 1`
    # holds on entry to every iteration, by induction from their initializer
    # and their increment -- so it is a loop-local, and the induction step is
    # written as the same arithmetic instead of a copy. Same values, same
    # order, same number of iterations. WHAT is said is unchanged; only HOW.
    var shift_mask = ~Int(0)
    while nwarps > 1:
        var split = (nwarps + 1) >> 1
        if warp_id < nwarps and warp_id >= split:
            var dst_warp_shift = (warp_id - (split & shift_mask)) * k
            queue.store(
                val_smem.unsafe_offset(dst_warp_shift),
                idx_smem.unsafe_offset(dst_warp_shift),
            )
        barrier()

        shift_mask = ~shift_mask  # invert the mask
        var src_warp_shift = (warp_id + (split & shift_mask)) * k
        # The last argument serves as a condition for loading -- to make sure
        # threads within a full warp do not diverge on `bitonic::merge()`.
        queue.load_sorted(
            val_smem.unsafe_offset(src_warp_shift),
            idx_smem.unsafe_offset(src_warp_shift),
            warp_id < nwarps - split,
        )

        # `nwarps = split, split = (nwarps + 1) >> 1`, `:710`, as arithmetic.
        nwarps = (nwarps + 1) >> 1


@always_inline
def block_sort_store[
    capacity: Int, ascending: Bool
](
    queue: WarpSortImmediate[capacity, ascending],
    out_val: MutPointer[Float32, MutAnyOrigin],
    out_idx: MutPointer[UInt32, MutAnyOrigin],
):
    """`block_sort::store`, `:728-737`, with `ValF = TwiddleOut`.

    Only the first subwarp holds the merged answer.
    """
    comptime WW = WarpSortImmediate[capacity, ascending].warp_width
    if Int(thread_idx.x) < WW:
        queue.store_twiddled(out_val, out_idx)


# =========================================================================
# `block_kernel`, `:753-806`, dense layout only.
# =========================================================================


def warpsort_topk_block_kernel[
    capacity: Int,
    ascending: Bool,
    block_warps: Int = DEFAULT_BLOCK_WARPS,
](
    in_val: MutPointer[Float32, MutAnyOrigin],
    in_idx: MutPointer[UInt32, MutAnyOrigin],
    out_val: MutPointer[Float32, MutAnyOrigin],
    out_idx: MutPointer[UInt32, MutAnyOrigin],
    len_in: Int32,
    k_in: Int32,
    has_in_idx_in: Int32,
):
    """`block_kernel<warp_sort_immediate, Capacity, Ascending, T, IdxT,
    dense_layout>`, `:753-806`.

    Uses the warp queue to sort chunks of data within one block with no
    interblock communication. Multiple blocks may process one row of input,
    in which case they output multiple results of length k each and a second
    pass is needed to merge them; see LAUNCH GEOMETRY in the module
    docstring.

    `in_idx` is read only when `has_in_idx_in != 0`; when it is 0 the payload
    is the column index `i`, which is what a k-NN caller wants. It must still
    be a valid, DISTINCT buffer -- Mojo refuses the same buffer as two
    mutable kernel arguments.
    """
    #: `calc_smem_size_for_block_wide`, `:661`, as two static halves.
    #: DEVIATION 3. `block_warps` sizes it and the runtime `blockDim.x` is
    #: what `block_sort_done` actually divides, so the caller's
    #: `block_dim.x == block_warps * warp_width` contract is what keeps the
    #: tree merge inside this allocation.
    comptime SMEM_SLOTS = ((block_warps + 1) // 2) * capacity

    var val_smem = stack_allocation[
        SMEM_SLOTS,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()
    var idx_smem = stack_allocation[
        SMEM_SLOTS,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()

    var length = Int(len_in)
    var k = Int(k_in)
    var has_in_idx = has_in_idx_in != 0

    # * per-block output, `:762-767`
    var n_blocks = Int(grid_dim.x)
    var block_id = Int(block_idx.x) + n_blocks * Int(block_idx.y)
    var o_val = out_val.unsafe_offset(block_id * k)
    var o_idx = out_idx.unsafe_offset(block_id * k)

    # * per-block input, `:768-775`. `dense_layout::compute` is
    #   `{len, batch_id * len}`, so the whole row is this block's input and
    #   the grid-stride loop below is what splits it.
    var batch = Int(block_idx.y)
    var i_val = in_val.unsafe_offset(batch * length)
    var i_idx = in_idx.unsafe_offset(batch * length)

    # * trim k if the input is too short, `:777`
    if k > length:
        k = length

    # * set up the queue, `:779-789`. `warp_sort_immediate::mem_required` is
    #   0 (`:146`), so no per-warp scratch is handed in.
    var queue = WarpSortImmediate[capacity, ascending](k)

    # * main loop, `:791-799`. The `len + laneId()` bound is load-bearing:
    #   see the module docstring.
    var stride = n_blocks * Int(block_dim.x)
    var per_thread_lim = length + Int(lane_id())
    var i = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    while i < per_thread_lim:
        var key = dummy_key[ascending]()
        if i < length:
            key = twiddle_in(i_val.unsafe_load(i))
        var payload = UInt32(i)
        if i < length and has_in_idx:
            payload = i_idx.unsafe_load(i)
        queue.add(key, payload)
        i += stride

    # * write out the result, `:801-805`
    block_sort_done[capacity, ascending](queue, val_smem, idx_smem)
    block_sort_store[capacity, ascending](queue, o_val, o_idx)
