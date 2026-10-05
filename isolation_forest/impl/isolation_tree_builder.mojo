# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The batched Isolation Forest tree builder: one block per tree, the whole
block draws the subsample and walks the stack (cpu-gpu-cleanup t-forest; see
the block comment above `IF_BUILD_TPB_MAX` for what changed from cuML's
thread-0 walk).

Reference: `cpp/src/isolation_forest/isolation_tree_builder.cuh` at
rapidsai/cuml v26.08.00, branch for branch and loop for loop:
`StackEntry` (`:37-42`), `compute_c_n` (`:48-55`), `IFNode` (`:63-69`),
`curand_u64` / `sample_bounded` (`:84-101`), `contains_sample` /
`contains_int_sample` (`:103-117`), `build_tree_iterative_global`
(`:125-243`), `build_isolation_trees_global_kernel` (`:245-340`),
`traverse_global_tree` (`:342-355`), `compute_path_lengths_global_kernel`
(`:357-375`) and `build_isolation_forest_global` (`:377-420`). The
compaction pair (`:422-502`) is host bookkeeping for Treelite export and
is NOT implemented (NOT_IMPLEMENTED.tsv).

Their design, kept: all trees in ONE launch (`<<<n_trees, 128>>>`), the
subsample gathered into a contiguous per-tree buffer by every thread of
the block, then THREAD 0 ALONE draws from one `curandState` seeded
`(seed, tree_id)` and builds the tree iteratively through a global-memory
stack and partition index array. The RNG consumption order is therefore
the serial stack walk, a pure function of (seed, tree_id, data bits), and
nothing in the tree depends on block size, grid shape or launch order.
The RNG itself is `isolation_forest/impl/rng/xorwow.mojo`
(XORWOW; the brief's DEVIATION 680 re-keying is not needed, see there).

Where the numerics live, and how each seam is pinned under IDENTICAL:
  * the per-feature min/max over a node's rows (`:176-186`) is THEIR
    serial loop with strict `<` / `>` from the first row's value. A
    compare is IEEE-defined on every vendor (a hardware `min`/`max` is
    NOT, IDENTITY_PATHS row 39), so the fold is kept positional: on a
    (-0.0, +0.0) tie the FIRST row in partition order wins. NaN never
    reaches it (DEVIATION 680 refuses non-finite inputs by name in
    `isolation_forest.mojo`).
  * `threshold = min + frac * (max - min)` (`:198`) is ONE multiply-add:
    `identical_mul_add(frac, max - min, min)` -- nvcc contracts it too, so
    this is their likely bits as well as ours (DEVIATION 682).
  * `compute_c_n`'s `log` is `identical_log`; the Euler constant is
    `T(0.5772156649015329)` = 0x3f13c468 in float32 (DEVIATION 682).
  * the path-length sum over trees (`:368-370`) is their serial ascending
    loop per sample; pinned as written, `ftz` at the stored seams.

================= DEVIATION BLOCK =================
DEVIATION 682. THE FLOAT SEAMS OF THE BUILDER AND THE SCORER GO THROUGH
`checks/numerics.mojo`. Theirs: `log` (CUDA libm, vendor bits), a
contraction the compiler decides, no denormal policy. Ours under
IDENTICAL: `identical_log` in `compute_c_n`, `identical_mul_add` for the
threshold, `ftz` on every stored float (leaf path length, threshold, the
per-sample path-length accumulator); under FAST the stdlib / naive
spellings, so Apple's FAST bits do not move. Measured: the std-exp
sabotage in the README (the scorer's `identical_pow` -> `**`) moves
scores device-vs-oracle under IDENTICAL; on Apple the mul-add pin is
bit-inert (Metal contracts) exactly as numerics.mojo says.

DEVIATION 685. NODE STORAGE IS FOUR ARRAYS, NOT AN ARRAY OF `IFNode`.
Reference: `IFNode<T>{int feature_idx; T threshold; int left_child; int
right_child;}` in one `rmm::device_buffer`. Here: `node_feature` (Int32),
`node_threshold` (Float32), `node_left` (Int32), `node_right` (Int32),
same indices, same `tree_offsets`. WHAT is said is unchanged (every field,
every index); HOW changed because a whole-struct load through a pointer
in a kernel is a known Metal-compiler wall (CONTRIBUTING.md) and
because the identity card hashes each field as its own dtype
(`if.treeNNN.structure.{feat,thr,left,right}`), which a packed struct of
mixed dtypes could not express without a byte view.

DEVIATION 750. `curand_u64` DRAWS THE HIGH WORD FIRST, BY CHOICE, BECAUSE
THEIRS DOES NOT SAY. Theirs (`:83-86`):

    return (static_cast<uint64_t>(curand(rng_state)) << 32) | curand(rng_state);

The two `curand()` calls are operands of `|`, and C++ does not sequence
them relative to each other (this is not fixed by C++17, which orders the
operands only of `<<`/`>>` as stream operators, `->*`, subscript,
assignment and the comma). So WHICH of the two consecutive XORWOW draws
becomes the high 32 bits is the compiler's choice, and both choices are
conforming. Since every index in the forest comes out of `sample_bounded`
and `sample_bounded` is built out of `curand_u64`, the two choices give
two DIFFERENT FORESTS from the same seed: not a rounding difference, a
different tree.

This is a REPRODUCIBILITY DEFECT OF THEIRS, and the standing rule is to
fix it rather than implementation it. Fixed here by making the order explicit and
naming it: the first draw is the high word (two named locals, so no
reader and no compiler has a choice left). That is also what the Python
reference (`checks/xorwow_reference.py`) and the host oracle assume,
so all three agree by construction and the gates cannot certify the
order -- only that we are self-consistent.

WHAT IS NOT CLOSED: nothing here has been checked against a cuML binary.
If NVIDIA's front end picks the other order, cuML's forest for a given
seed differs from ours in every tree while every gate in this lane stays
green, because the gates compare us to us. Closing it needs ONE number
off a real GPU, and the two orders are far apart in it: fit cuML's
Isolation Forest with `seed = 42`, `n_estimators = 1`, `bootstrap = true`,
`max_samples = 1` on 1000 rows, and read `tree_sample_indices[0]` (tree 0's
first sampled row, which is `sample_bounded(&rng_state, 1000)` on the very
first `curand_u64`). The two draws are 300737663 then 2363150160
(`checks/xorwow_reference.tsv`, seed 42 tree 0, verified against
`curand_precalc.h`'s own constants), so:

    high word first (ours)   -> 408
    low word first (swapped) -> 23

408 says the order above is theirs; 23 says flip it. The swapped-order arm
is built in as `-D MOJOLEARN_IF_SABOTAGE_U64_SWAP=1`, which must turn the
whole gate red -- that is the lane's proof the order is load-bearing rather
than a preference. NOT_IMPLEMENTED.tsv carries this as the lane's one open
cross-vendor question.

DEVIATION 686. `size_t` INDICES ARE `Int64`, `StackEntry` IS FOUR `Int32`
WORDS. Their `sample_indices` is `size_t*` and the stack an array of
structs; ours are an `Int64` buffer and a flat `Int32` buffer of four
words per entry (`node_idx, start_idx, end_idx, depth`), same LIFO
discipline, same push order (right then left). Spelling only.
===================================================
"""

from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace

from std.sys import is_defined
from std.sys.info import has_apple_gpu_accelerator

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.sync import barrier

from isolation_forest.impl.rng.xorwow import (
    curandStateXORWOW,
    curand,
    curand_init,
    curand_uniform,
)
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_FAST,
    ftz,
    identical_log,
    identical_mul_add,
)


#: FAST on Apple (lane apple-fast-trees2): THE FIT'S X STAYS ROW-MAJOR ON THE
#: DEVICE. The IDENTICAL fit uploads the column-major transpose cuML's
#: `fit` takes (`order="F"`), which the binding builds on the host: a
#: threaded transpose of all n x d cells into an n x d pinned stage, then a
#: blit, for a forest that reads max_samples rows per tree (256 x 100 of a
#: million). Under FAST the binding's borrowed row-major block is copied to
#: the device as it is (a raw host-pointer copy, the fastest Metal upload
#: measured, b046c2de7), the finiteness refusal runs as a device scan of the
#: same cells, and the gather below reads `data[row * n_cols + col]`: the
#: same cell values (`ftz` is a no-op under FAST), the same trees, a
#: coalesced read of each sampled row. `-D MOJOLEARN_IF_ROWMAJOR_OFF` keeps
#: the column-major upload (the A/B arm).
comptime IF_FAST_ROWMAJOR = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_IF_ROWMAJOR_OFF"]()
)


comptime EULER_MASCHERONI_F32 = Float32(0.5772156649015329)
"""`T(0.5772156649015329)` with T = float: 0x3f13c468. Printed and gated
as hex by `if_check.mojo` so the constant cannot drift by a decimal."""

comptime IF_BUILD_TPB = 128
"""`build_isolation_trees_global_kernel<T><<<n_trees, 128, 0, stream>>>`
(`:397`). A scheduling width: the sampling, the gather and every node's
min/max and partition are split over the block's threads; no bit depends
on it (at most `IF_BUILD_TPB_MAX`)."""

comptime IF_DECISION_WORDS = 6
"""Int32 words of RECORDED DECISION per node (archive/plans/CARD_GAPS.md's isolation
forest items 1-4). See `build_tree_iterative_global` for the layout.

These are decisions the ALGORITHM makes, never ones the scheduler makes:
every word is a pure function of (seed, tree_id, data bits), and the node
count is a pure function of (max_depth, max_samples). Nothing here is the
machine-sized scratch that `core/identity_trace.mojo` rule 3 forbids."""

comptime IF_RNG_STATE_WORDS = 6
"""`curandStateXORWOW`'s d, v0..v4: the stream POSITION a tree finished at."""

comptime IF_STACK_WORDS = 4
"""`StackEntry` as four Int32 words (DEVIATION 686)."""

comptime IF_SCRATCH_WORDS_PER_NODE = IF_STACK_WORDS + IF_DECISION_WORDS
"""10. One tree's scratch is `max_nodes_per_tree * 10 + IF_RNG_STATE_WORDS`
Int32, carved into three disjoint slices. IT IS ONE KERNEL ARGUMENT, not
three, ON PURPOSE: Metal caps a kernel at 31 arguments and this one already
stands at 25. A new output here REUSES A SLICE."""

comptime IF_PATH_TPB = 256
"""`compute_path_lengths`' `threads = 256` (`isolation_forest.cuh:146`)."""


# ---------------------------------------------------------------------------
# `:48-55`
# ---------------------------------------------------------------------------


def compute_c_n(n_samples: Int) -> Float32:
    """`compute_c_n<T>(int n_samples)`: c(n) = 2H(n-1) - 2(n-1)/n, the
    expected path length of an unsuccessful BST search, with H(n-1) =
    ln(n-1) + gamma. `<= 1 -> 0`, `== 2 -> 1`."""
    if n_samples <= 1:
        return Float32(0.0)
    if n_samples == 2:
        return Float32(1.0)
    var n = Float32(n_samples)
    var h = ftz(identical_log(n - Float32(1.0)) + EULER_MASCHERONI_F32)
    var tail = ftz(Float32(2.0) * (n - Float32(1.0)) / n)
    return ftz(identical_mul_add(Float32(2.0), h, -tail))  # the default build's fused op (lane/pinned-mul-contract-free)


# ---------------------------------------------------------------------------
# `:83-115`: the draws. See DEVIATION 750 in the block above for why
# `curand_u64`'s two draws had to be ordered by hand.
# ---------------------------------------------------------------------------


#: SABOTAGE (a no-op in every build that does not name it): take the two
#: draws of `curand_u64` in the OTHER order, which is the other conforming
#: reading of their unsequenced `|` (DEVIATION 750). This is the lane's
#: proof that the order is load-bearing rather than a preference: it
#: perturbs the RNG stream and nothing else, and every structure, path
#: length and score in the forest must move.
comptime SAB_U64_SWAP = is_defined["MOJOLEARN_IF_SABOTAGE_U64_SWAP"]()

#: DIAGNOSTIC BISECT GUARDS (2026-08-29, the RTX 4090 hang). Each is a
#: build define, each is a no-op unless named, and NONE of them may appear
#: in a shipped build: they truncate the kernel so a hang can be placed.
#: `tools/diag/rtx4090_hang.sh` builds `checks/if_hang_probe.mojo` once
#: per guard on the rented box and records which truncations still hang.
#:   MOJOLEARN_IF_DIAG_ENTRY_RETURN  the build kernel returns at entry
#:   MOJOLEARN_IF_DIAG_GATHER_ONLY   returns after the subsample gather
#:                                   barrier (sampling + gather, no walk)
#:   MOJOLEARN_IF_DIAG_NO_REJECT     `sample_bounded` is `value % bound`,
#:                                   no rejection loop (the one data-
#:                                   dependent `while` in the walk)
#:   MOJOLEARN_IF_DIAG_NO_RECORD     `_record_decision` writes nothing
comptime DIAG_ENTRY_RETURN = is_defined["MOJOLEARN_IF_DIAG_ENTRY_RETURN"]()
comptime DIAG_GATHER_ONLY = is_defined["MOJOLEARN_IF_DIAG_GATHER_ONLY"]()
comptime DIAG_NO_REJECT = is_defined["MOJOLEARN_IF_DIAG_NO_REJECT"]()
comptime DIAG_NO_RECORD = is_defined["MOJOLEARN_IF_DIAG_NO_RECORD"]()


def curand_u64(mut rng_state: curandStateXORWOW) -> UInt64:
    """`curand_u64` (`:83-86`). DEVIATION 750: the FIRST draw is the HIGH
    word. Theirs is one expression, `(static_cast<uint64_t>(curand(s)) <<
    32) | curand(s)`, whose two operands C++ leaves UNSEQUENCED -- there
    is no spelling in Mojo (or in any language) that reproduces "either
    order", so the order is a choice this implementation had to make and name."""
    var first = UInt64(curand(rng_state))
    var second = UInt64(curand(rng_state))
    comptime if SAB_U64_SWAP:
        return (second << 32) | first
    else:
        return (first << 32) | second


def sample_bounded(mut rng_state: curandStateXORWOW, bound: UInt64) -> UInt64:
    """`sample_bounded(curandState*, size_t bound)`: bounded rejection
    sampling against `max_uint64 - (max_uint64 % bound)`, "avoids modulo
    bias" (their comment at `:290`)."""
    if bound <= 1:
        return 0
    var max_uint64: UInt64 = 0xFFFFFFFFFFFFFFFF
    var limit: UInt64 = max_uint64 - (max_uint64 % bound)
    var value = curand_u64(rng_state)
    comptime if DIAG_NO_REJECT:
        return value % bound
    while value >= limit:
        value = curand_u64(rng_state)
    return value % bound


def contains_sample(
    samples: MutPointer[Int64, MutAnyOrigin], n_samples: Int, candidate: Int64
) -> Bool:
    for i in range(n_samples):
        if samples.unsafe_load(i) == candidate:
            return True
    return False


def contains_int_sample(
    samples: MutPointer[Int32, MutAnyOrigin], n_samples: Int, candidate: Int32
) -> Bool:
    for i in range(n_samples):
        if samples.unsafe_load(i) == candidate:
            return True
    return False


# ---------------------------------------------------------------------------
# `:125-243`: build_tree_iterative_global. Shared by the device kernel
# (thread 0) and called by no one else; the host oracle is an independent
# transcription (`checks/if_oracle.mojo`) so a pointer slip here has a
# witness.
# ---------------------------------------------------------------------------


def _stack_push(
    stack: MutPointer[Int32, MutAnyOrigin],
    mut top: Int,
    node_idx: Int,
    start_idx: Int,
    end_idx: Int,
    depth: Int,
):
    stack.unsafe_store(4 * top + 0, Int32(node_idx))
    stack.unsafe_store(4 * top + 1, Int32(start_idx))
    stack.unsafe_store(4 * top + 2, Int32(end_idx))
    stack.unsafe_store(4 * top + 3, Int32(depth))
    top += 1


def _record_decision(
    decisions: MutPointer[Int32, MutAnyOrigin],
    node_idx: Int,
    min_val: Float32,
    max_val: Float32,
    rand_frac: Float32,
    feature_start: Int,
    local_feature: Int,
    flags: Int,
):
    """Write one node's decision record. Six Int32 words at
    `IF_DECISION_WORDS * node_idx`:

        0  min_val bits      the node's per-feature minimum, as float bits
        1  max_val bits      its maximum
        2  rand_frac bits    `curand_uniform`'s draw for this split
        3  feature_start     the random feature offset the retry loop began at
        4  flags             see below
        5  local_feature     the column chosen, -1 when none was

    flags, one bit each:

        1   this node is a leaf
        2   stopping arm `depth >= max_depth` held
        4   stopping arm `n_node_samples <= 1` held
        8   stopping arm `n_nodes + 2 > max_nodes_per_tree` held
        16  leaf because no feature had `min < max`
        32  THE REPARTITION FALLBACK FIRED

    WHY EACH ONE EARNS ITS BYTES. (1) `threshold = fma(rand_frac,
    ftz(max_val - min_val), min_val)`, so when `(max - min)` is small
    against `|min|` the whole product is ABSORBED and a divergence in
    either bound produces the IDENTICAL recorded threshold. Recording the
    two bounds and the draw is that absorption result applied verbatim.
    (2) The repartition fallback OVERWRITES `threshold` with `max_val`, so
    `structure.thr` is a hash taken AFTER a repair and whether the repair
    fired was recorded nowhere: one bit per node fixes it. (3) When two
    stopping arms hold at once the leaf written is byte-identical either
    way -- the `CRIT_ORDER` shape that holtwinters measured to move ZERO
    cells and to be catchable only by a recorded decision. (4)
    `feature_start` is the per-node draw that decides WHICH column is
    tried first, and it is invisible in the tree whenever the first
    candidate happens to be splittable."""
    comptime if DIAG_NO_RECORD:
        return
    var base = IF_DECISION_WORDS * node_idx
    decisions.unsafe_store(base + 0, bitcast[DType.int32](min_val))
    decisions.unsafe_store(base + 1, bitcast[DType.int32](max_val))
    decisions.unsafe_store(base + 2, bitcast[DType.int32](rand_frac))
    decisions.unsafe_store(base + 3, Int32(feature_start))
    decisions.unsafe_store(base + 4, Int32(flags))
    decisions.unsafe_store(base + 5, Int32(local_feature))


# ---------------------------------------------------------------------------
# cpu-gpu-cleanup t-forest: THE BLOCK BUILDS ITS TREE TOGETHER.
#
# Until this lane one thread of each block drew the subsample (bootstrap
# draws and Floyd's sampler on the tree's XORWOW stream) and then walked the
# whole tree alone: every node's per-feature min/max over its rows and its
# partition ran on thread 0 while the other threads of the block waited at
# the final barrier. Now:
#
#   * the SUBSAMPLE is a counter draw. With replacement, draw i is
#     `if_bootstrap_draw(base, i, n_rows)` (SplitMix64 keyed by (seed, tree,
#     i), rejection keyed by the attempt). Without replacement, the tree takes
#     the `k` rows with the smallest `(if_row_key(base, r), r)`, listed in row
#     order: a 4-bit-digit radix select of the k-th key (eight block-wide
#     integer count passes, no atomics), then one block scan that compacts
#     the chosen rows. Features are chosen the same way from the columns.
#     Every thread computes the same counts and the same digits, so no thread
#     owns the walk.
#   * the WALK is block-uniform: every thread runs the same stack and draws
#     the same XORWOW values (the node draws are the tree stream's, from its
#     start), so the control flow needs no broadcast; the data work of a node
#     is split over the block. The per-feature min/max is a block reduction
#     whose tie goes to the LOWER position, which is exactly the serial
#     strict-`<` / strict-`>` fold's answer (the first row in partition order
#     wins a signed-zero tie). The partition is STABLE (rows below the
#     threshold first, each side in its previous order) through two block
#     scans into the tree's scratch half of `work_indices`.
#
# Bits: integer counts and scans, compares and copies; the float arithmetic
# (threshold, path length) is the serial walk's. The answer does not depend
# on the block width. The host oracle (`checks/if_oracle.mojo`, the CPU-only
# install's fit) draws and partitions the same way. Changed from cuML's
# serial semantics (Floyd's sample, the in-place swap partition): every
# forest's bits move once, on every vendor and the CPU column together.
# ---------------------------------------------------------------------------

comptime IF_BUILD_TPB_MAX = 256
"""The widest build block the shared scratch is sized for (the launch
invariance gates run 32..256); the launch refuses a wider one."""

comptime IF_KEY_BUCKETS = 16
"""4-bit digits of the 32-bit selection key: eight count passes."""

comptime IF_GOLDEN: UInt64 = 0x9E3779B97F4A7C15
comptime IF_REJECT_STEP: UInt64 = 0xD1B54A32D192ED03

comptime SHI32 = UnsafePointer[Int32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]
comptime SHF32 = UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]
comptime SHU32 = UnsafePointer[UInt32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]


@always_inline
def _if_mix64(z_in: UInt64) -> UInt64:
    """SplitMix64's finalizer (Steele, Lea, Flood 2014)."""
    var z = z_in
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


@always_inline
def if_sample_base(seed: UInt64, tree: UInt64, stream: UInt64) -> UInt64:
    """The counter base of one tree's draws; stream 0 rows, stream 1
    features."""
    return _if_mix64(
        _if_mix64(seed + IF_GOLDEN) + tree * IF_GOLDEN + stream * IF_REJECT_STEP + 1
    )


@always_inline
def if_row_key(base: UInt64, r: Int) -> UInt32:
    """Row (or column) `r`'s selection key; the sample is the `k` smallest
    `(key, r)`."""
    return UInt32(_if_mix64(base + UInt64(r + 1) * IF_GOLDEN) >> 32)


@always_inline
def if_bootstrap_draw(base: UInt64, i: Int, bound: UInt64) -> UInt64:
    """Draw `i` with replacement from `[0, bound)`, bounded rejection on a
    counter (attempt `j` is `mix(v + j * step)`), so no modulo bias and no
    shared stream."""
    if bound <= 1:
        return 0
    var max_uint64: UInt64 = 0xFFFFFFFFFFFFFFFF
    var limit: UInt64 = max_uint64 - (max_uint64 % bound)
    var v = _if_mix64(base + UInt64(i + 1) * IF_GOLDEN)
    var j: UInt64 = 0
    while v >= limit:
        j += 1
        v = _if_mix64(v + j * IF_REJECT_STEP)
    return v % bound


@always_inline
def _block_scan(scan: SHI32, tid: Int, n_threads: Int, flag: Int32) -> Tuple[Int, Int]:
    """(exclusive prefix, block total) of `flag` over the block, a
    Hillis-Steele scan in shared memory. Integer, exact."""
    scan[tid] = flag
    barrier()
    var off = 1
    while off < n_threads:
        var add = Int32(0)
        if tid >= off:
            add = scan[tid - off]
        barrier()
        scan[tid] = scan[tid] + add
        barrier()
        off *= 2
    var inclusive = Int(scan[tid])
    var total = Int(scan[n_threads - 1])
    barrier()
    return (inclusive - Int(flag), total)


def _block_select_smallest(
    base: UInt64,
    n: Int,
    k: Int,
    out64: MutPointer[Int64, MutAnyOrigin],
    out32: MutPointer[Int32, MutAnyOrigin],
    wide: Bool,
    counts: SHU32,
    totals: SHU32,
    scan: SHI32,
    tid: Int,
    n_threads: Int,
):
    """The `k` of `[0, n)` with the smallest `(if_row_key(base, r), r)`, in
    increasing `r`, written to `out64` (`wide`) or `out32`. Block-uniform:
    every thread derives the same threshold from the same integer counts."""
    if k >= n:
        var r = tid
        while r < n:
            if wide:
                out64.unsafe_store(r, Int64(r))
            else:
                out32.unsafe_store(r, Int32(r))
            r += n_threads
        barrier()
        return
    var prefix = UInt32(0)
    var mask = UInt32(0)
    var k_rem = k
    for p in range(8):
        var shift = UInt32(28 - 4 * p)
        var local = InlineArray[UInt32, IF_KEY_BUCKETS](fill=UInt32(0))
        var r = tid
        while r < n:
            var key = if_row_key(base, r)
            if (key & mask) == prefix:
                var d = Int((key >> shift) & UInt32(15))
                local[d] = local[d] + UInt32(1)
            r += n_threads
        for d in range(IF_KEY_BUCKETS):
            counts[d * IF_BUILD_TPB_MAX + tid] = local[d]
        barrier()
        if tid < IF_KEY_BUCKETS:
            var total = UInt32(0)
            for t in range(n_threads):
                total += counts[tid * IF_BUILD_TPB_MAX + t]
            totals[tid] = total
        barrier()
        var cum = 0
        var digit = IF_KEY_BUCKETS - 1
        for d in range(IF_KEY_BUCKETS):
            var c = Int(totals[d])
            if cum + c >= k_rem:
                digit = d
                break
            cum += c
        k_rem -= cum
        prefix = prefix | (UInt32(digit) << shift)
        mask = mask | (UInt32(15) << shift)
        barrier()
    # `prefix` is the k-th smallest key; `k_rem` rows holding it are taken,
    # lowest row first
    var eq_before = 0
    var sel_before = 0
    var base_r = 0
    while base_r < n:
        var r = base_r + tid
        var less = False
        var eq = False
        if r < n:
            var key = if_row_key(base, r)
            less = key < prefix
            eq = key == prefix
        var eq_scan = _block_scan(scan, tid, n_threads, Int32(1) if eq else Int32(0))
        var take = less or (eq and eq_before + eq_scan[0] < k_rem)
        var take_scan = _block_scan(scan, tid, n_threads, Int32(1) if take else Int32(0))
        if take:
            var pos = sel_before + take_scan[0]
            if wide:
                out64.unsafe_store(pos, Int64(r))
            else:
                out32.unsafe_store(pos, Int32(r))
        eq_before += eq_scan[1]
        sel_before += take_scan[1]
        base_r += n_threads
    barrier()


@always_inline
def _better_min(av: Float32, ai: Int, bv: Float32, bi: Int) -> Bool:
    """True when (bv, bi) beats (av, ai) for the minimum: a smaller value,
    or an equal one at a lower position (the serial strict-`<` fold keeps
    the first). `ai < 0` is no element."""
    if bi < 0:
        return False
    if ai < 0:
        return True
    if bv < av:
        return True
    if av < bv:
        return False
    return bi < ai


@always_inline
def _better_max(av: Float32, ai: Int, bv: Float32, bi: Int) -> Bool:
    if bi < 0:
        return False
    if ai < 0:
        return True
    if bv > av:
        return True
    if av > bv:
        return False
    return bi < ai


def _block_min_max(
    local_data: MutPointer[Float32, MutAnyOrigin],
    work_indices: MutPointer[Int32, MutAnyOrigin],
    start: Int,
    end: Int,
    n_cols: Int,
    candidate: Int,
    minv: SHF32,
    mini: SHI32,
    maxv: SHF32,
    maxi: SHI32,
    tid: Int,
    n_threads: Int,
) -> Tuple[Float32, Float32]:
    """(min, max) of column `candidate` over the node's rows, each the
    serial positional fold's answer (ties to the lower position)."""
    var lmin = Float32(0.0)
    var lmin_i = -1
    var lmax = Float32(0.0)
    var lmax_i = -1
    var r = start + tid
    while r < end:
        var v = local_data.unsafe_load(Int(work_indices.unsafe_load(r)) * n_cols + candidate)
        if _better_min(lmin, lmin_i, v, r):
            lmin = v
            lmin_i = r
        if _better_max(lmax, lmax_i, v, r):
            lmax = v
            lmax_i = r
        r += n_threads
    minv[tid] = lmin
    mini[tid] = Int32(lmin_i)
    maxv[tid] = lmax
    maxi[tid] = Int32(lmax_i)
    barrier()
    var s = 1
    while s < n_threads:
        if tid % (2 * s) == 0 and tid + s < n_threads:
            if _better_min(minv[tid], Int(mini[tid]), minv[tid + s], Int(mini[tid + s])):
                minv[tid] = minv[tid + s]
                mini[tid] = mini[tid + s]
            if _better_max(maxv[tid], Int(maxi[tid]), maxv[tid + s], Int(maxi[tid + s])):
                maxv[tid] = maxv[tid + s]
                maxi[tid] = maxi[tid + s]
        barrier()
        s *= 2
    var out_min = minv[0]
    var out_max = maxv[0]
    barrier()
    return (out_min, out_max)


def _block_stable_partition(
    local_data: MutPointer[Float32, MutAnyOrigin],
    work_indices: MutPointer[Int32, MutAnyOrigin],
    temp: MutPointer[Int32, MutAnyOrigin],
    start: Int,
    end: Int,
    n_cols: Int,
    feature: Int,
    threshold: Float32,
    scan: SHI32,
    tid: Int,
    n_threads: Int,
) -> Int:
    """Rows of `[start, end)` with `value < threshold` first, then the rest,
    each side in its previous order; returns the split position."""
    var n_less = 0
    var base_r = start
    while base_r < end:
        var r = base_r + tid
        var f = Int32(0)
        var idx = Int32(0)
        if r < end:
            idx = work_indices.unsafe_load(r)
            if local_data.unsafe_load(Int(idx) * n_cols + feature) < threshold:
                f = Int32(1)
        var sc = _block_scan(scan, tid, n_threads, f)
        if f != 0:
            temp.unsafe_store(n_less + sc[0], idx)
        n_less += sc[1]
        base_r += n_threads
    var n_more = 0
    base_r = start
    while base_r < end:
        var r = base_r + tid
        var g = Int32(0)
        var idx = Int32(0)
        if r < end:
            idx = work_indices.unsafe_load(r)
            if not (local_data.unsafe_load(Int(idx) * n_cols + feature) < threshold):
                g = Int32(1)
        var sc = _block_scan(scan, tid, n_threads, g)
        if g != 0:
            temp.unsafe_store(n_less + n_more + sc[0], idx)
        n_more += sc[1]
        base_r += n_threads
    barrier()
    var r2 = start + tid
    while r2 < end:
        work_indices.unsafe_store(r2, temp.unsafe_load(r2 - start))
        r2 += n_threads
    barrier()
    return start + n_less


def build_tree_iterative_global(
    local_data: MutPointer[Float32, MutAnyOrigin],
    n_samples: Int,
    n_cols: Int,
    has_feature_indices: Bool,
    feature_indices: MutPointer[Int32, MutAnyOrigin],
    max_depth: Int,
    max_nodes_per_tree: Int,
    mut rng_state: curandStateXORWOW,
    node_feature: MutPointer[Int32, MutAnyOrigin],
    node_threshold: MutPointer[Float32, MutAnyOrigin],
    node_left: MutPointer[Int32, MutAnyOrigin],
    node_right: MutPointer[Int32, MutAnyOrigin],
    n_nodes_out: MutPointer[Int32, MutAnyOrigin],
    max_depth_out: MutPointer[Int32, MutAnyOrigin],
    work_indices: MutPointer[Int32, MutAnyOrigin],
    work_temp: MutPointer[Int32, MutAnyOrigin],
    stack: MutPointer[Int32, MutAnyOrigin],
    decisions: MutPointer[Int32, MutAnyOrigin],
    minv: SHF32,
    mini: SHI32,
    maxv: SHF32,
    maxi: SHI32,
    scan: SHI32,
    tid: Int,
    n_threads: Int,
):
    """`build_tree_iterative_global<T>` (`:125-243`), walked by the WHOLE
    block (cpu-gpu-cleanup t-forest, the comment block above). `local_data`
    is the tree's gathered subsample, row-major `n_samples x n_cols` where
    `n_cols` is `max_features`; `feature_indices` maps a local column to the
    original one when `has_feature_indices`. The node pointers are already
    offset to this tree. Every thread runs the same stack and draws the same
    XORWOW values; a node's min/max and partition are split over the block;
    single stores (nodes, records, results) are thread 0's. `work_temp` is
    the tree's partition scratch, `n_samples` words.

    `decisions` is the card's per-node decision slice, `IF_DECISION_WORDS`
    Int32 per node, written by `_record_decision` at every node the walk
    finishes. It is an OUTPUT ONLY: nothing in the build reads it back, so
    it cannot change a single bit of the tree."""
    var i = tid
    while i < n_samples:
        work_indices.unsafe_store(i, Int32(i))
        i += n_threads
    barrier()

    var n_nodes = 1
    var observed_max_depth = 0
    var stack_top = 0
    # every thread pushes the same words to the same cells and reads back
    # its own: the stack needs no barrier of its own
    _stack_push(stack, stack_top, 0, 0, n_samples, 0)

    while stack_top > 0:
        stack_top -= 1
        var node_idx = Int(stack.unsafe_load(4 * stack_top + 0))
        var start = Int(stack.unsafe_load(4 * stack_top + 1))
        var end = Int(stack.unsafe_load(4 * stack_top + 2))
        var depth = Int(stack.unsafe_load(4 * stack_top + 3))
        var n_node_samples = end - start
        observed_max_depth = (
            observed_max_depth if observed_max_depth > depth else depth
        )

        # `:158`, cuML's defensive guard; unreachable (every pushed index is
        # below the capacity, see the stopping condition's capacity arm).
        if node_idx >= max_nodes_per_tree:
            continue

        var stop_depth = depth >= max_depth
        var stop_isolated = n_node_samples <= 1
        var stop_capacity = n_nodes + 2 > max_nodes_per_tree
        if stop_depth or stop_isolated or stop_capacity:
            if tid == 0:
                var path_length = ftz(
                    Float32(depth) + compute_c_n(n_node_samples)
                )
                node_feature.unsafe_store(node_idx, Int32(-1))
                node_threshold.unsafe_store(node_idx, path_length)
                node_left.unsafe_store(node_idx, Int32(-1))
                node_right.unsafe_store(node_idx, Int32(-1))
                var flags = 1
                if stop_depth:
                    flags += 2
                if stop_isolated:
                    flags += 4
                if stop_capacity:
                    flags += 8
                _record_decision(
                    decisions, node_idx, Float32(0.0), Float32(0.0),
                    Float32(0.0), -1, -1, flags,
                )
            continue

        # Try every feature, starting from a random offset, before concluding
        # that this node cannot be split; each candidate's min/max is a block
        # reduction, the first splittable candidate in order wins.
        var local_feature = -1
        var min_val = Float32(0.0)
        var max_val = Float32(0.0)
        var feature_start = Int(sample_bounded(rng_state, UInt64(n_cols)))
        for attempt in range(n_cols):
            var candidate = (feature_start + attempt) % n_cols
            var mm = _block_min_max(
                local_data, work_indices, start, end, n_cols, candidate,
                minv, mini, maxv, maxi, tid, n_threads,
            )
            if mm[0] < mm[1]:
                local_feature = candidate
                min_val = mm[0]
                max_val = mm[1]
                break

        if local_feature < 0:
            if tid == 0:
                var path_length = ftz(
                    Float32(depth) + compute_c_n(n_node_samples)
                )
                node_feature.unsafe_store(node_idx, Int32(-1))
                node_threshold.unsafe_store(node_idx, path_length)
                node_left.unsafe_store(node_idx, Int32(-1))
                node_right.unsafe_store(node_idx, Int32(-1))
                _record_decision(
                    decisions, node_idx, Float32(0.0), Float32(0.0),
                    Float32(0.0), feature_start, -1, 1 + 16,
                )
            continue

        var original_feature = Int32(local_feature)
        if has_feature_indices:
            original_feature = feature_indices.unsafe_load(local_feature)
        var rand_frac = curand_uniform(rng_state)
        # T threshold = min_val + static_cast<T>(rand_frac) * (max_val - min_val);
        var threshold = ftz(
            identical_mul_add(rand_frac, ftz(max_val - min_val), min_val)
        )

        var left_end = _block_stable_partition(
            local_data, work_indices, work_temp, start, end, n_cols,
            local_feature, threshold, scan, tid, n_threads,
        )
        var repartitioned = left_end == start or left_end == end
        if repartitioned:
            # Numerical rounding can move the random threshold onto an
            # endpoint. Repartition with max_val so the stored split and the
            # training partition agree; min_val < max_val, so both children
            # are nonempty.
            threshold = max_val
            left_end = _block_stable_partition(
                local_data, work_indices, work_temp, start, end, n_cols,
                local_feature, threshold, scan, tid, n_threads,
            )

        var left_child = n_nodes
        var right_child = n_nodes + 1
        n_nodes += 2
        if tid == 0:
            var split_flags = 0
            if repartitioned:
                split_flags += 32
            _record_decision(
                decisions, node_idx, min_val, max_val, rand_frac,
                feature_start, local_feature, split_flags,
            )
            node_feature.unsafe_store(node_idx, original_feature)
            node_threshold.unsafe_store(node_idx, threshold)
            node_left.unsafe_store(node_idx, Int32(left_child))
            node_right.unsafe_store(node_idx, Int32(right_child))

        _stack_push(stack, stack_top, right_child, left_end, end, depth + 1)
        _stack_push(stack, stack_top, left_child, start, left_end, depth + 1)

    if tid == 0:
        n_nodes_out.unsafe_store(0, Int32(n_nodes))
        max_depth_out.unsafe_store(0, Int32(observed_max_depth))


# ---------------------------------------------------------------------------
# `:245-340`: build_isolation_trees_global_kernel
# ---------------------------------------------------------------------------


def build_isolation_trees_global_kernel(
    data: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int64,
    n_cols_in: Int32,
    n_trees_in: Int32,
    max_samples_in: Int32,
    max_features_in: Int32,
    max_depth_in: Int32,
    max_nodes_per_tree_in: Int32,
    bootstrap_in: Int32,
    seed: UInt64,
    has_feature_indices_in: Int32,
    feature_indices: MutPointer[Int32, MutAnyOrigin],
    node_feature: MutPointer[Int32, MutAnyOrigin],
    node_threshold: MutPointer[Float32, MutAnyOrigin],
    node_left: MutPointer[Int32, MutAnyOrigin],
    node_right: MutPointer[Int32, MutAnyOrigin],
    tree_offsets: MutPointer[Int32, MutAnyOrigin],
    tree_n_nodes: MutPointer[Int32, MutAnyOrigin],
    tree_max_depth: MutPointer[Int32, MutAnyOrigin],
    subsample_buffer: MutPointer[Float32, MutAnyOrigin],
    sample_indices: MutPointer[Int64, MutAnyOrigin],
    work_indices: MutPointer[Int32, MutAnyOrigin],
    stack: MutPointer[Int32, MutAnyOrigin],
    xorwow_sequence_table: MutPointer[UInt32, MutAnyOrigin],
    xorwow_offset_table: MutPointer[UInt32, MutAnyOrigin],
    global_tree_start: Int32 = 0,
):
    """`build_isolation_trees_global_kernel<T>` (`:245-340`): `tree_id =
    blockIdx.x`; `curand_init(seed, tree_id, 0)`; thread 0 samples rows
    (bootstrap: with replacement; else Floyd's without replacement,
    `:291-305`) and features (`:307-315`); every thread gathers the
    subsample into `local_data` (`:319-326`, column-major source); then
    `build_tree_iterative_global`. `data` is column-major `n_rows x
    n_cols` exactly as theirs (`isolation_forest.hpp:118`)."""
    comptime if DIAG_ENTRY_RETURN:
        return
    var tree_id = Int(block_idx.x)
    var n_trees = Int(n_trees_in)
    if tree_id >= n_trees:
        return
    var n_rows = Int(n_rows_in)
    var n_cols = Int(n_cols_in)
    var max_samples = Int(max_samples_in)
    var max_features = Int(max_features_in)
    var max_depth = Int(max_depth_in)
    var max_nodes_per_tree = Int(max_nodes_per_tree_in)
    var bootstrap = bootstrap_in != 0
    var has_feature_indices = has_feature_indices_in != 0
    var tid = Int(thread_idx.x)
    var n_threads = Int(block_dim.x)

    var rng_state = curandStateXORWOW.zero()
    curand_init(
        seed,
        UInt64(tree_id + Int(global_tree_start)),
        UInt64(0),
        rng_state,
        xorwow_sequence_table,
        xorwow_offset_table,
    )

    var tree_offset = tree_id * max_nodes_per_tree
    if tid == 0:
        tree_offsets.unsafe_store(tree_id, Int32(tree_offset))

    var local_data = subsample_buffer.unsafe_offset(tree_id * max_samples * max_features)
    var tree_sample_indices = sample_indices.unsafe_offset(tree_id * max_samples)
    var tree_feature_indices = feature_indices.unsafe_offset(tree_id * max_features)
    # two halves per tree: the partition order, then its scratch
    var tree_work_indices = work_indices.unsafe_offset(tree_id * 2 * max_samples)
    var tree_work_temp = work_indices.unsafe_offset(tree_id * 2 * max_samples + max_samples)
    var sh_counts = stack_allocation[
        IF_KEY_BUCKETS * IF_BUILD_TPB_MAX, Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()
    var sh_totals = stack_allocation[
        IF_KEY_BUCKETS, Scalar[DType.uint32], address_space = AddressSpace.SHARED
    ]()
    var sh_scan = stack_allocation[
        IF_BUILD_TPB_MAX, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sh_minv = stack_allocation[
        IF_BUILD_TPB_MAX, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var sh_mini = stack_allocation[
        IF_BUILD_TPB_MAX, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sh_maxv = stack_allocation[
        IF_BUILD_TPB_MAX, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var sh_maxi = stack_allocation[
        IF_BUILD_TPB_MAX, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    # ONE buffer, THREE disjoint slices, so the kernel's argument count
    # stays at 25 (Metal caps a kernel at 31; holtwinters died on the 32nd).
    #   [0, 4*mn)              the stack
    #   [4*mn, 10*mn)          the per-node decision records
    #   [10*mn, 10*mn + 6)     this tree's final RNG state
    var tree_scratch_base = tree_id * (
        max_nodes_per_tree * IF_SCRATCH_WORDS_PER_NODE + IF_RNG_STATE_WORDS
    )
    var tree_stack = stack.unsafe_offset(tree_scratch_base)
    var tree_decisions = stack.unsafe_offset(
        tree_scratch_base + max_nodes_per_tree * IF_STACK_WORDS
    )
    var tree_rng_out = stack.unsafe_offset(
        tree_scratch_base + max_nodes_per_tree * IF_SCRATCH_WORDS_PER_NODE
    )
    var t_feature = node_feature.unsafe_offset(tree_offset)
    var t_threshold = node_threshold.unsafe_offset(tree_offset)
    var t_left = node_left.unsafe_offset(tree_offset)
    var t_right = node_right.unsafe_offset(tree_offset)

    # The subsample, drawn by the whole block (the comment block above
    # `IF_BUILD_TPB_MAX`): bootstrap=True samples with replacement, one
    # counter draw per sample; bootstrap=False takes the `max_samples` rows
    # of smallest key, in row order. Features likewise without replacement.
    var global_tree = UInt64(tree_id + Int(global_tree_start))
    var row_base = if_sample_base(seed, global_tree, 0)
    if bootstrap:
        var si = tid
        while si < max_samples:
            tree_sample_indices.unsafe_store(
                si, Int64(if_bootstrap_draw(row_base, si, UInt64(n_rows)))
            )
            si += n_threads
        barrier()
    else:
        _block_select_smallest(
            row_base, n_rows, max_samples, tree_sample_indices,
            tree_feature_indices, True, sh_counts, sh_totals, sh_scan,
            tid, n_threads,
        )
    if has_feature_indices:
        _block_select_smallest(
            if_sample_base(seed, global_tree, 1), n_cols, max_features,
            tree_sample_indices, tree_feature_indices, False, sh_counts,
            sh_totals, sh_scan, tid, n_threads,
        )
    barrier()

    for s in range(max_samples):
        var src_row = Int(tree_sample_indices.unsafe_load(s))
        var f = tid
        while f < max_features:
            var src_col = f
            if has_feature_indices:
                src_col = Int(tree_feature_indices.unsafe_load(f))
            comptime if IF_FAST_ROWMAJOR:
                local_data.unsafe_store(
                    s * max_features + f,
                    data.unsafe_load(src_row * n_cols + src_col),
                )
            else:
                local_data.unsafe_store(
                    s * max_features + f,
                    data.unsafe_load(src_row + src_col * n_rows),
                )
            f += n_threads
    barrier()
    comptime if DIAG_GATHER_ONLY:
        return

    build_tree_iterative_global(
        local_data,
        max_samples,
        max_features,
        has_feature_indices,
        tree_feature_indices,
        max_depth,
        max_nodes_per_tree,
        rng_state,
        t_feature,
        t_threshold,
        t_left,
        t_right,
        tree_n_nodes.unsafe_offset(tree_id),
        tree_max_depth.unsafe_offset(tree_id),
        tree_work_indices,
        tree_work_temp,
        tree_stack,
        tree_decisions,
        sh_minv,
        sh_mini,
        sh_maxv,
        sh_maxi,
        sh_scan,
        tid,
        n_threads,
    )

    # The stream POSITION this tree finished at. `if.rng.probe` verifies the
    # IMPLEMENTATION (that our XORWOW is cuRAND's); this verifies that a tree consumed
    # the draws we think it did. A vendor that took one extra rejection in
    # `sample_bounded`, or one fewer, lands here and nowhere else.
    if tid == 0:
        tree_rng_out.unsafe_store(0, bitcast[DType.int32](rng_state.d))
        tree_rng_out.unsafe_store(1, bitcast[DType.int32](rng_state.v0))
        tree_rng_out.unsafe_store(2, bitcast[DType.int32](rng_state.v1))
        tree_rng_out.unsafe_store(3, bitcast[DType.int32](rng_state.v2))
        tree_rng_out.unsafe_store(4, bitcast[DType.int32](rng_state.v3))
        tree_rng_out.unsafe_store(5, bitcast[DType.int32](rng_state.v4))


# ---------------------------------------------------------------------------
# `:342-375`: traversal and the path-length kernel
# ---------------------------------------------------------------------------


def traverse_global_tree(
    node_feature: MutPointer[Int32, MutAnyOrigin],
    node_threshold: MutPointer[Float32, MutAnyOrigin],
    node_left: MutPointer[Int32, MutAnyOrigin],
    node_right: MutPointer[Int32, MutAnyOrigin],
    tree_offset: Int,
    sample: MutPointer[Float32, MutAnyOrigin],
) -> Float32:
    """`traverse_global_tree<T>` (`:342-355`): descend by `val <
    threshold`; a leaf (`feature_idx < 0`) returns its stored path
    length."""
    var node_idx = 0
    while True:
        var f = Int(node_feature.unsafe_load(tree_offset + node_idx))
        var thr = node_threshold.unsafe_load(tree_offset + node_idx)
        if f < 0:
            return thr
        var val = sample.unsafe_load(f)
        if val < thr:
            node_idx = Int(node_left.unsafe_load(tree_offset + node_idx))
        else:
            node_idx = Int(node_right.unsafe_load(tree_offset + node_idx))
    return Float32(0.0)


def compute_path_lengths_global_kernel(
    data: MutPointer[Float32, MutAnyOrigin],
    n_samples_in: Int64,
    n_cols_in: Int32,
    node_feature: MutPointer[Int32, MutAnyOrigin],
    node_threshold: MutPointer[Float32, MutAnyOrigin],
    node_left: MutPointer[Int32, MutAnyOrigin],
    node_right: MutPointer[Int32, MutAnyOrigin],
    tree_offsets: MutPointer[Int32, MutAnyOrigin],
    n_trees_in: Int32,
    path_lengths: MutPointer[Float32, MutAnyOrigin],
):
    """`compute_path_lengths_global_kernel<T>` (`:357-375`): one thread per
    ROW-MAJOR sample, `total_path += traverse(tree t)` for t ascending,
    then `/ n_trees`. The sum is a serial fold whose order is a pure
    function of `n_trees`; nothing crosses threads."""
    var sample_idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if sample_idx >= Int(n_samples_in):
        return
    var n_cols = Int(n_cols_in)
    var n_trees = Int(n_trees_in)
    var sample = data.unsafe_offset(sample_idx * n_cols)
    var total_path = Float32(0.0)
    for t in range(n_trees):
        var off = Int(tree_offsets.unsafe_load(t))
        total_path = ftz(
            total_path
            + traverse_global_tree(
                node_feature, node_threshold, node_left, node_right, off, sample
            )
        )
    var out = Float32(0.0)
    if n_trees > 0:
        out = ftz(total_path / Float32(n_trees))
    path_lengths.unsafe_store(sample_idx, out)


def compute_path_lengths_range_kernel(
    data: MutPointer[Float32, MutAnyOrigin],
    n_samples_in: Int64,
    n_cols_in: Int32,
    node_feature: MutPointer[Int32, MutAnyOrigin],
    node_threshold: MutPointer[Float32, MutAnyOrigin],
    node_left: MutPointer[Int32, MutAnyOrigin],
    node_right: MutPointer[Int32, MutAnyOrigin],
    tree_offsets: MutPointer[Int32, MutAnyOrigin],
    n_local_trees: Int32,
    n_global_trees: Int32,
    finalize: Int32,
    path_lengths: MutPointer[Float32, MutAnyOrigin],
):
    """Continue the original serial tree fold; divide only after its last tree.

    The incoming value is the accumulator after every preceding global tree,
    not a separately rounded shard sum. Traversal and each FTZ add are the
    same operations as compute_path_lengths_global_kernel.
    """
    var sample_idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if sample_idx >= Int(n_samples_in):
        return
    var sample = data.unsafe_offset(sample_idx * Int(n_cols_in))
    var total_path = path_lengths.unsafe_load(sample_idx)
    for t in range(Int(n_local_trees)):
        var off = Int(tree_offsets.unsafe_load(t))
        total_path = ftz(
            total_path
            + traverse_global_tree(
                node_feature, node_threshold, node_left, node_right, off, sample
            )
        )
    if finalize != 0:
        total_path = ftz(total_path / Float32(n_global_trees))
    path_lengths.unsafe_store(sample_idx, total_path)


def if_finite_scan_kernel(
    data: MutPointer[Float32, MutAnyOrigin],
    n: Int64,
    pad: Int64,
    poison: Float32,
    flag: MutPointer[Int32, MutAnyOrigin],
):
    """IF_FAST_ROWMAJOR's device half of DEVIATION 680's finiteness scan:
    every thread walks the cells `i = gid, gid + grid, ...` of `[0, n)`
    and stores 1 to `flag` on a non-finite one (every writer stores the
    same word), and writes `poison` into the `pad` tail the column-major
    stage wrote on the host."""
    var gid = Int64(block_idx.x) * Int64(block_dim.x) + Int64(thread_idx.x)
    var stride = Int64(grid_dim.x) * Int64(block_dim.x)
    var bad = False
    var i = gid
    while i < n:
        var bits = bitcast[DType.uint32](data.unsafe_load(Int(i)))
        if (bits & UInt32(0x7F800000)) == UInt32(0x7F800000):
            bad = True
        i += stride
    if bad:
        flag.unsafe_store(0, Int32(1))
    if gid < pad:
        data.unsafe_store(Int(n + gid), poison)


def if_finite_scan_ftz_kernel(
    data: MutPointer[Float32, MutAnyOrigin],
    n: Int64,
    pad: Int64,
    poison: Float32,
    flag: MutPointer[Int32, MutAnyOrigin],
):
    """Lane fam-forests (`IDN_IF_QUERY_DEVICE`): `if_finite_scan_kernel`
    that also stores `ftz(cell)` back in place, so a raw host-pointer copy of
    a ROW-major query block becomes the words `_upload_f32` stages on the
    host (`ftz` per cell, `poison` in the `pad` tail). Each cell is read and
    written by exactly one thread (`i = gid, gid + grid, ...`)."""
    var gid = Int64(block_idx.x) * Int64(block_dim.x) + Int64(thread_idx.x)
    var stride = Int64(grid_dim.x) * Int64(block_dim.x)
    var bad = False
    var i = gid
    while i < n:
        var v = data.unsafe_load(Int(i))
        var bits = bitcast[DType.uint32](v)
        if (bits & UInt32(0x7F800000)) == UInt32(0x7F800000):
            bad = True
        data.unsafe_store(Int(i), ftz(v))
        i += stride
    if bad:
        flag.unsafe_store(0, Int32(1))
    if gid < pad:
        data.unsafe_store(Int(n + gid), poison)
