"""The batched Isolation Forest tree builder: one block per tree, thread 0
walks the stack.

MIRRORS `cpp/src/isolation_forest/isolation_tree_builder.cuh` at
rapidsai/cuml v26.08.00, branch for branch and loop for loop:
`StackEntry` (`:37-42`), `compute_c_n` (`:48-55`), `IFNode` (`:63-69`),
`curand_u64` / `sample_bounded` (`:84-101`), `contains_sample` /
`contains_int_sample` (`:103-117`), `build_tree_iterative_global`
(`:125-243`), `build_isolation_trees_global_kernel` (`:245-340`),
`traverse_global_tree` (`:342-355`), `compute_path_lengths_global_kernel`
(`:357-375`) and `build_isolation_forest_global` (`:377-420`). The
compaction pair (`:422-502`) is host bookkeeping for Treelite export and
is NOT ported (UNPORTED.tsv).

Their design, kept: all trees in ONE launch (`<<<n_trees, 128>>>`), the
subsample gathered into a contiguous per-tree buffer by every thread of
the block, then THREAD 0 ALONE draws from one `curandState` seeded
`(seed, tree_id)` and builds the tree iteratively through a global-memory
stack and partition index array. The RNG consumption order is therefore
the serial stack walk, a pure function of (seed, tree_id, data bits), and
nothing in the tree depends on block size, grid shape or launch order.
The RNG itself is `isolation_forest/ported/curand/curand_kernel.mojo`
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
`mojo_only/numerics.mojo`. Theirs: `log` (CUDA libm, vendor bits), a
contraction the compiler decides, no denormal policy. Ours under
IDENTICAL: `identical_log` in `compute_c_n`, `identical_mul_add` for the
threshold, `ftz` on every stored float (leaf path length, threshold, the
per-sample path-length accumulator); under FAST the stdlib / naive
spellings, so Apple's FAST bits do not move. Measured: the std-exp
sabotage in the README (the scorer's `identical_pow` -> `**`) moves
scores device-vs-oracle under IDENTICAL; on Apple the mul-add pin is
bit-inert (Metal contracts) exactly as numerics.mojo says.

DEVIATION 685. NODE STORAGE IS FOUR ARRAYS, NOT AN ARRAY OF `IFNode`.
Theirs: `IFNode<T>{int feature_idx; T threshold; int left_child; int
right_child;}` in one `rmm::device_buffer`. Ours: `node_feature` (Int32),
`node_threshold` (Float32), `node_left` (Int32), `node_right` (Int32),
same indices, same `tree_offsets`. WHAT is said is unchanged (every field,
every index); HOW changed because a whole-struct load through a pointer
in a kernel is a known Metal-compiler wall (PORTING_RULES.md rule 4) and
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
fix it rather than port it. Fixed here by making the order explicit and
naming it: the first draw is the high word (two named locals, so no
reader and no compiler has a choice left). That is also what the Python
reference (`mojo_only/xorwow_reference.py`) and the host oracle assume,
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
(`mojo_only/xorwow_reference.tsv`, seed 42 tree 0, verified against
`curand_precalc.h`'s own constants), so:

    high word first (ours)   -> 408
    low word first (swapped) -> 23

408 says the order above is theirs; 23 says flip it. The swapped-order arm
is built in as `-D MOJOLEARN_IF_SABOTAGE_U64_SWAP=1`, which must turn the
whole gate red -- that is the lane's proof the order is load-bearing rather
than a preference. UNPORTED.tsv carries this as the lane's one open
cross-vendor question.

DEVIATION 686. `size_t` INDICES ARE `Int64`, `StackEntry` IS FOUR `Int32`
WORDS. Their `sample_indices` is `size_t*` and the stack an array of
structs; ours are an `Int64` buffer and a flat `Int32` buffer of four
words per entry (`node_idx, start_idx, end_idx, depth`), same LIFO
discipline, same push order (right then left). Spelling only.
===================================================
"""

from std.memory import bitcast

from std.sys import is_defined

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.sync import barrier

from isolation_forest.ported.curand.curand_kernel import (
    curandStateXORWOW,
    curand,
    curand_init,
    curand_uniform,
)
from mojo_only.numerics import ftz, identical_log, identical_mul_add


comptime EULER_MASCHERONI_F32 = Float32(0.5772156649015329)
"""`T(0.5772156649015329)` with T = float: 0x3f13c468. Printed and gated
as hex by `if_check.mojo` so the constant cannot drift by a decimal."""

comptime IF_BUILD_TPB = 128
"""`build_isolation_trees_global_kernel<T><<<n_trees, 128, 0, stream>>>`
(`:397`). A scheduling width: only the gather uses the block's threads."""

comptime IF_DECISION_WORDS = 6
"""Int32 words of RECORDED DECISION per node (CARD_GAPS.md's isolation
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
    return ftz(Float32(2.0) * h - tail)


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


def curand_u64(mut rng_state: curandStateXORWOW) -> UInt64:
    """`curand_u64` (`:83-86`). DEVIATION 750: the FIRST draw is the HIGH
    word. Theirs is one expression, `(static_cast<uint64_t>(curand(s)) <<
    32) | curand(s)`, whose two operands C++ leaves UNSEQUENCED -- there
    is no spelling in Mojo (or in any language) that reproduces "either
    order", so the order is a choice this port had to make and name."""
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
# transcription (`mojo_only/if_oracle.mojo`) so a pointer slip here has a
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
    var base = IF_DECISION_WORDS * node_idx
    decisions.unsafe_store(base + 0, bitcast[DType.int32](min_val))
    decisions.unsafe_store(base + 1, bitcast[DType.int32](max_val))
    decisions.unsafe_store(base + 2, bitcast[DType.int32](rand_frac))
    decisions.unsafe_store(base + 3, Int32(feature_start))
    decisions.unsafe_store(base + 4, Int32(flags))
    decisions.unsafe_store(base + 5, Int32(local_feature))


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
    stack: MutPointer[Int32, MutAnyOrigin],
    decisions: MutPointer[Int32, MutAnyOrigin],
    tid: Int,
    n_threads: Int,
):
    """`build_tree_iterative_global<T>` (`:125-243`). `local_data` is the
    tree's gathered subsample, row-major `n_samples x n_cols` where
    `n_cols` is `max_features`; `feature_indices` maps a local column to
    the original one when `has_feature_indices`. The node pointers are
    already offset to this tree. `tid`/`n_threads` are the block's thread
    index and width (the `work_indices` fill is the only parallel line;
    everything after `if tid == 0` is one thread's serial walk).

    `decisions` is the card's per-node decision slice, `IF_DECISION_WORDS`
    Int32 per node, written by `_record_decision` at every node the walk
    finishes. It is an OUTPUT ONLY: nothing in the build reads it back, so
    it cannot change a single bit of the tree."""
    var i = tid
    while i < n_samples:
        work_indices.unsafe_store(i, Int32(i))
        i += n_threads
    barrier()

    if tid == 0:
        var n_nodes = 1
        var observed_max_depth = 0
        var stack_top = 0
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

            # `:158`, ported verbatim. THIS BRANCH IS UNREACHABLE, and the
            # proof is short enough to keep next to it so nobody writes a
            # gate that can never fire (adjudicated 2026-08-24; the
            # repo-wide card audit flagged it as a possible dangling-node
            # defect, CARD_GAPS.md:232-236).
            #
            #   `max_nodes_per_tree = min(2*max_samples - 1,
            #   2^(max_depth+1) - 1)` with `max_samples >= 1` and
            #   `max_depth >= 0` both enforced by a raise in
            #   `compute_global_max_nodes_per_tree`, so it is >= 1.
            #
            #   There are exactly THREE sites that ever push a node index:
            #   the root push of 0, and the two child pushes below. 0 <
            #   max_nodes_per_tree because the bound is >= 1. The child
            #   pushes are reachable ONLY past the capacity arm of the
            #   stopping condition, so at the moment they run
            #   `n_nodes + 2 <= max_nodes_per_tree`, and they push
            #   `left_child = n_nodes <= max_nodes_per_tree - 2` and
            #   `right_child = n_nodes + 1 <= max_nodes_per_tree - 1`.
            #   Both are < max_nodes_per_tree, and an index does not change
            #   after it is pushed.
            #
            #   Therefore every popped `node_idx` is < max_nodes_per_tree
            #   and the `continue` never runs. It is cuML's defensive
            #   spelling and we keep it because we mirror them, not because
            #   any shape reaches it. NO DEVIATION: theirs is identical
            #   (`isolation_tree_builder.cuh:158`), so there is no bug of
            #   theirs to fix and nothing of ours to number.
            #
            # If it ever DID run it would be a real defect, not a hash gap:
            # the node stays at the poison fill while its parent still
            # points at it. It would NOT be invisible, though -- the audit's
            # secondary claim is wrong. A dropped index is always < n_nodes
            # (it was produced by incrementing n_nodes), and the card
            # records `for i in range(n_used)` with `n_used = n_nodes`
            # (`isolation_forest.mojo:608,627`), so a poisoned dangling
            # child lands INSIDE the hashed range and diverges from the
            # oracle on `structure.feat` at that index.
            if node_idx >= max_nodes_per_tree:
                continue

            # Stopping condition: max depth, isolated sample, or exhausted capacity.
            # EVERY arm is evaluated for the record, not just the first
            # one `or` stops at: when two hold, the leaf written is
            # byte-identical either way, and which held is the only thing
            # that can tell two vendors apart here.
            var stop_depth = depth >= max_depth
            var stop_isolated = n_node_samples <= 1
            var stop_capacity = n_nodes + 2 > max_nodes_per_tree
            if stop_depth or stop_isolated or stop_capacity:
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
                    decisions,
                    node_idx,
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                    -1,
                    -1,
                    flags,
                )
                continue

            # Try every feature, starting from a random offset, before concluding
            # that this node cannot be split.
            var local_feature = -1
            var min_val = Float32(0.0)
            var max_val = Float32(0.0)
            var feature_start = Int(sample_bounded(rng_state, UInt64(n_cols)))
            for attempt in range(n_cols):
                var candidate = (feature_start + attempt) % n_cols
                var candidate_min = local_data.unsafe_load(
                    Int(work_indices.unsafe_load(start)) * n_cols + candidate
                )
                var candidate_max = candidate_min
                for r in range(start + 1, end):
                    var val = local_data.unsafe_load(
                        Int(work_indices.unsafe_load(r)) * n_cols + candidate
                    )
                    if val < candidate_min:
                        candidate_min = val
                    if val > candidate_max:
                        candidate_max = val
                if candidate_min < candidate_max:
                    local_feature = candidate
                    min_val = candidate_min
                    max_val = candidate_max
                    break

            if local_feature < 0:
                var path_length = ftz(
                    Float32(depth) + compute_c_n(n_node_samples)
                )
                node_feature.unsafe_store(node_idx, Int32(-1))
                node_threshold.unsafe_store(node_idx, path_length)
                node_left.unsafe_store(node_idx, Int32(-1))
                node_right.unsafe_store(node_idx, Int32(-1))
                _record_decision(
                    decisions,
                    node_idx,
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                    feature_start,
                    -1,
                    1 + 16,
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

            var left_end = start
            for r in range(start, end):
                var val = local_data.unsafe_load(
                    Int(work_indices.unsafe_load(r)) * n_cols + local_feature
                )
                if val < threshold:
                    var tmp = work_indices.unsafe_load(left_end)
                    work_indices.unsafe_store(
                        left_end, work_indices.unsafe_load(r)
                    )
                    work_indices.unsafe_store(r, tmp)
                    left_end += 1

            var repartitioned = left_end == start or left_end == end
            if repartitioned:
                # Numerical rounding can move the random threshold onto an endpoint.
                # Repartition with max_val so the stored split and training partition
                # remain consistent. Since min_val < max_val, both children are nonempty.
                threshold = max_val
                left_end = start
                for r in range(start, end):
                    var val = local_data.unsafe_load(
                        Int(work_indices.unsafe_load(r)) * n_cols + local_feature
                    )
                    if val < threshold:
                        var tmp = work_indices.unsafe_load(left_end)
                        work_indices.unsafe_store(
                            left_end, work_indices.unsafe_load(r)
                        )
                        work_indices.unsafe_store(r, tmp)
                        left_end += 1

            # AFTER the fallback, so the record says whether the stored
            # `structure.thr` is the drawn threshold or the repaired one.
            var split_flags = 0
            if repartitioned:
                split_flags += 32
            _record_decision(
                decisions,
                node_idx,
                min_val,
                max_val,
                rand_frac,
                feature_start,
                local_feature,
                split_flags,
            )

            var left_child = n_nodes
            var right_child = n_nodes + 1
            n_nodes += 2

            node_feature.unsafe_store(node_idx, original_feature)
            node_threshold.unsafe_store(node_idx, threshold)
            node_left.unsafe_store(node_idx, Int32(left_child))
            node_right.unsafe_store(node_idx, Int32(right_child))

            _stack_push(stack, stack_top, right_child, left_end, end, depth + 1)
            _stack_push(stack, stack_top, left_child, start, left_end, depth + 1)

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
):
    """`build_isolation_trees_global_kernel<T>` (`:245-340`): `tree_id =
    blockIdx.x`; `curand_init(seed, tree_id, 0)`; thread 0 samples rows
    (bootstrap: with replacement; else Floyd's without replacement,
    `:291-305`) and features (`:307-315`); every thread gathers the
    subsample into `local_data` (`:319-326`, column-major source); then
    `build_tree_iterative_global`. `data` is column-major `n_rows x
    n_cols` exactly as theirs (`isolation_forest.hpp:118`)."""
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
        UInt64(tree_id),
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
    var tree_work_indices = work_indices.unsafe_offset(tree_id * max_samples)
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

    # Thread 0 samples source rows using sklearn IsolationForest semantics:
    # bootstrap=True samples with replacement; bootstrap=False samples without
    # replacement. Bounded rejection sampling avoids modulo bias.
    if tid == 0:
        if bootstrap:
            for i in range(max_samples):
                tree_sample_indices.unsafe_store(
                    i, Int64(sample_bounded(rng_state, UInt64(n_rows)))
                )
        else:
            var start = n_rows - max_samples
            for i in range(max_samples):
                var j = start + i
                var t = Int64(sample_bounded(rng_state, UInt64(j + 1)))
                if contains_sample(tree_sample_indices, i, t):
                    tree_sample_indices.unsafe_store(i, Int64(j))
                else:
                    tree_sample_indices.unsafe_store(i, t)

        if has_feature_indices:
            var start = n_cols - max_features
            for i in range(max_features):
                var j = start + i
                var t = Int32(sample_bounded(rng_state, UInt64(j + 1)))
                if contains_int_sample(tree_feature_indices, i, t):
                    tree_feature_indices.unsafe_store(i, Int32(j))
                else:
                    tree_feature_indices.unsafe_store(i, t)
    barrier()

    for s in range(max_samples):
        var src_row = Int(tree_sample_indices.unsafe_load(s))
        var f = tid
        while f < max_features:
            var src_col = f
            if has_feature_indices:
                src_col = Int(tree_feature_indices.unsafe_load(f))
            local_data.unsafe_store(
                s * max_features + f,
                data.unsafe_load(src_row + src_col * n_rows),
            )
            f += n_threads
    barrier()

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
        tree_stack,
        tree_decisions,
        tid,
        n_threads,
    )

    # The stream POSITION this tree finished at. `if.rng.probe` verifies the
    # PORT (that our XORWOW is cuRAND's); this verifies that a tree consumed
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
