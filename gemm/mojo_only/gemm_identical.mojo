# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device GEMM of profile `mojolearn.identical.gemm.fp32.v1`.

Phase 2b of `IDENTICAL_GEMM_PLAN.md`'s lane charter. The contract is
`gemm/IDENTICAL_FP32_CONTRACT.md`; the answer is
`gemm/mojo_only/gemm_oracle.mojo::gemm_oracle`, bit for bit, for `OP_NN`,
`OP_NT` and `OP_TN`.

**NOT A PORT.** RAFT's standalone matrix product is cuBLASLt, a closed
library (`gemm_oracle.mojo`'s header carries the citation). What IS mirrored
is the STRUCTURE: RAFT's `pairwise_distance_base.cuh:139-149` gives one
thread block the entire `k` range of its output tile and walks `kidx`
ascending, with no split-K and no cross-block combination anywhere. Plans
`TILE_*` below are that structure with the contract's leaf partition and fold
named on top of it.

THE SEPARATION THIS FILE EXISTS TO MAKE STRUCTURAL
---------------------------------------------------
The charter splits the NUMERICAL PLAN (logical k leaves, product order in a
leaf, the fold tree, FMA policy, denormal policy, accumulator precision) from
the EXECUTION PLAN (output tiles, block and thread counts, staging,
vectorization, the order blocks compute logical nodes in). The first may not
vary by vendor or by launch; the second may vary freely.

Here that separation is three structural facts, not three promises:

1. **`_contract_partition` is the ONLY producer of `(L, P)` in this file**,
   and it takes `k`. Not `m`, not `n`, not `ctx`, not a plan, not a block
   size. Every kernel receives `L` and `P` as arguments and has no other way
   to learn a leaf boundary: the words `m`, `n`, `block_dim`, `grid_dim`,
   `block_idx` and `thread_idx` do not appear in any expression that reaches
   `leaf_begin` / `leaf_end` or a tree level. Contract section 6.
2. **No float ever crosses a thread boundary in the TILE_* and FLAT plans.**
   One thread owns cell `(i, j)` and computes every one of its `k` products
   and every one of its `P - 1` fold additions in its own registers. Shared
   memory carries OPERANDS (a bit-exact copy of `A` and `B`), never partial
   sums. So there is no cross-thread combination for a block size to
   reorder, and batch invariance is a property of the shape of the kernel
   rather than of a check that happens to pass.
3. **The fold is a REGISTER STACK whose merge rule is a pure function of the
   leaf index.** `_fold_push` merges when the level is occupied, which
   happens exactly when the contract's tree pairs. Proven equal to
   `fold_balanced_tree` for every `P` in `1 .. 2049` by
   `check_stack_fold_is_the_contract_tree` in `gemm_device_check.mojo`, and
   the argument is in `_fold_push`'s docstring. Contract section 7.2.

THE WORKSPACE QUESTION, CONTRACT 13.5
--------------------------------------
A split-K arm materializes `m * n * P` floats: 2 GB at `m = n = 4096,
k = 4096`, 64 GB at `k = 4,000,000`. Section 6.1 forbids fixing that by
letting `L` depend on `m` or `n`, because `m` is the batch dimension.

**The DEFAULT arm here is 13.5's second escape: one block owns an output tile
and ALL of its `k` leaves, folding in registers. There is no global scratch
at any shape, and `identical_gemm_workspace_floats` returns 0 for it.**
That is the right arm wherever `m * n` is large enough to fill the machine on
its own, which is every transformer row in `bench/gemm_shapes.mojo`.

The SPLITK plans are kept for the opposite shape -- small `m n`, enormous
`k`, which is the shipped Gram aspect (`32 x 32 x 1,000,000`) -- where a
fused arm has only `m * n` threads of parallelism and `m * n * P` is small
precisely because `m n` is. Choosing between the arms on `m`, `n` and the
device is legal and expected (13.5); choosing the NUMERICAL TREE on them is
not, and nothing here does.

`PLAN_SPLITK_STAGED` is a third realization and the DISPATCHER NEVER PICKS
IT. It materializes every node of the tree and runs `D` separate fold
launches, which is contract 13.4's "fully staged" implementation -- the one
13.4 argues is never necessary and 13.6.2 requires to be PRICED against the
fused one. It is in the source for that measurement and because it is the
most distant realization of the arithmetic DAG available: no register stack,
no threadgroup memory, every intermediate node round-tripped through DRAM.
`check_device_is_launch_invariant` requiring it to equal `PLAN_FLAT` bit for
bit is the strongest form of the claim this file makes.

WHAT `NUMERIC_FAST` DOES HERE
------------------------------
Everything, with the two pins compiled away. `identical_mul_add` becomes
`a*b + c` and `ftz` becomes the identity, so the leaf loop rounds however the
backend's codegen decides and a denormal survives or does not according to
the hardware. The partition, the tree, the addressing, the tiling and every
launch are UNCHANGED -- so FAST is a correct GEMM, it is the same code on the
same path (not dead code, and not a separate kernel), and it makes no
identity claim. `check_device_is_launch_invariant` and
`check_device_is_batch_invariant` are asserted in BOTH modes, because they
are properties of the kernel's shape rather than of the arithmetic pins;
`check_device_matches_oracle` asserts under IDENTICAL and reports under FAST,
for the reason `core/gemm_identity_check.mojo` gives at the same seam.

DEVIATIONS 530 (the register-stack realization of the contract's fold tree),
531 (the fused one-block-owns-all-k arm as the default, and the workspace
escape it is), 532 (the SPLITK arm and its level-wise threadgroup fold over
the normative `(d, q)` addressing).

`[[mojo-buffer-freed-at-last-use]]`: every launcher here takes buffers the
CALLER owns and the caller keeps alive past `ctx.synchronize()`.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.sys.compile import is_defined

from gemm.mojo_only.gemm_oracle import (
    CONTRACT_MAX_LEAVES,
    OP_NN,
    OP_NT,
    OP_TN,
    contract_leaf_count,
    contract_leaf_size,
    fold_level_base,
    fold_level_count,
    fold_level_width,
    fold_node_total,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add

# DEVIATION 1876 -- THE FAST ARM REACHES THE VENDOR KERNEL. See
# `_fast_vendor_gemm` below for why these three imports are here at all.
from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from linalg.gemv import gemv_gpu


# ===========================================================================
# THE SABOTAGE SWITCHES
# ===========================================================================
# OFF in every build that does not name them. Each one is a specific way to
# break the contract that a plausible implementation could reach by accident,
# and each exists so that the checks in `gemm_device_check.mojo` can be shown
# to FAIL -- "a check that has never failed is a check nobody has tested".
#
# Build with, e.g.:
#     tools/with_identical_mode.sh pixi run mojo run \
#         -D MOJOLEARN_GEMM_SABOTAGE_LEAF_READS_LAUNCH=1 \
#         -I . gemm/mojo_only/gemm_device_check.mojo
#
#: The leaf boundary reads the LAUNCH instead of `k`: `L` is scaled by the
#: block size. Contract section 6's first sentence, violated.
comptime SAB_LEAF_READS_LAUNCH = is_defined[
    "MOJOLEARN_GEMM_SABOTAGE_LEAF_READS_LAUNCH"
]()
#: The fold pairs by STRIDE (`red[t] += red[t + step]`) instead of by
#: adjacent logical leaves. `core/pinned_reduce.mojo::pinned_block_sum`'s
#: shape, and the one a Phase 2b kernel gets for free by letting a thread
#: index stand in for a logical leaf index. Contract 7.2 clause 1, fixture F8.
comptime SAB_FOLD_STRIDE = is_defined["MOJOLEARN_GEMM_SABOTAGE_FOLD_STRIDE"]()
#: The odd tail is PADDED with `+0.0` instead of carried bit for bit.
#: Contract 7.2 clause 3 and 4, fixture F7.
comptime SAB_PAD_PLUS_ZERO = is_defined[
    "MOJOLEARN_GEMM_SABOTAGE_PAD_PLUS_ZERO"
]()
#: The fold is SERIAL ascending over the partials -- the SUPERSEDED rule, and
#: `core/gram_splitk.mojo::gram_splitk_reduce_kernel`'s shipped spelling.
#: Contract 7.2, fixture F5.
comptime SAB_FOLD_SERIAL = is_defined["MOJOLEARN_GEMM_SABOTAGE_FOLD_SERIAL"]()
#: A NODE IS COMPUTED IN THE WRONG ORDER: the SPLITK leaf kernel writes its
#: partial at the address its BLOCK arrived in rather than at its logical
#: leaf index, so the fold pairs the wrong leaves. Contract 7.2.2 -- "the
#: pairing is over logical leaf indices, never over physical block, warp or
#: thread indices" -- and 7.2 clause 2.
comptime SAB_NODE_ORDER = is_defined["MOJOLEARN_GEMM_SABOTAGE_NODE_ORDER"]()
#: THE LEAF START IS ROTATED BY THE BLOCK INDEX: position `t` of the fold
#: visits logical leaf `(t + block_idx.x) mod P` instead of leaf `t`, in every
#: plan. The leaf ARITHMETIC and the tree are both untouched -- only WHICH
#: leaf stands at which tree position depends on the physical block, so the
#: same `(i, j, k)` cell folds in a different order whenever the launch puts
#: it in a different block. That is exactly what a change of batch
#: composition does, and it is the defect `check_device_is_batch_invariant`'s
#: `n: 4 vs 4096` and shared-workspace arms exist to see. Contract 7.2.2 and
#: the charter's "never of batch composition". Added 2026-08-23 for the
#: batch-composition arms; a NO-OP in every build that does not name it.
comptime SAB_LEAF_ROTATE = is_defined["MOJOLEARN_GEMM_SABOTAGE_LEAF_ROTATE"]()

comptime ANY_SABOTAGE = (
    SAB_LEAF_READS_LAUNCH
    or SAB_FOLD_STRIDE
    or SAB_PAD_PLUS_ZERO
    or SAB_FOLD_SERIAL
    or SAB_NODE_ORDER
    or SAB_LEAF_ROTATE
)


def gemm_sabotage_name() -> String:
    """Which sabotage this binary compiled with, for the check's banner."""
    comptime if SAB_LEAF_READS_LAUNCH:
        return String("LEAF_READS_LAUNCH")
    comptime if SAB_FOLD_STRIDE:
        return String("FOLD_STRIDE")
    comptime if SAB_PAD_PLUS_ZERO:
        return String("PAD_PLUS_ZERO")
    comptime if SAB_FOLD_SERIAL:
        return String("FOLD_SERIAL")
    comptime if SAB_NODE_ORDER:
        return String("NODE_ORDER")
    comptime if SAB_LEAF_ROTATE:
        return String("LEAF_ROTATE")
    return String("none")


# ===========================================================================
# THE NUMERICAL PLAN'S ONE DOOR (contract section 6)
# ===========================================================================


#: `ceil(log2(MAX_LEAVES)) + 1` levels, plus headroom, is the depth of the
#: register fold stack. `D = 10` at the profile cap `P = 1024`, so 12 is two
#: spare; `check_fold_stack_depth_covers_the_profile` asserts it against
#: `CONTRACT_MAX_LEAVES` rather than against this comment.
comptime GEMM_FOLD_LEVELS = 12

#: The width of the SIMD register the fold stack lives in. Must be >=
#: `GEMM_FOLD_LEVELS` and a power of two.
comptime GEMM_FOLD_SLOTS = 16


def contract_partition(k: Int) -> Tuple[Int, Int]:
    """**THE ONLY PLACE `(L, P)` IS PRODUCED IN THIS FILE.** Contract 6.

    It takes `k`. There is no overload that takes `m`, `n`, a plan, a block
    size, a `DeviceContext` or a device, and every launcher below calls this
    and passes the two results through as kernel arguments. That is what
    makes "launch geometry cannot reach the arithmetic" a property of the
    call graph instead of a claim in a comment: to move a leaf boundary you
    would have to add an argument to this function, and there is exactly one
    line to look at.

    Both values come from `gemm_oracle`'s own functions -- imported, not
    re-derived -- so the kernel and the oracle cannot hold two opinions
    about the partition. A second spelling of the leaf rule would be a
    second thing that can be wrong, and the shape table already shipped one
    such re-spelling and got it wrong (`bench/gemm_shapes.mojo`'s header).
    """
    return (contract_leaf_size(k), contract_leaf_count(k))


def gemm_operand_strides(op: Int, m: Int, n: Int, k: Int) -> Tuple[Int, Int, Int, Int]:
    """`(a_si, a_sp, b_sp, b_sj)`: contract section 3's four lines, as strides.

        A_eff[i, p] = a[i * a_si + p * a_sp]
        B_eff[p, j] = b[p * b_sp + j * b_sj]

    and the contract's four cases are exactly

        OP_NN, OP_NT :  A is m x k row-major  ->  (a_si, a_sp) = (k, 1)
        OP_TN        :  A is k x m row-major  ->  (a_si, a_sp) = (1, m)
        OP_NN, OP_TN :  B is k x n row-major  ->  (b_sp, b_sj) = (n, 1)
        OP_NT        :  B is n x k row-major  ->  (b_sp, b_sj) = (1, k)

    **Strides rather than a branch inside the k loop, and it is the same
    arithmetic.** Contract section 3 requires the three orientations to be
    ONE numerical implementation differing only in the two index
    expressions; a stride pair IS that index expression with the `if` lifted
    to the host, so the inner loop is character for character identical
    across the three ops and cannot round differently.
    `check_strides_match_the_contract_addressing` asserts the four lines
    element by element against the contract's own spelling.
    """
    var a_si = k
    var a_sp = 1
    if op == OP_TN:
        a_si = 1
        a_sp = m
    var b_sp = n
    var b_sj = 1
    if op == OP_NT:
        b_sp = 1
        b_sj = k
    return (a_si, a_sp, b_sp, b_sj)


# ===========================================================================
# THE FOLD, IN REGISTERS (contract section 7.2)
# ===========================================================================


def _fold_push(
    mut stack: SIMD[DType.float32, GEMM_FOLD_SLOTS],
    mut occ: Int,
    value: Float32,
) -> Bool:
    """Push leaf partial `t` into the balanced tree. Returns False on overflow.

    **THIS IS THE CONTRACT'S TREE, NOT AN APPROXIMATION OF IT**, and the
    argument is short enough to check.

    `occ` is a bitmask: bit `d` set means slot `d` holds a completed node of
    level `d`, which by construction covers a FULL block of `2^d` consecutive
    leaves ending just before the leaf about to be pushed. Pushing merges
    upward while the level is occupied. So a merge at level `d` combines the
    node covering `[q*2^d, (q+1)*2^d)` with the node covering
    `[(q+1)*2^d, (q+2)*2^d)` -- adjacent, ascending, in logical leaf order,
    which is `node(d+1, q/2) = node(d, 2q') + node(d, 2q'+1)`.

    An UNPAIRED leaf never merges: it sits in its slot and is combined only
    at drain time. That is the contract's CARRY -- a bit-for-bit copy with no
    arithmetic -- realized as "no instruction at all", which is the strongest
    form of it. There is no padding anywhere in this function and no `+0.0`
    is ever added.

    THE TREE IS THEREFORE A PURE FUNCTION OF `t` AND `P`, and `P` is a pure
    function of `k` (contract section 6). Nothing here reads a block size, a
    thread index, a lane width or a vendor, and there is nothing in scope
    that could.

    Proven, not argued: `check_stack_fold_is_the_contract_tree` runs this
    spelling against `gemm_oracle.fold_balanced_tree` for every `P` in
    `1 .. 2049` on hashed partials and requires bit equality at every one.

    Contract sections 5 (rows 5e, 5f), 7.2 and 7.3.
    """
    var val = value
    var placed = False
    comptime for d in range(GEMM_FOLD_LEVELS):
        if not placed:
            if ((occ >> d) & 1) == 1:
                comptime if SAB_FOLD_STRIDE:
                    # SABOTAGE: pair the far end of the level instead of the
                    # adjacent node -- `pinned_block_sum`'s stride shape.
                    val = ftz(ftz(val) + ftz(stack[GEMM_FOLD_LEVELS - 1 - d]))
                else:
                    # THE ARITHMETIC NODE. Contract 5e (both children
                    # flushed as read) and 5f (the node's own result
                    # flushed). The occupied slot holds the EARLIER leaves,
                    # so it is the left operand.
                    val = ftz(ftz(stack[d]) + ftz(val))
                occ = occ - (1 << d)
            else:
                stack[d] = val
                occ = occ + (1 << d)
                placed = True
    return placed


def _fold_drain(
    stack: SIMD[DType.float32, GEMM_FOLD_SLOTS], occ: Int
) -> Float32:
    """Combine the leftover slots into the root, LOWEST LEVEL FIRST.

    At the end of the push sequence the occupied slots are the set bits of
    `P`, and slot `d` holds the pairwise sum of a full `2^d` block. The
    contract's root is `node(D, 0)`, which unrolls as
    `full_block(d_max) + (the same construction on the remainder)` -- so the
    remainder is accumulated first and each larger block joins on its left.
    That is this loop, ascending `d`.

    **`P == 1` performs NO addition** (contract 7.3): one slot is occupied,
    `have` is False on the only iteration that fires, and the value is
    returned unchanged. The `-0.0` of section 9.2(b) survives, and this is
    the case where the rule and the "optimization" coincide, so there is
    nothing to get wrong.

    `P == 0` (`k == 0`) returns `+0.0` and the caller STORES it -- contract
    section 8 requires the value to be written, not the store to be skipped.
    """
    var have = False
    var acc = Float32(0.0)
    comptime for d in range(GEMM_FOLD_LEVELS):
        if ((occ >> d) & 1) == 1:
            if have:
                acc = ftz(ftz(stack[d]) + ftz(acc))
            else:
                acc = stack[d]
                have = True
    return acc


# ===========================================================================
# PLAN NAMES (EXECUTION PLAN ONLY -- none of these can move a bit)
# ===========================================================================
# Every plan below computes `gemm_oracle`. They differ in which thread owns
# which cell, how many threads a block has, what the grid looks like, whether
# operands are staged through threadgroup memory, what order blocks take
# tiles in, and whether the fold runs in registers, in one threadgroup
# launch, or in `D` separate global launches.
# `check_device_is_launch_invariant` requires all eight to produce the same
# bits, and that check is the reason the profile exists.

comptime PLAN_FLAT = 0
comptime PLAN_TILE_16_16_32 = 1
comptime PLAN_TILE_8_32_32 = 2
comptime PLAN_TILE_32_8_16 = 3
comptime PLAN_TILE_16_16_8_REV = 4
comptime PLAN_TILE_4_4_32_TSP = 5
comptime PLAN_SPLITK = 6
comptime PLAN_SPLITK_STAGED = 7
comptime GEMM_PLAN_COUNT = 8

#: Threads per block for `PLAN_FLAT`. SCHEDULING: each thread owns a whole
#: output cell, so this moves WHICH thread computes a cell and never the
#: order any cell is accumulated in.
comptime FLAT_TPB = 256
#: Threads per block for the SPLITK leaf-partial kernel. Same argument.
comptime SPLITK_LEAF_TPB = 128
#: Threads per block for the SPLITK fold kernel: one BLOCK per output cell,
#: cooperating over the nodes of one tree level. The level WIDTHS come from
#: `P` alone (contract 7.2.2); this number only decides how the nodes of a
#: level are handed out.
comptime SPLITK_FOLD_TPB = 256

#: Tile order permutations. Each is an exact bijection on `[0, n_tiles)`, so
#: every tile is computed exactly once whichever is chosen; which BLOCK
#: computes which tile is the "order blocks compute logical nodes in" that
#: the charter puts on the execution-plan side of the table.
comptime SWIZZLE_NONE = 0
comptime SWIZZLE_REVERSE = 1
comptime SWIZZLE_TRANSPOSE = 2


def gemm_plan_name(plan: Int) -> String:
    if plan == PLAN_FLAT:
        return String("FLAT(1 thread/cell, tpb=256, 1-D grid)")
    if plan == PLAN_TILE_16_16_32:
        return String("TILE 16x16 KS=32 (256 thr, 1-D grid, no swizzle)")
    if plan == PLAN_TILE_8_32_32:
        return String("TILE 8x32 KS=32 (256 thr, 2-D grid, no swizzle)")
    if plan == PLAN_TILE_32_8_16:
        return String("TILE 32x8 KS=16 (256 thr, 1-D grid, reversed tiles)")
    if plan == PLAN_TILE_16_16_8_REV:
        return String("TILE 16x16 KS=8 (256 thr, 2-D grid, transposed tiles)")
    if plan == PLAN_TILE_4_4_32_TSP:
        return String("TILE 4x4 KS=32 (16 thr, 1-D grid, transposed tiles)")
    if plan == PLAN_SPLITK:
        return String("SPLITK(leaf kernel -> global workspace -> level-wise threadgroup fold)")
    if plan == PLAN_SPLITK_STAGED:
        return String("SPLITK_STAGED(leaf kernel -> D separate global fold-level launches -> emit)")
    return String("PLAN?")


# ===========================================================================
# THE LEAF LOOP (contract sections 4, 5, 7.1)
# ===========================================================================


def _leaf_bounds(t: Int, leaf: Int, k: Int) -> Tuple[Int, Int]:
    """Leaf `t` covers `[t*L, min((t+1)*L, k))`. Contract sections 6 and 8.

    Only the LAST leaf is ever short and by section 6.2 none is ever empty.
    NO PADDING: the ragged leaf is masked by `pe`, never extended with zeros
    (section 8 forbids a padded operand because `0.0 * 0.0` is not inert for
    the signed-zero and NaN cases of section 9).

    `leaf` and `k` are the only inputs. That is the clause.
    """
    var pb = t * leaf
    var pe = pb + leaf
    if pe > k:
        pe = k
    return (pb, pe)


def _leaf_at(t: Int, p_count: Int) -> Int:
    """The logical leaf that stands at fold position `t`. **The identity**,
    and it exists only so that `SAB_LEAF_ROTATE` has one place to break it
    in every plan: under the sabotage it is `(t + block_idx.x) mod P`, and
    the block a cell was scheduled in decides its summation order."""
    comptime if SAB_LEAF_ROTATE:
        if p_count > 0:
            return (t + Int(block_idx.x)) % p_count
    return t


# ===========================================================================
# PLAN FLAT: one thread per output cell, its whole k range, no staging
# ===========================================================================


def identical_gemm_flat_kernel(
    c: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    leaf_in: Int32,
    p_in: Int32,
    a_si_in: Int32,
    a_sp_in: Int32,
    b_sp_in: Int32,
    b_sj_in: Int32,
):
    """`C[i, j]` for one thread: every leaf, then the tree, all in registers.

    The simplest realization of the profile and the reference the tiled plans
    are checked against. No shared memory, no cross-thread anything: the
    launch decides only which linear index this thread holds.

    `leaf_in` and `p_in` come from `contract_partition(k)` and from nowhere
    else. This kernel cannot compute a leaf boundary from anything but them.
    """
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var leaf = Int(leaf_in)
    var p_count = Int(p_in)
    var a_si = Int(a_si_in)
    var a_sp = Int(a_sp_in)
    var b_sp = Int(b_sp_in)
    var b_sj = Int(b_sj_in)

    comptime if SAB_LEAF_READS_LAUNCH:
        # SABOTAGE: the leaf boundary reads the LAUNCH. Contract section 6's
        # first sentence forbids exactly this.
        leaf = leaf * Int(block_dim.x) // 64
        if leaf < 1:
            leaf = 1
        p_count = (k + leaf - 1) // leaf
        if k <= 0:
            p_count = 0

    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= m * n:
        return
    var i = cell // n
    var j = cell - i * n

    var stack = SIMD[DType.float32, GEMM_FOLD_SLOTS](0.0)
    var occ = 0
    var serial = Float32(0.0)
    var a_row = i * a_si
    var b_col = j * b_sj
    for t in range(p_count):
        var bounds = _leaf_bounds(_leaf_at(t, p_count), leaf, k)
        var acc = Float32(0.0)
        for p in range(bounds[0], bounds[1]):
            # Contract 7.1: ascending `p`, one `identical_mul_add` per step,
            # operands flushed as loaded (5a, 5b) and the accumulator flushed
            # after every step (5c). No sub-partition of a leaf -- section
            # 7.1's clause about register tiling and vectorization.
            acc = ftz(
                identical_mul_add(
                    ftz(a.unsafe_load(a_row + p * a_sp)),
                    ftz(b.unsafe_load(p * b_sp + b_col)),
                    acc,
                )
            )
        # 5d: the leaf partial as written.
        var part = ftz(acc)
        comptime if SAB_FOLD_SERIAL:
            # SABOTAGE: the SUPERSEDED serial ascending fold.
            serial = ftz(ftz(serial) + ftz(part))
        else:
            _ = _fold_push(stack, occ, part)
    comptime if SAB_PAD_PLUS_ZERO:
        # SABOTAGE: pad the level-0 width up to the next power of two with
        # `+0.0` instead of carrying the odd tail. Contract 7.2 clause 4.
        var padded = 1
        while padded < p_count:
            padded = padded * 2
        for _pad in range(p_count, padded):
            _ = _fold_push(stack, occ, Float32(0.0))
    var out = _fold_drain(stack, occ)
    comptime if SAB_FOLD_SERIAL:
        out = serial
    # 5g: the output cell as stored. At `P == 1` this is the ONLY operation
    # between the leaf partial and memory (contract 7.3); at `P == 0` it
    # stores the `+0.0` section 8 requires to be written rather than skipped.
    c.unsafe_store(cell, ftz(out))


# ===========================================================================
# PLAN TILE_*: one block owns a TM x TN output tile and ALL of its k leaves
# ===========================================================================


def identical_gemm_tiled_kernel[
    TM: Int, TN: Int, KS: Int
](
    c: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    leaf_in: Int32,
    p_in: Int32,
    a_si_in: Int32,
    a_sp_in: Int32,
    b_sp_in: Int32,
    b_sj_in: Int32,
    swizzle_in: Int32,
):
    """RAFT's structure: one block owns a `TM x TN` output tile and the whole
    `k` range of it. Contract 13.5's second workspace escape -- NO GLOBAL
    SCRATCH AT ANY SHAPE.

    Operands are staged through threadgroup memory in `KS`-wide k windows.
    **Threadgroup memory carries OPERANDS, never partial sums**, and a
    staging copy is bit-exact, so the value this thread flushes and
    multiplies is the same float `A_eff[i, p]` the FLAT plan loads straight
    from DRAM. `TM`, `TN` and `KS` are therefore pure execution plan: they
    change which thread owns a cell and how many DRAM transactions serve it,
    and they cannot change the sequence of values accumulated into it.

    THE STAGING WINDOW IS NOT A LEAF, and the loop nest is written to make
    that impossible to confuse: the OUTER loop is over logical leaves `t`
    (from `leaf` and `k` alone) and the staging window is the INNER loop,
    clamped to the leaf's end. So a `KS` of 8 and a `KS` of 32 walk the same
    leaf in a different number of windows and accumulate into the same
    register in the same order. `check_device_is_launch_invariant` runs both.

    Every thread of the block reaches every `barrier()`: the loop bounds are
    functions of `t`, `leaf`, `k` and `KS`, which are block-uniform, and a
    thread whose cell is out of range still stages and still barriers, and
    only its final STORE is masked. `[[metal-hardware-gaps]]`: no block
    reduction primitive is used anywhere, so nothing here carries the
    `Block size must be greater than warp size` constraint.
    """
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var leaf = Int(leaf_in)
    var p_count = Int(p_in)
    var a_si = Int(a_si_in)
    var a_sp = Int(a_sp_in)
    var b_sp = Int(b_sp_in)
    var b_sj = Int(b_sj_in)
    var swizzle = Int(swizzle_in)

    comptime if SAB_LEAF_READS_LAUNCH:
        leaf = leaf * Int(block_dim.x) // 64
        if leaf < 1:
            leaf = 1
        p_count = (k + leaf - 1) // leaf
        if k <= 0:
            p_count = 0

    comptime NTH = TM * TN
    var as_ = stack_allocation[
        TM * KS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var bs_ = stack_allocation[
        KS * TN, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()

    var tiles_i = (m + TM - 1) // TM
    var tiles_j = (n + TN - 1) // TN
    var n_tiles = tiles_i * tiles_j
    var raw = Int(block_idx.y) * Int(grid_dim.x) + Int(block_idx.x)
    if raw >= n_tiles:
        return
    # The tile permutation: an EXACT bijection, so every tile runs once.
    var tile = raw
    if swizzle == SWIZZLE_REVERSE:
        tile = n_tiles - 1 - raw
    elif swizzle == SWIZZLE_TRANSPOSE:
        tile = (raw % tiles_i) * tiles_j + (raw // tiles_i)
    var ti = tile // tiles_j
    var tj = tile - ti * tiles_j
    var i0 = ti * TM
    var j0 = tj * TN

    var tid = Int(thread_idx.x)
    var r = tid // TN
    var s = tid - r * TN
    var gi = i0 + r
    var gj = j0 + s

    var stack = SIMD[DType.float32, GEMM_FOLD_SLOTS](0.0)
    var occ = 0
    var serial = Float32(0.0)

    for t in range(p_count):
        var bounds = _leaf_bounds(_leaf_at(t, p_count), leaf, k)
        var pe = bounds[1]
        var acc = Float32(0.0)
        var p = bounds[0]
        while p < pe:
            var chunk = pe - p
            if chunk > KS:
                chunk = KS
            # Stage A_eff[i0 .. i0+TM, p .. p+chunk). `KS` is comptime, so
            # `idx // KS` is a shift; slots past `chunk` are written zero and
            # never read, which is staging hygiene and NOT operand padding --
            # the accumulation loop below runs `chunk` steps, not `KS`.
            var ia = tid
            while ia < TM * KS:
                var rr = ia // KS
                var cc = ia - rr * KS
                var v = Float32(0.0)
                if i0 + rr < m and cc < chunk:
                    v = a.unsafe_load((i0 + rr) * a_si + (p + cc) * a_sp)
                as_[unsafe_offset=ia] = v
                ia += NTH
            var ib = tid
            while ib < KS * TN:
                var cc2 = ib // TN
                var ss = ib - cc2 * TN
                var v2 = Float32(0.0)
                if j0 + ss < n and cc2 < chunk:
                    v2 = b.unsafe_load((p + cc2) * b_sp + (j0 + ss) * b_sj)
                bs_[unsafe_offset=ib] = v2
                ib += NTH
            barrier()
            for cc3 in range(chunk):
                # Contract 7.1 again, character for character the FLAT
                # plan's step with the two loads served from threadgroup
                # memory. Seams 5a, 5b, 5c.
                acc = ftz(
                    identical_mul_add(
                        ftz(as_[unsafe_offset = r * KS + cc3]),
                        ftz(bs_[unsafe_offset = cc3 * TN + s]),
                        acc,
                    )
                )
            barrier()
            p += chunk
        var part = ftz(acc)
        comptime if SAB_FOLD_SERIAL:
            serial = ftz(ftz(serial) + ftz(part))
        else:
            _ = _fold_push(stack, occ, part)

    comptime if SAB_PAD_PLUS_ZERO:
        var padded = 1
        while padded < p_count:
            padded = padded * 2
        for _pad in range(p_count, padded):
            _ = _fold_push(stack, occ, Float32(0.0))

    var out = _fold_drain(stack, occ)
    comptime if SAB_FOLD_SERIAL:
        out = serial
    if gi < m and gj < n:
        c.unsafe_store(gi * n + gj, ftz(out))


# ===========================================================================
# PLAN SPLITK: named partials into a predetermined workspace, then a fold
# ===========================================================================
# `core/gram_splitk.mojo`'s architecture, with v1's numerical policy instead
# of that file's (which pins a chunk COUNT and folds SERIALLY, and stays that
# way as its own profile -- IDENTICAL_GEMM_PLAN.md's LANE BOUNDARY item 2).


def identical_gemm_leaf_kernel(
    ws: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    leaf_in: Int32,
    p_in: Int32,
    a_si_in: Int32,
    a_sp_in: Int32,
    b_sp_in: Int32,
    b_sj_in: Int32,
    stride_in: Int32,
):
    """Level 0 of the tree: every `(cell, leaf)` partial, at its PREDETERMINED
    address `ws[cell * stride + t]`. Contract 7.2.2.

    `stride` is the per-cell block of the workspace: `P` when only level 0 is
    materialized (`PLAN_SPLITK`, whose fold keeps the upper levels in
    threadgroup memory) and `fold_node_total(P)` when EVERY level is
    materialized (`PLAN_SPLITK_STAGED`). It is a LAYOUT number and the only
    thing it changes is where a node lives, never which node is added to
    which -- contract 7.2.2's "the (d, q) pair is the normative address; the
    flat integer is one legal layout of it".

    `t` is the LOGICAL leaf index, derived from the flat work index and never
    from `block_idx`. That distinction is the whole of contract 7.2 clause 2,
    and `SAB_NODE_ORDER` is the version that gets it wrong.

    Blocks may run in any order and any number of them may be resident: each
    partial has one writer and one address, so the workspace after the launch
    is the same array of floats whatever the schedule was.
    """
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var leaf = Int(leaf_in)
    var p_count = Int(p_in)
    var a_si = Int(a_si_in)
    var a_sp = Int(a_sp_in)
    var b_sp = Int(b_sp_in)
    var b_sj = Int(b_sj_in)
    var stride = Int(stride_in)

    comptime if SAB_LEAF_READS_LAUNCH:
        leaf = leaf * Int(block_dim.x) // 64
        if leaf < 1:
            leaf = 1
        p_count = (k + leaf - 1) // leaf
        if k <= 0:
            p_count = 0

    var total = m * n * p_count
    var gid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if gid >= total:
        return
    var cell = gid // p_count
    var t = gid - cell * p_count
    var i = cell // n
    var j = cell - i * n

    var bounds = _leaf_bounds(_leaf_at(t, p_count), leaf, k)
    var acc = Float32(0.0)
    var a_row = i * a_si
    var b_col = j * b_sj
    for p in range(bounds[0], bounds[1]):
        acc = ftz(
            identical_mul_add(
                ftz(a.unsafe_load(a_row + p * a_sp)),
                ftz(b.unsafe_load(p * b_sp + b_col)),
                acc,
            )
        )
    var slot = t
    comptime if SAB_NODE_ORDER:
        # SABOTAGE: address the partial by the PHYSICAL BLOCK that produced
        # it rather than by its logical leaf index, so the fold below pairs
        # whichever leaves happened to share a block. Contract 7.2.2.
        slot = (t + Int(block_idx.x)) % p_count
    ws.unsafe_store(cell * stride + slot, ftz(acc))


def identical_gemm_fold_kernel(
    c: MutPointer[Float32, MutAnyOrigin],
    ws: MutPointer[Float32, MutAnyOrigin],
    mn_in: Int32,
    p_in: Int32,
    stride_in: Int32,
):
    """One BLOCK per output cell folds that cell's tree LEVEL BY LEVEL in
    threadgroup memory. Contract 7.2.2's normative `(d, q)` addressing, run
    as the addressing rather than as a comment about it.

    Contract 13.4's design: `MAX_LEAVES = 1024` caps `P`, so 1024 partials is
    4 KB and ONE threadgroup folds all `D <= 10` levels with a barrier
    between them, in ONE launch, executing exactly the pairings of section
    7.2.

    THE LEVEL WIDTHS COME FROM `P` ALONE. `w` starts at `P` and halves with a
    CEILING; the block size decides only how the `pairs` nodes of a level are
    handed out to threads (`q` strides by `block_dim.x`), which is the
    charter's "scheduling of logical work". A thread index is never a leaf
    index here: `q` addresses the node, and node `q` reads children `2q` and
    `2q + 1` whichever thread computes it.

    Ping-pong buffers rather than in-place: at level `d`, node `q` writes
    slot `q` while node `q/2`'s children live at `2q` and `2q + 1`, and
    `q <= 2q`, so an in-place level would race a reader against a writer.
    That is a correctness bug a small `P` would hide, because at `P <= 2`
    there is one node per level.

    `[[metal-hardware-gaps]]`: hand-written, no `max.gpu.primitives.block.*`
    reduction, so no `Block size must be greater than warp size` constraint
    and this compiles at any block size on a 64-wide wavefront.
    """
    var mn = Int(mn_in)
    var p_count = Int(p_in)
    var stride = Int(stride_in)
    var cell = Int(block_idx.x)
    if cell >= mn:
        return
    var tid = Int(thread_idx.x)
    var nth = Int(block_dim.x)

    var buf = stack_allocation[
        2 * CONTRACT_MAX_LEAVES,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var cur = 0
    var nxt = CONTRACT_MAX_LEAVES

    var q0 = tid
    while q0 < p_count:
        buf[unsafe_offset = cur + q0] = ws.unsafe_load(cell * stride + q0)
        q0 += nth
    barrier()

    var w = p_count
    comptime if SAB_FOLD_SERIAL:
        # SABOTAGE: the SUPERSEDED serial ascending fold, one thread.
        if tid == 0:
            var acc = Float32(0.0)
            for t in range(p_count):
                acc = ftz(ftz(acc) + ftz(buf[unsafe_offset = cur + t]))
            c.unsafe_store(cell, ftz(acc))
        return
    while w > 1:
        var pairs = w // 2
        var q = tid
        while q < pairs:
            comptime if SAB_FOLD_STRIDE:
                # SABOTAGE: `red[q] += red[q + step]`, `pinned_block_sum`'s
                # stride pairing. Same depth, different pairs, different
                # answer. Contract 7.2 clause 1, fixture F8.
                buf[unsafe_offset = nxt + q] = ftz(
                    ftz(buf[unsafe_offset = cur + q])
                    + ftz(buf[unsafe_offset = cur + q + pairs])
                )
            else:
                # THE ARITHMETIC NODE, contract 7.2.2: adjacent children,
                # ascending logical order, both flushed as read (5e), the
                # result flushed (5f).
                buf[unsafe_offset = nxt + q] = ftz(
                    ftz(buf[unsafe_offset = cur + 2 * q])
                    + ftz(buf[unsafe_offset = cur + 2 * q + 1])
                )
            q += nth
        var w_next = pairs
        if w % 2 == 1:
            comptime if SAB_PAD_PLUS_ZERO:
                # SABOTAGE: pad with `+0.0` instead of carrying. `x + (+0.0)`
                # is not the identity at `x = -0.0`. Contract 7.2 clause 4,
                # fixture F7.
                if tid == 0:
                    buf[unsafe_offset = nxt + pairs] = ftz(
                        ftz(buf[unsafe_offset = cur + w - 1]) + Float32(0.0)
                    )
            else:
                # THE CARRY NODE, contract 7.2.2: a BIT-FOR-BIT copy. No
                # arithmetic, no flush of its own -- it inherits the flush
                # its source already carries (contract section 5's note that
                # a carry needs no seam).
                if tid == 0:
                    buf[unsafe_offset = nxt + pairs] = buf[
                        unsafe_offset = cur + w - 1
                    ]
            w_next = pairs + 1
        barrier()
        var swap = cur
        cur = nxt
        nxt = swap
        w = w_next
    if tid == 0:
        # 5g, and contract section 8's `k == 0` case: `p_count == 0` skips
        # the loop entirely and `+0.0` is STORED rather than the store being
        # skipped.
        var root = Float32(0.0)
        if p_count > 0:
            root = buf[unsafe_offset=cur]
        c.unsafe_store(cell, ftz(root))


def identical_gemm_fold_level_kernel(
    ws: MutPointer[Float32, MutAnyOrigin],
    mn_in: Int32,
    stride_in: Int32,
    src_base_in: Int32,
    dst_base_in: Int32,
    w_prev_in: Int32,
    w_next_in: Int32,
):
    """ONE LOGICAL LEVEL of the tree, as ONE LAUNCH. Contract 7.2.2's flat
    address space, used literally.

    `src_base` is `fold_level_base(P, d-1)` and `dst_base` is
    `fold_level_base(P, d)`, both computed ON THE HOST by `gemm_oracle`'s own
    functions, so the device never re-derives the topology and cannot hold a
    second opinion about it.

        q <  pairs   ARITHMETIC:  ftz( ftz(node(d-1, 2q)) + ftz(node(d-1, 2q+1)) )
        q == pairs   CARRY:       node(d-1, w_prev - 1), BIT FOR BIT

    This is the FULLY STAGED realization contract 13.4 says is not required
    and 13.6.2 says must be priced against the fused one. It is here for two
    reasons and the first is not performance: it is an eighth EXECUTION PLAN
    whose fold runs in `D` separate global-memory launches with no
    threadgroup memory and no register stack anywhere, so
    `check_device_is_launch_invariant` comparing it against `PLAN_FLAT` is
    about as far apart as two realizations of one arithmetic DAG can get.
    """
    var mn = Int(mn_in)
    var stride = Int(stride_in)
    var src_base = Int(src_base_in)
    var dst_base = Int(dst_base_in)
    var w_prev = Int(w_prev_in)
    var w_next = Int(w_next_in)
    var gid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if gid >= mn * w_next:
        return
    var cell = gid // w_next
    var q = gid - cell * w_next
    var base = cell * stride
    var pairs = w_prev // 2
    if q < pairs:
        comptime if SAB_FOLD_STRIDE:
            ws.unsafe_store(
                base + dst_base + q,
                ftz(
                    ftz(ws.unsafe_load(base + src_base + q))
                    + ftz(ws.unsafe_load(base + src_base + q + pairs))
                ),
            )
        else:
            ws.unsafe_store(
                base + dst_base + q,
                ftz(
                    ftz(ws.unsafe_load(base + src_base + 2 * q))
                    + ftz(ws.unsafe_load(base + src_base + 2 * q + 1))
                ),
            )
        return
    # THE CARRY. `q == pairs` can only be reached when `w_prev` is odd.
    comptime if SAB_PAD_PLUS_ZERO:
        ws.unsafe_store(
            base + dst_base + q,
            ftz(ftz(ws.unsafe_load(base + src_base + w_prev - 1)) + Float32(0.0)),
        )
    else:
        ws.unsafe_store(
            base + dst_base + q, ws.unsafe_load(base + src_base + w_prev - 1)
        )


def identical_gemm_emit_kernel(
    c: MutPointer[Float32, MutAnyOrigin],
    ws: MutPointer[Float32, MutAnyOrigin],
    mn_in: Int32,
    stride_in: Int32,
    root_base_in: Int32,
    p_in: Int32,
):
    """The output seam (contract 5g) for the staged plan: `C[cell] =
    ftz(node(D, 0))`.

    `p_count == 0` (`k == 0`) stores `+0.0` -- contract section 8 requires the
    value to be WRITTEN, not the store skipped.
    """
    var mn = Int(mn_in)
    var stride = Int(stride_in)
    var root_base = Int(root_base_in)
    var p_count = Int(p_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= mn:
        return
    var v = Float32(0.0)
    if p_count > 0:
        comptime if SAB_FOLD_SERIAL:
            # SABOTAGE: ignore the tree in the workspace and fold level 0
            # SERIALLY -- the superseded rule, and gram_splitk's spelling.
            var acc = Float32(0.0)
            for t in range(p_count):
                acc = ftz(ftz(acc) + ftz(ws.unsafe_load(cell * stride + t)))
            v = acc
        else:
            v = ws.unsafe_load(cell * stride + root_base)
    c.unsafe_store(cell, ftz(v))


# ===========================================================================
# THE HOST ENTRY POINTS
# ===========================================================================


def identical_gemm_workspace_floats(m: Int, n: Int, k: Int, plan: Int) -> Int:
    """Floats of global scratch `plan` needs at this shape. Phase 3 and Phase
    4 size the buffer with this and nothing else.

    **Zero for FLAT and every TILE plan**, which is the whole content of
    contract 13.5's second escape: the fused arm folds in registers and
    materializes no partials at any `m`, `n` or `k`.

    `PLAN_SPLITK` needs `m * n * P` (13.5's table: 128 MB at
    `1024 x 1024 x 4096`, 2 GB at `4096 x 4096 x 4096`, 64 GB at
    `4096 x 4096 x 4,000,000`) because it materializes LEVEL 0 only and folds
    the rest in threadgroup memory.

    `PLAN_SPLITK_STAGED` needs `m * n * fold_node_total(P)`, EVERY node of
    the tree. **`fold_node_total` is not bounded by `2P`** -- contract 7.2.2:
    `P = 5` gives 11 against `2P = 10`, because each level's ceiling rounds
    up -- so this calls the function rather than a bound. A scratch allocator
    that budgeted `2P` would be one node short at the first odd `P` it met.

    `identical_gemm_splitk_fits` is the predicate the dispatcher uses; the
    answer to a shape that does not fit is a different EXECUTION plan, never
    a different leaf rule (section 6.1).
    """
    if m <= 0 or n <= 0 or k <= 0:
        return 0
    var part = contract_partition(k)
    if plan == PLAN_SPLITK:
        return m * n * part[1]
    if plan == PLAN_SPLITK_STAGED:
        return m * n * fold_node_total(part[1])
    return 0


#: The largest SPLITK workspace this file will allocate on its own: 64 M
#: floats, 256 MB. A shape past it is served by a TILE plan, which is a
#: change of EXECUTION plan and moves no bit. Contract 13.5.
comptime SPLITK_MAX_WORKSPACE_FLOATS = 64 * 1024 * 1024


def identical_gemm_splitk_fits(m: Int, n: Int, k: Int) -> Bool:
    return (
        identical_gemm_workspace_floats(m, n, k, PLAN_SPLITK)
        <= SPLITK_MAX_WORKSPACE_FLOATS
    )


def choose_gemm_plan(m: Int, n: Int, k: Int) -> Int:
    """Pick an EXECUTION plan. **Reads `m`, `n`, `k` and is allowed to**
    (contract 6.1: "the EXECUTION plan may look at `m`, `n`, the device, the
    occupancy and anything else it likes, because under section 7 none of
    that can reach the arithmetic").

    What it may NOT do -- and structurally cannot, because it returns a plan
    id and nothing else, and every plan calls `contract_partition(k)` -- is
    influence a leaf boundary or a tree level. There is no
    `if P < 32 use the serial fold` here and there never may be (13.5's last
    sentence).

    The rule: SPLITK when the output is small enough that a fused arm would
    leave the machine idle AND the workspace fits; a wide tile when the
    output is large enough to tile; FLAT otherwise. All three are the same
    bits and `check_device_is_launch_invariant` is what keeps them so.
    """
    if m <= 0 or n <= 0:
        return PLAN_FLAT
    var part = contract_partition(k)
    var p_count = part[1]
    if (
        m * n <= 4096
        and p_count >= 4
        and identical_gemm_splitk_fits(m, n, k)
    ):
        return PLAN_SPLITK
    if m >= 16 and n >= 16:
        return PLAN_TILE_16_16_32
    if n >= 32:
        return PLAN_TILE_8_32_32
    if m >= 32:
        return PLAN_TILE_32_8_16
    return PLAN_FLAT


def _tile_grid(m: Int, n: Int, tm: Int, tn: Int, two_d: Bool) -> Tuple[Int, Int]:
    var tiles_i = (m + tm - 1) // tm
    var tiles_j = (n + tn - 1) // tn
    var total = tiles_i * tiles_j
    if total < 1:
        total = 1
    if not two_d:
        return (total, 1)
    # A 2-D grid over the same tiles: `gy = ceil(total / 64)`, and the
    # kernel's `block_idx.y * grid_dim.x + block_idx.x` linearizes it back.
    # Blocks past `total` return immediately.
    var gx = 64
    if gx > total:
        gx = total
    var gy = (total + gx - 1) // gx
    return (gx, gy)


def _launch_tiled[
    TM: Int, TN: Int, KS: Int
](
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    leaf: Int,
    p_count: Int,
    st: Tuple[Int, Int, Int, Int],
    swizzle: Int,
    two_d: Bool,
) raises:
    comptime kern = identical_gemm_tiled_kernel[TM, TN, KS]
    var g = _tile_grid(m, n, TM, TN, two_d)
    ctx.enqueue_function[kern](
        c.unsafe_ptr(),
        a.unsafe_ptr(),
        b.unsafe_ptr(),
        Int32(m),
        Int32(n),
        Int32(k),
        Int32(leaf),
        Int32(p_count),
        Int32(st[0]),
        Int32(st[1]),
        Int32(st[2]),
        Int32(st[3]),
        Int32(swizzle),
        grid_dim=(g[0], g[1], 1),
        block_dim=(TM * TN, 1, 1),
    )


def identical_gemm_with_plan(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
    plan: Int,
) raises:
    """`C = op(A) . op(B)` under `mojolearn.identical.gemm.fp32.v1`, on a
    NAMED execution plan. The gates call this; production calls
    `identical_gemm` below.

    `ws` must hold at least `identical_gemm_workspace_floats(m, n, k, plan)`
    floats and is UNTOUCHED by every plan but SPLITK. The caller owns every
    buffer and synchronizes; nothing here waits.

    THE THREE LINES THAT ARE THE NUMERICAL PLAN are the next three: the
    partition from `k`, the strides from the orientation, and then a switch
    that chooses geometry. After `part` is computed there is no path back
    into the arithmetic -- every branch below passes the SAME `leaf` and
    `p_count` to its kernel.
    """
    if m <= 0 or n <= 0:
        return
    var part = contract_partition(k)
    var leaf = part[0]
    var p_count = part[1]
    var st = gemm_operand_strides(op, m, n, k)

    if (plan == PLAN_SPLITK or plan == PLAN_SPLITK_STAGED) and p_count > 0:
        var staged = plan == PLAN_SPLITK_STAGED
        var stride = p_count
        if staged:
            stride = fold_node_total(p_count)
        ctx.enqueue_function[identical_gemm_leaf_kernel](
            ws.unsafe_ptr(),
            a.unsafe_ptr(),
            b.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(k),
            Int32(leaf),
            Int32(p_count),
            Int32(st[0]),
            Int32(st[1]),
            Int32(st[2]),
            Int32(st[3]),
            Int32(stride),
            grid_dim=(
                (m * n * p_count + SPLITK_LEAF_TPB - 1) // SPLITK_LEAF_TPB,
                1,
                1,
            ),
            block_dim=(SPLITK_LEAF_TPB, 1, 1),
        )
        if not staged:
            ctx.enqueue_function[identical_gemm_fold_kernel](
                c.unsafe_ptr(),
                ws.unsafe_ptr(),
                Int32(m * n),
                Int32(p_count),
                Int32(stride),
                grid_dim=(m * n, 1, 1),
                block_dim=(SPLITK_FOLD_TPB, 1, 1),
            )
            return
        # THE FULLY STAGED FOLD: one launch per LOGICAL LEVEL. Every base and
        # width below comes from `gemm_oracle`'s own addressing functions, on
        # the host, from `p_count` alone -- contract 7.2.2. The launch that
        # computes a level cannot influence which nodes the level HAS.
        var levels = fold_level_count(p_count)
        for d in range(1, levels):
            var w_prev = fold_level_width(p_count, d - 1)
            var w_next = fold_level_width(p_count, d)
            ctx.enqueue_function[identical_gemm_fold_level_kernel](
                ws.unsafe_ptr(),
                Int32(m * n),
                Int32(stride),
                Int32(fold_level_base(p_count, d - 1)),
                Int32(fold_level_base(p_count, d)),
                Int32(w_prev),
                Int32(w_next),
                grid_dim=(
                    (m * n * w_next + SPLITK_LEAF_TPB - 1) // SPLITK_LEAF_TPB,
                    1,
                    1,
                ),
                block_dim=(SPLITK_LEAF_TPB, 1, 1),
            )
        ctx.enqueue_function[identical_gemm_emit_kernel](
            c.unsafe_ptr(),
            ws.unsafe_ptr(),
            Int32(m * n),
            Int32(stride),
            Int32(fold_level_base(p_count, levels - 1)),
            Int32(p_count),
            grid_dim=((m * n + FLAT_TPB - 1) // FLAT_TPB, 1, 1),
            block_dim=(FLAT_TPB, 1, 1),
        )
        return
    if plan == PLAN_TILE_16_16_32:
        _launch_tiled[16, 16, 32](
            ctx, c, a, b, m, n, k, leaf, p_count, st, SWIZZLE_NONE, False
        )
        return
    if plan == PLAN_TILE_8_32_32:
        _launch_tiled[8, 32, 32](
            ctx, c, a, b, m, n, k, leaf, p_count, st, SWIZZLE_NONE, True
        )
        return
    if plan == PLAN_TILE_32_8_16:
        _launch_tiled[32, 8, 16](
            ctx, c, a, b, m, n, k, leaf, p_count, st, SWIZZLE_REVERSE, False
        )
        return
    if plan == PLAN_TILE_16_16_8_REV:
        _launch_tiled[16, 16, 8](
            ctx, c, a, b, m, n, k, leaf, p_count, st, SWIZZLE_TRANSPOSE, True
        )
        return
    if plan == PLAN_TILE_4_4_32_TSP:
        _launch_tiled[4, 4, 32](
            ctx, c, a, b, m, n, k, leaf, p_count, st, SWIZZLE_TRANSPOSE, False
        )
        return
    # PLAN_FLAT, and the fallback for both SPLITK plans at `k == 0` (there
    # are no partials to write, and section 8 still requires `+0.0` to be
    # STORED rather than the store skipped).
    ctx.enqueue_function[identical_gemm_flat_kernel](
        c.unsafe_ptr(),
        a.unsafe_ptr(),
        b.unsafe_ptr(),
        Int32(m),
        Int32(n),
        Int32(k),
        Int32(leaf),
        Int32(p_count),
        Int32(st[0]),
        Int32(st[1]),
        Int32(st[2]),
        Int32(st[3]),
        grid_dim=((m * n + FLAT_TPB - 1) // FLAT_TPB, 1, 1),
        block_dim=(FLAT_TPB, 1, 1),
    )


def identical_gemm_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises:
    """`C = op(A) . op(B)` into a CALLER-OWNED workspace, on the plan
    `choose_gemm_plan` picks. ASYNCHRONOUS: nothing here waits, and the
    caller must keep every buffer alive past its own `ctx.synchronize()`.

    `ws` must hold at least `identical_gemm_workspace_max_floats(m, n, k)`
    floats. **Sizing it for one plan and letting the dispatcher pick another
    is an out-of-bounds write that a small shape will not show you** -- it
    cost this lane a run: a 1-float workspace passed to a SPLITK dispatch at
    `64 x 4` still produced the right answer because the allocation had
    slack, and only at `64 x 64` (512 KB of partials) did whole regions of
    the output come back `+0.0`. `check_device_is_batch_invariant` is what
    caught it. Use the helper, not a guess.
    """
    # DEVIATION 1900 -- DEVIATION 1876'S SIBLING: THE _INTO FORM HAD NO MODE
    # BRANCH EITHER. `identical_gemm` grew its FAST branch (1876, below) and
    # this entry did not, so every caller that owns its workspace -- the
    # training loop's lm_head, the loss reductions, the GP mean -- kept
    # landing on the pinned kernel under FAST (~70x off the vendor GEMM on
    # H100). Same branch, same boundaries: `_fast_vendor_gemm` writes the
    # SAME caller-owned `c` with the same row-major shape semantics, is
    # stream-ordered on the same ctx and returns without waiting -- which is
    # this entry's documented contract already (ASYNCHRONOUS above), so no
    # postcondition moves. `ws` is simply unused on the vendor path; it is
    # the caller's buffer, not ours to free. OP_TN and any shape the vendor
    # arm declines return False and FALL THROUGH to the pinned plan exactly
    # as 1876 does. Under IDENTICAL this branch is not compiled at all:
    # bit-unchanged by construction.
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        if _fast_vendor_gemm(ctx, c, a, b, m, n, k, op):
            return
    identical_gemm_with_plan(
        ctx, c, a, b, ws, m, n, k, op, choose_gemm_plan(m, n, k)
    )


def _fast_vendor_gemm(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises -> Bool:
    """MAX's tuned matmul for the FAST arm. True if it served the shape.

    DEVIATION 1876 -- **THE FAST ARM HAD NO FAST PATH, AND THAT IS WHAT THE
    SEQUENCE LANES WERE PAYING.**

    `identical_gemm` had NO MODE BRANCH AT ALL. It ran the pinned
    balanced-tree kernel in both modes, and FAST only stripped `ftz` and the
    FMA pins from inside it. `core/gemm.mojo` DOES branch -- under FAST it
    calls `linalg.matmul` -- but `transformer/` and `mamba/` do not go
    through `core/gemm.mojo`; they import this file directly. So every
    projection, every score and every MLP matmul in those two lanes ran the
    hand-written kernel even when nobody had asked for identity.

    MEASURED ON AN H100, 2026-08-25, ONE RUN, SAME BOX, SAME SHAPES:

        our MLP block, llama8b t512                  51.07 ms
        the same three GEMMs via linalg.matmul   0.23 + 0.23 + 0.24 ms

    Seventy times, and not one instruction of it is the algorithm. It is the
    same three matrix products handed to the wrong kernel.

    WHAT THIS DOES NOT DO, and the boundaries are the point:

    * **NOTHING UNDER `NUMERIC_IDENTICAL`.** The caller of this function is
      comptime-gated, so under IDENTICAL this code is not compiled in. A
      vendor kernel's tile shape and k-split are per-vendor and a k-split IS
      a summation order; routing identity through a closed library is the
      one thing this whole profile exists to refuse.
    * **NOT `OP_TN`.** The vendor kernel expresses `transpose_b` and not
      `transpose_a`, so a TN shape would need a materialized transpose. That
      is a real cost with a real tradeoff and neither sequence lane asks for
      it -- `modeling_llama.mojo:282` imports `OP_NN` and `OP_NT` and
      nothing else. TN returns False here and falls through to the pinned
      kernel, which is correct, just not fast.
    * **NOT `n == 1` UNDER `OP_NT`.** `core/gemm.mojo` records what happens
      there and it is not an edge case to wave at: measured 2026-08-19 at
      m=64, n=1, k=32 with the output poisoned first, **63 of the 64 rows
      still held the poison afterwards.** `transpose_b=True` does not write
      them. A caller reusing a buffer reads whatever was in it last time,
      which is worse than zeros. The same product with `transpose_b=False`
      is correct at the identical shape, so the fault is the flag and not
      the shape. At `n == 1` this IS a matrix-vector product and it goes to
      `gemv_gpu`, which is what RAFT does for the same case.

    DEVIATION 1877 -- **THIS PATH DOES NOT SYNCHRONIZE AND DOES NOT NEED TO.**

    `identical_gemm`'s docstring says "THIS FORM SYNCHRONIZES BEFORE IT
    RETURNS, and it has to", and the reason it gives is specific: it
    ALLOCATES ITS OWN WORKSPACE, and `[[mojo-buffer-freed-at-last-use]]`
    means a buffer created in that function is dead at its `.unsafe_ptr()`,
    so returning without waiting would free it out from under a kernel that
    had not run yet.

    NONE OF THAT APPLIES HERE. This path allocates nothing. Every buffer it
    touches -- `a`, `b`, `c` -- is the caller's and outlives the call, and
    the enqueue is stream-ordered, so anything the caller enqueues next
    already sees the result. A caller that reads `c` on the host does it
    with `enqueue_copy` followed by a synchronize, which is ordered on the
    same stream.

    WHY IT IS WORTH REMOVING RATHER THAN LEAVING ALONE. A Llama block issues
    roughly thirty of these, and a sync per call turns a pipeline into thirty
    serialized round trips. The shape of that cost is visible in the numbers
    already taken: `mlp` at ONE token costs 1.446 ms while its three GEMMs
    measure about 0.27 ms in the gemm lane, a floor of over a millisecond
    that does not scale with work. `rmsnorm`, `mlp` and `attention` all sit
    within a factor of two of each other at t1 despite doing wildly
    different amounts of arithmetic, which is what a fixed per-call overhead
    looks like rather than a slow kernel.

    THIS CHANGES A DOCUMENTED POSTCONDITION and that is not something to do
    on reasoning alone. It is why the gemmseq leg runs `transformer_check`
    and `mamba_check` on the box: if any in-tree caller depended on the
    implicit wait, those gates are where it shows.

    FAST IS NOT BIT-STABLE AND NEVER WAS. This changes which bits FAST
    produces for the sequence lanes. The profile's promise lives entirely on
    the IDENTICAL side and is untouched here: the whole entry is gated behind
    `comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL`.

    **AND IT IS NOT ONLY A SUMMATION ORDER. IT IS A SHORTER MANTISSA.**
    DEVIATION 1885. The sentence that stood here said the bits move "because
    a different kernel sums in a different order", which is true and is not
    the whole truth, and being almost right is exactly why nobody re-derived
    it. `matmul` on an NVIDIA target is a TENSOR-CORE PATH: on the H100 it
    measured 200 TFLOP/s against `cublas-fp32`'s 44.4 and `cublas-tf32`'s
    207.5. It is on the TF32 line. **TF32 carries 10 explicit mantissa bits
    against fp32's 23.**

    THE CONTROL THAT PROVES IT, measured 2026-08-26 on one H100, agreement
    against the SAME torch fp32 reference before and after this entry landed:

        rmsnorm     (NO GEMM)   4.77e-06 -> 4.77e-06     UNCHANGED
        attention   (GEMM)        0.0183 -> 0.859         47x worse
        transformer (GEMM)        0.0889 -> 4.334         49x worse
        mlp         (GEMM)        0.0163 -> 3.888        239x worse

    A reordered summation does not move an absolute error by two orders of
    magnitude. `rmsnorm` routes no GEMM and did not move by one bit.

    SO THE CALLER OWES ITS READER A SENTENCE. A FAST path may be
    non-deterministic and may take the vendor's fastest kernel; it may NOT
    take a precision cut silently. Every lane that reaches this entry must
    say so beside its number, and its fair vendor opponent is a TF32 arm --
    `torch-gpu-tf32`, `cublas-tf32` -- not an fp32 one, for the same reason
    `bench/speed/gemm_speed_main.mojo`'s docstring already gives for the
    gemm lane. That reasoning was written down for `gemm` and then this
    entry let the identical cut into `transformer`, `attention` and `mlp`
    through the back door without anyone re-deriving it.

    Any FAST-mode tolerance gate downstream has to be RE-RUN rather than
    assumed. `verify.mamba_block.fast` FAILED on the leg that landed this
    (15 stages differ from the oracle, first at `norm.out`) while
    `verify.mamba_block.identical` PASSED -- the comptime gate held, and
    that failure is this precision cut arriving at a bit-exactness
    assertion, not a regression in the block.
    """
    if op == OP_NT:
        if n == 1:
            # `z[m] = a[m x k] . b[k]`: b is a contiguous k-vector whether
            # it is read as (1, k) or (k, 1), and c is a contiguous m-vector.
            var vz = TileTensor(c, row_major(m, 1))
            var vx = TileTensor(a, row_major(m, k))
            var vy = TileTensor(b, row_major(k, 1))
            gemv_gpu(vz, vx, vy, ctx)
            return True
        var tc = TileTensor(c, row_major(m, n))
        var ta = TileTensor(a, row_major(m, k))
        var tb = TileTensor(b, row_major(n, k))
        matmul[transpose_b=True, target="gpu"](tc, ta, tb, ctx)
        return True
    if op == OP_NN:
        var tc2 = TileTensor(c, row_major(m, n))
        var ta2 = TileTensor(a, row_major(m, k))
        var tb2 = TileTensor(b, row_major(k, n))
        matmul[transpose_b=False, target="gpu"](tc2, ta2, tb2, ctx)
        return True
    return False


def identical_gemm(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises:
    """**THE HOST-VISIBLE ENTRY POINT.** `C[m x n] = op(A) . op(B)`,
    row-major and contiguous throughout (contract section 2), under profile
    `mojolearn.identical.gemm.fp32.v1`.

    `op` is `OP_NN`, `OP_NT` or `OP_TN` from `gemm_oracle.mojo`. **The caller
    does not name a plan, does not size a workspace and does not know
    whether one was used** -- that is the point of the profile, and Phase 3
    and Phase 4 call this form.

    **THIS FORM SYNCHRONIZES BEFORE IT RETURNS, and it has to.**
    `[[mojo-buffer-freed-at-last-use]]`: a `DeviceBuffer` created here is
    dead at its `.unsafe_ptr()`, so a scratch buffer allocated inside this
    function and handed to an enqueued kernel would be FREED BEFORE THE
    KERNEL RAN if the function returned without waiting. A caller that wants
    to stay asynchronous owns the workspace itself and calls
    `identical_gemm_into`; the `_ = ws` below is what keeps this one's alive
    past the wait.
    """
    # DEVIATION 1876: under FAST, hand the shape to the vendor kernel. Under
    # IDENTICAL this branch is not compiled at all. Read `_fast_vendor_gemm`
    # before changing anything here; the `n == 1` clause in it is a
    # correctness requirement and not an optimization.
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        if _fast_vendor_gemm(ctx, c, a, b, m, n, k, op):
            return
    var nws = identical_gemm_workspace_max_floats(m, n, k)
    var ws = ctx.enqueue_create_buffer[DType.float32](nws)
    ctx.synchronize()
    identical_gemm_into(ctx, c, a, b, ws, m, n, k, op)
    ctx.synchronize()
    _ = ws


def identical_gemm_workspace_max_floats(m: Int, n: Int, k: Int) -> Int:
    """The workspace `identical_gemm` may need at this shape: what
    `choose_gemm_plan`'s answer costs. Phase 3 and Phase 4 allocate with
    this. Never less than 1, so the buffer is always constructible."""
    var w = identical_gemm_workspace_floats(
        m, n, k, choose_gemm_plan(m, n, k)
    )
    if w < 1:
        return 1
    return w
