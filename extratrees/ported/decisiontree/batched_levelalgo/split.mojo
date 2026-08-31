# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""The split record and its total order.

A PORT of cuML `cpp/src/decisiontree/batched-levelalgo/split.cuh`, pinned at
`00094f7` in `~/CascadeProjects/upstream/cuml`. Fields are theirs
(`split.cuh:38-45`), the default construction is theirs (`split.cuh:47-52`) and
`update` is their branch chain transcribed in their order (`split.cuh:76-90`).

WHY A cuML FILE IS THE UPSTREAM FOR A PAPER PORT
------------------------------------------------
The split RULE in this directory comes from Geurts, Ernst & Wehenkel 2006 by
way of scikit-learn's `RandomSplitter`, because no GPU library ships that
formulation. The split RECORD does not need inventing: cuML already wrote the
struct a GPU tree builder reduces over, and `PORTING_RULES.md` 0c is a list of
eleven times this project invented something that a competitor's file already
answered better. So the record, the tie-break and the validity test are ports.

`warpReduce` and `evalBestSplit` (`split.cuh:92-152`) and `initSplit`
(`:158-166`) ARE in this file, at the bottom, as `split_warp_reduce`,
`split_reduce_kernel` and `split_reduce_init_kernel`; they were added
2026-08-21 and are enqueued and checked by
`extratrees/mojo_only/split_reduce_check.mojo`. They carry DEVIATION BLOCKS
166-169, the largest of which is that OUR reduction key is not their float
field but the exact rational of DEVIATION 145, with their `update` -- the
function below, unchanged and CALLED rather than re-transcribed -- as its
tie-break.

**Two files outside this lane's ownership still say otherwise and are OPEN
items for their owners**: `extratrees/UNPORTED.tsv:15` still lists
`split.cuh::warpReduce+evalBestSplit` as "not yet", and
`extratrees/PORTED_MAP.tsv:10` still scopes this file to `split.cuh:32-90`
with "warpReduce/evalBestSplit still unported". Both sentences are false as of
this commit.
"""


# ==========================================================================
# DEVIATION BLOCK 133 -- the tie-break is a total order, sklearn's is not
#
# THEIRS (sklearn, `_splitter.pyx:693`): `if current_proxy_improvement >
#   best_proxy_improvement` -- strictly greater, so a tie is resolved by
#   whichever candidate the SEQUENTIAL draw loop reached first. That is a
#   statement about loop order, not an order on the candidates.
# OURS: `Split.update` below, cuML's `split.cuh:78-90` branch for branch --
#   greater metric wins; equal metric, greater colid wins; equal colid,
#   greater quesval wins.
# WHY: this builder evaluates candidates in parallel and reduces them in an
#   unspecified order, so "first" does not exist (DEVIATION 130). A total
#   order over the candidate's own fields is the only tie-break that survives
#   an unspecified reduction order, and cuML already wrote it.
# PRICE: on an exact tie we take a different feature than sklearn would. Ties
#   are what the duplicate-feature analytic fixture exists to produce, so the
#   behaviour is pinned by a check rather than left to chance.
# AMENDED BY DEVIATION 463 (2026-08-26): the price above turned out to be
#   PAID IN ACCURACY on covtype -- "greater colid wins" is a systematic bias
#   toward high column ids, not a neutral coin. The shipping tie order is now
#   the keyed pseudorandom rank of DEVIATION BLOCK 463 below; `Split.update`
#   ITSELF is unchanged (it is cuML's transcription, `split_check`'s
#   subject, and still the whole no-key reduction), and the max-colid rule
#   stays reachable via `MOJOLEARN_ET_TIE_MAX_COLID` /
#   `SPLIT_SAB_MAX_COLID_TIE` so this block's gate keeps its subject.
# ==========================================================================


@fieldwise_init
struct Split(ImplicitlyCopyable, Movable):
    """All info pertaining to splitting a node. `split.cuh:32-45`."""

    var quesval: Float32
    """Threshold to compare in this node. Theirs is `DataT quesval`."""

    var colid: Int32
    """Feature index. Theirs is `IdxT colid`."""

    var best_metric_val: Float32
    """Best info gain on this node. Theirs is `DataT best_metric_val`."""

    var n_left: Int32
    """Number of samples in the left child. Theirs is `int nLeft`."""

    comptime Min = Float32.MIN_FINITE
    """`split.cuh:36`: `-std::numeric_limits<DataT>::max()`.

    NOT negative infinity. Mojo spells the negative of the largest finite
    float `MIN_FINITE`; it is the same bit pattern their expression produces.
    """

    def __init__(out self):
        """`split.cuh:54-59`, the default constructor."""
        self.quesval = Self.Min
        self.best_metric_val = Self.Min
        self.colid = -1
        self.n_left = 0

    def update(mut self, other: Self) -> Bool:
        """Updates the current split if the input gain is better.

        `split.cuh:76-90`, transcribed branch for branch. Returns whether the
        update happened, as theirs does.
        """
        var update_result = False
        if other.best_metric_val > self.best_metric_val:
            update_result = True
        elif other.best_metric_val == self.best_metric_val:
            if other.colid > self.colid:
                update_result = True
            elif other.colid == self.colid:
                if other.quesval > self.quesval:
                    update_result = True
        if update_result:
            self = other
        return update_result

    def is_valid(self) -> Bool:
        """Whether a candidate was ever recorded here. `split.cuh:145` tests
        `this->colid != -1` before touching the shared best split."""
        return self.colid != -1


# ============================================================================
# THE DEVICE-SIDE REDUCTION. `split.cuh:92-152` (`warpReduce`,
# `evalBestSplit`) and `:158-166` (`initSplit`).
#
#     Verified by:  extratrees/mojo_only/split_reduce_check.mojo
#
# `Split.update` above is untouched: it IS the tie-break the reduction below
# imposes, and the shipping arm of the kernel calls it rather than
# transcribing it a second time (see DEVIATION BLOCK 166 for why that
# mattered enough to be a design constraint).
#
#     ==================================================================
#     DEVIATION BLOCK 166 -- the reduction carries the EXACT RATIONAL KEY
#     beside the `Split`, and `Split.update` becomes its TIE-BREAK
#
#     THEIRS. `Split::warpReduce` (`split.cuh:92-105`) shuffles exactly
#     four fields -- `quesval`, `colid`, `best_metric_val`, `nLeft` -- and
#     `update` (`:76-90`) decides on the first of them it can. The score
#     IS the reduction key, and it is one `DataT`.
#
#     OURS. The reduced payload is `SplitExact`: their four fields plus
#     an `ExactKey` (`num`, `den`, `valid`), and the comparison is
#
#         1. `compare_exact_key` -- DEVIATION 145's `CompareProxyExact`,
#            the Int128 cross-multiply, evaluated FIRST; then, only on an
#            exact tie,
#        2. `Split.update`'s branch chain, UNCHANGED and CALLED, not
#            re-transcribed: greater metric, then greater colid, then
#            greater quesval.
#
#     WHY IT HAD TO BE CARRIED AND COULD NOT BE COLLAPSED. DEVIATION 145
#     is a contract on this exact code: two candidates whose exact proxies
#     DIFFER can round to the same `Float32` (24 mantissa bits against
#     counts reaching 2^26), and a reduction keyed on the float alone then
#     falls through to the `colid` arm and picks by FEATURE INDEX. There
#     is no single scalar that can stand in for the rational: `num/den` in
#     `Float32` is the very rounding being avoided, `Float64` does not
#     exist on device, and a rational cannot be normalised to a common
#     denominator without the same overflow the comparison already has to
#     survive. So the pair travels.
#
#     HOW `update` STAYS THE ONE COPY. `SplitExact.update` does not
#     re-implement the chain. On an exact tie it takes a COPY of its own
#     `Split`, calls `Split.update` on the copy, and uses only the
#     BOOLEAN THAT FUNCTION RETURNS as the predicate; the assignment is
#     then done on the whole payload so the key travels with the split it
#     belongs to. The shipping path therefore executes the transcription
#     that `split_check.mojo` already checks against an independently
#     written comparator, and there is no second copy to drift.
#
#     THE ORDER IS TOTAL, AND HERE IS THE EXACT CONDITION. The relation is
#     lexicographic on (exact rational, metric, colid, quesval), each arm
#     a total order on its own field, so the composition is a total
#     PREORDER whose only ties are payloads equal in all four. Two such
#     payloads can still differ in `n_left` and in the REPRESENTATION of
#     an equal rational (`2/4` against `1/2`), and for those the incumbent
#     survives -- which is order-dependent. That case is unreachable in
#     the pipeline, because (colid, quesval) plus the node's rows
#     DETERMINE `n_left` and the class counts the key is built from, and
#     `split_reduce_check.mojo` asserts the fixture contains no such pair
#     rather than assuming it. DEVIATION 154 already recorded the same
#     boundary for `Split.update` alone.
#
#     REGRESSION IS THE SAME CODE WITH THE KEY SWITCHED OFF. DEVIATION 153
#     says the regression reduction key is sklearn's FLOAT proxy, and no
#     exact rational exists for MSE. A caller passes `ExactKey()` --
#     `valid = 0` -- for every candidate; `compare_exact_key` then returns
#     0 for every pair (two invalid keys tie, `objectives.mojo`'s own
#     rule), the composition degenerates to `Split.update`, and the
#     kernel is cuML's reduction exactly. One kernel, no branch on task.
#
#     PRICE, and it is paid in three places. (1) The payload is 36 bytes
#     rather than 16, so each butterfly step shuffles SEVEN values rather
#     than four and the per-warp shared scratch is 36 bytes a warp rather
#     than 16. (2) Every comparison that reaches the key costs two 64x64
#     -> 128 multiplies (DEVIATION BLOCK 167) instead of one `float`
#     compare. (3) `num` and `den` are `Int64`, so a 32-bit-only lane
#     shuffle would not carry them; `warp.shuffle_xor` on `Int64` was
#     measured to work on this target and is used directly. **No timing
#     number is attached to any of this and none will be until the perf
#     round** (`extratrees/PLAN.md`).
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 167 -- the Int128 cross-multiply is HAND-WIDENED
#     from 64-bit limbs, because `Int128` in a device kernel does not
#     compile
#
#     THEIRS. No counterpart: cuML compares one float.
#
#     OURS, on the host, is `GiniObjectiveFunction.CompareProxyExact`
#     (`objectives.mojo`), `Int128(a.num) * Int128(b.den)` against
#     `Int128(b.num) * Int128(a.den)`. `MAX_ROWS_EXACT` proves that is
#     exact to 2^26 rows.
#
#     OURS, ON DEVICE, IS THE SAME COMPARISON COMPUTED DIFFERENTLY.
#     `compare_exact_key` below splits each `UInt64` into two 32-bit limbs
#     and forms the 128-bit product as a (hi, lo) pair, then compares the
#     pairs lexicographically. Nothing about it is vendor-specific: four
#     32x32 -> 64 multiplies, two adds and two shifts, the schoolbook
#     widening every 128-bit integer library starts from.
#
#     WHY, and it is a MEASURED platform wall, not a preference. A kernel
#     containing `Int128(a) * Int128(b)` fails to build: the Metal
#     pipeline-state creation dies with `Compilation failed due to an
#     interrupted connection: XPC_ERROR_CONNECTION_INTERRUPTED. This error
#     occurred after multiple retries.` The identical kernel with the
#     products taken in `Int64` builds and runs, so the failure is the
#     128-bit multiply and not the harness. Reproduced twice on
#     2026-08-21 in a 35-line standalone probe, Mojo 1.0.0 (ed45d567). It
#     is the same shape of Metal-backend refusal DEVIATION 162 records for
#     a whole-struct kernel argument.
#
#     WHY THE HAND-WRITTEN FORM IS NOT AN INVENTION UNDER
#     `VENDOR_LIBRARIES.md`. The rule is to call the vendor's library
#     where the incumbent calls one. There is no `linalg`, `algorithm` or
#     `math` entry point in MAX for a 128-bit integer product, and there
#     is no vendor intrinsic to call on ANY of the three targets, since
#     the type itself is what the backend refuses. Nothing exists to call.
#
#     WHY IT IS SAFE TO BELIEVE. The check does not assert this function
#     against itself or against a hand-tallied expectation: it runs
#     `compare_exact_key` against the HOST's `Int128` form
#     (`CompareProxyExact`) pairwise, per cell, over products deliberately
#     built to exceed 2^64 so that the high limb is load-bearing, and it
#     COUNTS how many of those comparisons were decided by the high word
#     alone. A fixture whose products all fit in 64 bits would check the
#     easy half of this function and nothing else.
#
#     ALSO OURS, and slightly wider than 145 needs. The sign handling
#     covers a NEGATIVE numerator, which Gini's `sq_L*nR + sq_R*nL` cannot
#     produce. It is there because DEVIATION 153 leaves the regression key
#     open and an MSE numerator is not sign-constrained; the check
#     exercises both signs so the branch is not a dead one.
#
#     PRICE. Roughly ten integer operations where the host writes one
#     multiply, and a function that must be read against the host's to be
#     believed rather than read on its own.
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 168 -- `warpReduce` is a `shuffle_xor` butterfly
#     with NO ASSUMED WARP WIDTH
#
#     THEIRS. `split.cuh:92-105`: `for (int i = raft::WarpSize / 2; i >=
#     1; i /= 2) { auto id = lane + i; ... raft::shfl(quesval, id) ... }`.
#     `raft::WarpSize` is the compile-time constant 32, and the source
#     lane is `lane + i` taken modulo the warp width.
#
#     OURS. The same loop over `WARP_SIZE // 2 ... 1`, with
#     `warp.shuffle_xor` (`std.gpu.primitives.warp`) and `WARP_SIZE`
#     (`std.gpu`) -- the TARGET's warp width, 32 on NVIDIA and Apple, 64
#     on AMD CDNA, never a literal.
#
#     WHY XOR AND NOT `lane + i`. For lane 0 the two are the SAME
#     PERMUTATION at every step (`0 + i == 0 XOR i`), so lane 0's result
#     is bit-identical to theirs and their `@note` ("Best split will be
#     with 0th lane") still holds. They differ for other lanes, where
#     theirs leaves partial answers and XOR leaves the FULL answer in
#     every lane. That is a strengthening, and it is the one the platform
#     documents: `shuffle_xor` is the reduction idiom with the best
#     instruction scheduling, and the "only lane 0 is meaningful" caveat
#     is a footgun this reduction is called twice in a row by.
#
#     WHY NOT `warp.reduce` OR A BLOCK COLLECTIVE, which
#     `VENDOR_LIBRARIES.md` would otherwise require. `max.gpu.primitives`
#     offers `sum`, `max`, `min`, `broadcast` and `prefix_sum` over a
#     SCALAR. The thing being reduced here is a 36-byte record under a
#     four-level lexicographic order, which is none of those, and no
#     collective in MAX takes a user comparator. The shuffle primitives
#     ARE the vendor library here and they are what is called; only the
#     combine is ours, and the combine is `Split.update`.
#
#     PRICE. `log2(WARP_SIZE)` steps of seven shuffles each. On a 64-wide
#     wavefront that is one more step than on a 32-wide one, which is the
#     correct behaviour and not a cost.
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 169 -- `evalBestSplit`'s publish: a portable mutex,
#     a struct-of-arrays cell, and the device's own report of which path
#     it ran
#
#     THEIRS, `split.cuh:107-152`. `warpReduce`; lane 0 of each warp
#     writes its `Split` to `smem`; `__syncthreads()`; warp 0 loads
#     `sbest[lane]` for `lane < nWarps` and a default `Split()` otherwise;
#     `warpReduce` again; then `threadIdx.x == 0`, if `colid != -1`,
#     `while (atomicCAS(mutex, 0, 1));`, reads the node's `volatile
#     Split*` field by field, `update`s, writes back the four fields,
#     `__threadfence()`, `atomicExch(mutex, 0)`.
#
#     OURS. The same six steps in the same order. Three things are
#     spelled differently and each is forced:
#
#     (a) THE LOCK. `threadfence` is comptime-asserted NVIDIA-only in Mojo
#         1.0 (DEVIATION 106) and Metal rejects a strong compare-exchange
#         by name, so the established translation is used -- the one this
#         repository has already enqueued twice, for cuVS's cross-block
#         mutex and for `node_feature_range_kernel` (DEVIATION 161): spin
#         on an ACQUIRE load until the mutex reads free, claim with a WEAK
#         RELAXED compare-exchange, hand back with a RELEASE store, which
#         is where their `__threadfence(); atomicExch()` pair goes. Only
#         thread 0 of a block ever takes it, exactly as theirs does, so no
#         two threads of one warp can contend and the spin cannot
#         livelock a warp.
#
#     (b) THE CELL IS SEVEN ARRAYS, not one `volatile Split*`. DEVIATION
#         162 records the measured Metal-compiler failure on whole-struct
#         access through a pointer; their code is already field-by-field
#         (`:135-138`, `:146-149`) BECAUSE of `volatile`, so this is the
#         same shape for a different reason. The seven are `out_quesval`,
#         `out_colid`, `out_metric`, `out_nleft`, `out_num`, `out_den`,
#         `out_valid`, indexed by the node's slot.
#
#     (c) TWO COUNTERS WITH NO cuML COUNTERPART, written unconditionally
#         by the SHIPPING kernel and not behind a flag. `out_n_merges` is
#         how many BLOCKS published into the cell; `out_n_warps` is how
#         many WARPS carried a valid candidate into the cross-warp
#         combine. They exist because a reduction check that cannot NAME
#         the path it ran can pass about a path it never took: with one
#         warp per block the cross-warp step is a no-op that copies lane
#         0's own value, and a fixture that only ever ran that shape would
#         report the combine correct without ever exercising it. The
#         device reports the path; the host does not infer it.
#
#     WHY THE PUBLISH ORDER CANNOT CHANGE THE ANSWER. The merge is
#     `update` under the total order of DEVIATION BLOCK 166, so the cell
#     holds the maximum over the blocks that have published so far, and a
#     maximum under a total order is independent of the order the
#     candidates arrive in. That is the same argument DEVIATION 161 makes
#     for min/max, made for a lexicographic order instead of a numeric
#     one, and the check proves it by PERMUTING the candidate array rather
#     than by asserting it.
#
#     PRICE. One `Int32` mutex per node that the caller must zero, one
#     `split_reduce_init_kernel` launch to seed the cells, and a
#     serialized publish per block instead of a wait-free atomic -- which
#     no float rational could have used anyway.
#     ==================================================================
# ============================================================================

from std.atomic import Atomic, Ordering
from std.gpu import WARP_SIZE, block_dim, block_idx, grid_dim, lane_id, thread_idx
from std.gpu.primitives import warp
from std.memory import stack_allocation
from std.sys.compile import is_defined
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from extratrees.mojo_only.pcg_rng import FNV1A32_BASIS, fmix32, fnv1a32


# ==========================================================================
# DEVIATION BLOCK 463 -- on an EXACT tie the winner is a keyed pseudorandom
# rank over (tree, node, colid), not the highest column id
#
# THEIRS (sklearn, `_splitter.pyx:693`): strict `>`, so a tie goes to the
#   FIRST candidate in a uniformly-random visit order -- which makes the
#   tied winner UNIFORM AMONG THE TIED, per node, per draw.
# OURS UNTIL 2026-08-26: cuML's `colid` arm -- every exact tie resolved
#   toward the HIGHEST column id, deterministically, tree after tree. Our
#   exact integer rational (DEVIATION 145) makes ties REAL events, and on
#   covtype (44 one-hot indicator columns at ids 10-53, 10 continuous ones
#   at 0-9) the max-colid rule systematically funneled every tied node into
#   the one-hot block: 0.6701 against sklearn's 0.6768 at depth 16.
# NOW: on `compare_exact_key(...) == 0` (both keys valid), challengers are
#   ranked by `split_tie_rank` -- `fmix32(fnv1a32(tie_salt, colid))` with
#   `tie_salt` chained from `(SPLIT_TIE_SALT, tree_id, node_id)` -- and the
#   GREATEST rank wins; only a rank collision falls back to the colid arm
#   (then quesval), so the order stays TOTAL and the reduction stays
#   order-independent. Uniform-among-ties in sklearn's sense, deterministic
#   from `(seed-independent) (tree, node, colid)` alone, identical on host
#   (`host_splitter._wins_on_total_order`) and device (this file), integer
#   math only, so the IDENTICAL cross-vendor mode is untouched.
# ARMS: comptime `-D MOJOLEARN_ET_TIE_MAX_COLID=1` restores the max-colid
#   rule build-wide (host AND device together, so oracle parity holds in
#   both builds); runtime `SPLIT_SAB_MAX_COLID_TIE` restores it in the
#   shipping binary for the reduce check, so DEVIATION 133's
#   reproducibility gate keeps its subject. The NO-KEY degeneration
#   (DEVIATION 166: both keys invalid) is untouched and still reduces by
#   `Split.update` exactly -- cuML's own chain -- because no shipping
#   caller reaches it (device regression carries valid keys, DEVIATION
#   189) and `split_reduce_check` arm A' pins that shape by name.
# ==========================================================================

comptime ET_TIE_BREAK_KEYED = not is_defined["MOJOLEARN_ET_TIE_MAX_COLID"]()
"""DEVIATION 463's build switch. Default: keyed rank. `-D
MOJOLEARN_ET_TIE_MAX_COLID=1`: the pre-463 max-colid tie-break, everywhere."""

comptime SPLIT_TIE_SALT: UInt32 = 0x7B31C853
"""DEVIATION 463's salt. Fresh constant (grep: used nowhere else), declared
the way `EXCESS_SELECTION_SALT` and `THRESHOLD_KEY_SALT` are, so the rank
stream is disjoint from every other fnv1a32 stream in the tree."""


def split_tie_salt_for(tree_id: UInt32, node_id: UInt32) -> UInt32:
    """The per-node half of DEVIATION 463's rank key, computed ONCE per node.

    `fnv1a32(fnv1a32(fnv1a32(BASIS, SPLIT_TIE_SALT), tree_id), node_id)`.
    Host and device must call THIS function -- the builder stages one value
    per work item (`node_tie_salt`), the host splitter computes it at the
    top of the node loop -- so the rule cannot drift between the two.
    """
    var h = fnv1a32(FNV1A32_BASIS, SPLIT_TIE_SALT)
    h = fnv1a32(h, tree_id)
    return fnv1a32(h, node_id)


def split_tie_rank(tie_salt: UInt32, colid: Int32) -> UInt32:
    """DEVIATION 463's rank: one more fnv1a32 round for the column, then the
    `fmix32` avalanche (DEVIATION 464's lesson: an un-finalized fnv chain on
    a small word is a Weyl rotation, not a permutation)."""
    return fmix32(fnv1a32(tie_salt, colid.cast[DType.uint32]()))


def keyed_tie_wins(tie_salt: UInt32, cand: Split, best: Split) -> Bool:
    """Does `cand` beat `best` under DEVIATION 463's tie order?

    Greatest `split_tie_rank` wins; a rank collision falls back to the colid
    arm and then the quesval arm (cuML `split.cuh:85`, `:88`), so the
    relation is a total order on the same boundary DEVIATION 154 recorded.
    ONE copy: `SplitExact.update` and `host_splitter._wins_on_total_order`
    both call this function rather than transcribing it.
    """
    var rc = split_tie_rank(tie_salt, cand.colid)
    var rb = split_tie_rank(tie_salt, best.colid)
    if rc != rb:
        return rc > rb
    if cand.colid > best.colid:
        return True
    if cand.colid == best.colid:
        return cand.quesval > best.quesval
    return False


# Sabotage selectors. A kernel ARGUMENT and not a comptime parameter, for the
# reason `builder_kernels_impl.mojo` states: a sabotage compiled into a
# different binary proves nothing about the binary that ships, so every arm of
# `split_reduce_check.mojo` runs THIS kernel.
comptime SPLIT_SAB_NONE = 0
"""No sabotage. The shipping path."""

comptime SPLIT_SAB_FLOAT_KEY = 1
"""Drop the exact rational and reduce on `best_metric_val` alone -- cuML's
own key, which DEVIATION 145 says is the wrong one for classification."""

comptime SPLIT_SAB_NO_COLID_ARM = 2
"""Drop `update`'s second arm (`other.colid > colid`, `split.cuh:85`)."""

comptime SPLIT_SAB_NO_QUESVAL_ARM = 3
"""Drop `update`'s third arm (`other.quesval > quesval`, `split.cuh:88`)."""

comptime SPLIT_SAB_NO_CROSS_WARP = 4
"""Warp 0 reads slot 0 for every lane, so only warp 0's own result survives
the cross-warp combine (`split.cuh:126-127`)."""

comptime SPLIT_SAB_INVERT_GUARD = 5
"""Invert the publish guard at `split.cuh:143-146` -- write when
`split_reg.update()` says NOT to, and not when it says to. That guard is what
makes the cell a running maximum over the blocks.

**It is inverted rather than simply dropped, and the reason is a measured
defect in the CHECK.** The first version of this arm published
unconditionally, "last writer wins"; that turned `split_reduce_check.mojo`
red in only 6 runs of 8, because with three multi-block cells the last block
to take the lock sometimes IS the one holding the winner. A sabotage that is
only usually red is not evidence. Inverting is deterministic: the first block
to reach a freshly seeded cell always improves it, so the inverted guard never
writes, and every non-empty cell keeps its seed no matter which block gets
there first."""

comptime SPLIT_SAB_NO_LOCK = 6
"""Publish without taking the mutex at all (`split.cuh:133-134`), with the
read-modify-write window deliberately widened so the race is observable
rather than merely possible."""

comptime SPLIT_SAB_BLOCK0_ONLY = 7
"""Publish only from block 0, so a node served by several blocks sees one
block's slice. Nothing in cuML corresponds; it is the cross-BLOCK counterpart
of NO_CROSS_WARP."""

comptime SPLIT_SAB_MAX_COLID_TIE = 8
"""Restore the pre-463 tie-break: on an exact key tie the HIGHEST column id
wins outright, no keyed rank. This is the arm DEVIATION 133's reproducibility
gate keeps as its subject, and the arm the orchestrator A/Bs against the
keyed rule; on covtype it is the one that funnels ties into the one-hot
columns. Inert when the build itself selects the old rule
(`MOJOLEARN_ET_TIE_MAX_COLID`)."""


@fieldwise_init
struct ExactKey(ImplicitlyCopyable, Movable):
    """The exact rational a classification candidate is ordered by.

    OURS, DEVIATION 166. It is `GiniProxyExact` (`objectives.mojo`) with
    `length` dropped -- `length` is a node constant that only `value()` uses,
    and the reduction never needs it -- and `valid` as an `Int32` rather than
    a `Bool` because it is shuffled between lanes with the numeric fields.
    """

    var num: Int64
    """`GiniProxyExact.num`. `sq_L*nR + sq_R*nL` for Gini."""

    var den: Int64
    """`GiniProxyExact.den`. `nL*nR`, strictly positive when `valid != 0`."""

    var valid: Int32
    """`GiniProxyExact.valid` as 0/1. **ZERO ALSO MEANS ABSENT**: a caller
    with no exact key at all (regression, DEVIATION 153) passes `ExactKey()`
    for every candidate, every comparison ties, and the composed order
    degenerates to `Split.update`."""

    def __init__(out self):
        """The absent key. Orders below every present one, and ties with
        another absent one -- `CompareProxyExact`'s own rule for two invalid
        candidates (`objectives.mojo`)."""
        self.num = 0
        self.den = 0
        self.valid = 0


def _mul_wide_u64(a: UInt64, b: UInt64, mut hi: UInt64, mut lo: UInt64):
    """The 128-bit product of two `UInt64`s as two `UInt64` limbs.

    Schoolbook widening on 32-bit limbs. DEVIATION BLOCK 167: `Int128` is
    what the host uses and what the Metal backend refuses to compile.
    """
    var a0 = a & 0xFFFFFFFF
    var a1 = a >> 32
    var b0 = b & 0xFFFFFFFF
    var b1 = b >> 32

    var t = a0 * b0
    var w0 = t & 0xFFFFFFFF
    var k = t >> 32

    t = a1 * b0 + k
    var w1 = t & 0xFFFFFFFF
    var w2 = t >> 32

    t = a0 * b1 + w1
    k = t >> 32

    lo = (t << 32) + w0
    hi = a1 * b1 + w2 + k


def _cmp_wide_products(a: UInt64, b: UInt64, c: UInt64, d: UInt64) -> Int32:
    """`a*b` against `c*d`, exactly, in 64-bit arithmetic. -1 / 0 / +1."""
    var hi1 = UInt64(0)
    var lo1 = UInt64(0)
    var hi2 = UInt64(0)
    var lo2 = UInt64(0)
    _mul_wide_u64(a, b, hi1, lo1)
    _mul_wide_u64(c, d, hi2, lo2)
    if hi1 != hi2:
        return Int32(1) if hi1 > hi2 else Int32(-1)
    if lo1 != lo2:
        return Int32(1) if lo1 > lo2 else Int32(-1)
    return Int32(0)


def compare_exact_key(a: ExactKey, b: ExactKey) -> Int32:
    """`GiniObjectiveFunction.CompareProxyExact`, without `Int128`.

    Returns -1 if `a` scores below `b`, +1 if above, 0 on an EXACT tie. The
    validity arms are theirs verbatim: an invalid (or absent) key ranks below
    every valid one, and two invalid keys tie. The rational arm is
    `a.num*b.den` against `b.num*a.den` -- the same cross-multiply, taken in
    two 64-bit limbs because the 128-bit type does not survive the device
    compiler (DEVIATION BLOCK 167).

    The sign split is wider than Gini needs and is deliberate; see the same
    block.
    """
    if a.valid == 0 and b.valid == 0:
        return Int32(0)
    if a.valid == 0:
        return Int32(-1)
    if b.valid == 0:
        return Int32(1)

    var sa = Int32(0)
    if a.num > 0:
        sa = Int32(1)
    elif a.num < 0:
        sa = Int32(-1)
    var sb = Int32(0)
    if b.num > 0:
        sb = Int32(1)
    elif b.num < 0:
        sb = Int32(-1)
    if sa != sb:
        return Int32(1) if sa > sb else Int32(-1)
    if sa == 0:
        return Int32(0)

    var an = UInt64(a.num) if a.num > 0 else UInt64(-a.num)
    var bn = UInt64(b.num) if b.num > 0 else UInt64(-b.num)
    var mag = _cmp_wide_products(an, UInt64(b.den), bn, UInt64(a.den))
    if sa > 0:
        return mag
    return -mag


@fieldwise_init
struct SplitExact(ImplicitlyCopyable, Movable):
    """The record the device reduction actually moves. DEVIATION 166.

    `split` is theirs, unchanged. `key` is the exact rational DEVIATION 145
    makes the authority for classification, and `ExactKey()` switches it off
    for a caller that has none.
    """

    var split: Split
    var key: ExactKey

    def __init__(out self):
        """`split.cuh:54-59` extended: their default `Split`, and the absent
        key. This is the identity of the reduction -- it loses to every real
        candidate and `is_valid()` reports it as never written."""
        self.split = Split()
        self.key = ExactKey()

    def update(mut self, other: Self, sabotage: Int32, tie_salt: UInt32) -> Bool:
        """Their `update` (`split.cuh:76-90`) with the exact key ahead of it.

        The chain is: exact rational first; on an exact tie, DEVIATION 463's
        keyed rank (falling back to colid, then quesval); with no rational at
        all, `Split.update`'s own three arms, IN THEIR CODE and not
        transcribed again -- the copy below exists only so that this function
        can use the boolean their function returns as a predicate and then
        assign the WHOLE payload, key included. Returns whether the update
        happened, as theirs does.

        `tie_salt` is `split_tie_salt_for(tree_id, node_id)`, block-uniform
        within one node's reduction; it only matters on an exact tie between
        two VALID keys. `sabotage` selects the arms; `SPLIT_SAB_NONE` is the
        shipping path.
        """
        var c = compare_exact_key(other.key, self.key)
        if sabotage == SPLIT_SAB_FLOAT_KEY:
            # THE SABOTAGE MEANS "reduce on best_metric_val alone", which is
            # cuML's own key. Forcing the rational to tie is only half of that
            # now: since DEVIATION 194 the tie branch SKIPS the metric when
            # both keys are valid, so `c = 0` on its own would produce a
            # colid-keyed fold rather than a float-keyed one. The arm
            # therefore also takes the no-key path below, which is
            # `Split.update` untouched -- and that IS reducing on the float.
            c = Int32(0)

        var update_result = False
        if c > 0:
            update_result = True
        elif c == 0:
            if (
                sabotage == SPLIT_SAB_NO_COLID_ARM
                or sabotage == SPLIT_SAB_NO_QUESVAL_ARM
            ):
                # `split.cuh:78-90` again, with one arm removed. This branch
                # is REACHED ONLY BY A SABOTAGE ARM; the shipping path is the
                # `else` below, which calls the checked transcription.
                if other.split.best_metric_val > self.split.best_metric_val:
                    update_result = True
                elif other.split.best_metric_val == self.split.best_metric_val:
                    if (
                        other.split.colid > self.split.colid
                        and sabotage != SPLIT_SAB_NO_COLID_ARM
                    ):
                        update_result = True
                    elif other.split.colid == self.split.colid:
                        if (
                            other.split.quesval > self.split.quesval
                            and sabotage != SPLIT_SAB_NO_QUESVAL_ARM
                        ):
                            update_result = True
            else:
                # ==========================================================
                # DEVIATION BLOCK 194 -- on an exact tie the metric arm is
                # SKIPPED, on this side as it already was on the host's.
                #
                # THEIRS: `Split::update` (`split.cuh:78-90`) tests
                #   best_metric_val, then colid, then quesval. For cuML that
                #   is right, because the metric IS their key.
                # OURS: the exact rational is the key (DEVIATION 145), and
                #   `host_splitter.mojo::_wins_on_total_order` already orders
                #   an exact tie by (colid, quesval) with the metric arm
                #   dropped. This side called `Split.update`, which still had
                #   it -- harmless while the device published a constant
                #   metric, and not harmless once it published a real one.
                #
                # MEASURED, which is how it was found. Computing cuML's
                # GainPerSplit on the device (DEVIATION 183, second form) made
                # the device metric real; the device and host metrics then
                # differed on 4 of 747 nodes by at most 4 ulp -- a float
                # DIVISION rounds differently on Metal than on the host -- and
                # those 4 flipped exact-rational ties, cascading into 11 of
                # 747 nodes with a different split and 18 differing leaf
                # values.
                #
                # WHY DROPPING IT IS MORE CORRECT, NOT A CONCESSION. Within a
                # node, gain = num/(den*n) - sq_total/n^2 and sq_total/n is a
                # node constant, so the gain is a strictly increasing function
                # of the exact rational. Two candidates whose rationals are
                # EQUAL therefore have EQUAL true gains, and any difference
                # between their floats is pure rounding. Ordering by it is
                # ordering by noise, which is the thing DEVIATION 145 exists
                # to forbid -- 145 argued it from a Float32 collision, and
                # this is the same argument with a second measurement.
                #
                # THE CONDITION IS "BOTH KEYS VALID", NOT "EXACT TIE", AND
                # THE FIRST VERSION OF THIS BLOCK GOT THAT WRONG. Written as
                # an unconditional skip it also stripped the metric from the
                # NO-KEY path -- where `compare_exact_key` ties every pair and
                # the metric is the only ranking that exists -- so a caller
                # with no exact key would have ranked by feature index alone.
                # `split_reduce_check`'s arm A' is what caught it: "no-key node
                # did not degenerate to Split.update".
                #
                # `Split.update` itself is UNCHANGED, still checked by
                # `split_check`, and still the whole tie-break when there is no
                # rational to reason from.
                # ==========================================================
                if (
                    other.key.valid != 0
                    and self.key.valid != 0
                    and sabotage != SPLIT_SAB_FLOAT_KEY
                ):
                    # BOTH rationals are valid and EQUAL, so the true gains
                    # are equal and any float difference is noise: skip the
                    # metric arm. DEVIATION 463: the keyed rank decides the
                    # tie (colid, then quesval, only on a rank collision);
                    # the comptime switch and the MAX_COLID_TIE sabotage arm
                    # keep the pre-463 max-colid rule reachable.
                    var keyed = False
                    comptime if ET_TIE_BREAK_KEYED:
                        keyed = sabotage != SPLIT_SAB_MAX_COLID_TIE
                    if keyed:
                        update_result = keyed_tie_wins(
                            tie_salt, other.split, self.split
                        )
                    elif other.split.colid > self.split.colid:
                        update_result = True
                    elif other.split.colid == self.split.colid:
                        if other.split.quesval > self.split.quesval:
                            update_result = True
                else:
                    # NO rational to reason from -- `compare_exact_key` ties
                    # every pair when a key is invalid -- so the metric is the
                    # only ranking there is, and `Split.update` is exactly
                    # cuML's. This is the arm a caller with no exact key takes,
                    # and it must NOT be weakened: dropping the metric there
                    # would rank by feature index alone.
                    var probe = self.split
                    update_result = probe.update(other.split)

        if update_result:
            self = other
        return update_result


def split_warp_reduce(
    s: SplitExact, sabotage: Int32, tie_salt: UInt32
) -> SplitExact:
    """`Split::warpReduce`, `split.cuh:92-105`. DEVIATION BLOCK 168.

    A butterfly over the TARGET's `WARP_SIZE`, seven shuffles a step. Every
    lane ends holding the warp's best; theirs guarantees only lane 0, and for
    lane 0 the two permutations are identical.

    Every lane of the warp must reach this function: the shuffles carry their
    own membership mask and a lane that never arrives hangs the rest.
    """
    var acc = s
    var off = UInt32(WARP_SIZE // 2)
    while off >= 1:
        var other = SplitExact(
            Split(
                warp.shuffle_xor(acc.split.quesval, off),
                warp.shuffle_xor(acc.split.colid, off),
                warp.shuffle_xor(acc.split.best_metric_val, off),
                warp.shuffle_xor(acc.split.n_left, off),
            ),
            ExactKey(
                warp.shuffle_xor(acc.key.num, off),
                warp.shuffle_xor(acc.key.den, off),
                warp.shuffle_xor(acc.key.valid, off),
            ),
        )
        _ = acc.update(other, sabotage, tie_salt)
        off //= 2
    return acc


def split_reduce_shared_bytes(tpb: Int, warp_size: Int) -> Int:
    """Shared memory `split_reduce_kernel[tpb]` needs, in bytes.

    HOST-side, so a caller can price the launch against the device's queried
    `MAX_SHARED_MEMORY_PER_BLOCK` instead of against this laptop. One
    `SplitExact` per warp, unbundled: 4+4+4+4+8+8+4 bytes.
    """
    return (tpb // warp_size) * 36


def split_reduce_init_kernel(
    out_quesval: MutPointer[Float32, MutAnyOrigin],
    out_colid: MutPointer[Int32, MutAnyOrigin],
    out_metric: MutPointer[Float32, MutAnyOrigin],
    out_nleft: MutPointer[Int32, MutAnyOrigin],
    out_num: MutPointer[Int64, MutAnyOrigin],
    out_den: MutPointer[Int64, MutAnyOrigin],
    out_valid: MutPointer[Int32, MutAnyOrigin],
    out_n_merges: MutPointer[Int32, MutAnyOrigin],
    out_n_warps: MutPointer[Int32, MutAnyOrigin],
    len: Int32,
):
    """`initSplit`, `split.cuh:158-166`, as a grid-stride write-only map.

    Theirs writes a default-constructed `Split` over the array with
    `raft::linalg::writeOnlyUnaryOp`; ours writes the same four fields plus
    the absent key and zeroes the two path counters. The publish below READS
    the cell it updates, so an unseeded cell is read garbage.
    """
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while idx < Int(len):
        out_quesval[unsafe_offset=idx] = Split.Min
        out_colid[unsafe_offset=idx] = Int32(-1)
        out_metric[unsafe_offset=idx] = Split.Min
        out_nleft[unsafe_offset=idx] = Int32(0)
        out_num[unsafe_offset=idx] = Int64(0)
        out_den[unsafe_offset=idx] = Int64(0)
        out_valid[unsafe_offset=idx] = Int32(0)
        out_n_merges[unsafe_offset=idx] = Int32(0)
        out_n_warps[unsafe_offset=idx] = Int32(0)
        idx += stride


def split_reduce_kernel[
    TPB: Int
](
    out_quesval: MutPointer[Float32, MutAnyOrigin],
    out_colid: MutPointer[Int32, MutAnyOrigin],
    out_metric: MutPointer[Float32, MutAnyOrigin],
    out_nleft: MutPointer[Int32, MutAnyOrigin],
    out_num: MutPointer[Int64, MutAnyOrigin],
    out_den: MutPointer[Int64, MutAnyOrigin],
    out_valid: MutPointer[Int32, MutAnyOrigin],
    out_n_merges: MutPointer[Int32, MutAnyOrigin],
    out_n_warps: MutPointer[Int32, MutAnyOrigin],
    mutexes: MutPointer[Int32, MutAnyOrigin],
    cand_quesval: MutPointer[Float32, MutAnyOrigin],
    cand_colid: MutPointer[Int32, MutAnyOrigin],
    cand_metric: MutPointer[Float32, MutAnyOrigin],
    cand_nleft: MutPointer[Int32, MutAnyOrigin],
    cand_num: MutPointer[Int64, MutAnyOrigin],
    cand_den: MutPointer[Int64, MutAnyOrigin],
    cand_valid: MutPointer[Int32, MutAnyOrigin],
    node_begin: MutPointer[Int32, MutAnyOrigin],
    node_count: MutPointer[Int32, MutAnyOrigin],
    node_tie_salt: MutPointer[UInt32, MutAnyOrigin],
    blocks_per_node_in: Int32,
    sabotage_in: Int32,
):
    """The best candidate per node, reduced on device. `split.cuh:92-152`.

    `node_tie_salt[nid]` is `split_tie_salt_for(tree_id, node_id)` for the
    node this cell reduces -- DEVIATION 463's per-node rank key, staged by
    the caller (the builder computes it in `stage_batch` from the SAME
    `(item_trees[i], work_items[i].idx)` the host oracle keys on).

    GRID. `grid_dim = (blocks_per_node, n_nodes)`, `block_dim = TPB`.
    `block_idx.y` is the node -- the cell -- and `block_idx.x` is which slice
    of that node's candidate list this block folds, the same
    several-blocks-per-node shape `computeSplitKernel` gets from
    `WorkloadInfo` (`builder_kernels_impl.cuh:236-244`).

    THE CANDIDATES ARE SEVEN PARALLEL ARRAYS, `node_begin[nid] ..
    +node_count[nid]`, and the output is seven more indexed by `nid`;
    DEVIATION BLOCK 169(b) for why nothing is a struct on either side of a
    pointer. A caller with no exact key writes `cand_valid = 0` and gets
    cuML's own reduction (DEVIATION BLOCK 166).

    THE ANSWER DOES NOT DEPEND ON THE ORDER. Per-thread fold, butterfly warp
    reduce, per-warp shared scratch, cross-warp reduce, cross-block mutex
    merge -- five different regroupings of one `update`, which is a maximum
    under a total order and therefore blind to grouping. The check proves it
    by permuting the input, not by repeating this sentence.

    SHARED MEMORY. `split_reduce_shared_bytes(TPB, WARP_SIZE)`, one
    `SplitExact` per warp, unbundled into seven arrays. The caller prices it
    against the device's queried budget; a `comptime assert` below refuses a
    `TPB` that is not a whole number of warps, because `nWarps = blockDim.x /
    WarpSize` (`split.cuh:124`) silently becomes zero otherwise on a 64-wide
    wavefront.
    """
    comptime assert (
        TPB % WARP_SIZE == 0
    ), "split_reduce_kernel: TPB must be a whole number of warps on this target"
    comptime NW = TPB // WARP_SIZE

    var nid = Int(block_idx.y)
    var blk = Int(block_idx.x)
    var sab = sabotage_in
    var begin = Int(node_begin[unsafe_offset=nid])
    var count = Int(node_count[unsafe_offset=nid])
    var tie_salt = node_tie_salt[unsafe_offset=nid]
    var bpn = Int(blocks_per_node_in)

    # The per-thread fold. Their `computeSplitKernel` walks the bins of one
    # feature this way (`builder_kernels_impl.cuh:264-265`, `:322-330`); the
    # candidates are already scored here, so the body is one `update`.
    var acc = SplitExact()
    var stride = TPB * bpn
    var end = begin + count
    var i = begin + Int(thread_idx.x) + blk * TPB
    while i < end:
        var cand = SplitExact(
            Split(
                cand_quesval[unsafe_offset=i],
                cand_colid[unsafe_offset=i],
                cand_metric[unsafe_offset=i],
                cand_nleft[unsafe_offset=i],
            ),
            ExactKey(
                cand_num[unsafe_offset=i],
                cand_den[unsafe_offset=i],
                cand_valid[unsafe_offset=i],
            ),
        )
        _ = acc.update(cand, sab, tie_salt)
        i += stride

    # ---- `Split::evalBestSplit`, `split.cuh:107-152` ----------------------
    # `:120` -- warpReduce().
    acc = split_warp_reduce(acc, sab, tie_salt)

    var lane = Int(lane_id())
    var warp_id = Int(thread_idx.x) // WARP_SIZE

    # `:118` -- `auto* sbest = reinterpret_cast<SplitT*>(smem)`, unbundled.
    var s_quesval = stack_allocation[
        NW, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var s_colid = stack_allocation[
        NW, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var s_metric = stack_allocation[
        NW, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var s_nleft = stack_allocation[
        NW, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var s_num = stack_allocation[
        NW, Scalar[DType.int64], address_space = AddressSpace.SHARED
    ]()
    var s_den = stack_allocation[
        NW, Scalar[DType.int64], address_space = AddressSpace.SHARED
    ]()
    var s_valid = stack_allocation[
        NW, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()

    # `:125` -- `if (lane == 0) sbest[warp] = *this;`
    if lane == 0:
        s_quesval[unsafe_offset=warp_id] = acc.split.quesval
        s_colid[unsafe_offset=warp_id] = acc.split.colid
        s_metric[unsafe_offset=warp_id] = acc.split.best_metric_val
        s_nleft[unsafe_offset=warp_id] = acc.split.n_left
        s_num[unsafe_offset=warp_id] = acc.key.num
        s_den[unsafe_offset=warp_id] = acc.key.den
        s_valid[unsafe_offset=warp_id] = acc.key.valid

    # `:126` -- `__syncthreads()`. Unconditional: every thread reaches it.
    barrier()

    if warp_id == 0:
        # `:128-131` -- `if (lane < nWarps) *this = sbest[lane]; else *this =
        # SplitT();`
        var src = lane
        if sab == SPLIT_SAB_NO_CROSS_WARP:
            src = 0
        if lane < NW:
            acc = SplitExact(
                Split(
                    s_quesval[unsafe_offset=src],
                    s_colid[unsafe_offset=src],
                    s_metric[unsafe_offset=src],
                    s_nleft[unsafe_offset=src],
                ),
                ExactKey(
                    s_num[unsafe_offset=src],
                    s_den[unsafe_offset=src],
                    s_valid[unsafe_offset=src],
                ),
            )
        else:
            acc = SplitExact()

        # OURS, DEVIATION 169(c): how many warps carried a real candidate
        # into this combine, read from the warp's OWN slot so a sabotage of
        # the combine cannot also forge the report. A warp collective, so
        # every lane of warp 0 must reach it -- which they do, the branch is
        # warp-uniform.
        var mine = Int32(0)
        if lane < NW:
            if s_colid[unsafe_offset=lane] != Int32(-1):
                mine = Int32(1)
        var n_contrib = warp.sum(mine)

        # `:132` -- warpReduce() again.
        acc = split_warp_reduce(acc, sab, tie_salt)

        # `:134` -- `if (threadIdx.x == 0 && this->colid != -1)`.
        if Int(thread_idx.x) == 0 and acc.split.is_valid():
            if not (sab == SPLIT_SAB_BLOCK0_ONLY and blk != 0):
                var slot = nid

                # `:135-136` -- `while (atomicCAS(mutex, 0, 1));`, as the
                # acquire-spin plus weak relaxed claim of DEVIATION 169(a).
                if sab != SPLIT_SAB_NO_LOCK:
                    while True:
                        if (
                            Atomic.load[ordering = Ordering.ACQUIRE](
                                mutexes.unsafe_offset(slot)
                            )
                            != Int32(0)
                        ):
                            continue
                        var expected = Int32(0)
                        if Atomic.compare_exchange[
                            success_ordering = Ordering.RELAXED,
                            failure_ordering = Ordering.RELAXED,
                            weak=True,
                        ](mutexes.unsafe_offset(slot), expected, Int32(1)):
                            break

                # `:137-142` -- `SplitT split_reg; split_reg.quesval =
                # split->quesval; ...` field by field off the cell.
                var cur = SplitExact(
                    Split(
                        out_quesval[unsafe_offset=slot],
                        out_colid[unsafe_offset=slot],
                        out_metric[unsafe_offset=slot],
                        out_nleft[unsafe_offset=slot],
                    ),
                    ExactKey(
                        out_num[unsafe_offset=slot],
                        out_den[unsafe_offset=slot],
                        out_valid[unsafe_offset=slot],
                    ),
                )

                # The NO_LOCK arm widens the read-modify-write window so that
                # the missing mutual exclusion is OBSERVED rather than merely
                # possible. The load keeps the loop from being folded away;
                # the comparison it feeds can never be true.
                if sab == SPLIT_SAB_NO_LOCK:
                    var spin = 0
                    while spin < 4096:
                        if Atomic.load[ordering = Ordering.RELAXED](
                            out_n_merges.unsafe_offset(slot)
                        ) == Int32(-424242):
                            spin += 1 << 20
                        spin += 1

                # `:143-144` -- `bool update_result = split_reg.update({...})`.
                var moved = cur.update(acc, sab, tie_salt)
                if sab == SPLIT_SAB_INVERT_GUARD:
                    cur = acc
                    moved = not moved

                # `:145-150` -- write back the fields, only if it moved.
                if moved:
                    out_quesval[unsafe_offset=slot] = cur.split.quesval
                    out_colid[unsafe_offset=slot] = cur.split.colid
                    out_metric[unsafe_offset=slot] = cur.split.best_metric_val
                    out_nleft[unsafe_offset=slot] = cur.split.n_left
                    out_num[unsafe_offset=slot] = cur.key.num
                    out_den[unsafe_offset=slot] = cur.key.den
                    out_valid[unsafe_offset=slot] = cur.key.valid

                # OURS, DEVIATION 169(c). Inside the critical section, so the
                # counts are exact on every arm that holds the lock.
                out_n_merges[unsafe_offset=slot] = (
                    out_n_merges[unsafe_offset=slot] + Int32(1)
                )
                if n_contrib > out_n_warps[unsafe_offset=slot]:
                    out_n_warps[unsafe_offset=slot] = n_contrib

                # `:151` -- `__threadfence(); atomicExch(mutex, 0);` folded
                # into one RELEASE store, which orders every write above it
                # before the handback.
                if sab != SPLIT_SAB_NO_LOCK:
                    Atomic.store[ordering = Ordering.RELEASE](
                        mutexes.unsafe_offset(slot), Int32(0)
                    )


def split_tie_count_kernel(
    out_ties: MutPointer[Int32, MutAnyOrigin],
    win_colid: MutPointer[Int32, MutAnyOrigin],
    win_num: MutPointer[Int64, MutAnyOrigin],
    win_den: MutPointer[Int64, MutAnyOrigin],
    win_valid: MutPointer[Int32, MutAnyOrigin],
    cand_num: MutPointer[Int64, MutAnyOrigin],
    cand_den: MutPointer[Int64, MutAnyOrigin],
    cand_valid: MutPointer[Int32, MutAnyOrigin],
    node_begin: MutPointer[Int32, MutAnyOrigin],
    node_count: MutPointer[Int32, MutAnyOrigin],
    len: Int32,
):
    """DEVIATION 463's exact-tie counter, per node, AFTER the reduce.

    `out_ties[nid]` = how many VALID candidates in the node's cell range
    exactly tie the reduced winner's key under `compare_exact_key` (the
    winner itself included), or 0 when the node has no valid winner. A value
    `>= 2` means the tie-break arm -- not the key -- decided the node, so
    `count(out_ties >= 2) / count(winner valid)` is the exact-tie RATE the
    covtype audit pre-registered as high at depth 16. Order-independent (a
    property of the candidate set), and the same quantity the host oracle
    reports as `HostSplitResult.n_tied_best`. No cuML counterpart; launched
    only under `-D MOJOLEARN_ET_TIE_STATS=1` (the builder's gate), so the
    shipping fit pays nothing for it.
    """
    var nid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while nid < Int(len):
        var ties = Int32(0)
        if win_colid[unsafe_offset=nid] >= 0:
            var wkey = ExactKey(
                win_num[unsafe_offset=nid],
                win_den[unsafe_offset=nid],
                win_valid[unsafe_offset=nid],
            )
            var b = Int(node_begin[unsafe_offset=nid])
            var e = b + Int(node_count[unsafe_offset=nid])
            for i in range(b, e):
                if cand_valid[unsafe_offset=i] == 0:
                    continue
                var ck = ExactKey(
                    cand_num[unsafe_offset=i],
                    cand_den[unsafe_offset=i],
                    cand_valid[unsafe_offset=i],
                )
                if compare_exact_key(ck, wkey) == Int32(0):
                    ties += 1
        out_ties[unsafe_offset=nid] = ties
        nid += stride
