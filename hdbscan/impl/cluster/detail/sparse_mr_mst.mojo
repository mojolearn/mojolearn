# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The mutual reachability MST without the m x m graph, on the device.

DEVIATION 1620 (`hdbscan/impl/detail/sparse_mr.mojo` has the block): past
`PAIRWISE_MAX_ROWS` the dense graph cannot exist, so Boruvka runs with
every edge weight computed when it is read. The answer is the dense arm's
tree edge for edge and bit for bit, because both minimize under the one
total order (weight key, lo, hi) and every weight is the dense cell's
arithmetic (`mr_edge_weight`).

THE ROUND. For each point `i` of a listed set, `sparse_mr_search_kernel`
finds the cheapest edge to a point of ANOTHER component: one thread per
(point, slice of the j axis), j ascending, the minimum taken on the
INTEGER weight key with a STRICT `<` so a tie keeps the lower j. For a
fixed `i` the order (key, lo, hi) among equal keys IS ascending j (j < i
reads (j, i), j > i reads (i, j), and every j < i sorts first), so the
per-point minimum is the per-point minimum of the total order. The slices
fold in ascending slice order under the same (key, j) comparison, each
component's minimum is taken under the full triple, and components join by
hooking and pointer jumping, all on the device (THE ROUND ON THE DEVICE,
below). No float reduction; the only atomics are integer minimums over a
total order, which have one answer whatever order they land in.

THE PRUNING (the plan of `hierarchy/.../fast_boruvka.mojo`, exact here).
A point whose previous best edge still leaves its component keeps that
edge: the candidate set only shrinks and still holds it. A point whose
best joined its own component is LISTED, with its previous key as a lower
bound. A listed point whose lower bound is STRICTLY above its component's
best known key cannot supply the component's edge and sits the round out
(a tie is still searched). Phase A searches each component's listed point
with the smallest bound, phase B the rest after A's exact keys tighten the
bound. None of this changes which edge a component takes.

APPLE LAUNCH BOUND. macOS silently aborts a Metal command buffer that
holds the GPU for seconds (the output stays partly stale and
`synchronize` reports nothing). Every search launch is bounded to
`SPARSE_MR_LAUNCH_MACS` multiply-adds and drained, its output is POISONED
before the launch and every cell is folded on the device after it; a
poisoned cell that survives is refused by name. The same bound runs on every column so the arithmetic
and the launches are one program.

NON-FINITE WEIGHTS. Round 1 searches every point against every other, so
every edge weight is computed at least once; a NaN or infinite weight is
reported by its point and refused by name (DEVIATION 1607's rule).
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_NVIDIA, COLUMN_AMD
from core.column_stats import CUDA_MAX_GRID_YZ, TRANSPOSE_TILE, transpose_kernel
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add
from core.fast_radix_sort import fast_radix_sort_pairs_u32, frs_counts_len
from core.row_norms import NORM_TPB, row_norm_kernel
from hdbscan.checks.hdbscan_sabotage import HDB_SAB_NONE
from hdbscan.impl.detail.sparse_mr import mr_edge_weight
from hdbscan.impl.detail.fast_apple import HDB_SMR_TILED
from hierarchy.checks.edge_order import (
    WEIGHT_KEY_SENTINEL,
    weight_order_key,
    weight_order_unkey,
)


comptime SPARSE_MR_TPB = 128
"""SCHEDULING: one thread per (point, slice); nothing folds across threads."""

comptime SPARSE_MR_LAUNCH_MACS = 1 << 30
"""The most multiply-adds one search launch may issue (the Apple bound)."""

comptime SPARSE_MR_TARGET_THREADS = 1 << 16
"""Slices per launch are chosen so a launch has about this many threads."""

comptime SMR_NONE_J: Int32 = -1
comptime SMR_BAD_J: Int32 = -2
comptime SMR_POISON_J: Int32 = -3
comptime SMR_KEY_MIN: Int32 = -0x7FFFFFFF - 1


def sparse_mr_search_kernel(
    out_key: MutPointer[Int32, MutAnyOrigin],
    out_j: MutPointer[Int32, MutAnyOrigin],
    xt: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    norms: MutPointer[Float32, MutAnyOrigin],
    core: MutPointer[Float32, MutAnyOrigin],
    comp: MutPointer[Int32, MutAnyOrigin],
    todo: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    d_in: Int32,
    n_todo_in: Int32,
    j0_in: Int32,
    j1_in: Int32,
    slice_in: Int32,
    inv_alpha: Float32,
    sabotage: Int32,
):
    """Cell `(s, t)`: the cheapest edge from listed point `todo[t]` to a
    point of another component among `j` in slice `s` of `[j0, j1)`.

    `xt` is X feature-major (`xt[f * m + i]`, adjacent threads read
    adjacent words), `x` row-major (every thread of a block reads the same
    `x[j * d + f]`). The chain is `pinned_distance_tile_kernel`'s cell
    (row i, col j): `fma(ftz(x_i[f]), ftz(x_j[f]), acc)`, f ascending.
    Every cell is written, a non-finite weight as `SMR_BAD_J`."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n_todo = Int(n_todo_in)
    if t >= n_todo:
        return
    var m = Int(m_in)
    var d = Int(d_in)
    var s = Int(block_idx.y)
    var ja = Int(j0_in) + s * Int(slice_in)
    var jb = ja + Int(slice_in)
    if jb > Int(j1_in):
        jb = Int(j1_in)
    var i = Int(todo[t])
    var ci = comp[i]
    var ni = norms[i]
    var cri = core[i]
    var bk = WEIGHT_KEY_SENTINEL
    var bj = SMR_NONE_J
    var bad = False
    for j in range(ja, jb):
        if comp[j] == ci:
            continue
        var acc = Float32(0.0)
        for f in range(d):
            acc = ftz(identical_mul_add(ftz(xt[f * m + i]), ftz(x[j * d + f]), acc))
        var v = mr_edge_weight(acc, ni, norms[j], cri, core[j], inv_alpha, sabotage)
        if (bitcast[DType.uint32](v) & 0x7F800000) == 0x7F800000:
            bad = True
            continue
        var key = weight_order_key(v)
        if key < bk:
            bk = key
            bj = Int32(j)
    var cell = s * n_todo + t
    out_key[cell] = bk
    out_j[cell] = SMR_BAD_J if bad else bj


# ===========================================================================
# lane/gap-nv-classical2: THE TILED SEARCH ON NVIDIA AND AMD
# ===========================================================================
#
# `sparse_mr_search_kernel` reads row i's d words from global memory once per
# candidate j (no reuse), and the Apple launch bound (2^30 multiply-adds)
# cut istella's first round (m = 100k, d = 220) into ~2,000 launches, each
# read back whole and folded on the host. Off Apple the search is
# `sparse_mr_search_tiled_kernel`: a block owns SMR_TI listed points and
# walks its slice of j in SMR_TJ tiles; both sides are staged SMR_KC
# features at a time from `xt` (feature-major, coalesced on both sides) and
# every thread keeps SMR_RI x SMR_RJ cells in registers. Each cell is the
# SAME chain, `ftz(identical_mul_add(ftz(x_i[f]), ftz(x_j[f]), acc))` with f
# ascending from 0.0, then `mr_edge_weight`; the per-point minimum is over
# the total order (key, j), so the merge order of threads, slices and
# launches cannot move it. `smr_fold_slices_kernel` folds the slices on the
# device and only n_todo pairs come back. No launch bound off Apple. -D
# MOJOLEARN_SMR_TILED_OFF=1 restores the per-pair kernel and the bound.
comptime SMR_TILED = (
    (TARGET_COLUMN == COLUMN_NVIDIA or TARGET_COLUMN == COLUMN_AMD)
    and not is_defined["MOJOLEARN_SMR_TILED_OFF"]()
) or HDB_SMR_TILED
#: lane af-hdbscan2 (2026-10-03), FAST on Apple only, -D MOJOLEARN_HDB_SMR_TILED:
#: the tiled kernel above runs on Apple too, under a launch bound of
#: SMR_APPLE_TILED_MACS multiply-adds (16x the per-pair kernel's: the tiled
#: kernel issues 16 FMAs per 8 threadgroup reads, so a launch of this size is
#: milliseconds, far under the macOS command-buffer cut) and a drain every
#: SMR_APPLE_DRAIN_EVERY launches instead of after every launch. The poison
#: and the device fold stay, so a cut launch is still refused by name. istella
#: round 1: ~2,084 drained 48-column launches become ~128 launches, 16 drains.
#: IDENTICAL and every build off Apple compile main's code unchanged.
comptime SMR_APPLE_TILED_MACS = 1 << 34
comptime SMR_APPLE_DRAIN_EVERY = 8
comptime SMR_TI = 64
comptime SMR_TJ = 64
comptime SMR_KC = 16
comptime SMR_TX = 16
comptime SMR_TY = 16
comptime SMR_RI = SMR_TI // SMR_TY
comptime SMR_RJ = SMR_TJ // SMR_TX
comptime SMR_TILED_TPB = SMR_TX * SMR_TY
comptime SMR_TARGET_BLOCKS = 1024


@always_inline
def _smr_better(ka: Int32, ja: Int32, kb: Int32, jb: Int32) -> Bool:
    """(ka, ja) strictly before (kb, jb) in the total order (key, j); an
    empty slot (jb < 0) loses to every real candidate."""
    if ja < 0:
        return False
    if jb < 0:
        return True
    if ka != kb:
        return ka < kb
    return ja < jb


def sparse_mr_search_tiled_kernel(
    out_key: MutPointer[Int32, MutAnyOrigin],
    out_j: MutPointer[Int32, MutAnyOrigin],
    xt: MutPointer[Float32, MutAnyOrigin],
    norms: MutPointer[Float32, MutAnyOrigin],
    core: MutPointer[Float32, MutAnyOrigin],
    comp: MutPointer[Int32, MutAnyOrigin],
    todo: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    d_in: Int32,
    n_todo_in: Int32,
    j0_in: Int32,
    j1_in: Int32,
    slice_in: Int32,
    inv_alpha: Float32,
    sabotage: Int32,
):
    """Cells `(s, t)` for the SMR_TI listed points of this block: the same
    answer as `sparse_mr_search_kernel`'s cell, every cell written."""
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * SMR_TX + tx
    var m = Int(m_in)
    var d = Int(d_in)
    var n_todo = Int(n_todo_in)
    var t0 = Int(block_idx.x) * SMR_TI
    var s = Int(block_idx.y)
    var ja = Int(j0_in) + s * Int(slice_in)
    var jb = ja + Int(slice_in)
    if jb > Int(j1_in):
        jb = Int(j1_in)
    var a_s = stack_allocation[
        SMR_KC * SMR_TI, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var b_s = stack_allocation[
        SMR_KC * SMR_TJ, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var ti_s = stack_allocation[
        SMR_TI, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var cj_s = stack_allocation[
        SMR_TJ, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var nj_s = stack_allocation[
        SMR_TJ, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var crj_s = stack_allocation[
        SMR_TJ, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var rk_s = stack_allocation[
        SMR_TI * SMR_TX, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var rj_s = stack_allocation[
        SMR_TI * SMR_TX, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    if tid < SMR_TI:
        var t = t0 + tid
        ti_s[unsafe_offset=tid] = todo[t] if t < n_todo else Int32(-1)
    barrier()
    var ci = InlineArray[Int32, SMR_RI](fill=Int32(-1))
    var ni = InlineArray[Float32, SMR_RI](fill=Float32(0.0))
    var cri = InlineArray[Float32, SMR_RI](fill=Float32(0.0))
    var bk = InlineArray[Int32, SMR_RI](fill=WEIGHT_KEY_SENTINEL)
    var bj = InlineArray[Int32, SMR_RI](fill=SMR_NONE_J)
    var bad = InlineArray[Bool, SMR_RI](fill=False)
    comptime for r in range(SMR_RI):
        var i = Int(ti_s[unsafe_offset = ty + r * SMR_TY])
        if i >= 0:
            ci[r] = comp[i]
            ni[r] = norms[i]
            cri[r] = core[i]
    var jt = ja
    while jt < jb:
        if tid < SMR_TJ:
            var j = jt + tid
            if j < jb:
                cj_s[unsafe_offset=tid] = comp[j]
                nj_s[unsafe_offset=tid] = norms[j]
                crj_s[unsafe_offset=tid] = core[j]
            else:
                cj_s[unsafe_offset=tid] = Int32(-1)
        var acc = InlineArray[Float32, SMR_RI * SMR_RJ](fill=Float32(0.0))
        var k0 = 0
        while k0 < d:
            comptime for q in range(SMR_KC * SMR_TI // SMR_TILED_TPB):
                var e = tid + q * SMR_TILED_TPB
                var kk = e // SMR_TI
                var ii = e - kk * SMR_TI
                var f = k0 + kk
                var i = Int(ti_s[unsafe_offset=ii])
                var v = Float32(0.0)
                if f < d and i >= 0:
                    v = ftz(xt[f * m + i])
                a_s[unsafe_offset=e] = v
            comptime for q in range(SMR_KC * SMR_TJ // SMR_TILED_TPB):
                var e = tid + q * SMR_TILED_TPB
                var kk = e // SMR_TJ
                var jj = e - kk * SMR_TJ
                var f = k0 + kk
                var j = jt + jj
                var v = Float32(0.0)
                if f < d and j < jb:
                    v = ftz(xt[f * m + j])
                b_s[unsafe_offset=e] = v
            barrier()
            var kmax = d - k0
            if kmax > SMR_KC:
                kmax = SMR_KC
            for kk in range(kmax):
                var av = InlineArray[Float32, SMR_RI](fill=Float32(0.0))
                var bv = InlineArray[Float32, SMR_RJ](fill=Float32(0.0))
                comptime for r in range(SMR_RI):
                    av[r] = a_s[unsafe_offset = kk * SMR_TI + ty + r * SMR_TY]
                comptime for c in range(SMR_RJ):
                    bv[c] = b_s[unsafe_offset = kk * SMR_TJ + tx + c * SMR_TX]
                comptime for r in range(SMR_RI):
                    comptime for c in range(SMR_RJ):
                        acc[r * SMR_RJ + c] = ftz(
                            identical_mul_add(av[r], bv[c], acc[r * SMR_RJ + c])
                        )
            barrier()
            k0 += SMR_KC
        comptime for c in range(SMR_RJ):
            var jj = tx + c * SMR_TX
            var j = jt + jj
            var cj = cj_s[unsafe_offset=jj]
            if j < jb:
                comptime for r in range(SMR_RI):
                    if ci[r] >= 0 and cj != ci[r]:
                        var v = mr_edge_weight(
                            acc[r * SMR_RJ + c], ni[r], nj_s[unsafe_offset=jj],
                            cri[r], crj_s[unsafe_offset=jj], inv_alpha, sabotage,
                        )
                        if (bitcast[DType.uint32](v) & 0x7F800000) == 0x7F800000:
                            bad[r] = True
                        else:
                            var key = weight_order_key(v)
                            if _smr_better(key, Int32(j), bk[r], bj[r]):
                                bk[r] = key
                                bj[r] = Int32(j)
        barrier()
        jt += SMR_TJ
    comptime for r in range(SMR_RI):
        var ii = ty + r * SMR_TY
        rk_s[unsafe_offset = ii * SMR_TX + tx] = bk[r]
        rj_s[unsafe_offset = ii * SMR_TX + tx] = SMR_BAD_J if bad[r] else bj[r]
    barrier()
    if tid < SMR_TI:
        var t = t0 + tid
        if t < n_todo:
            var fk = WEIGHT_KEY_SENTINEL
            var fj = SMR_NONE_J
            var fbad = False
            for q in range(SMR_TX):
                var k = rk_s[unsafe_offset = tid * SMR_TX + q]
                var j = rj_s[unsafe_offset = tid * SMR_TX + q]
                if j == SMR_BAD_J:
                    fbad = True
                elif _smr_better(k, j, fk, fj):
                    fk = k
                    fj = j
            var cell = s * n_todo + t
            out_key[cell] = fk
            out_j[cell] = SMR_BAD_J if fbad else fj


def smr_fold_slices_kernel(
    fold_key: MutPointer[Int32, MutAnyOrigin],
    fold_j: MutPointer[Int32, MutAnyOrigin],
    out_key: MutPointer[Int32, MutAnyOrigin],
    out_j: MutPointer[Int32, MutAnyOrigin],
    n_todo_in: Int32,
    n_s_in: Int32,
):
    """Point `t`'s slices in ascending order under (key, j): the host fold of
    `_search`, on the device. A poisoned or non-finite cell is passed on as
    its marker (the first one in slice order) for the host to refuse."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n_todo = Int(n_todo_in)
    if t >= n_todo:
        return
    var fk = WEIGHT_KEY_SENTINEL
    var fj = SMR_NONE_J
    for s in range(Int(n_s_in)):
        var c = s * n_todo + t
        var jj = out_j[c]
        if jj == SMR_POISON_J or jj == SMR_BAD_J:
            fj = jj
            break
        if jj < 0:
            continue
        var kk = out_key[c]
        if _smr_better(kk, jj, fk, fj):
            fk = kk
            fj = jj
    fold_key[t] = fk
    fold_j[t] = fj


@fieldwise_init
struct SparseMst(Movable):
    """The m - 1 tree edges sorted by (weight key, lo, hi), oriented
    (lo, hi), and the dense solver's round count. A CHECK's view:
    `sparse_mr_mst` reads the device result back for the comparison
    driver; the fit calls `sparse_mr_mst_device` and nothing comes back."""

    var lo: List[Int32]
    var hi: List[Int32]
    var w: List[Float32]
    var rounds: Int


# ===========================================================================
# lane cgr3-hdbscan-mst (2026-10-03): THE ROUND ON THE DEVICE
# ===========================================================================
#
# Before, every round downloaded the per-point results, built the listed
# sets, took each component's minimum, joined components with a union-find
# and relabeled on the host, then uploaded the labels. Now every step is a
# parallel kernel over the points or the components, and the host reads
# back three status words per round (the list sizes that size the search
# grids, the join count, an error code):
#
#   classify    a point whose best edge still leaves its component is
#               EXACT and lowers the component's bound `ub` (integer
#               atomic min: one answer whatever order they land in); the
#               rest are LISTED with their previous key as a lower bound.
#   phase A     a listed point strictly above its component's bound sits
#               out; of the rest, each component's point with the smallest
#               (bound, index) is searched first (two integer atomic mins).
#   phase B     A's exact keys lower the bounds; the deferred points still
#               at or under the bound are searched.
#   compact     the searched lists come from a flag scan: per-block counts,
#               an exclusive scan of the block counts, a stable in-block
#               scatter (ascending point order, as the host lists were).
#   component   each component's cheapest edge under (key, lo, hi): three
#               integer atomic-min phases (key, then lo among the key's
#               holders, then hi), `hierarchy/.../mst_kernels.mojo`'s
#               DEVIATION 620 pattern. The one point holding the triple
#               names the target component.
#   hook        component c hooks to its target d. The only cycle a unique
#               minimum edge can make is the mutual pair (c -> d -> c, the
#               same edge); the lower label stays the root and the edge is
#               recorded once, at the hooked component's slot. Each vertex
#               is a hooked root at most once, so slot c is never reused.
#   jump        pointer jumping, ceil(log2 m) + 1 synchronous ping-pong
#               passes (depth < m), then every point takes its root.
#
# The tree is THE minimum spanning tree under the total order (key, lo, hi)
# and the edges leave sorted by that order (a rank sort on the triple), so
# the output does not depend on the labels, the launch shapes or which
# atomic lands first: the same edges and bits as before on every column.
# The host column (`hdbscan/host/hdbscan_host_oracle.mojo::hdbh_sparse_prim`)
# returns the same sorted list and is unchanged.
#
# SMALL HOST STEPS THAT REMAIN (scalars): the three status words per round
# (list sizes for the launch grids, the join count for termination and the
# "joined nothing" refusal, the poison / non-finite error code with its
# row). `smr_scan_blocks_kernel` is one block over the ceil(m / 256) block
# counts (the second level of the scan, k = m / 256), not over the points.

comptime SMR_DEV_TPB = 256
comptime SMR_INT_MAX: Int32 = 0x7FFFFFFF
comptime SMR_ST_ERR = 0
comptime SMR_ST_ERR_ROW = 1
comptime SMR_ST_COUNT = 2
comptime SMR_ST_ADDED = 3
comptime SMR_ST_LEN = 4
comptime SMR_ERR_BAD: Int32 = 1
comptime SMR_ERR_POISON: Int32 = 2

comptime SMR_EXACT: Int32 = 0
comptime SMR_LIST_A: Int32 = 1
comptime SMR_LIST_B: Int32 = 2
comptime SMR_DROPPED: Int32 = 3

comptime _I32P = MutPointer[Int32, MutAnyOrigin]
comptime _F32P = MutPointer[Float32, MutAnyOrigin]


@always_inline
def _gid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _grid(n: Int) -> Int:
    return (n + SMR_DEV_TPB - 1) // SMR_DEV_TPB if n > 0 else 1


@always_inline
def _triple_less(
    ka: Int32, la: Int32, ha: Int32, kb: Int32, lb: Int32, hb: Int32
) -> Bool:
    if ka != kb:
        return ka < kb
    if la != lb:
        return la < lb
    return ha < hb


def smr_init_kernel(
    comp: _I32P, pk: _I32P, pj: _I32P, e_key: _I32P, e_lo: _I32P,
    e_hi: _I32P, st: _I32P, m_in: Int32,
):
    """Every point its own component, no best edge, every edge slot empty."""
    var i = _gid()
    if i < SMR_ST_LEN:
        st[i] = SMR_INT_MAX if i == SMR_ST_ERR_ROW else Int32(0)
    if i >= Int(m_in):
        return
    comp[i] = Int32(i)
    pk[i] = WEIGHT_KEY_SENTINEL
    pj[i] = SMR_NONE_J
    e_key[i] = WEIGHT_KEY_SENTINEL
    e_lo[i] = SMR_INT_MAX
    e_hi[i] = SMR_INT_MAX


def smr_round_reset_kernel(
    ub: _I32P, amin: _I32P, aidx: _I32P, ckey: _I32P, clo: _I32P,
    chi: _I32P, nxt: _I32P, win: _I32P, st: _I32P, m_in: Int32,
):
    var c = _gid()
    if c == 0:
        st[SMR_ST_ADDED] = 0
    if c >= Int(m_in):
        return
    ub[c] = WEIGHT_KEY_SENTINEL
    amin[c] = WEIGHT_KEY_SENTINEL
    aidx[c] = SMR_INT_MAX
    ckey[c] = WEIGHT_KEY_SENTINEL
    clo[c] = SMR_INT_MAX
    chi[c] = SMR_INT_MAX
    nxt[c] = -1
    win[c] = -1


def smr_classify_kernel(
    comp: _I32P, pk: _I32P, pj: _I32P, ub: _I32P, lb: _I32P, state: _I32P,
    m_in: Int32,
):
    """EXACT points set the component bound; the rest are LISTED with a
    lower bound (their previous key, or the floor before any search)."""
    var i = _gid()
    if i >= Int(m_in):
        return
    var ci = comp[i]
    var j = pj[i]
    if j >= 0 and comp[Int(j)] != ci:
        state[i] = SMR_EXACT
        _ = Atomic.min(ub.unsafe_offset(Int(ci)), pk[i])
    else:
        state[i] = SMR_LIST_A
        lb[i] = pk[i] if j >= 0 else SMR_KEY_MIN


def smr_drop_arg_kernel(
    comp: _I32P, lb: _I32P, ub: _I32P, state: _I32P, amin: _I32P,
    m_in: Int32,
):
    """A listed point strictly above its component's bound sits out; the
    others publish their bound for the component's (bound, index) minimum."""
    var i = _gid()
    if i >= Int(m_in) or state[i] != SMR_LIST_A:
        return
    var ci = Int(comp[i])
    if lb[i] > ub[ci]:
        state[i] = SMR_DROPPED
    else:
        _ = Atomic.min(amin.unsafe_offset(ci), lb[i])


def smr_arg_idx_kernel(
    comp: _I32P, lb: _I32P, state: _I32P, amin: _I32P, aidx: _I32P,
    m_in: Int32,
):
    var i = _gid()
    if i >= Int(m_in) or state[i] != SMR_LIST_A:
        return
    var ci = Int(comp[i])
    if lb[i] == amin[ci]:
        _ = Atomic.min(aidx.unsafe_offset(ci), Int32(i))


def smr_assign_kernel(comp: _I32P, state: _I32P, aidx: _I32P, m_in: Int32):
    """Phase A: each component's (bound, index)-first listed point; every
    other surviving listed point is deferred to phase B."""
    var i = _gid()
    if i >= Int(m_in) or state[i] != SMR_LIST_A:
        return
    if aidx[Int(comp[i])] != Int32(i):
        state[i] = SMR_LIST_B


def smr_ub_from_a_kernel(
    comp: _I32P, pk: _I32P, pj: _I32P, state: _I32P, ub: _I32P, m_in: Int32
):
    var i = _gid()
    if i >= Int(m_in) or state[i] != SMR_LIST_A or pj[i] < 0:
        return
    _ = Atomic.min(ub.unsafe_offset(Int(comp[i])), pk[i])


def smr_drop_b_kernel(
    comp: _I32P, lb: _I32P, ub: _I32P, state: _I32P, m_in: Int32
):
    var i = _gid()
    if i >= Int(m_in) or state[i] != SMR_LIST_B:
        return
    if lb[i] > ub[Int(comp[i])]:
        state[i] = SMR_DROPPED


def smr_flag_count_kernel(
    state: _I32P, want: Int32, bcount: _I32P, m_in: Int32
):
    """bcount[b] = the number of points of block b whose state is `want`
    (an integer tree in threadgroup memory)."""
    var sh = stack_allocation[
        SMR_DEV_TPB, Int32, address_space = AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    var i = _gid()
    sh[tid] = Int32(1) if (i < Int(m_in) and state[i] == want) else Int32(0)
    barrier()
    var s = SMR_DEV_TPB // 2
    while s > 0:
        if tid < s:
            sh[tid] = sh[tid] + sh[tid + s]
        barrier()
        s //= 2
    if tid == 0:
        bcount[Int(block_idx.x)] = sh[0]


def smr_scan_blocks_kernel(
    bcount: _I32P, boff: _I32P, nb_in: Int32, st: _I32P
):
    """Exclusive scan of the `nb` block counts, ONE block: thread t owns a
    contiguous chunk, the chunk sums are scanned in threadgroup memory,
    then each chunk is written from its base. The total goes to
    st[SMR_ST_COUNT]. Integers only."""
    var sh = stack_allocation[
        SMR_DEV_TPB, Int32, address_space = AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    var nb = Int(nb_in)
    var per = (nb + SMR_DEV_TPB - 1) // SMR_DEV_TPB
    var a = tid * per
    var b = min(nb, a + per)
    var sum = Int32(0)
    for q in range(a, b):
        sum += bcount[q]
    sh[tid] = sum
    barrier()
    var step = 1
    while step < SMR_DEV_TPB:
        var add = Int32(0)
        if tid >= step:
            add = sh[tid - step]
        barrier()
        sh[tid] = sh[tid] + add
        barrier()
        step *= 2
    var run = sh[tid] - sum
    for q in range(a, b):
        boff[q] = run
        run += bcount[q]
    if tid == SMR_DEV_TPB - 1:
        st[SMR_ST_COUNT] = sh[tid]


def smr_flag_scatter_kernel(
    state: _I32P, want: Int32, boff: _I32P, todo: _I32P, m_in: Int32
):
    """todo[boff[b] + rank in block] = i for every point whose state is
    `want`: a stable compaction (ascending i)."""
    var sh = stack_allocation[
        SMR_DEV_TPB, Int32, address_space = AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    var i = _gid()
    var take = Int32(1) if (i < Int(m_in) and state[i] == want) else Int32(0)
    sh[tid] = take
    barrier()
    var step = 1
    while step < SMR_DEV_TPB:
        var add = Int32(0)
        if tid >= step:
            add = sh[tid - step]
        barrier()
        sh[tid] = sh[tid] + add
        barrier()
        step *= 2
    if take != 0:
        todo[Int(boff[Int(block_idx.x)]) + Int(sh[tid]) - 1] = Int32(i)


def smr_merge_kernel(
    pk: _I32P, pj: _I32P, fk: _I32P, fj: _I32P, todo: _I32P, st: _I32P,
    n_todo_in: Int32, first: Int32,
):
    """One launch's folded (key, j) per listed point into that point's best
    under (key, j), strict; the first launch of a search starts from empty.
    A surviving poison or a non-finite weight is reported in `st`."""
    var t = _gid()
    if t >= Int(n_todo_in):
        return
    var i = Int(todo[t])
    if first != 0:
        pk[i] = WEIGHT_KEY_SENTINEL
        pj[i] = SMR_NONE_J
    var jj = fj[t]
    if jj == SMR_POISON_J:
        _ = Atomic.max(st.unsafe_offset(SMR_ST_ERR), SMR_ERR_POISON)
        return
    if jj == SMR_BAD_J:
        _ = Atomic.max(st.unsafe_offset(SMR_ST_ERR), SMR_ERR_BAD)
        _ = Atomic.min(st.unsafe_offset(SMR_ST_ERR_ROW), Int32(i))
        return
    if jj < 0:
        return
    var kk = fk[t]
    if pj[i] < 0 or kk < pk[i] or (kk == pk[i] and jj < pj[i]):
        pk[i] = kk
        pj[i] = jj


def smr_cmin_key_kernel(
    comp: _I32P, pk: _I32P, pj: _I32P, state: _I32P, ckey: _I32P,
    m_in: Int32,
):
    var i = _gid()
    if i >= Int(m_in) or state[i] == SMR_DROPPED or pj[i] < 0:
        return
    _ = Atomic.min(ckey.unsafe_offset(Int(comp[i])), pk[i])


def smr_cmin_lo_kernel(
    comp: _I32P, pk: _I32P, pj: _I32P, state: _I32P, ckey: _I32P,
    clo: _I32P, m_in: Int32,
):
    var i = _gid()
    if i >= Int(m_in) or state[i] == SMR_DROPPED or pj[i] < 0:
        return
    var ci = Int(comp[i])
    if pk[i] == ckey[ci]:
        _ = Atomic.min(clo.unsafe_offset(ci), min(Int32(i), pj[i]))


def smr_cmin_hi_kernel(
    comp: _I32P, pk: _I32P, pj: _I32P, state: _I32P, ckey: _I32P,
    clo: _I32P, chi: _I32P, m_in: Int32,
):
    var i = _gid()
    if i >= Int(m_in) or state[i] == SMR_DROPPED or pj[i] < 0:
        return
    var ci = Int(comp[i])
    if pk[i] == ckey[ci] and min(Int32(i), pj[i]) == clo[ci]:
        _ = Atomic.min(chi.unsafe_offset(ci), max(Int32(i), pj[i]))


def smr_winner_kernel(
    comp: _I32P, pk: _I32P, pj: _I32P, state: _I32P, ckey: _I32P,
    clo: _I32P, chi: _I32P, nxt: _I32P, win: _I32P, m_in: Int32,
):
    """The one point of the component holding its minimum triple (an edge
    has one endpoint inside) names the target component."""
    var i = _gid()
    if i >= Int(m_in) or state[i] == SMR_DROPPED or pj[i] < 0:
        return
    var ci = Int(comp[i])
    var j = pj[i]
    if (
        pk[i] == ckey[ci]
        and min(Int32(i), j) == clo[ci]
        and max(Int32(i), j) == chi[ci]
    ):
        nxt[ci] = comp[Int(j)]
        win[ci] = Int32(i)


def smr_hook_kernel(
    comp: _I32P, pk: _I32P, pj: _I32P, nxt: _I32P, win: _I32P, par: _I32P,
    e_key: _I32P, e_lo: _I32P, e_hi: _I32P, st: _I32P, m_in: Int32,
):
    """Root c hooks to its target; of a mutual pair the lower label stays
    the root. The hooked root records its edge in slot c."""
    var c = _gid()
    if c >= Int(m_in) or comp[c] != Int32(c):
        return
    var d = nxt[c]
    if d < 0 or (nxt[Int(d)] == Int32(c) and Int32(c) < d):
        par[c] = Int32(c)
        return
    par[c] = d
    var i = Int(win[c])
    var j = pj[i]
    e_key[c] = pk[i]
    e_lo[c] = min(Int32(i), j)
    e_hi[c] = max(Int32(i), j)
    _ = Atomic.fetch_add(st.unsafe_offset(SMR_ST_ADDED), Int32(1))


def smr_jump_kernel(comp: _I32P, src: _I32P, dst: _I32P, m_in: Int32):
    var c = _gid()
    if c >= Int(m_in) or comp[c] != Int32(c):
        return
    dst[c] = src[Int(src[c])]


def smr_relabel_kernel(comp: _I32P, par: _I32P, m_in: Int32):
    var v = _gid()
    if v >= Int(m_in):
        return
    comp[v] = par[Int(comp[v])]


def smr_edge_rank_kernel(
    e_key: _I32P, e_lo: _I32P, e_hi: _I32P, rank: _I32P, m_in: Int32
):
    """rank[s] = the number of slots whose (key, lo, hi) is below slot s's
    (triples are distinct; the one empty slot sorts last), the slots read a
    tile at a time through threadgroup memory."""
    var tk = stack_allocation[SMR_DEV_TPB, Int32, address_space = AddressSpace.SHARED]()
    var tl = stack_allocation[SMR_DEV_TPB, Int32, address_space = AddressSpace.SHARED]()
    var th = stack_allocation[SMR_DEV_TPB, Int32, address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var s = _gid()
    var m = Int(m_in)
    var mk = WEIGHT_KEY_SENTINEL
    var ml = SMR_INT_MAX
    var mh = SMR_INT_MAX
    if s < m:
        mk = e_key[s]
        ml = e_lo[s]
        mh = e_hi[s]
    var r = 0
    var t0 = 0
    while t0 < m:
        var q = t0 + tid
        if q < m:
            tk[tid] = e_key[q]
            tl[tid] = e_lo[q]
            th[tid] = e_hi[q]
        barrier()
        var lim = min(SMR_DEV_TPB, m - t0)
        for u in range(lim):
            if _triple_less(tk[u], tl[u], th[u], mk, ml, mh):
                r += 1
        barrier()
        t0 += SMR_DEV_TPB
    if s < m:
        rank[s] = Int32(r)


#: fam-cluster (2026-10-04), IDENTICAL: the final order of the m edge slots
#: comes from three stable radix sorts (hi, then lo, then key; `core/
#: fast_radix_sort.mojo`) instead of `smr_edge_rank_kernel`'s all-pairs
#: count, which is m^2 triple compares (10^12 at a million points). The
#: triples are distinct, so the stable lexicographic order IS the rank the
#: count produces: same permutation, same MST arrays.
#: `-D MOJOLEARN_IDN_SMR_RADIX_RANK_OFF=1` restores the count.
comptime IDN_SMR_RADIX_RANK = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (
        is_defined["MOJOLEARN_IDN_SMR_RADIX_RANK_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)
comptime _U32P = MutPointer[UInt32, MutAnyOrigin]


def smr_rank_key_kernel(
    src: _I32P, idx: _U32P, keys: _U32P, m_in: Int32, seed: Int32
):
    """keys[p] = the unsigned image of `src` at the slot position p holds
    (signed order preserved by flipping the sign bit). `seed != 0` starts
    the permutation at the identity; otherwise it is read from `idx`."""
    var p = _gid()
    if p >= Int(m_in):
        return
    var s = p
    if Int(seed) != 0:
        idx[p] = UInt32(p)
    else:
        s = Int(idx[p])
    # a same-width integer cast keeps the bits (fam2-cluster: was a
    # `bitcast` of the loaded scalar)
    keys[p] = src[s].cast[DType.uint32]() ^ UInt32(0x80000000)


def smr_rank_from_perm_kernel(idx: _U32P, rank: _I32P, m_in: Int32):
    """rank[slot] = its position in the sorted order."""
    var p = _gid()
    if p < Int(m_in):
        rank[Int(idx[p])] = Int32(p)


def smr_edge_scatter_kernel(
    e_key: _I32P, e_lo: _I32P, e_hi: _I32P, rank: _I32P, rows: _I32P,
    cols: _I32P, weights: _F32P, m_in: Int32,
):
    var s = _gid()
    if s >= Int(m_in):
        return
    var p = Int(rank[s])
    if p >= Int(m_in) - 1:
        return
    rows[p] = e_lo[s]
    cols[p] = e_hi[s]
    weights[p] = weight_order_unkey(e_key[s])


@fieldwise_init
struct _Search(Movable):
    var key_d: DeviceBuffer[DType.int32]
    var j_d: DeviceBuffer[DType.int32]
    var todo_d: DeviceBuffer[DType.int32]
    var cap: Int
    var fk_d: DeviceBuffer[DType.int32]
    var fj_d: DeviceBuffer[DType.int32]


def _search(
    ctx: DeviceContext,
    mut sb: _Search,
    mut xt_d: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut norms_d: DeviceBuffer[DType.float32],
    mut core_d: DeviceBuffer[DType.float32],
    mut comp_d: DeviceBuffer[DType.int32],
    mut pk_d: DeviceBuffer[DType.int32],
    mut pj_d: DeviceBuffer[DType.int32],
    mut st_d: DeviceBuffer[DType.int32],
    n_todo: Int,
    m: Int,
    d: Int,
    inv_alpha: Float32,
    sabotage: Int32,
    launch_macs: Int,
) raises -> Int:
    """Exact cheapest other-component edge of every listed point
    (`sb.todo_d[0:n_todo]`, on the device), merged into `pk_d` / `pj_d` on
    the device. Each launch's output is POISONED first, its slices folded
    in ascending order under (key, j) (`smr_fold_slices_kernel`) and merged
    (`smr_merge_kernel`); a surviving poison or a non-finite weight lands
    in `st_d`. Off Apple the tiled kernel and no launch bound (an explicit
    smaller one, the checks', still splits the j axis, which may move no
    bit); on Apple the per-pair kernel under `SPARSE_MR_LAUNCH_MACS` with
    a drain after every launch so no command buffer holds the GPU long.
    Returns the number of launches."""
    if n_todo == 0:
        return 0
    var dd = d if d > 0 else 1
    var span = m
    comptime if HDB_SMR_TILED:
        span = max(SMR_TJ, min(m, SMR_APPLE_TILED_MACS // (n_todo * dd)))
        if launch_macs < SPARSE_MR_LAUNCH_MACS:
            span = max(1, launch_macs // (n_todo * dd))
    else:
        comptime if SMR_TILED:
            if launch_macs < SPARSE_MR_LAUNCH_MACS:
                span = max(1, launch_macs // (n_todo * dd))
        else:
            span = max(1, launch_macs // (n_todo * dd))
    var i_tiles = (n_todo + SMR_TI - 1) // SMR_TI
    var want_s = (SPARSE_MR_TARGET_THREADS + n_todo - 1) // n_todo
    var tgrid = (n_todo + SPARSE_MR_TPB - 1) // SPARSE_MR_TPB
    var launches = 0
    var j0 = 0
    while j0 < m:
        var j1 = min(m, j0 + span)
        var width = j1 - j0
        var n_s: Int
        var sl: Int
        comptime if SMR_TILED:
            n_s = (SMR_TARGET_BLOCKS + i_tiles - 1) // i_tiles
            n_s = min(n_s, (width + SMR_TJ - 1) // SMR_TJ)
            if n_s * n_todo > sb.cap:
                n_s = sb.cap // n_todo
            n_s = max(1, min(n_s, CUDA_MAX_GRID_YZ))
            sl = (width + n_s - 1) // n_s
            sl = ((sl + SMR_TJ - 1) // SMR_TJ) * SMR_TJ
            n_s = (width + sl - 1) // sl
        else:
            n_s = min(width, max(1, want_s))
            if n_s * n_todo > sb.cap:
                n_s = max(1, sb.cap // n_todo)
            sl = (width + n_s - 1) // n_s
            n_s = (width + sl - 1) // sl
        var cells = n_s * n_todo
        var vj = sb.j_d.create_sub_buffer[DType.int32](0, cells)
        ctx.enqueue_memset(vj, SMR_POISON_J)
        comptime if SMR_TILED:
            ctx.enqueue_function[sparse_mr_search_tiled_kernel](
                sb.key_d.unsafe_ptr(), sb.j_d.unsafe_ptr(), xt_d.unsafe_ptr(),
                norms_d.unsafe_ptr(), core_d.unsafe_ptr(), comp_d.unsafe_ptr(),
                sb.todo_d.unsafe_ptr(), Int32(m), Int32(d), Int32(n_todo),
                Int32(j0), Int32(j1), Int32(sl), inv_alpha, sabotage,
                grid_dim=(i_tiles, n_s, 1), block_dim=(SMR_TX, SMR_TY, 1),
            )
        else:
            ctx.enqueue_function[sparse_mr_search_kernel](
                sb.key_d.unsafe_ptr(), sb.j_d.unsafe_ptr(),
                xt_d.unsafe_ptr(), x.unsafe_ptr(), norms_d.unsafe_ptr(),
                core_d.unsafe_ptr(), comp_d.unsafe_ptr(), sb.todo_d.unsafe_ptr(),
                Int32(m), Int32(d), Int32(n_todo), Int32(j0), Int32(j1),
                Int32(sl), inv_alpha, sabotage,
                grid_dim=(tgrid, n_s, 1),
                block_dim=(SPARSE_MR_TPB, 1, 1),
            )
        ctx.enqueue_function[smr_fold_slices_kernel](
            sb.fk_d.unsafe_ptr(), sb.fj_d.unsafe_ptr(), sb.key_d.unsafe_ptr(),
            sb.j_d.unsafe_ptr(), Int32(n_todo), Int32(n_s),
            grid_dim=(tgrid, 1, 1), block_dim=(SPARSE_MR_TPB, 1, 1),
        )
        ctx.enqueue_function[smr_merge_kernel](
            pk_d.unsafe_ptr(), pj_d.unsafe_ptr(), sb.fk_d.unsafe_ptr(),
            sb.fj_d.unsafe_ptr(), sb.todo_d.unsafe_ptr(), st_d.unsafe_ptr(),
            Int32(n_todo), Int32(1) if launches == 0 else Int32(0),
            grid_dim=(tgrid, 1, 1), block_dim=(SPARSE_MR_TPB, 1, 1),
        )
        comptime if not SMR_TILED:
            ctx.synchronize()
        launches += 1
        comptime if HDB_SMR_TILED:
            if launches % SMR_APPLE_DRAIN_EVERY == 0:
                ctx.synchronize()
        _ = vj^
        j0 = j1
    return launches


def _compact(
    ctx: DeviceContext,
    mut state_d: DeviceBuffer[DType.int32],
    want: Int32,
    mut bcount_d: DeviceBuffer[DType.int32],
    mut boff_d: DeviceBuffer[DType.int32],
    mut todo_d: DeviceBuffer[DType.int32],
    mut st_d: DeviceBuffer[DType.int32],
    m: Int,
) raises:
    """The points whose state is `want`, ascending, into `todo_d`; the
    count into st[SMR_ST_COUNT]. Enqueued only; the caller drains."""
    var g = _grid(m)
    ctx.enqueue_function[smr_flag_count_kernel](
        state_d.unsafe_ptr(), want, bcount_d.unsafe_ptr(), Int32(m),
        grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
    )
    ctx.enqueue_function[smr_scan_blocks_kernel](
        bcount_d.unsafe_ptr(), boff_d.unsafe_ptr(), Int32(g),
        st_d.unsafe_ptr(),
        grid_dim=(1, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
    )
    ctx.enqueue_function[smr_flag_scatter_kernel](
        state_d.unsafe_ptr(), want, boff_d.unsafe_ptr(), todo_d.unsafe_ptr(),
        Int32(m),
        grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
    )


def _read_status(
    ctx: DeviceContext,
    mut st_d: DeviceBuffer[DType.int32],
    mut st_h: HostBuffer[DType.int32],
) raises:
    """Drain, read the status words, refuse a poison or a non-finite weight
    by name."""
    ctx.enqueue_copy(dst_ptr=st_h.unsafe_ptr(), src_buf=st_d)
    ctx.synchronize()
    var err = st_h.unsafe_ptr().unsafe_load(SMR_ST_ERR)
    if err == SMR_ERR_POISON:
        raise Error(
            "hdbscan.sparse_mr_mst: a search launch left a cell unwritten"
            " (the poison survived); the launch was cut short (on Apple,"
            " macOS aborts a long command buffer without reporting it)."
            " Refused by name"
        )
    if err == SMR_ERR_BAD:
        raise Error(
            "hdbscan.build_mr_linkage: a mutual reachability weight from row "
            + String(Int(st_h.unsafe_ptr().unsafe_load(SMR_ST_ERR_ROW)))
            + " is NaN or infinite (a non-finite input row, or rows whose"
            " squared difference overflows Float32); refused by name"
            " (DEVIATION 623 / 1607, IDENTITY_PATHS row 39)"
        )


def sparse_mr_mst_device(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut core_d: DeviceBuffer[DType.float32],
    m: Int,
    d: Int,
    inv_alpha: Float32,
    mut mst_rows: DeviceBuffer[DType.int32],
    mut mst_cols: DeviceBuffer[DType.int32],
    mut mst_weights: DeviceBuffer[DType.float32],
    sabotage: Int32 = HDB_SAB_NONE,
    launch_macs: Int = SPARSE_MR_LAUNCH_MACS,
) raises -> Int:
    """The dense arm's `build_sorted_mst` result on the mutual reachability
    graph, with no m x m array, written to `mst_rows` / `mst_cols` /
    `mst_weights` (the first m - 1 cells): edges sorted by (weight key, lo,
    hi), oriented (lo, hi), weights bit for bit. Returns the round count.
    `launch_macs` bounds one launch's work; the check lowers it so small
    inputs take many launches and slices, which may move no bit."""
    if m < 2:
        raise Error("hdbscan.sparse_mr_mst: m=" + String(m) + " < 2")
    # `pairwise_distances`'s norms: the same kernel, the same launch.
    var norms_d = ctx.enqueue_create_buffer[DType.float32](m)
    ctx.enqueue_function[row_norm_kernel](
        norms_d.unsafe_ptr(), x.unsafe_ptr(), Int32(d), Int32(0),
        grid_dim=(m, 1, 1), block_dim=(NORM_TPB, 1, 1),
    )
    # the feature-major copy on the device (moves words, no arithmetic)
    var xt_d = ctx.enqueue_create_buffer[DType.float32](m * d)
    ctx.enqueue_function[transpose_kernel](
        xt_d.unsafe_ptr(), x.unsafe_ptr(), Int32(m), Int32(d),
        grid_dim=(
            (d + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE,
            min((m + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE, CUDA_MAX_GRID_YZ),
            1,
        ),
        block_dim=(TRANSPOSE_TILE, TRANSPOSE_TILE, 1),
    )
    var cap = m + SPARSE_MR_TARGET_THREADS
    var sb = _Search(
        ctx.enqueue_create_buffer[DType.int32](cap),
        ctx.enqueue_create_buffer[DType.int32](cap),
        ctx.enqueue_create_buffer[DType.int32](m),
        cap,
        ctx.enqueue_create_buffer[DType.int32](m),
        ctx.enqueue_create_buffer[DType.int32](m),
    )
    var comp_d = ctx.enqueue_create_buffer[DType.int32](m)
    var pk_d = ctx.enqueue_create_buffer[DType.int32](m)
    var pj_d = ctx.enqueue_create_buffer[DType.int32](m)
    var lb_d = ctx.enqueue_create_buffer[DType.int32](m)
    var state_d = ctx.enqueue_create_buffer[DType.int32](m)
    var ub_d = ctx.enqueue_create_buffer[DType.int32](m)
    var amin_d = ctx.enqueue_create_buffer[DType.int32](m)
    var aidx_d = ctx.enqueue_create_buffer[DType.int32](m)
    var ckey_d = ctx.enqueue_create_buffer[DType.int32](m)
    var clo_d = ctx.enqueue_create_buffer[DType.int32](m)
    var chi_d = ctx.enqueue_create_buffer[DType.int32](m)
    var nxt_d = ctx.enqueue_create_buffer[DType.int32](m)
    var win_d = ctx.enqueue_create_buffer[DType.int32](m)
    var par_a = ctx.enqueue_create_buffer[DType.int32](m)
    var par_b = ctx.enqueue_create_buffer[DType.int32](m)
    var e_key = ctx.enqueue_create_buffer[DType.int32](m)
    var e_lo = ctx.enqueue_create_buffer[DType.int32](m)
    var e_hi = ctx.enqueue_create_buffer[DType.int32](m)
    var g = _grid(m)
    var bcount_d = ctx.enqueue_create_buffer[DType.int32](g)
    var boff_d = ctx.enqueue_create_buffer[DType.int32](g)
    var st_d = ctx.enqueue_create_buffer[DType.int32](SMR_ST_LEN)
    var st_h = ctx.enqueue_create_host_buffer[DType.int32](SMR_ST_LEN)
    ctx.enqueue_function[smr_init_kernel](
        comp_d.unsafe_ptr(), pk_d.unsafe_ptr(), pj_d.unsafe_ptr(),
        e_key.unsafe_ptr(), e_lo.unsafe_ptr(), e_hi.unsafe_ptr(),
        st_d.unsafe_ptr(), Int32(m),
        grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
    )
    var jumps = 1
    while (1 << jumps) < m:
        jumps += 1
    jumps += 1

    var n_comp = m
    var merge_rounds = 0
    while n_comp > 1:
        merge_rounds += 1
        if merge_rounds > 64:
            raise Error("hdbscan.sparse_mr_mst: Boruvka did not converge")
        ctx.enqueue_function[smr_round_reset_kernel](
            ub_d.unsafe_ptr(), amin_d.unsafe_ptr(), aidx_d.unsafe_ptr(),
            ckey_d.unsafe_ptr(), clo_d.unsafe_ptr(), chi_d.unsafe_ptr(),
            nxt_d.unsafe_ptr(), win_d.unsafe_ptr(), st_d.unsafe_ptr(),
            Int32(m), grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        ctx.enqueue_function[smr_classify_kernel](
            comp_d.unsafe_ptr(), pk_d.unsafe_ptr(), pj_d.unsafe_ptr(),
            ub_d.unsafe_ptr(), lb_d.unsafe_ptr(), state_d.unsafe_ptr(),
            Int32(m), grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        ctx.enqueue_function[smr_drop_arg_kernel](
            comp_d.unsafe_ptr(), lb_d.unsafe_ptr(), ub_d.unsafe_ptr(),
            state_d.unsafe_ptr(), amin_d.unsafe_ptr(), Int32(m),
            grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        ctx.enqueue_function[smr_arg_idx_kernel](
            comp_d.unsafe_ptr(), lb_d.unsafe_ptr(), state_d.unsafe_ptr(),
            amin_d.unsafe_ptr(), aidx_d.unsafe_ptr(), Int32(m),
            grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        ctx.enqueue_function[smr_assign_kernel](
            comp_d.unsafe_ptr(), state_d.unsafe_ptr(), aidx_d.unsafe_ptr(),
            Int32(m), grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        _compact(ctx, state_d, SMR_LIST_A, bcount_d, boff_d, sb.todo_d, st_d, m)
        _read_status(ctx, st_d, st_h)
        var n_a = Int(st_h.unsafe_ptr().unsafe_load(SMR_ST_COUNT))
        _ = _search(
            ctx, sb, xt_d, x, norms_d, core_d, comp_d, pk_d, pj_d, st_d,
            n_a, m, d, inv_alpha, sabotage, launch_macs,
        )
        # Phase B: A's exact keys tighten the bounds.
        ctx.enqueue_function[smr_ub_from_a_kernel](
            comp_d.unsafe_ptr(), pk_d.unsafe_ptr(), pj_d.unsafe_ptr(),
            state_d.unsafe_ptr(), ub_d.unsafe_ptr(), Int32(m),
            grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        ctx.enqueue_function[smr_drop_b_kernel](
            comp_d.unsafe_ptr(), lb_d.unsafe_ptr(), ub_d.unsafe_ptr(),
            state_d.unsafe_ptr(), Int32(m),
            grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        _compact(ctx, state_d, SMR_LIST_B, bcount_d, boff_d, sb.todo_d, st_d, m)
        _read_status(ctx, st_d, st_h)
        var n_b = Int(st_h.unsafe_ptr().unsafe_load(SMR_ST_COUNT))
        _ = _search(
            ctx, sb, xt_d, x, norms_d, core_d, comp_d, pk_d, pj_d, st_d,
            n_b, m, d, inv_alpha, sabotage, launch_macs,
        )
        # Each component's cheapest edge under (key, lo, hi).
        ctx.enqueue_function[smr_cmin_key_kernel](
            comp_d.unsafe_ptr(), pk_d.unsafe_ptr(), pj_d.unsafe_ptr(),
            state_d.unsafe_ptr(), ckey_d.unsafe_ptr(), Int32(m),
            grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        ctx.enqueue_function[smr_cmin_lo_kernel](
            comp_d.unsafe_ptr(), pk_d.unsafe_ptr(), pj_d.unsafe_ptr(),
            state_d.unsafe_ptr(), ckey_d.unsafe_ptr(), clo_d.unsafe_ptr(),
            Int32(m), grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        ctx.enqueue_function[smr_cmin_hi_kernel](
            comp_d.unsafe_ptr(), pk_d.unsafe_ptr(), pj_d.unsafe_ptr(),
            state_d.unsafe_ptr(), ckey_d.unsafe_ptr(), clo_d.unsafe_ptr(),
            chi_d.unsafe_ptr(), Int32(m),
            grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        ctx.enqueue_function[smr_winner_kernel](
            comp_d.unsafe_ptr(), pk_d.unsafe_ptr(), pj_d.unsafe_ptr(),
            state_d.unsafe_ptr(), ckey_d.unsafe_ptr(), clo_d.unsafe_ptr(),
            chi_d.unsafe_ptr(), nxt_d.unsafe_ptr(), win_d.unsafe_ptr(),
            Int32(m), grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        ctx.enqueue_function[smr_hook_kernel](
            comp_d.unsafe_ptr(), pk_d.unsafe_ptr(), pj_d.unsafe_ptr(),
            nxt_d.unsafe_ptr(), win_d.unsafe_ptr(), par_a.unsafe_ptr(),
            e_key.unsafe_ptr(), e_lo.unsafe_ptr(), e_hi.unsafe_ptr(),
            st_d.unsafe_ptr(), Int32(m),
            grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        for q in range(jumps):
            if q % 2 == 0:
                ctx.enqueue_function[smr_jump_kernel](
                    comp_d.unsafe_ptr(), par_a.unsafe_ptr(),
                    par_b.unsafe_ptr(), Int32(m),
                    grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
                )
            else:
                ctx.enqueue_function[smr_jump_kernel](
                    comp_d.unsafe_ptr(), par_b.unsafe_ptr(),
                    par_a.unsafe_ptr(), Int32(m),
                    grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
                )
        if jumps % 2 == 1:
            ctx.enqueue_function[smr_relabel_kernel](
                comp_d.unsafe_ptr(), par_b.unsafe_ptr(), Int32(m),
                grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
            )
        else:
            ctx.enqueue_function[smr_relabel_kernel](
                comp_d.unsafe_ptr(), par_a.unsafe_ptr(), Int32(m),
                grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
            )
        _read_status(ctx, st_d, st_h)
        var added = Int(st_h.unsafe_ptr().unsafe_load(SMR_ST_ADDED))
        if added == 0:
            raise Error(
                "hdbscan.sparse_mr_mst: a Boruvka round joined nothing with "
                + String(n_comp) + " components left"
            )
        n_comp -= added

    # The m - 1 recorded slots (plus the one empty slot, last) by rank.
    var rank_d = ctx.enqueue_create_buffer[DType.int32](m)
    comptime if IDN_SMR_RADIX_RANK:
        var rk_keys = ctx.enqueue_create_buffer[DType.uint32](m)
        var rk_idx = ctx.enqueue_create_buffer[DType.uint32](m)
        var rk_tk = ctx.enqueue_create_buffer[DType.uint32](m)
        var rk_tv = ctx.enqueue_create_buffer[DType.uint32](m)
        var rk_counts = ctx.enqueue_create_buffer[DType.int32](
            frs_counts_len(m)
        )
        # least significant component first; each sort is stable
        ctx.enqueue_function[smr_rank_key_kernel](
            e_hi.unsafe_ptr(), rk_idx.unsafe_ptr(), rk_keys.unsafe_ptr(),
            Int32(m), Int32(1),
            grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        fast_radix_sort_pairs_u32(
            ctx, m, rk_keys, rk_idx, rk_tk, rk_tv, rk_counts
        )
        ctx.enqueue_function[smr_rank_key_kernel](
            e_lo.unsafe_ptr(), rk_idx.unsafe_ptr(), rk_keys.unsafe_ptr(),
            Int32(m), Int32(0),
            grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        fast_radix_sort_pairs_u32(
            ctx, m, rk_keys, rk_idx, rk_tk, rk_tv, rk_counts
        )
        ctx.enqueue_function[smr_rank_key_kernel](
            e_key.unsafe_ptr(), rk_idx.unsafe_ptr(), rk_keys.unsafe_ptr(),
            Int32(m), Int32(0),
            grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        fast_radix_sort_pairs_u32(
            ctx, m, rk_keys, rk_idx, rk_tk, rk_tv, rk_counts
        )
        ctx.enqueue_function[smr_rank_from_perm_kernel](
            rk_idx.unsafe_ptr(), rank_d.unsafe_ptr(), Int32(m),
            grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
        # the launches hold raw pointers into the sort's scratch
        ctx.synchronize()
        _ = rk_keys^
        _ = rk_idx^
        _ = rk_tk^
        _ = rk_tv^
        _ = rk_counts^
    else:
        ctx.enqueue_function[smr_edge_rank_kernel](
            e_key.unsafe_ptr(), e_lo.unsafe_ptr(), e_hi.unsafe_ptr(),
            rank_d.unsafe_ptr(), Int32(m),
            grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
        )
    ctx.enqueue_function[smr_edge_scatter_kernel](
        e_key.unsafe_ptr(), e_lo.unsafe_ptr(), e_hi.unsafe_ptr(),
        rank_d.unsafe_ptr(), mst_rows.unsafe_ptr(), mst_cols.unsafe_ptr(),
        mst_weights.unsafe_ptr(), Int32(m),
        grid_dim=(g, 1, 1), block_dim=(SMR_DEV_TPB, 1, 1),
    )
    ctx.synchronize()
    _ = norms_d^
    _ = xt_d^
    _ = sb^
    _ = comp_d^
    _ = pk_d^
    _ = pj_d^
    _ = lb_d^
    _ = state_d^
    _ = ub_d^
    _ = amin_d^
    _ = aidx_d^
    _ = ckey_d^
    _ = clo_d^
    _ = chi_d^
    _ = nxt_d^
    _ = win_d^
    _ = par_a^
    _ = par_b^
    _ = e_key^
    _ = e_lo^
    _ = e_hi^
    _ = bcount_d^
    _ = boff_d^
    _ = st_d^
    _ = st_h^
    _ = rank_d^
    # The dense solver's count: the merge rounds plus its final round.
    return merge_rounds + 1


def sparse_mr_mst(
    ctx: DeviceContext,
    x_host: List[Float32],
    mut x: DeviceBuffer[DType.float32],
    mut core_d: DeviceBuffer[DType.float32],
    m: Int,
    d: Int,
    inv_alpha: Float32,
    sabotage: Int32 = HDB_SAB_NONE,
    launch_macs: Int = SPARSE_MR_LAUNCH_MACS,
) raises -> SparseMst:
    """`sparse_mr_mst_device` read back for a CHECK (the comparison driver
    compares host lists). `x_host` is not read; the fit never calls this."""
    var rows = ctx.enqueue_create_buffer[DType.int32](m - 1)
    var cols = ctx.enqueue_create_buffer[DType.int32](m - 1)
    var wts = ctx.enqueue_create_buffer[DType.float32](m - 1)
    var rounds = sparse_mr_mst_device(
        ctx, x, core_d, m, d, inv_alpha, rows, cols, wts, sabotage,
        launch_macs,
    )
    var h_r = ctx.enqueue_create_host_buffer[DType.int32](m - 1)
    var h_c = ctx.enqueue_create_host_buffer[DType.int32](m - 1)
    var h_w = ctx.enqueue_create_host_buffer[DType.float32](m - 1)
    ctx.enqueue_copy(dst_ptr=h_r.unsafe_ptr(), src_buf=rows)
    ctx.enqueue_copy(dst_ptr=h_c.unsafe_ptr(), src_buf=cols)
    ctx.enqueue_copy(dst_ptr=h_w.unsafe_ptr(), src_buf=wts)
    ctx.synchronize()
    var lo = List[Int32](capacity=m - 1)
    var hi = List[Int32](capacity=m - 1)
    var w = List[Float32](capacity=m - 1)
    for e in range(m - 1):
        lo.append(h_r.unsafe_ptr().unsafe_load(e))
        hi.append(h_c.unsafe_ptr().unsafe_load(e))
        w.append(h_w.unsafe_ptr().unsafe_load(e))
    _ = rows^
    _ = cols^
    _ = wts^
    _ = h_r^
    _ = h_c^
    _ = h_w^
    return SparseMst(lo^, hi^, w^, rounds)
