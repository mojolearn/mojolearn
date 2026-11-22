# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LARGE SEQUENCE PRODUCTS ON THE NEURAL PACKED BODY, ONE LEAF
(lane fx-gemm-seq-packed, 2026-11-18; idea 1 of the GEMM headroom review).

THE DEFAULT on the NVIDIA and AMD columns since 2026-11-19 (the experiment
`MOJOLEARN_GEMM_FAST_SEQ_PACKED`, promoted; A/B numbers beside
`SEQ_GEMM_PACKED` below; docs/apple-fast/EXPERIMENTS.md). On the Apple
column nothing here is compiled into a launch.

WHAT MOVES. The 16 x 16 sequence tile (sequence/gemm_tiled.mojo) gives every
thread ONE cell and reads two threadgroup words per fused multiply-add, so it
runs at the threadgroup-memory rate at every size. The neural packed body
(`identical_gemm_kpack_kernel`) gives every thread a register tile of 16 or 32
cells and reads 0.375 to 0.5 words per multiply-add. Under the arm a product
the rule below selects is launched on that body with `leaf = K` and ONE leaf.

WHY THE WORDS ARE THE SAME (the claim the A/B's hash check tests). Write
A(m, k) = p0[m i3 + k i4], B(k, n) = p1[k i5 + n i6], `ftz` the flush of a
subnormal to its signed zero, `fma` the fused multiply-add with one rounding
to nearest.

  The sequence chain (`gemm_dot`, the tile kernels, the host's op_gemm), cell
  (m, n), no accumulate:
      acc_0   = +0.0
      acc_k+1 = ftz(fma(ftz(A(m, k)), ftz(B(k, n)), acc_k))     k = 0, 1, ..., K - 1
      C[m, n] = ftz(acc_K)                                      (`st`; acc_K is already flushed)
  One chain a cell, k ascending, every term folded exactly once, nothing
  padded (the slab width only cuts the k range into consecutive pieces).

  The packed body with (leaf, P) = (K, 1), cell (i, j):
      the one leaf is [0, min(K, K)) = [0, K)                   (`_leaf_bounds`)
      it is walked in ceil(K / 16) windows, window w covering
      [16 w, min(16 w + 16, K)), windows ascending, steps ascending in a
      window (`_tuned_window_at`; a full window is an unrolled loop, the last
      short window a counted loop over its `chunk` live terms, so no zero
      slot is folded)
      acc_0   = +0.0
      acc_k+1 = step(ftz(A(i, k)), ftz(B(k, j)), acc_k)         operands flushed at the gather
      part    = ftz(acc_K)                                      the leaf partial, pushed once
      root    = part                                            `_fold_drain_local`: no addition at P = 1
      C[i, j] = ftz(root) = ftz(ftz(acc_K)) = ftz(acc_K)
  `step` is the contract step `ftz(fma(a, b, acc))` in each of its spellings:
    - NVIDIA `_tuned_step`: `fma.rn`, then the hardware flush by one. The
      sequence step on NVIDIA is `ftz(fma(...))` with `ftz` spelled as the
      same hardware flush by one: the same two instructions.
    - NVIDIA, an ADMITTED window: the bare `fma.rn`, taken only where the
      staged exponents prove every step result of the window is zero or
      normal, i.e. where the flush is the identity. The proof reads the
      window's operands and that the accumulators entering it are multiples
      of 2^-126 (true at the leaf start, kept by admitted windows); it does
      not read the leaf length, so one leaf of K terms changes nothing in it.
    - AMD `_tuned_step`: `fma`, then the class flush; the sequence `ftz` on
      AMD is the same class test and select.
    - AMD, the DETECT seam: bare packed FMAs and a per-thread flag set by any
      subnormal step result; a flagged thread recomputes each of its cells
      with `_rtf_cell`, the exact chain above over the same (leaf, P).
  So both bodies fold the same K flushed operand pairs into the same seed in
  the same ascending order with the same rounding at every step: the same
  word, cell for cell. Rows and columns past the matrix are staged as zeros
  by the packed gather and never stored. The tile, the thread that owns a
  cell and the window width are scheduling.
  NaN: both bodies return the column's arithmetic NaN for a NaN cell; a
  payload is outside the identity contract at every GEMM seam.

THE EPILOGUE. The packed body stores the plain product. An epilogue opcode
(OP_GEMM_EPI, OP_AGK_GEMM_EPI) then runs its follower as its own launch,
OP_GEMM_EPI_TAIL: `gemm_epi_tail` on the stored word, the launch pair
`HostExec` already runs for both opcodes (sequence/exec.mojo). The follower
reloads `ld(C) = ftz(ftz(acc)) = acc`, so its operands and operations are the
tile's in-register epilogue's (sequence/mlp.mojo, "the tape's in-register
epilogue").

NOT ROUTED (today's route kept): the accumulate form (i7 != 0: the chain's
seed is C, the packed body's is +0.0), a C that is not dense (i8 != N), the
column-sum launches (i11 != 0), every product the split-K rule or the coop
rule takes first (they are tested before this rule and are their own bit
versions), Apple (its sequence GEMM is one thread a cell), the host column
(`HostExec._gemm`, unchanged: the chain above).

THE RULE (`seq_gemm_packed_shape`): sizes and hardware only.
  1. K >= 4 windows of 16 terms (64). A packed block pays a gather round
     trip and a barrier per 16-term window and a prologue per launch; under
     four windows the staging is not amortized over the chain.
  2. Both edges of C hold at least one whole wide packed tile (128 rows; 64
     columns on NVIDIA, 128 on AMD: the larger of the two, so one rule), so
     no launched block is mostly dead cells.
  3. The packed tiles FILL THE DEVICE: ceil(M / BM) ceil(N / BN) is at least
     the column's block floor (`GEMM_LEAF_PTR_MIN_BLOCKS`: 192 on NVIDIA, 1024
     on AMD), first for the wide tile, then (NVIDIA) for the stepped-down
     64-row tile, the same two steps the neural dispatch takes
     (`gemm_nv_step_kpack_rows`). A product with fewer tiles than SMs runs a
     few 256-thread blocks on a mostly idle device, and with one leaf there
     is no split over k to make more blocks (that would be another order), so
     it keeps the 16 x 16 tile, whose blocks are 64 times as many.
  What that selects, NVIDIA (floor 192; wide tile 128 x 64, small 64 x 64):
     4096 x 256, 2048 x 512, 1024 x 1024 (256 small tiles each), 3072 x 512
     (192 wide tiles), 8192 x 256 (256 wide tiles), any K >= 64.
  What stays on the tile: 2048 x 256 and 1024 x 512 (128 small tiles),
     every long-K small output such as 256 x 256 x 4096 (16 small tiles: the
     weight gradients of a wide layer), K < 64, an edge under 128.
  AMD (floor 1024, wide tile 128 x 128 only): from about 16.7 M cells, e.g.
     16384 x 1024; smaller outputs keep the tile until the AMD leg measures
     this body there.
"""
from std.sys import is_defined
from max.gpu.host import DeviceContext

from core.accelerator_target import TARGET_HAS_APPLE_GPU
from sequence.ops import Args, OP_GEMM, OP_GEMM_EPI
from autograd.ops import OP_AGK_GEMM_EPI
from gemm.neural_dispatch import (
    GEMM_LEAF_PTR_COLUMN, GEMM_LEAF_PTR_SMALL_ON,
    GEMM_LEAF_PTR_RPT_WIDE, GEMM_LEAF_PTR_RPT_SMALL,
    GEMM_LEAF_PTR_FS_WIDE, GEMM_LEAF_PTR_FS_SMALL,
    GEMM_LEAF_PTR_BM_WIDE, GEMM_LEAF_PTR_BM_SMALL, GEMM_LEAF_PTR_BN,
    GEMM_LEAF_PTR_KS, GEMM_LEAF_PTR_MIN_BLOCKS,
    identical_gemm_leaf_ptr_into,
)

#: The route switch. IDENTICAL on the NVIDIA and AMD columns (`GEMM_LEAF_PTR_COLUMN`);
#: never Apple, whose sequence GEMM is not tiled (SEQ_GEMM_TILED).
# MOJOLEARN_GEMM_FAST_SEQ_PACKED: PROMOTED 2026-11-19 (lane fx-verdicts3). The packed route is the default;
# (the old opt-in name and its _OFF are refused, core/experiment_guards_neural.mojo). Evidence: A/B round 1, NVIDIA L40S (host
# bd9c355c19c0, 14-core cgroup), wheel 0c68d6cd27 and its B set, base and arm back to back in ONE pod job
# (nvc1-0006, tag nvc-ab-20261119T125554Z-6f7afc), one scored run per arm; digests and quality equal on all 8
# cells. Base -> arm ms: mlp-clf istella fit 1462.4 -> 868.3 (0.59), infer 52.80 -> 28.63 (0.54); mlp-clf taxi
# fit 1013.1 -> 672.9 (0.66), infer 15.91 (one call) -> 7.97 (median of 20; warm-up calls 16.4 -> 9.9, 0.60);
# mlp-reg istella fit 1463.8 -> 857.0 (0.59), infer 25.28 -> 17.09 (0.68); mlp-reg taxi fit 1004.3 -> 660.1
# (0.66), infer 6.42 -> 2.77 (0.43, median of 20 both). Geomean 0.58 over 8 cells; no cell slower. Same words
# (the module docstring's order argument), so no column, oracle or verify record moves.
# The rule is sizes and the column's tile and block floor only (`seq_gemm_packed_shape`): no shape or dataset key.
# Limits: NVIDIA only (AGENTS.md part 8); the AMD column takes the route only from 1024 wide tiles (about 16.7 M
# cells), not timed. OWED (does not hold the flip): the driver row gemm-seq-packed-shapes (neighbour
# 4096 x 256 x 128, non-board 2048 x 512 x 512 and 8192 x 192 x 96, control 1024 x 256) against the _OFF arm.
# A/B round 3, 2026-11-20 (lane fx-verdicts5; NVIDIA L40S host 2ef132a6de04, wheel 28a780604c and its _OFF set, one pod
# job, tag nvc-ab-20261120T063324Z-c630ad), that driver row, MLPClassifier.fit, packed (default) -> 16 x 16 tile ms:
# 4096 x 256 over K 128 (neighbour) 70.99 -> 79.62 (1.12), 2048 x 512 x 512 (non-board) 303.2 -> 505.4 (1.67),
# 8192 x 192 over K 96 249.4 -> 254.2 (1.02), control batch 1024 x 256 (below the rule) 89.58 -> 90.03 (1.005).
# CONFIRMED a second time, on shapes the board does not hold: the route stands and its switch is deleted. The 16 x 16
# tile itself stays: it serves every product the rule does not select, the Apple column and fast mode.
# TOMBSTONE: MOJOLEARN_GEMM_FAST_SEQ_PACKED_OFF (old path's switch; the promotion confirmed by A/B rounds 1 and 3, numbers above) deleted 2026-11-20 by lane/fx-verdicts5; code recoverable at e25e35dab8.
# Restore: git apply experiments/removed/MOJOLEARN_GEMM_FAST_SEQ_PACKED_OFF.patch; record in docs/TOMBSTONES.md.
comptime SEQ_GEMM_PACKED = (
    GEMM_LEAF_PTR_COLUMN
    and not TARGET_HAS_APPLE_GPU
)

#: `seq_gemm_packed_shape`'s answers: keep today's route, the wide packed
#: tile, the stepped-down packed tile.
comptime SP_NONE = 0
comptime SP_WIDE = 1
comptime SP_SMALL = 2
#: rule 1: four staged windows of the chain
comptime SP_MIN_K = 4 * GEMM_LEAF_PTR_KS
#: rule 2: one whole wide packed tile along each edge of C
comptime SP_MIN_EDGE = GEMM_LEAF_PTR_BM_WIDE if GEMM_LEAF_PTR_BM_WIDE > GEMM_LEAF_PTR_BN else GEMM_LEAF_PTR_BN
#: the launch profile's route code of a packed launch (sequence/gemm_tiled.mojo
#: GR_* are 0-5, GR_PROF_* 8-12; autograd/profile.mojo names it "packed")
comptime GR_PROF_PACKED = 13


@always_inline
def seq_gemm_packed_shape(M: Int, N: Int, K: Int) -> Int:
    """SP_* for an M x N x K product: the cost rule in the module docstring
    (sizes and the column's tile and block floor; no data shape)."""
    comptime if not SEQ_GEMM_PACKED:
        return SP_NONE
    if K < SP_MIN_K or M < SP_MIN_EDGE or N < SP_MIN_EDGE:
        return SP_NONE
    var cols = (N + GEMM_LEAF_PTR_BN - 1) // GEMM_LEAF_PTR_BN
    if ((M + GEMM_LEAF_PTR_BM_WIDE - 1) // GEMM_LEAF_PTR_BM_WIDE) * cols >= GEMM_LEAF_PTR_MIN_BLOCKS:
        return SP_WIDE
    comptime if GEMM_LEAF_PTR_SMALL_ON:
        if ((M + GEMM_LEAF_PTR_BM_SMALL - 1) // GEMM_LEAF_PTR_BM_SMALL) * cols >= GEMM_LEAF_PTR_MIN_BLOCKS:
            return SP_SMALL
    return SP_NONE


@always_inline
def seq_gemm_packed_on[OP: Int](a: Args) -> Int:
    """SP_* for one launch of OP_GEMM, OP_GEMM_EPI or OP_AGK_GEMM_EPI
    (i0 = M, i1 = N, i2 = K): the shape rule, and only the plain dense form
    (no accumulate, row stride N, no column sums). `DeviceExec._launch` asks
    after its split-K and coop tests."""
    comptime assert OP == OP_GEMM or OP == OP_GEMM_EPI or OP == OP_AGK_GEMM_EPI, "seq_gemm_packed_on: a GEMM opcode"
    comptime if not SEQ_GEMM_PACKED:
        return SP_NONE
    if a.i7 != 0 or a.i8 != a.i1 or a.i11 != 0:
        return SP_NONE
    return seq_gemm_packed_shape(a.i0, a.i1, a.i2)


def seq_gemm_packed_launch(ctx: DeviceContext, a: Args, geom: Int) raises:
    """The product of a GEMM launch `seq_gemm_packed_on` selected (geom =
    SP_WIDE or SP_SMALL), queued on the packed body with ONE leaf of K terms:
    C = p2 (dense), A(m, k) = p0[m i3 + k i4], B(k, n) = p1[k i5 + n i6],
    which are the body's `(a_si, a_sp, b_sp, b_sj)` in that order. One
    asynchronous launch, no workspace, no wait; the pointers are the
    executor's own (live to its sync). The epilogue, if any, is the caller's
    next launch."""
    comptime if SEQ_GEMM_PACKED:
        var pc = a.p2.unsafe_origin_cast[MutAnyOrigin]()
        var pa = a.p0.unsafe_origin_cast[MutAnyOrigin]()
        var pb = a.p1.unsafe_origin_cast[MutAnyOrigin]()
        comptime if GEMM_LEAF_PTR_SMALL_ON:
            if geom == SP_SMALL:
                identical_gemm_leaf_ptr_into[GEMM_LEAF_PTR_RPT_SMALL, GEMM_LEAF_PTR_FS_SMALL](
                    ctx, pc, pa, pb, a.i0, a.i1, a.i2, a.i2, 1, a.i3, a.i4, a.i5, a.i6
                )
                return
        identical_gemm_leaf_ptr_into[GEMM_LEAF_PTR_RPT_WIDE, GEMM_LEAF_PTR_FS_WIDE](
            ctx, pc, pa, pb, a.i0, a.i1, a.i2, a.i2, 1, a.i3, a.i4, a.i5, a.i6
        )


# The fused store of a follower (this route with `-D MOJOLEARN_GEMM_FAST_EPI=1`: a bias, a bias and an activation, or
# the activation derivative computed in the packed body's own store, one launch for two) was measured and deleted:
# A/B round 3, 2026-11-20 (lane fx-verdicts5; NVIDIA L40S host 2ef132a6de04, wheel 28a780604c and its B set, base and
# arm back to back in one pod job, tag nvc-ab-20261120T063324Z-c630ad, workers on the GPU's NUMA node 0; digests and
# quality equal on all 8 cells). Follower launch (default) -> fused store ms: fit mlp-clf istella 862.9 -> 850.5
# (0.986), taxi 673.9 -> 665.4 (0.987), mlp-reg istella 855.5 -> 842.2 (0.984), taxi 662.5 -> 653.4 (0.986); infer
# taxi 7.955 -> 7.722 (0.971) and 2.762 -> 2.694 (0.975), medians of 20; infer istella 24.17 -> 29.44 (1.218) and
# 15.83 -> 16.59 (1.048), one call each, and their warm-up calls say the same (29.0 -> 33.2, 17.0 -> 17.8).
# NOT PROMOTED: MIXED. The fits gain 1.3 to 1.6 % (the fit rows' bases agree to 0.6 % in this job) and the predict of
# the wide input loses 5 to 22 %; the arm's rule wants every fit and infer faster. The neighbouring and non-board
# shapes of its row were not timed. The follower launch (OP_GEMM_EPI_TAIL) is the one form.
# TOMBSTONE: MOJOLEARN_GEMM_FAST_EPI (MIXED, not promoted: A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm, digests equal on 8 cells: mlp fits 0.984-0.987, infer taxi 0.971 / 0.975, infer istella 1.218 / 1.048) SEQ_GEMM_PACKED_EPI, seq_gemm_packed_epi_kind, _seq_gemm_packed_epi_go and seq_gemm_packed_epi_launch deleted 2026-11-22 by lane/verdicts-apply; code recoverable at 5dfd4a2e67.
# Restore: git apply experiments/removed/MOJOLEARN_GEMM_FAST_EPI.patch; record in docs/TOMBSTONES.md.
