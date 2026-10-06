# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MCD_BMMA (lane/apple-fast-w2-mcd2, 2026-10-04, DEFAULT in FAST +
Apple, rollback MOJOLEARN_MCD_BMMA_OFF): ONE launch of the matrix-unit GEMM over every candidate of an
MCD phase instead of one `_launch_gemm_mma` per candidate.

Why: x_decomp/mcd_fast.mojo's MCD_BATCH_MMA default enqueues three GEMMs per
candidate per C-step (covariance, weighted Gram, Mahalanobis product), for
every candidate of the phase, active or not. Taxi phase B is 3,330
candidates x up to 31 steps x 3 = up to ~310,000 Metal launches (~10-20 us
of enqueue each, memory metal-enqueue-costs), which is the bulk of the
3.6 s fit; the matrices themselves are 11 x 11.

What is kept: the tile kernel below is `gemm/afn_apple_fast.mojo`'s
`afn_gemm_mma_kernel` at AFN_TILE_SQUARE (2 x 2 simdgroups, 4 x 4
fragments, AFN_GEMM_KB window), statement for statement, with the
candidate taken from `block_idx.z` and the three operand pointers moved by
that candidate's batch stride. The host launcher computes `tiles`,
`splits` and the K chunk exactly as `x_decomp/device.mojo::_launch_gemm_mma`
does for ONE candidate (every candidate of a phase has the same m, k, n),
so each candidate's output cell is the same matrix-unit sum over the same
K windows and the same split-K ranges as main. Split-K partials still meet
through f32 atomics (order free, as on main). A candidate whose gate word
is 0 (inactive; for the weighted Gram, pinvh did not run) launches no work:
main zeroed its operands and discarded its output (guarded publication).
"""
from std.ffi import _Global
from std.python import PythonObject
from std.atomic import Atomic
from std.sys import llvm_intrinsic
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from gemm.afn_apple_fast import AFN_GEMM_APPLE
from experiments.apple_fast.gemm.scoped_dispatch import scoped_kernel
from x_decomp.mcd_experiments import MCD_FAST_ACTIVE_COMPACT, McdCompactWorkspace, compact_candidate_count, mcd_experiment_hit

# SOURCE-READY / UNBUILT, 2026-10-04, lane/apple-fast-mcd-g1-gram-remote
# source35c712d9c: no matrix/fitted quality or timing evidence. Default OFF.
# Non-split self-Gram of any shape (no window since 2026-10-04); split plans
# keep the incumbent atomic path.
# Admission requires gated batched FP64/no-regression checks, actual MCD/EE
# fitted-state/support/rank gates, positive caller reach, then M3 A/B timing.
# Existing ordered-covariance HOLD is not waived by this separate candidate.
# See docs/apple-fast/ab/mcd-g1-gram.md and EXPERIMENTS.md (MCD_FAST_G1_GRAM).
comptime MCD_G1_GRAM = AFN_GEMM_APPLE and is_defined["MOJOLEARN_MCD_FAST_G1_GRAM"]()
comptime MCD_G1_AUDIT = AFN_GEMM_APPLE and is_defined["MOJOLEARN_MCD_FAST_G1_GRAM_AUDIT"]()
# LEGACY, default OFF: the old window admitted only d 129..256 features and
# K 128..1023 selected rows, which brackets the board (istella 220 features).
# Removed as benchmark-tuned on 2026-10-04; the window-free replacement is
# UNMEASURED.
comptime MCD_G1_LEGACY_WINDOW = is_defined["MOJOLEARN_LEGACY_NARROW_MCD_G1_GRAM"]()

struct MCDG1Audit(Defaultable, Movable):
    var counts: InlineArray[Int, 4]
    var last: InlineArray[Int, 7]
    def __init__(out self):
        self.counts = InlineArray[Int, 4](fill=0)
        self.last = InlineArray[Int, 7](fill=0)

comptime MCD_G1_STATE = _Global[StorageType=MCDG1Audit, name="MCDG1GramAuditV1", init_fn=MCDG1Audit.__init__]

def mcd_g1_count(index: Int) raises -> Int:
    if index < 0 or index >= 4:
        raise Error("MCD G1 count index out of range")
    return MCD_G1_STATE.get_or_create_ptr()[].counts[index]

def mcd_g1_last(index: Int) raises -> Int:
    if index < 0 or index >= 7:
        raise Error("MCD G1 metadata index out of range")
    return MCD_G1_STATE.get_or_create_ptr()[].last[index]

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.kernel_matrix import COLUMN_APPLE, lib_smem_page_fits_for, lib_smem_pages_for
from gemm.afn_apple_fast import AFN_GEMM_KB, _M64, _afn_gload, _afn_load_t, _afn_mma, _afn_stage
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.device import DFG_BLOCK_TARGET, DFG_MIN_SPLIT_STEPS

#: FAST Apple candidate, default OFF: `-D MOJOLEARN_MCD_ORDERED_COV`.
#: Source lane/apple-fast-mcd-ordered-quality-r3@f35a57bd4 (ported
#: 2026-10-04, lane apple-fast-rec-misc). MinCovDet / EllipticEnvelope
#: covariance (raw C-step batched Gram and final masked covariance): split-K
#: partials to disjoint storage, then one GPU thread per cell folds them in
#: split order with Neumaier compensation (no atomics). No SKIP_PINVH or
#: DEFLATE change. Known: compiled (r3); M3 mcd-ordered-direct-q-r3 quality
#: HOLD (strict per-key f64 B <= A on rel-L2 AND max-abs, no tolerance;
#: repeat_identical true, so B was deterministic). Cause: B kept main's
#: split count (2..9 at the direct shapes), so most error sat in each
#: split's fp32 MMA chain, untouched by the compensated fold. Fixed here:
#: splits of MCD_ORD_CHAIN rows (ordered_cov_shape). Quality before timing:
#: tools/mcd_ordered_pair.py direct, then the fitted cases (MCD_ORDERED_COV.md).
comptime MCD_ORDERED_COV = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_MCD_ORDERED_COV"]())

# Bound the queued candidate plane independently of features/datasets. This
# scheduling experiment preserves inactive-candidate gates and input strides;
# it changes neither support selection nor candidate count.
# NEVER RUN — PENDING VALIDATION: bounded MCD candidate batches.
comptime MCD_FAST_BOUND_BATCH = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_MCD_FAST_BOUND_BATCH"]())
comptime MCD_CANDIDATE_BATCH = 32

struct McdBatchAudit(Defaultable, Movable):
    var launches: Int
    def __init__(out self):
        self.launches = 0

comptime BATCH_STATE = _Global[StorageType=McdBatchAudit, name="McdCandidateBatchAudit", init_fn=McdBatchAudit.__init__]

def mcd_fast_batch_count() raises -> Int:
    return BATCH_STATE.get_or_create_ptr()[].launches


#: AFN_TILE_SQUARE's shape (`afn_launch_tile_aux`'s default branch).
comptime MB_SGM = 2
comptime MB_SGN = 2
comptime MB_FM = 4
comptime MB_FN = 4
comptime MB_BM = 8 * MB_FM * MB_SGM
comptime MB_BN = 8 * MB_FN * MB_SGN
comptime MB_NT = MB_SGM * MB_SGN * 32
comptime MB_ZERO_TPB = 256


def mcd_bmma_kernel[SPLIT: Bool, ORDERED: Bool = False, COMPACT: Bool = False](
    c_in: F32Ptr,
    a_in: F32Ptr,
    b_in: F32Ptr,
    gate: I32Ptr,
    gate_all: Int32,
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    a_si_in: Int32,
    a_sp_in: Int32,
    b_sp_in: Int32,
    b_sj_in: Int32,
    k_split_in: Int32,
    a_bs_in: Int32,
    b_bs_in: Int32,
    c_bs_in: Int32,
    ids: I32Ptr,
):
    """`afn_gemm_mma_kernel[2, 2, 4, 4, AFN_GEMM_KB, f32, f32, SPLIT,
    AFN_EPI_NONE]` for candidate `block_idx.z`: C_z (+)= A_z . B_z with
    A_z = a + z * a_bs, B_z = b + z * b_bs, C_z = c + z * c_bs. Skipped
    (whole block, before any barrier) when gate[z] == 0 and not gate_all."""
    comptime SGM = MB_SGM
    comptime SGN = MB_SGN
    comptime FM = MB_FM
    comptime FN = MB_FN
    comptime KB = AFN_GEMM_KB
    comptime NSG = SGM * SGN
    comptime NT = NSG * 32
    comptime BM = 8 * FM * SGM
    comptime BN = 8 * FN * SGN
    comptime AST = BM + 4
    comptime BST = KB + 4
    comptime NF = FM * FN
    comptime ASZ = KB * AST
    comptime BSZ = BN * BST
    comptime PAGE_BYTES = (ASZ + BSZ) * 4
    comptime NPG = lib_smem_pages_for[COLUMN_APPLE, PAGE_BYTES]()
    comptime assert lib_smem_page_fits_for[COLUMN_APPLE, PAGE_BYTES](), (
        "mcd_bmma_kernel: one staged page must fit Apple's threadgroup memory"
    )
    comptime assert KB % 8 == 0, "mcd_bmma_kernel: whole 8-step fragments"
    var z = Int(block_idx.z)
    comptime if COMPACT:
        z = Int(ids.unsafe_load(z))
    if gate_all == 0 and gate.unsafe_load(z) == 0:
        return
    var a = a_in + z * Int(a_bs_in)
    var b = b_in + z * Int(b_bs_in)
    var c = c_in + z * Int(c_bs_in)
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    comptime if ORDERED:
        comptime assert SPLIT, "ordered partials require split windows"
        c = c + Int(block_idx.y) * m * n
    var a_si = Int(a_si_in)
    var a_sp = Int(a_sp_in)
    var b_sp = Int(b_sp_in)
    var b_sj = Int(b_sj_in)
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var lane = tid % 32
    var sgm = sg // SGN
    var sgn = sg % SGN
    var nbn = (n + BN - 1) // BN
    var bid = Int(block_idx.x)
    var m0 = (bid // nbn) * BM
    var n0 = (bid % nbn) * BN
    var qd = lane // 4
    var frow = (qd & 4) + ((lane // 2) % 4)
    var fcol = (qd & 2) * 2 + (lane % 2) * 2
    var at = stack_allocation[NPG * ASZ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var bt = stack_allocation[NPG * BSZ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acc = InlineArray[_M64, NF](fill=_M64(0))
    var a_ofast = a_si == 1 and a_sp != 1
    var b_ofast = b_sj == 1 and b_sp != 1
    var kb = 0
    var ke = k
    comptime if SPLIT:
        kb = Int(block_idx.y) * Int(k_split_in)
        ke = min(k, kb + Int(k_split_in))
    var windows = (ke - kb + KB - 1) // KB
    var ra = _afn_gload[BM, KB, NT, DType.float32](a, a_si, a_sp, m0, m, kb, min(KB, ke - kb), tid, a_ofast)
    var rb = _afn_gload[BN, KB, NT, DType.float32](b, b_sj, b_sp, n0, n, kb, min(KB, ke - kb), tid, b_ofast)
    comptime if NPG == 2:
        _afn_stage[BM, KB, NT, True, AST](at, ra, tid, a_ofast)
        _afn_stage[BN, KB, NT, False, BST](bt, rb, tid, b_ofast)
        barrier()
        if windows > 1:
            ra = _afn_gload[BM, KB, NT, DType.float32](a, a_si, a_sp, m0, m, kb + KB, min(KB, ke - kb - KB), tid, a_ofast)
            rb = _afn_gload[BN, KB, NT, DType.float32](b, b_sj, b_sp, n0, n, kb + KB, min(KB, ke - kb - KB), tid, b_ofast)
    for w in range(windows):
        var k0 = kb + w * KB
        var cur = w % NPG
        var atc = at + cur * ASZ
        var btc = bt + cur * BSZ
        comptime if NPG == 1:
            _afn_stage[BM, KB, NT, True, AST](at, ra, tid, a_ofast)
            _afn_stage[BN, KB, NT, False, BST](bt, rb, tid, b_ofast)
            barrier()
            if w + 1 < windows:
                var k1 = k0 + KB
                ra = _afn_gload[BM, KB, NT, DType.float32](a, a_si, a_sp, m0, m, k1, min(KB, ke - k1), tid, a_ofast)
                rb = _afn_gload[BN, KB, NT, DType.float32](b, b_sj, b_sp, n0, n, k1, min(KB, ke - k1), tid, b_ofast)
        comptime for p8 in range(KB // 8):
            var af = InlineArray[_M64, FM](fill=_M64(0))
            var bf = InlineArray[_M64, FN](fill=_M64(0))
            comptime for fm in range(FM):
                af[fm] = _afn_load_t(atc + (8 * p8) * AST + (sgm * FM + fm) * 8, AST)
            comptime for fq in range(FN):
                bf[fq] = _afn_load_t(btc + ((sgn * FN + fq) * 8) * BST + 8 * p8, BST)
            comptime for fm in range(FM):
                comptime for fq in range(FN):
                    acc[fm * FN + fq] = _afn_mma(af[fm], bf[fq], acc[fm * FN + fq])
        comptime if NPG == 2:
            if w + 1 < windows:
                var nxt = (w + 1) % NPG
                _afn_stage[BM, KB, NT, True, AST](at + nxt * ASZ, ra, tid, a_ofast)
                _afn_stage[BN, KB, NT, False, BST](bt + nxt * BSZ, rb, tid, b_ofast)
                if w + 2 < windows:
                    var k2 = k0 + 2 * KB
                    ra = _afn_gload[BM, KB, NT, DType.float32](a, a_si, a_sp, m0, m, k2, min(KB, ke - k2), tid, a_ofast)
                    rb = _afn_gload[BN, KB, NT, DType.float32](b, b_sj, b_sp, n0, n, k2, min(KB, ke - k2), tid, b_ofast)
        barrier()
    comptime for fm in range(FM):
        comptime for fq in range(FN):
            comptime for e in range(2):
                var gi = m0 + (sgm * FM + fm) * 8 + frow
                var gj = n0 + (sgn * FN + fq) * 8 + fcol + e
                if gi < m and gj < n:
                    var v = acc[fm * FN + fq][e]
                    comptime if SPLIT and not ORDERED:
                        _ = Atomic.fetch_add(c.unsafe_offset(gi * n + gj), v)
                    else:
                        c.unsafe_store(gi * n + gj, v)


def mcd_g1_gram_batched_kernel(
    c: F32Ptr, a: F32Ptr, gate: I32Ptr, gate_all: Int32,
    m: Int32, k: Int32, a_bs: Int32, c_bs: Int32,
):
    """Non-split TN self-Gram, with the incumbent candidate gate and strides.
    The common G1 implementation reads block_idx.x; z selects only the batch.
    Every block of an inactive candidate returns before touching any operand.
    """
    var z = Int(block_idx.z)
    if gate_all == 0 and gate.unsafe_load(z) == 0:
        return
    var x = a + z * Int(a_bs)
    scoped_kernel[64, 64, False](
        c + z * Int(c_bs), x, x, m, m, k,
        Int32(1), m, m, Int32(1), k,
    )


def mcd_bmma_zero_kernel(c: F32Ptr, gate: I32Ptr, gate_all: Int32, nc: Int32, cells: Int32, c_bs: Int32):
    """C_z[0, cells) = +0.0 for every gated candidate z (the split sum's
    seed, `afn_zero_kernel` per candidate)."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var mn = Int(cells)
    if t < Int(nc) * mn:
        var z = t // mn
        if gate_all != 0 or gate.unsafe_load(z) != 0:
            c.unsafe_store(z * Int(c_bs) + (t - z * mn), Float32(0.0))


def launch_gemm_mma_batched(
    ctx: DeviceContext, a: F32Ptr, b: F32Ptr, c: F32Ptr, m: Int, k: Int, n: Int, ta: Bool, tb: Bool,
    nc: Int, a_bs: Int, b_bs: Int, c_bs: Int, gate: I32Ptr, gate_all: Bool,
    compact: Optional[McdCompactWorkspace] = None,
) raises:
    """`_launch_gemm_mma(ctx, a + z a_bs, b + z b_bs, c + z c_bs, m, k, n,
    ta, tb)` for every candidate z < nc whose gate word is set (all when
    gate_all), as one launch (plus one zero launch when split)."""
    if nc <= 0 or m <= 0 or n <= 0 or k <= 0:
        return
    # the strides, tiles and split of _launch_gemm_mma, unchanged
    var a_si = 1 if ta else k
    var a_sp = m if ta else 1
    var b_sp = 1 if tb else n
    var b_sj = k if tb else 1
    var tiles = ((m + MB_BM - 1) // MB_BM) * ((n + MB_BN - 1) // MB_BN)
    var splits = 1
    if tiles < DFG_BLOCK_TARGET and k >= 2 * DFG_MIN_SPLIT_STEPS:
        splits = min(DFG_BLOCK_TARGET // tiles, k // DFG_MIN_SPLIT_STEPS)
    var ga = Int32(1) if gate_all else Int32(0)
    # Correctness limits only, no shape window. The batched kernel is a TN
    # self-Gram (C = A^T A, one operand pointer and stride), so it needs
    # m == n, ta, not tb and a == b with equal batch strides. It is the
    # non-split scoped_kernel[64, 64, False], whose tiles equal MB_BM x MB_BN,
    # so it applies only where the incumbent plan picks splits == 1 (a split
    # plan needs the zero seed and atomics below). Every index is
    # bounds-checked, so any m, k >= 1 is correct. Shapes and batch strides
    # travel as Int32, so they must fit in Int32. No overlap of C with A.
    var eligible = (
        m == n and ta and not tb and a == b and a_bs == b_bs and splits == 1
        and a_bs >= m * k and c_bs >= m * n and c != a and c != b
        and max(a_bs, max(c_bs, max(m * k, m * n))) <= 2147483647
    )
    comptime if MCD_G1_LEGACY_WINDOW:
        eligible = eligible and m >= 129 and m <= 256 and k >= 128 and k <= 1023
    comptime if MCD_G1_AUDIT:
        var state = MCD_G1_STATE.get_or_create_ptr()
        state[].counts[0] += 1
        if eligible:
            state[].counts[1] += 1
            state[].counts[3] += nc  # potential slots, NOT active device candidates
            state[].last[0] = m
            state[].last[1] = k
            state[].last[2] = n
            state[].last[3] = nc
            state[].last[4] = a_bs
            state[].last[5] = c_bs
            state[].last[6] = splits
    comptime if MCD_G1_GRAM:
        if eligible:
            comptime if MCD_G1_AUDIT:
                MCD_G1_STATE.get_or_create_ptr()[].counts[2] += 1
            ctx.enqueue_function[mcd_g1_gram_batched_kernel](
                c, a, gate, ga, Int32(m), Int32(k), Int32(a_bs), Int32(c_bs),
                grid_dim=(tiles, 1, nc), block_dim=(MB_NT, 1, 1),
            )
            return
    # F06 PENDING: reduced active grid is an explicit FAST Apple experiment.
    # Count-read completion/scan overhead is included in whole-call timing.
    # Original IDs own output slots, gate rechecks and split-K windows/seeds.
    comptime if MCD_FAST_ACTIVE_COMPACT:
        if Bool(compact) and not gate_all:
            var live = compact_candidate_count(ctx, compact[], gate, nc)
            mcd_experiment_hit(0)
            if live == 0: return
            var ids = I32Ptr(unsafe_from_address=Int(compact[].ids.unsafe_ptr()))
            if splits > 1:
                var per = ((k + splits - 1)//splits + AFN_GEMM_KB - 1)//AFN_GEMM_KB*AFN_GEMM_KB
                splits = (k + per - 1)//per
                ctx.enqueue_function[mcd_bmma_zero_kernel](c, gate, ga, Int32(nc), Int32(m*n), Int32(c_bs), grid_dim=max((nc*m*n+MB_ZERO_TPB-1)//MB_ZERO_TPB,1), block_dim=MB_ZERO_TPB)
                ctx.enqueue_function[mcd_bmma_kernel[True, False, True]](
                    c, a, b, gate, ga, Int32(m), Int32(n), Int32(k), Int32(a_si), Int32(a_sp), Int32(b_sp), Int32(b_sj), Int32(per), Int32(a_bs), Int32(b_bs), Int32(c_bs), ids,
                    grid_dim=(tiles,splits,live), block_dim=(MB_NT,1,1))
            else:
                ctx.enqueue_function[mcd_bmma_kernel[False, False, True]](
                    c, a, b, gate, ga, Int32(m), Int32(n), Int32(k), Int32(a_si), Int32(a_sp), Int32(b_sp), Int32(b_sj), Int32(k), Int32(a_bs), Int32(b_bs), Int32(c_bs), ids,
                    grid_dim=(tiles,1,live), block_dim=(MB_NT,1,1))
            return
    if splits > 1:
        var per = (k + splits - 1) // splits
        per = ((per + AFN_GEMM_KB - 1) // AFN_GEMM_KB) * AFN_GEMM_KB
        splits = (k + per - 1) // per
        ctx.enqueue_function[mcd_bmma_zero_kernel](
            c, gate, ga, Int32(nc), Int32(m * n), Int32(c_bs),
            grid_dim=max((nc * m * n + MB_ZERO_TPB - 1) // MB_ZERO_TPB, 1), block_dim=MB_ZERO_TPB,
        )
        ctx.enqueue_function[mcd_bmma_kernel[True]](
            c, a, b, gate, ga, Int32(m), Int32(n), Int32(k),
            Int32(a_si), Int32(a_sp), Int32(b_sp), Int32(b_sj), Int32(per),
            Int32(a_bs), Int32(b_bs), Int32(c_bs), gate,
            grid_dim=(tiles, splits, nc), block_dim=(MB_NT, 1, 1),
        )
    else:
        comptime if MCD_FAST_BOUND_BATCH:
            for first in range(0, nc, MCD_CANDIDATE_BATCH):
                var count = min(MCD_CANDIDATE_BATCH, nc - first)
                ctx.enqueue_function[mcd_bmma_kernel[False]](
                    c + first * c_bs, a + first * a_bs, b + first * b_bs, gate + first, ga,
                    Int32(m), Int32(n), Int32(k),
                    Int32(a_si), Int32(a_sp), Int32(b_sp), Int32(b_sj), Int32(k),
                    Int32(a_bs), Int32(b_bs), Int32(c_bs), gate,
                    grid_dim=(tiles, 1, count), block_dim=(MB_NT, 1, 1),
                )
                BATCH_STATE.get_or_create_ptr()[].launches += 1
            return
        ctx.enqueue_function[mcd_bmma_kernel[False]](
            c, a, b, gate, ga, Int32(m), Int32(n), Int32(k),
            Int32(a_si), Int32(a_sp), Int32(b_sp), Int32(b_sj), Int32(k),
            Int32(a_bs), Int32(b_bs), Int32(c_bs), gate,
            grid_dim=(tiles, 1, nc), block_dim=(MB_NT, 1, 1),
        )


@always_inline
def _cov_add(a: Float32, b: Float32) -> Float32:
    # Explicit intrinsic without fast-math flags. Multiplication by one is
    # exact; the remaining operation is one rounded addition. Ordinary FAST
    # expressions must not reassociate the compensated fold into a plain sum.
    return llvm_intrinsic["llvm.fma.f32", Float32, has_side_effect=False](Float32(1), a, b)


#: Rows per ordered split: MCD_ORD_CHAIN_WINDOWS matrix-unit K windows
#: (AFN_GEMM_KB rows each). The quality HOLD (mcd-ordered-direct-q-r3) came
#: from keeping main's split count: with 2..9 splits per cell the error is
#: the fp32 accumulation INSIDE each split's MMA chain (600 rows at
#: 3000 x 220), not the cross-split fold the candidate compensated, so B and
#: A differed by noise and the strict per-key B <= A max-abs gate became a
#: coin flip. Short chains move the error into the compensated fold.
comptime MCD_ORD_CHAIN_WINDOWS = 4
comptime MCD_ORD_CHAIN = MCD_ORD_CHAIN_WINDOWS * AFN_GEMM_KB
#: Partial-region cap (float words): the kernels' Int32 addressing bound
#: (2^31) / 8 = 256M words (1 GiB). Past it the split count falls back
#: toward main's (longer chains) instead of failing.
comptime MCD_ORD_MAX_WORDS = 2147483647 // 8


def ordered_cov_shape(nc: Int, rows: Int, d: Int) -> Tuple[Int, Int]:
    """(splits, rows per split) of the ordered covariance: chains of
    MCD_ORD_CHAIN rows (a multiple of AFN_GEMM_KB), at least main's split
    count, at most what MCD_ORD_MAX_WORDS of partials hold."""
    var tiles = ((d + MB_BM - 1) // MB_BM) * ((d + MB_BN - 1) // MB_BN)
    var base = 1
    if tiles < DFG_BLOCK_TARGET and rows >= 2 * DFG_MIN_SPLIT_STEPS:
        base = min(DFG_BLOCK_TARGET // tiles, rows // DFG_MIN_SPLIT_STEPS)
    var splits = max(base, (rows + MCD_ORD_CHAIN - 1) // MCD_ORD_CHAIN)
    var cap = MCD_ORD_MAX_WORDS // max(max(nc, 1) * d * d, 1)
    splits = max(min(splits, cap), 1)
    var per = rows
    if splits > 1:
        per = (rows + splits - 1) // splits
        per = ((per + AFN_GEMM_KB - 1) // AFN_GEMM_KB) * AFN_GEMM_KB
        splits = (rows + per - 1) // per
    return splits, per


def ordered_cov_scratch(nc: Int, rows: Int, d: Int) raises -> Int:
    if nc <= 0 or rows <= 0 or d <= 0:
        return 1
    var shape = ordered_cov_shape(nc, rows, d)
    var words = nc * shape[0] * d * d if shape[0] > 1 else 1
    if words > 2147483647:
        raise Error("ordered covariance partials exceed Int32 addressing")
    return words


def mcd_cov_fold_kernel(part: F32Ptr, dst: F32Ptr, gate: I32Ptr, gate_all: Int32,
                        nc_: Int32, d_: Int32, splits_: Int32, stride_: Int32):
    # Independent thread per candidate/output cell; no atomics, global serial
    # loop or cooperative single-block reduction. One ordered Neumaier chain
    # over the cell's splits (ceil(rows / MCD_ORD_CHAIN), capped by
    # MCD_ORD_MAX_WORDS).
    var at = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var dd = Int(d_)
    var cells = dd * dd
    var nc = Int(nc_)
    if at >= nc * cells:
        return
    var candidate = at // cells
    if gate_all == 0 and gate.unsafe_load(candidate) == 0:
        return
    var cell = at % cells
    var splits = Int(splits_)
    var base = candidate * splits * cells + cell
    var total = Float32(0)
    var correction = Float32(0)
    for split in range(splits):
        var value = part.unsafe_load(base + split * cells)
        var next = _cov_add(total, value)
        var error = Float32(0)
        if abs(total) >= abs(value):
            error = _cov_add(_cov_add(total, -next), value)
        else:
            error = _cov_add(_cov_add(value, -next), total)
        correction = _cov_add(correction, error)
        total = next
    dst.unsafe_store(candidate * Int(stride_) + cell, _cov_add(total, correction))


def launch_mcd_cov_ordered(ctx: DeviceContext, x: F32Ptr, dst: F32Ptr, part: F32Ptr,
                           rows: Int, d: Int, nc: Int, x_stride: Int, out_stride: Int,
                           gate: I32Ptr, gate_all: Bool) raises:
    """Only X^T X; caller owns sufficient partial storage through completion.
    Same MMA tile/window operands as main, changed cross-window accumulation.
    Unsplit inputs retain the original kernel and arithmetic exactly."""
    if nc <= 0 or rows <= 0 or d <= 0:
        return
    var shape = ordered_cov_shape(nc, rows, d)
    var splits = shape[0]
    if splits == 1:
        launch_gemm_mma_batched(ctx, x, x, dst, d, rows, d, True, False,
                                nc, x_stride, x_stride, out_stride, gate, gate_all)
        return
    _ = ordered_cov_scratch(nc, rows, d)  # validate before narrowing strides
    var ga = Int32(1 if gate_all else 0)
    var tiles = ((d + MB_BM - 1) // MB_BM) * ((d + MB_BN - 1) // MB_BN)
    ctx.enqueue_function[mcd_bmma_kernel[True, True]](
        part, x, x, gate, ga, Int32(d), Int32(d), Int32(rows),
        Int32(1), Int32(d), Int32(d), Int32(1), Int32(shape[1]),
        Int32(x_stride), Int32(x_stride), Int32(splits*d*d), gate,
        grid_dim=(tiles, splits, nc), block_dim=(MB_NT, 1, 1))
    ctx.enqueue_function[mcd_cov_fold_kernel](
        part, dst, gate, ga, Int32(nc), Int32(d), Int32(splits), Int32(out_stride),
        grid_dim=(nc*d*d + MB_ZERO_TPB - 1) // MB_ZERO_TPB, block_dim=MB_ZERO_TPB)


struct _CovReach(Defaultable, Movable):
    var raw: Int
    var final_count: Int
    var raw_split: Int
    var final_split: Int

    def __init__(out self):
        self.raw = 0
        self.final_count = 0
        self.raw_split = 0
        self.final_split = 0


comptime COV_REACH = _Global[StorageType=_CovReach, name="MojoMcdOrderedCovReach", init_fn=_CovReach.__init__]


def note_cov_route(raw: Bool, rows: Int, d: Int) raises:
    # Host metadata only; never consumes matrix values. Counts make no-op
    # quality passes observable. No production dispatch outside the opt-in.
    comptime if MCD_ORDERED_COV:
        var r = COV_REACH.get_or_create_ptr()
        var split = ordered_cov_shape(1, rows, d)[0] > 1
        if raw:
            r[].raw += 1
            if split:
                r[].raw_split += 1
        else:
            r[].final_count += 1
            if split:
                r[].final_split += 1


def mcd_cov_reach_py(which: PythonObject) raises -> PythonObject:
    var i = Int(py=which)
    var r = COV_REACH.get_or_create_ptr()
    if i == -1:
        r[].raw = 0
        r[].final_count = 0
        r[].raw_split = 0
        r[].final_split = 0
        return PythonObject(0)
    if i == 0:
        return PythonObject(1 if MCD_ORDERED_COV else 0)
    if i == 1:
        return PythonObject(r[].raw)
    if i == 2:
        return PythonObject(r[].final_count)
    if i == 3:
        return PythonObject(r[].raw_split)
    if i == 4:
        return PythonObject(r[].final_split)
    raise Error("unknown ordered covariance reach counter")
