# SPDX-License-Identifier: Apache-2.0
"""Neural-only model dispatcher for the NN01--NN16 A/B source experiments.

Existing signatures, unchanged OFF route, one selected host/device numerical
profile for every routed forward/backward product. Classical code retains its
original imports. Allocation, staging and synchronization remain in the model
operation; no benchmark controller prepares hidden runtime work. Unverified.
"""
from gemm.experiments.neural_profile_device import neural_profile_device
from max.gpu import block_idx,block_dim,thread_idx
from max.gpu.host import DeviceContext,DeviceBuffer
from checks.kernel_matrix import TARGET_COLUMN,COLUMN_NVIDIA
from checks.numerics import ftz
from checks.rtf_seam import rtf_mul_add
from gemm.checks.gemm_epilogue import (
    identical_gemm_epi_takes as _incumbent_epi_takes,
    identical_gemm_epi_into as _incumbent_epi_into,
)
from core.step_phase import step_count_device_alloc,step_count_sync
from gemm.checks.gemm_identical import (
    identical_gemm_into as _incumbent_into,
    identical_gemm as _incumbent_gemm,
    identical_gemm_workspace_max_floats as _incumbent_workspace,
    # re-exported: training/checks/loss_check.mojo, training/byte_lm.mojo and the training checks import these from here
    identical_gemm_with_plan,PLAN_FLAT,ANY_SABOTAGE,
)
# lane/fx-gemm-worklist: the work-list arm (-D MOJOLEARN_GEMM_FAST_WORKLIST=1, default off).
from gemm.checks.gemm_identical import GEMM_FAST_WORKLIST,GemmWorklist
from gemm.experiments.neural_profile import (
    NEURAL_PROFILE_CHANGED,NEURAL_LEAF,NEURAL_CHAINS,neural_validate,
)
from gemm.experiments.neural_switches import ROLE_PROJECTION,ROLE_HEAD,ROLE_WGRAD
# lane fx-fast-matrix-units (2026-11-19, -D MOJOLEARN_GEMM_FAST_MATRIX_UNITS=1, default OFF, A/B owed with quality): the
# matrix-units relaxation of an NVIDIA `fast` build (gemm/fast_matrix_units.mojo). False on every other build.
from gemm.fast_matrix_units import (
    GEMM_FAST_MATRIX_UNITS,GEMM_FAST_MATRIX_UNITS_ALL,
    matrix_units_takes,matrix_units_workspace_floats,matrix_units_gemm_into,
)
# lane fx-gemm-seq-packed (2026-11-18): the names `identical_gemm_leaf_ptr_into` below reads; nothing else uses them here.
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE,NUMERIC_IDENTICAL
from checks.kernel_matrix import COLUMN_AMD,lib_smem_pages_for,lib_smem_page_fits_for
from gemm.checks.gemm_identical import (
    identical_gemm_kpack_kernel,_tile_grid,NI06_ONE_PAGE,TUNED_TPB,TUNED_TC,TUNED_RPT,
    GEMM_KPACK_RPT,GEMM_KPACK_CPT,GEMM_KPACK_KS,GEMM_KPACK_FS,GEMM_KPACK_PAD,GEMM_KPACK_ALIGN,
    GEMM_KPACK_PAGE_GUARD_BYTES,GEMM_TILE_MIN_BLOCKS_DEFAULT,
)
# lane fx-leftovers-product (2026-11-19): the store epilogue kinds that read a second buffer, and their kernel
# (`identical_gemm_leaf_ptr_epi_into` below). The kinds are re-exported: sequence/gemm_packed_route.mojo imports them here.
from gemm.checks.gemm_identical import (
    identical_gemm_kpack_epi_ptr_kernel,GEMM_EPI_PLUS_BIAS,GEMM_EPI_BIAS_ACT,GEMM_EPI_ACT_BWD,
)
# (Sweep 2026-11-15: with MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE (lane/expdel2-25-neural) and the NN07/NN12/NN16 arms deleted,
# NN01, NN02, NN07, NN09, NN10, NN11, NN12, NN15 and NN16 were all False. Removed with them: gemm/experiments/
# neural_streaming.mojo, neural_tiled.mojo, neural_plans.mojo, neural_grouped.mojo; here NEURAL_PAIR_ENABLED,
# NEURAL_WORKSPACE_REUSE_ENABLED, the schedule constants, _scheduled/_unscheduled workspace sizing,
# _profile_or_cost_into, _selected_cost_plan, the scheduled routes of identical_gemm_into, the NN07 operand pack
# of identical_gemm_pair_into (then the pair entry itself, whose last caller was NEURAL_PAIR_ENABLED's) and the NN12
# retained-plan path and plan fields of GemmWorkspace. Markers moved from the four deleted files:)
# TOMBSTONE: MOJOLEARN_IDN_NEURAL_NN16 (opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-25-neural; code recoverable at 1362c86435.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_NN16.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE=8 (slower) deleted 2026-10-07 by 225398f14; code recoverable at 225398f14^.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE-arm8.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_IDN_NEURAL_NN12 (opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-25-neural; code recoverable at a9cf951197.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_NN12.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_IDN_NEURAL_NN07 (opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-25-neural; code recoverable at dd47bfb212.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_NN07.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_IDN_NEURAL_GEMM_OZAKI_SLICES (NN-OZ int8 Ozaki-split neural GEMM; DELETE-slower: grid g50ebe26a5, slices 4/5
# on 10 neural workloads, e.g. lm-train-step NV L40S 35.3 -> 3172.6 / 3709.4 ms, AMD MI325X 42.7 -> 334.3 / 407.5 ms; cnn-clf avg
# 25.9x / 30.1x; mlp-train-step 1.20x / 1.22x, the closest) deleted 2026-10-11 by lane/grid-act-16; code recoverable at 7aa609453.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_OZAKI_SLICES.patch; record in docs/TOMBSTONES.md.

# NN05 (the fused pair kernel arm) deleted 2026-10-08 (lane grid-act-4, grid
# ge123e6f9, 2.2-3.9x slower); see the tombstone at _neural_pair_kernel's old site.
comptime NEURAL_GEMM_EXPERIMENT_ENABLED = NEURAL_PROFILE_CHANGED
#: The work-list arm as the model call sites see it: on only where the kernel file compiles it (its column and
#: body tests) and the neural profile is the incumbent (a changed profile runs another kernel per product).
#: False compiles `GemmWorkspace.list_add` to `run` and the other two list calls to nothing.
comptime NEURAL_GEMM_WORKLIST = GEMM_FAST_WORKLIST and not NEURAL_GEMM_EXPERIMENT_ENABLED
# TOMBSTONE: MOJOLEARN_IDN_NEURAL_STREAM_GROUP (DELETE-ARM, unmeasured opt-in, rule a) deleted 2026-11-15 by lane/expdel2-26-neural; code recoverable at 5fc464dce4.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_STREAM_GROUP.patch
# TOMBSTONE: MOJOLEARN_IDN_NEURAL_STAGE_DEPTH (DELETE-ARM, unmeasured opt-in, rule a) deleted 2026-11-15 by lane/expdel2-26-neural; code recoverable at 5fc464dce4.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_STAGE_DEPTH.patch
# TOMBSTONE: MOJOLEARN_IDN_NEURAL_STAGE_PAD (DELETE-ARM, unmeasured opt-in, rule a) deleted 2026-11-15 by lane/expdel2-26-neural; code recoverable at 5fc464dce4.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_STAGE_PAD.patch
# TOMBSTONE: MOJOLEARN_IDN_NEURAL_STAGE_SWIZZLE (DELETE-ARM, unmeasured opt-in, rule a) deleted 2026-11-15 by lane/expdel2-26-neural; code recoverable at 5fc464dce4.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_STAGE_SWIZZLE.patch
# TOMBSTONE: MOJOLEARN_IDN_NEURAL_GEMM_ARM (DELETE-ARM, unmeasured opt-in, rule a) deleted 2026-11-15 by lane/expdel2-26-neural; code recoverable at 5fc464dce4.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_ARM.patch
# TOMBSTONE: MOJOLEARN_IDN_NEURAL_FILL_BLOCKS (DELETE-ARM, unmeasured opt-in, rule a) deleted 2026-11-15 by lane/expdel2-26-neural; code recoverable at 5fc464dce4.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_FILL_BLOCKS.patch
# TOMBSTONE: MOJOLEARN_IDN_NEURAL_RETAINED_FLOATS (DELETE-ARM, unmeasured opt-in, rule a) deleted 2026-11-15 by lane/expdel2-26-neural; code recoverable at 5fc464dce4.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_RETAINED_FLOATS.patch
# TOMBSTONE: MOJOLEARN_IDN_NEURAL_GEMM_CONTROL (DELETE-ARM, unmeasured opt-in, rule a) deleted 2026-11-15 by lane/expdel2-26-neural; code recoverable at 5fc464dce4.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_CONTROL.patch
# TOMBSTONE: MOJOLEARN_IDN_NEURAL_PAIR_CONTROL (DELETE-ARM, unmeasured opt-in, rule a) deleted 2026-11-15 by lane/expdel2-26-neural; code recoverable at 5fc464dce4.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_PAIR_CONTROL.patch
# TOMBSTONE: MOJOLEARN_IDN_NEURAL_FOLD_SHARED (DELETE-ARM, unmeasured opt-in, rule a) deleted 2026-11-15 by lane/expdel2-26-neural; code recoverable at 5fc464dce4.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_FOLD_SHARED.patch


def _overlaps(mut a: DeviceBuffer[DType.float32],na: Int,
              mut b: DeviceBuffer[DType.float32],nb: Int) -> Bool:
    if na<=0 or nb<=0:
        return False
    var ap = Int(a.unsafe_ptr())
    var bp = Int(b.unsafe_ptr())
    return ap<bp+4*nb and bp<ap+4*na


def identical_gemm_workspace_max_floats(m: Int,n: Int,k: Int) -> Int:
    """Covers the route the build takes for (m, n, k)."""
    comptime if NEURAL_PROFILE_CHANGED:
        return 1
    return _incumbent_workspace(m,n,k)


def _admit_product(mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int) raises:
    """Admission shared by direct and model-owned cached-plan routes."""
    neural_validate(m,n,k,op)
    if len(c)<m*n or len(a)<m*k or len(b)<n*k or len(ws)<identical_gemm_workspace_max_floats(m,n,k):
        raise Error("neural GEMM operand/workspace storage too short")
    if (_overlaps(c,m*n,a,m*k) or _overlaps(c,m*n,b,n*k)
        or _overlaps(ws,len(ws),c,m*n) or _overlaps(ws,len(ws),a,m*k)
        or _overlaps(ws,len(ws),b,n*k)):
        raise Error("neural GEMM output/workspace must not overlap inputs or each other")


def _matrix_units_admits(mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int) -> Bool:
    """Whether the matrix-units arm may take this product: the size rule, operands long enough and an output that
    overlaps no input, and for the form that transposes A into the workspace a workspace that overlaps nothing.
    False sends the call to the route below with its own refusals, so no refusal moves or changes its wording.
    A and B hold m k and n k floats in all three addressings."""
    if not matrix_units_takes(m,n,k):
        return False
    if len(c)<m*n or len(a)<m*k or len(b)<n*k:
        return False
    if _overlaps(c,m*n,a,m*k) or _overlaps(c,m*n,b,n*k):
        return False
    var wsn = matrix_units_workspace_floats(m,n,k,op)
    if wsn>0:
        if len(ws)<wsn or _overlaps(ws,wsn,c,m*n) or _overlaps(ws,wsn,a,m*k) or _overlaps(ws,wsn,b,n*k):
            return False
    return True


def identical_gemm_into[allow_vendor: Bool = True,ROLE: Int = ROLE_PROJECTION](
    ctx: DeviceContext,mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int,
) raises:
    """ROLE is the caller class (ROLE_PROJECTION, ROLE_HEAD, ROLE_WGRAD); it
    never changes the route."""
    comptime assert ROLE == ROLE_PROJECTION or ROLE == ROLE_HEAD or ROLE == ROLE_WGRAD, "neural GEMM role"
    # lane fx-fast-matrix-units: an NVIDIA `fast` build with the matrix-units row relaxed sends a large product of a
    # caller that left the vendor route open (forward, dInput and dWeight alike) to the TF32 tensor-core product,
    # FP32 accumulation. A caller that closed it (`allow_vendor=False`; taken too under the arm's value 3, its own
    # owed row), a product under the size rule and every
    # refusal keep the route below. Per call under the arm: the integer tests of `_matrix_units_admits`, then one
    # launch in place of the route's (two for OP_TN: the transpose of A into `ws`); no allocation, lookup or wait.
    # Asynchronous like the route below. Not compiled without the define.
    comptime if GEMM_FAST_MATRIX_UNITS and (allow_vendor or GEMM_FAST_MATRIX_UNITS_ALL):
        if _matrix_units_admits(c,a,b,ws,m,n,k,op):
            if matrix_units_gemm_into(ctx,c,a,b,ws,m,n,k,op):
                return
    comptime if not NEURAL_GEMM_EXPERIMENT_ENABLED:
        _incumbent_into[allow_vendor](ctx,c,a,b,ws,m,n,k,op)
        return
    _admit_product(c,a,b,ws,m,n,k,op)
    if m==0 or n==0:
        return
    # TOMBSTONE: MOJOLEARN_IDN_NEURAL_GEMM_OZAKI_SLICES (NN-OZ int8 Ozaki-split neural GEMM; DELETE-slower: grid g50ebe26a5, slices 4/5
    # on 10 neural workloads, e.g. lm-train-step NV L40S 35.3 -> 3172.6 / 3709.4 ms, AMD MI325X 42.7 -> 334.3 / 407.5 ms; cnn-clf avg
    # 25.9x / 30.1x; mlp-train-step 1.20x / 1.22x, the closest) deleted 2026-10-11 by lane/grid-act-16; code recoverable at 7aa609453.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_OZAKI_SLICES.patch; record in docs/TOMBSTONES.md.
    neural_profile_device[NEURAL_LEAF,NEURAL_CHAINS](ctx,c,a,b,m,n,k,op)


# TOMBSTONE: neural_model_leaf_into (its one caller, the deleted small MLP step's MLP_GEMM_LEAF128 dispatch, plan row X2) deleted 2026-11-09 by lane/x2-small-mlp-on-executor; code recoverable at f59d913b0b.
# Restore: git apply experiments/removed/MOJOLEARN_X2_SMALL_MLP_HAND_ROUTES.patch
# TOMBSTONE restored 2026-11-09 by lane/r1-restore-measured-winners as the GEMM of op small_mlp (OPS_SMALL_MLP,
# autograd/small_mlp_op.mojo); body from f59d913b0b, unchanged.
def neural_model_leaf_into[MIN_LEAF: Int](ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int,
) raises:
    """A MODEL call site's own leaf choice (lane grid-act-5, 2026-10-08): the
    NN03 neural profile body at MIN_LEAF for one model's products, in a build
    whose other neural callers keep the incumbent. This is the body
    identical_gemm_into reaches under -D MOJOLEARN_IDN_GEMM_LEAF=1|2 with no
    schedule arm (the last `elif NEURAL_PROFILE_CHANGED` branch), with the
    same validation and overlap refusals; the profile kernel reads no
    workspace, so none is taken. Caller: the small MLP op
    (autograd/small_mlp_op.mojo::SMALL_MLP_GEMM_LEAF128)."""
    comptime assert MIN_LEAF == 128 or MIN_LEAF == 256, "neural profile leaf 128|256 (leaf 64 deleted: I04 loser)"
    neural_validate(m,n,k,op)
    if len(c)<m*n or len(a)<m*k or len(b)<n*k:
        raise Error("neural GEMM operand storage too short")
    if _overlaps(c,m*n,a,m*k) or _overlaps(c,m*n,b,n*k):
        raise Error("neural GEMM output must not overlap inputs")
    if m==0 or n==0:
        return
    neural_profile_device[MIN_LEAF,NEURAL_CHAINS](ctx,c,a,b,m,n,k,op)


# ---------------------------------------------------------------------------
# lane fx-gemm-seq-packed (2026-11-18, idea 1 of briefs-20261118/ideas-gemm-headroom-RESULT.md):
# THE PACKED BODY AT A CALLER'S PARTITION, OVER RAW POINTERS.
# Reached from the sequence packed route (sequence/gemm_packed_route.mojo; the default since 2026-11-19,
# -D MOJOLEARN_GEMM_FAST_SEQ_PACKED_OFF removes the caller).
# Every neural caller keeps `identical_gemm_into`, whose `(L, P)` is `contract_partition(k)` and nothing else.
# ---------------------------------------------------------------------------
#: The columns whose build carries `identical_gemm_kpack_kernel` with the gather staging and the hardware fold
#: flush (the `kpack_hg` body): NVIDIA and AMD under IDENTICAL. Apple's packed page is bounded differently
#: (`PLAN_APPLE_MMA` is its body) and the CPU column has no kernel.
comptime GEMM_LEAF_PTR_COLUMN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and (
    TARGET_COLUMN == COLUMN_NVIDIA or TARGET_COLUMN == COLUMN_AMD
)
#: The two packed tiles the shipped dispatch already instantiates (`_shipped_body_kpack_hg`): the wide one,
#: `GEMM_KPACK_RPT` rows a thread (128 x 64 cells on NVIDIA, 128 x 128 on AMD), and the stepped-down one,
#: `TUNED_RPT` rows a thread (64 x 64 on NVIDIA, `gemm_nv_step_kpack_rows`; NVIDIA only there, so NVIDIA only here).
comptime GEMM_LEAF_PTR_RPT_WIDE = GEMM_KPACK_RPT
comptime GEMM_LEAF_PTR_RPT_SMALL = TUNED_RPT
comptime GEMM_LEAF_PTR_SMALL_ON = TARGET_COLUMN == COLUMN_NVIDIA
comptime GEMM_LEAF_PTR_BM_WIDE = GEMM_KPACK_RPT * (TUNED_TPB // TUNED_TC)
comptime GEMM_LEAF_PTR_BM_SMALL = TUNED_RPT * (TUNED_TPB // TUNED_TC)
comptime GEMM_LEAF_PTR_BN = GEMM_KPACK_CPT * TUNED_TC
#: The kernel's staged window, in terms of `k` (one barrier and one gather round trip a window).
comptime GEMM_LEAF_PTR_KS = GEMM_KPACK_KS
#: The local fold stack of each instantiation: unused storage above the highest reachable level, never a bit
#: (`_kpack_hg_run_with_ws`). 4 levels cover 8 leaves and are the wide NVIDIA kernel the shipped dispatch runs
#: for short folds (the kernel admits FS 4 on NVIDIA only); every other instantiation keeps the profile-wide stack.
comptime GEMM_LEAF_PTR_FS_WIDE = 4 if TARGET_COLUMN == COLUMN_NVIDIA else GEMM_KPACK_FS
comptime GEMM_LEAF_PTR_FS_SMALL = GEMM_KPACK_FS
#: The column's block floor: fewer packed tiles than this leave SMs / CUs idle (`GEMM_TILE_MIN_BLOCKS_DEFAULT`,
#: measured per column for the neural tiles: 192 on the L40S's 142 SMs, which is 1.35 blocks an SM (an earlier
#: comment here said "two blocks an SM"; 2 x 142 = 284, so that was wrong), 1024 on the MI325X's 304 CUs). The
#: comptime column constant, so the packed route's rule reads no process slot per call. The shipped tile
#: step-down reads the floor scaled by the device's SM count (`gemm_nv_block_floor`,
#: gemm/checks/gemm_identical.mojo: equal to this constant from 95 to 142 SMs, so on the L40S and the A100);
#: this default-off route keeps the constant until its owed A/B, whose row names the SM count it ran on.
comptime GEMM_LEAF_PTR_MIN_BLOCKS = GEMM_TILE_MIN_BLOCKS_DEFAULT


def identical_gemm_leaf_ptr_into[RPT: Int,FS: Int](
    ctx: DeviceContext,
    c: MutPointer[Float32,MutAnyOrigin],
    a: MutPointer[Float32,MutAnyOrigin],
    b: MutPointer[Float32,MutAnyOrigin],
    m: Int,n: Int,k: Int,leaf: Int,p_count: Int,
    a_si: Int,a_sp: Int,b_sp: Int,b_sj: Int,
) raises:
    """One asynchronous all-leaves launch of the packed body (`identical_gemm_kpack_kernel`, the `kpack_hg`
    spelling: padded 16-byte page, gather staging, hardware fold flush) with the partition `(leaf, p_count)`
    GIVEN BY THE CALLER instead of `contract_partition(k)`, over raw device pointers and operand strides:

        A_eff[i, p] = a[i * a_si + p * a_sp]      B_eff[p, j] = b[p * b_sp + j * b_sj]      C[i, j] = c[i * n + j]

    The kernel reads `leaf` and `p_count` as arguments and nothing else about the partition (`_leaf_bounds`:
    leaf t is `[t leaf, min((t + 1) leaf, k))`), so the order it computes is, per cell,

        s_t = chain over p in leaf t, ascending, seeded +0.0: acc = ftz(fma(ftz(A_eff[i, p]), ftz(B_eff[p, j]), acc))
        C[i, j] = ftz(T(ftz(s_0), ..., ftz(s_{P-1})))        T = the contract's 7.2 tree; no addition at P = 1

    With `leaf = k, p_count = 1` that is ONE ascending chain over the whole of k and `C = ftz(ftz(acc_k)) =
    acc_k`: the sequence GEMM's word (sequence/gemm_packed_route.mojo writes both orders out side by side).
    The partition must cover k exactly (`(p_count - 1) leaf < k <= p_count leaf`) and fit the instantiation's
    fold stack (`p_count <= 2^(FS - 1)`); anything else is refused before the launch. `c` must be dense
    (row stride `n`) and must not overlap `a` or `b` (blocks store C while others gather the operands): the
    caller's contract, as for every sequence GEMM kernel; no buffer is looked up here.

    MIRRORS `gemm_identical._kpack_launch` (the page count, the page-fits assert, the grid) because that
    launcher takes `DeviceBuffer`s and the sequence executor holds raw pointers: a view per operand per call
    would be three constructions on a hot path. A change to the kernel's argument block (lane fx-gemm-worklist)
    must be made here too. `RPT, FS` = `(GEMM_LEAF_PTR_RPT_WIDE, GEMM_LEAF_PTR_FS_WIDE)` or
    `(GEMM_LEAF_PTR_RPT_SMALL, GEMM_LEAF_PTR_FS_SMALL)`: the two kernels the NVIDIA build already carries, so
    the arm adds no kernel there."""
    comptime assert GEMM_LEAF_PTR_COLUMN, "identical_gemm_leaf_ptr_into: IDENTICAL on the NVIDIA or AMD column only"
    comptime assert RPT == GEMM_KPACK_RPT or RPT == TUNED_RPT, "identical_gemm_leaf_ptr_into: the wide or the stepped-down packed tile"
    if m<=0 or n<=0:
        return
    if k<=0 or leaf<=0 or p_count<=0 or (p_count-1)*leaf>=k or p_count*leaf<k:
        raise Error("identical_gemm_leaf_ptr_into: (leaf, p_count) does not partition k")
    if p_count>(1<<(FS-1)):
        raise Error("identical_gemm_leaf_ptr_into: more leaves than this kernel's fold stack covers")
    comptime NTH = TUNED_TPB
    comptime TR = NTH // TUNED_TC
    comptime BM = RPT * TR
    comptime BN = GEMM_KPACK_CPT * TUNED_TC
    # `_kpack_launch`'s page: TR A line groups of `KS RPT + PAD` words and TC B line groups of `KS CPT + PAD`.
    comptime PAGE_BYTES = (BM * GEMM_KPACK_KS + TR * GEMM_KPACK_PAD + BN * GEMM_KPACK_KS + TUNED_TC * GEMM_KPACK_PAD) * 4
    comptime PAGES = 1 if (NI06_ONE_PAGE or is_defined["MOJOLEARN_GEMM_ONE_PAGE"]()) else lib_smem_pages_for[
        TARGET_COLUMN, PAGE_BYTES + GEMM_KPACK_PAGE_GUARD_BYTES
    ]()
    comptime assert lib_smem_page_fits_for[TARGET_COLUMN, PAGE_BYTES](), (
        "identical_gemm_leaf_ptr_into: one packed page of this geometry exceeds the column's shared limit"
    )
    comptime kern = identical_gemm_kpack_kernel[
        RPT, GEMM_KPACK_CPT, TUNED_TC, GEMM_KPACK_KS, FS, PAGES, False, False,
        GEMM_KPACK_PAD, GEMM_KPACK_ALIGN, 0, True, True
    ]
    var g = _tile_grid(m,n,BM,BN,False)
    ctx.enqueue_function[kern](
        c,
        a,
        b,
        Int32(m),
        Int32(n),
        Int32(k),
        Int32(leaf),
        Int32(p_count),
        Int32(a_si),
        Int32(a_sp),
        Int32(b_sp),
        Int32(b_sj),
        Int32(0),
        Int32(0),
        grid_dim=(g[0], 1, 1),
        block_dim=(NTH, 1, 1),
    )


#: lane fx-leftovers-product (2026-11-19; `-D MOJOLEARN_GEMM_FAST_EPI=1`, default OFF, A/B owed): whether this build
#: fuses a sequence follower (bias, bias + activation, the activation derivative) into the packed store of
#: `identical_gemm_leaf_ptr_epi_into`. The packed body's columns, as `GEMM_LEAF_PTR_COLUMN`; a GEMM sabotage build keeps
#: the product and its follower launch (the epilogue is the clean body's store).
comptime GEMM_LEAF_PTR_EPI = GEMM_LEAF_PTR_COLUMN and is_defined["MOJOLEARN_GEMM_FAST_EPI"]() and not ANY_SABOTAGE


def identical_gemm_leaf_ptr_epi_into[RPT: Int,FS: Int,EPI: Int](
    ctx: DeviceContext,
    c: MutPointer[Float32,MutAnyOrigin],
    a: MutPointer[Float32,MutAnyOrigin],
    b: MutPointer[Float32,MutAnyOrigin],
    m: Int,n: Int,k: Int,leaf: Int,p_count: Int,
    a_si: Int,a_sp: Int,b_sp: Int,b_sj: Int,
    epi_p: MutPointer[Float32,MutAnyOrigin],
    epi_i: Int,
) raises:
    """`identical_gemm_leaf_ptr_into` with the follower in the product's store: ONE asynchronous all-leaves launch of
    `identical_gemm_kpack_epi_ptr_kernel[..., EPI]` where the plain route issues the product and then a follower
    launch over the same m n cells. `EPI` is `GEMM_EPI_PLUS_BIAS` (`epi_p` = the bias row, n words),
    `GEMM_EPI_BIAS_ACT` (the bias row; `epi_i` = the activation code) or `GEMM_EPI_ACT_BWD` (`epi_p` = the forward
    activations, m n words dense; `epi_i` = the activation code). The product's leaves, chain and fold are
    `identical_gemm_leaf_ptr_into`'s; the kinds and the unchanged-bits argument are beside `GEMM_EPI_BIAS_ACT` in
    gemm_identical.mojo. The same refusals; `epi_p` must not overlap `c` (a block stores C cells another block's
    thread could otherwise read as the operand). The page, the grid and the first twelve arguments MIRROR
    `identical_gemm_leaf_ptr_into` (and so `_kpack_launch`): a change there is made here too. Arm only: a build
    without `GEMM_LEAF_PTR_EPI` raises before any launch (the caller's rule answers 0 there and never calls)."""
    comptime assert RPT == GEMM_KPACK_RPT or RPT == TUNED_RPT, "identical_gemm_leaf_ptr_epi_into: the wide or the stepped-down packed tile"
    comptime assert EPI == GEMM_EPI_PLUS_BIAS or EPI == GEMM_EPI_BIAS_ACT or EPI == GEMM_EPI_ACT_BWD, (
        "identical_gemm_leaf_ptr_epi_into: a store epilogue kind that reads a second buffer"
    )
    comptime if not GEMM_LEAF_PTR_EPI:
        raise Error("identical_gemm_leaf_ptr_epi_into: this build has no fused packed store (GEMM_LEAF_PTR_EPI is False)")
    else:
        if m<=0 or n<=0:
            return
        if k<=0 or leaf<=0 or p_count<=0 or (p_count-1)*leaf>=k or p_count*leaf<k:
            raise Error("identical_gemm_leaf_ptr_epi_into: (leaf, p_count) does not partition k")
        if p_count>(1<<(FS-1)):
            raise Error("identical_gemm_leaf_ptr_epi_into: more leaves than this kernel's fold stack covers")
        comptime NTH = TUNED_TPB
        comptime TR = NTH // TUNED_TC
        comptime BM = RPT * TR
        comptime BN = GEMM_KPACK_CPT * TUNED_TC
        comptime PAGE_BYTES = (BM * GEMM_KPACK_KS + TR * GEMM_KPACK_PAD + BN * GEMM_KPACK_KS + TUNED_TC * GEMM_KPACK_PAD) * 4
        comptime PAGES = 1 if (NI06_ONE_PAGE or is_defined["MOJOLEARN_GEMM_ONE_PAGE"]()) else lib_smem_pages_for[
            TARGET_COLUMN, PAGE_BYTES + GEMM_KPACK_PAGE_GUARD_BYTES
        ]()
        comptime assert lib_smem_page_fits_for[TARGET_COLUMN, PAGE_BYTES](), (
            "identical_gemm_leaf_ptr_epi_into: one packed page of this geometry exceeds the column's shared limit"
        )
        # The epi-ptr kernel's comptime list: the single-product kernel's without GROUP, SAB and DIAG.
        comptime kern = identical_gemm_kpack_epi_ptr_kernel[
            RPT, GEMM_KPACK_CPT, TUNED_TC, GEMM_KPACK_KS, FS, PAGES,
            GEMM_KPACK_PAD, GEMM_KPACK_ALIGN, True, True, EPI
        ]
        var g = _tile_grid(m,n,BM,BN,False)
        ctx.enqueue_function[kern](
            c,
            a,
            b,
            Int32(m),
            Int32(n),
            Int32(k),
            Int32(leaf),
            Int32(p_count),
            Int32(a_si),
            Int32(a_sp),
            Int32(b_sp),
            Int32(b_sj),
            epi_p,
            Int32(epi_i),
            grid_dim=(g[0], 1, 1),
            block_dim=(NTH, 1, 1),
        )


def identical_gemm[allow_vendor: Bool = True,ROLE: Int = ROLE_PROJECTION](
    ctx: DeviceContext,mut c: DeviceBuffer[DType.float32],mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int,
) raises:
    comptime if not NEURAL_GEMM_EXPERIMENT_ENABLED:
        _incumbent_gemm[allow_vendor](ctx,c,a,b,m,n,k,op)
        return
    neural_validate(m,n,k,op)
    var workspace = ctx.enqueue_create_buffer[DType.float32](identical_gemm_workspace_max_floats(m,n,k))
    try:
        identical_gemm_into[allow_vendor,ROLE](ctx,c,a,b,workspace,m,n,k,op)
    except error:
        ctx.synchronize()
        raise error
    ctx.synchronize()
    _ = workspace^


# lane fx-gemm-epi (2026-11-18, -D MOJOLEARN_GEMM_FAST_EPI=1, default OFF, OPEN: no A/B yet): the store epilogue.
# A product joined to the value already in its destination, in the product's own final store
# (gemm/checks/gemm_epilogue.mojo; kinds and the unchanged-bits argument beside gemm_epi_cell in gemm_identical.mojo).
# Two free functions, so a caller decides where its operands go before any launch and GemmWorkspace gains no method.
def gemm_workspace_epi_takes(m: Int,n: Int,k: Int) -> Bool:
    """Whether gemm_workspace_run_epi fuses the join at (m, n, k) on this build. False without the define, on a
    column or plan with no fused store, and under a changed neural profile (its kernel has no epilogue): the caller
    then runs GemmWorkspace.run and its follower launch, unchanged."""
    comptime if NEURAL_GEMM_EXPERIMENT_ENABLED:
        return False
    return _incumbent_epi_takes(m,n,k)


def gemm_workspace_run_epi[EPI: Int](ctx: DeviceContext,mut workspace: GemmWorkspace,
    mut c: DeviceBuffer[DType.float32],mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int) raises:
    """GemmWorkspace.run with the join in the store: c = EPI(op(A) . op(B), c), EPI 1 = ftz(ftz(c) + ftz(cell)),
    2 = ftz(ftz(cell) + ftz(c)). Call only where gemm_workspace_epi_takes(m, n, k) is True (raises otherwise, before
    any launch). Same workspace sizing and the same asynchronous contract as run."""
    workspace._ensure(ctx,identical_gemm_workspace_max_floats(m,n,k))
    _incumbent_epi_into[EPI](ctx,c,a,b,workspace.buffer,m,n,k,op)


# TOMBSTONE NN05 (MOJOLEARN_IDN_NEURAL_NN05, neural_gemm_pair=nn05, deleted
# 2026-10-08 by lane grid-act-4): _neural_pair_kernel ran the gate and up
# projections as one kernel, one A load feeding both contractions, same leaf
# chains and fold. IDENTICAL grid ge123e6f9 nn05/off NV/AMD: lm-forward
# 2.19x/2.80x, lm-train-step 3.72x/2.44x, transformer-forward 3.85x/3.47x
# SLOWER (one thread per cell loses the tiled GEMM); bits same. Recoverable at
# main 4e3da4282.


# TOMBSTONE: MOJOLEARN_IDN_GEMM_FILL_SPLITK (noise/loser: mamba2 NV 1.248, transformer NV 1.39) deleted 2026-10-11 by 026a811d5; code recoverable at 026a811d5^.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_GEMM_FILL_SPLITK.patch
struct GemmWorkspace(Movable):
    """One in-order model context; no mutable global plan or operand cache."""
    var buffer: DeviceBuffer[DType.float32]
    # lane/fx-gemm-worklist: the pending work list and its device tables (empty unless the arm is compiled).
    var worklist: GemmWorklist

    def __init__(out self,ctx: DeviceContext) raises:
        step_count_device_alloc()
        self.buffer = ctx.enqueue_create_buffer[DType.float32](1)
        self.worklist = GemmWorklist()

    def _ensure(mut self,ctx: DeviceContext,required: Int) raises:
        if required>len(self.buffer):
            step_count_sync()
            ctx.synchronize()
            step_count_device_alloc()
            self.buffer = ctx.enqueue_create_buffer[DType.float32](required)

    def run[allow_vendor: Bool = True,ROLE: Int = ROLE_PROJECTION](mut self,ctx: DeviceContext,
        mut c: DeviceBuffer[DType.float32],mut a: DeviceBuffer[DType.float32],
        mut b: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int) raises:
        comptime if NEURAL_GEMM_EXPERIMENT_ENABLED:
            neural_validate(m,n,k,op)
        var required = identical_gemm_workspace_max_floats(m,n,k)
        # lane fx-fast-matrix-units (arm only): room for the transpose of A of an OP_TN the matrix unit takes.
        comptime if GEMM_FAST_MATRIX_UNITS and (allow_vendor or GEMM_FAST_MATRIX_UNITS_ALL):
            required = max(required,matrix_units_workspace_floats(m,n,k,op))
        self._ensure(ctx,required)
        identical_gemm_into[allow_vendor,ROLE](ctx,c,a,b,self.buffer,m,n,k,op)

    # ---- THE WORK LIST (lane/fx-gemm-worklist; arm MOJOLEARN_GEMM_FAST_WORKLIST, default off) ----
    # list_begin(); list_add(...) once per product; list_run(ctx). The products of one list must be INDEPENDENT:
    # none reads or writes another's output (their inputs are all ready before the first list_add). With the arm
    # off list_add IS run, in the caller's order, and the other two calls compile to nothing. With it on, every
    # product whose own launch is the packed body shares one launch per mode with the others (gemm_identical.mojo,
    # "THE WORK LIST": same blocks, same body, same words); any other product runs its own launch inside list_add.
    # Lifetime: list_add records device addresses. The caller keeps every c, a and b alive past the synchronize that
    # follows list_run (name them after list_run, or hold them in a struct the caller keeps), as for run.

    def list_begin(mut self):
        """Start a work list, dropping any product still pending."""
        comptime if NEURAL_GEMM_WORKLIST:
            self.worklist.begin()

    def list_add[allow_vendor: Bool = True,ROLE: Int = ROLE_PROJECTION](mut self,ctx: DeviceContext,
        mut c: DeviceBuffer[DType.float32],mut a: DeviceBuffer[DType.float32],
        mut b: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int) raises:
        """`run`'s product, queued for `list_run`. A refusal is this product's own, raised here with the wording
        its single launch uses; the products queued before it are launched first, as they would have been."""
        # lane fx-fast-matrix-units (arm only): a product the matrix unit takes is its own launch, never a list entry
        # (the list's shared launch is the FP32 packed body). List products are independent, so its place is free.
        comptime if GEMM_FAST_MATRIX_UNITS and (allow_vendor or GEMM_FAST_MATRIX_UNITS_ALL):
            if matrix_units_takes(m,n,k):
                self.run[allow_vendor,ROLE](ctx,c,a,b,m,n,k,op)
                return
        comptime if not NEURAL_GEMM_WORKLIST:
            self.run[allow_vendor,ROLE](ctx,c,a,b,m,n,k,op)
            return
        if self.worklist.full():
            self.list_run(ctx)
        var taken = False
        try:
            taken = self.worklist.add(ctx,c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),m,n,k,op)
        except error:
            self.list_run(ctx)
            raise error
        if not taken:
            self.run[allow_vendor,ROLE](ctx,c,a,b,m,n,k,op)

    def list_run(mut self,ctx: DeviceContext) raises:
        """Launch the pending products (asynchronous, as `run`) and clear the list."""
        comptime if NEURAL_GEMM_WORKLIST:
            var need = self.worklist.node_floats()
            if need>0:
                self._ensure(ctx,need)
            var nodes = self.buffer.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            self.worklist.launch(ctx,nodes)

    def close(mut self,ctx: DeviceContext) raises:
        ctx.synchronize()
        self.buffer = ctx.enqueue_create_buffer[DType.float32](1)
