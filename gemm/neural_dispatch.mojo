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


def identical_gemm_into[allow_vendor: Bool = True,ROLE: Int = ROLE_PROJECTION](
    ctx: DeviceContext,mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int,
) raises:
    """ROLE is the caller class (ROLE_PROJECTION, ROLE_HEAD, ROLE_WGRAD); it
    never changes the route."""
    comptime assert ROLE == ROLE_PROJECTION or ROLE == ROLE_HEAD or ROLE == ROLE_WGRAD, "neural GEMM role"
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
        self._ensure(ctx,required)
        identical_gemm_into[allow_vendor,ROLE](ctx,c,a,b,self.buffer,m,n,k,op)

    # ---- THE WORK LIST (lane/fx-gemm-worklist; arm MOJOLEARN_GEMM_FAST_WORKLIST, default off) ----
    # list_begin(); list_add(...) once per product; list_run(ctx). The products of one list must be INDEPENDENT:
    # none reads or writes another's output (their inputs are all ready before the first list_add). With the arm
    # off list_add IS run, in the caller's order, and the other two calls compile to nothing. With it on, every
    # product whose own launch is the packed body shares one launch per mode with the others (gemm_identical.mojo,
    # "THE WORK LIST": same blocks, same body, same words); any other product runs its own launch inside list_add.
    # Lifetime: list_add records pointers. The caller keeps every c, a and b alive past the synchronize that
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
