# SPDX-License-Identifier: Apache-2.0
"""Neural-only model dispatcher for the NN01--NN16 A/B source experiments.

Existing signatures, unchanged OFF route, one selected host/device numerical
profile for every routed forward/backward product. Classical code retains its
original imports. Allocation, staging and synchronization remain in the model
operation; no benchmark controller prepares hidden runtime work. Unverified.
"""
from gemm.experiments.neural_profile_device import neural_profile_device
from std.sys.compile import is_defined,get_defined_int
from std.gpu import block_idx,block_dim,thread_idx
from max.gpu.host import DeviceContext,DeviceBuffer
from max.gpu.host.device_attribute import DeviceAttribute
from checks.kernel_matrix import TARGET_COLUMN,COLUMN_NVIDIA,COLUMN_AMD
from checks.numerics import GLOBAL_NUMERIC_MODE,NUMERIC_IDENTICAL
from checks.numerics import ftz
from checks.rtf_seam import rtf_mul_add
from core.step_phase import step_count_device_alloc,step_count_sync
from gemm.contract import OP_NN,OP_NT,OP_TN
from gemm.checks.gemm_identical import (
    identical_gemm_into as _incumbent_into,
    identical_gemm as _incumbent_gemm,
    identical_gemm_workspace_max_floats as _incumbent_workspace,
    identical_gemm_with_plan,identical_gemm_workspace_floats,choose_gemm_plan,
    GEMM_ARM_KPACK,PLAN_FLAT,ANY_SABOTAGE,
    PLAN_TUNED_128_8X8,GEMM_IDENTICAL_MFMA,GEMM_REUSE_GROUP_WS,
    GEMM_KPACK_RPT,GEMM_KPACK_CPT,GEMM_KPACK_KS,GEMM_KPACK_FS,GEMM_KPACK_PAD,GEMM_KPACK_ALIGN,TUNED_TC,
    contract_partition,gemm_operand_strides,gemm_amd_short_k_tuned,gemm_default_ksplit_leaves,
    gemm_step_ksplit_finest_leaves,_mfma_group_leaves,_mfma_run_ws,_kpack_launch,
    _ksplit_resolve_leaves,_ksplit_fold_launch,
)
from gemm.experiments.neural_profile import (
    NEURAL_EXPERIMENTS_ALLOWED,NEURAL_PROFILE_CHANGED,NEURAL_LEAF,NEURAL_CHAINS,
    neural_partition,neural_strides,neural_validate,
    neural_merge_chains,neural_fold_push,neural_fold_drain,
)
from gemm.experiments.neural_streaming import (
    NN02,NN11,NN16,neural_streaming_ab,neural_streaming_floats,neural_fold_capacity_ab,
)
from gemm.experiments.neural_tiled import NN09,NN15,neural_tiled_ab
from gemm.experiments.neural_plans import NN01,NN10,NN12,neural_schedule_ab,neural_cost_plan
from gemm.experiments.neural_grouped import NN07,_neural_operand_pack_kernel
from gemm.experiments.neural_switches import (
    ROLE_PROJECTION,ROLE_HEAD,ROLE_WGRAD,ROLE_ALL,NEURAL_GEMM_ROLES,
)
# TOMBSTONE: MOJOLEARN_IDN_NEURAL_GEMM_OZAKI_SLICES (NN-OZ int8 Ozaki-split neural GEMM; DELETE-slower: grid g50ebe26a5, slices 4/5
# on 10 neural workloads, e.g. lm-train-step NV L40S 35.3 -> 3172.6 / 3709.4 ms, AMD MI325X 42.7 -> 334.3 / 407.5 ms; cnn-clf avg
# 25.9x / 30.1x; mlp-train-step 1.20x / 1.22x, the closest) deleted 2026-10-11 by lane/grid-act-16; code recoverable at 7aa609453.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_OZAKI_SLICES.patch; record in docs/TOMBSTONES.md.

# NN05 (the fused pair kernel arm) deleted 2026-10-08 (lane grid-act-4, grid
# ge123e6f9, 2.2-3.9x slower); see the tombstone at _neural_pair_kernel's old site.
comptime NEURAL_PAIR_ENABLED = NN07
comptime NEURAL_GEMM_EXPERIMENT_ENABLED = (
    NEURAL_PROFILE_CHANGED or NN01 or NN02 or NN09 or NN10 or NN11 or NN12 or NN15 or NN16
)
comptime _STREAM_GROUP = get_defined_int["MOJOLEARN_IDN_NEURAL_STREAM_GROUP",8]()
comptime _DEPTH = get_defined_int["MOJOLEARN_IDN_NEURAL_STAGE_DEPTH",2]()
comptime _PAD = get_defined_int["MOJOLEARN_IDN_NEURAL_STAGE_PAD",0]()
comptime _SWIZZLE = is_defined["MOJOLEARN_IDN_NEURAL_STAGE_SWIZZLE"]()
comptime _ARM = get_defined_int["MOJOLEARN_IDN_NEURAL_GEMM_ARM",GEMM_ARM_KPACK]()
comptime _FILL_BLOCKS = get_defined_int["MOJOLEARN_IDN_NEURAL_FILL_BLOCKS",1]()
comptime _RETAINED = get_defined_int["MOJOLEARN_IDN_NEURAL_RETAINED_FLOATS",16777216]()
# Resource budget, not a matrix-size route fit: 64 MiB retained per owned
# workspace, with exact-shaped temporary allocation above that memory budget.
comptime _CONTROL = is_defined["MOJOLEARN_IDN_NEURAL_GEMM_CONTROL"]()
comptime _PAIR_CONTROL = is_defined["MOJOLEARN_IDN_NEURAL_PAIR_CONTROL"]()
comptime NEURAL_WORKSPACE_REUSE_ENABLED = NN12 and not _CONTROL
comptime _STREAM = NN02
comptime _STAGING = NN09 or NN15
# NN11 has two real placements: global fold slots inside NN02, or a
# standalone register/shared fold kernel. Other schedule families do not
# implement its storage arm; reject those combinations instead of masking it.
comptime _ROUTES = Int(NN01)+Int(NN10)+Int(_STREAM)+Int(_STAGING)+Int(NN11 and not _STREAM)
# One arm of MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE is selected. Products whose
# caller role is outside MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE_ROLES keep the
# unscheduled route (selected profile, else the incumbent) in the same build.
comptime _SCHEDULED = _ROUTES > 0
comptime _ROLE_SPLIT = _SCHEDULED and NEURAL_GEMM_ROLES != ROLE_ALL


def _overlaps(mut a: DeviceBuffer[DType.float32],na: Int,
              mut b: DeviceBuffer[DType.float32],nb: Int) -> Bool:
    if na<=0 or nb<=0:
        return False
    var ap = Int(a.unsafe_ptr())
    var bp = Int(b.unsafe_ptr())
    return ap<bp+4*nb and bp<ap+4*na


def _unscheduled_workspace_max_floats(m: Int,n: Int,k: Int) -> Int:
    comptime if NEURAL_PROFILE_CHANGED:
        return 1
    return _incumbent_workspace(m,n,k)


def identical_gemm_workspace_max_floats(m: Int,n: Int,k: Int) -> Int:
    """Covers every route the build can take for (m, n, k): with a role mask,
    one workspace serves both the scheduled and the unscheduled roles."""
    var count = _scheduled_workspace_max_floats(m,n,k)
    comptime if _ROLE_SPLIT:
        count = max(count,_unscheduled_workspace_max_floats(m,n,k))
    return count


def _scheduled_workspace_max_floats(m: Int,n: Int,k: Int) -> Int:
    comptime if _STREAM:
        var leaves = neural_partition[NEURAL_LEAF](k)[1]
        comptime if NN02 and not _CONTROL:
            return neural_streaming_floats[_STREAM_GROUP,True](k,m*n)
        return max(1,m*n*leaves)
    comptime if NEURAL_PROFILE_CHANGED or _STAGING or NN11:
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


def _profile_or_cost_into(ctx: DeviceContext,mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int,plan: Int) raises:
    comptime if NEURAL_PROFILE_CHANGED:
        # New-profile cost arm compares actual shared-tile reuse with scalar
        # loads. Existing v1 native geometries cannot define a different sum.
        if plan == PLAN_FLAT:
            neural_profile_device[NEURAL_LEAF,NEURAL_CHAINS](ctx,c,a,b,m,n,k,op)
        else:
            neural_tiled_ab[False](ctx,c,a,b,m,n,k,op)
    else:
        identical_gemm_with_plan(ctx,c,a,b,ws,m,n,k,op,plan)


def _selected_cost_plan(m: Int,n: Int,k: Int) raises -> Int:
    comptime if not NEURAL_PROFILE_CHANGED:
        return neural_cost_plan(m,n,k,_FILL_BLOCKS)
    if m<=0 or n<=0:
        return PLAN_FLAT
    # The changed-profile candidate currently has a real 8x16 shared tile,
    # not the incumbent 32/64/128 bodies. Price precisely the available tile
    # instead of borrowing the old body's more favorable reuse estimate.
    var blocks = (m*n+127)//128
    var flat = Float64(2)*Float64(m*n)*Float64(max(k,1))*Float64(max(_FILL_BLOCKS,blocks))/Float64(blocks)
    var tiles = ((m+7)//8)*((n+15)//16)
    var staged = Float64(tiles*24+tiles*128-m*n)*Float64(max(k,1))*Float64(max(_FILL_BLOCKS,tiles))/Float64(tiles)
    # Any non-FLAT tag selects the explicit selected-profile 8x16 launcher.
    return 1 if staged<flat else PLAN_FLAT


def identical_gemm_into[allow_vendor: Bool = True,ROLE: Int = ROLE_PROJECTION](
    ctx: DeviceContext,mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int,
) raises:
    """ROLE is the caller class (ROLE_PROJECTION, ROLE_HEAD, ROLE_WGRAD) that
    MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE_ROLES selects on; it never changes the
    route of a build without a schedule arm."""
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
    comptime if _SCHEDULED and not ((NEURAL_GEMM_ROLES & ROLE) != 0):
        comptime if NEURAL_PROFILE_CHANGED:
            neural_profile_device[NEURAL_LEAF,NEURAL_CHAINS](ctx,c,a,b,m,n,k,op)
        else:
            _incumbent_into[allow_vendor](ctx,c,a,b,ws,m,n,k,op)
        return
    comptime if NN01:
        _ = neural_schedule_ab[not _CONTROL](ctx,c,a,b,ws,m,n,k,op,_ARM)
    elif _STREAM:
        neural_streaming_ab[not _CONTROL,_STREAM_GROUP,not _CONTROL,not _CONTROL](ctx,c,a,b,ws,m,n,k,op)
    elif _STAGING:
        neural_tiled_ab[not _CONTROL,_DEPTH,_PAD,_SWIZZLE,NN15](ctx,c,a,b,m,n,k,op)
    elif NN10:
        var plan = _selected_cost_plan(m,n,k)
        comptime if _CONTROL:
            plan = choose_gemm_plan(m,n,k)
        _profile_or_cost_into(ctx,c,a,b,ws,m,n,k,op,plan)
    elif NN11:
        neural_fold_capacity_ab[not _CONTROL,is_defined["MOJOLEARN_IDN_NEURAL_FOLD_SHARED"]()](ctx,c,a,b,m,n,k,op)
    elif NEURAL_PROFILE_CHANGED:
        neural_profile_device[NEURAL_LEAF,NEURAL_CHAINS](ctx,c,a,b,m,n,k,op)
    else:
        _incumbent_into[allow_vendor](ctx,c,a,b,ws,m,n,k,op)


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
    workspace, so none is taken. First caller: the MLP training step
    (training/mlp_ops.mojo::MLP_GEMM_LEAF128)."""
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


def identical_gemm_pair_workspace_max_floats(m: Int,n: Int,k: Int) -> Int:
    var count = identical_gemm_workspace_max_floats(m,n,k)
    comptime if NN07:
        count += m*k
    return max(1,count)


def identical_gemm_pair_into(ctx: DeviceContext,
    mut c1: DeviceBuffer[DType.float32],mut c2: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],mut b1: DeviceBuffer[DType.float32],
    mut b2: DeviceBuffer[DType.float32],mut ws: DeviceBuffer[DType.float32],
    m: Int,n: Int,k: Int,op: Int) raises:
    neural_validate(m,n,k,op)
    if len(c1)<m*n or len(c2)<m*n or len(a)<m*k or len(b1)<n*k or len(b2)<n*k:
        raise Error("neural pair operand storage too short")
    if len(ws)<identical_gemm_pair_workspace_max_floats(m,n,k):
        raise Error("neural pair workspace too short")
    if m==0 or n==0:
        return
    if (_overlaps(c1,m*n,c2,m*n) or _overlaps(c1,m*n,a,m*k)
        or _overlaps(c2,m*n,a,m*k) or _overlaps(c1,m*n,b1,n*k)
        or _overlaps(c1,m*n,b2,n*k) or _overlaps(c2,m*n,b1,n*k)
        or _overlaps(c2,m*n,b2,n*k) or _overlaps(ws,len(ws),a,m*k)
        or _overlaps(ws,len(ws),b1,n*k) or _overlaps(ws,len(ws),b2,n*k)
        or _overlaps(ws,len(ws),c1,m*n) or _overlaps(ws,len(ws),c2,m*n)):
        raise Error("neural pair outputs/workspace must be disjoint from live operands")
    var actual_op = op
    var input = DeviceBuffer[DType.float32](ctx,a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),len(a),owning=False)
    var scratch_offset = 0
    var st = neural_strides(op,m,n,k)
    comptime if NN07:
        if k>0:
            input = DeviceBuffer[DType.float32](ctx,ws.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),m*k,owning=False)
            ctx.enqueue_function[_neural_operand_pack_kernel](input,a,Int32(m),Int32(k),Int32(st[0]),Int32(st[1]),
                grid_dim=((m*k+127)//128,1,1),block_dim=(128,1,1))
            scratch_offset = m*k
            if op==OP_TN:
                actual_op = OP_NN
    var scratch = DeviceBuffer[DType.float32](ctx,(ws.unsafe_ptr()+scratch_offset).unsafe_origin_cast[MutAnyOrigin](),
        len(ws)-scratch_offset,owning=False)
    # NN05's fused pair kernel arm (deleted 2026-10-08, see the tombstone above)
    # ran here; both projections are the identical GEMM.
    identical_gemm_into(ctx,c1,input,b1,scratch,m,n,k,actual_op)
    comptime if NN07 and _PAIR_CONTROL:
        if k>0:
            # B repeats the same producer; A shares its completed stage.
            # Both include packing in the whole model operation.
            ctx.enqueue_function[_neural_operand_pack_kernel](input,a,Int32(m),Int32(k),Int32(st[0]),Int32(st[1]),
                grid_dim=((m*k+127)//128,1,1),block_dim=(128,1,1))
    identical_gemm_into(ctx,c2,input,b2,scratch,m,n,k,actual_op)
    _ = input^
    _ = scratch^


# ---------------------------------------------------------------------------
# H2 (2026-10-11 gap program, lane gemm-fill-splitk): MOJOLEARN_IDN_GEMM_FILL_SPLITK, default OFF.
# STRUCTURAL REASON: a TUNED 128x128 call with few output tiles leaves most compute units idle
# (e.g. 48-128 tiles on the L40S's 142 SMs / the MI325X's 304 CUs). The shipped rules split k only
# below fixed rows (NVIDIA S=132 blocks; AMD the leaf split raised to 4-leaf groups, so k <= 512
# never splits). This arm reads the compute-unit count from the device (a hardware count, never a
# shape) and, when tiles < 2 x units, splits the contract's leaves over the grid's y axis: the
# coarsest power-of-two group that still issues >= 2 x units blocks. Groups are powers of two
# aligned at leaf 0 and the second launch is the existing contract fold (`_ksplit_fold_launch`),
# so the leaves and the fold DAG are the contract's: NO BIT CHANGES on any column (the same
# group/fold kernels the shipped grouped plan runs). Only block assignment changes, never the leaf
# size (unlike the deleted GEMM_LEAF128). A call with one leaf, or one resulting group, or that the
# shipped rule already splits at least as finely, keeps the shipped dispatch. The node partials go
# in the model workspace (grown once by `run`), so no allocation and no wait.
# EXPECTED: under-filled projections ~1/2-1/3 the time; lm-train-step NV 35 -> ~27 ms, AMD 43 -> ~30.
# Revert: build without the define.
comptime IDN_GEMM_FILL_SPLITK = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_IDN_GEMM_FILL_SPLITK"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    and not ANY_SABOTAGE
    and (TARGET_COLUMN == COLUMN_NVIDIA or TARGET_COLUMN == COLUMN_AMD)
)


def _fill_splitk_leaves(m: Int,n: Int,k: Int,units: Int) -> Int:
    """Leaves per group for the fill split, or 0 to keep the shipped dispatch.
    Target: 2 blocks per compute unit (one to run, one to hide the tail)."""
    if units<=0 or m<=0 or n<=0:
        return 0
    if choose_gemm_plan(m,n,k)!=PLAN_TUNED_128_8X8:
        return 0
    comptime if TARGET_COLUMN==COLUMN_AMD:
        # The short-k band forces its own plan before the TUNED dispatch.
        if gemm_amd_short_k_tuned(m,n,k):
            return 0
    var p_count = contract_partition(k)[1]
    if p_count<2:
        return 0
    var target = 2*units
    var tiles = ((m+127)//128)*((n+127)//128)
    if tiles>=target:
        return 0
    var gl = gemm_step_ksplit_finest_leaves(m,n,k)
    if gl<1 or (p_count+gl-1)//gl<2:
        return 0
    while (p_count+2*gl-1)//(2*gl)>=2 and tiles*((p_count+2*gl-1)//(2*gl))>=target:
        gl = gl*2
    var shipped: Int
    comptime if GEMM_IDENTICAL_MFMA:
        shipped = _mfma_group_leaves(m,n,k)
    else:
        shipped = gemm_default_ksplit_leaves(m,n,k)
    if shipped>0 and shipped<=gl:
        return 0
    return gl


def _fill_splitk_floats(m: Int,n: Int,k: Int,units: Int) -> Int:
    """Node workspace the fill split needs at this call (0 when it declines)."""
    var gl = _fill_splitk_leaves(m,n,k,units)
    if gl<=0:
        return 0
    var p_count = contract_partition(k)[1]
    return m*n*((p_count+gl-1)//gl)


def _fill_splitk_into(ctx: DeviceContext,mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int,units: Int) raises -> Bool:
    """Runs the fill split and answers True, or answers False (nothing launched)."""
    var gl = _fill_splitk_leaves(m,n,k,units)
    if gl<=0:
        return False
    var part = contract_partition(k)
    var rg = _ksplit_resolve_leaves(gl,part[1])
    if len(ws)<m*n*rg[1]:
        return False
    comptime if GEMM_IDENTICAL_MFMA:
        # AMD matrix-core group launch + fold into the caller's workspace (asynchronous).
        _mfma_run_ws(ctx,c,a,b,ws,m,n,k,op,gl)
        return True
    else:
        comptime if not GEMM_REUSE_GROUP_WS:
            return False
        # The shipped packed (kpack-hg) group launch, as `_kpack_hg_run_with_ws` issues it,
        # at the fill group size; FS is the profile-wide stack (local storage only, same DAG).
        _kpack_launch[
            GEMM_KPACK_RPT,GEMM_KPACK_CPT,TUNED_TC,
            GEMM_KPACK_KS,GEMM_KPACK_FS,True,False,
            GEMM_KPACK_PAD,GEMM_KPACK_ALIGN,0,True,True,
        ](ctx,ws,a,b,m,n,k,part[0],part[1],gemm_operand_strides(op,m,n,k),rg[0],rg[1])
        _ksplit_fold_launch(ctx,c,ws,m,n,rg[1])
        return True


struct GemmWorkspace(Movable):
    """One in-order model context; no mutable global plan or operand cache."""
    var buffer: DeviceBuffer[DType.float32]
    var last_m: Int
    var last_n: Int
    var last_k: Int
    var cached_plan: Int
    var fill_units: Int

    def __init__(out self,ctx: DeviceContext) raises:
        step_count_device_alloc()
        self.buffer = ctx.enqueue_create_buffer[DType.float32](1)
        self.last_m = -1
        self.last_n = -1
        self.last_k = -1
        self.cached_plan = PLAN_FLAT
        self.fill_units = 0
        comptime if IDN_GEMM_FILL_SPLITK:
            # H2: compute units read once per model context (hardware count, not a shape).
            self.fill_units = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)

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
        comptime if NN12 and _CONTROL:
            # B charges fresh scratch and its last-consumer wait per product.
            # OFF keeps the pre-existing model workspace reuse unchanged.
            identical_gemm[allow_vendor,ROLE](ctx,c,a,b,m,n,k,op)
            return
        var required = identical_gemm_workspace_max_floats(m,n,k)
        comptime if NN12 and not _CONTROL:
            if required>_RETAINED:
                identical_gemm[allow_vendor,ROLE](ctx,c,a,b,m,n,k,op)
                return
        self._ensure(ctx,required)
        comptime if NN12 and NN10 and not _CONTROL and ((NEURAL_GEMM_ROLES & ROLE) != 0):
            _admit_product(c,a,b,self.buffer,m,n,k,op)
            if m==0 or n==0:
                return
            if self.last_m!=m or self.last_n!=n or self.last_k!=k:
                self.cached_plan = _selected_cost_plan(m,n,k)
                self.last_m = m
                self.last_n = n
                self.last_k = k
            _profile_or_cost_into(ctx,c,a,b,self.buffer,m,n,k,op,self.cached_plan)
        else:
            comptime if IDN_GEMM_FILL_SPLITK:
                var fill = _fill_splitk_floats(m,n,k,self.fill_units)
                if fill>0:
                    self._ensure(ctx,fill)
                    if _fill_splitk_into(ctx,c,a,b,self.buffer,m,n,k,op,self.fill_units):
                        return
            identical_gemm_into[allow_vendor,ROLE](ctx,c,a,b,self.buffer,m,n,k,op)

    def run_pair(mut self,ctx: DeviceContext,mut c1: DeviceBuffer[DType.float32],
        mut c2: DeviceBuffer[DType.float32],mut a: DeviceBuffer[DType.float32],
        mut b1: DeviceBuffer[DType.float32],mut b2: DeviceBuffer[DType.float32],
        m: Int,n: Int,k: Int,op: Int) raises:
        neural_validate(m,n,k,op)
        var required = identical_gemm_pair_workspace_max_floats(m,n,k)
        comptime if NN12 and _CONTROL:
            var temporary = ctx.enqueue_create_buffer[DType.float32](required)
            try:
                identical_gemm_pair_into(ctx,c1,c2,a,b1,b2,temporary,m,n,k,op)
            except error:
                ctx.synchronize()
                raise error
            ctx.synchronize()
            _ = temporary^
            return
        comptime if NN12 and not _CONTROL:
            if required>_RETAINED:
                var temporary = ctx.enqueue_create_buffer[DType.float32](required)
                try:
                    identical_gemm_pair_into(ctx,c1,c2,a,b1,b2,temporary,m,n,k,op)
                except error:
                    ctx.synchronize()
                    raise error
                ctx.synchronize()
                _ = temporary^
                return
        self._ensure(ctx,required)
        identical_gemm_pair_into(ctx,c1,c2,a,b1,b2,self.buffer,m,n,k,op)

    def close(mut self,ctx: DeviceContext) raises:
        ctx.synchronize()
        self.buffer = ctx.enqueue_create_buffer[DType.float32](1)
        self.last_m = -1
