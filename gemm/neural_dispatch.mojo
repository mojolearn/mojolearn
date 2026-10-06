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
from checks.kernel_matrix import TARGET_COLUMN,COLUMN_NVIDIA
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
from gemm.experiments.neural_plans import NN01,NN08,NN10,NN12,neural_schedule_ab,neural_async_ab,neural_cost_plan
from gemm.experiments.neural_grouped import NN05,NN07,_neural_operand_pack_kernel

comptime NEURAL_PAIR_ENABLED = NN05 or NN07
comptime NEURAL_GEMM_EXPERIMENT_ENABLED = (
    NEURAL_PROFILE_CHANGED or NN01 or NN02 or NN08 or NN09 or NN10 or NN11 or NN12 or NN15 or NN16
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
comptime _ROUTES = Int(NN01)+Int(NN08)+Int(NN10)+Int(_STREAM)+Int(_STAGING)
comptime assert _ROUTES <= 1, "select one GEMM schedule family per A/B; NN11/NN12/profile/pair controls compose"
comptime assert not (NN01 and NEURAL_PROFILE_CHANGED), "legacy NN01 instruction geometry is v1-only; do not silently mix arithmetic profiles"
comptime assert not NN10 or is_defined["MOJOLEARN_IDN_NEURAL_FILL_BLOCKS"](), "NN10 requires a recorded device-fill budget, never guessed hardware"
comptime assert _STREAM_GROUP>0 and _RETAINED>0, "positive neural resource budgets"


def _overlaps(mut a: DeviceBuffer[DType.float32],na: Int,
              mut b: DeviceBuffer[DType.float32],nb: Int) -> Bool:
    if na<=0 or nb<=0:
        return False
    var ap = Int(a.unsafe_ptr())
    var bp = Int(b.unsafe_ptr())
    return ap<bp+4*nb and bp<ap+4*na


def identical_gemm_workspace_max_floats(m: Int,n: Int,k: Int) -> Int:
    comptime if _STREAM:
        var leaves = neural_partition[NEURAL_LEAF](k)[1]
        comptime if NN02 and not _CONTROL:
            return neural_streaming_floats[_STREAM_GROUP,True](k,m*n)
        return max(1,m*n*leaves)
    comptime if NEURAL_PROFILE_CHANGED or _STAGING or NN08 or NN11:
        return 1
    return _incumbent_workspace(m,n,k)


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


def identical_gemm_into[allow_vendor: Bool = True](
    ctx: DeviceContext,mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int,
) raises:
    comptime if not NEURAL_GEMM_EXPERIMENT_ENABLED:
        _incumbent_into[allow_vendor](ctx,c,a,b,ws,m,n,k,op)
        return
    neural_validate(m,n,k,op)
    if len(c)<m*n or len(a)<m*k or len(b)<n*k or len(ws)<identical_gemm_workspace_max_floats(m,n,k):
        raise Error("neural GEMM operand/workspace storage too short")
    if m==0 or n==0:
        return
    if (_overlaps(c,m*n,a,m*k) or _overlaps(c,m*n,b,n*k)
        or _overlaps(ws,len(ws),c,m*n) or _overlaps(ws,len(ws),a,m*k)
        or _overlaps(ws,len(ws),b,n*k)):
        raise Error("neural GEMM output/workspace must not overlap inputs or each other")
    comptime if NN01:
        _ = neural_schedule_ab[not _CONTROL](ctx,c,a,b,ws,m,n,k,op,_ARM)
    elif _STREAM:
        neural_streaming_ab[not _CONTROL,_STREAM_GROUP,not _CONTROL,not _CONTROL](ctx,c,a,b,ws,m,n,k,op)
    elif _STAGING:
        neural_tiled_ab[not _CONTROL,_DEPTH,_PAD,_SWIZZLE,NN15](ctx,c,a,b,m,n,k,op)
    elif NN08:
        if k==0:
            neural_profile_device[NEURAL_LEAF,NEURAL_CHAINS](ctx,c,a,b,m,n,k,op)
        else:
            # Async is a physical NVIDIA schedule. Other columns use the
            # synchronous exact graph; their timing is still whole-model.
            comptime if TARGET_COLUMN == COLUMN_NVIDIA:
                neural_async_ab[not _CONTROL](ctx,c,a,b,m,n,k,op)
            else:
                neural_async_ab[False](ctx,c,a,b,m,n,k,op)
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


def identical_gemm[allow_vendor: Bool = True](
    ctx: DeviceContext,mut c: DeviceBuffer[DType.float32],mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int,
) raises:
    comptime if not NEURAL_GEMM_EXPERIMENT_ENABLED:
        _incumbent_gemm[allow_vendor](ctx,c,a,b,m,n,k,op)
        return
    var workspace = ctx.enqueue_create_buffer[DType.float32](identical_gemm_workspace_max_floats(m,n,k))
    try:
        identical_gemm_into[allow_vendor](ctx,c,a,b,workspace,m,n,k,op)
    except error:
        ctx.synchronize()
        raise error
    ctx.synchronize()
    _ = workspace^


def _neural_pair_kernel(
    c1: MutPointer[Float32,MutAnyOrigin],c2: MutPointer[Float32,MutAnyOrigin],
    a: MutPointer[Float32,MutAnyOrigin],b1: MutPointer[Float32,MutAnyOrigin],
    b2: MutPointer[Float32,MutAnyOrigin],m: Int32,n: Int32,k: Int32,
    leaf: Int32,leaves: Int32,asi: Int32,asp: Int32,bsp: Int32,bsj: Int32,
):
    var cell = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell>=Int(m)*Int(n):
        return
    var row = cell//Int(n)
    var col = cell%Int(n)
    var stack1 = SIMD[DType.float32,16](0.0)
    var stack2 = SIMD[DType.float32,16](0.0)
    var occupied1 = 0
    var occupied2 = 0
    for t in range(Int(leaves)):
        var acc1 = SIMD[DType.float32,NEURAL_CHAINS](0.0)
        var acc2 = SIMD[DType.float32,NEURAL_CHAINS](0.0)
        var begin = t*Int(leaf)
        for p in range(begin,min((t+1)*Int(leaf),Int(k))):
            var av = ftz(a.unsafe_load(row*Int(asi)+p*Int(asp)))
            var off = p*Int(bsp)+col*Int(bsj)
            var chain = (p-begin)%NEURAL_CHAINS
            acc1[chain] = rtf_mul_add(av,ftz(b1.unsafe_load(off)),acc1[chain])
            acc2[chain] = rtf_mul_add(av,ftz(b2.unsafe_load(off)),acc2[chain])
        neural_fold_push[16](stack1,occupied1,neural_merge_chains[NEURAL_CHAINS](acc1))
        neural_fold_push[16](stack2,occupied2,neural_merge_chains[NEURAL_CHAINS](acc2))
    c1.unsafe_store(cell,neural_fold_drain[16](stack1,occupied1))
    c2.unsafe_store(cell,neural_fold_drain[16](stack2,occupied2))


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
    comptime if NN05 and not _PAIR_CONTROL:
        var part = neural_partition[NEURAL_LEAF](k)
        var actual = neural_strides(actual_op,m,n,k)
        ctx.enqueue_function[_neural_pair_kernel](c1,c2,input,b1,b2,Int32(m),Int32(n),Int32(k),
            Int32(part[0]),Int32(part[1]),Int32(actual[0]),Int32(actual[1]),Int32(actual[2]),Int32(actual[3]),
            grid_dim=((m*n+127)//128,1,1),block_dim=(128,1,1))
    else:
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


struct GemmWorkspace(Movable):
    """One in-order model context; no mutable global plan or operand cache."""
    var buffer: DeviceBuffer[DType.float32]
    var last_m: Int
    var last_n: Int
    var last_k: Int
    var cached_plan: Int

    def __init__(out self,ctx: DeviceContext) raises:
        step_count_device_alloc()
        self.buffer = ctx.enqueue_create_buffer[DType.float32](1)
        self.last_m = -1
        self.last_n = -1
        self.last_k = -1
        self.cached_plan = PLAN_FLAT

    def _ensure(mut self,ctx: DeviceContext,required: Int) raises:
        if required>len(self.buffer):
            step_count_sync()
            ctx.synchronize()
            step_count_device_alloc()
            self.buffer = ctx.enqueue_create_buffer[DType.float32](required)

    def run[allow_vendor: Bool = True](mut self,ctx: DeviceContext,
        mut c: DeviceBuffer[DType.float32],mut a: DeviceBuffer[DType.float32],
        mut b: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int) raises:
        var required = identical_gemm_workspace_max_floats(m,n,k)
        comptime if NN12 and not _CONTROL:
            if required>_RETAINED:
                identical_gemm[allow_vendor](ctx,c,a,b,m,n,k,op)
                return
        self._ensure(ctx,required)
        comptime if NN12 and NN10 and not _CONTROL:
            if self.last_m!=m or self.last_n!=n or self.last_k!=k:
                self.cached_plan = _selected_cost_plan(m,n,k)
                self.last_m = m
                self.last_n = n
                self.last_k = k
            _profile_or_cost_into(ctx,c,a,b,self.buffer,m,n,k,op,self.cached_plan)
        else:
            identical_gemm_into[allow_vendor](ctx,c,a,b,self.buffer,m,n,k,op)

    def run_pair(mut self,ctx: DeviceContext,mut c1: DeviceBuffer[DType.float32],
        mut c2: DeviceBuffer[DType.float32],mut a: DeviceBuffer[DType.float32],
        mut b1: DeviceBuffer[DType.float32],mut b2: DeviceBuffer[DType.float32],
        m: Int,n: Int,k: Int,op: Int) raises:
        var required = identical_gemm_pair_workspace_max_floats(m,n,k)
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
