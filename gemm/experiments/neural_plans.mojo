# SPDX-License-Identifier: Apache-2.0
"""NN01/NN08/NN10/NN12: explicit schedules and bounded model-owned plans.

All entry points are neural component adapters, not production dispatch.
NN01 and NN08 deliberately reuse existing implementations; no old component
win is relabelled as a new full-model result. This source is uncompiled and
unverified by request. Public model ownership and full recipes remain pending.
"""
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer,DeviceContext
from checks.kernel_matrix import TARGET_COLUMN,COLUMN_NVIDIA
from gemm.contract import CONTRACT_K_LEAF_MIN
from gemm.checks.gemm_identical import (
    GEMM_ARM_TRIAL,GEMM_ARM_SHIPPED,gemm_step_arm_geometry,gemm_step_geometry_name,
    gemm_shipped_dispatch_name,identical_gemm_step_geometry_into,
    identical_gemm_shipped_into,identical_gemm_workspace_max_floats,
    identical_gemm_workspace_floats,identical_gemm_with_plan,choose_gemm_plan,
    PLAN_FLAT,PLAN_TUNED_32_2X2,PLAN_TUNED_64_4X4,PLAN_TUNED_128_8X8,
)
from gemm.experiments.neural_profile import NEURAL_EXPERIMENTS_ALLOWED,NEURAL_LEAF,NEURAL_CHAINS,neural_validate
from gemm.experiments.async_operand_pipeline import pipeline_gemm

comptime NN01 = NEURAL_EXPERIMENTS_ALLOWED and is_defined["MOJOLEARN_IDN_NEURAL_NN01"]()
comptime NN08 = NEURAL_EXPERIMENTS_ALLOWED and is_defined["MOJOLEARN_IDN_NEURAL_NN08"]()
comptime NN10 = NEURAL_EXPERIMENTS_ALLOWED and is_defined["MOJOLEARN_IDN_NEURAL_NN10"]()
comptime NN12 = NEURAL_EXPERIMENTS_ALLOWED and is_defined["MOJOLEARN_IDN_NEURAL_NN12"]()


def neural_schedule_ab[CANDIDATE: Bool = False](
    ctx: DeviceContext,mut c: DeviceBuffer[DType.float32],mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],mut workspace: DeviceBuffer[DType.float32],
    m: Int,n: Int,k: Int,op: Int,arm: Int = GEMM_ARM_SHIPPED,
) raises -> String:
    """NN01: select a real existing geometry and return its actual route label.

    Timing must encompass the caller and consume outputs. Route text records
    only which adapter was reached; it is not a measurement/correctness claim.
    """
    neural_validate(m,n,k,op)
    if len(c)<m*n or len(a)<m*k or len(b)<n*k:
        raise Error("neural schedule operand buffer too short")
    if len(workspace)<identical_gemm_workspace_max_floats(m,n,k):
        raise Error("neural schedule workspace too short")
    comptime if CANDIDATE and NN01:
        comptime if not GEMM_ARM_TRIAL:
            raise Error("NN01 candidate needs MOJOLEARN_GEMM_ARM_TRIAL; refusing silent shipped fallback")
        var geometry = gemm_step_arm_geometry(arm,m,n,k)
        identical_gemm_step_geometry_into(ctx,c,a,b,workspace,m,n,k,op,geometry,False)
        return gemm_step_geometry_name(geometry)
    else:
        identical_gemm_shipped_into(ctx,c,a,b,workspace,m,n,k,op)
        return gemm_shipped_dispatch_name(m,n,k)


def neural_async_ab[CANDIDATE: Bool = False](
    ctx: DeviceContext,mut c: DeviceBuffer[DType.float32],mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int,
) raises:
    """NN08: supported NVIDIA copy pipeline versus its synchronous control.

    Existing N03 mechanism only. No AMD asynchronous primitive is invented.
    The native AMD/Apple schedule can retain the same profile; a candidate
    request for this particular unsupported scheduling arm fails explicitly.
    """
    neural_validate(m,n,k,op)
    comptime if CANDIDATE and NN08:
        comptime if TARGET_COLUMN != COLUMN_NVIDIA:
            raise Error("NN08 async candidate available only on supported NVIDIA path; Modular support pending elsewhere")
        pipeline_gemm[True,NEURAL_LEAF,NEURAL_CHAINS](ctx,c,a,b,m,n,k,op)
    else:
        pipeline_gemm[False,NEURAL_LEAF,NEURAL_CHAINS](ctx,c,a,b,m,n,k,op)


def _neural_tile_cost(m: Int,n: Int,k: Int,tile: Int,target_blocks: Int) -> Float64:
    var tiles = ((m+tile-1)//tile)*((n+tile-1)//tile)
    var padded = tiles*tile*tile
    # Read traffic for one shared A/B tile at each contraction term, plus
    # masked arithmetic issue work. Scale by a declared device-fill budget
    # when there are fewer independent tiles. These costs vary continuously
    # across neighborhoods except for actual tile boundaries. No board row,
    # benchmark dataset, or guessed hardware register count enters the rule.
    # Host-side shape metadata only. Float64 avoids integer overflow on legal
    # large matrices; these values never participate in the numerical graph.
    var work = Float64(tiles*2*tile + padded-m*n)*Float64(max(k,1))
    return work*Float64(max(target_blocks,tiles))/Float64(max(tiles,1))


def neural_cost_plan(m: Int,n: Int,k: Int,target_blocks: Int) raises -> Int:
    if target_blocks<1:
        raise Error("cost dispatch requires an explicit hardware fill budget")
    if m<=0 or n<=0:
        return PLAN_FLAT
    var flat_blocks = (m*n+127)//128
    var cost = Float64(2)*Float64(m*n)*Float64(max(k,1))*Float64(max(target_blocks,flat_blocks))/Float64(max(flat_blocks,1))
    var plan = PLAN_FLAT
    var c32 = _neural_tile_cost(m,n,k,32,target_blocks)
    if c32<cost:
        cost = c32
        plan = PLAN_TUNED_32_2X2
    var c64 = _neural_tile_cost(m,n,k,64,target_blocks)
    if c64<cost:
        cost = c64
        plan = PLAN_TUNED_64_4X4
    var c128 = _neural_tile_cost(m,n,k,128,target_blocks)
    if c128<cost:
        plan = PLAN_TUNED_128_8X8
    return plan


struct NeuralGemmPlan(Copyable,Movable):
    var m: Int
    var n: Int
    var k: Int
    var op: Int
    var plan: Int
    var workspace_floats: Int
    var target_blocks: Int
    var settings_generation: Int
    var cost_candidate: Bool

    def __init__(out self,m: Int,n: Int,k: Int,op: Int,
        target_blocks: Int,settings_generation: Int,cost_candidate: Bool = False) raises:
        neural_validate(m,n,k,op)
        self.m = m
        self.n = n
        self.k = k
        self.op = op
        self.target_blocks = target_blocks
        self.settings_generation = settings_generation
        self.cost_candidate = cost_candidate and NN10
        # NN10 intentionally names choose_gemm_plan as its B, rather than
        # claiming that this whole-dispatch control isolates one old rule.
        # Splitting each implicated legacy rule into its own removal arm is
        # still required before a public default change.
        self.plan = choose_gemm_plan(m,n,k)
        comptime if NN10:
            if cost_candidate:
                self.plan = neural_cost_plan(m,n,k,target_blocks)
        self.workspace_floats = max(1,identical_gemm_workspace_floats(m,n,k,self.plan))

    def matches(self,m: Int,n: Int,k: Int,op: Int,target_blocks: Int,generation: Int,
        cost_candidate: Bool) -> Bool:
        return (self.m==m and self.n==n and self.k==k and self.op==op
                and self.target_blocks==target_blocks and self.settings_generation==generation
                and self.cost_candidate==(cost_candidate and NN10))

    def run(self,ctx: DeviceContext,mut c: DeviceBuffer[DType.float32],
        mut a: DeviceBuffer[DType.float32],mut b: DeviceBuffer[DType.float32],
        mut workspace: DeviceBuffer[DType.float32]) raises:
        if len(c)<self.m*self.n or len(a)<self.m*self.k or len(b)<self.n*self.k:
            raise Error("neural plan operand storage too short")
        if len(workspace)<self.workspace_floats:
            raise Error("neural plan workspace too short")
        identical_gemm_with_plan(ctx,c,a,b,workspace,self.m,self.n,self.k,self.op,self.plan)


struct NeuralPlanWorkspace(Movable):
    """One model, one in-order context, bounded retained scratch and one plan.

    A includes first-use allocation. B re-plans and provisions every call.
    Growth drains old users before releasing their memory. The owner must
    call close on its context after success OR failure, before destruction.
    No implicit thread/stream safety and no global cache are claimed.
    """
    var workspace: DeviceBuffer[DType.float32]
    var plan: NeuralGemmPlan
    var valid: Bool
    var max_retained_floats: Int
    var owner: Int
    var oversized_calls: Int

    def __init__(out self,ctx: DeviceContext,max_retained_floats: Int,owner: Int) raises:
        if max_retained_floats<1:
            raise Error("positive neural retained scratch budget required")
        self.workspace = ctx.enqueue_create_buffer[DType.float32](1)
        self.plan = NeuralGemmPlan(0,0,0,0,1,0)
        self.valid = False
        self.max_retained_floats = max_retained_floats
        self.owner = owner
        self.oversized_calls = 0

    def run[CANDIDATE: Bool = False,COST_CANDIDATE: Bool = False](mut self,ctx: DeviceContext,
        mut c: DeviceBuffer[DType.float32],mut a: DeviceBuffer[DType.float32],
        mut b: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int,
        owner: Int,settings_generation: Int,target_blocks: Int) raises -> Int:
        if owner != self.owner:
            raise Error("neural plan/workspace belongs to another model owner")
        comptime if CANDIDATE and NN12:
            if not self.valid or not self.plan.matches(m,n,k,op,target_blocks,settings_generation,COST_CANDIDATE):
                self.plan = NeuralGemmPlan(m,n,k,op,target_blocks,settings_generation,COST_CANDIDATE)
                self.valid = True
            if self.plan.workspace_floats<=self.max_retained_floats:
                if len(self.workspace)<self.plan.workspace_floats:
                    ctx.synchronize()
                    self.workspace = ctx.enqueue_create_buffer[DType.float32](self.plan.workspace_floats)
                self.plan.run(ctx,c,a,b,self.workspace)
                return self.plan.plan
            self.oversized_calls += 1
        # Cold/per-call arm and above-budget calls charge the allocation and
        # last-consumer synchronization to the whole operation.
        var temporary_plan = NeuralGemmPlan(m,n,k,op,target_blocks,settings_generation,COST_CANDIDATE)
        var temporary = ctx.enqueue_create_buffer[DType.float32](temporary_plan.workspace_floats)
        try:
            temporary_plan.run(ctx,c,a,b,temporary)
        except error:
            ctx.synchronize()
            raise error
        ctx.synchronize()
        return temporary_plan.plan

    def invalidate(mut self):
        self.valid = False

    def close(mut self,ctx: DeviceContext) raises:
        ctx.synchronize()
        self.valid = False
        self.workspace = ctx.enqueue_create_buffer[DType.float32](1)
