# SPDX-License-Identifier: Apache-2.0
"""Full objective trial scheduling versus the unchanged sequential control."""
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext
from metrics.checks.device_io import upload_f32,download_f32
from glm.checks.logistic_check import _fixture
from glm.impl.qn.glm_base import GLMDims,GLMWithData
from glm.impl.linear_model.qn import QN_LOSS_LOGISTIC
from glm.impl.qn.qn_util import LBFGSParam,LS_SUCCESS,LS_INVALID_DIR,LS_INVALID_STEP,LS_INVALID_STEP_MIN,LS_INVALID_STEP_MAX,LS_MAX_ITERS_REACHED
from glm.impl.qn.qn_linesearch import ls_backtrack,ls_backtrack_sequential
from glm.impl.qn.simple_mat.dense import ax,copy_vec

struct TrialResult(Movable):
    var words: List[UInt32]
    var status: Int
    var logical: Int
    var physical: Int
    var considered: Int
    var fresh: Bool
    def __init__(out self,var words: List[UInt32],status: Int,logical: Int,physical: Int,considered: Int,fresh: Bool):
        self.words=words^; self.status=status; self.logical=logical
        self.physical=physical; self.considered=considered; self.fresh=fresh

def run_trial(ctx: DeviceContext,case: Int,control: Bool) raises -> TrialResult:
    var rows=257
    var d=7
    var n=d+1
    var data=_fixture(rows,d,1.0)
    var dx=upload_f32(ctx,data[0])
    var dy=upload_f32(ctx,data[1])
    var zeros=List[Float32](length=n,fill=Float32(0))
    var x=upload_f32(ctx,zeros)
    var xp=upload_f32(ctx,zeros)
    var grad=ctx.enqueue_create_buffer[DType.float32](n)
    var gradp=ctx.enqueue_create_buffer[DType.float32](n)
    var direction=ctx.enqueue_create_buffer[DType.float32](n)
    var scalar=ctx.enqueue_create_buffer[DType.float32](4)
    var stage=ctx.enqueue_create_host_buffer[DType.float32](4)
    var f=GLMWithData(ctx,dx^,dy^,rows,GLMDims.make(1,d,True),QN_LOSS_LOGISTIC,Float32(0.125))
    var fx=f.evaluate(ctx,x,grad)
    copy_vec(ctx,gradp,grad)
    ax(ctx,direction,Float32(1) if case==2 else Float32(-1),grad,n)
    var param=LBFGSParam.defaults()
    var step=Float32(1)
    var expected=LS_SUCCESS
    if case==1:
        step=Float32(64)  # reject full steps before finding the first acceptance
    elif case==2:
        expected=LS_INVALID_DIR
    elif case==3:
        step=Float32(0); expected=LS_INVALID_STEP
    elif case==4:
        param.ftol=Float32(1e6); param.min_step=Float32(2)
        expected=LS_INVALID_STEP_MIN
    elif case==5:
        param.ftol=Float32(1e6); param.max_step=Float32(0.5)
        expected=LS_INVALID_STEP_MAX
    elif case==6 or case==7:
        param.ftol=Float32(1e6)
        param.max_linesearch=2 if case==6 else 7
        expected=LS_MAX_ITERS_REACHED
    var iterations=0
    var fresh=False
    var ret=0
    if control:
        ret=ls_backtrack_sequential(ctx,param,f,fx,x,grad,step,direction,xp,n,scalar,iterations,stage,fresh,gradp,False)
    else:
        ret=ls_backtrack(ctx,param,f,fx,x,grad,step,direction,xp,n,scalar,iterations,stage,fresh,gradp,False)
    if ret!=expected:
        raise Error("I12 planted Armijo fixture did not reach its intended status")
    var hw=download_f32(ctx,x,n)
    var hg=download_f32(ctx,grad,n)
    var active=download_f32(ctx,f.slots,3)
    var words=List[UInt32]()
    words.append(bitcast[DType.uint32](fx)); words.append(bitcast[DType.uint32](step))
    words.append(bitcast[DType.uint32](f.gnorm_raw))
    for value in hw:
        words.append(bitcast[DType.uint32](value))
    for value in hg:
        words.append(bitcast[DType.uint32](value))
    # Invalid-direction undo intentionally forgets its speculative norm;
    # those objective scratch words are not consumed after that status.
    if ret!=LS_INVALID_DIR:
        for value in active:
            words.append(bitcast[DType.uint32](value))
    var result=TrialResult(words^,ret,f.n_evals,f.speculative_evals,iterations,fresh)
    _ = x^; _ = xp^; _ = grad^; _ = gradp^; _ = direction^; _ = scalar^; _ = stage^
    return result^

def check_exact_trial_schedule(ctx: DeviceContext) raises:
    for case in range(8):
        var expected=run_trial(ctx,case,True)
        var actual=run_trial(ctx,case,False)
        if actual.status!=expected.status or actual.logical!=expected.logical or actual.considered!=expected.considered or actual.fresh!=expected.fresh:
            raise Error("I12 speculative line-search status/count/first acceptance differs")
        if len(actual.words)!=len(expected.words):
            raise Error("I12 trial result shape differs")
        for i in range(len(actual.words)):
            if actual.words[i]!=expected.words[i]:
                raise Error("I12 selected trial/objective/gradient/cache word differs at "+String(i))
        var should_reach=False
        comptime if is_defined["MOJOLEARN_IDN_QN_EXACT_TRIALS"]():
            should_reach=case!=3
        if (actual.physical>0)!=should_reach or expected.physical!=0:
            raise Error("I12 full speculative evaluator did not reach the requested arm")
        if case==1 and actual.considered<=1:
            raise Error("I12 rejection fixture accepted its initial full step")
        print("I12_TRIAL case",case,"status",actual.status,"considered",actual.considered,"logical",actual.logical,"physical",actual.physical)
