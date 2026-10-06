# SPDX-License-Identifier: Apache-2.0
"""F16 compensated tail from actual production Kalman innovations/variances.

Initializer, Jones transform, covariance recursion, finite-difference step
and optimizer layouts are the production ones. Only likelihood summation
and retained-low-word differencing change, under an Apple FAST opt-in. This
never dispatches the private supplied-state scalar probe. Two bounded GPU
reduction levels own independent model members; no host arithmetic or wait.
"""
from std.gpu import block_idx,thread_idx,block_dim
from std.atomic import Atomic,Ordering
from std.ffi import _Global
from std.memory import stack_allocation
from std.math import isfinite,isinf,inf
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceContext,DeviceBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE,NUMERIC_FAST,ftz
from arima.impl.fast_scalar_df import DF,df_add,df_sub,df_mul,df_div,df_round,df_log_positive
from arima.impl.tsa.arima_common import ARIMAOrder

# F16 PENDING: compile evidence alone does not qualify device correctness, quality or speed.
# Tail requires Apple FAST + MOJOLEARN_ARIMA_FAST_PRODUCT_DF_TAIL; original Kalman path retained.
# NEVER RUN — PENDING VALIDATION: actual production compensated likelihood/gradient tail.
comptime PRODUCT_DF_ON=(GLOBAL_NUMERIC_MODE==NUMERIC_FAST and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_ARIMA_FAST_PRODUCT_DF_TAIL"]())
comptime PRODUCT_DF_TPB=256


struct _ProductDfAudit(Defaultable,Movable):
    var likelihoods: Int64
    var tails: Int64

    def __init__(out self):
        self.likelihoods=Int64(0)
        self.tails=Int64(0)


comptime _PRODUCT_DF_AUDIT=_Global[StorageType=_ProductDfAudit,name="MojolearnProductDfAuditV1",init_fn=_ProductDfAudit.__init__]


def product_df_count(stage: Int) raises -> Int:
    """Enqueued actual product routes: 0 likelihood, 1 finite-difference tail.

    The default build returns zero without creating an audit global. A hit
    proves source route reach, never device completion or numerical quality.
    Device completion and quality are independently owed by the caller gate.
    """
    if stage<0 or stage>1:raise Error("invalid product DF audit stage")
    comptime if PRODUCT_DF_ON:
        ref audit=_PRODUCT_DF_AUDIT.get_or_create_ptr()[]
        if stage==0:return Int(Atomic.load[ordering=Ordering.RELAXED](MutPointer(to=audit.likelihoods)))
        return Int(Atomic.load[ordering=Ordering.RELAXED](MutPointer(to=audit.tails)))
    return 0


@always_inline
def product_df_hit(stage: Int) raises:
    comptime if PRODUCT_DF_ON:
        ref audit=_PRODUCT_DF_AUDIT.get_or_create_ptr()[]
        if stage==0:_=Atomic.fetch_add[ordering=Ordering.RELAXED](MutPointer(to=audit.likelihoods),Int64(1))
        else:_=Atomic.fetch_add[ordering=Ordering.RELAXED](MutPointer(to=audit.tails),Int64(1))


def product_df_eligible(order: ARIMAOrder,nobs: Int) -> Bool:
    # One actual scalar state, no diffuse/exogenous/seasonal model ambiguity.
    # The two-level reduction can hold at most256 chunks of256 observations.
    return order.rd()==1 and order.n_diff()==0 and order.n_exog==0 and order.P==0 and order.Q==0 and nobs>0 and nobs<=65536


def product_parts_kernel(vs: MutPointer[Float32,MutAnyOrigin],Fs: MutPointer[Float32,MutAnyOrigin],
    parts: MutPointer[Float32,MutAnyOrigin],nobs_in: Int32,members_in: Int32):
    var nobs=Int(nobs_in);var members=Int(members_in);var chunks=(nobs+255)//256
    var bid=Int(block_idx.x)//chunks;var chunk=Int(block_idx.x)%chunks;var tid=Int(thread_idx.x)
    var pos=chunk*256+tid
    var logs=DF(Float32(0));var square=DF(Float32(0))
    if pos<nobs:
        var F=Fs.unsafe_load(bid*nobs+pos);var v=vs.unsafe_load(bid*nobs+pos)
        if F>Float32(0) and isfinite(F) and isfinite(v):
            logs=df_log_positive(F)
            square=df_div(df_mul(DF(v),DF(v)),DF(F))
        else:
            # Propagate nonfinite actual stages to the finish, which retains
            # the original LL/status instead of inventing a valid objective.
            logs=DF(inf[DType.float32]())
    var lh=stack_allocation[256,Float32,address_space=AddressSpace.SHARED]()
    var ll=stack_allocation[256,Float32,address_space=AddressSpace.SHARED]()
    var qh=stack_allocation[256,Float32,address_space=AddressSpace.SHARED]()
    var ql=stack_allocation[256,Float32,address_space=AddressSpace.SHARED]()
    lh[tid]=logs.hi;ll[tid]=logs.lo;qh[tid]=square.hi;ql[tid]=square.lo
    barrier()
    var step=128
    while step>0:
        if tid<step:
            var l=df_add(DF(lh[tid],ll[tid]),DF(lh[tid+step],ll[tid+step]))
            var q=df_add(DF(qh[tid],ql[tid]),DF(qh[tid+step],ql[tid+step]))
            lh[tid]=l.hi;ll[tid]=l.lo;qh[tid]=q.hi;ql[tid]=q.lo
        barrier()
        step//=2
    if tid==0:
        var index=bid*chunks+chunk;var count=members*chunks
        parts.unsafe_store(index,lh[0]);parts.unsafe_store(count+index,ll[0])
        parts.unsafe_store(2*count+index,qh[0]);parts.unsafe_store(3*count+index,ql[0])


def product_finish_kernel(parts: MutPointer[Float32,MutAnyOrigin],words: MutPointer[Float32,MutAnyOrigin],
    ll_out: MutPointer[Float32,MutAnyOrigin],info_init: MutPointer[Int32,MutAnyOrigin],
    info_loop: MutPointer[Int32,MutAnyOrigin],nobs_in: Int32,members_in: Int32):
    var bid=Int(block_idx.x);var tid=Int(thread_idx.x)
    var members=Int(members_in);var nobs=Int(nobs_in);var chunks=(nobs+255)//256;var count=members*chunks
    var lh=stack_allocation[256,Float32,address_space=AddressSpace.SHARED]()
    var ll=stack_allocation[256,Float32,address_space=AddressSpace.SHARED]()
    var qh=stack_allocation[256,Float32,address_space=AddressSpace.SHARED]()
    var ql=stack_allocation[256,Float32,address_space=AddressSpace.SHARED]()
    lh[tid]=Float32(0);ll[tid]=Float32(0);qh[tid]=Float32(0);ql[tid]=Float32(0)
    if tid<chunks:
        var index=bid*chunks+tid
        lh[tid]=parts.unsafe_load(index);ll[tid]=parts.unsafe_load(count+index)
        qh[tid]=parts.unsafe_load(2*count+index);ql[tid]=parts.unsafe_load(3*count+index)
    barrier()
    var step=128
    while step>0:
        if tid<step:
            var l=df_add(DF(lh[tid],ll[tid]),DF(lh[tid+step],ll[tid+step]))
            var q=df_add(DF(qh[tid],ql[tid]),DF(qh[tid+step],ql[tid+step]))
            lh[tid]=l.hi;ll[tid]=l.lo;qh[tid]=q.hi;ql[tid]=q.lo
        barrier()
        step//=2
    if tid==0:
        var original=ll_out.unsafe_load(bid)
        var n=DF(Float32(nobs))
        var log2pi=DF(Float32(1.8378770664093453),Float32(3.1268354230284965e-8))
        var value=df_mul(DF(Float32(-0.5)),df_add(df_add(DF(qh[0],ql[0]),df_mul(n,log2pi)),DF(lh[0],ll[0])))
        # Keep the first actual production refusal; never clear its stage code.
        if info_init.unsafe_load(bid)!=0 or info_loop.unsafe_load(bid)!=0 or not isfinite(original) or not isfinite(df_round(value)):
            value=DF(original)
        ll_out.unsafe_store(bid,df_round(value))
        words.unsafe_store(bid,value.hi);words.unsafe_store(members+bid,value.lo)
        # Preserve baseline scratch gradients on a refused series even when
        # another finite-difference member was numerically valid.
        words.unsafe_store(2*members+bid,original)


def product_tail_kernel(f: MutPointer[Float32,MutAnyOrigin],g: MutPointer[Float32,MutAnyOrigin],
    raw: MutPointer[Float32,MutAnyOrigin],xpert: MutPointer[Float32,MutAnyOrigin],x: MutPointer[Float32,MutAnyOrigin],
    words: MutPointer[Float32,MutAnyOrigin],info_init: MutPointer[Int32,MutAnyOrigin],
    info_loop: MutPointer[Int32,MutAnyOrigin],bad: MutPointer[Int32,MutAnyOrigin],nb_in: Int32,N_in: Int32,h: Float32,scale: Float32):
    var b=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x);var nb=Int(nb_in);var N=Int(N_in)
    if b>=nb:return
    var members=nb*(N+1);var invalid=False
    for member in range(N+1):
        var index=member*nb+b
        if info_init.unsafe_load(index)!=0 or info_loop.unsafe_load(index)!=0 or (isinf(words.unsafe_load(index)) and words.unsafe_load(index)<Float32(0)):
            invalid=True
    bad.unsafe_store(b,Int32(1) if invalid else Int32(0))
    var base=DF(words.unsafe_load(b),words.unsafe_load(members+b))
    f.unsafe_store(b,inf[DType.float32]() if invalid else ftz(df_round(df_div(df_mul(DF(Float32(-1)),base),DF(scale)))))
    for i in range(N):
        var member=(i+1)*nb+b
        var diff=df_sub(DF(words.unsafe_load(member),words.unsafe_load(members+member)),base)
        var gradient=df_div(diff,DF(h))
        var raw_gradient=ftz(df_round(gradient))
        if invalid:
            var olddiff=ftz(ftz(words.unsafe_load(2*members+member))-ftz(words.unsafe_load(2*members+b)))
            raw_gradient=ftz(olddiff/h)
        raw.unsafe_store(b*N+i,raw_gradient)
        xpert.unsafe_store(b*N+i,x.unsafe_load(b*N+i))
        g.unsafe_store(b*N+i,Float32(0) if invalid else ftz(df_round(df_div(df_mul(DF(Float32(-1)),gradient),DF(scale)))))


struct ProductDfScratch(Movable):
    var parts: DeviceBuffer[DType.float32]
    var words: DeviceBuffer[DType.float32]

    def __init__(out self,ctx: DeviceContext,members: Int,nobs: Int) raises:
        self.parts=ctx.enqueue_create_buffer[DType.float32](4*members*((nobs+255)//256))
        self.words=ctx.enqueue_create_buffer[DType.float32](3*members)
