# SPDX-License-Identifier: Apache-2.0
"""Actual Jones/init/filter stages and each production finite-difference.

CPU Float64 here is an independent verification-only oracle of the captured
actual innovations and variances. Every gradient component must have error
no greater than the actual baseline tail. A failure is a quality hold, never
permission to average away a worsened component or loosen its allowance.
No timing or same-bit FAST gate is used.
"""
from std.math import log,abs
from std.memory import bitcast
from max.gpu.host import DeviceContext
from arima.impl.fast_eval_ws import FastEvalWS,ew_finish_kernel
from arima.impl.fast_eval_df import PRODUCT_DF_ON,product_df_eligible,product_parts_kernel,product_finish_kernel,product_df_count
from arima.impl.batched_kalman import fast_kalman_into
from arima.impl.tsa.arima_common import ARIMAOrder


def _case(nobs: Int,refuse: Bool=False) raises:
    comptime assert PRODUCT_DF_ON,"compile the Apple FAST product DF candidate"
    var ctx=DeviceContext()
    var order=ARIMAOrder(1,0,0,0,0,0,0,1,0)
    var nb=3;var N=order.complexity();var eb=nb*(N+1)
    var hy=ctx.enqueue_create_host_buffer[DType.float32](nb*nobs)
    var hx=ctx.enqueue_create_host_buffer[DType.float32](nb*N)
    var y=ctx.enqueue_create_buffer[DType.float32](nb*nobs)
    var x=ctx.enqueue_create_buffer[DType.float32](nb*N)
    var grad=ctx.enqueue_create_buffer[DType.float32](nb*N)
    var xp=ctx.enqueue_create_buffer[DType.float32](nb*N)
    var f=ctx.enqueue_create_buffer[DType.float32](nb)
    var g=ctx.enqueue_create_buffer[DType.float32](nb*N)
    var bad=ctx.enqueue_create_buffer[DType.int32](nb)
    var llh=ctx.enqueue_create_host_buffer[DType.float32](eb)
    var vh=ctx.enqueue_create_host_buffer[DType.float32](eb*nobs)
    var Fh=ctx.enqueue_create_host_buffer[DType.float32](eb*nobs)
    var gh=ctx.enqueue_create_host_buffer[DType.float32](nb*N)
    var rh=ctx.enqueue_create_host_buffer[DType.float32](nb*N)
    var fh=ctx.enqueue_create_host_buffer[DType.float32](nb)
    var bh=ctx.enqueue_create_host_buffer[DType.int32](nb)
    ctx.synchronize()
    for b in range(nb):
        hx[b*N]=Float32(0.1)
        hx[b*N+1]=Float32(0.3)+Float32(b*2)
        hx[b*N+2]=Float32(0) if refuse and b==1 else Float32(1.0)
        for t in range(nobs):hy[b*nobs+t]=Float32((t*17+b*11)%31-15)/Float32(8)
    ctx.enqueue_copy(dst_buf=y,src_buf=hy)
    ctx.enqueue_copy(dst_buf=x,src_buf=hx)
    ctx.synchronize()
    var held=FastEvalWS(ctx,y,nb,nobs,order)
    if not held.compensated:raise Error("actual product adapter not reached")
    var h=Float32(0.0009765625);var scale=Float32(nobs)
    held.prepare(ctx,order,h,x,bad)
    fast_kalman_into(ctx,held.y_ext,held.t_params,order,eb,nobs,held.ws,32,0,True)
    ctx.enqueue_copy(dst_ptr=llh.unsafe_ptr(),src_buf=held.ws.loglike)
    ctx.enqueue_copy(dst_ptr=vh.unsafe_ptr(),src_buf=held.ws.vs)
    ctx.enqueue_copy(dst_ptr=Fh.unsafe_ptr(),src_buf=held.ws.Fs)
    ctx.enqueue_function[ew_finish_kernel](f,g,grad,xp,x,held.ws.loglike,held.ws.info_init,held.ws.info_loop,bad,Int32(nb),Int32(N),h,scale,grid_dim=(1,1,1),block_dim=(128,1,1))
    ctx.enqueue_copy(dst_ptr=gh.unsafe_ptr(),src_buf=g)
    ctx.enqueue_copy(dst_ptr=rh.unsafe_ptr(),src_buf=grad)
    ctx.enqueue_copy(dst_ptr=fh.unsafe_ptr(),src_buf=f)
    ctx.enqueue_copy(dst_ptr=bh.unsafe_ptr(),src_buf=bad)
    ctx.synchronize()
    var baseline=List[Float32]()
    var basef=List[Float32]()
    var baser=List[Float32]()
    var basebad=List[Int32]()
    var reference=List[Float64]()
    for cell in range(nb*N):
        baseline.append(gh[cell]);baser.append(rh[cell])
    for b in range(nb):
        if bh[b]!=0 and not refuse:raise Error("valid actual initializer/filter refused")
        basef.append(fh[b])
        basebad.append(bh[b])
    if refuse and basebad[1]==0:raise Error("invalid production variance did not refuse")
    # cpu-route: independent captured-stage likelihood oracle for qualification.
    for member in range(eb):
        var total=Float64(0)
        if basebad[member%nb]==0:
            for t in range(nobs):
                var F=Float64(Fh[member*nobs+t]);var v=Float64(vh[member*nobs+t])
                total+=log(F)+v*v/F+Float64(1.8378770664093453)
        reference.append(Float64(-0.5)*total)
    ref sc=held.compensated.value()
    ctx.enqueue_function[product_parts_kernel](held.ws.vs,held.ws.Fs,sc.parts,Int32(nobs),Int32(eb),grid_dim=(eb*((nobs+255)//256),1,1),block_dim=(256,1,1))
    ctx.enqueue_function[product_finish_kernel](sc.parts,sc.words,held.ws.loglike,held.ws.info_init,held.ws.info_loop,Int32(nobs),Int32(eb),grid_dim=(eb,1,1),block_dim=(256,1,1))
    held.finish(ctx,h,scale,x,grad,xp,f,g,bad)
    ctx.enqueue_copy(dst_ptr=gh.unsafe_ptr(),src_buf=g)
    ctx.enqueue_copy(dst_ptr=rh.unsafe_ptr(),src_buf=grad)
    ctx.enqueue_copy(dst_ptr=fh.unsafe_ptr(),src_buf=f)
    ctx.enqueue_copy(dst_ptr=bh.unsafe_ptr(),src_buf=bad)
    ctx.synchronize()
    var failures=0
    for b in range(nb):
        if bh[b]!=basebad[b]:raise Error("candidate changed filter refusal")
        if basebad[b]!=0:
            if bitcast[DType.uint32](fh[b])!=bitcast[DType.uint32](basef[b]):raise Error("refusal changed objective")
            for i in range(N):
                if bitcast[DType.uint32](gh[b*N+i])!=bitcast[DType.uint32](baseline[b*N+i]):raise Error("refusal changed optimizer gradient")
                if bitcast[DType.uint32](rh[b*N+i])!=bitcast[DType.uint32](baser[b*N+i]):raise Error("refusal changed raw scratch gradient")
            continue
        var expected=-reference[b]/Float64(scale)
        if abs(Float64(fh[b])-expected)>abs(Float64(basef[b])-expected):failures+=1
        for i in range(N):
            expected=-(reference[(i+1)*nb+b]-reference[b])/Float64(h)/Float64(scale)
            var olderror=abs(Float64(baseline[b*N+i])-expected)
            var newerror=abs(Float64(gh[b*N+i])-expected)
            print("PRODUCT_DF_COMPONENT nobs="+String(nobs)+" series="+String(b)+" parameter="+String(i)+" baseline_error="+String(olderror)+" candidate_error="+String(newerror))
            if newerror>olderror:failures+=1
    if failures:raise Error("PRODUCT_DF_QUALITY_HOLD worsened_fields="+String(failures))
    print("PRODUCT_DF_QUALITY_PASS nobs="+String(nobs)+" components="+String(nb*N))
    # Reevaluate through the actual production workspace entry after the
    # independent stage oracle. A compile flag alone never proves reach.
    var route0=product_df_count(0);var route1=product_df_count(1)
    var routed_f=ctx.enqueue_create_host_buffer[DType.float32](nb)
    var routed_g=ctx.enqueue_create_host_buffer[DType.float32](nb*N)
    held.loglike_at(ctx,order,h,x,bad)
    held.finish(ctx,h,scale,x,grad,xp,f,g,bad)
    ctx.enqueue_copy(dst_ptr=routed_f.unsafe_ptr(),src_buf=f)
    ctx.enqueue_copy(dst_ptr=routed_g.unsafe_ptr(),src_buf=g)
    ctx.synchronize()
    if product_df_count(0)-route0!=1 or product_df_count(1)-route1!=1:raise Error("actual product DF route not reached")
    for b in range(nb):
        if bitcast[DType.uint32](routed_f[b])!=bitcast[DType.uint32](fh[b]):raise Error("actual likelihood entry differs from qualified stages")
    for cell in range(nb*N):
        if bitcast[DType.uint32](routed_g[cell])!=bitcast[DType.uint32](gh[cell]):raise Error("actual gradient entry differs from qualified stages")
    print("PRODUCT_DF_ROUTE_PASS likelihood_launches=1 tail_launches=1")


def main() raises:
    _case(31)
    _case(257)
    _case(1025)
    _case(31,True)
    if product_df_eligible(ARIMAOrder(2,0,0,0,0,0,0,1,0),31):raise Error("multi-state model admitted")
    if product_df_eligible(ARIMAOrder(1,0,0,0,0,0,0,1,0),65537):raise Error("resource bound exceeded")
