# SPDX-License-Identifier: Apache-2.0
"""Exact partial-pool reuse, overwritten queries/weights, budget/lifetime gates."""
from std.memory import bitcast
from max.gpu.host import DeviceContext
# TOMBSTONE: MOJOLEARN_IDN_KDE_PARTIAL_POOL (unmeasured opt-in arm (I20 mixed, no promotion)) deleted 2026-11-15 by lane/expdel2-19-kernel; code recoverable at 3af082d6f4.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_KDE_PARTIAL_POOL.patch; record in docs/TOMBSTONES.md.
from kde.impl.chunk_workspace import KdeChunkWorkspace,KDE_CHUNK_POOL_ON
from kde.impl.neighbors.kernel_density import kde_score_samples_chunk_lse_identical,kde_score_samples_chunk_lse_reused,DIST_L2_SQRT_UNEXPANDED,KDE_KERNEL_GAUSSIAN,KDE_KERNEL_TOPHAT,host_sum_weights
from kde.checks.kde_check import _train_fixture,_query_fixture,_weight_fixture,_upload,_scores_by_path
from kde.resident_fit import kde_fit_prepare,kde_score_samples_resident,kde_fit_release,KDE_FIT_REGISTRY
from kde.impl.neighbors.kernel_density import metric_from_name,kernel_from_name
from checks.native_tests.kit import download_host_into


def check_resident() raises:
    var ctx=DeviceContext()
    var nt=131;var dims=7
    var train=_train_fixture(nt,dims,29);var weights=_weight_fixture(nt,41)
    for metricname in [String("euclidean"),String("manhattan"),String("chebyshev")]:
        for kernelname in [String("gaussian"),String("tophat")]:
            var handle=kde_fit_prepare(train.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),nt,dims,Float32(0.25),kernelname,metricname,weights,True)
            for nq in [9,3]:
                var query=_query_fixture(train,nt,nq,dims,37+nq)
                var output=List[Float32](length=nq,fill=Float32(123456))
                _=kde_score_samples_resident(handle,query.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),nq,dims,Float32(0.25),kernelname,metricname,output.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
                var expected=_scores_by_path(ctx,train,query,weights,True,nt,nq,dims,Float32(0.25),kernel_from_name(kernelname),metric_from_name(metricname),0,0,0)
                for i in range(nq):
                    if bitcast[DType.uint32](output[i])!=bitcast[DType.uint32](expected[i]):raise Error("public retained KDE changed density bits")
            kde_fit_release(handle)
            print("I20_PUBLIC_POOL_PASS metric="+metricname+" kernel="+kernelname+" pool_enabled="+String(KDE_CHUNK_POOL_ON))


def run_checks() raises:
    var ctx=DeviceContext()
    var nt=513;var dims=7
    var hosttrain=_train_fixture(nt,dims,29)
    var hostweights=_weight_fixture(nt,41)
    var train=_upload(ctx,hosttrain,0,Float32(-987))
    var weights=_upload(ctx,hostweights,0,Float32(-987))
    var pool=KdeChunkWorkspace(4096)
    pool.bind(ctx,UInt64(7),nt,dims)
    for fixture in range(6):
        var nq=3
        if fixture==1:nq=9
        if fixture==2:nq=129  # oversized weighted request, preserved small pool
        if fixture==3:nq=2
        if fixture==5:nq=2
        var queryhost=_query_fixture(hosttrain,nt,nq,dims,37+fixture)
        var query=_upload(ctx,queryhost,0,Float32(-987))
        var expected=ctx.enqueue_create_buffer[DType.float32](nq)
        var actual=ctx.enqueue_create_buffer[DType.float32](nq)
        var eh=ctx.enqueue_create_host_buffer[DType.float32](nq)
        var ah=ctx.enqueue_create_host_buffer[DType.float32](nq)
        ctx.synchronize()
        var weighted=fixture!=3 and fixture!=5
        var sumw=host_sum_weights(hostweights) if weighted else Float32(nt)
        var kernel=KDE_KERNEL_TOPHAT if fixture==4 else KDE_KERNEL_GAUSSIAN
        kde_score_samples_chunk_lse_identical(ctx,train,query,weights,weighted,sumw,nt,nq,dims,Float32(0.25),kernel,DIST_L2_SQRT_UNEXPANDED,expected)
        ctx.enqueue_memset(actual,Float32(123456))
        if fixture==3 and not pool.try_lease():raise Error("unclaimed KDE scratch lease unavailable")
        var admitted=kde_score_samples_chunk_lse_reused(ctx,train,query,weights,weighted,sumw,nt,nq,dims,Float32(0.25),kernel,DIST_L2_SQRT_UNEXPANDED,actual,pool,UInt64(7))
        if fixture==3:
            if admitted:raise Error("overlapping KDE scratch lease admitted")
            pool.release_lease()
        if not admitted:
            kde_score_samples_chunk_lse_identical(ctx,train,query,weights,weighted,sumw,nt,nq,dims,Float32(0.25),kernel,DIST_L2_SQRT_UNEXPANDED,actual)
        if fixture==2 and admitted:raise Error("KDE oversized request retained beyond budget")
        if fixture!=2 and fixture!=3 and not admitted:raise Error("small KDE request did not reuse pool")
        download_host_into[DType.float32](ctx, expected, eh)
        download_host_into[DType.float32](ctx, actual, ah)
        ctx.synchronize()
        for i in range(nq):
            if bitcast[DType.uint32](eh[i])!=bitcast[DType.uint32](ah[i]):raise Error("retained partials changed KDE score bits or left poison")
        if pool.retained_bytes()>pool.budget_bytes:raise Error("KDE retained bytes exceeded budget")
        print("I20_POOL_PASS fixture="+String(fixture)+" admitted="+String(admitted)+" retained_bytes="+String(pool.retained_bytes())+" allocations="+String(pool.allocations)+" reuse="+String(pool.reuse_calls))
    if pool.oversized_calls!=1 or pool.reuse_calls<2:raise Error("KDE pool allocation/reuse reach failed")
    var refused=False
    try:
        _=pool.prepare(ctx,UInt64(8),1,1,False)
    except:refused=True
    if not refused:raise Error("KDE pool accepted a foreign immutable source")
    pool.bind(ctx,UInt64(8),nt,dims)
    if pool.retained_bytes()!=0 or pool.invalidations!=1:raise Error("source rebinding retained stale partials")
    _=pool.prepare(ctx,UInt64(8),1,1,False)
    pool.begin()
    refused=False
    try:
        _=pool.prepare(ctx,UInt64(8),1,1,False)
    except:refused=True
    if not refused:raise Error("in-flight KDE pool reuse admitted")
    pool.close(ctx)
    if pool.retained_bytes()!=0:raise Error("closed KDE pool retained storage")
    refused=False
    try:
        _=pool.prepare(ctx,UInt64(8),1,1,False)
    except:refused=True
    if not refused:raise Error("closed KDE pool reuse admitted")
    print("I20_POOL_LIFECYCLE_PASS")
    check_resident()
