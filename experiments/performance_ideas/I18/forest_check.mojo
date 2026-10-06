# SPDX-License-Identifier: Apache-2.0
"""Real retained forest histograms, per-column route witnesses and forest
digests. Device queue only. Compare full-fit digests against the flag-off arm.
No timings are inferred from this correctness/attribution check."""
from std.sys.info import size_of
from max.gpu.host import DeviceContext
from ensemble.checks.rf_perf_candidates_check import Fixture, ObjT, BinT, N_COLS, N_CLASSES, MAX_N_BINS, build_workload, upload_structs, upload_i32
from ensemble.checks.fingerprint_probe import _rf_params, _fit_clf, _fit_reg
from ensemble.decisiontree.decisiontree import GINI, MSE
from ensemble.decisiontree.batched_levelalgo.retained_count_histograms import RetainedCountHistograms
from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels import NodeWorkItem, InstanceRange, WorkloadInfo, SharedMemoryConfig
from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import DeviceArgs, HistogramArgs, launch_build_histograms_kernel

def phase(ctx: DeviceContext, mut fx: Fixture, mut cache: RetainedCountHistograms, items: List[NodeWorkItem], samples: List[Int32], candidate: Bool) raises -> List[UInt32]:
    var n=len(items)
    var counts=List[Int]()
    for item in items:
        counts.append(item.instances.count)
    var workloads=build_workload(counts,128)
    var di=ctx.enqueue_create_buffer[DType.uint8](n*size_of[NodeWorkItem]())
    var dw=ctx.enqueue_create_buffer[DType.uint8](len(workloads)*size_of[WorkloadInfo]())
    var ds=ctx.enqueue_create_buffer[DType.int32](n*N_COLS)
    upload_structs(ctx,di,items); upload_structs(ctx,dw,workloads); upload_i32(ctx,ds,samples)
    var ip=di.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]().unsafe_bitcast[NodeWorkItem]()
    var wp=dw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]().unsafe_bitcast[WorkloadInfo]()
    var sp=ds.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var data=fx.dataset()
    data.n_sampled_cols=Int32(N_COLS)
    var obj=ObjT(Int32(N_CLASSES),Int32(1),Int32(GINI))
    var blob=DeviceArgs[HistogramArgs[ObjT]](ctx)
    var ap=blob.upload(ctx,HistogramArgs[ObjT](data.copy(),fx.quantiles(),obj^))
    var slots=MAX_N_BINS*N_CLASSES
    var hist=ctx.enqueue_create_buffer[DType.uint32](n*2*slots)
    var result=List[UInt32]()
    for i in range(n*N_COLS*slots):
        result.append(UInt32(0xdeadbeef))
    if candidate:
        cache.prepare(ctx,ip,n)
    for col in [0,2]:
        hist.enqueue_fill(UInt32(0))
        if candidate:
            cache.enqueue[ObjT,True,False](ctx,hist.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]().unsafe_bitcast[BinT](),ip,wp,sp,ap.unsafe_origin_cast[MutAnyOrigin](),col,MAX_N_BINS,len(workloads),2,n)
        else:
            launch_build_histograms_kernel[ObjT](ctx,hist.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]().unsafe_bitcast[BinT](),MAX_N_BINS,data,ip.unsafe_origin_cast[MutUntrackedOrigin](),col,sp.unsafe_origin_cast[MutUntrackedOrigin](),wp.unsafe_origin_cast[MutUntrackedOrigin](),len(workloads),2,SharedMemoryConfig(True,0),ap)
        ctx.synchronize()
        with hist.map_to_host() as h:
            for node in range(n):
                for c in range(2):
                    var physical=Int(samples[node*N_COLS+col+c])
                    for cell in range(slots):
                        result[(node*N_COLS+physical)*slots+cell]=h[(node*2+c)*slots+cell]
    _ = di^; _ = dw^; _ = ds^; _ = blob^; _ = hist^
    return result^

def equal(a: List[UInt32],b: List[UInt32]) raises:
    if len(a)!=len(b):
        raise Error("I18 length changed")
    for i in range(len(a)):
        if a[i]!=b[i]:
            raise Error("I18 retained count differs at cell "+String(i))

def mechanism(ctx: DeviceContext) raises:
    var fx=Fixture(ctx)
    var cache=RetainedCountHistograms(ctx,3,N_COLS,MAX_N_BINS*N_CLASSES)
    cache.reset(ctx)
    var root=List[NodeWorkItem]()
    root.append(NodeWorkItem(0,Int32(0),InstanceRange(0,fx.n_rows)))
    var root_samples=List[Int32]()
    for col in [0,1,2,3]:
        root_samples.append(Int32(col))
    equal(phase(ctx,fx,cache,root,root_samples,True),phase(ctx,fx,cache,root,root_samples,False))
    var left=fx.n_rows//3
    var children=List[NodeWorkItem]()
    children.append(NodeWorkItem(1,Int32(1),InstanceRange(0,left)))
    children.append(NodeWorkItem(2,Int32(1),InstanceRange(left,fx.n_rows-left)))
    # Different per-node permutations cross a pass boundary. Column2 is
    # unavailable in left's first pass, so only that right column falls back.
    var samples=List[Int32]()
    for col in [3,0,2,1,0,2,1,3]:
        samples.append(Int32(col))
    equal(phase(ctx,fx,cache,children,samples,True),phase(ctx,fx,cache,children,samples,False))
    ctx.synchronize()
    with cache.skip.map_to_host() as mask:
        for col in range(N_COLS):
            if mask[2*N_COLS+col]!=(UInt8(0) if col==2 else UInt8(1)):
                raise Error("I18 sibling route/permutation witness failed")
    # Beyond the byte-bounded node cache, every column must use the row loop.
    var overflow=List[NodeWorkItem]()
    overflow.append(NodeWorkItem(4,Int32(2),InstanceRange(left,fx.n_rows-left)))
    equal(phase(ctx,fx,cache,overflow,root_samples,True),phase(ctx,fx,cache,overflow,root_samples,False))
    cache.reset(ctx)
    equal(phase(ctx,fx,cache,children,samples,True),phase(ctx,fx,cache,children,samples,False))
    ctx.synchronize()
    with cache.skip.map_to_host() as mask:
        for col in range(N_COLS):
            if mask[2*N_COLS+col]!=UInt8(0):
                raise Error("I18 stale parent survived tree reset")
    _ = fx^; _ = cache^

def main() raises:
    var ctx=DeviceContext()
    mechanism(ctx)
    for streams in [1,2]:
        var full=_rf_params(3,True,Float32(1),6,16,GINI,n_streams=streams)
        print("I18_FULL",streams,_fit_clf(ctx,521,43,4,full,False,UInt64(871)))
        var partial=_rf_params(3,True,Float32(.5),6,16,GINI,n_streams=streams)
        print("I18_SUBSAMPLED_FALLBACK",streams,_fit_clf(ctx,521,43,4,partial,False,UInt64(871)))
        var regression=_rf_params(2,True,Float32(1),5,16,MSE,n_streams=streams)
        print("I18_REGRESSION_FALLBACK",streams,_fit_reg(ctx,521,7,regression,UInt64(972)))
    print("I18 PASS mechanism permutations cache_bound reset full_forest_digests; compare OFF arm")
