# SPDX-License-Identifier: Apache-2.0
"""Actual public coarse/rescore/fallback path; existing Apple support only."""
from std.memory import bitcast
from max.gpu.host import DeviceContext
from neighbors.estimator import knn_search
from neighbors.impl.detail.knn_brute_force import KNN_METHOD_AUTO,KNN_METHOD_TILED,KNN_CERTIFIED_MMA,certified_knn_reach_clear,certified_knn_reach_read

def check_certified_caller(ctx: DeviceContext) raises:
    var ni=257
    var nq=37
    for fixture in range(3):
        for d in [1,7,31]:
            for k in [1,17]:
                var index=ctx.enqueue_create_host_buffer[DType.float32](ni*d)
                var query=ctx.enqueue_create_host_buffer[DType.float32](nq*d)
                var dist=ctx.enqueue_create_host_buffer[DType.float32](nq*k)
                var ids=ctx.enqueue_create_host_buffer[DType.uint32](nq*k)
                var expected=List[UInt32]()
                for row in range(ni):
                    for f in range(d):
                        var value=Float32(row)*Float32(0.125)+Float32(f)*Float32(0.03125)
                        if fixture==1:
                            value=Float32(0)  # every candidate/excluded row ties
                        elif fixture==2:
                            value=Float32(4096)+Float32((row*17+f)%257)*Float32(0.000244140625)
                        index[row*d+f]=value
                for row in range(nq):
                    for f in range(d):
                        var value=Float32(row%5)*Float32(0.03125)+Float32(f)*Float32(0.03125)
                        if fixture==1:
                            value=Float32(0)
                        elif fixture==2:
                            value=Float32(4096)+Float32((row*7+f)%257)*Float32(0.000244140625)
                        query[row*d+f]=value
                for root in [False,True]:
                    expected.clear()
                    certified_knn_reach_clear()
                    _ = knn_search(ctx,index.unsafe_ptr(),ni,query.unsafe_ptr(),nq,d,k,dist.unsafe_ptr(),ids.unsafe_ptr(),return_sqrt=root,requested_query_tile=64,knn_method=KNN_METHOD_TILED)
                    var before=certified_knn_reach_read()
                    if before[0]!=0:
                        raise Error("I15 forced exact control unexpectedly used coarse candidates")
                    for cell in range(nq*k):
                        expected.append(ids[cell]); expected.append(bitcast[DType.uint32](dist[cell]))
                    _ = knn_search(ctx,index.unsafe_ptr(),ni,query.unsafe_ptr(),nq,d,k,dist.unsafe_ptr(),ids.unsafe_ptr(),return_sqrt=root,requested_query_tile=64,knn_method=KNN_METHOD_AUTO)
                    for cell in range(nq*k):
                        if ids[cell]!=expected[2*cell] or bitcast[DType.uint32](dist[cell])!=expected[2*cell+1]:
                            raise Error("I15 actual coarse/rescore/fallback caller differs from exact TILED")
                    var reached=certified_knn_reach_read()
                    comptime if KNN_CERTIFIED_MMA:
                        if reached[0]!=1 or reached[1]!=nq or reached[2]<0 or reached[2]>nq:
                            raise Error("I15 public candidate did not reach existing coarse/rescore driver")
                        if fixture==0 and reached[2]==nq:
                            raise Error("I15 separated-neighbor fixture certified no query")
                        if fixture==1 and reached[2]!=nq:
                            raise Error("I15 excluded duplicate ties did not force complete exact fallback")
                    else:
                        if reached[0]!=0:
                            raise Error("I15 unsupported coarse vendor unexpectedly reached driver")
                    print("I15_CERT_CALLER fixture",fixture,"d",d,"k",k,"root",root,"supported",KNN_CERTIFIED_MMA,"queries",reached[1],"fallback",reached[2])
                _ = index^; _ = query^; _ = dist^; _ = ids^
