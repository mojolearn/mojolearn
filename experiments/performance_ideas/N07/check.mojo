# SPDX-License-Identifier: Apache-2.0
"""Exact signed weighted histogram arms, skew, tails and invalid addends."""
from experiments.performance_ideas.N07.production_check import run_checks as check_production
from max.gpu.host import DeviceContext
from experiments.performance_ideas.N07.streamed_histogram import streamed_histogram


def main() raises:
    check_production()
    var ctx=DeviceContext()
    var rows=769;var features=11
    comptime BINS=32
    var hb=ctx.enqueue_create_host_buffer[DType.int32](rows*features)
    var hw=ctx.enqueue_create_host_buffer[DType.int32](rows)
    var bins=ctx.enqueue_create_buffer[DType.int32](rows*features)
    var weights=ctx.enqueue_create_buffer[DType.int32](rows)
    var partials=ctx.enqueue_create_buffer[DType.int64](4*features*BINS)
    var output=ctx.enqueue_create_buffer[DType.int64](features*BINS)
    var status=ctx.enqueue_create_buffer[DType.int32](4*3)
    var ho=ctx.enqueue_create_host_buffer[DType.int64](features*BINS)
    var hs=ctx.enqueue_create_host_buffer[DType.int32](4*3)
    ctx.synchronize()
    for fixture in range(3):
        var reference=List[Int64]()
        for cell in range(features*BINS):reference.append(Int64(0))
        for row in range(rows):
            hw[row]=Int32(row%17-8)
            for f in range(features):
                var bin=0 if fixture==1 else (row*7+f*11)%BINS
                hb[row*features+f]=Int32(bin)
                reference[f*BINS+bin]+=Int64(hw[row])
        if fixture==2:
            hw[rows-1]=Int32(9)
        ctx.enqueue_copy(dst_buf=bins,src_buf=hb)
        ctx.enqueue_copy(dst_buf=weights,src_buf=hw)
        ctx.synchronize()
        for arm in range(3):
            var groups=3
            if arm==0:
                streamed_histogram[4,BINS,1](ctx,bins,weights,partials,output,status,rows,features,8)
            elif arm==1:
                streamed_histogram[4,BINS,4](ctx,bins,weights,partials,output,status,rows,features,8)
            else:
                groups=2
                streamed_histogram[8,BINS,4](ctx,bins,weights,partials,output,status,rows,features,8)
            ctx.enqueue_copy(dst_ptr=ho.unsafe_ptr(),src_buf=output)
            ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(),src_buf=status)
            ctx.synchronize()
            var invalid=0
            for cell in range(4*groups):invalid+=Int(hs[cell])
            if fixture==2:
                if invalid==0:raise Error("out-of-contract addend was accepted")
            else:
                if invalid!=0:raise Error("valid integer histogram rejected")
                for cell in range(features*BINS):
                    if ho[cell]!=reference[cell]:raise Error("signed histogram differs from exact integer oracle")
            print("N07_PASS fixture="+String(fixture)+" arm="+String(arm)+" invalid_partitions="+String(invalid))
