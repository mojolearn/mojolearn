# SPDX-License-Identifier: Apache-2.0
"""Real skewed node spans, uniform spans, tails, signed weights, dense bins."""
from max.gpu.host import DeviceContext
from experiments.performance_ideas.A07.histogram_tasks import histogram_tasks


def _case(skew: Bool,dense: Bool,invalid: Bool=False) raises:
    var ctx=DeviceContext()
    var sizes=List[Int](length=17,fill=31)
    sizes[0]=0;sizes[16]=0
    if skew:sizes[8]=4097
    var rows=0
    for n in sizes:rows+=n
    var nodes=len(sizes);var features=5
    comptime BINS=8
    var capacity=(rows+255)//256+nodes;var groups=(features+3)//4
    var hb=ctx.enqueue_create_host_buffer[DType.int32](rows*features)
    var hw=ctx.enqueue_create_host_buffer[DType.int32](rows)
    var hi=ctx.enqueue_create_host_buffer[DType.int32](nodes+1)
    var ho=ctx.enqueue_create_host_buffer[DType.int64](nodes*features*BINS)
    var hp=ctx.enqueue_create_host_buffer[DType.int32](nodes+1)
    var hn=ctx.enqueue_create_host_buffer[DType.int32](nodes+1)
    var hs=ctx.enqueue_create_host_buffer[DType.int32](capacity*groups)
    var bins=ctx.enqueue_create_buffer[DType.int32](rows*features)
    var weights=ctx.enqueue_create_buffer[DType.int32](rows)
    var offsets=ctx.enqueue_create_buffer[DType.int32](nodes+1)
    var counts=ctx.enqueue_create_buffer[DType.int32](nodes)
    var prefix=ctx.enqueue_create_buffer[DType.int32](nodes+1)
    var begins=ctx.enqueue_create_buffer[DType.int32](capacity)
    var ends=ctx.enqueue_create_buffer[DType.int32](capacity)
    var partials=ctx.enqueue_create_buffer[DType.int64](capacity*features*BINS)
    var output=ctx.enqueue_create_buffer[DType.int64](nodes*features*BINS)
    var status=ctx.enqueue_create_buffer[DType.int32](capacity*groups)
    var node_status=ctx.enqueue_create_buffer[DType.int32](nodes+1)
    ctx.synchronize()
    var reference=List[Int64](length=nodes*features*BINS,fill=Int64(0))
    var begin=0
    for node in range(nodes):
        hi[node]=Int32(begin)
        for row in range(begin,begin+sizes[node]):
            hw[row]=Int32(row%17-8)
            for f in range(features):
                var bin=0 if dense else (row*3+f)%BINS
                hb[row*features+f]=Int32(bin)
                reference[(node*features+f)*BINS+bin]+=Int64(hw[row])
        begin+=sizes[node]
    hi[nodes]=Int32(rows)
    if invalid:hi[4]=Int32(-1)
    ctx.enqueue_copy(dst_buf=bins,src_buf=hb)
    ctx.enqueue_copy(dst_buf=weights,src_buf=hw)
    ctx.enqueue_copy(dst_buf=offsets,src_buf=hi)
    ctx.synchronize()
    for arm in range(2):
        ctx.enqueue_memset(status,Int32(-7))
        ctx.enqueue_memset(output,Int64(-999))
        if arm==0:
            histogram_tasks[4,BINS,4,False](ctx,offsets,bins,weights,counts,prefix,begins,ends,partials,output,status,node_status,nodes,rows,features,8)
        else:
            histogram_tasks[4,BINS,4,True](ctx,offsets,bins,weights,counts,prefix,begins,ends,partials,output,status,node_status,nodes,rows,features,8)
        ctx.enqueue_copy(dst_ptr=ho.unsafe_ptr(),src_buf=output)
        ctx.enqueue_copy(dst_ptr=hp.unsafe_ptr(),src_buf=prefix)
        ctx.enqueue_copy(dst_ptr=hn.unsafe_ptr(),src_buf=node_status)
        ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(),src_buf=status)
        ctx.synchronize()
        if invalid:
            if hn[nodes]==0 or hp[nodes]!=0:raise Error("malformed offsets admitted histogram tasks")
        else:
            var expected_tasks=0
            for node in range(nodes):
                if hp[node]!=Int32(expected_tasks):raise Error("task prefix differs")
                expected_tasks+=(sizes[node]+255)//256
            if hp[nodes]!=Int32(expected_tasks):raise Error("task count differs")
            if hn[nodes]!=0:raise Error("valid node spans refused")
            for cell in range(expected_tasks*groups):
                if hs[cell]!=0:raise Error("valid signed histogram addend refused")
            for cell in range(nodes*features*BINS):
                if ho[cell]!=reference[cell]:raise Error("node histogram differs from independent integer oracle")
        print("A07_HISTOGRAM_TASKS_PASS skew="+String(skew)+" dense="+String(dense)+" invalid="+String(invalid)+" arm="+String(arm)+" tasks="+String(hp[nodes]))


def run_checks() raises:
    _case(False,False)
    _case(True,False)
    _case(True,True)
    _case(True,False,True)


def main() raises:
    run_checks()
