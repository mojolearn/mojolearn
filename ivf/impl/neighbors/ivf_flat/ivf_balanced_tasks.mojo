# SPDX-License-Identifier: Apache-2.0
"""I16 compact exact query/list chunks. Eight logical warp-width batches
per task bounds serial candidate visits while amortizing task descriptors.
The 256-row work unit is independent of vendor wave width and board rows.
Every point uses the existing ascending feature FMA chain; task top-k and
query merge compare the same total (distance,index) keys. Partial lists are
exact: any globally selected point must occur in its task's KM best.
One integer task-total readback sizes scratch; no input arithmetic on host."""
from std.gpu import block_idx,thread_idx,lane_id,shuffle_xor
from std.collections import InlineArray
from max.gpu.host import DeviceBuffer,DeviceContext
from checks.numerics import ftz,identical_mul_add
from ivf.impl.neighbors.ivf_flat.identical_ivf_scan import WARP_SIZE,_key,_kless
from ivf.impl.neighbors.ivf_flat.ivf_group_device import device_exclusive_scan_total_from
comptime TASK_ROWS=256

def _task_counts(probes: MutPointer[UInt32,MutAnyOrigin],off: MutPointer[Int32,MutAnyOrigin],counts: MutPointer[Int32,MutAnyOrigin],pairs: Int32):
    var i=Int(block_idx.x)*256+Int(thread_idx.x)
    if i<Int(pairs):
        var l=Int(probes[i])
        counts[i]=Int32((Int(off[l+1]-off[l])+TASK_ROWS-1)//TASK_ROWS)

def _scan_task[KM:Int](queries: MutPointer[Float32,MutAnyOrigin],qn: MutPointer[Float32,MutAnyOrigin],data: MutPointer[Float32,MutAnyOrigin],norm: MutPointer[Float32,MutAnyOrigin],off: MutPointer[Int32,MutAnyOrigin],ids: MutPointer[UInt32,MutAnyOrigin],probes: MutPointer[UInt32,MutAnyOrigin],task_offsets: MutPointer[Int32,MutAnyOrigin],pd: MutPointer[Float32,MutAnyOrigin],pi: MutPointer[UInt32,MutAnyOrigin],keep: MutPointer[Int32,MutAnyOrigin],keep_len: Int32,dim_in: Int32,nprobes: Int32,pairs: Int32):
    var task=Int(block_idx.x)
    var low=0
    var high=Int(pairs)
    # upper_bound skips empty query/list tasks; unique owner per task.
    while low<high:
        var middle=(low+high)//2
        if Int(task_offsets[middle+1])<=task:
            low=middle+1
        else:
            high=middle
    var pair=low
    var q=pair//Int(nprobes)
    var list_id=Int(probes[pair])
    var start=Int(off[list_id])+(task-Int(task_offsets[pair]))*TASK_ROWS
    var end=min(start+TASK_ROWS,Int(off[list_id+1]))
    var lane=Int(lane_id())
    var dim=Int(dim_in)
    var tk=InlineArray[UInt32,KM](fill=UInt32.MAX)
    var td=InlineArray[Float32,KM](fill=Float32(0))
    var ti=InlineArray[UInt32,KM](fill=UInt32.MAX)
    var pos=start+lane
    while pos<end:
        var id=ids[pos]
        if keep_len==0 or keep[Int(id)]!=0:
            var acc=Float32(0)
            for f in range(dim):
                acc=ftz(identical_mul_add(ftz(queries[q*dim+f]),ftz(data[pos*dim+f]),acc))
            var d=ftz(identical_mul_add(Float32(-2),acc,ftz(ftz(qn[q])+ftz(norm[pos]))))
            if d<=Float32(0):
                d=Float32(0)
            var key=_key(d)
            var ck=key
            var cd=d
            var ci=id
            comptime for j in range(KM):
                if _kless(ck,ci,tk[j],ti[j]):
                    var sk=tk[j];var sd=td[j];var si=ti[j]
                    tk[j]=ck;td[j]=cd;ti[j]=ci
                    ck=sk;cd=sd;ci=si
        pos+=WARP_SIZE
    for r in range(KM):
        var bk=tk[0];var bd=td[0];var bi=ti[0]
        comptime for shift in [32,16,8,4,2,1]:
            comptime if shift<WARP_SIZE:
                var ok=shuffle_xor(bk,UInt32(shift))
                var od=shuffle_xor(bd,UInt32(shift))
                var oi=shuffle_xor(bi,UInt32(shift))
                if _kless(ok,oi,bk,bi):
                    bk=ok;bd=od;bi=oi
        if tk[0]==bk and ti[0]==bi:
            comptime for j in range(KM-1):
                tk[j]=tk[j+1];td[j]=td[j+1];ti[j]=ti[j+1]
            tk[KM-1]=UInt32.MAX;ti[KM-1]=UInt32.MAX
        if lane==0:
            pd[task*KM+r]=bd
            pi[task*KM+r]=bi

def _merge_tasks[KM:Int](pd: MutPointer[Float32,MutAnyOrigin],pi: MutPointer[UInt32,MutAnyOrigin],off: MutPointer[Int32,MutAnyOrigin],od: MutPointer[Float32,MutAnyOrigin],oi: MutPointer[UInt32,MutAnyOrigin],nq: Int32,np: Int32,k: Int32):
    var q=Int(block_idx.x)*256+Int(thread_idx.x)
    if q>=Int(nq):
        return
    var tk=InlineArray[UInt32,KM](fill=UInt32.MAX)
    var td=InlineArray[Float32,KM](fill=Float32(0))
    var ti=InlineArray[UInt32,KM](fill=UInt32.MAX)
    for task in range(Int(off[q*Int(np)]),Int(off[(q+1)*Int(np)])):
        for r in range(KM):
            var id=pi[task*KM+r]
            if id==UInt32.MAX:
                break
            var d=pd[task*KM+r]
            var ck=_key(d);var cd=d;var ci=id
            comptime for j in range(KM):
                if _kless(ck,ci,tk[j],ti[j]):
                    var sk=tk[j];var sd=td[j];var si=ti[j]
                    tk[j]=ck;td[j]=cd;ti[j]=ci
                    ck=sk;cd=sd;ci=si
    for r in range(Int(k)):
        od[q*Int(k)+r]=td[r];oi[q*Int(k)+r]=ti[r]

def ivf_balanced_scan[KM:Int](ctx: DeviceContext,mut query: DeviceBuffer[DType.float32],mut qn: DeviceBuffer[DType.float32],mut data: DeviceBuffer[DType.float32],mut norm: DeviceBuffer[DType.float32],mut off: DeviceBuffer[DType.int32],mut ids: DeviceBuffer[DType.uint32],mut probes: DeviceBuffer[DType.uint32],mut keep: DeviceBuffer[DType.int32],keep_len: Int,mut outd: DeviceBuffer[DType.float32],mut outi: DeviceBuffer[DType.uint32],nq: Int,np: Int,dim: Int,k: Int) raises:
    var pairs=nq*np
    if nq<0 or np<1 or dim<1 or k<1 or k>KM or pairs>2147483647:
        raise Error("I16 balanced task shape refused")
    if pairs==0:
        return
    var counts=ctx.enqueue_create_buffer[DType.int32](pairs+1)
    var offsets=ctx.enqueue_create_buffer[DType.int32](pairs+1)
    counts.enqueue_fill(Int32(0))
    ctx.enqueue_function[_task_counts](probes.unsafe_ptr(),off.unsafe_ptr(),counts.unsafe_ptr(),Int32(pairs),grid_dim=((pairs+255)//256,1,1),block_dim=(256,1,1))
    device_exclusive_scan_total_from(ctx,counts,offsets,pairs)
    var h=ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(),src_buf=offsets.create_sub_buffer[DType.int32](pairs,1))
    ctx.synchronize()
    var tasks=Int(h[0])
    if tasks<0 or tasks>2147483647//KM:
        raise Error("I16 balanced task scratch exceeds int32 indexing")
    var pd=ctx.enqueue_create_buffer[DType.float32](max(tasks*KM,1))
    var pi=ctx.enqueue_create_buffer[DType.uint32](max(tasks*KM,1))
    if tasks>0:
        ctx.enqueue_function[_scan_task[KM]](query.unsafe_ptr(),qn.unsafe_ptr(),data.unsafe_ptr(),norm.unsafe_ptr(),off.unsafe_ptr(),ids.unsafe_ptr(),probes.unsafe_ptr(),offsets.unsafe_ptr(),pd.unsafe_ptr(),pi.unsafe_ptr(),keep.unsafe_ptr(),Int32(keep_len),Int32(dim),Int32(np),Int32(pairs),grid_dim=(tasks,1,1),block_dim=(WARP_SIZE,1,1))
    ctx.enqueue_function[_merge_tasks[KM]](pd.unsafe_ptr(),pi.unsafe_ptr(),offsets.unsafe_ptr(),outd.unsafe_ptr(),outi.unsafe_ptr(),Int32(nq),Int32(np),Int32(k),grid_dim=((nq+255)//256,1,1),block_dim=(256,1,1))
    ctx.synchronize()
    _ = counts^;_ = offsets^;_ = pd^;_ = pi^
