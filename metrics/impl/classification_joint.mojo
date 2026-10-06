# SPDX-License-Identifier: Apache-2.0
"""I24 one input pass for resident confusion and PRF counts.
Exactly encoded finite labels (0<=label<classes) have already passed public
admission; this operator refuses mismatched buffer shapes before indexing.
Counts are exact Int32 for n<=Int32.max, matching classification.mojo.
Outputs remain on device for existing finish kernels; no array readback."""
from std.atomic import Atomic
from std.gpu import block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from metrics.impl.classification import count_labels_kernel, MAX_CONFUSION_CLASSES

def joint_classification_count_kernel(truth: MutPointer[Int32, MutAnyOrigin],prediction: MutPointer[Int32, MutAnyOrigin],n: Int32,classes: Int32,matrix: MutPointer[Int32, MutAnyOrigin],prf: MutPointer[Int32, MutAnyOrigin]):
    var i=Int(block_idx.x)*256+Int(thread_idx.x)
    if i<Int(n):
        var y=Int(truth[i])
        var p=Int(prediction[i])
        var k=Int(classes)
        _ = Atomic.fetch_add(matrix+y*k+p,Int32(1))
        _ = Atomic.fetch_add(matrix+k*k,Int32(1))
        _ = Atomic.fetch_add(prf+k+y,Int32(1))
        _ = Atomic.fetch_add(prf+2*k+p,Int32(1))
        if y==p:
            _ = Atomic.fetch_add(prf+y,Int32(1))

def enqueue_joint_classification_counts(ctx: DeviceContext,mut truth: DeviceBuffer[DType.int32],mut prediction: DeviceBuffer[DType.int32],mut matrix: DeviceBuffer[DType.int32],mut prf: DeviceBuffer[DType.int32],n: Int,classes: Int) raises:
    if n<0 or n>2147483647 or classes<1 or classes>MAX_CONFUSION_CLASSES:
        raise Error("joint classification shape refused")
    if len(truth)<n or len(prediction)<n or len(matrix)<classes*classes+1 or len(prf)<3*classes:
        raise Error("joint classification capacity refused")
    matrix.enqueue_fill(Int32(0))
    prf.enqueue_fill(Int32(0))
    if n==0:
        return
    comptime if is_defined["MOJOLEARN_METRICS_JOINT_COUNTS"]():
        ctx.enqueue_function[joint_classification_count_kernel](truth.unsafe_ptr(),prediction.unsafe_ptr(),Int32(n),Int32(classes),matrix.unsafe_ptr(),prf.unsafe_ptr(),grid_dim=((n+255)//256,1,1),block_dim=(256,1,1))
    else:
        ctx.enqueue_function[count_labels_kernel[True,True]](truth.unsafe_ptr(),prediction.unsafe_ptr(),Int32(n),Int32(classes),matrix.unsafe_ptr(),grid_dim=((n+255)//256,1,1),block_dim=(256,1,1))
        ctx.enqueue_function[count_labels_kernel[False,False]](truth.unsafe_ptr(),prediction.unsafe_ptr(),Int32(n),Int32(classes),prf.unsafe_ptr(),grid_dim=((n+255)//256,1,1),block_dim=(256,1,1))
