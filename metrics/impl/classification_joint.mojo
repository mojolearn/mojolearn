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
from metrics.impl.classification import count_labels_kernel, MAX_CONFUSION_CLASSES, confusion_finish_kernel, prf_finish_kernel
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

# I24 PENDING qualification; default OFF. Only an explicit IDENTICAL define
# admits the joint count pass; the existing finish kernels remain authoritative.

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
    # I24 2026-10-06 L40S representative caller: WIN 3.609 vs 6.955 ms
    # (1M rows, 33 classes); LOSS 3.874 vs 3.085 ms (1048573 rows, 17 classes).
    # One warmup/score; mixed scope-limited result, default OFF. Evidence:
    # overnight-ab-20261006/nvidia/default-repair-normalized-measurements.json.
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_METRICS_JOINT_COUNTS"]():
        ctx.enqueue_function[joint_classification_count_kernel](truth.unsafe_ptr(),prediction.unsafe_ptr(),Int32(n),Int32(classes),matrix.unsafe_ptr(),prf.unsafe_ptr(),grid_dim=((n+255)//256,1,1),block_dim=(256,1,1))
    else:
        ctx.enqueue_function[count_labels_kernel[True,True]](truth.unsafe_ptr(),prediction.unsafe_ptr(),Int32(n),Int32(classes),matrix.unsafe_ptr(),grid_dim=((n+255)//256,1,1),block_dim=(256,1,1))
        ctx.enqueue_function[count_labels_kernel[False,False]](truth.unsafe_ptr(),prediction.unsafe_ptr(),Int32(n),Int32(classes),prf.unsafe_ptr(),grid_dim=((n+255)//256,1,1),block_dim=(256,1,1))

def classification_report_dev(ctx: DeviceContext,mut truth: DeviceBuffer[DType.int32],mut prediction: DeviceBuffer[DType.int32],n: Int,classes: Int,normalization: Int,average: Int,positive: Int,zero: Int,selected: Int) raises -> Tuple[DeviceBuffer[DType.float32],DeviceBuffer[DType.float32]]:
    """Resident normalized confusion and PRF from already admitted dense labels.

    This combined-operation API preserves the existing finish kernels and their
    warning layout. One consumer drain protects both count buffers. Input label
    validation remains the caller's responsibility, as for the existing device
    classification entry points; sentinel labels are outside this joint API.
    """
    if normalization<1 or normalization>3:
        raise Error("confusion_matrix: normalized output requires mode1..3")
    if average<0 or average>4 or zero<0 or zero>1:
        raise Error("precision_recall_fscore: invalid average/zero_division")
    if selected<=0 or selected>classes or (average==1 and (positive<0 or positive>=classes)):
        raise Error("precision_recall_fscore: invalid selected/positive class")
    if n<=0 or n>2147483647 or classes<1 or classes>MAX_CONFUSION_CLASSES or len(truth)<n or len(prediction)<n:
        raise Error("joint classification shape refused")
    var matrix_counts=ctx.enqueue_create_buffer[DType.int32](classes*classes+1)
    var prf_counts=ctx.enqueue_create_buffer[DType.int32](3*classes)
    enqueue_joint_classification_counts(ctx,truth,prediction,matrix_counts,prf_counts,n,classes)
    var matrix=ctx.enqueue_create_buffer[DType.float32](classes*classes)
    var width=selected if average==0 else 1
    var scores=ctx.enqueue_create_buffer[DType.float32](3*width+3)
    var matrix_cells=classes*classes if normalization==3 else classes
    ctx.enqueue_function[confusion_finish_kernel[DType.float32]](matrix_counts.unsafe_ptr(),Int32(classes),Int32(normalization),matrix.unsafe_ptr(),grid_dim=((matrix_cells+255)//256,1,1),block_dim=(256,1,1))
    ctx.enqueue_function[prf_finish_kernel](prf_counts.unsafe_ptr(),Int32(classes),Int32(selected),Int32(average),Int32(positive),Int32(zero),scores.unsafe_ptr(),grid_dim=((selected+255)//256 if average==0 else 1,1,1),block_dim=(256 if average==0 else 32,1,1))
    ctx.synchronize()
    _ = matrix_counts^; _ = prf_counts^
    return (matrix^,scores^)
