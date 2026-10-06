# SPDX-License-Identifier: Apache-2.0
"""Actual fused integer count adapter versus two production count passes.
Existing normalized confusion/PRF final kernels consume resident outputs;
exact count and floating result bits agree. Scaling/finite checks separately
exercise resident statistics versus public materialization and fitted attrs."""
from std.memory import bitcast
from max.gpu.host import DeviceContext
from metrics.checks.device_io import upload_i32, upload_f32, download_f32
from metrics.impl.classification_joint import enqueue_joint_classification_counts, classification_report_dev
from metrics.impl.classification import count_labels_kernel, confusion_finish_kernel, prf_finish_kernel, confusion_matrix, precision_recall_fscore
from preprocessing.standard import standard_fit_dev, standard_fit
from preprocessing.minmax import minmax_fit_dev, minmax_fit

def check(ctx: DeviceContext,n: Int,k: Int) raises:
    var y=List[Int32]()
    var p=List[Int32]()
    for i in range(n):
        y.append(Int32((i*17+i//5)%k))
        p.append(Int32((i*13+i//7)%k))
    var dy=upload_i32(ctx,y)
    var dp=upload_i32(ctx,p)
    var counts=ctx.enqueue_create_buffer[DType.int32](k*k+1)
    var prf=ctx.enqueue_create_buffer[DType.int32](3*k)
    var reference=ctx.enqueue_create_buffer[DType.int32](k*k+1)
    var reference_prf=ctx.enqueue_create_buffer[DType.int32](3*k)
    enqueue_joint_classification_counts(ctx,dy,dp,counts,prf,n,k)
    reference.enqueue_fill(Int32(0)); reference_prf.enqueue_fill(Int32(0))
    ctx.enqueue_function[count_labels_kernel[True,True]](dy.unsafe_ptr(),dp.unsafe_ptr(),Int32(n),Int32(k),reference.unsafe_ptr(),grid_dim=((n+255)//256,1,1),block_dim=(256,1,1))
    ctx.enqueue_function[count_labels_kernel[False,False]](dy.unsafe_ptr(),dp.unsafe_ptr(),Int32(n),Int32(k),reference_prf.unsafe_ptr(),grid_dim=((n+255)//256,1,1),block_dim=(256,1,1))
    var matrix=ctx.enqueue_create_buffer[DType.float32](k*k)
    var matrix_ref=ctx.enqueue_create_buffer[DType.float32](k*k)
    var output=ctx.enqueue_create_buffer[DType.float32](3*k+3)
    var output_ref=ctx.enqueue_create_buffer[DType.float32](3*k+3)
    for normalization in [1,2,3]:
        ctx.enqueue_function[confusion_finish_kernel[DType.float32]](counts.unsafe_ptr(),Int32(k),Int32(normalization),matrix.unsafe_ptr(),grid_dim=((k*k+255)//256,1,1),block_dim=(256,1,1))
        ctx.enqueue_function[confusion_finish_kernel[DType.float32]](reference.unsafe_ptr(),Int32(k),Int32(normalization),matrix_ref.unsafe_ptr(),grid_dim=((k*k+255)//256,1,1),block_dim=(256,1,1))
        var a=download_f32(ctx,matrix,k*k)
        var b=download_f32(ctx,matrix_ref,k*k)
        for i in range(k*k):
            if bitcast[DType.uint32](a[i])!=bitcast[DType.uint32](b[i]):
                raise Error("I24 confusion result changed bits")
    ctx.enqueue_function[prf_finish_kernel](prf.unsafe_ptr(),Int32(k),Int32(k),Int32(0),Int32(k-1),Int32(0),output.unsafe_ptr(),grid_dim=((k+255)//256,1,1),block_dim=(256,1,1))
    ctx.enqueue_function[prf_finish_kernel](reference_prf.unsafe_ptr(),Int32(k),Int32(k),Int32(0),Int32(k-1),Int32(0),output_ref.unsafe_ptr(),grid_dim=((k+255)//256,1,1),block_dim=(256,1,1))
    var a=download_f32(ctx,output,3*k+3)
    var b=download_f32(ctx,output_ref,3*k+3)
    for i in range(3*k+3):
        if bitcast[DType.uint32](a[i])!=bitcast[DType.uint32](b[i]):
            raise Error("I24 resident PRF result changed bits")
    # Complete resident consumer versus both original public device operations,
    # including every average and all three undefined-metric warning cells.
    for average in [0,1,2,3,4]:
        for zero in [0,1]:
            var combined=classification_report_dev(ctx,dy,dp,n,k,3,average,k-1,zero,k)
            var public_matrix=confusion_matrix[DType.float32](ctx,dy,dp,n,k,3)
            var public_scores=precision_recall_fscore(ctx,dy,dp,n,k,average,k-1,zero,k)
            var combined_matrix=download_f32(ctx,combined[0],k*k)
            var original_matrix=download_f32(ctx,public_matrix,k*k)
            for i in range(k*k):
                if bitcast[DType.uint32](combined_matrix[i])!=bitcast[DType.uint32](original_matrix[i]):
                    raise Error("I24 complete confusion consumer changed bits")
            var cells=3*(k if average==0 else 1)+3
            var combined_scores=download_f32(ctx,combined[1],cells)
            var original_scores=download_f32(ctx,public_scores,cells)
            for i in range(cells):
                if bitcast[DType.uint32](combined_scores[i])!=bitcast[DType.uint32](original_scores[i]):
                    raise Error("I24 complete PRF consumer or warning changed bits")
            _ = combined^; _ = public_matrix^; _ = public_scores^
    _ = dy^; _ = dp^; _ = counts^; _ = prf^; _ = reference^; _ = reference_prf^; _ = matrix^; _ = matrix_ref^; _ = output^; _ = output_ref^

def scaler_check(ctx: DeviceContext,n: Int,d: Int) raises:
    var x=List[Float32]()
    for row in range(n):
        for f in range(d):
            x.append(Float32(7) if f==0 else Float32((row*17+f*13)%251-125)/Float32(128))
    var data=upload_f32(ctx,x)
    var resident=standard_fit_dev(ctx,data,n,d,1,1)
    var words=download_f32(ctx,resident,3*d)
    var public=standard_fit(ctx,data,n,d,1,1)
    for i in range(3*d):
        if bitcast[DType.uint32](words[i])!=bitcast[DType.uint32](public[i]):
            raise Error("I24 resident standard fitted attributes moved")
    var mm=minmax_fit_dev(ctx,data,n,d,Float32(-1),Float32(1))
    var mm_words=download_f32(ctx,mm,5*d)
    var mm_public=minmax_fit(ctx,data,n,d,Float32(-1),Float32(1))
    for i in range(5*d):
        if bitcast[DType.uint32](mm_words[i])!=bitcast[DType.uint32](mm_public[i]):
            raise Error("I24 resident minmax fitted attributes moved")
    _ = data^; _ = resident^; _ = mm^

def main() raises:
    var ctx=DeviceContext()
    for n in [1,255,257,1031]:
        for k in [2,7,33]:
            check(ctx,n,k)
        scaler_check(ctx,n,7)
    print("I24 PASS joint_count_cases=12 resident_scaler_attributes=4 empty_class_warning_cases=3")
