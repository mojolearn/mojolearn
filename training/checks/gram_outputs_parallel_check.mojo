# SPDX-License-Identifier: Apache-2.0
"""Cloud-only bit comparisons of original Gram output contractions."""
from std.os import getenv, setenv
from std.memory import bitcast
from max.gpu.host import DeviceContext
from core.gemm import gemm_tn_identical_v1, gemm_nt_gram
from metrics.checks.device_io import upload_f32, download_f32


def check(m: Int,k: Int,tn: Bool) raises:
    var ctx = DeviceContext()
    var values = List[Float32](length=m*k,fill=Float32(0))
    for i in range(m*k):
        values[i] = Float32((i*37)%257-128)/Float32(64)
        if i%19 == 0:
            values[i] = Float32(1e-40)
        elif i%23 == 0:
            values[i] = Float32(-0.0)
    var x = upload_f32(ctx,values)
    var one = ctx.enqueue_create_buffer[DType.float32](m*m)
    var many = ctx.enqueue_create_buffer[DType.float32](m*m)
    var scratch = ctx.enqueue_create_buffer[DType.float32](m*k)
    ctx.synchronize()
    _ = setenv("MOJOLEARN_GRAM_DEVICE_COUNT","1",True)
    if tn:
        gemm_tn_identical_v1(ctx,one,x,scratch,m,k)
    else:
        gemm_nt_gram(ctx,one,x,m,m,k)
    ctx.synchronize()
    _ = setenv("MOJOLEARN_GRAM_DEVICE_COUNT","2",True)
    if tn:
        gemm_tn_identical_v1(ctx,many,x,scratch,m,k)
    else:
        gemm_nt_gram(ctx,many,x,m,m,k)
    ctx.synchronize()
    var a = download_f32(ctx,one,m*m)
    var b = download_f32(ctx,many,m*m)
    for i in range(m*m):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            raise Error("Gram cell differs: " + String(m) + "/" + String(k) + "/" + String(tn) + "/" + String(i))
    print("PASS Gram output bits",m,k,tn)
    _ = scratch^
    _ = many^
    _ = one^
    _ = x^
    ctx.synchronize()


def main() raises:
    if String(getenv("RUNPOD_POD_ID")) == "":
        raise Error("cloud host required")
    var widths: List[Int] = [129,257,513]
    var contractions: List[Int] = [3,127,259,1025]
    for m in widths:
        for k in contractions:
            check(m,k,True)
    var rows: List[Int] = [3,17,33]
    var features: List[Int] = [7,129,257]
    for m in rows:
        for k in features:
            check(m,k,False)
