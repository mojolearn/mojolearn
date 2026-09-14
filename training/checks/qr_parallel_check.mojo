# SPDX-License-Identifier: Apache-2.0
"""Cloud-only TSQR panel, destroyed-input, stacked-R and final-R bits."""
from std.os import getenv,setenv
from std.memory import bitcast
from max.gpu.host import DeviceContext,DeviceBuffer
from core.householder_qr import qr_factor,qr_slice_count
from core.device_zero import enqueue_fill
from metrics.checks.device_io import upload_f32,download_f32


def equal(ctx: DeviceContext,mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],n: Int,label: String) raises:
    var x = download_f32(ctx,a,n)
    var y = download_f32(ctx,b,n)
    for i in range(n):
        if bitcast[DType.uint32](x[i]) != bitcast[DType.uint32](y[i]):
            raise Error(label + " differs at " + String(i))


def check(m: Int,n: Int) raises:
    var ctx = DeviceContext()
    var values = List[Float32](length=m*n,fill=Float32(0))
    for i in range(m*n):
        values[i] = Float32((i*37)%257-128)/Float32(64)
        if i%19 == 0:
            values[i] = Float32(1e-40)
        elif i%23 == 0:
            values[i] = Float32(-0.0)
    var a = upload_f32(ctx,values)
    var b = upload_f32(ctx,values)
    var ns = qr_slice_count(m,n)
    var ra = ctx.enqueue_create_buffer[DType.float32](n*n)
    var rb = ctx.enqueue_create_buffer[DType.float32](n*n)
    var sa = ctx.enqueue_create_buffer[DType.float32](ns*n*n)
    var sb = ctx.enqueue_create_buffer[DType.float32](ns*n*n)
    enqueue_fill(ctx,sa,Float32(42))
    enqueue_fill(ctx,sb,Float32(42))
    ctx.synchronize()
    _ = setenv("MOJOLEARN_QR_DEVICE_COUNT","1",True)
    var one = qr_factor(ctx,a,sa,ra,m,n)
    _ = setenv("MOJOLEARN_QR_DEVICE_COUNT","2",True)
    var many = qr_factor(ctx,b,sb,rb,m,n)
    if one != ns or many != ns:
        raise Error("TSQR slice count changed")
    equal(ctx,a,b,m*n,"destroyed panel inputs")
    equal(ctx,sa,sb,ns*n*n,"stacked R scratch")
    equal(ctx,ra,rb,n*n,"final R")
    print("PASS TSQR bits",m,n,ns)
    _ = sb^
    _ = sa^
    _ = rb^
    _ = ra^
    _ = b^
    _ = a^
    ctx.synchronize()


def main() raises:
    if String(getenv("RUNPOD_POD_ID")) == "":
        raise Error("cloud host required")
    check(17,7)
    check(259,7)
    check(259,17)
    check(1025,65)
    check(4099,7)
