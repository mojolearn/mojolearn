# SPDX-License-Identifier: Apache-2.0
"""WP8 full-byte comparison, SIMD tail and unchanged-upload skip gate."""
from max.gpu.host import DeviceContext
from extratrees.impl.decisiontree.batched_levelalgo.builder import _stage_upload_if_changed


def check[dt: DType](ctx: DeviceContext, n: Int) raises:
    var src = ctx.enqueue_create_host_buffer[dt](n)
    var shadow = ctx.enqueue_create_host_buffer[dt](n)
    var output = ctx.enqueue_create_host_buffer[dt](n)
    var dst = ctx.enqueue_create_buffer[dt](n)
    ctx.synchronize()
    for i in range(n):
        src.unsafe_ptr()[i] = Scalar[dt](i * 19 + 7)
        shadow.unsafe_ptr()[i] = Scalar[dt](0)
    _stage_upload_if_changed(ctx, dst, src, shadow, n, False, False)
    ctx.enqueue_copy(dst_buf=output, src_buf=dst)
    ctx.synchronize()
    for i in range(n):
        if output.unsafe_ptr()[i] != src.unsafe_ptr()[i]:
            raise Error("WP8 first upload differs")
    # Corrupt the DEVICE only: equality must leave it untouched, proving skip.
    ctx.enqueue_memset(dst, Scalar[dt](0))
    _stage_upload_if_changed(ctx, dst, src, shadow, n, True, False)
    ctx.enqueue_copy(dst_buf=output, src_buf=dst)
    ctx.synchronize()
    for i in range(n):
        if output.unsafe_ptr()[i] != Scalar[dt](0):
            raise Error("WP8 equal bytes redundantly uploaded")
    # Scatter changes across every position, including each SIMD tail.
    for changed in range(n):
        src.unsafe_ptr()[changed] += Scalar[dt](1)
        _stage_upload_if_changed(ctx, dst, src, shadow, n, True, False)
        ctx.enqueue_copy(dst_buf=output, src_buf=dst)
        ctx.synchronize()
        for i in range(n):
            if output.unsafe_ptr()[i] != src.unsafe_ptr()[i]:
                raise Error("WP8 changed byte skipped or copy corrupted")
    print("PASS WP8", dt, "elements", n)


def main() raises:
    var ctx = DeviceContext()
    check[DType.uint8](ctx, 1)
    check[DType.uint8](ctx, 15)
    check[DType.uint8](ctx, 16)
    check[DType.uint8](ctx, 17)
    check[DType.uint8](ctx, 65)
    check[DType.int32](ctx, 31)
