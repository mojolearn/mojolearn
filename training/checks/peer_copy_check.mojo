# SPDX-License-Identifier: Apache-2.0
"""Cloud-only cross-device copy gate: does a kernel on device 1 read what was copied?

For sizes around 1 MiB, a root buffer on device 0 holding a position pattern
is copied to device 1 by (A) `core/multi_gpu.mojo::peer_clone` followed by a
drain of both contexts, and (B) host staging (device 0 to host, host to
device 1). A kernel on device 1 then copies the received buffer into a fresh
buffer cell by cell, and that result is downloaded and compared with the
pattern. No host readback of the received buffer happens before the kernel.
"""
from std.os import getenv
from std.memory import bitcast
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from core.multi_gpu import peer_clone


def copy_cells_kernel(dst: MutPointer[Float32, MutAnyOrigin], src: MutPointer[Float32, MutAnyOrigin], n_in: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        dst[i] = src[i]


def pattern(i: Int) -> Float32:
    return Float32((i * 2654435761) % 16777213) + Float32(0.5)


def check(n: Int, staged: Bool) raises -> Int:
    var root = DeviceContext(device_id=0)
    var owner = DeviceContext(device_id=1)
    var host = root.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        host.unsafe_ptr()[i] = pattern(i)
    var src = root.enqueue_create_buffer[DType.float32](n)
    root.enqueue_copy(dst_buf=src, src_ptr=host.unsafe_ptr())
    root.synchronize()
    var received: DeviceBuffer[DType.float32]
    if staged:
        var back = root.enqueue_create_host_buffer[DType.float32](n)
        root.enqueue_copy(dst_ptr=back.unsafe_ptr(), src_buf=src)
        root.synchronize()
        received = owner.enqueue_create_buffer[DType.float32](n)
        owner.enqueue_copy(dst_buf=received, src_ptr=back.unsafe_ptr())
        owner.synchronize()
        _ = back^
    else:
        received = peer_clone(root, owner, src)
        owner.synchronize()
        root.synchronize()
    var out = owner.enqueue_create_buffer[DType.float32](n)
    owner.enqueue_function[copy_cells_kernel](out.unsafe_ptr(), received.unsafe_ptr(), Int32(n),
        grid_dim=((n + 255) // 256, 1, 1), block_dim=(256, 1, 1))
    var result = owner.enqueue_create_host_buffer[DType.float32](n)
    owner.enqueue_copy(dst_ptr=result.unsafe_ptr(), src_buf=out)
    owner.synchronize()
    var bad = 0
    for i in range(n):
        if bitcast[DType.uint32](result.unsafe_ptr()[i]) != bitcast[DType.uint32](pattern(i)):
            bad += 1
    _ = result^
    _ = out^
    _ = received^
    _ = src^
    _ = host^
    _ = owner^
    _ = root^
    return bad


def main() raises:
    if String(getenv("RUNPOD_POD_ID")) == "":
        raise Error("RunPod required; no local execution")
    var sizes: List[Int] = [4096, 65536, 262144, 262145, 263169, 524288, 1048576, 4194304]
    var failures = 0
    for n in sizes:
        var a = check(n, False)
        var b = check(n, True)
        print("PEERCOPY cells", n, "bytes", n * 4, "peer_clone differing", a, "host_staged differing", b)
        if a != 0 or b != 0:
            failures += 1
    print("PEERCOPY cases with a difference:", failures)
