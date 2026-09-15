# SPDX-License-Identifier: Apache-2.0
"""Cloud-only cross-device copy gate: does a kernel on device 1 read what was copied?

Part 1 (PEERCOPY). For sizes around 1 MiB, a root buffer on device 0 holding a
position pattern is copied to device 1 by (A) `core/multi_gpu.mojo::peer_clone`
followed by a drain of both contexts, and (B) host staging (device 0 to host,
host to device 1). A kernel on device 1 then copies the received buffer into a
fresh buffer cell by cell, and that result is downloaded and compared with the
pattern. No host readback of the received buffer happens before the kernel.

Part 2 (PEERSOLVE). The device-to-device column solve that diverged on two
MI300X for n >= 513 (bench/results/multi_gpu/2026-09-14/cholesky-mi300x-diag/),
rebuilt here outside the estimator so one variant at a time can be moved
toward the host-staged form. Every variant's forward substitution is compared
bit for bit with the one-device forward substitution of the same factor and
right-hand sides. The variants are named in `variant_name`.
"""
from std.os import getenv
from std.memory import bitcast
from std.time import perf_counter_ns
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from core.multi_gpu import peer_clone
from cholesky.checks.trsm import trsm_lower_kernel, CHOL_SOLVE_TPB
from cholesky.multi_gpu import chol_gather_columns, chol_scatter_columns


def copy_cells_kernel(dst: MutPointer[Float32, MutAnyOrigin], src: MutPointer[Float32, MutAnyOrigin], n_in: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        dst[i] = src[i]


def fill_kernel(dst: MutPointer[Float32, MutAnyOrigin], n_in: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        dst[i] = Float32(7.25)


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


# ---------------------------------------------------------------- PEERSOLVE

comptime V_OLD = 0          # the pre-staging solve transport, unchanged
comptime V_SLEEP = 1        # V_OLD plus two seconds of host wall time before the kernels
comptime V_READBACK = 2     # V_OLD plus the transport diagnostic's host readbacks
comptime V_L_HOST = 3       # factor from host through the owner; columns device to device
comptime V_B_HOST = 4       # columns from host through the owner; factor device to device
comptime V_BACK_HOST = 5    # V_OLD transport to owners; results back through host
comptime V_L_FIRST = 6      # the factor copied before the columns
comptime V_SERIAL = 7       # V_OLD with every owner drained right after its kernel
comptime V_PREALLOC = 8     # owner buffers created before the gather, copied by enqueue_copy_to
comptime V_ROOT_RANK0 = 9   # rank 0 solves on the root context (no second device-0 context)
comptime V_COPYK = 10       # V_OLD transport, then a copy kernel instead of the solve
comptime V_DIRTY = 11       # V_OLD after a freed, filled factor-sized buffer on device 1
comptime V_DIRTY_COPYK = 12 # V_COPYK after the same dirty buffer
comptime V_COUNT = 13


def variant_name(v: Int) -> String:
    var names: List[String] = [
        "old", "sleep", "readback", "l_host", "b_host", "back_host", "l_first",
        "serial", "prealloc", "root_rank0", "copyk", "dirty", "dirty_copyk",
    ]
    return names[v]


def unit_cell(i: Int, k: Int) -> Float32:
    return Float32(((i * 31 + k * 17) % 97)) / Float32(1940.0) - Float32(0.025)


def make_factor(n: Int) -> List[Float32]:
    var l = List[Float32]()
    for i in range(n):
        for k in range(n):
            if k < i:
                l.append(unit_cell(i, k))
            elif k == i:
                l.append(Float32(1.0) + Float32(i % 7) / Float32(8.0))
            else:
                l.append(Float32(-3.5))
    return l^


def make_rhs(n: Int, nrhs: Int) -> List[Float32]:
    var b = List[Float32]()
    for i in range(n * nrhs):
        b.append(Float32(((i * 2654435761) % 2003)) / Float32(500.0) - Float32(2.0))
    return b^


def upload(ctx: DeviceContext, values: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        host.unsafe_ptr()[i] = values[i]
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def download(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int) raises -> List[Float32]:
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    var view = buf.create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(host.unsafe_ptr()[i])
    _ = view^
    _ = host^
    return out^


def launch_trsm(ctx: DeviceContext, mut l: DeviceBuffer[DType.float32], mut b: DeviceBuffer[DType.float32],
                n: Int, nrhs: Int) raises:
    var tpb = CHOL_SOLVE_TPB
    ctx.enqueue_function[trsm_lower_kernel](
        l.unsafe_ptr(), b.unsafe_ptr(), Int32(n), Int32(nrhs), Int32(n),
        grid_dim=((nrhs + tpb - 1) // tpb, 1, 1), block_dim=(tpb, 1, 1),
    )


def launch_copy(ctx: DeviceContext, mut dst: DeviceBuffer[DType.float32], mut src: DeviceBuffer[DType.float32],
                cells: Int) raises:
    ctx.enqueue_function[copy_cells_kernel](dst.unsafe_ptr(), src.unsafe_ptr(), Int32(cells),
        grid_dim=((cells + 255) // 256, 1, 1), block_dim=(256, 1, 1))


def wait_host(seconds: Int):
    var until = Int(perf_counter_ns()) + seconds * 1000000000
    var spins = 0
    while Int(perf_counter_ns()) < until:
        spins += 1


def one_device_forward(l: List[Float32], b: List[Float32], n: Int, nrhs: Int) raises -> List[Float32]:
    var ctx = DeviceContext()
    var dl = upload(ctx, l)
    var db = upload(ctx, b)
    launch_trsm(ctx, dl, db, n, nrhs)
    ctx.synchronize()
    var x = download(ctx, db, n * nrhs)
    _ = dl^
    _ = db^
    _ = ctx^
    return x^


def dirty_device_one(n: Int) raises:
    """Create, fill with 7.25 and free a factor-sized buffer on device 1 in a context that dies."""
    var dev = DeviceContext(device_id=1)
    var junk = dev.enqueue_create_buffer[DType.float32](n * n)
    dev.enqueue_function[fill_kernel](junk.unsafe_ptr(), Int32(n * n),
        grid_dim=((n * n + 255) // 256, 1, 1), block_dim=(256, 1, 1))
    dev.synchronize()
    _ = junk^
    dev.synchronize()
    _ = dev^


@fieldwise_init
struct Shard(Movable):
    var ctx: DeviceContext
    var l: DeviceBuffer[DType.float32]
    var b: DeviceBuffer[DType.float32]
    var first: Int
    var width: Int

    def __deinit__(deinit self):
        _ = self.b^
        _ = self.l^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


def columns_of(b: List[Float32], n: Int, nrhs: Int, first: Int, width: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        for c in range(width):
            out.append(b[i * nrhs + first + c])
    return out^


def two_device_forward(lh: List[Float32], bh: List[Float32], n: Int, nrhs: Int, v: Int) raises -> List[Float32]:
    """The old `_cho_solve_columns` forward stage with the variant `v` applied.

    For V_COPYK and V_DIRTY_COPYK the return holds, per owner, the count of
    factor cells and column cells its copy kernel read differently from the
    host values, and the count of factor cells read as +0.0 and as 7.25, as
    [rank0 factor bad, rank0 columns bad, rank0 zeros, rank0 sevens, rank1 ...].
    """
    if v == V_DIRTY or v == V_DIRTY_COPYK:
        dirty_device_one(n)
    var ctx = DeviceContext()
    var l = upload(ctx, lh)
    var b = upload(ctx, bh)
    ctx.synchronize()
    var active = min(2, nrhs)
    var shards = List[Shard]()
    var report = List[Float32]()
    for rank in range(active):
        var first = nrhs * rank // active
        var width = nrhs * (rank + 1) // active - first
        var device = DeviceContext(device_id=rank)
        if v == V_ROOT_RANK0 and rank == 0:
            device = ctx.copy()
        var sb: DeviceBuffer[DType.float32]
        var sl: DeviceBuffer[DType.float32]
        if v == V_PREALLOC:
            sb = device.enqueue_create_buffer[DType.float32](n * width)
            sl = device.enqueue_create_buffer[DType.float32](n * n)
            device.synchronize()
            var packed = chol_gather_columns(ctx, b, n, nrhs, first, width)
            packed.enqueue_copy_to(sb)
            l.enqueue_copy_to(sl)
            ctx.synchronize()
            device.synchronize()
            ctx.synchronize()
            _ = packed^
        elif v == V_L_HOST:
            var packed = chol_gather_columns(ctx, b, n, nrhs, first, width)
            sb = peer_clone(ctx, device, packed)
            sl = upload(device, lh)
            device.synchronize()
            ctx.synchronize()
            _ = packed^
        elif v == V_B_HOST:
            sb = upload(device, columns_of(bh, n, nrhs, first, width))
            sl = peer_clone(ctx, device, l)
            device.synchronize()
            ctx.synchronize()
        elif v == V_L_FIRST:
            var packed = chol_gather_columns(ctx, b, n, nrhs, first, width)
            sl = peer_clone(ctx, device, l)
            sb = peer_clone(ctx, device, packed)
            device.synchronize()
            ctx.synchronize()
            _ = packed^
        else:
            var packed = chol_gather_columns(ctx, b, n, nrhs, first, width)
            sb = peer_clone(ctx, device, packed)
            sl = peer_clone(ctx, device, l)
            device.synchronize()
            ctx.synchronize()
            _ = packed^
        if v == V_READBACK:
            var gotl = download(device, sl, n * n)
            var gotb = download(device, sb, n * width)
            var bad = 0
            for i in range(n * n):
                if bitcast[DType.uint32](gotl[i]) != bitcast[DType.uint32](lh[i]):
                    bad += 1
            var colsh = columns_of(bh, n, nrhs, first, width)
            for i in range(n * width):
                if bitcast[DType.uint32](gotb[i]) != bitcast[DType.uint32](colsh[i]):
                    bad += 1
            print("PEERSOLVE readback rank", rank, "n", n, "cells differing before the kernel", bad)
        shards.append(Shard(device^, sl^, sb^, first, width))
    if v == V_SLEEP:
        wait_host(2)
    if v == V_COPYK or v == V_DIRTY_COPYK:
        for rank in range(active):
            ref s = shards[rank]
            var outl = s.ctx.enqueue_create_buffer[DType.float32](n * n)
            var outb = s.ctx.enqueue_create_buffer[DType.float32](n * s.width)
            launch_copy(s.ctx, outl, s.l, n * n)
            launch_copy(s.ctx, outb, s.b, n * s.width)
            s.ctx.synchronize()
            var gotl = download(s.ctx, outl, n * n)
            var gotb = download(s.ctx, outb, n * s.width)
            var badl = 0
            var zeros = 0
            var sevens = 0
            for i in range(n * n):
                if bitcast[DType.uint32](gotl[i]) != bitcast[DType.uint32](lh[i]):
                    badl += 1
                if gotl[i] == Float32(0.0):
                    zeros += 1
                if gotl[i] == Float32(7.25):
                    sevens += 1
            var colsh = columns_of(bh, n, nrhs, s.first, s.width)
            var badb = 0
            for i in range(n * s.width):
                if bitcast[DType.uint32](gotb[i]) != bitcast[DType.uint32](colsh[i]):
                    badb += 1
            report.append(Float32(badl))
            report.append(Float32(badb))
            report.append(Float32(zeros))
            report.append(Float32(sevens))
            _ = outl^
            _ = outb^
        _ = shards^
        _ = b^
        _ = l^
        _ = ctx^
        return report^
    for rank in range(active):
        ref s = shards[rank]
        launch_trsm(s.ctx, s.l, s.b, n, s.width)
        if v == V_SERIAL:
            s.ctx.synchronize()
    var x = List[Float32]()
    if v == V_BACK_HOST:
        x = download(ctx, b, n * nrhs)
    for rank in range(active):
        ref s = shards[rank]
        s.ctx.synchronize()
        if v == V_BACK_HOST:
            var part = download(s.ctx, s.b, n * s.width)
            for i in range(n):
                for c in range(s.width):
                    x[i * nrhs + s.first + c] = part[i * s.width + c]
        else:
            var staged = ctx.enqueue_create_buffer[DType.float32](n * s.width)
            ctx.synchronize()
            s.b.enqueue_copy_to(staged)
            s.ctx.synchronize()
            ctx.synchronize()
            chol_scatter_columns(ctx, b, staged, n, nrhs, s.first, s.width)
            _ = staged^
    if v != V_BACK_HOST:
        ctx.synchronize()
        x = download(ctx, b, n * nrhs)
    _ = shards^
    _ = b^
    _ = l^
    _ = ctx^
    return x^


def peersolve(n: Int, nrhs: Int) raises -> Int:
    var lh = make_factor(n)
    var bh = make_rhs(n, nrhs)
    var ref_x = one_device_forward(lh, bh, n, nrhs)
    var failing = 0
    for v in range(V_COUNT):
        var got = two_device_forward(lh, bh, n, nrhs, v)
        if v == V_COPYK or v == V_DIRTY_COPYK:
            var line = "PEERSOLVE n " + String(n) + " nrhs " + String(nrhs) + " variant " + variant_name(v)
            var bad_any = False
            for rank in range(len(got) // 4):
                line += " | rank " + String(rank) + " factor differing " + String(Int(got[rank * 4]))
                line += " columns differing " + String(Int(got[rank * 4 + 1]))
                line += " factor zeros " + String(Int(got[rank * 4 + 2]))
                line += " factor sevens " + String(Int(got[rank * 4 + 3]))
                if got[rank * 4] != Float32(0.0) or got[rank * 4 + 1] != Float32(0.0):
                    bad_any = True
            print(line)
            if bad_any:
                failing += 1
            continue
        var line = "PEERSOLVE n " + String(n) + " nrhs " + String(nrhs) + " variant " + variant_name(v)
        var bad_any = False
        for j in range(nrhs):
            var count = 0
            var first = -1
            for i in range(n):
                if bitcast[DType.uint32](got[i * nrhs + j]) != bitcast[DType.uint32](ref_x[i * nrhs + j]):
                    count += 1
                    if first < 0:
                        first = i
            line += " | column " + String(j) + " rows differing " + String(count) + " first " + String(first)
            if count > 0:
                bad_any = True
                if first >= 0:
                    line += " got " + String(got[first * nrhs + j]) + " want " + String(ref_x[first * nrhs + j])
        print(line)
        if bad_any:
            failing += 1
    return failing


def main() raises:
    if String(getenv("RUNPOD_POD_ID")) == "":
        raise Error("RunPod required; no local execution")
    var failures = 0
    if String(getenv("MOJOLEARN_PEERCOPY_SKIP_BARE", "0")) != "1":
        var sizes: List[Int] = [4096, 65536, 262144, 262145, 263169, 524288, 1048576, 4194304]
        for n in sizes:
            var a = check(n, False)
            var b = check(n, True)
            print("PEERCOPY cells", n, "bytes", n * 4, "peer_clone differing", a, "host_staged differing", b)
            if a != 0 or b != 0:
                failures += 1
        print("PEERCOPY cases with a difference:", failures)
    var solve_failures = 0
    var ns: List[Int] = [512, 513, 1024]
    for n in ns:
        solve_failures += peersolve(n, 2)
    print("PEERSOLVE cases with a difference:", solve_failures)
