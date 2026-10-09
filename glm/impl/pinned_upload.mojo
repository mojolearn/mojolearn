# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lane fg-linear L4 (IDN_LINEAR_PINNED_UPLOAD, default on in IDENTICAL off
Apple; experiments/classical_identical_ideas/fg_linear_controls.mojo): the
linear fits' host-to-device uploads through a pinned double-buffered stage.

NO ARITHMETIC: copies and waits only, so no bit of any fit can move. The
stage is one process-lifetime pinned host buffer of two LPU_STAGE_FLOATS
halves (made on the first large upload, kept for the process: a pinned
allocation per call would cost more than it saves). A busy flag guards it;
a concurrent second caller (two Python threads, the GIL is released) takes
the direct copy instead."""
from std.atomic import Atomic, Ordering
from std.ffi import _Global
from std.memory import memcpy
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from experiments.classical_identical_ideas.fg_linear_controls import IDN_LINEAR_PINNED_UPLOAD

comptime _LP = MutPointer[Float32, MutUntrackedOrigin]

#: 8M floats = 32 MB per stage half.
comptime LPU_STAGE_FLOATS = 1 << 23
#: Uploads smaller than this (4 MB) go direct.
comptime LPU_MIN_FLOATS = 1 << 20

comptime LPU_ON = IDN_LINEAR_PINNED_UPLOAD and not has_apple_gpu_accelerator()


struct _LinearStage(Defaultable, Movable):
    var busy: Int64
    var stage: Optional[HostBuffer[DType.float32]]

    def __init__(out self):
        self.busy = 0
        self.stage = Optional[HostBuffer[DType.float32]]()


comptime LINEAR_STAGE = _Global[StorageType=_LinearStage, name="MojoLinearPinnedStage", init_fn=_LinearStage.__init__]


def linear_upload_f32(
    ctx: DeviceContext, mut dst: DeviceBuffer[DType.float32], src: MutPointer[Float32, MutUntrackedOrigin], n: Int
) raises:
    """Host floats `src[0, n)` into `dst[0, n)` (the whole of `dst` when
    `n == len(dst)`). Off, small, or with the stage busy: the direct
    `enqueue_copy` (enqueued, no wait), exactly the old statement. Staged: the
    copies are retired before return (the stage is free for the next call);
    kernels the caller enqueues next are ordered after them either way."""
    comptime if LPU_ON:
        if n >= LPU_MIN_FLOATS:
            var p = LINEAR_STAGE.get_or_create_ptr()
            var flag = UnsafePointer(to=p[].busy)
            var old = Atomic.fetch_add[ordering = Ordering.SEQUENTIAL](flag, Int64(1))
            if old == 0:
                if not p[].stage:
                    p[].stage = ctx.enqueue_create_host_buffer[DType.float32](2 * LPU_STAGE_FLOATS)
                    ctx.synchronize()
                var stage = _LP(unsafe_from_address=Int(p[].stage.value().unsafe_ptr()))
                # Double-buffered: chunk i is written into its half while
                # chunk i - 1's DMA runs; the wait before DMA i retires DMA
                # i - 1, so when chunk i + 1 refills that half nothing reads it.
                var off = 0
                var i = 0
                while off < n:
                    var cnt = min(LPU_STAGE_FLOATS, n - off)
                    var half = stage + (i % 2) * LPU_STAGE_FLOATS
                    memcpy(dest=half, src=src + off, count=cnt)
                    if i > 0:
                        ctx.synchronize()
                    ctx.enqueue_copy(dst_buf=dst.create_sub_buffer[DType.float32](off, cnt), src_ptr=half)
                    off += cnt
                    i += 1
                ctx.synchronize()
                _ = Atomic.fetch_add[ordering = Ordering.SEQUENTIAL](flag, Int64(-1))
                return
            _ = Atomic.fetch_add[ordering = Ordering.SEQUENTIAL](flag, Int64(-1))
    if n == len(dst):
        ctx.enqueue_copy(dst_buf=dst, src_ptr=src)
    else:
        ctx.enqueue_copy(dst_buf=dst.create_sub_buffer[DType.float32](0, n), src_ptr=src)
