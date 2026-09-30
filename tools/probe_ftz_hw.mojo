# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Is NVIDIA's `mul.rn.ftz.f32 x, 1.0` the word `ftz(x)` returns, for EVERY x?

lane/neural-net-experiment (2026-09-30, the S16 pass). `checks.numerics.ftz`
spells the denormal policy on the NVIDIA column as that one instruction
(the GEMM seam's spelling since DEVIATION 2706) where it used to be the
six-operation integer test. There are 2^32 float32 words; this probe runs
both spellings over all of them on the device and counts the words that
differ, NaN words (exponent 255, mantissa nonzero) counted apart from the
rest, because the arithmetic unit may canonicalize a NaN payload and the
identity contract refuses NaN payloads anyway (row 39). The verdict:

  PASS  no non-NaN word differs (the NaN count is reported, not judged)
  FAIL  a non-NaN word differs; the first few are printed as hex

It needs no build define and runs the integer spelling itself, so it does
not depend on the build's numeric mode.  Run:  mojo run tools/probe_ftz_hw.mojo
On a non-NVIDIA device it prints SKIP (both lanes are the integer spelling).
"""

from max.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from std.sys import llvm_intrinsic
from std.sys.info import is_nvidia_gpu

comptime THREADS = 1 << 20
comptime WORDS_PER_THREAD = 1 << 12  # THREADS x WORDS_PER_THREAD = 2^32


def _ftz_int(x: Float32) -> Float32:
    var b = bitcast[DType.uint32](x)
    if (b & UInt32(0x7F800000)) == UInt32(0) and (b & UInt32(0x007FFFFF)) != UInt32(0):
        return bitcast[DType.float32](b & UInt32(0x80000000))
    return x


def probe_kernel(
    diff_plain: MutPointer[Int32, MutAnyOrigin],
    diff_nan: MutPointer[Int32, MutAnyOrigin],
    first_plain: MutPointer[UInt32, MutAnyOrigin],
    first_seen: MutPointer[Int32, MutAnyOrigin],
):
    """Thread t: the words t * WORDS_PER_THREAD .. + WORDS_PER_THREAD."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= THREADS:
        return
    var plain = Int32(0); var nan = Int32(0); var first = UInt32(0); var seen = Int32(0)
    var base = UInt32(t) * UInt32(WORDS_PER_THREAD)
    for i in range(WORDS_PER_THREAD):
        var w = base + UInt32(i)
        var x = bitcast[DType.float32](w)
        var soft = bitcast[DType.uint32](_ftz_int(x))
        var hard = soft
        comptime if is_nvidia_gpu():
            hard = bitcast[DType.uint32](llvm_intrinsic["llvm.nvvm.mul.rn.ftz.f", Float32, has_side_effect=False](x, Float32(1.0)))
        if hard != soft:
            if (w & UInt32(0x7F800000)) == UInt32(0x7F800000) and (w & UInt32(0x007FFFFF)) != UInt32(0):
                nan += Int32(1)
            else:
                plain += Int32(1)
                if seen == Int32(0):
                    first = w; seen = Int32(1)
    diff_plain.unsafe_store(t, plain); diff_nan.unsafe_store(t, nan)
    first_plain.unsafe_store(t, first); first_seen.unsafe_store(t, seen)


def main() raises:
    var ctx = DeviceContext()
    var dev = ctx.name()
    var d_plain = ctx.enqueue_create_buffer[DType.int32](THREADS)
    var d_nan = ctx.enqueue_create_buffer[DType.int32](THREADS)
    var d_first = ctx.enqueue_create_buffer[DType.uint32](THREADS)
    var d_seen = ctx.enqueue_create_buffer[DType.int32](THREADS)
    ctx.synchronize()
    ctx.enqueue_function[probe_kernel](d_plain.unsafe_ptr(), d_nan.unsafe_ptr(), d_first.unsafe_ptr(), d_seen.unsafe_ptr(), grid_dim=(THREADS // 256, 1, 1), block_dim=(256, 1, 1))
    ctx.synchronize()
    var h_plain = ctx.enqueue_create_host_buffer[DType.int32](THREADS)
    var h_nan = ctx.enqueue_create_host_buffer[DType.int32](THREADS)
    var h_first = ctx.enqueue_create_host_buffer[DType.uint32](THREADS)
    var h_seen = ctx.enqueue_create_host_buffer[DType.int32](THREADS)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=h_plain.unsafe_ptr(), src_buf=d_plain); ctx.enqueue_copy(dst_ptr=h_nan.unsafe_ptr(), src_buf=d_nan)
    ctx.enqueue_copy(dst_ptr=h_first.unsafe_ptr(), src_buf=d_first); ctx.enqueue_copy(dst_ptr=h_seen.unsafe_ptr(), src_buf=d_seen)
    ctx.synchronize()
    var plain = 0; var nan = 0; var shown = 0
    for t in range(THREADS):
        plain += Int(h_plain.unsafe_ptr().unsafe_load(t)); nan += Int(h_nan.unsafe_ptr().unsafe_load(t))
        if h_seen.unsafe_ptr().unsafe_load(t) != Int32(0) and shown < 16:
            shown += 1
            print("DIFF word", hex(h_first.unsafe_ptr().unsafe_load(t)))
    print("words 4294967296  non-NaN words that differ:", plain, "  NaN words that differ:", nan)
    if "NVIDIA" not in dev and "nvidia" not in dev:
        print("PROBE_FTZ_HW SKIP: device", dev, "is not NVIDIA; both lanes ran the integer spelling")
    elif plain == 0:
        print("PROBE_FTZ_HW PASS on", dev, ": mul.rn.ftz by one returns ftz's word for every non-NaN float32")
    else:
        print("PROBE_FTZ_HW FAIL on", dev, ":", plain, "non-NaN words differ; build with -D MOJOLEARN_FTZ_HW_OFF=1")
        raise Error("ftz hardware probe: mismatches")
