# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/lowbit-amd-tuned: is the MI325X's device fault a property of a
process's FIRST LAUNCH OF A KERNEL THAT USES SCRATCH, with no mojolearn code
in it?

    pixi run mojo build -D MOJOLEARN_SCRATCH=1 -I . tools/lowbit_amd_tuned/scratch_first.mojo -o scratch
    pixi run mojo build -I . tools/lowbit_amd_tuned/scratch_first.mojo -o noscratch

One kernel, one block of 256 threads, launched once in a fresh process. With
`MOJOLEARN_SCRATCH` a lane's 256-word array is indexed by a value known only
at run time, which puts it in scratch (private) memory; without, every index
is a constant and the array lives in registers. Each prints one line and the
store it checks. `tools/lowbit_amd_tuned/scratch_first.sh` runs each binary
in many fresh processes and counts the faults.
"""

from std.gpu import thread_idx
from std.sys import is_defined
from max.gpu.host import DeviceContext

comptime SCRATCH = is_defined["MOJOLEARN_SCRATCH"]()
comptime WORDS = 256


def probe_kernel(out: MutPointer[Int32, MutAnyOrigin], salt_in: Int32):
    var t = Int(thread_idx.x)
    var salt = Int(salt_in)
    var arr = InlineArray[Int32, WORDS](fill=Int32(0))
    comptime if SCRATCH:
        for i in range(WORDS):
            arr[(i * 7 + t + salt) & (WORDS - 1)] += Int32(i)
        out.unsafe_store(t, arr[(t * 3 + salt) & (WORDS - 1)])
    else:
        comptime for i in range(WORDS):
            arr[i] += Int32(i + t + salt)
        var acc = Int32(0)
        comptime for i in range(WORDS):
            acc += arr[i]
        out.unsafe_store(t, acc)


def main() raises:
    var ctx = DeviceContext()
    var d = ctx.enqueue_create_buffer[DType.int32](256)
    ctx.enqueue_function[probe_kernel](d.unsafe_ptr(), Int32(0), grid_dim=(1, 1, 1), block_dim=(256, 1, 1))
    ctx.synchronize()
    var h = ctx.enqueue_create_host_buffer[DType.int32](256)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d)
    ctx.synchronize()
    var want = Int32(0)
    comptime if SCRATCH:
        # Thread 0 reads word 0 (salt 0): the steps i with 7 i = 0 mod 256,
        # i = 0 only, so the word holds 0.
        want = Int32(0)
    else:
        # the sum over i of (i + t + salt) at t = 0, salt = 0.
        want = Int32(WORDS * (WORDS - 1) // 2)
    var got = h.unsafe_ptr().unsafe_load(0)
    print("scratch_first scratch=" + String(SCRATCH) + " cell0=" + String(got) + " want=" + String(want))
    _ = h
    _ = d
    if got != want:
        raise Error("wrong cell")
