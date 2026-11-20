# SPDX-License-Identifier: Apache-2.0
"""Typed half-precision widening (from bindings/causal_checkpoint.mojo), split across host cores."""
from std.memory import bitcast
from gemm.f16_widen import f16_bits_to_f32
from core.host_parallel import host_parallelize


# Halves per host task: 65,536 (128 KB read, 256 KB written) keeps the pool
# hand-off (microseconds) under about 1% of a task's integer work at any
# tensor size; a tensor at or below one chunk runs on the calling thread
# (host_parallelize's one-task rule), so small reads pay no pool cost. Size
# and cost reasoning only, no shape of any checkpoint in mind.
comptime CHECKPOINT_WIDEN_CHUNK = 65536


def checkpoint_widen_halves(
    source: MutPointer[UInt16, MutUntrackedOrigin], dest: MutPointer[Float32, MutUntrackedOrigin],
    count: Int, dtype: String,
):
    # lane owed-deletions (2026-11-16): the dtype test moved out of the
    # element loop (it compared a String per element) and the elements split
    # into independent chunks across host cores. Each element is the same
    # integer bit construction as before (gemm/f16_widen.mojo for F16, `<< 16`
    # for BF16): bits unchanged, every column.
    var f16 = dtype == "F16"
    var chunks = (count + CHECKPOINT_WIDEN_CHUNK - 1) // CHECKPOINT_WIDEN_CHUNK

    def task(chunk: Int) {imm source, imm dest, imm count, imm f16}:
        var first = chunk * CHECKPOINT_WIDEN_CHUNK
        var last = first + CHECKPOINT_WIDEN_CHUNK
        if last > count:
            last = count
        if f16:
            for i in range(first, last):
                dest.unsafe_store(i, f16_bits_to_f32(source.unsafe_load(i)))
        else:
            for i in range(first, last):
                dest.unsafe_store(i, bitcast[DType.float32](UInt32(source.unsafe_load(i)) << 16))

    host_parallelize(task, chunks)
