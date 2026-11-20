# SPDX-License-Identifier: Apache-2.0
"""Copy admitted strided archive words into disjoint C-order storage.

Each destination word is independent. The existing host policy partitions
words across workers; source shape/strides and owners stay alive until join.
This is the shared gather previously inside NativeArchiveMember.prepare.
"""
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk


def archive_copy_bytes(source: Int, shape: List[Int], steps: List[Int], count: Int,
                       width: Int, destination: Int):
    var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=destination)
    var dimensions = shape.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var strides = steps.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var rank = len(shape)
    var tasks = host_predict_task_count_work(count, count * width)
    var chunk = host_predict_chunk(count, tasks)
    def gather(unit: Int) {imm dst, imm dimensions, imm strides, imm rank, imm width, imm source, imm count, imm chunk}:
        for i in range(unit * chunk, min(count, (unit + 1) * chunk)):
            var index = i
            var offset = 0
            for axis in range(rank - 1, -1, -1):
                offset += (index % dimensions[axis]) * strides[axis]
                index //= dimensions[axis]
            var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=source + offset)
            for byte in range(width):
                dst[i * width + byte] = src[byte]
    host_parallelize(gather, tasks)
    _ = shape
    _ = steps
