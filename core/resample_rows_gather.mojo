# SPDX-License-Identifier: Apache-2.0
"""The general row gather of `ResampleRows.take`
(providers/python/resample_rows_pyapi.mojo), moved here by lane
i3-move-data-loops (2026-11-09): a source whose rows are not one contiguous
run of bytes (strided trailing axes or PEP 3118 suboffsets) and holds no
object references. The bytes as they are, item by item in C order of the
trailing axes; the rows split across the host's cores (every output row is
a function of its own source row, so the split moves no bit). No Python."""
from std.memory import unsafe_memcpy
from core.borrowed_buffer import BorrowedBufferLayout
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk


def resample_gather_rows(layout: BorrowedBufferLayout, rows: MutPointer[Int32, MutUntrackedOrigin], count: Int,
                         row_items: Int, dst: Int):
    """Source row `rows[i]` (already checked in [0, n)) to output row `i`, a
    C-contiguous run of `row_items * layout.itemsize` bytes at
    `dst + i * row_items * layout.itemsize`."""
    if count <= 0 or row_items <= 0:
        return
    var base = layout.address
    var itemsize = layout.itemsize
    var rank = len(layout.shape)
    var shape_ptr = layout.shape.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var strides_ptr = layout.strides.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var sub_ptr = layout.suboffsets.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var tasks = host_predict_task_count_work(count, count * row_items * max(rank, 1))
    var chunk = host_predict_chunk(count, tasks)

    def _task(c: Int) {imm base, imm itemsize, imm rank, imm shape_ptr, imm strides_ptr, imm sub_ptr, imm rows,
                       imm count, imm row_items, imm dst, imm chunk}:
        var lo = c * chunk
        var hi = min(lo + chunk, count)
        var coordinates = List[Int](length=rank, fill=0)
        for i in range(lo, hi):
            for axis in range(rank):
                coordinates[axis] = 0
            coordinates[0] = Int(rows[i])
            for item in range(row_items):
                var at = dst + (i * row_items + item) * itemsize
                # BorrowedBufferLayout.item_address, word for word
                var address = base
                for axis in range(rank):
                    address += coordinates[axis] * strides_ptr[axis]
                    if sub_ptr[axis] >= 0:
                        address = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=address)[0] + sub_ptr[axis]
                unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=at),
                    src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=address), count=itemsize)
                for reverse in range(rank - 1):
                    var axis = rank - reverse - 1
                    coordinates[axis] += 1
                    if coordinates[axis] < shape_ptr[axis]:
                        break
                    coordinates[axis] = 0

    host_parallelize(_task, tasks)
