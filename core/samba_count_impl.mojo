# SPDX-License-Identifier: Apache-2.0
# The same exact-integer target count used by the public Samba boundary.
from core.hotpath_impl import equal_elements, reduce_stat


def samba_target_count(address: Int, rows: Int) raises -> Int:
    var ignored = List[Int32](length=rows, fill=-100)
    var mask = List[UInt8](length=rows, fill=0)
    _ = equal_elements(address, Int(ignored.unsafe_ptr()), 2, rows, Int(mask.unsafe_ptr()))
    var count = reduce_stat(Int(mask.unsafe_ptr()), 5, rows, 5)
    if count.hi != 0 or count.mid != 0:
        raise Error("Samba: ignored-target count exceeds kernel indexing")
    return rows - count.low
