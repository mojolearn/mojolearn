# SPDX-License-Identifier: Apache-2.0
# The AUC sort's in-block stages on one threadgroup (lane ag-pass6-rows,
# pass-6 idea I, MOJOLEARN_IDN_AUC_BLOCK_SORT; autograd/row_ops.mojo has the
# define, the launch arithmetic and op_row_bitonic_block, the host twin).
# DeviceExec launches it for OP_ROW_BITONIC_BLOCK: one block of B / 2
# threads per B elements, the block's keys and labels in threadgroup memory,
# one barrier after each stage. Thread q owns the stage's pair i = 2 j (q //
# j) + q % j of the block (the host twin's q), and the ascending test reads
# the GLOBAL index, so every compare-exchange is op_ls_bitonic's on the
# same values: the same words as one launch per stage. Values are only
# moved, never computed.
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from sequence.ops import FP, ld, st
from autograd.row_ops import ROW_SORT_BLOCK, row_sort_after


def row_bitonic_block_kernel(keys: FP, labels: FP, k_first: Int32, k_last: Int32, block_elems: Int32):
    """Block `block_idx.x` of the in-block stages (op_row_bitonic_block's
    arguments: merge sizes k_first .. k_last, B = block_elems)."""
    var B = Int(block_elems)
    var h = B // 2
    var lo = Int(block_idx.x) * B
    var q = Int(thread_idx.x)
    var ks = stack_allocation[ROW_SORT_BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var ls = stack_allocation[ROW_SORT_BLOCK, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    ks[q] = ld(keys, lo + q)
    ls[q] = ld(labels, lo + q)
    ks[q + h] = ld(keys, lo + q + h)
    ls[q + h] = ld(labels, lo + q + h)
    barrier()
    var k = Int(k_first)
    while k <= Int(k_last):  # uniform over the block
        var j = min(k, B) // 2
        while j > 0:
            var i = 2 * j * (q // j) + q % j
            var l = i + j
            var ka = ks[i]
            var la = ls[i]
            var kb = ks[l]
            var lb = ls[l]
            var swap: Bool
            if ((lo + i) & k) == 0:
                swap = row_sort_after(ka, la, kb, lb)
            else:
                swap = row_sort_after(kb, lb, ka, la)
            if swap:
                ks[i] = kb
                ls[i] = lb
                ks[l] = ka
                ls[l] = la
            barrier()
            j //= 2
        k *= 2
    st(keys, lo + q, ks[q])
    st(labels, lo + q, ls[q])
    st(keys, lo + q + h, ks[q + h])
    st(labels, lo + q + h, ls[q + h])
