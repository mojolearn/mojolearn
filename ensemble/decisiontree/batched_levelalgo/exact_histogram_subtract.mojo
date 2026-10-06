# SPDX-License-Identifier: Apache-2.0
"""I18 exact sibling algebra for unweighted ClassificationBin counts.
No weighted statistics/gradients are admissible. Supported forest row count
is <=Int32.max, so each class/bin count and parent sum fits UInt32 exactly.
The same sampled-feature fingerprint is a mandatory caller admission.
This enqueued operator leaves invalid cells untouched and emits one status
byte per cell; the caller must validate statuses before split evaluation."""
from std.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys.compile import is_defined

def exact_histogram_subtract_kernel(parent: MutPointer[UInt32, MutAnyOrigin], child: MutPointer[UInt32, MutAnyOrigin], sibling: MutPointer[UInt32, MutAnyOrigin], invalid: MutPointer[UInt8, MutAnyOrigin], cells: Int32):
    var i = Int(block_idx.x)*256+Int(thread_idx.x)
    if i < Int(cells):
        var p = parent[i]
        var c = child[i]
        var bad = c>p or p>UInt32(2147483647)
        invalid[i] = UInt8(1) if bad else UInt8(0)
        if not bad:
            sibling[i] = p-c

def enqueue_exact_histogram_subtract(ctx: DeviceContext, mut parent: DeviceBuffer[DType.uint32], mut child: DeviceBuffer[DType.uint32], mut sibling: DeviceBuffer[DType.uint32], mut invalid: DeviceBuffer[DType.uint8], cells: Int, rows_bound: Int, parent_features: UInt64, child_features: UInt64) raises:
    comptime assert is_defined["MOJOLEARN_TREE_EXACT_SIBLING_HIST"](), "I18 is an opt-in experiment"
    if cells<0 or rows_bound<0 or rows_bound>2147483647 or parent_features!=child_features:
        raise Error("I18 count bound or sampled-feature compatibility refused")
    if len(parent)<cells or len(child)<cells or len(sibling)<cells or len(invalid)<cells:
        raise Error("I18 histogram capacity refused")
    if cells>0:
        ctx.enqueue_function[exact_histogram_subtract_kernel](parent.unsafe_ptr(),child.unsafe_ptr(),sibling.unsafe_ptr(),invalid.unsafe_ptr(),Int32(cells),grid_dim=((cells+255)//256,1,1),block_dim=(256,1,1))
