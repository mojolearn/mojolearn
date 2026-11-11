# SPDX-License-Identifier: Apache-2.0
"""Fixed native request envelopes shared by neighbor algorithm plans."""
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from x_neighbors.descriptor_schema import xn_operation_id


@fieldwise_init
struct XnDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def run[NA: Int, NP: Int, NF: Int](self, name: String,
        a: InlineArray[Int, NA], p: InlineArray[Int, NP], f: InlineArray[Float64, NF],
    ) raises -> Int:
        return native_descriptor_call_fixed(self.entry, xn_operation_id(name), a, p, f)
