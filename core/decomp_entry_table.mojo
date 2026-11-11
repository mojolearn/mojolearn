# SPDX-License-Identifier: Apache-2.0
"""The x_decomp operations table for native callers (lane calltable-prep-decomp,
2026-11-11).

Both x_decomp images already publish their one descriptor dispatch
(`x_decomp/descriptor_device.mojo::x_decomp_device_v1`,
`x_decomp/descriptor_host.mojo::x_decomp_host_v1`, ABI in
`x_decomp/descriptor_schema.mojo`) at module init, in the linear family's
`LINEAR_TABLE_DECOMP` slots (`bindings/_mojolearn_x_decomp{,_host}.mojo`
`_publish_linear_entry_table`). This file is the decomposition family's
reader of that publication: no second slot, no second publication. A caller
reads the slot of the binding its owner connected (the key is the decomp
bridge's `x_decomp_native_key`, python/mojolearn/_native_decomp_kit_bridge.py)
and calls the pointer with plain spans; no Python object crosses.

Lookup is a lock-free acquire load and a schema compare per call, a small
metadata operation with no independent numerical work to split.
"""
from core.linear_entry_table import LinearEntryKey, linear_table_entry, LINEAR_TABLE_DECOMP
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from x_decomp.descriptor_schema import XD_ABI_NAME, XD_OPERATION_COUNT


def decomp_table_entry(key: LinearEntryKey) raises -> NativeDescriptorCall:
    """The x_decomp dispatch of the image `key` names; a missing or
    mismatched publication refuses with the table's error."""
    return linear_table_entry[LINEAR_TABLE_DECOMP](key, 0, String(XD_ABI_NAME), XD_OPERATION_COUNT)


def decomp_table_call(key: LinearEntryKey, operation: Int, addresses: List[Int], ints: List[Int],
                      floats: List[Float64]) raises -> Int:
    """One x_decomp operation (an `XD_*` code) in the image `key` names. The
    caller keeps every addressed buffer alive until the call returns."""
    var entry = decomp_table_entry(key)
    return native_descriptor_call(entry, operation, addresses, ints, floats)
