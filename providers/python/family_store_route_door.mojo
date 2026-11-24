# SPDX-License-Identifier: Apache-2.0
"""UNPACK of a member family's resident store (lane vdf-scope-cluster2b, 2026-11-24).

A family image with its own device store (core/family_store_transport.mojo: mixture, density, spectral,
manifold) exports the store's transport entry as a descriptor capsule (`<family>_store_v1`, ABI
`XP_TRANSPORT_ABI`, the x_prep transport numbering). The family's member boundary connects it ONCE per process
here, when the image answers its family's store capability, and hands it to the member in `route.second`
(`has_second`): the member then keeps a scoped X in the store (`arena_scope_store_put`) and takes each fold's rows
there (`arena_store_take_rows`). The returned provider is a pin the caller keeps with the member's other pins.

Serial by nature: one capsule read and one cached connection per member admission.
"""
from std.python import PythonObject
from core.device_route import DeviceRoute
from core.entry_table import EntryRef, entry_ref_word
from providers.python.neural_descriptor_pyapi_door import (
    NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect_factory_once,
)
from x_prep.transport_abi import XP_TRANSPORT_ABI, XP_OP_COUNT


def family_store_route(binding: PythonObject, entry_name: String, mut route: DeviceRoute) raises -> PythonObject:
    """Connect `binding`'s store transport `entry_name` and set it as `route.second` with the route's (mode,
    vendor) words; the provider (the caller pins it)."""
    var factory = neural_descriptor_entry(binding, entry_name)
    var provider = neural_descriptor_connect_factory_once(factory, String(XP_TRANSPORT_ABI), XP_OP_COUNT)
    var entry = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry
    route.second = EntryRef(entry, entry_ref_word(route.mode, route.vendor, XP_OP_COUNT))
    route.has_second = True
    return provider
