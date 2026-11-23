# SPDX-License-Identifier: Apache-2.0
"""The native owner of SAGEConv's saved host-form forward state (lane
bm-rest-graph, 2026-11-23).

`xcnn_sage_forward` (bindings/xcnn_graph_native_door.mojo) runs the typed
forward (x_cnn/graph_layer_run.mojo `sage_forward_host_run`) into one
`SageSaved` and stores ONE `SageSavedOwner` in the layer's `_sage_saved`
field; `_sage_backward_host` reads it back with one downcast and hands the
typed state to `sage_backward_saved_run`. Before, the same buffers were nine
Python arrays in the fields of the layer, lin_l, lin and the ReLU.

`keep` holds the returned output when the forward normalizes: the saved
state's `y` address points into it (the forward wrote it there and the
backward's tape records it), exactly as the layer's `_y` field kept that
same array before. It is a PACK-side reference: the RUN sees only the
address.

The type is registered ONCE, by `_mojolearn_runtime_xseq_host` (the image
that publishes `xcnn_native_call`, the only door that allocates it);
registration looks the type up first, the pattern of
bindings/member_held_owner_unpack.mojo. SERIAL BY NATURE: one type
registration per process.
"""
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.py_type_registry import py_type_absent, py_type_attach
from x_cnn.graph_layer_run import SageSaved


struct SageSavedOwner(Movable, Writable):
    """One SAGEConv host-form forward's saved state and the returned array
    its `y` address points into (None without normalize)."""

    var saved: SageSaved
    var keep: PythonObject

    def __init__(out self, var saved: SageSaved, keep: PythonObject):
        self.saved = saved^
        self.keep = keep

    def write_to(self, mut writer: Some[Writer]):
        writer.write("SageSavedOwner(native)")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def register_sage_saved_owner(mut module: PythonModuleBuilder) raises:
    """Module setup of the ONE registering image (before `finalize()`)."""
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[SageSavedOwner]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[SageSavedOwner]("_SageSavedOwner")
        return
    py_type_attach(module.module, "_SageSavedOwner", existing)


def sage_saved_owner_new(var saved: SageSaved, keep: PythonObject) raises -> PythonObject:
    """PACK: the owner the layer's `_sage_saved` field holds."""
    var owner = PythonObject(alloc=SageSavedOwner(saved^, keep))
    return owner
