# SPDX-License-Identifier: Apache-2.0
# PCA fitted-state admission: one buffer metadata snapshot per fitted array,
# no cast or contiguity repair. X is admitted by decomp_admit.
from std.python import Python, PythonObject
from bindings.py_balanced import py_contains
from bindings.classical_validate import classical_view
from providers.python.array_metadata_pyapi import ArrayMetadata, array_shape_once, array_metadata_once
from core.pca_exact_impl import pca_check_state


# TOMBSTONE: pca_input_snapshot (PCA's own X admission: a zero-filled cast destination and its own cast/scan call; its refusals were admit_x's wording with est PCA) deleted 2026-11-08 by lane a3-admit-decomp-prep-metrics; code recoverable at f7d10d2b09.
# Restore: experiments/removed/A3_ADMISSION_HELPERS.patch. PCA reads
# bindings/decomp_inputs.mojo decomp_admit (admit_x_unrecorded).


def pca_state_snapshot(fields: PythonObject, name: String, expected: List[Int],
                       context: PythonObject, finite: Bool = False,
                       nonnegative: Bool = False) raises -> ArrayMetadata:
    if not py_contains(fields, name):
        raise Error("PCA fitted state is missing " + name)
    var source = fields[name]
    var view = classical_view(source, context, name)
    var shape = array_shape_once(view, context)
    var captured = array_metadata_once(view, shape^, name, context,
                                      allow_non_native=True, allow_unsupported=True)
    var same_shape = len(captured.shape) == len(expected)
    if same_shape:
        for axis in range(len(expected)):  # small-loop(fitted vector/matrix dimensions)
            same_shape = same_shape and captured.shape[axis] == expected[axis]
    # No cast or contiguity repair is permitted for fitted state. The public
    # contract requires the supplied fitted arrays themselves to be C f32.
    if not same_shape or not captured.castable or not captured.native_byte_order or not captured.f32_c():
        raise Error("PCA requires C-order float32 " + name + " with the fitted shape")
    captured.owner = Python.tuple(source, captured.owner)
    if finite or nonnegative:
        pca_check_state(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=captured.source.addr),
                        captured.count, name, finite, nonnegative)
    return captured^
