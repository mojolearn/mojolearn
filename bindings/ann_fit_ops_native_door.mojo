# SPDX-License-Identifier: Apache-2.0
"""The boundary of the ANN fit operations (lane dw-neighbors, 2026-11-21):
what the four fit doors of bindings/neighbors_ann_native_door.mojo
(`IVFPQIndex`, `IVFSQIndex`, `IVFRaBitQIndex`, `CagraIndex`) share to reach
the x_ann image through its operations table
(core/neighbors_entry_table_ann.mojo).

UNPACK is `ann_ops_key` (bindings/native_ops_key_ann_unpack.mojo: one exact
read of the image's module dictionary for the key it recorded when it
published) and one table lookup for the entry, once per fit. RUN is
x_ann/neighbors_ann_fit_run.mojo (typed requests, no Python). The
`ann_door_*` calls here hold the interpreter lock rule of the old exports:
  - the five builds run with the GIL RELEASED around the whole table call,
    on a GPU image and on the CPU column alike;
  - the registration of a resident IVF-PQ build and the release of a
    resident handle run with the GIL HELD: the resident registry
    (x_ann/resident.mojo) is guarded by it and has no lock word.
The resident IVF-PQ fit is therefore two table calls, the two phases its
export had: the build is never under the GIL and the registry is never
reached without it.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.native_ops_key_ann_unpack import ann_ops_key
from core.native_descriptor_abi import NativeDescriptorCall
from core.neighbors_entry_table import NeighborsEntryKey
from x_ann.neighbors_ann_fit_run import (
    ANN_NO_TICKET, AnnCagraFitRequest, AnnPqFitRequest, AnnRabitqFitRequest, AnnSqFitRequest,
    ann_cagra_fit_run, ann_pq_fit_resident_run, ann_pq_fit_run, ann_pq_register_run, ann_rabitq_fit_run,
    ann_release_run, ann_sq_fit_run,
)

#: "No resident handle" (the registry mints handles from 1).
comptime ANN_NO_HANDLE = 0


def ann_ops_key_probe(binding: PythonObject) raises -> NeighborsEntryKey:
    """`ann_ops_key`, or a key with vendor -1 when `binding` records none
    (not a compiled module; an image that predates the table). No code of
    `binding` runs."""
    try:
        return ann_ops_key(binding)
    except:
        return NeighborsEntryKey(-1, -1)


def ann_door_pq_fit(entry: NativeDescriptorCall, request: AnnPqFitRequest) raises:
    with GILReleased(Python()):
        ann_pq_fit_run(entry, request)


def ann_door_pq_fit_resident(entry: NativeDescriptorCall, request: AnnPqFitRequest) raises -> Int:
    """The resident IVF-PQ fit: the build with the GIL released, then the
    registration with it held. The handle, or `ANN_NO_HANDLE` when the image
    keeps no resident fit (nothing was built: take `ann_door_pq_fit`)."""
    var ticket = ANN_NO_TICKET
    with GILReleased(Python()):
        ticket = ann_pq_fit_resident_run(entry, request)
    if ticket == ANN_NO_TICKET:
        return ANN_NO_HANDLE
    return ann_pq_register_run(entry, ticket)


def ann_door_sq_fit(entry: NativeDescriptorCall, request: AnnSqFitRequest) raises:
    with GILReleased(Python()):
        ann_sq_fit_run(entry, request)


def ann_door_rabitq_fit(entry: NativeDescriptorCall, request: AnnRabitqFitRequest) raises:
    with GILReleased(Python()):
        ann_rabitq_fit_run(entry, request)


def ann_door_cagra_fit(entry: NativeDescriptorCall, request: AnnCagraFitRequest) raises:
    with GILReleased(Python()):
        ann_cagra_fit_run(entry, request)


def ann_door_release(entry: NativeDescriptorCall, handle: Int) raises:
    # GIL held: a pop of the resident registry.
    ann_release_run(entry, handle)
