# SPDX-License-Identifier: Apache-2.0
"""The boundary of the IVF-Flat operations table (lane dw-neighbors,
2026-11-21), shared by the three doors that reach the IVF image:
`IVFIndex.fit` (bindings/neighbors_ann_native_door.mojo), `search` / `extend`
/ the list-data read (bindings/neighbor_query_native_door.mojo) and the shard
worker (bindings/parallel_neighbor_worker_door.mojo).

UNPACK is `ivf_ops_key`: one exact read of the image's module dictionary for
the `(mode, vendor, image)` it recorded when it published. RUN is
ivf/ivf_query_run.mojo (typed requests, no Python). The `ivf_door_*` calls
here hold the interpreter lock rule of the old exports, decided by the key
the door already holds (one integer compare, nothing read):
  - a GPU image runs with the GIL HELD, as its exports did (its resident
    registry and its process DeviceContext are serialized by it);
  - a CPU image runs with the GIL released around the whole entry (its
    registry is locked and its handles are held by reference count). The
    old host exports released it around the search only.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_fields, py_get, py_type
from core.ivf_entry_table import IvfEntryKey, IvfIndexSpans, IvfIndexWords
from ivf.ivf_query_run import (
    IVF_NO_HANDLE, IvfBuildRequest, IvfExtendRequest, IvfSearchRequest, ivf_build_run, ivf_export_run,
    ivf_extend_run, ivf_prepare_run, ivf_release_run, ivf_search_run,
)


def ivf_ops_key_probe(binding: PythonObject) raises -> IvfEntryKey:
    """The IVF-Flat table key of the image behind `binding` (an exact
    compiled module, or the import-time captured table of one), or a key
    with vendor -1 when it records none (not a module; an image that
    predates the table). One exact dictionary read, never `__getattr__`."""
    var b = native_loaded_module("builtins")
    if py_type(binding) is not py_type(b):
        return IvfEntryKey(-1, -1, -1)
    var key = py_get(py_fields(binding), "ivf_native_key")
    if key is None or py_type(key) is not b.tuple or len(key) != 3:
        return IvfEntryKey(-1, -1, -1)
    return IvfEntryKey(Int(py=key[0]), Int(py=key[1]), Int(py=key[2]))


def ivf_ops_key(binding: PythonObject) raises -> IvfEntryKey:
    """`ivf_ops_key_probe`, refused in words when the image records no key."""
    var key = ivf_ops_key_probe(binding)
    if key.vendor < 0:
        raise Error("ImportError: mojolearn IVFIndex: the IVF binding publishes no native operations table;"
                    + " rebuild it with bash bindings/build_ivf.sh")
    return key


def ivf_door_build(key: IvfEntryKey, request: IvfBuildRequest) raises -> Int:
    if key.on_device():
        return ivf_build_run(key, request)
    var handle = IVF_NO_HANDLE
    with GILReleased(Python()):
        handle = ivf_build_run(key, request)
    return handle


def ivf_door_search(key: IvfEntryKey, request: IvfSearchRequest) raises:
    if key.on_device():
        ivf_search_run(key, request)
        return
    with GILReleased(Python()):
        ivf_search_run(key, request)


def ivf_door_prepare(key: IvfEntryKey, spans: IvfIndexSpans, index: IvfIndexWords, partial: Bool) raises -> Int:
    if key.on_device():
        return ivf_prepare_run(key, spans, index, partial)
    var handle = IVF_NO_HANDLE
    with GILReleased(Python()):
        handle = ivf_prepare_run(key, spans, index, partial)
    return handle


def ivf_door_release(key: IvfEntryKey, handle: Int) raises:
    # A registry pop: no lock release is worth its two calls on either image.
    ivf_release_run(key, handle)


def ivf_door_export(key: IvfEntryKey, handle: Int, list_data: Int, count: Int) raises:
    """The list-data read of a device index (a GPU image only; a CPU image
    refuses it in words: its fitted arrays are the index)."""
    ivf_export_run(key, handle, list_data, count)


def ivf_door_extend(key: IvfEntryKey, request: IvfExtendRequest) raises:
    if key.on_device():
        ivf_extend_run(key, request)
        return
    with GILReleased(Python()):
        ivf_extend_run(key, request)
