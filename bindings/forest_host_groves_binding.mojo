# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The resident `parallel_groves` entries of a CPU-only install, under the GPU
bindings' names and address contract (`bindings/forest_inference_binding.mojo`:
`forest_prepare_gpu`, `forest_predict_resident_reuse_gpu`,
`forest_release_gpu`), over `core/forest_host_groves.mojo`.

HOST ONLY. The `rf` host binding specializes these with `RF_INPUT=True` and
the `trees` host binding with `RF_INPUT=False`, each over its own registry,
as `core/forest_inference_model.mojo` keeps one registry per specialization.
The caller holds the GIL through prepare, predict and release, as the GPU
binding's caller does.
"""
from std.python import PythonObject

from core.forest_host_groves import HostGroveForest, HostGroveRegistry
from core.forest_host_groves_binding_impl import (
    _f32_ptr,
    _i32_ptr,
    host_grove_forest_from,
)
from core.shared_slot import SharedSlot


comptime RF_GROVE_REGISTRY = SharedSlot[HostGroveRegistry, "MojolearnRFResidentForestHostV1"]
comptime ET_GROVE_REGISTRY = SharedSlot[HostGroveRegistry, "MojolearnETResidentForestHostV1"]


def forest_prepare_host_native[RF_INPUT: Bool](model: List[Int], features: Int, trees: Int, outputs: Int,
                                              ordered_word: Int) raises -> Int:
    """`forest_prepare_gpu_native` (`bindings/forest_inference_binding.mojo`)
    on the host (XF_RESIDENT_PREPARE; lane calltable-forest): the same
    checks in the same words, then the host snapshot. The host images take
    no ordered word (they report no XF_CAP_ORDERED_RESIDENT)."""
    if ordered_word != -1:
        raise Error("forest_prepare_gpu requires features, trees, outputs")
    if features < 1 or trees < 1 or trees >= 2147483647 or outputs < 1:
        raise Error("invalid resident forest dimensions")
    var snapshot = host_grove_forest_from(model[0], model[1], model[2], model[3], model[4], features, trees, outputs)
    comptime if RF_INPUT:
        return RF_GROVE_REGISTRY.get_or_create_ptr()[].prepare(snapshot^)
    else:
        return ET_GROVE_REGISTRY.get_or_create_ptr()[].prepare(snapshot^)


def forest_prepare_host_binding[RF_INPUT: Bool](
    offsets_addr: PythonObject, colid_addr: PythonObject,
    quesval_addr: PythonObject, left_child_addr: PythonObject,
    leaves_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`forest_prepare_gpu` of a host binding: UNPACK of
    `forest_prepare_host_native`."""
    if len(params) != 3:
        raise Error("forest_prepare_gpu requires features, trees, outputs")
    var model: List[Int] = [Int(py=offsets_addr), Int(py=colid_addr), Int(py=quesval_addr),
                            Int(py=left_child_addr), Int(py=leaves_addr)]
    return PythonObject(forest_prepare_host_native[RF_INPUT](
        model, Int(py=params[0]), Int(py=params[1]), Int(py=params[2]), -1))


def forest_predict_resident_host_native[RF_INPUT: Bool](
    handle: Int, x_addr: Int, out_addr: Int, rows: Int, features: Int, outputs: Int,
) raises -> Int:
    """`forest_predict_resident_into_gpu_native[RF_INPUT, True]` on the host
    (XF_RESIDENT_PREDICT): borrowed ROW-major `x`; returns rows."""
    if rows < 0 or features < 1 or outputs < 1:
        raise Error("invalid resident prediction dimensions")
    if rows > 2147483647 // features or rows > 2147483647 // outputs:
        raise Error("resident prediction dimensions exceed Int32")
    var xp = _f32_ptr(x_addr)
    var op = _f32_ptr(out_addr)
    var state = RF_GROVE_REGISTRY.get_or_create_ptr()
    comptime if not RF_INPUT:
        state = ET_GROVE_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident forest handle")
    state[].entries[handle].predict_into[RF_INPUT](xp, op, rows, features, outputs)
    return rows


def forest_predict_resident_host_binding[RF_INPUT: Bool](
    handle: PythonObject, x_addr: PythonObject, out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`forest_predict_resident_reuse_gpu` of a host binding (UNPACK)."""
    if len(params) != 3:
        raise Error("resident prediction requires rows, features, outputs")
    return PythonObject(forest_predict_resident_host_native[RF_INPUT](
        Int(py=handle), Int(py=x_addr), Int(py=out_addr), Int(py=params[0]), Int(py=params[1]), Int(py=params[2])))


def forest_release_host_native[RF_INPUT: Bool](handle: Int) raises:
    """`forest_release_gpu` on the host (XF_RESIDENT_RELEASE)."""
    comptime if RF_INPUT:
        RF_GROVE_REGISTRY.get_or_create_ptr()[].release(handle)
    else:
        ET_GROVE_REGISTRY.get_or_create_ptr()[].release(handle)


def forest_release_host_binding[RF_INPUT: Bool](handle: PythonObject) raises -> PythonObject:
    """`forest_release_gpu_binding` (`bindings/forest_inference_binding.mojo`)."""
    forest_release_host_native[RF_INPUT](Int(py=handle))
    return PythonObject(None)
