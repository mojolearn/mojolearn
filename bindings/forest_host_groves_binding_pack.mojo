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

from core.forest_host_groves_run import (
    forest_prepare_host_native,
    forest_predict_resident_host_native,
    forest_release_host_native,
)


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


def forest_predict_resident_host_binding[RF_INPUT: Bool](
    handle: PythonObject, x_addr: PythonObject, out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`forest_predict_resident_reuse_gpu` of a host binding (UNPACK)."""
    if len(params) != 3:
        raise Error("resident prediction requires rows, features, outputs")
    return PythonObject(forest_predict_resident_host_native[RF_INPUT](
        Int(py=handle), Int(py=x_addr), Int(py=out_addr), Int(py=params[0]), Int(py=params[1]), Int(py=params[2])))


def forest_release_host_binding[RF_INPUT: Bool](handle: PythonObject) raises -> PythonObject:
    """`forest_release_gpu_binding` (`bindings/forest_inference_binding.mojo`)."""
    forest_release_host_native[RF_INPUT](Int(py=handle))
    return PythonObject(None)
