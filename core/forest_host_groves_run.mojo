# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN of `bindings/forest_host_groves_binding.mojo` (wave 8 run split): the
host resident `parallel_groves` registries and their typed prepare, predict
and release. Imports no Python; the binding keeps the UNPACK doors."""
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


def forest_release_host_native[RF_INPUT: Bool](handle: Int) raises:
    """`forest_release_gpu` on the host (XF_RESIDENT_RELEASE)."""
    comptime if RF_INPUT:
        RF_GROVE_REGISTRY.get_or_create_ptr()[].release(handle)
    else:
        ET_GROVE_REGISTRY.get_or_create_ptr()[].release(handle)
