# SPDX-License-Identifier: Apache-2.0
"""Optimizer descriptor UNPACK: exact factories and lifetime pins stay here."""
from std.python import PythonObject
from core.shared_slot import SharedSlot
from providers.python.neural_descriptor_pyapi_door import (
    _neural_descriptor_function_owner,
    NeuralDescriptorProvider, neural_descriptor_connect_once, neural_descriptor_connect_factory_once,
)
from training.optimizer_native_api import OPTIMIZER_NATIVE_ABI, OPT_NATIVE_COUNT, OptimizerNativeDriver
from training.optimizer_descriptor_boundary_run import (
    optimizer_connection_index, optimizer_connection_add, optimizer_record_driver, optimizer_record_mode,
    optimizer_record_device,
)
from bindings.py_provider_pins import ProviderPins, provider_word


# GIL-owned boundary pins: providers pin extension images, capsule tables and
# factories; algorithm calls receive only the typed driver copy from the RUN
# record at the same index (training/optimizer_descriptor_boundary_run.mojo).
comptime OPTIMIZER_PROVIDER_PINS = SharedSlot[ProviderPins, "MojolearnOptimizerProviderPinsV1"]


def optimizer_provider(module: PythonObject) raises -> PythonObject:
    return neural_descriptor_connect_once(module, "optimizer_native_api_v1", String(OPTIMIZER_NATIVE_ABI), OPT_NATIVE_COUNT)


def optimizer_provider_entries(entries: PythonObject) raises -> PythonObject:
    var factory = entries["optimizer_native_api_v1"]
    var provider = neural_descriptor_connect_factory_once(factory, String(OPTIMIZER_NATIVE_ABI), OPT_NATIVE_COUNT)
    # The provider's module is the compiled owner of the factory function
    # (PyCFunction self). The bridge's `entries["module"]` is the public
    # binding module, which since the merged-binding layout re-exports the
    # host unit's functions (`mojolearn._mojolearn_training` ->
    # `mojolearn._host._mojolearn_training_host`), so an identity test against
    # it refused every optimizer endpoint (verify 2026-11-12: 84 public API
    # failures, training-mlp-sessions). Accept the admitted module or the
    # factory's own compiled owner; anything else is a foreign descriptor.
    ref owner = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module
    if owner is not entries["module"] and owner._obj_ptr != _neural_descriptor_function_owner(factory):
        raise Error("TypeError: optimizer descriptor must belong to its connected native extension")
    return provider


def _optimizer_connection(provider: PythonObject) raises -> Int:
    var index = optimizer_connection_index(provider_word(provider))
    if index >= 0:
        return index
    # One complete typed record is published by RUN (its metadata reads may
    # raise before anything is appended); the pin append that follows cannot
    # raise, so the record and pin lists keep the same length.
    index = optimizer_connection_add(provider_word(provider),
        provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry)
    OPTIMIZER_PROVIDER_PINS.get_or_create_ptr()[].providers.append(provider)
    return index


def optimizer_driver(provider: PythonObject) raises -> OptimizerNativeDriver:
    return optimizer_record_driver(_optimizer_connection(provider))


def optimizer_native_mode(provider: PythonObject) raises -> Int:
    return optimizer_record_mode(_optimizer_connection(provider))


def optimizer_device_available(provider: PythonObject) raises -> Bool:
    return optimizer_record_device(_optimizer_connection(provider))
