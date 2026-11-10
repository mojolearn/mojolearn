# SPDX-License-Identifier: Apache-2.0
"""Admitted provider and image pin for typed class-weight calls."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.native_loaded_module import native_loaded_module
from core.label_weights_native_api import LABEL_WEIGHTS_ABI, LABEL_WEIGHTS_COUNT, LabelWeightsDriver, LabelWeightsRequest
from providers.python.neural_descriptor_pyapi import (
    NeuralDescriptorProvider, neural_descriptor_connect_factory_once, neural_descriptor_connect_once,
    _neural_descriptor_dictionary,
)


def label_weights_run[ModuleProvider: Bool = False](entries: PythonObject, operation: Int,
                      request: LabelWeightsRequest) raises -> Int:
    var provider: PythonObject
    @parameter
    if ModuleProvider:
        provider = neural_descriptor_connect_once(entries, "label_weights_native_api_v1",
            String(LABEL_WEIGHTS_ABI), LABEL_WEIGHTS_COUNT)
    else:
        # Admit the exact dictionary and its string-key namespace before
        # indexing: neither a mapping hook nor a colliding custom key may run.
        # The provider's mutation-aware namespace admission reuses its stamp.
        _neural_descriptor_dictionary(entries, native_loaded_module("builtins"))
        provider = neural_descriptor_connect_factory_once(entries["label_weights_native_api_v1"],
            String(LABEL_WEIGHTS_ABI), LABEL_WEIGHTS_COUNT)
    var driver = LabelWeightsDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
    try:
        with GILReleased(Python()):
            return driver.run(operation, request)
    finally:
        _ = provider
