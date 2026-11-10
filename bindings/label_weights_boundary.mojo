# SPDX-License-Identifier: Apache-2.0
"""Admitted provider and image pin for typed class-weight calls."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.label_weights_native_api import LABEL_WEIGHTS_ABI, LABEL_WEIGHTS_COUNT, LabelWeightsDriver, LabelWeightsRequest
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once


def label_weights_run(entries: PythonObject, operation: Int,
                      request: LabelWeightsRequest) raises -> Int:
    var provider = neural_descriptor_connect_factory_once(entries["label_weights_native_api_v1"],
        String(LABEL_WEIGHTS_ABI), LABEL_WEIGHTS_COUNT)
    var driver = LabelWeightsDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
    try:
        with GILReleased(Python()):
            return driver.run(operation, request)
    finally:
        _ = provider
