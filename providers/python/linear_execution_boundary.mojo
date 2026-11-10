# SPDX-License-Identifier: Apache-2.0
"""Admit a compiled linear provider and pack its native build metadata."""
from std.python import PythonObject
from core.linear_fit_types import LINEAR_FIT_ABI, LINEAR_FIT_OPERATION_COUNT
from core.native_descriptor_call import native_descriptor_call
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once


def linear_execution_word(entries: PythonObject, operation: Int) raises -> Int:
    var provider = neural_descriptor_connect_factory_once(
        entries["linear_fit_native_api_v1"], String(LINEAR_FIT_ABI), LINEAR_FIT_OPERATION_COUNT)
    var entry = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry
    try:
        return native_descriptor_call(entry, operation, List[Int](), List[Int](), List[Float64]())
    finally:
        _ = provider
        _ = entries
