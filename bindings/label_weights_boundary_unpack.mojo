# SPDX-License-Identifier: Apache-2.0
"""Admitted provider and image pin for typed class-weight calls."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_class_admission_door import dict_keys_admitted, dict_keys_admit
from bindings.py_balanced import py_type
from core.label_weights_native_api import LABEL_WEIGHTS_ABI, LABEL_WEIGHTS_COUNT, LabelWeightsDriver, LabelWeightsRequest
from providers.python.neural_descriptor_pyapi_door import (
    NeuralDescriptorProvider, neural_descriptor_connect_factory_once, neural_descriptor_connect_once,
    _neural_descriptor_dictionary,
)


def label_weights_run[ModuleProvider: Bool = False](entries: PythonObject, operation: Int,
                      request: LabelWeightsRequest) raises -> Int:
    var provider: PythonObject
    comptime if ModuleProvider:
        provider = neural_descriptor_connect_once(entries, "label_weights_native_api_v1",
            String(LABEL_WEIGHTS_ABI), LABEL_WEIGHTS_COUNT)
    else:
        provider = label_weights_provider(entries)
    var driver = label_weights_driver(provider)
    try:
        with GILReleased(Python()):
            return driver.run(operation, request)
    finally:
        _ = provider


def label_weights_provider(entries: PythonObject) raises -> PythonObject:
    """UNPACK: the admitted label weights provider of an x_trees image's
    entry dictionary. A typed caller holds it (the pin of the driver's entry)
    while it calls `label_weights_driver(provider).run` (lane mf-ada: the
    AdaBoost rounds multiply their class_weight rows through it)."""
    # Admit the exact dictionary and its string-key namespace before
    # indexing: neither a mapping hook nor a colliding custom key may run.
    # The provider's mutation-aware namespace admission reuses its stamp.
    var b = native_loaded_module("builtins")
    if py_type(entries) is not b.dict:
        raise Error("TypeError: neural provider metadata requires an exact dictionary")
    if not dict_keys_admitted(entries):
        _neural_descriptor_dictionary(entries, b)
        dict_keys_admit(entries)
    return neural_descriptor_connect_factory_once(entries["label_weights_native_api_v1"],
        String(LABEL_WEIGHTS_ABI), LABEL_WEIGHTS_COUNT)


def label_weights_driver(provider: PythonObject) raises -> LabelWeightsDriver:
    """The typed driver of an admitted provider (no Python in its calls;
    the caller keeps `provider` alive while it runs)."""
    return LabelWeightsDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry)
