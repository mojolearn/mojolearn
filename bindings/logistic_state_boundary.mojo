# SPDX-License-Identifier: Apache-2.0
"""Public Logistic fitted-buffer PACK and owner pins."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.logistic_model_state import LogisticFitLayout, logistic_publish_coefficients
from bindings.admitted_input import admitted_storage_native
from bindings.classical_labels import labels_address


def logistic_parameters_pack(weights: PythonObject, layout: LogisticFitLayout,
                              context: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    # Public coef_/intercept_ remain independent writable arrays. Their
    # established alias/rebinding contract is unchanged; the private QN _w
    # buffer remains the numerical operation's weight storage.
    var coefficients = admitted_storage_native([layout.targets, layout.features], "<f4", context, as_array=True)
    var intercepts = admitted_storage_native([layout.targets], "<f4", context, as_array=True)
    var source = labels_address(weights)
    var coefficient_address = labels_address(coefficients)
    var intercept_address = labels_address(intercepts)
    try:
        with GILReleased(Python()):
            logistic_publish_coefficients(layout, source, coefficient_address, intercept_address)
    finally:
        _ = weights
        _ = coefficients
        _ = intercepts
    return (coefficients^, intercepts^)
