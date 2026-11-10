# SPDX-License-Identifier: Apache-2.0
"""Public Logistic fitted-buffer PACK and owner pins."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.logistic_model_state import LogisticFitLayout, logistic_publish_coefficients
from bindings.admitted_input import admitted_storage_native
from bindings.classical_labels import labels_address
from bindings.py_balanced import py_get, py_contains
from bindings.classifier_public_lifecycle import classifier_label_list


struct LogisticQueryOwners(Movable):
    """One admitted public-field snapshot, confined to UNPACK/PACK.

    These references pin actual public values against rebinding during RUN;
    no private Python dictionary or algorithm state is retained. Numerical
    requests receive only admitted addresses, dimensions and scalar values.
    Class label decoding remains a separate shared boundary migration.
    """
    var weights: PythonObject
    var classes: PythonObject
    var feature_owner: PythonObject
    var intercept_owner: PythonObject
    var mode_owner: PythonObject
    var has_weights: Bool
    var has_features: Bool

    def __init__(out self, fields: PythonObject, context: PythonObject) raises:
        self.has_weights = py_contains(fields, "_w")
        self.has_features = py_contains(fields, "n_features_in_")
        self.weights = py_get(fields, "_w")
        var classes = py_get(fields, "classes_")
        self.classes = classifier_label_list(classes, context) if py_contains(fields, "classes_") else PythonObject(None)
        self.feature_owner = py_get(fields, "n_features_in_")
        self.intercept_owner = fields["fit_intercept"]
        self.mode_owner = fields["numeric_mode"]

    def features(self) raises -> PythonObject:
        if not self.has_features:
            raise Error("KeyError: n_features_in_")
        return self.feature_owner


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
