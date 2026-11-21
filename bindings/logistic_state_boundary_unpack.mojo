# SPDX-License-Identifier: Apache-2.0
"""Public Logistic fitted-buffer PACK and owner pins."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.logistic_model_state import LogisticFitLayout, logistic_publish_coefficients
from bindings.admitted_input_unpack import admitted_storage_native
from bindings.classical_labels_door import labels_address
from bindings.py_balanced import py_get, py_contains, py_setitem
from bindings.classifier_public_lifecycle_door import classifier_label_list
from bindings.logistic_keys_unpack import logistic_keys
from bindings.py_balanced import py_item


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
    var has_weights: Bool
    var has_features: Bool

    def __init__(out self, fields: PythonObject, context: PythonObject) raises:
        var nk = logistic_keys()
        self.has_weights = py_contains(fields, nk[].k__w)
        self.has_features = py_contains(fields, nk[].k_n_features_in_)
        self.weights = py_get(fields, nk[].k__w)
        var classes = py_get(fields, nk[].k_classes_)
        self.classes = classifier_label_list(classes, context) if py_contains(fields, nk[].k_classes_) else PythonObject(None)
        self.feature_owner = py_get(fields, nk[].k_n_features_in_)
        self.intercept_owner = py_item(fields, nk[].k_fit_intercept)

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


comptime _LogisticI64 = MutPointer[Int64, MutUntrackedOrigin]


def logistic_fit_pack(fields: PythonObject, weights: PythonObject, layout: LogisticFitLayout, iterations: Int,
                      objective: Float32, retcode: Float32, context: PythonObject) raises:
    """PACK of fit: publish the solver's weight block, its public coefficient
    and intercept copies, the iteration count, the objective, the return
    code and the feature count on the estimator's admitted field storage.
    Nothing here is read by RUN; the typed results arrive as values."""
    var nk = logistic_keys()
    # Keep the new weight buffer private while native PACK releases the GIL.
    # Preserve the old failed-allocation state (_w was published before the
    # public coefficient allocations) without exposing an in-progress copy.
    var public_parameters: Tuple[PythonObject, PythonObject]
    try:
        public_parameters = logistic_parameters_pack(weights, layout, context)
    except error:
        py_setitem(fields, nk[].k__w, weights)
        raise error^
    py_setitem(fields, nk[].k__w, weights)
    var coef = public_parameters[0]
    var inter = public_parameters[1]
    py_setitem(fields, nk[].k_coef_, coef)
    py_setitem(fields, nk[].k_intercept_, inter)
    var count = admitted_storage_native([1], "<i8", context, as_array=True)
    _LogisticI64(unsafe_from_address=labels_address(count))[0] = Int64(iterations)
    _ = count  # keep the Python object alive through the native write above
    _ = weights  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_n_iter_, count)
    py_setitem(fields, nk[].k_objective_, PythonObject(Float64(objective)))
    py_setitem(fields, nk[].k_retcode_, PythonObject(Int(retcode)))
    py_setitem(fields, nk[].k_n_features_in_, layout.features)
