# SPDX-License-Identifier: Apache-2.0
# Native PCA transform/inverse_transform envelope: admitted private state,
# captured inputs, existing native execution, and the finished owned ndarray.
from std.python import Python, PythonObject
from bindings.py_balanced import py_contains, py_get
from bindings.py_typed_call import py_call_typed
from bindings.classical_validate import classical_check_width
from bindings.sequential_arrays import sequential_array, sequential_address
from bindings.pca_exact import pca_whiten_entries, pca_refusal
from providers.python.pca_model_pyapi import pca_query_fields, pca_query_entries
from providers.python.pca_input_pyapi import pca_input_snapshot, pca_state_snapshot


def pca_query_operation(model: PythonObject, source: PythonObject, inverse: Bool,
                        context: PythonObject) raises -> PythonObject:
    var fields = pca_query_fields(model, context)
    if py_get(context, "ndarray") is None:
        raise Error("ImportError: mojolearn: PCA requires optional NumPy support. Install it with: "
                    + 'python -m pip install "mojolearn[numpy]" ("mojolearn[verify]" also includes NumPy).')
    var entries = pca_query_entries(fields, context)
    if not py_contains(fields, "components_"):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var fit_rows = Int(py=fields["n_samples_"])
    var features = Int(py=fields["n_features_in_"])
    var count = Int(py=fields["n_components_"])
    if features < 1 or count < 0:
        raise Error("PCA fitted dimensions are invalid")
    var x = pca_input_snapshot(source, context)
    var rows = x.shape[0]
    if x.shape[1] != (count if inverse else features):
        if inverse:
            raise Error("mojolearn PCA component count differs from fit")
        classical_check_width(x.shape[1], features, "PCA")
    var whiten = Bool(py=fields["whiten"])
    var components = pca_state_snapshot(fields, "components_", [count, features], context, whiten)
    var mean = pca_state_snapshot(fields, "mean_", [features], context, whiten)
    # NumPy's native ndarray constructor owns this fresh bytearray-backed
    # C-order float32 output. No later Python wrapper converts its result.
    var output = sequential_array(Python.tuple(rows, features if inverse else count), context["float32"], context)
    var output_address = sequential_address(output)
    if whiten:
        pca_whiten_entries(entries)
        if fit_rows < 2:
            raise Error("PCA whitening requires fit rows >=2")
        var singular = pca_state_snapshot(fields, "singular_values_", [count], context, True, True)
        var variance = pca_state_snapshot(fields, "explained_variance_", [count], context, True, True)
        # The fit row count governs whitening; the current input's row count
        # only governs output shape. All owners and addresses are now frozen.
        try:
            if inverse:
                _ = py_call_typed(entries["pca_whiten_inverse_transform"], x.source.addr, components.source.addr,
                    singular.source.addr, mean.source.addr, output_address, Python.list(rows, features, count, fit_rows))
            else:
                _ = py_call_typed(entries["pca_whiten_transform"], x.source.addr, mean.source.addr,
                    components.source.addr, singular.source.addr, output_address, Python.list(rows, features, count, fit_rows))
        except error:
            raise pca_refusal(error^)
        _ = singular^
        _ = variance^
    elif inverse:
        _ = py_call_typed(entries["inverse_transform"], x.source.addr, components.source.addr,
            mean.source.addr, output_address, Python.list(rows, features, count, 1))
    else:
        _ = py_call_typed(entries["pca_transform"], x.source.addr, mean.source.addr,
            components.source.addr, output_address, Python.list(rows, features, count))
    _ = x^
    _ = components^
    _ = mean^
    return output
