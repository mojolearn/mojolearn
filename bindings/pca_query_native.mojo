# SPDX-License-Identifier: Apache-2.0
# Native PCA transform/inverse_transform envelope: admitted private state,
# captured inputs, existing native execution, and the finished owned ndarray.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains, py_get, py_type
from bindings.device_rows_door import door_rows_refuse
from bindings.admitted_input import admitted_finite_refuse
from bindings.py_typed_call import py_call_typed
from bindings.classical_validate import classical_check_width
from bindings.sequential_arrays import sequential_array, sequential_address
from bindings.pca_exact import pca_whiten_entries, pca_refusal
from providers.python.pca_model_pyapi import pca_query_admitted
from providers.python.pca_input_pyapi import pca_state_snapshot
from bindings.decomp_inputs import decomp_admit


def pca_query_operation(model: PythonObject, source: PythonObject, inverse: Bool,
                        context: PythonObject) raises -> PythonObject:
    # Settings, fitted fields and binding entries: full admission on first
    # use and after any change, else the per-model record
    # (providers/python/pca_model_pyapi.mojo, pca_query_admitted).
    var admitted = pca_query_admitted(model, context, String("inverse_transform") if inverse else String("transform"))
    var fields = admitted[0]
    var entries = admitted[1]
    if not py_contains(fields, "components_"):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var fit_rows = Int(py=fields["n_samples_"])
    var features = Int(py=fields["n_features_in_"])
    var count = Int(py=fields["n_components_"])
    if features < 1 or count < 0:
        raise Error("PCA fitted dimensions are invalid")
    var whiten = Bool(py=fields["whiten"])
    # Lane a6: a cross-validation fold of a plain transform is gathered on
    # the device out of model selection's resident base by the GPU entry
    # `pca_transform_fold` (no host gather, no upload of X; connected from
    # the identical base binding that holds the fold store); without it (the
    # host column) the door gets the host rows.
    var fold_entry = py_get(context, "pca_transform_fold")
    # Lane a8d: only an identical model takes it (the entry is the identical
    # base binding's, on the identical context); a `fast` model's fold is
    # gathered on the host and its own `fast` binding uploads it on the
    # `fast` context, as QN's door does (bindings/qn_native.mojo).
    var mode = py_get(fields, "numeric_mode")
    var identical = (mode is not None and py_type(mode) is py_type(PythonObject(""))
                     and String(py_call(py_call(mode.strip).lower)) == "identical")
    var fold_ok = (not inverse and not whiten and fold_entry is not None and identical
                   and py_type(fold_entry) is py_type(entries["pca_transform"]))
    # The plain transform's entry tests a borrowed X on the device after its
    # upload and returns the finite word (lane a6, defer_finite): no host
    # scan of X here.
    var x = decomp_admit(source, "PCA", context, fold_ok, not inverse and not whiten)
    var rows = x.rows
    if x.cols != (count if inverse else features):
        if inverse:
            raise Error("mojolearn PCA component count differs from fit")
        classical_check_width(x.cols, features, "PCA")
    # The whitening path refuses non-finite (and negative) fitted state on
    # every call, as 0.8.55 did: one native scan of the four fitted buffers,
    # so a value written in place into components_ or mean_ after an earlier
    # whitened transform is still refused. No Python walk is involved.
    var scan = whiten
    var components = pca_state_snapshot(fields, "components_", [count, features], context, scan)
    var mean = pca_state_snapshot(fields, "mean_", [features], context, scan)
    # NumPy's native ndarray constructor owns this fresh bytearray-backed
    # C-order float32 output. No later Python wrapper converts its result.
    var output = sequential_array(Python.tuple(rows, features if inverse else count), context["float32"], context)
    var output_address = sequential_address(output)
    if whiten:
        pca_whiten_entries(entries)
        if fit_rows < 2:
            raise Error("PCA whitening requires fit rows >=2")
        var singular = pca_state_snapshot(fields, "singular_values_", [count], context, scan, scan)
        var variance = pca_state_snapshot(fields, "explained_variance_", [count], context, scan, scan)
        # The fit row count governs whitening; the current input's row count
        # only governs output shape. All owners and addresses are now frozen.
        try:
            if inverse:
                _ = py_call_typed(entries["pca_whiten_inverse_transform"], x.address, components.source.addr,
                    singular.source.addr, mean.source.addr, output_address, Python.list(rows, features, count, fit_rows))
            else:
                _ = py_call_typed(entries["pca_whiten_transform"], x.address, mean.source.addr,
                    components.source.addr, singular.source.addr, output_address, Python.list(rows, features, count, fit_rows))
        except error:
            raise pca_refusal(error^)
        _ = singular^
        _ = variance^
    elif inverse:
        _ = py_call_typed(entries["inverse_transform"], x.address, components.source.addr,
            mean.source.addr, output_address, Python.list(rows, features, count, 1))
    elif x.store.active():
        var words = py_call_typed(fold_entry, Python.list(x.store.base_id, x.store.base_rows, x.store.index_id),
            mean.source.addr, components.source.addr, output_address, Python.list(rows, features, count))
        door_rows_refuse(Int(py=words[0]), Int(py=words[1]), "PCA", x.store.base_rows)
    else:
        var status = py_call_typed(entries["pca_transform"], x.address, mean.source.addr,
            components.source.addr, output_address, Python.list(rows, features, count))
        if not x.finite:
            admitted_finite_refuse(Int(py=status), "PCA")
    _ = x^
    _ = components^
    _ = mean^
    return output
