# SPDX-License-Identifier: Apache-2.0
# Native PCA transform/inverse_transform envelope: admitted private state,
# captured inputs, existing native execution, and the finished owned ndarray.
from std.python import Python, PythonObject
from bindings.device_rows_door import door_rows_refuse
from bindings.admitted_input import admitted_finite_refuse
from bindings.classical_validate import classical_check_width
from bindings.sequential_arrays import sequential_array, sequential_address
from bindings.pca_exact import pca_refusal
from bindings.pca_descriptor_boundary import pca_mode, pca_fold_connection, pca_whiten_capability, pca_run_query
from core.pca_native_api import PCAQueryRequest, PCAFoldStore
from providers.python.pca_model_pyapi import pca_query_admitted
from providers.python.pca_input_pyapi import pca_state_snapshot
from bindings.decomp_inputs import decomp_admit


def pca_query_operation(model: PythonObject, source: PythonObject, inverse: Bool,
                        context: PythonObject) raises -> PythonObject:
    # Settings, fitted fields and binding entries: full admission on first
    # use and after any change, else the per-model record
    # (providers/python/pca_model_pyapi.mojo, pca_query_admitted).
    var admitted = pca_query_admitted(model, context, String("inverse_transform") if inverse else String("transform"))
    var fields = admitted.fields
    var fit_rows = admitted.state.fit_rows
    var features = admitted.state.features
    var count = admitted.state.count
    var whiten = admitted.state.whiten
    var connection = admitted.state.connection
    # Lane a6: a cross-validation fold of a plain transform is gathered on
    # the device out of model selection's resident base by the GPU entry
    # `pca_transform_fold` (no host gather, no upload of X; connected from
    # the identical base binding that holds the fold store); without it (the
    # host column) the door gets the host rows.
    # Lane a8d: only an identical model takes it (the entry is the identical
    # base binding's, on the identical context); a `fast` model's fold is
    # gathered on the host and its own `fast` binding uploads it on the
    # `fast` context, as QN's door does (bindings/qn_native.mojo).
    var fold_connection = -1
    if not inverse and not whiten and pca_mode(connection) == 1:
        fold_connection = pca_fold_connection(context)
    var fold_ok = fold_connection >= 0
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
    var output_address = Int(py=sequential_address(output))
    var fold = PCAFoldStore()
    if x.store.active():
        fold = PCAFoldStore(x.store.base_id, x.store.base_rows, x.store.index_id)
    var request = PCAQueryRequest(x.address, mean.source.addr, components.source.addr, 0,
        output_address, rows, features, count, fit_rows, inverse, whiten, fold)
    if whiten:
        pca_whiten_capability(connection)
        if fit_rows < 2:
            raise Error("PCA whitening requires fit rows >=2")
        var singular = pca_state_snapshot(fields, "singular_values_", [count], context, scan, scan)
        var variance = pca_state_snapshot(fields, "explained_variance_", [count], context, scan, scan)
        # The fit row count governs whitening; the current input's row count
        # only governs output shape. All owners and addresses are now frozen.
        request.singular = singular.source.addr
        try:
            _ = pca_run_query(connection, request)
        except error:
            raise pca_refusal(error^)
        _ = singular^
        _ = variance^
    elif inverse:
        _ = pca_run_query(connection, request)
    elif x.store.active():
        var words = pca_run_query(fold_connection, request)
        door_rows_refuse(words.finite, words.in_range, "PCA", x.store.base_rows)
    else:
        var status = pca_run_query(connection, request)
        if not x.finite:
            admitted_finite_refuse(status.finite, "PCA")
    _ = x^
    _ = components^
    _ = mean^
    return output
