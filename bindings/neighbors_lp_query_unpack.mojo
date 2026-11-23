# SPDX-License-Identifier: Apache-2.0
"""UNPACK of a LabelPropagation / LabelSpreading query (lane mf-shap3b, 2026-11-23): the fitted state and the
query words of `_LabelPropagationBase.predict_proba` / `predict` read once into the typed `LpQueryRequest`
(neighbors/label_propagation_run.mojo), with the rows, the output and the row count left for the caller. ONE BODY
for both callers: the public query (bindings/neighbors_semi_native_door.mojo `_lp_query`) and Kernel / Permutation
SHAP over a label propagation model (bindings/trees_agnostic_native_door.mojo, the RUN's `AgnLabelProp`), so the
explained output is the public predict_proba's, word for word.

The Python reads are the door's, in its order: `kernel`, `X_`, `n_neighbors` (knn) or `gamma` (rbf), the
import-time switches the family context holds (`family_xn_semi_unfused_knn`, `family_svc_xn_old_items`,
`family_svc_xn_kernels`), `label_distributions_`, and the image's dense-proba capability."""
from std.python import PythonObject
from bindings.py_balanced import py_fields
from bindings.family_common_pack import family_call, family_field
from bindings.native_loaded_module_door import native_loaded_module
from bindings.classical_arrays_door import classical_address
from providers.python.x_neighbors_descriptor_pyapi_door import xn_module, xn_is_module
from providers.python.x_neighbors_admission_pyapi_door import xn_capabilities
from neighbors.neighbors_semi_run import xn_semi_knn_k
from neighbors.label_propagation_run import LpQueryRequest


struct LpQueryUnpack(Movable):
    """`request`: the query words with `q`, `dst`, `labels` and `rows` 0 (the caller sets them per call).
    `fit_x` / `distributions`: the fitted `X_` and `label_distributions_` whose addresses the request holds
    (references only, no copy); the caller keeps this value alive through every RUN that reads the request."""
    var request: LpQueryRequest
    var fit_x: PythonObject
    var distributions: PythonObject

    def __init__(out self, var request: LpQueryRequest, fit_x: PythonObject, distributions: PythonObject):
        self.request = request^
        self.fit_x = fit_x
        self.distributions = distributions


def _lp_b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _lp_truth(value: PythonObject) raises -> Bool:
    """`bool(value)`."""
    return Bool(py=family_call(_lp_b().bool, value))


def _lp_eq(a: PythonObject, b: PythonObject) raises -> Bool:
    """`bool(a == b)`, Python's own comparison."""
    return _lp_truth(family_call(native_loaded_module("operator").eq, a, b))


def lp_dense_available(binding: PythonObject) raises -> Bool:
    """The binding carries `lp_dense_proba` (export on a loaded module,
    capability bit 6 on the native descriptor route)."""
    if xn_is_module(binding):
        return family_call(_lp_b().getattr, xn_module(binding), PythonObject("x_neighbors_lp_dense_proba"),
                           PythonObject(None)) is not None
    return (xn_capabilities(binding) & 64) != 0


def lp_query_unpack(model: PythonObject, binding: PythonObject, want_labels: Bool,
                    context: PythonObject) raises -> LpQueryUnpack:
    """The fitted query words of `model` on the x_neighbors `binding` (`context`: the shared family context the
    public method is called with). Refusals in the door's words (an unfitted model's missing `X_`)."""
    var fields = py_fields(model)
    var knn = _lp_eq(family_field(fields, "kernel"), PythonObject("knn"))
    var fused = not knn and lp_dense_available(binding)
    var xfit = family_field(fields, "X_")
    var n = Int(py=xfit.shape[0])
    var d = Int(py=xfit.shape[1])
    var k = 0
    var kind = 0
    var gamma = Float64(0.0)
    var unfused = False
    var old_items = False
    if knn:
        k = xn_semi_knn_k(Int(py=family_call(_lp_b().int, family_field(fields, "n_neighbors"))), n)
        unfused = _lp_truth(context["family_xn_semi_unfused_knn"])
        if not unfused:
            old_items = _lp_truth(context["family_svc_xn_old_items"])
    else:
        var kind_word = context["family_svc_xn_kernels"]["rbf"]
        gamma = Float64(py=family_call(_lp_b().float, family_field(fields, "gamma")))
        kind = Int(py=family_call(_lp_b().int, kind_word))
    var dist = family_field(fields, "label_distributions_")
    var c = Int(py=dist.shape[1])
    var request = LpQueryRequest(0, Int(py=classical_address(xfit)), Int(py=classical_address(dist)), 0, 0, 0, n,
                                 d, c, knn, k, kind, gamma, fused, want_labels, unfused, old_items)
    return LpQueryUnpack(request^, xfit, dist)
