# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_estimators` family: KernelDensity (the
CPU training lane, phase 1, 2026-09-13; brief
docs/lanes/BRIEF_cpu_training_2026-09-13.md sections 1.1 kde and 3.2) and,
since the classical host inference lane the same day, the INFERENCE entries
of LinearRegression, Ridge, TruncatedSVD, LogisticRegression and PCA
(docs/lanes/BRIEF_forest_host_inference_2026-09-13.md, "Classical lanes").

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The KDE arithmetic is
`kde/checks/kde_oracle.mojo::oracle_score_samples`, the float32 serial
replay the device arm is gated against bit for bit under IDENTICAL ("every
formula spelled here a SECOND time rather than imported from `kde/impl/`").
The validation is the GPU entry's, in the GPU entry's order
(`kde/estimator.mojo::kde_score_samples_host_ptr`: kernel and metric names,
`kde_fit_validate`, `n_query`, train data, query data), through the same
host-only functions of `kde/impl/neighbors/kernel_density.mojo`, so a bad
call raises the same error and nothing is written on a refusal. The
classical inference arithmetic is `core/classical_host_predict.mojo`, the
statement-for-statement restatement of the pinned gemm/gemv kernel, the
intercept and bias epilogues, the centering kernel and the host sigmoid;
that file's header names every original by file and line.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES for what this covers, so
`python/mojolearn/density.py::KernelDensity`, `linear_model.py` and
`decomposition.py` run unchanged on a CPU-only install through
`_backend._HOST_MODULES` (`"_mojolearn_estimators":
"_mojolearn_estimators_host"`): `kde_score_samples` with the SAME address
contract (train, query, weights, out, the five-value params list, kernel,
metric; mirrored word for word in `density.py`), `ols_predict`,
`tsvd_transform`, `qn_decision_function`, `qn_sigmoid`, `pca_transform`,
since lane/logistic-multiclass (2026-09-14) `qn_softmax` and the 4-field
`qn_decision_function` (the softmax shape of LogisticRegression with more
than two classes), and, since the kde svc host lane (2026-09-14), the whitened pair
`pca_whiten_transform` and `pca_whiten_inverse_transform`, with the SAME
address contracts as `bindings/_mojolearn_estimators.mojo` (each docstring
below repeats its params list), `estimators_numeric_mode` and
`estimators_vendor` (answering "cpu"). Every other function of the GPU
binding (dbscan_fit, pca_fit, pca_fit_full, tsvd_fit, inverse_transform,
ols_fit, ridge_fit, qn_fit, ...) is deliberately absent, so those surfaces
refuse BY NAME through `_HostBinding` and never hash something else.
"""
from std.math import isfinite
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import f32_ptr, f64_ptr, read_f32
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.classical_host_predict import (
    CLASSICAL_HOST_SABOTAGE,
    host_ols_predict,
    host_pca_transform,
    host_pca_whiten_inverse_transform,
    host_pca_whiten_transform,
    host_qn_decision,
    host_qn_decision_multi,
    host_qn_sigmoid,
    host_qn_softmax,
    host_tsvd_transform,
)
from kde.checks.kde_oracle import KDE_ORACLE_HOST_SABOTAGE, oracle_score_samples
from kde.impl.neighbors.kernel_density import (
    kde_fit_validate,
    kde_validate_data_ptr,
    kernel_from_name,
    metric_from_name,
)


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("estimators host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


def estimators_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def estimators_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def estimators_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_estimators_host.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "estimators host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_estimators_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


# There is no `estimators_host_detected_column` read-back, for the reason 8d16ce2f
# removed it from the forest and byte LM host bindings: the detected column
# folds to the GPU of the machine that ran the build, so its name would land
# in the vendor-neutral binary. The comptime assert above is the check.


def estimators_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary sums every logsumexp row and walks every dot
    product descending on purpose (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's
    negative control; one define, both arithmetics)."""
    return PythonObject(KDE_ORACLE_HOST_SABOTAGE or CLASSICAL_HOST_SABOTAGE)


# The GPU binding's names, same contract.


def estimators_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: always 1 here, a host
    binding is IDENTICAL only (the build script refuses any other)."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def estimators_vendor_binding() raises -> PythonObject:
    """"cpu". On a CPU-only install `_backend.vendor()` is "cpu" and the
    read-back cross-check expects that string from every host binding."""
    return PythonObject(String("cpu"))


def kde_score_samples_binding(
    train_addr: PythonObject,
    query_addr: PythonObject,
    weights_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
    kernel: PythonObject,
    metric: PythonObject,
) raises -> PythonObject:
    """KernelDensity.score_samples on the host: log density of each query
    row under the fitted training set, by `oracle_score_samples`. Writes
    `n_query` float32 to `out_addr`. `params`, in this order (mirrored in
    `python/mojolearn/density.py` and in the GPU binding):

        0  n_train
        1  n_query
        2  n_features
        3  bandwidth   (float)
        4  has_weights (0/1; weights_addr is read only when 1)

    `kernel` and `metric` are the sklearn/cuML names; every unimplemented
    one is refused BY NAME by `kernel_from_name` and `metric_from_name`, as
    on the device. Returns n_query."""
    if len(params) != 5:
        raise Error(
            "kde_score_samples: params must contain 5 values, got "
            + String(len(params))
        )
    var tp = f32_ptr(_index(train_addr))
    var qp = f32_ptr(_index(query_addr))
    var op = f32_ptr(_index(out_addr))
    var n_train = _index(params[0])
    var n_query = _index(params[1])
    var n_features = _index(params[2])
    var bandwidth = Float32(Float64(py=params[3]))
    var has_weights = _index(params[4]) != 0
    var kname = String(py=kernel)
    var mname = String(py=metric)
    var weights = List[Float32]()
    if has_weights:
        weights = read_f32(_index(weights_addr), max(0, n_train))
    var train_address = _index(train_addr)
    var query_address = _index(query_addr)
    with GILReleased(Python()):
        # The GPU entry's checks, in its order, so a bad call raises the
        # same error before anything is read or written.
        var k = kernel_from_name(kname)
        var m = metric_from_name(mname)
        kde_fit_validate(n_train, n_features, bandwidth, k, m, weights, has_weights)
        if n_query <= 0:
            raise Error("kde: X must have at least one row (n_query)")
        kde_validate_data_ptr(tp, n_train, n_features, m, "train")
        kde_validate_data_ptr(qp, n_query, n_features, m, "query")
        var train = read_f32(train_address, n_train * n_features)
        var query = read_f32(query_address, n_query * n_features)
        # THE ONE CALL THAT COMPUTES ANYTHING. metric_arg is Minkowski's p,
        # 2.0 here as in the GPU binding, which passes no other value.
        var stages = oracle_score_samples(
            train, query, weights, has_weights, n_train, n_query, n_features,
            bandwidth, k, m, Float32(2.0),
        )
        for i in range(n_query):
            op[i] = stages.scores[i]
    return PythonObject(n_query)


# The classical inference entries (the classical host inference lane,
# 2026-09-13). Each keeps the GPU binding's name, arity and params list.


def _positive(value: Int, what: String) raises:
    if value < 1:
        raise Error("estimators host: " + what + " must be at least 1, got " + String(value))


def ols_predict_binding(
    x_addr: PythonObject,
    coef_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`LinearRegression.predict` and `Ridge.predict` on the host:
    `out[n_rows] = X[n_rows x n_features] . coef + intercept` by
    `host_ols_predict`. params, as in the GPU binding: n_rows, n_features,
    intercept (a float; `Float32(Float64(...))` as there). Returns 0."""
    if len(params) != 3:
        raise Error("ols_predict: params must contain n_rows, n_features, intercept")
    var x_address = _index(x_addr)
    var coef_address = _index(coef_addr)
    var op = f32_ptr(_index(out_addr))
    var nr = _index(params[0])
    var nf = _index(params[1])
    var intercept = Float32(Float64(py=params[2]))
    with GILReleased(Python()):
        _positive(nr, "n_rows")
        _positive(nf, "n_features")
        var x = read_f32(x_address, nr * nf)
        var coef = read_f32(coef_address, nf)
        var out = host_ols_predict(x, coef, nr, nf, intercept)
        for i in range(nr):
            op[i] = out[i]
    return PythonObject(0)


def tsvd_transform_binding(
    x_addr: PythonObject,
    components_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`TruncatedSVD.transform` on the host: `out[n_rows x n_components] =
    X . components^T` by `host_tsvd_transform`. params: n_rows, n_features,
    n_components. Returns 0."""
    if len(params) != 3:
        raise Error("tsvd_transform: params must contain 3 values")
    var x_address = _index(x_addr)
    var c_address = _index(components_addr)
    var op = f32_ptr(_index(out_addr))
    var nr = _index(params[0])
    var nf = _index(params[1])
    var nc = _index(params[2])
    with GILReleased(Python()):
        _positive(nr, "n_rows")
        _positive(nf, "n_features")
        _positive(nc, "n_components")
        var x = read_f32(x_address, nr * nf)
        var components = read_f32(c_address, nc * nf)
        var out = host_tsvd_transform(x, components, nr, nf, nc)
        for i in range(nr * nc):
            op[i] = out[i]
    return PythonObject(0)


def pca_transform_binding(
    x_addr: PythonObject,
    mean_addr: PythonObject,
    components_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`PCA.transform` (whiten=False) on the host: center by `mean_`, then
    `out = (X - mean) . components^T` by `host_pca_transform`. params:
    n_rows, n_features, n_components. Returns 0. The whitened pair is
    `pca_whiten_transform` and `pca_whiten_inverse_transform` below."""
    if len(params) != 3:
        raise Error("pca_transform: params must contain 3 values")
    var x_address = _index(x_addr)
    var m_address = _index(mean_addr)
    var c_address = _index(components_addr)
    var op = f32_ptr(_index(out_addr))
    var nr = _index(params[0])
    var nf = _index(params[1])
    var nc = _index(params[2])
    with GILReleased(Python()):
        _positive(nr, "n_rows")
        _positive(nf, "n_features")
        _positive(nc, "n_components")
        var x = read_f32(x_address, nr * nf)
        var mu = read_f32(m_address, nf)
        var components = read_f32(c_address, nc * nf)
        var out = host_pca_transform(x, mu, components, nr, nf, nc)
        for i in range(nr * nc):
            op[i] = out[i]
    return PythonObject(0)


def _whiten_finite(values: List[Float32], what: String) raises:
    """`_pca_whiten_finite` of the GPU binding, over the copy this side
    reads, with the GPU binding's sentence."""
    for i in range(len(values)):
        if not isfinite(values[i]):
            raise Error("PCA whitening requires finite inputs and outputs")


def _pca_whiten_apply(
    input_addr: PythonObject,
    mean_addr: PythonObject,
    components_addr: PythonObject,
    singular_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
    inverse: Bool,
) raises -> PythonObject:
    """`bindings/_mojolearn_estimators.mojo::_pca_whiten_apply` without the
    DeviceContext: the same params check, the same dimension refusals in the
    same words, the same overlap refusal, the same finiteness and sign
    refusals BEFORE anything is computed, the same finiteness refusal of the
    output after, and the arithmetic by `host_pca_whiten_transform` or
    `host_pca_whiten_inverse_transform`. params: n_rows, n_features,
    n_components, n_fit_rows (the FIT's row count, DEVIATION 580)."""
    if len(params) != 4:
        raise Error("PCA whitening params require rows, features, components, fit_rows")
    var nr = _index(params[0])
    var nf = _index(params[1])
    var nc = _index(params[2])
    var nfit = _index(params[3])
    if nr < 1 or nf < 2 or nc < 1 or nc > nf or nfit < 2:
        raise Error("PCA whitening requires positive rows/components, features>=2 and fit_rows>=2")
    if nr > 2147483647 or nf > 2147483647 or nc > 2147483647 or nfit > 2147483647:
        raise Error("PCA whitening dimensions exceed Int32")
    if nr > 2147483647 // nf or nc > 2147483647 // nf:
        raise Error("PCA whitening matrix cell count exceeds Int32")
    var input_count = nr * (nc if inverse else nf)
    var output_count = nr * (nf if inverse else nc)
    var xa = _index(input_addr)
    var ma = _index(mean_addr)
    var ca = _index(components_addr)
    var sa = _index(singular_addr)
    var oa = _index(out_addr)
    var starts = List[Int]()
    starts.append(xa)
    starts.append(ma)
    starts.append(ca)
    starts.append(sa)
    var counts = List[Int]()
    counts.append(input_count)
    counts.append(nf)
    counts.append(nc * nf)
    counts.append(nc)
    for i in range(4):
        if starts[i] <= 0 or starts[i] % 4 != 0:
            raise Error("PCA whitening requires positive aligned FP32 pointers")
    if oa <= 0 or oa % 4 != 0:
        raise Error("PCA whitening requires positive aligned FP32 pointers")
    for i in range(4):
        if oa < starts[i] + counts[i] * 4 and starts[i] < oa + output_count * 4:
            raise Error("PCA whitening output must not overlap any input")
    var op = f32_ptr(oa)
    with GILReleased(Python()):
        var x = read_f32(xa, input_count)
        var mu = read_f32(ma, nf)
        var components = read_f32(ca, nc * nf)
        var singular = read_f32(sa, nc)
        _whiten_finite(x, "input")
        _whiten_finite(mu, "mean")
        _whiten_finite(components, "components")
        _whiten_finite(singular, "singular")
        for i in range(nc):
            if singular[i] < Float32(0):
                raise Error("PCA whitening singular values must be nonnegative")
        var out = (
            host_pca_whiten_inverse_transform(
                x, components, singular, mu, nr, nf, nc, nfit
            ) if inverse else host_pca_whiten_transform(
                x, mu, components, singular, nr, nf, nc, nfit
            )
        )
        _whiten_finite(out, "output")
        for i in range(output_count):
            op[i] = out[i]
    return PythonObject(0)


def pca_whiten_transform_binding(
    x_addr: PythonObject, mean_addr: PythonObject,
    components_addr: PythonObject, singular_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`PCA.transform` (whiten=True) on the host, the GPU binding's arity
    and argument order (x, mean, components, singular_values, out, params).
    Returns 0."""
    return _pca_whiten_apply(
        x_addr, mean_addr, components_addr, singular_addr, out_addr, params, False)


def pca_whiten_inverse_transform_binding(
    scores_addr: PythonObject, components_addr: PythonObject,
    singular_addr: PythonObject, mean_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`PCA.inverse_transform` (whiten=True) on the host, the GPU binding's
    arity and argument order (scores, components, singular_values, mean, out,
    params). Returns 0."""
    return _pca_whiten_apply(
        scores_addr, mean_addr, components_addr, singular_addr, out_addr, params, True)


def qn_decision_function_binding(
    x_addr: PythonObject,
    coef_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`LogisticRegression.decision_function` on the host: `scores = X w + b`
    by `host_qn_decision`, `w` the fitted `_w` of `n_features +
    fit_intercept` entries, the bias its LAST entry. params: n_rows,
    n_features, fit_intercept (0/1) and, since lane/logistic-multiclass
    (2026-09-14), an OPTIONAL 4th, n_classes, the GPU binding's contract:
    absent or below 3 is the binary shape; `n_classes > 2` the softmax
    shape by `host_qn_decision_multi`, `w` the column-major `C x dims`
    block and out `n_rows * n_classes` floats row-major. Returns 0."""
    if len(params) != 3 and len(params) != 4:
        raise Error("qn_decision_function: params must contain n_rows, n_features, fit_intercept and an optional n_classes")
    var x_address = _index(x_addr)
    var w_address = _index(coef_addr)
    var op = f32_ptr(_index(out_addr))
    var nr = _index(params[0])
    var nf = _index(params[1])
    var fi = _index(params[2]) != 0
    var nc = _index(params[3]) if len(params) == 4 else 1
    with GILReleased(Python()):
        _positive(nr, "n_rows")
        _positive(nf, "n_features")
        var x = read_f32(x_address, nr * nf)
        if nc > 2:
            var wm = read_f32(w_address, (nf + (1 if fi else 0)) * nc)
            var outm = host_qn_decision_multi(x, wm, nr, nf, nc, fi)
            for i in range(nr * nc):
                op[i] = outm[i]
        else:
            var w = read_f32(w_address, nf + (1 if fi else 0))
            var out = host_qn_decision(x, w, nr, nf, fi)
            for i in range(nr):
                op[i] = out[i]
    return PythonObject(0)


def qn_softmax_binding(
    scores_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """The multinomial `predict_proba` link (lane/logistic-multiclass,
    2026-09-14) by `host_qn_softmax`, the GPU binding's contract: scores
    float32 `(n_rows, n_classes)` row-major, out float64 `(n_rows,
    n_classes)`. params: n_rows, n_classes. Returns 0."""
    if len(params) != 2:
        raise Error("qn_softmax: params must contain n_rows, n_classes")
    var s_address = _index(scores_addr)
    var op = f64_ptr(_index(out_addr))
    var nr = _index(params[0])
    var nc = _index(params[1])
    if nc < 3:
        raise Error("qn_softmax: n_classes must be at least 3; the binary link is qn_sigmoid")
    with GILReleased(Python()):
        _positive(nr, "n_rows")
        var scores = read_f32(s_address, nr * nc)
        var out = host_qn_softmax(scores, nr, nc)
        for i in range(nr * nc):
            op[i] = out[i]
    return PythonObject(0)


def qn_sigmoid_binding(
    scores_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """The binary `predict_proba` link (DEVIATION 549) by `host_qn_sigmoid`:
    out is float64 `(n_rows, 2)`, `[1 - p, p]`. params: n_rows. Returns 0."""
    if len(params) != 1:
        raise Error("qn_sigmoid: params must contain n_rows")
    var s_address = _index(scores_addr)
    var op = f64_ptr(_index(out_addr))
    var nr = _index(params[0])
    with GILReleased(Python()):
        _positive(nr, "n_rows")
        var scores = read_f32(s_address, nr)
        var out = host_qn_sigmoid(scores, nr)
        for i in range(2 * nr):
            op[i] = out[i]
    return PythonObject(0)


@export
def PyInit__mojolearn_estimators_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_estimators_host")
        module.def_function[estimators_host_numeric_mode_binding]("estimators_host_numeric_mode")
        module.def_function[estimators_host_vendor_binding]("estimators_host_vendor")
        module.def_function[estimators_host_column_binding]("estimators_host_column")
        module.def_function[estimators_host_sabotage_binding]("estimators_host_sabotage")
        module.def_function[estimators_vendor_binding]("estimators_vendor")
        module.def_function[estimators_numeric_mode_binding]("estimators_numeric_mode")
        module.def_function[kde_score_samples_binding]("kde_score_samples")
        module.def_function[ols_predict_binding]("ols_predict")
        module.def_function[tsvd_transform_binding]("tsvd_transform")
        module.def_function[pca_transform_binding]("pca_transform")
        module.def_function[pca_whiten_transform_binding]("pca_whiten_transform")
        module.def_function[pca_whiten_inverse_transform_binding]("pca_whiten_inverse_transform")
        module.def_function[qn_decision_function_binding]("qn_decision_function")
        module.def_function[qn_sigmoid_binding]("qn_sigmoid")
        module.def_function[qn_softmax_binding]("qn_softmax")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_estimators_host: ", error))
