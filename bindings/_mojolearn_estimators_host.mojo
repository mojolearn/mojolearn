# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_estimators` family, KernelDensity today
(the CPU training lane, phase 1, 2026-09-13; brief
docs/lanes/BRIEF_cpu_training_2026-09-13.md sections 1.1 kde and 3.2).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The arithmetic is
`kde/checks/kde_oracle.mojo::oracle_score_samples`, the float32 serial
replay the device arm is gated against bit for bit under IDENTICAL ("every
formula spelled here a SECOND time rather than imported from `kde/impl/`").
The validation is the GPU entry's, in the GPU entry's order
(`kde/estimator.mojo::kde_score_samples_host_ptr`: kernel and metric names,
`kde_fit_validate`, `n_query`, train data, query data), through the same
host-only functions of `kde/impl/neighbors/kernel_density.mojo`, so a bad
call raises the same error and nothing is written on a refusal.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES for the fit this covers, so
`python/mojolearn/density.py::KernelDensity` runs unchanged on a CPU-only
install through `_backend._HOST_MODULES` (`"_mojolearn_estimators":
"_mojolearn_estimators_host"`): `kde_score_samples` with the SAME address
contract (train, query, weights, out, the five-value params list, kernel,
metric; mirrored word for word in `density.py`), `estimators_numeric_mode`
and `estimators_vendor` (answering "cpu"). Every other function of
`bindings/_mojolearn_estimators.mojo` (dbscan_fit, pca_fit, tsvd_fit,
ols_fit, ridge_fit, qn_fit, ...) is deliberately absent, so those lanes
refuse BY NAME through `_HostBinding` and never hash something else.
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import f32_ptr, read_f32
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
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
    """Whether this binary sums every logsumexp row descending on purpose
    (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control)."""
    return PythonObject(KDE_ORACLE_HOST_SABOTAGE)


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
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_estimators_host: ", error))
