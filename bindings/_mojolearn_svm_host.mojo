# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_svm` family, SVC today (the CPU training
lane, phase 1, 2026-09-13; brief docs/lanes/BRIEF_cpu_training_2026-09-13.md
sections 1.1 svc and 3.2).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The fit is
`svm/host/smo_oracle.mojo::smo_oracle_fit[DType.float32]`, "the SAME SMO,
serial ... one thread, one loop, ascending", the arm
`svc_check::check_device_matches_oracle` holds the device to bit for bit
under IDENTICAL (working-set sequence, alpha and f per outer iteration, b,
dual coefficients, support indices, decision function); prediction uses
`smo_oracle_decision_into`, the borrowed-pointer, row-parallel door over the
same per-row arithmetic as `smo_oracle_decision`. The guards are the GPU
entry's, in the GPU entry's order and words
(`svm/estimator.mojo::svc_fit_host_borrowed`, then
`svm/impl/svc_impl.mojo::svc_fit_borrowed` and `_svc_label_model`), through
the same host-only functions of `svm/impl/svm_parameter.mojo` and
`unique_labels_sorted` of `svc_impl.mojo`; the one-vs-rest targets are
`ovr_labels_kernel`'s rule (+1 where the label equals the LARGER of the two
sorted distinct labels), restated as `svc_check::ovr_y` restates it. The
support matrix is `CollectSupportVectorMatrix`'s gather, the rows
`support_idx` of X in support order.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES for the fits this covers, so
`python/mojolearn/_svm_impl.py::SVC` runs unchanged on a CPU-only install
through `_backend._HOST_MODULES` (`"_mojolearn_svm": "_mojolearn_svm_host"`):
`svc_fit` and `svc_predict` with the SAME address contract and worst-case
sized outputs (the eight-value and ten-value params lists and the five
float64 info slots, mirrored word for word in `_svm_impl.py`), `svm_vendor`
answering "cpu" and `svm_numeric_mode`; and, since the iforest lane
(2026-09-14), `iforest_run` under the GPU binding's 16-slot contract over
`isolation_forest/checks/if_oracle.mojo` (the docstring of
`iforest_run_binding` below). `svr_fit` and `svr_predict` joined on
2026-09-14 (lane/cpu-training-batch3, the svr and svr-linear lanes): they were
absent only because no lane had landed them, not for a reason in the
arithmetic. `smo_oracle_fit` has solved EPSILON_SVR through the same loop
since 2026-08-31 (its header, "THE REGRESSION ARM"), the SVR device gates hold
the device to that oracle, and the regression estimate is `svcPredict` with
the class epilogue off, which is `smo_oracle_decision`. The two entries keep
the GPU binding's nine-value and seven-value params lists and its three
float64 info slots.

What the CPU column certifies is what the lane hashes, the decision
function and the predicted labels on the training rows and on held-out
rows. `n_iter_` is written from the oracle's inner iteration count
(`OracleResult.n_iter`), which `svc_check` compares per outer iteration
with the device's trace, but no identity_break cell hashes it.
"""
from std.math import isnan
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32
from core.host_predict_threads import host_predict_task_count
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, ftz
from isolation_forest.checks.if_oracle import (
    OracleForest,
    oracle_fit,
    oracle_path_lengths,
    oracle_scores,
)
from isolation_forest.estimator import (
    IF_WANT_DECISION_FUNCTION,
    IF_WANT_PREDICT,
    IF_WANT_SCORE_SAMPLES,
    percentile_linear,
)
from isolation_forest.impl.isolation_forest import (
    IF_params,
    check_finite_by_name,
)
from isolation_forest.impl.rng.xorwow import (
    XORWOW_HOST_SABOTAGE,
    build_xorwow_tables,
)
from svm.host.smo_oracle import (
    SMO_ORACLE_HOST_SABOTAGE,
    OracleResult,
    smo_oracle_decision_into,
    smo_oracle_fit,
)
from svm.impl.svc_impl import unique_labels_sorted
from svm.impl.svm_parameter import (
    C_SVC,
    EPSILON_SVR,
    KERNEL_LINEAR,
    KERNEL_POLYNOMIAL,
    KERNEL_RBF,
    KernelParams,
    SvmParameter,
    check_finite_list,
    check_finite_ptr,
    check_rung1_scope,
)


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("svm host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


def svm_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def svm_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def svm_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_svm_host.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "svm host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_svm_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


# There is no `svm_host_detected_column` read-back, for the reason 8d16ce2f
# removed it from the forest and byte LM host bindings: the detected column
# folds to the GPU of the machine that ran the build, so its name would land
# in the vendor-neutral binary. The comptime assert above is the check.


def svm_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary walks every GEMM leaf of the SMO oracle
    descending on purpose, and advances the isolation forest's XORWOW one
    extra step per split fraction (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's
    negative control; one define, both arms)."""
    return PythonObject(SMO_ORACLE_HOST_SABOTAGE or XORWOW_HOST_SABOTAGE)


# The GPU binding's names, same contract.


def svm_vendor_binding() raises -> PythonObject:
    """"cpu". On a CPU-only install `_backend.vendor()` is "cpu" and the
    read-back cross-check expects that string from every host binding."""
    return PythonObject(String("cpu"))


def svm_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: always 1 here, a host
    binding is IDENTICAL only (the build script refuses any other)."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def _kernel_params(
    kernel: Int, gamma: Float64, degree: Int = 3, coef0: Float64 = 0.0
) raises -> KernelParams:
    """`svm/estimator.mojo::_kernel_params`, the same refusal and the same
    constructor defaults (degree 3, coef0 0, read by no implemented
    kernel)."""
    if kernel != KERNEL_LINEAR and kernel != KERNEL_RBF and kernel != KERNEL_POLYNOMIAL:
        raise Error(
            "svm: kernel=" + String(kernel) + " is not implemented in rung 1;"
            + " only LINEAR (" + String(KERNEL_LINEAR) + ") and RBF ("
            + String(KERNEL_RBF) + ") are (svm/NOT_IMPLEMENTED.tsv)"
        )
    return KernelParams(kernel, degree, gamma, coef0)


def svc_fit_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    dual_addr: PythonObject,
    support_idx_addr: PythonObject,
    support_matrix_addr: PythonObject,
    info_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`SVC.fit` on the host by `smo_oracle_fit`: binary C-SVC, dense FP32,
    LINEAR or RBF. Returns `n_support`.

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_svm_impl.py` and in the GPU binding):

        0  n_rows
        1  n_features
        2  kernel          (0 = LINEAR, 2 = RBF; cuML's KernelType values)
        3  gamma           (float; read only by RBF)
        4  C               (float)
        5  tol             (float)
        6  max_iter        (-1 = no limit, cuML's default)
        7  nochange_steps
        8  degree          (POLYNOMIAL only; 3 otherwise)
        9  coef0           (float; POLYNOMIAL only; 0 otherwise)

    The OUTPUT buffers are worst-case sized by the caller (`dual_addr` and
    `support_idx_addr` hold `n_rows` entries, `support_matrix_addr`
    `n_rows * n_features` float32); only the first `n_support` (and
    `n_support * n_features`) are written. `info_addr` is FIVE float64:
    b, n_support, n_iter, classes[0] (the SMALLER sorted distinct label),
    classes[1] (the LARGER, mapped to +1)."""
    if len(params) != 10:
        raise Error(
            "svc_fit: params must contain 10 values, got " + String(len(params))
        )
    var xp = f32_ptr(_index(x_addr))
    var y_address = _index(y_addr)
    var dp = f32_ptr(_index(dual_addr))
    var sip = i32_ptr(_index(support_idx_addr))
    var smp = f32_ptr(_index(support_matrix_addr))
    var ip = f64_ptr(_index(info_addr))
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    var kernel = _index(params[2])
    var gamma = Float64(py=params[3])
    var c = Float64(py=params[4])
    var tol = Float64(py=params[5])
    var max_iter = _index(params[6])
    var nochange_steps = _index(params[7])
    var degree = _index(params[8])
    var coef0 = Float64(py=params[9])
    if n_rows <= 0 or n_cols <= 0:
        raise Error("svc_fit: n_rows and n_features must both be positive")
    var labels = read_f32(y_address, max(0, n_rows))
    var n_support = 0
    with GILReleased(Python()):
        # `svc_fit_host_borrowed`'s guards and parameter pins, in its order.
        if n_rows <= 0:
            raise Error("Parameter n_rows: number of rows cannot be less than one")
        if n_cols <= 0:
            raise Error("Parameter n_cols: number of columns cannot be less than one")
        if len(labels) != n_rows:
            raise Error(
                "svc_fit_host: y has " + String(len(labels)) + " values, n_rows is "
                + String(n_rows)
            )
        var kp = _kernel_params(kernel, gamma, degree, coef0)
        var param = SvmParameter.default()
        param.C = c
        param.tol = tol
        param.max_iter = max_iter
        param.max_outer_iter = -1
        param.nochange_steps = nochange_steps
        param.cache_size = 0.0
        param.epsilon = 0.0
        param.svmType = C_SVC
        param.verbosity = 0
        check_rung1_scope(param, kp, False)
        # `svc_fit_borrowed`'s, then `_svc_label_model`'s.
        check_rung1_scope(param, kp, False)
        check_finite_ptr(xp, n_rows * n_cols, "X")
        check_finite_list(labels, "labels")
        var unique = unique_labels_sorted(labels)
        if len(unique) != 2:
            raise Error(
                "Only binary classification is implemented at the moment (got "
                + String(len(unique)) + " classes)"
            )
        var label0 = unique[0]
        var label1 = unique[1]
        # `ovr_labels_kernel`: +1 where the label equals the LARGER one.
        var y = List[Float32]()
        y.reserve(n_rows)
        for i in range(n_rows):
            y.append(Float32(1.0) if labels[i] == label1 else Float32(-1.0))
        var x = read_f32(Int(xp), n_rows * n_cols)
        # THE ONE CALL THAT COMPUTES ANYTHING.
        var res = smo_oracle_fit[DType.float32](x, y, n_rows, n_cols, param, kp)
        if isnan(res.b):
            # DEVIATION 637, the device's refusal in its words.
            raise Error(
                "SMO error: NaN found during fitting (DEVIATION 637: the"
                " intercept b is NaN, floating point overflow in f)"
            )
        n_support = len(res.dual_coefs)
        if len(res.support_idx) != n_support or n_support > n_rows:
            raise Error("svc_fit: the host oracle returned support of an unexpected length; nothing written")
        for j in range(n_support):
            dp[j] = res.dual_coefs[j]
            sip[j] = res.support_idx[j]
            # `CollectSupportVectorMatrix`: row `support_idx[j]` of X.
            var r = Int(res.support_idx[j])
            if r < 0 or r >= n_rows:
                raise Error("svc_fit: the host oracle returned a support index out of range; nothing written")
            for col in range(n_cols):
                smp[j * n_cols + col] = x[r * n_cols + col]
        ip[0] = Float64(res.b)
        ip[1] = Float64(n_support)
        ip[2] = Float64(res.n_iter)
        ip[3] = Float64(label0)
        ip[4] = Float64(label1)
    return PythonObject(n_support)


def svc_predict_binding(
    x_addr: PythonObject,
    dual_addr: PythonObject,
    support_matrix_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`SVC.decision_function` / `SVC.predict` on the host by
    `smo_oracle_decision_into` over the support matrix handed back in. Writes
    `n_rows` float32 to `out_addr` and returns `n_rows`.

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_svm_impl.py` and in the GPU binding):

        0  n_rows
        1  n_features
        2  n_support
        3  b               (float)
        4  classes[0]      (float)
        5  classes[1]      (float)
        6  kernel          (0 = LINEAR, 2 = RBF)
        7  gamma           (float; the gamma the FIT resolved)
        8  predict_class   (0 = the raw decision value, 1 = the class label)
        9  cache_size_mib  (float; the device's prediction BATCH knob,
                            launch-invariant by gate; only its positivity
                            is checked here, as on the device)
        10 degree          (POLYNOMIAL only; 3 otherwise)
        11 coef0           (float; POLYNOMIAL only; 0 otherwise)

    `predict_class` is `applyPrediction`'s epilogue, `label0 if val < 0
    else label1`, the spelling of `decision_kernel`."""
    if len(params) != 12:
        raise Error(
            "svc_predict: params must contain 12 values, got " + String(len(params))
        )
    var x_address = _index(x_addr)
    var op = f32_ptr(_index(out_addr))
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    var n_support = _index(params[2])
    var b = Float32(Float64(py=params[3]))
    var label0 = Float32(Float64(py=params[4]))
    var label1 = Float32(Float64(py=params[5]))
    var kernel = _index(params[6])
    var gamma = Float64(py=params[7])
    var predict_class = _index(params[8]) != 0
    var buffer_mib = Float64(py=params[9])
    var degree = _index(params[10])
    var coef0 = Float64(py=params[11])
    if n_rows <= 0 or n_cols <= 0:
        raise Error("svc_predict: n_rows and n_features must both be positive")
    if n_support < 0:
        raise Error("svc_predict: n_support cannot be negative")
    var dual_address = 0
    var support_address = 0
    if n_support > 0:
        dual_address = _index(dual_addr)
        support_address = _index(support_matrix_addr)
    with GILReleased(Python()):
        # `svc_predict_host`'s guards, in its order and words.
        if n_rows <= 0:
            raise Error("svc_predict_host: n_rows must be at least one")
        if n_cols <= 0:
            raise Error("svc_predict_host: n_cols must be at least one")
        if not (buffer_mib > 0.0):
            raise Error(
                "svc_predict_host: the predict buffer (cache_size) must be a"
                " positive number of MiB, got " + String(buffer_mib)
            )
        var kp = _kernel_params(kernel, gamma, degree, coef0)
        var xp = f32_ptr(x_address)
        if n_support == 0:
            for i in range(n_rows):
                op[i] = label0 if predict_class and b < Float32(0.0) else (
                    label1 if predict_class else b
                )
        else:
            # Borrow all three arrays and write decisions directly into the
            # caller's output. Class prediction safely maps those completed
            # rows in place; no model/query/output List is materialized.
            smo_oracle_decision_into(
                f32_ptr(dual_address), f32_ptr(support_address), n_support,
                xp, n_rows, n_cols, kp, b, op,
                host_predict_task_count(n_rows),
            )
            if predict_class:
                for i in range(n_rows):
                    op[i] = label0 if op[i] < Float32(0.0) else label1
    return PythonObject(n_rows)


def svr_fit_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    dual_addr: PythonObject,
    support_idx_addr: PythonObject,
    support_matrix_addr: PythonObject,
    info_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`SVR.fit` on the host by `smo_oracle_fit` under EPSILON_SVR:
    epsilon-SVR, dense FP32, LINEAR or RBF. Returns `n_support`.

    `params` is, in this exact order (the GPU binding's
    `bindings/_mojolearn_svm.mojo::svr_fit_binding`, mirrored in
    `python/mojolearn/_svm_impl.py`):

        0  n_rows
        1  n_features
        2  kernel          (0 = LINEAR, 2 = RBF)
        3  gamma           (float; read only by RBF)
        4  C               (float)
        5  epsilon         (float; the width of the insensitive tube)
        6  tol             (float)
        7  max_iter        (-1 = no limit)
        8  nochange_steps

    The output buffers are the classifier's worst-case sizes (`n_rows`,
    `n_rows`, `n_rows * n_features`); the doubled alpha domain is internal
    and `CombineCoefs` folds it back to `n_rows` before the selection, in
    the oracle as on the device. `info_addr` is THREE float64: b, n_support,
    n_iter. The guards are `svr_fit_host`'s, then `svr_fit`'s
    (`svm/estimator.mojo`, `svm/impl/svr_impl.mojo`), in their order."""
    if len(params) != 9:
        raise Error(
            "svr_fit: params must contain 9 values, got " + String(len(params))
        )
    var xp = f32_ptr(_index(x_addr))
    var y_address = _index(y_addr)
    var dp = f32_ptr(_index(dual_addr))
    var sip = i32_ptr(_index(support_idx_addr))
    var smp = f32_ptr(_index(support_matrix_addr))
    var ip = f64_ptr(_index(info_addr))
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    var kernel = _index(params[2])
    var gamma = Float64(py=params[3])
    var c = Float64(py=params[4])
    var epsilon = Float64(py=params[5])
    var tol = Float64(py=params[6])
    var max_iter = _index(params[7])
    var nochange_steps = _index(params[8])
    if n_rows <= 0 or n_cols <= 0:
        raise Error("svr_fit: n_rows and n_features must both be positive")
    var targets = read_f32(y_address, max(0, n_rows))
    var n_support = 0
    with GILReleased(Python()):
        # `svr_fit_host`'s guards and parameter pins, in its order.
        if n_rows <= 0:
            raise Error("Parameter n_rows: number of rows cannot be less than one")
        if n_cols <= 0:
            raise Error("Parameter n_cols: number of columns cannot be less than one")
        if len(targets) != n_rows:
            raise Error(
                "svr_fit_host: y has " + String(len(targets)) + " values, n_rows is "
                + String(n_rows)
            )
        var kp = _kernel_params(kernel, gamma)
        var param = SvmParameter.default()
        param.C = c
        param.tol = tol
        param.max_iter = max_iter
        param.max_outer_iter = -1
        param.nochange_steps = nochange_steps
        param.cache_size = 0.0
        param.epsilon = epsilon
        param.svmType = EPSILON_SVR
        param.verbosity = 0
        check_rung1_scope(param, kp, False)
        # `svr_fit`'s: the scope again, then DEVIATION 636's finite scans.
        check_rung1_scope(param, kp, False)
        check_finite_ptr(xp, n_rows * n_cols, "X")
        check_finite_list(targets, "labels")
        var x = read_f32(Int(xp), n_rows * n_cols)
        # THE ONE CALL THAT COMPUTES ANYTHING. `y` is the regression
        # targets; the oracle builds the +-1 label vector and the gradient
        # as `SvrInit` does.
        var res = smo_oracle_fit[DType.float32](x, targets, n_rows, n_cols, param, kp)
        if isnan(res.b):
            # DEVIATION 637, the device's refusal in its words.
            raise Error(
                "SMO error: NaN found during fitting (DEVIATION 637: the"
                " intercept b is NaN, floating point overflow in f)"
            )
        n_support = len(res.dual_coefs)
        if len(res.support_idx) != n_support or n_support > n_rows:
            raise Error("svr_fit: the host oracle returned support of an unexpected length; nothing written")
        for j in range(n_support):
            var r = Int(res.support_idx[j])
            if r < 0 or r >= n_rows:
                raise Error("svr_fit: the host oracle returned a support index out of range; nothing written")
        for j in range(n_support):
            dp[j] = res.dual_coefs[j]
            sip[j] = res.support_idx[j]
            # `CollectSupportVectorMatrix`: row `support_idx[j]` of X.
            var r = Int(res.support_idx[j])
            for col in range(n_cols):
                smp[j * n_cols + col] = x[r * n_cols + col]
        ip[0] = Float64(res.b)
        ip[1] = Float64(n_support)
        ip[2] = Float64(res.n_iter)
    return PythonObject(n_support)


def svr_predict_binding(
    x_addr: PythonObject,
    dual_addr: PythonObject,
    support_matrix_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`SVR.predict` on the host: `smo_oracle_decision_into` over the support
    matrix handed back in, `sum_j K(x, sv_j) dual_j + b`, the class
    epilogue off (the reference's `svcPredict(..., predict_class = false)`).
    Writes `n_rows` float32 to `out_addr` and returns `n_rows`.

    `params` is, in this exact order (the GPU binding's):

        0  n_rows
        1  n_features
        2  n_support
        3  b               (float)
        4  kernel          (0 = LINEAR, 2 = RBF)
        5  gamma           (float; the gamma the FIT resolved)
        6  cache_size_mib  (float; launch-invariant on the device, only its
                            positivity is checked here, as there)"""
    if len(params) != 7:
        raise Error(
            "svr_predict: params must contain 7 values, got " + String(len(params))
        )
    var x_address = _index(x_addr)
    var op = f32_ptr(_index(out_addr))
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    var n_support = _index(params[2])
    var b = Float32(Float64(py=params[3]))
    var kernel = _index(params[4])
    var gamma = Float64(py=params[5])
    var buffer_mib = Float64(py=params[6])
    if n_rows <= 0 or n_cols <= 0:
        raise Error("svr_predict: n_rows and n_features must both be positive")
    if n_support < 0:
        raise Error("svr_predict: n_support cannot be negative")
    var dual_address = 0
    var support_address = 0
    if n_support > 0:
        dual_address = _index(dual_addr)
        support_address = _index(support_matrix_addr)
    with GILReleased(Python()):
        # `svr_predict_host`'s guards, in its order and words.
        if n_rows <= 0:
            raise Error("svr_predict_host: n_rows must be at least one")
        if n_cols <= 0:
            raise Error("svr_predict_host: n_cols must be at least one")
        if not (buffer_mib > 0.0):
            raise Error(
                "svr_predict_host: the predict buffer (cache_size) must be a"
                " positive number of MiB, got " + String(buffer_mib)
            )
        var kp = _kernel_params(kernel, gamma)
        if n_support == 0:
            for i in range(n_rows):
                op[i] = b
        else:
            smo_oracle_decision_into(
                f32_ptr(dual_address), f32_ptr(support_address), n_support,
                f32_ptr(x_address), n_rows, n_cols, kp, b, op,
                host_predict_task_count(n_rows),
            )
    return PythonObject(n_rows)


def _iforest_host_scores(
    forest: OracleForest, x: List[Float32], n_rows: Int, n_cols: Int
) raises -> List[Float32]:
    """`_score_samples_device`'s guards in its words, then the oracle's
    path lengths and PAPER scores (1 = anomaly, 0.5 = normal; the Python
    layer's negation is the caller's)."""
    if n_rows <= 0:
        raise Error("Invalid n_rows " + String(n_rows))
    if n_cols != forest.n_features:
        raise Error(
            "X_query has "
            + String(n_cols)
            + " features, the model was fitted with "
            + String(forest.n_features)
        )
    check_finite_by_name("X_query", x, n_rows, n_cols)
    var pl = oracle_path_lengths(forest, x, n_rows, n_cols)
    return oracle_scores(forest, pl)


def iforest_run_binding(
    train_addr: PythonObject,
    query_addr: PythonObject,
    out_f32_addr: PythonObject,
    out_i32_addr: PythonObject,
    info_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`IsolationForest.fit` plus ONE of `score_samples`,
    `decision_function` or `predict`, in one call, on the host: the GPU
    binding's contract (`bindings/_mojolearn_svm.mojo::iforest_run_binding`,
    the 16-slot params list, `want` selecting the output buffer, the three
    float64 info slots), DEVIATION 874's fit-on-every-call kept so the
    identity surface is the GPU's.

    THE FIT AND SCORE ARE `isolation_forest/checks/if_oracle.mojo`'s
    (`oracle_fit`, `oracle_path_lengths`, `oracle_scores`), the second
    transcription of cuML's builder and scorer that `if_check` holds the
    kernels to, over the same XORWOW stream (`build_xorwow_tables`,
    DEVIATION 683, rebuilt on the host per call). The parameter resolution
    is `IsolationForestEstimator.fit`'s (`isolation_forest/estimator.mojo`,
    max_features, contamination, max_samples, the seed range) in its order
    and words, the guards `iforest_run_host`'s and `IsolationForest.
    error_checking`'s, the finite scan `check_finite_by_name`'s. The
    epilogues are the estimator's: `score_samples = -paper`,
    `decision_function = score_samples - Float32(offset_)`, `predict =
    -(paper > Float32(-offset_) ? 1 : -1)`, `offset_` the contamination
    quantile through `percentile_linear` over the training scores or -0.5.
    """
    if len(params) != 16:
        raise Error(
            "iforest_run: params must contain 16 values, got " + String(len(params))
        )
    var train_address = _index(train_addr)
    var query_address = _index(query_addr)
    var ip = f64_ptr(_index(info_addr))
    var n_train = _index(params[0])
    var n_features = _index(params[1])
    var n_query = _index(params[2])
    var n_estimators = _index(params[3])
    var max_samples_mode = _index(params[4])
    var max_samples_int = _index(params[5])
    var max_samples_frac = Float64(py=params[6])
    var max_depth = _index(params[7])
    var max_features_mode = _index(params[8])
    var max_features_int = _index(params[9])
    var max_features_frac = Float64(py=params[10])
    var bootstrap = _index(params[11]) != 0
    var random_state = _index(params[12])
    var contamination_auto = _index(params[13]) != 0
    var contamination = Float64(py=params[14])
    var want = _index(params[15])
    if n_train <= 0 or n_features <= 0 or n_query <= 0:
        raise Error("iforest_run: n_train, n_features and n_query must all be positive")
    var out_f32_address = 0
    var out_i32_address = 0
    if want == IF_WANT_PREDICT:
        out_i32_address = _index(out_i32_addr)
    else:
        out_f32_address = _index(out_f32_addr)
    var values = List[Float32]()
    var labels = List[Int32]()
    var offset_ = Float64(-0.5)
    var max_samples_ = 0
    with GILReleased(Python()):
        # `iforest_run_host`'s remaining guard.
        if want < IF_WANT_SCORE_SAMPLES or want > IF_WANT_PREDICT:
            raise Error(
                "iforest_run_host: want=" + String(want) + " is not one of 0"
                " (score_samples), 1 (decision_function), 2 (predict)"
            )
        var train = read_f32(train_address, n_train * n_features)
        var query = read_f32(query_address, n_query * n_features)
        # THE DEVICE STAGES EVERY INPUT CELL THROUGH `ftz` AT UPLOAD
        # (`_upload_f32`, `_upload_rowmajor_as_colmajor` in
        # `isolation_forest/impl/isolation_forest.mojo`; DEVIATION 1942 row),
        # for the training matrix and for every query, so a denormal input
        # is a signed zero to every kernel. The oracle reads its lists raw,
        # so the flush is applied here, once, at the same boundary. MEASURED
        # 2026-09-14: without it the CPU column's iforest/denormal train and
        # infer cells diverged from all three GPU columns on `scores` (the
        # GPU columns' denormal and denormal_ftz hashes are equal), the other
        # eight fixtures identical.
        for i in range(n_train * n_features):
            train[i] = ftz(train[i])
        for i in range(n_query * n_features):
            query[i] = ftz(query[i])
        # `IsolationForestEstimator.fit` (`:616-712`), in its order.
        var actual_max_features: Int
        if max_features_mode == 1:
            if max_features_int < 1 or max_features_int > n_features:
                raise Error(
                    "max_features must be an int in [1, n_features] or a float in (0.0, 1.0]."
                )
            actual_max_features = max_features_int
        else:
            if max_features_frac <= 0.0 or max_features_frac > 1.0:
                raise Error(
                    "max_features must be an int in [1, n_features] or a float in (0.0, 1.0]."
                )
            actual_max_features = Int(max_features_frac * Float64(n_features))
            if actual_max_features < 1:
                actual_max_features = 1
        var use_quantile = False
        if not contamination_auto:
            if contamination <= 0.0 or contamination > 0.5:
                raise Error(
                    "contamination must be 'auto' or a float in the range (0.0, 0.5]."
                )
            use_quantile = True
        var actual_max_samples: Int
        if max_samples_mode == 0:
            actual_max_samples = 256 if n_train > 256 else n_train
        elif max_samples_mode == 1:
            if max_samples_int <= 0:
                raise Error("max_samples must be a positive integer.")
            actual_max_samples = max_samples_int if max_samples_int < n_train else n_train
        else:
            if max_samples_frac <= 0.0 or max_samples_frac > 1.0:
                raise Error("float max_samples must be in (0.0, 1.0].")
            actual_max_samples = Int(max_samples_frac * Float64(n_train))
            if actual_max_samples < 1:
                raise Error(
                    "max_samples resolves to 0 samples; increase max_samples or the number of rows."
                )
        max_samples_ = actual_max_samples
        if random_state < 0 or random_state >= 4294967296:
            raise Error(
                "Expected `0 <= random_state <= 2**32 - 1`, got " + String(random_state)
            )
        var if_params = IF_params.default()
        if_params.n_estimators = n_estimators
        if_params.max_samples = actual_max_samples
        if_params.max_depth = max_depth if max_depth > 0 else -1
        if_params.max_features = actual_max_features
        if_params.bootstrap = bootstrap
        if_params.seed = UInt64(random_state)
        # `IsolationForest.error_checking` (`:538-549`), then DEVIATION 680's
        # finite scan (the borrowed path's threaded scan raises the same
        # words), then the fit.
        if if_params.n_estimators <= 0:
            raise Error(
                "n_estimators must be > 0, got " + String(if_params.n_estimators)
            )
        check_finite_by_name("X", train, n_train, n_features)
        var tables = build_xorwow_tables()
        var forest = oracle_fit(train, n_train, n_features, if_params, tables)
        if use_quantile:
            var paper_train = _iforest_host_scores(forest, train, n_train, n_features)
            var training_scores = List[Float32](capacity=n_train)
            for i in range(n_train):
                training_scores.append(-paper_train[i])
            offset_ = percentile_linear(training_scores, 100.0 * contamination)
        else:
            offset_ = -0.5
        var paper = _iforest_host_scores(forest, query, n_query, n_features)
        if want == IF_WANT_PREDICT:
            var threshold = Float32(-offset_)
            for i in range(n_query):
                var raw = Int32(1) if paper[i] > threshold else Int32(-1)
                labels.append(-raw)
        elif want == IF_WANT_DECISION_FUNCTION:
            var off = Float32(offset_)
            for i in range(n_query):
                values.append((-paper[i]) - off)
        else:
            for i in range(n_query):
                values.append(-paper[i])
        _ = tables^
    if want == IF_WANT_PREDICT:
        var oi = i32_ptr(out_i32_address)
        for i in range(n_query):
            oi[i] = labels[i]
    else:
        var of = f32_ptr(out_f32_address)
        for i in range(n_query):
            of[i] = values[i]
    ip[0] = offset_
    ip[1] = Float64(max_samples_)
    ip[2] = Float64(n_features)
    return PythonObject(n_query)


@export
def PyInit__mojolearn_svm_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_svm_host")
        module.def_function[svm_host_numeric_mode_binding]("svm_host_numeric_mode")
        module.def_function[svm_host_vendor_binding]("svm_host_vendor")
        module.def_function[svm_host_column_binding]("svm_host_column")
        module.def_function[svm_host_sabotage_binding]("svm_host_sabotage")
        module.def_function[svm_vendor_binding]("svm_vendor")
        module.def_function[svm_numeric_mode_binding]("svm_numeric_mode")
        module.def_function[svc_fit_binding]("svc_fit")
        module.def_function[svc_predict_binding]("svc_predict")
        module.def_function[svr_fit_binding]("svr_fit")
        module.def_function[svr_predict_binding]("svr_predict")
        module.def_function[iforest_run_binding]("iforest_run")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_svm_host: ", error))
