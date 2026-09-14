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
dual coefficients, support indices, decision function); the decision is
`smo_oracle_decision`. The guards are the GPU entry's, in the GPU entry's
order and words (`svm/estimator.mojo::svc_fit_host_borrowed`, then
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
answering "cpu" and `svm_numeric_mode`. `svr_fit`, `svr_predict` and
`iforest_run` are deliberately absent and refuse BY NAME through
`_HostBinding` until their lanes land.

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
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from svm.host.smo_oracle import (
    SMO_ORACLE_HOST_SABOTAGE,
    OracleResult,
    smo_oracle_decision,
    smo_oracle_fit,
)
from svm.impl.svc_impl import unique_labels_sorted
from svm.impl.svm_parameter import (
    C_SVC,
    KERNEL_LINEAR,
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
    descending on purpose (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's
    negative control)."""
    return PythonObject(SMO_ORACLE_HOST_SABOTAGE)


# The GPU binding's names, same contract.


def svm_vendor_binding() raises -> PythonObject:
    """"cpu". On a CPU-only install `_backend.vendor()` is "cpu" and the
    read-back cross-check expects that string from every host binding."""
    return PythonObject(String("cpu"))


def svm_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: always 1 here, a host
    binding is IDENTICAL only (the build script refuses any other)."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def _kernel_params(kernel: Int, gamma: Float64) raises -> KernelParams:
    """`svm/estimator.mojo::_kernel_params`, the same refusal and the same
    constructor defaults (degree 3, coef0 0, read by no implemented
    kernel)."""
    if kernel != KERNEL_LINEAR and kernel != KERNEL_RBF:
        raise Error(
            "svm: kernel=" + String(kernel) + " is not implemented in rung 1;"
            + " only LINEAR (" + String(KERNEL_LINEAR) + ") and RBF ("
            + String(KERNEL_RBF) + ") are (svm/NOT_IMPLEMENTED.tsv)"
        )
    return KernelParams(kernel, 3, gamma, 0.0)


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

    The OUTPUT buffers are worst-case sized by the caller (`dual_addr` and
    `support_idx_addr` hold `n_rows` entries, `support_matrix_addr`
    `n_rows * n_features` float32); only the first `n_support` (and
    `n_support * n_features`) are written. `info_addr` is FIVE float64:
    b, n_support, n_iter, classes[0] (the SMALLER sorted distinct label),
    classes[1] (the LARGER, mapped to +1)."""
    if len(params) != 8:
        raise Error(
            "svc_fit: params must contain 8 values, got " + String(len(params))
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
        var kp = _kernel_params(kernel, gamma)
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
    `smo_oracle_decision` over the support matrix handed back in. Writes
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

    `predict_class` is `applyPrediction`'s epilogue, `label0 if val < 0
    else label1`, the spelling of `decision_kernel`."""
    if len(params) != 10:
        raise Error(
            "svc_predict: params must contain 10 values, got " + String(len(params))
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
        var kp = _kernel_params(kernel, gamma)
        var x = read_f32(x_address, n_rows * n_cols)
        var res = OracleResult[DType.float32]()
        res.b = b
        var support = List[Float32]()
        if n_support > 0:
            res.dual_coefs = read_f32(dual_address, n_support)
            support = read_f32(support_address, n_support * n_cols)
            for j in range(n_support):
                res.support_idx.append(Int32(j))
        # `smo_oracle_decision` gathers the support rows from the training
        # matrix by `support_idx`; the support matrix IS those rows in that
        # order, so it is passed as the "training" matrix with the identity
        # index. The arithmetic is `decision_kernel`'s.
        var dec = smo_oracle_decision[DType.float32](
            res, support, n_support, x, n_rows, n_cols, kp
        )
        for i in range(n_rows):
            var val = dec[i]
            if predict_class:
                op[i] = label0 if val < Float32(0.0) else label1
            else:
                op[i] = val
    return PythonObject(n_rows)


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
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_svm_host: ", error))
