"""CPython boundary for the C-SVC and Isolation Forest lanes.

A THIRD extension module, and a separate one on purpose. The header of
`bindings/_mojolearn_estimators.mojo` states the reason and it is the same
reason here: an independently changing binding must not become a merge
point. Two lanes share this module because they landed in one slot, not
because they share code -- `svm/` is cuML's SMO solver and cuVS's kernel
matrices, `isolation_forest/` is cuML's Isolation Forest and cuRAND's
XORWOW, and nothing crosses between them.

Arrays cross as borrowed NumPy addresses; all device buffers and contexts
live for one call and no pointer is retained. The Python wrapper owns the
arrays and keeps them alive for the duration of the call
(`python/mojolearn/_arrays.py` is where that contract is written down).

SCALARS ARRIVE AS ONE LIST, NOT AS SEPARATE ARGUMENTS.
`PythonModuleBuilder.def_function` infers its signature from arity and
stops working above roughly nine arguments, so buffer addresses go
positionally and every scalar goes in one `params` list. THE ORDER OF THAT
LIST IS WRITTEN OUT IN A COMMENT ON BOTH SIDES IN THE SAME WORDS. A silent
reordering is a wrong answer, not a failure, and the length check below is
the only thing standing between a swapped pair and a plausible number.

DEVIATIONS 870-879 are this surface's. 870-875 are used and each is named
where it bites; 876-879 are unassigned.

NOT WIRED into `python/mojolearn/__init__.py` and not registered in
`python/mojolearn/_backend.py`'s `_MODULES`. Those two files are the
operator's convergence points. `_svm_impl.py` therefore loads this module
itself, mode-aware, and cross-checks `svm_numeric_mode()` against the mode
the package asked for; see the comment there.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from isolation_forest.estimator import (
    IF_WANT_PREDICT,
    IFRunOutputs,
    iforest_run_host,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from svm.estimator import SvcFitOutputs, svc_fit_host, svc_predict_host


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int32 buffer address")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)


def _f64_ptr(addr: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float64 buffer address")
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)


def svm_numeric_mode_binding() raises -> PythonObject:
    """1 when this binding was built under NUMERIC_IDENTICAL, else 0. The
    same shape as `gbdt_numeric_mode`, and for the same reason: the wrapper
    reads it once and refuses to run if the binary it loaded disagrees with
    the mode the package asked for. A wrong-arm measurement that is
    correctly labelled by accident is the failure this prevents."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return PythonObject(1)
    return PythonObject(0)


def svc_fit_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    dual_addr: PythonObject,
    support_idx_addr: PythonObject,
    support_matrix_addr: PythonObject,
    info_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`SVC.fit` (svm/, DEVIATIONS 630-637): binary C-SVC, dense FP32,
    LINEAR or RBF. Returns `n_support`.

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_svm_impl.py`):

        0  n_rows
        1  n_features
        2  kernel          (0 = LINEAR, 2 = RBF; cuML's KernelType values)
        3  gamma           (float; read only by RBF)
        4  C               (float)
        5  tol             (float)
        6  max_iter        (-1 = no limit, cuML's default)
        7  nochange_steps

    The OUTPUT buffers are worst-case sized by the caller, because
    `n_support` is not known until the solve finishes (DEVIATION 873):
    `dual_addr` and `support_idx_addr` hold `n_rows` entries,
    `support_matrix_addr` holds `n_rows * n_features` float32. Only the
    first `n_support` (and `n_support * n_features`) are written.

    `info_addr` is FIVE float64, written in this order:

        0  b            (the intercept; a float32 value widened exactly)
        1  n_support
        2  n_iter
        3  classes[0]   (the SMALLER of the two sorted distinct labels)
        4  classes[1]   (the LARGER; the one mapped to +1, so it fixes the
                         sign of the decision function)

    `cache_size`, `epsilon`, `svmType`, `sample_weight`, POLYNOMIAL, TANH
    and PRECOMPUTED are not on this surface at all: `svm/estimator.mojo`
    pins the first three at the only values rung 1 accepts and refuses the
    rest by name. `max_outer_iter` is pinned at -1, which is what cuML's
    own Python layer does (`svm_base.pyx:371`).
    """
    if len(params) != 8:
        raise Error(
            "svc_fit: params must contain 8 values, got " + String(len(params))
        )
    var xp = _f32_ptr(Int(py=x_addr))
    var yp = _f32_ptr(Int(py=y_addr))
    var dp = _f32_ptr(Int(py=dual_addr))
    var sip = _i32_ptr(Int(py=support_idx_addr))
    var smp = _f32_ptr(Int(py=support_matrix_addr))
    var ip = _f64_ptr(Int(py=info_addr))
    var n_rows = Int(py=params[0])
    var n_cols = Int(py=params[1])
    var kernel = Int(py=params[2])
    var gamma = Float64(py=params[3])
    var c = Float64(py=params[4])
    var tol = Float64(py=params[5])
    var max_iter = Int(py=params[6])
    var nochange_steps = Int(py=params[7])
    if n_rows <= 0 or n_cols <= 0:
        raise Error("svc_fit: n_rows and n_features must both be positive")
    var x = List[Float32]()
    var y = List[Float32]()
    for i in range(n_rows * n_cols):
        x.append(xp.unsafe_load(i))
    for i in range(n_rows):
        y.append(yp.unsafe_load(i))
    var res = SvcFitOutputs()
    with GILReleased(Python()):
        res = svc_fit_host(
            x, y, n_rows, n_cols, kernel, gamma, c, tol, max_iter,
            nochange_steps,
        )
    for i in range(res.n_support):
        dp.unsafe_store(i, res.dual_coefs[i])
        sip.unsafe_store(i, res.support_idx[i])
    for i in range(res.n_support * n_cols):
        smp.unsafe_store(i, res.support_matrix[i])
    ip.unsafe_store(0, Float64(res.b))
    ip.unsafe_store(1, Float64(res.n_support))
    ip.unsafe_store(2, Float64(res.n_iter))
    ip.unsafe_store(3, Float64(res.label0))
    ip.unsafe_store(4, Float64(res.label1))
    return PythonObject(res.n_support)


def svc_predict_binding(
    x_addr: PythonObject,
    dual_addr: PythonObject,
    support_matrix_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`SVC.decision_function` / `SVC.predict` on a model handed back in.
    Writes `n_rows` float32 to `out_addr` and returns `n_rows`.

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_svm_impl.py`):

        0  n_rows
        1  n_features
        2  n_support
        3  b               (float; the intercept from fit's info[0])
        4  classes[0]      (float)
        5  classes[1]      (float)
        6  kernel          (0 = LINEAR, 2 = RBF)
        7  gamma           (float; MUST be the gamma the fit resolved)
        8  predict_class   (0 = the raw decision value, 1 = the class label)
        9  cache_size_mib  (float; their `param.cache_size` at the
                            `SVC::predict` call site -- the prediction
                            BATCH knob, launch-invariant by gate)

    Slot 7 is the trap in this list. `gamma` here must be the value the
    FIT resolved, not the constructor's, or the kernel matrix at predict
    is not the one the dual coefficients were solved against and the
    answer is quietly wrong. `_svm_impl.py` stores it as `_gamma` at fit
    and passes that, which is what cuML does (`svm_base.pyx:464, 532`).
    """
    if len(params) != 10:
        raise Error(
            "svc_predict: params must contain 10 values, got " + String(len(params))
        )
    var xp = _f32_ptr(Int(py=x_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var n_rows = Int(py=params[0])
    var n_cols = Int(py=params[1])
    var n_support = Int(py=params[2])
    var b = Float32(Float64(py=params[3]))
    var label0 = Float32(Float64(py=params[4]))
    var label1 = Float32(Float64(py=params[5]))
    var kernel = Int(py=params[6])
    var gamma = Float64(py=params[7])
    var predict_class = Int(py=params[8]) != 0
    var buffer_mib = Float64(py=params[9])
    if n_rows <= 0 or n_cols <= 0:
        raise Error("svc_predict: n_rows and n_features must both be positive")
    if n_support < 0:
        raise Error("svc_predict: n_support cannot be negative")
    var x = List[Float32]()
    for i in range(n_rows * n_cols):
        x.append(xp.unsafe_load(i))
    var dual = List[Float32]()
    var support = List[Float32]()
    if n_support > 0:
        var dp = _f32_ptr(Int(py=dual_addr))
        var smp = _f32_ptr(Int(py=support_matrix_addr))
        for i in range(n_support):
            dual.append(dp.unsafe_load(i))
        for i in range(n_support * n_cols):
            support.append(smp.unsafe_load(i))
    var out = List[Float32]()
    with GILReleased(Python()):
        out = svc_predict_host(
            x, n_rows, n_cols, support, dual, n_support, b, label0, label1,
            kernel, gamma, predict_class, buffer_mib,
        )
    for i in range(n_rows):
        op.unsafe_store(i, out[i])
    return PythonObject(n_rows)


def iforest_run_binding(
    train_addr: PythonObject,
    query_addr: PythonObject,
    out_f32_addr: PythonObject,
    out_i32_addr: PythonObject,
    info_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`IsolationForest.fit` plus ONE of `score_samples`,
    `decision_function` or `predict`, in one call (isolation_forest/,
    DEVIATIONS 680-686 and 750-751). Returns `n_query`.

    DEVIATION 874: the fit happens on EVERY call, because the forest is
    eight device buffers and this boundary retains no device pointer. See
    the block comment at `isolation_forest/estimator.mojo::
    iforest_run_host`.

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_iforest_impl.py`):

         0  n_train
         1  n_features
         2  n_query
         3  n_estimators
         4  max_samples_mode     (0 = 'auto', 1 = int, 2 = float fraction)
         5  max_samples_int
         6  max_samples_frac     (float)
         7  max_depth            (-1 = None, their auto)
         8  max_features_mode    (0 = float fraction, 1 = int)
         9  max_features_int
        10  max_features_frac    (float)
        11  bootstrap            (0/1)
        12  random_state         (an int in [0, 2^32-1]; None is 0 here)
        13  contamination_auto   (0/1)
        14  contamination        (float; read only when slot 13 is 0)
        15  want                 (0 = score_samples, 1 = decision_function,
                                  2 = predict)

    Slot 15 picks which output buffer is written. `want` 0 and 1 write
    `n_query` float32 to `out_f32_addr` and touch `out_i32_addr` not at
    all; `want` 2 writes `n_query` int32 to `out_i32_addr` and touches
    `out_f32_addr` not at all. The unused address may be 0.

    `info_addr` is THREE float64, written on every call:

        0  offset_          (the contamination quantile, or -0.5)
        1  max_samples_     (the resolved subsample size)
        2  n_features_in_
    """
    if len(params) != 16:
        raise Error(
            "iforest_run: params must contain 16 values, got " + String(len(params))
        )
    var tp = _f32_ptr(Int(py=train_addr))
    var qp = _f32_ptr(Int(py=query_addr))
    var ip = _f64_ptr(Int(py=info_addr))
    var n_train = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_query = Int(py=params[2])
    var n_estimators = Int(py=params[3])
    var max_samples_mode = Int(py=params[4])
    var max_samples_int = Int(py=params[5])
    var max_samples_frac = Float64(py=params[6])
    var max_depth = Int(py=params[7])
    var max_features_mode = Int(py=params[8])
    var max_features_int = Int(py=params[9])
    var max_features_frac = Float64(py=params[10])
    var bootstrap = Int(py=params[11]) != 0
    var random_state = Int(py=params[12])
    var contamination_auto = Int(py=params[13]) != 0
    var contamination = Float64(py=params[14])
    var want = Int(py=params[15])
    if n_train <= 0 or n_features <= 0 or n_query <= 0:
        raise Error("iforest_run: n_train, n_features and n_query must all be positive")
    var train = List[Float32]()
    var query = List[Float32]()
    for i in range(n_train * n_features):
        train.append(tp.unsafe_load(i))
    for i in range(n_query * n_features):
        query.append(qp.unsafe_load(i))
    var res = IFRunOutputs()
    with GILReleased(Python()):
        res = iforest_run_host(
            train, n_train, n_features, query, n_query, n_estimators,
            max_samples_mode, max_samples_int, max_samples_frac, max_depth,
            max_features_mode, max_features_int, max_features_frac,
            bootstrap, random_state, contamination_auto, contamination,
            want,
        )
    if want == IF_WANT_PREDICT:
        var oi = _i32_ptr(Int(py=out_i32_addr))
        for i in range(n_query):
            oi.unsafe_store(i, res.labels[i])
    else:
        var of = _f32_ptr(Int(py=out_f32_addr))
        for i in range(n_query):
            of.unsafe_store(i, res.values[i])
    ip.unsafe_store(0, res.offset_)
    ip.unsafe_store(1, Float64(res.max_samples_))
    ip.unsafe_store(2, Float64(res.n_features_in_))
    return PythonObject(n_query)


@export
def PyInit__mojolearn_svm() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_svm")
        m.def_function[svm_numeric_mode_binding]("svm_numeric_mode")
        m.def_function[svc_fit_binding]("svc_fit")
        m.def_function[svc_predict_binding]("svc_predict")
        m.def_function[iforest_run_binding]("iforest_run")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_svm: ", e))
