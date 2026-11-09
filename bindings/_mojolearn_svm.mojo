# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the C-SVC, epsilon-SVR and Isolation Forest lanes.

A THIRD extension module, and a separate one on purpose. The header of
`bindings/_mojolearn_estimators.mojo` states the reason and it is the same
reason here: an independently changing binding must not become a merge
point. Two lanes share this module because they landed in one slot, not
because they share code -- `svm/` is cuML's SMO solver and cuVS's kernel
matrices, `isolation_forest/` is cuML's Isolation Forest and cuRAND's
XORWOW, and nothing crosses between them.

THE SVR PAIR RIDES THIS SAME MODULE AND NEEDED NO NEW EXTENSION (added
2026-09-01). `svr_fit` and `svr_predict` call the same `SmoSolver` the
classifier does, differing in the gradient initialization, the domain size
and how the coefficients are combined, so `_backend.py`'s `_MODULES`,
`pyproject.toml` and `build_release_wheel.sh`'s `EXT_NAMES` are unchanged.
The whole cost of the regression surface is the two functions below.

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

`SVC`, `SVR` and `IsolationForest` ARE wired into
`python/mojolearn/__init__.py`; this header said "NOT WIRED" for a week
after they were. `_mojolearn_svm` is also registered in
`python/mojolearn/_backend.py`'s `_MODULES` (added with DEVIATION 869, when
five bindings were found missing from that tuple and were therefore
resolving to the FAST binary under the identical label). `_svm_impl.py`
still loads this module itself, mode-aware, and cross-checks
`svm_numeric_mode()` against the mode the package asked for; see the comment
there.
"""

# DEVIATION 2486: shared byte-preserving host copies.
from core.experiment_guards_kernel import KERNEL_GUARDS_OK
from bindings.device_plan_binding import device_plan_enter_binding, device_plan_exit_binding, device_plan_log_binding
from core.device_plan import device_plan_current
from core.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32, copy_f32, list_f32
from std.os import abort
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from checks.vendor import COMPILED_VENDOR
from svm.native_descriptor_dispatch import SVC_NATIVE_API
from providers.python.neural_descriptor_pyapi import neural_descriptor_capsule
from isolation_forest.impl.isolation_forest import (
    IF_DEVICE_TRANSPOSE,
    IDN_IF_QUERY_DEVICE,
    IDN_IF_RESIDENT,
)

from isolation_forest.impl.isolation_tree_builder import IF_FAST_ROWMAJOR
from isolation_forest.estimator import (
    IF_WANT_PREDICT,
    IFResidentKey,
    IFRunOutputs,
    if_resident_new_token,
    if_resident_put,
    if_resident_release,
    if_resident_take,
    iforest_run_host,
    iforest_run_resident,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.sys import is_defined

# Lane p3-svm-gp-kernel: the resident forest on every tier. `IDN_IF_RESIDENT`
# is IDENTICAL only, so a FAST binary refit the whole forest (and re-ran the
# contamination quantile) inside every score_samples / decision_function /
# predict call. The resident entry is mode-agnostic (the forest `fit` built,
# scored without a refit, isolation_forest/estimator.mojo "THE RESIDENT
# ENTRY"), so FAST takes it too; `-D MOJOLEARN_IDN_IF_RESIDENT_OFF` still
# restores the refit-per-call form on both tiers. FAST outputs are those of
# the fit-time forest, the refit's own forest for unchanged training data.
comptime _IF_RESIDENT_ROUTE = IDN_IF_RESIDENT or (
    GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_IDN_IF_RESIDENT_OFF"]()
)
# cgfin-c-svm: the epilogues, Platt, the shuffle and the row glue run on
# the device (the CPU binding runs svm/host/svc_proba.mojo's host twin)
from providers.python.svm_impl_svc_epilogue_pyapi import (
    svc_pair_epilogue_device_binding,
    svc_platt_train_device_binding,
    svc_splitmix_perm_device_binding,
)
from svm.estimator import (
    SvcFitOutputs,
    SvrFitOutputs,
    svc_fit_host,
    svc_fit_host_borrowed,
    svc_predict_host,
    svr_fit_host_borrowed,
    svr_predict_host,
)


from providers.python.svm_impl_scale_gamma_device_pyapi import scale_gamma_limbs_device_binding, py2mojo_linear_flags_binding
from svm.estimator import _family_ctx as _sg_family_ctx
from core.device_query_x import QueryFold
from svm.svm_resident_state import (
    SvmResidentLease,
    svm_resident_lease_open,
    svm_resident_stats_words,
    svc_predict_resident_host,
    svr_predict_resident_host,
)
from providers.python.svm_impl_svc_ovo_layout_pyapi import svc_ovo_layout_device_binding, svc_dual_gemv_device_binding

def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return i32_ptr(addr)


def _f64_ptr(addr: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    return f64_ptr(addr)


def svm_native_api_binding() raises -> PythonObject:
    var table = SVC_NATIVE_API.get_or_create_ptr()
    return neural_descriptor_capsule(table[].name, table[].entries)


def svm_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER, as the `NUMERIC_*` code itself: 0 FAST, 1
    IDENTICAL. The same shape as `gbdt_numeric_mode`, and for
    the same reason: the wrapper reads it once and refuses to run if the
    binary it loaded disagrees with the mode the package asked for. A
    wrong-arm measurement that is correctly labelled by accident is the
    failure this prevents."""
    return PythonObject(GLOBAL_NUMERIC_MODE)



def _c_rows(params: PythonObject, slot: Int, n_rows: Int, who: String) raises -> List[Float32]:
    """The OPTIONAL per-row bounds `C * sample_weight` (float32, formed and
    rounded once by the caller; `InitPenalty`'s weighted arm) at the address
    in `params[slot]`; absent or 0 is the unweighted arm. The solver checks
    them (`check_c_rows`)."""
    var out = List[Float32]()
    if len(params) <= slot:
        return out^
    var a = Int(py=params[slot])
    if a == 0:
        return out^
    out = read_f32(a, n_rows)
    return out^

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
        8  degree          (POLYNOMIAL only; 3 otherwise)
        9  coef0           (float; POLYNOMIAL only; 0 otherwise)

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

    THE DEVICE-X FORM (lane a8a): 15 values, the ten above, then

        10 the per-row bounds' address (0: unweighted)
        11 a cross-validation fold's base id   (-1: no fold)
        12 the fold's base rows
        13 the fold's index id                 (-1: no fold)
        14 scale_gamma     (1: gamma='scale' is resolved from the fit's own
                            device copy of X; slot 3 is then ignored)

    With a fold, `x_addr` is 0 and the rows are gathered on the device.
    `info_addr` then holds EIGHT float64: the five above, then X's finite
    word (5), its range word (6) and the gamma solved with (7). A flagged X
    (5 or 6 nonzero) is not solved: the door refuses it in admit_x's
    wording 'Input X contains NaN or infinity (SVC).'
    """
    if len(params) != 10 and len(params) != 11 and len(params) != 15:
        raise Error(
            "svc_fit: params must contain 10 values (11 with the per-row bounds, 15 with a fold and"
            " scale_gamma), got " + String(len(params))
        )
    var device_form = len(params) == 15
    var fold = QueryFold.none()
    var scale_gamma = False
    if device_form:
        fold = QueryFold(Int(py=params[11]), Int(py=params[12]), Int(py=params[13]))
        scale_gamma = Int(py=params[14]) != 0
    var x_word = 0 if fold.active() else Int(py=x_addr)
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
    var degree = Int(py=params[8])
    var coef0 = Float64(py=params[9])
    if n_rows <= 0 or n_cols <= 0:
        raise Error("svc_fit: n_rows and n_features must both be positive")
    var y = List[Float32]()
    y = read_f32(Int(yp), max(0, n_rows))
    var c_rows = _c_rows(params, 10, n_rows, "svc_fit")
    var res = SvcFitOutputs()
    with GILReleased(Python()):
        # DEVIATION 2665: X is not copied into a host List; the fit stages
        # the caller's buffer (or gathers the fold) on the device once.
        res = svc_fit_host_borrowed(
            x_word, fold, y, n_rows, n_cols, kernel, gamma, c, tol, max_iter,
            nochange_steps, degree, coef0, c_rows, scale_gamma,
        )
    copy_f32(res.dual_coefs.unsafe_ptr(), dp, res.n_support)
    memcpy(dest=sip, src=res.support_idx.unsafe_ptr(), count=res.n_support)
    copy_f32(res.support_matrix.unsafe_ptr(), smp, res.n_support * n_cols)
    ip.unsafe_store(0, Float64(res.b))
    ip.unsafe_store(1, Float64(res.n_support))
    ip.unsafe_store(2, Float64(res.n_iter))
    ip.unsafe_store(3, Float64(res.label0))
    ip.unsafe_store(4, Float64(res.label1))
    if device_form:
        ip.unsafe_store(5, Float64(res.finite_word))
        ip.unsafe_store(6, Float64(res.range_word))
        ip.unsafe_store(7, res.gamma)
    elif res.finite_word != 0:
        # The short forms carry no status slots: refuse in admit_x's wording.
        raise Error("Input X contains NaN or infinity (SVC).")
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
        10 degree          (POLYNOMIAL only; 3 otherwise)
        11 coef0           (float; POLYNOMIAL only; 0 otherwise)

    Slot 7 is the trap in this list. `gamma` here must be the value the
    FIT resolved, not the constructor's, or the kernel matrix at predict
    is not the one the dual coefficients were solved against and the
    answer is quietly wrong. `_svm_impl.py` stores it as `_gamma` at fit
    and passes that, which is what cuML does (`svm_base.pyx:464, 532`).
    """
    if len(params) != 12:
        raise Error(
            "svc_predict: params must contain 12 values, got " + String(len(params))
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
    var degree = Int(py=params[10])
    var coef0 = Float64(py=params[11])
    if n_rows <= 0 or n_cols <= 0:
        raise Error("svc_predict: n_rows and n_features must both be positive")
    if n_support < 0:
        raise Error("svc_predict: n_support cannot be negative")
    var x = List[Float32]()
    x = read_f32(Int(xp), max(0, n_rows * n_cols))
    var dual = List[Float32]()
    var support = List[Float32]()
    if n_support > 0:
        var dp = _f32_ptr(Int(py=dual_addr))
        var smp = _f32_ptr(Int(py=support_matrix_addr))
        dual = read_f32(Int(dp), max(0, n_support))
        if kernel != 4:
            # kernel='precomputed' (4) reads no support rows: X is the
            # cross-kernel and the caller passes no support matrix.
            support = read_f32(Int(smp), max(0, n_support * n_cols))
    var out = List[Float32]()
    with GILReleased(Python()):
        out = svc_predict_host(
            x, n_rows, n_cols, support, dual, n_support, b, label0, label1,
            kernel, gamma, predict_class, buffer_mib, degree, coef0,
        )
    copy_f32(out.unsafe_ptr(), op, n_rows)
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
    """`SVR.fit` (svm/, DEVIATIONS 630-637 plus the SVR pieces gated at
    `fea6becc`): epsilon-SVR, dense FP32, LINEAR or RBF. Returns
    `n_support`.

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_svm_impl.py`):

        0  n_rows
        1  n_features
        2  kernel          (0 = LINEAR, 2 = RBF; cuML's KernelType values)
        3  gamma           (float; read only by RBF)
        4  C               (float)
        5  epsilon         (float; the width of the insensitive tube)
        6  tol             (float)
        7  max_iter        (-1 = no limit, cuML's default)
        8  nochange_steps

    SLOT 5 IS THE ONE THIS LIST HAS THAT `svc_fit`'s DOES NOT, and it sits
    between `C` and `tol` rather than at the end, so a reader comparing the
    two lists sees one insertion instead of a reordering. `epsilon` crosses
    UNCLAMPED: `check_rung1_scope` refuses a non-finite or negative value by
    name and that refusal has to stay reachable from Python.

    The OUTPUT buffers are worst-case sized by the caller, because
    `n_support` is not known until the solve finishes (DEVIATION 873), and
    they are the SAME SIZES the classifier uses: `dual_addr` and
    `support_idx_addr` hold `n_rows` entries, `support_matrix_addr` holds
    `n_rows * n_features` float32. NOT `2 * n_rows`. The solver's domain is
    doubled but `Results` folds the two alpha halves before it selects, so
    at most `n_rows` coefficients can ever be written
    (`svm/estimator.mojo::svr_fit_host` spells the argument out).

    `info_addr` is THREE float64, written in this order:

        0  b            (the intercept; a float32 value widened exactly)
        1  n_support
        2  n_iter

    THREE, NOT FIVE. `svc_fit`'s slots 3 and 4 are the class labels and a
    regressor has none.

    THE DEVICE-X FORM (lane a8a): 16 values, the nine above, then the per-row
    bounds' address (9; 0: unweighted), degree (10), coef0 (11), a
    cross-validation fold's base id (12; -1: no fold), base rows (13), index
    id (14; -1: no fold) and scale_gamma (15; 1: gamma='scale' resolved from
    the fit's own device copy of X, slot 3 ignored). With a fold `x_addr` is
    0 and the rows are gathered on the device. `info_addr` then holds SIX
    float64: the three below, then X's finite word, its range word and the
    gamma solved with; a flagged X is not solved and the door refuses it.
    X is never copied into a host List: one copy to the device from
    `x_addr` (the former form read it element by element into a List,
    scanned it serially and staged it again).

    `cache_size`, `svmType`, `sample_weight`, POLYNOMIAL, TANH and
    PRECOMPUTED are not on this surface at all: `svm/estimator.mojo` pins
    the first two at the only values this rung accepts and refuses the rest
    by name. `max_outer_iter` is pinned at -1, which is what cuML's own
    Python layer does (`svm_base.pyx:371`), and -1 resolves over the DOUBLED
    `n_train`.
    """
    if len(params) != 9 and len(params) != 10 and len(params) != 12 and len(params) != 16:
        raise Error(
            "svr_fit: params must contain 9 values (10 with the per-row bounds, 12 with"
            " degree and coef0, 16 with a fold and scale_gamma), got " + String(len(params))
        )
    var device_form = len(params) == 16
    var fold = QueryFold.none()
    var scale_gamma = False
    if device_form:
        fold = QueryFold(Int(py=params[12]), Int(py=params[13]), Int(py=params[14]))
        scale_gamma = Int(py=params[15]) != 0
    var x_word = 0 if fold.active() else Int(py=x_addr)
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
    var epsilon = Float64(py=params[5])
    var tol = Float64(py=params[6])
    var max_iter = Int(py=params[7])
    var nochange_steps = Int(py=params[8])
    if n_rows <= 0 or n_cols <= 0:
        raise Error("svr_fit: n_rows and n_features must both be positive")
    # TOMBSTONE: the element-by-element host List copy of X (read_f32) before svr_fit_host removed 2026-11-09 by lane a8a-device-x-fit-svm-gp-krr: one device copy from x_addr; recoverable at 0ce88411d3 (docs/TOMBSTONES.md).
    var y = List[Float32]()
    y = read_f32(Int(yp), max(0, n_rows))
    var c_rows = _c_rows(params, 9, n_rows, "svr_fit")
    var degree = 3
    var coef0 = 0.0
    if len(params) >= 12:
        degree = Int(py=params[10])
        coef0 = Float64(py=params[11])
    var res = SvrFitOutputs()
    with GILReleased(Python()):
        res = svr_fit_host_borrowed(
            x_word, fold, y, n_rows, n_cols, kernel, gamma, c, epsilon, tol, max_iter,
            nochange_steps, c_rows, degree, coef0, scale_gamma,
        )
    copy_f32(res.dual_coefs.unsafe_ptr(), dp, res.n_support)
    memcpy(dest=sip, src=res.support_idx.unsafe_ptr(), count=res.n_support)
    copy_f32(res.support_matrix.unsafe_ptr(), smp, res.n_support * n_cols)
    ip.unsafe_store(0, Float64(res.b))
    ip.unsafe_store(1, Float64(res.n_support))
    ip.unsafe_store(2, Float64(res.n_iter))
    if device_form:
        ip.unsafe_store(3, Float64(res.finite_word))
        ip.unsafe_store(4, Float64(res.range_word))
        ip.unsafe_store(5, res.gamma)
    elif res.finite_word != 0:
        # The short forms carry no status slots: refuse in admit_x's wording.
        raise Error("Input X contains NaN or infinity (SVR).")
    return PythonObject(res.n_support)


def svr_predict_binding(
    x_addr: PythonObject,
    dual_addr: PythonObject,
    support_matrix_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`SVR.predict` on a model handed back in. Writes `n_rows` float32 to
    `out_addr` and returns `n_rows`.

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_svm_impl.py`):

        0  n_rows
        1  n_features
        2  n_support
        3  b               (float; the intercept from fit's info[0])
        4  kernel          (0 = LINEAR, 2 = RBF)
        5  gamma           (float; MUST be the gamma the fit resolved)
        6  cache_size_mib  (float; their `param.cache_size` at the
                            `SVM::predict` call site -- the prediction
                            BATCH knob, launch-invariant by gate)

    SEVEN SLOTS, NOT TEN. `svc_predict`'s `classes[0]`, `classes[1]` and
    `predict_class` are all absent: there are no classes and there is no
    epilogue to select, so the regression estimate is the only thing this
    call can return.

    Slot 5 is the trap in this list, exactly as slot 7 is in `svc_predict`'s.
    `gamma` here must be the value the FIT resolved, not the constructor's,
    or the kernel matrix at predict is not the one the dual coefficients
    were solved against and the answer is quietly wrong. `_svm_impl.py`
    stores it as `_gamma` at fit and passes that, which is what cuML does
    (`svm_base.pyx:464, 532`).
    """
    if len(params) != 7 and len(params) != 9:
        raise Error(
            "svr_predict: params must contain 7 values (9 with degree and coef0), got "
            + String(len(params))
        )
    var xp = _f32_ptr(Int(py=x_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var n_rows = Int(py=params[0])
    var n_cols = Int(py=params[1])
    var n_support = Int(py=params[2])
    var b = Float32(Float64(py=params[3]))
    var kernel = Int(py=params[4])
    var gamma = Float64(py=params[5])
    var buffer_mib = Float64(py=params[6])
    var degree = 3
    var coef0 = 0.0
    if len(params) == 9:
        degree = Int(py=params[7])
        coef0 = Float64(py=params[8])
    if n_rows <= 0 or n_cols <= 0:
        raise Error("svr_predict: n_rows and n_features must both be positive")
    if n_support < 0:
        raise Error("svr_predict: n_support cannot be negative")
    var x = List[Float32]()
    x = read_f32(Int(xp), max(0, n_rows * n_cols))
    var dual = List[Float32]()
    var support = List[Float32]()
    if n_support > 0:
        var dp = _f32_ptr(Int(py=dual_addr))
        var smp = _f32_ptr(Int(py=support_matrix_addr))
        dual = read_f32(Int(dp), max(0, n_support))
        if kernel != 4:
            # kernel='precomputed' (4) reads no support rows: X is the
            # cross-kernel and the caller passes no support matrix.
            support = read_f32(Int(smp), max(0, n_support * n_cols))
    var out = List[Float32]()
    with GILReleased(Python()):
        out = svr_predict_host(
            x, n_rows, n_cols, support, dual, n_support, b, kernel, gamma,
            buffer_mib, degree, coef0,
        )
    copy_f32(out.unsafe_ptr(), op, n_rows)
    return PythonObject(n_rows)


# ---------------------------------------------------------------- resident
# Device-resident fitted state (lane ne1-a-gp-svm, 2026-11-06;
# svm/svm_resident_state.mojo, core/fitted_resident_store.mojo). The public
# door takes a lease once per fitted state and names it, with the machine's
# slot, on every predict: the support rows and duals cross once, the query
# and the decisions cross between the caller's memory and the device with no
# host list.


def svm_resident_lease_binding() raises -> PythonObject:
    """A new lease: `(owner, lease)`. `owner` is the native object whose
    destruction frees every device copy the lease holds."""
    var lease = svm_resident_lease_open()
    var owner = PythonObject(alloc=SvmResidentLease(lease))
    return Python.tuple(owner, PythonObject(lease))


def svm_resident_stats_binding() raises -> PythonObject:
    """`(live leases, kept entries, kept bytes, uploads, cap bytes)` of this
    binding's store: a test-only readout."""
    var words = svm_resident_stats_words()
    return Python.tuple(PythonObject(words[0]), PythonObject(words[1]), PythonObject(words[2]),
                        PythonObject(words[3]), PythonObject(words[4]))


def _svm_query_fold(params: PythonObject, at: Int) raises -> QueryFold:
    """The fold's store ids at `params[at:at + 3]` (lane a7a), or none."""
    if len(params) < at + 3:
        return QueryFold.none()
    return QueryFold(Int(py=params[at]), Int(py=params[at + 1]), Int(py=params[at + 2]))


def svc_predict_resident_binding(
    x_addr: PythonObject,
    dual_addr: PythonObject,
    support_matrix_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`svc_predict` with the same arguments and `params` = its 12 values
    followed by two words: the lease (`svm_resident_lease`; 0 = this call
    only) and the machine's slot, then optionally three more: a
    cross-validation fold's store ids [base id, base rows, index id]
    (bindings/device_rows_door.mojo door_rows_store; `x_addr` is then 0 and
    the rows are gathered on the device). Writes `n_rows` float32 to
    `out_addr` and returns [finite word, range word] of the query (0, 0:
    finite and in range; the door refuses otherwise, lane a7a)."""
    if len(params) != 14 and len(params) != 17:
        raise Error(
            "svc_predict_resident: params must contain 14 values (svc_predict's 12, lease, slot) or 17"
            " (and the fold's base id, base rows, index id), got " + String(len(params))
        )
    var x = Int(py=x_addr)
    var out_word = Int(py=out_addr)
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
    var degree = Int(py=params[10])
    var coef0 = Float64(py=params[11])
    var lease = Int(py=params[12])
    var slot = Int(py=params[13])
    if n_rows <= 0 or n_cols <= 0:
        raise Error("svc_predict: n_rows and n_features must both be positive")
    if n_support < 0:
        raise Error("svc_predict: n_support cannot be negative")
    var dual = Int(py=dual_addr) if n_support > 0 else 0
    var support = Int(py=support_matrix_addr) if n_support > 0 and kernel != 4 else 0
    var fold = _svm_query_fold(params, 14)
    var words: Tuple[Int32, Int32]
    with GILReleased(Python()):
        words = svc_predict_resident_host(
            lease, slot, x, n_rows, n_cols, support, dual, n_support, b, label0, label1,
            kernel, gamma, predict_class, buffer_mib, degree, coef0, out_word, fold,
        )
    return Python.list(Int(words[0]), Int(words[1]))


def svr_predict_resident_binding(
    x_addr: PythonObject,
    dual_addr: PythonObject,
    support_matrix_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`svr_predict` with the same arguments and `params` = its 9 values
    (degree and coef0 included) followed by two words: the lease and the
    slot, then optionally a fold's three store ids (as
    `svc_predict_resident`). Writes `n_rows` float32 to `out_addr` and
    returns [finite word, range word] of the query (lane a7a)."""
    if len(params) != 11 and len(params) != 14:
        raise Error(
            "svr_predict_resident: params must contain 11 values (svr_predict's 9, lease, slot) or 14"
            " (and the fold's base id, base rows, index id), got " + String(len(params))
        )
    var x = Int(py=x_addr)
    var out_word = Int(py=out_addr)
    var n_rows = Int(py=params[0])
    var n_cols = Int(py=params[1])
    var n_support = Int(py=params[2])
    var b = Float32(Float64(py=params[3]))
    var kernel = Int(py=params[4])
    var gamma = Float64(py=params[5])
    var buffer_mib = Float64(py=params[6])
    var degree = Int(py=params[7])
    var coef0 = Float64(py=params[8])
    var lease = Int(py=params[9])
    var slot = Int(py=params[10])
    if n_rows <= 0 or n_cols <= 0:
        raise Error("svr_predict: n_rows and n_features must both be positive")
    if n_support < 0:
        raise Error("svr_predict: n_support cannot be negative")
    var dual = Int(py=dual_addr) if n_support > 0 else 0
    var support = Int(py=support_matrix_addr) if n_support > 0 and kernel != 4 else 0
    var fold = _svm_query_fold(params, 11)
    var words: Tuple[Int32, Int32]
    with GILReleased(Python()):
        words = svr_predict_resident_host(
            lease, slot, x, n_rows, n_cols, support, dual, n_support, b, kernel, gamma,
            buffer_mib, degree, coef0, out_word, fold,
        )
    return Python.list(Int(words[0]), Int(words[1]))


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

    Lane fam2-forests (`IDN_IF_RESIDENT`, isolation_forest/estimator.mojo
    "THE RESIDENT ENTRY"): a binary whose `iforest_resident()` is 1 also
    takes SEVENTEEN params and FOUR info slots:

        params 16  token    (0 = fit a fresh forest and keep it; else the
                             token an earlier call returned: that resident
                             forest scores the query with NO fit; a token
                             that is gone or whose key differs refits as
                             DEVIATION 874 does and keeps the new forest)
        info    3  token    (the resident forest's token after this call)

    Sixteen params is the DEVIATION 874 form, unchanged.
    """
    var resident = False
    comptime if _IF_RESIDENT_ROUTE:
        resident = len(params) == 17
    if len(params) != 16 and not resident:
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
    # DEVIATION 2638: the training matrix is LENT by address (the Python
    # caller keeps it alive through this synchronous call) and transposed
    # into the pinned upload stage across the host pool; it is no longer
    # appended cell by cell into a List and transposed again cell by cell.
    var train = List[Float32]()
    var query = List[Float32]()
    # Lane fam-forests (`IDN_IF_QUERY_DEVICE`, isolation_forest/impl/
    # isolation_forest.mojo): the query matrix is LENT by address too and
    # flushed and scanned on the device; otherwise the List copy as before.
    var q_addr = 0
    comptime if IDN_IF_QUERY_DEVICE:
        q_addr = Int(qp)
    else:
        # the define's off arm: one memcpy (no per-cell host loop)
        query = list_f32(qp, n_query * n_features)
    # The IsolationForest device plan, read ONCE here at the binding entry
    # (the thread-scoped plan, else MOJOLEARN_IFOREST_DEVICE_COUNT); nothing
    # below reads it again (isolation_forest/multi_gpu.mojo::fit_dispatch).
    var plan = device_plan_current("MOJOLEARN_IFOREST_DEVICE_COUNT")
    var res = IFRunOutputs()
    comptime if _IF_RESIDENT_ROUTE:
        if resident:
            # The take and the put run under the GIL; the fit (on a miss) and
            # the scoring run with it released, on an entry no other thread
            # can reach while it is out of the store.
            var token = Int(py=params[16])
            var out_f32 = Int(py=out_f32_addr)
            var out_i32 = Int(py=out_i32_addr)
            var key = IFResidentKey(
                Int(tp), n_train, n_features, n_estimators, max_samples_mode,
                max_samples_int, max_samples_frac, max_depth, max_features_mode,
                max_features_int, max_features_frac, bootstrap, random_state,
                contamination_auto, contamination,
            )
            var held = if_resident_take(token, key)
            var new_token = 0
            if len(held) == 0:
                new_token = if_resident_new_token()
            try:
                with GILReleased(Python()):
                    res = iforest_run_resident(
                        plan, held, key, new_token, query, q_addr, n_query, want, out_f32, out_i32,
                    )
            except e:
                # Preserve an existing caller-owned forest after a refused
                # query. A new token has not reached Python yet: caching it
                # here would orphan its buffers until native-global teardown.
                if new_token == 0:
                    if_resident_put(held)
                raise e
            if_resident_put(held)
            ip.unsafe_store(0, res.offset_)
            ip.unsafe_store(1, Float64(res.max_samples_))
            ip.unsafe_store(2, Float64(res.n_features_in_))
            ip.unsafe_store(3, Float64(res.token))
            return PythonObject(n_query)
    with GILReleased(Python()):
        res = iforest_run_host(
            plan, train, n_train, n_features, query, n_query, n_estimators,
            max_samples_mode, max_samples_int, max_samples_frac, max_depth,
            max_features_mode, max_features_int, max_features_frac,
            bootstrap, random_state, contamination_auto, contamination,
            want, train_addr=Int(tp), query_addr=q_addr,
        )
    if want == IF_WANT_PREDICT:
        var oi = _i32_ptr(Int(py=out_i32_addr))
        memcpy(dest=oi, src=res.labels.unsafe_ptr(), count=n_query)
    else:
        var of = _f32_ptr(Int(py=out_f32_addr))
        memcpy(dest=of, src=res.values.unsafe_ptr(), count=n_query)
    ip.unsafe_store(0, res.offset_)
    ip.unsafe_store(1, Float64(res.max_samples_))
    ip.unsafe_store(2, Float64(res.n_features_in_))
    return PythonObject(n_query)


def svm_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR: 'metal', 'cuda',
    'hip' or 'none'. A compile-time constant folded in from
    `checks/vendor.mojo`, the same shape as the tier read-back
    (`gbdt_numeric_mode`): the answer comes from the binary that actually
    loaded, never from the directory it sat in or from the environment.
    `python/mojolearn/_backend.py` refuses at import when this disagrees
    with the vendor directory the set was loaded from."""
    return PythonObject(String(COMPILED_VENDOR))


# TOMBSTONE: the *_parallel_available multi-GPU capability probes (dead native entries: constant 1, no driver or package code looks them up) deleted 2026-10-26 by b3cad1ef6; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_parallel_probes.patch


def iforest_device_finite_scan_binding() raises -> PythonObject:
    """1 when this build's IsolationForest fit refuses a non-finite training
    cell through a device scan (IF_FAST_ROWMAJOR, FAST on Apple), so the
    Python layer skips its host scan of the same cells; 0 otherwise."""
    return PythonObject(1 if IF_FAST_ROWMAJOR else 0)


def iforest_resident_binding() raises -> PythonObject:
    """1 when this binary keeps the fitted forest resident between calls
    (`IDN_IF_RESIDENT`, lane fam2-forests) and `iforest_run` takes the
    17-value params with a token; 0 otherwise (DEVIATION 874's refit per
    call)."""
    comptime if _IF_RESIDENT_ROUTE:
        return PythonObject(1)
    return PythonObject(0)


def iforest_resident_release_binding(token: PythonObject) raises -> PythonObject:
    """Drop the resident forest `token` names (the Python estimator was
    refitted or collected). A token nobody holds is a no-op. Returns 0."""
    comptime if _IF_RESIDENT_ROUTE:
        if_resident_release(Int(py=token))
    return PythonObject(0)


def iforest_device_scan_binding() raises -> PythonObject:
    """1 when this binary's fit uploads the training matrix as raw row-major
    bytes and scans it for non-finite cells on the device
    (`IF_DEVICE_TRANSPOSE`, lane gap-trees-nv), so the Python layer's host
    scan before the fit is the same refusal twice; 0 otherwise."""
    comptime if IF_DEVICE_TRANSPOSE:
        return PythonObject(1)
    return PythonObject(0)



def svm_scale_gamma_limbs_binding(
    x_addr: PythonObject, count: PythonObject, out_addr: PythonObject
) raises -> PythonObject:
    """lane/apple-fast-py2mojo-linear: gamma='scale' exact sums on the device
    (`svm/impl/scale_gamma_device.mojo`)."""
    return scale_gamma_limbs_device_binding(_sg_family_ctx(), x_addr, count, out_addr)


def svc_ovo_layout_binding(
    sup_addrs: PythonObject, dual_addrs: PythonObject, meta: PythonObject, out_addrs: PythonObject
) raises -> PythonObject:
    """lane/apple-fast-py2mojo-linear: SVC's one-vs-one attribute layout on
    the device (`svm/impl/svc_ovo_layout.mojo`)."""
    return svc_ovo_layout_device_binding(_sg_family_ctx(), sup_addrs, dual_addrs, meta, out_addrs)


def svc_dual_gemv_binding(
    dual_addr: PythonObject, sv_addr: PythonObject, dims: PythonObject, out_addr: PythonObject
) raises -> PythonObject:
    """lane/apple-fast-py2mojo-linear: SVC coef_ (dual @ sv) on the device."""
    return svc_dual_gemv_device_binding(_sg_family_ctx(), dual_addr, sv_addr, dims, out_addr)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_svm.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_svm() -> PythonObject:
    comptime assert KERNEL_GUARDS_OK, "invalid integrated experiment configuration (kernel guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_svm")
        # TOMBSTONE: the *_parallel_available multi-GPU capability probes (dead native entries: constant 1, no driver or package code looks them up) deleted 2026-10-26 by b3cad1ef6; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_parallel_probes.patch
        m.def_function[device_plan_enter_binding]("device_plan_enter")
        m.def_function[device_plan_exit_binding]("device_plan_exit")
        m.def_function[device_plan_log_binding]("device_plan_log")
        # TOMBSTONE: the *_parallel_available multi-GPU capability probes (dead native entries: constant 1, no driver or package code looks them up) deleted 2026-10-26 by b3cad1ef6; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_parallel_probes.patch
        m.def_function[svm_vendor_binding]("svm_vendor")
        m.def_function[svm_numeric_mode_binding]("svm_numeric_mode")
        m.def_function[svc_fit_binding]("svc_fit")
        m.def_function[svm_native_api_binding]("svm_binary_api_v1")
        m.def_function[svc_predict_binding]("svc_predict")
        m.def_function[svc_pair_epilogue_device_binding]("svc_pair_epilogue")
        m.def_function[svc_platt_train_device_binding]("svc_platt_train")
        m.def_function[svc_splitmix_perm_device_binding]("svc_splitmix_perm")
        # TOMBSTONE: svc_portable_math (dead native entry: only the one-off py-consolidation check tools/py_consolidated/svc_equal.py called it) deleted 2026-10-26 by f6700e77c; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_svm.patch
        m.def_function[svr_fit_binding]("svr_fit")
        m.def_function[svr_predict_binding]("svr_predict")
        m.def_function[iforest_run_binding]("iforest_run")
        m.def_function[iforest_resident_binding]("iforest_resident")
        m.def_function[iforest_resident_release_binding]("iforest_resident_release")
        m.def_function[iforest_device_scan_binding]("iforest_device_scan")
        m.def_function[iforest_device_finite_scan_binding]("iforest_device_finite_scan")
        m.def_function[svm_scale_gamma_limbs_binding]("scale_gamma_limbs")
        m.def_function[py2mojo_linear_flags_binding]("py2mojo_linear_flags")
        m.def_function[svc_ovo_layout_binding]("svc_ovo_layout")
        m.def_function[svc_dual_gemv_binding]("svc_dual_gemv")
        # Device-resident fitted state (lane ne1-a-gp-svm).
        _ = m.add_type[SvmResidentLease]("_SvmResidentLease")
        m.def_function[svm_resident_lease_binding]("svm_resident_lease")
        m.def_function[svm_resident_stats_binding]("svm_resident_stats")
        m.def_function[svc_predict_resident_binding]("svc_predict_resident")
        m.def_function[svr_predict_resident_binding]("svr_predict_resident")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_svm: ", e))
