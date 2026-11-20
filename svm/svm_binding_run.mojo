# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN half of `bindings/_mojolearn_svm.mojo` (wave 8 run split, 2026-11-12):
the typed work of the SVC/SVR fit and predict doors and the IsolationForest
run door. Each door reads its Python addresses and parameter lists into ints
and floats, calls these pieces, and keeps every GIL release exactly where it
was before the split (around the solve, the predict, the forest run). Typed
Mojo only, no Python import (tools/python_free_dirs.sh). Moved word for word:
same copies, same stores, same order, so the bits are unchanged."""
from std.memory import unsafe_memcpy
from core.hostptr import f32_ptr, i32_ptr, read_f32, copy_f32
from svm.estimator import SvcFitOutputs, SvrFitOutputs
from isolation_forest.estimator import IF_WANT_PREDICT, IFRunOutputs


def svm_c_rows_run(a: Int, n_rows: Int) raises -> List[Float32]:
    """The OPTIONAL per-row bounds `C * sample_weight` at address `a`
    (float32, n_rows); 0 is the unweighted arm (an empty list)."""
    var out = List[Float32]()
    if a == 0:
        return out^
    out = read_f32(a, n_rows)
    return out^


def svm_predict_inputs_run(
    x_word: Int,
    n_rows: Int,
    n_cols: Int,
    n_support: Int,
    kernel: Int,
    dual_word: Int,
    support_word: Int,
    mut x: List[Float32],
    mut dual: List[Float32],
    mut support: List[Float32],
) raises:
    """The predict's host copies: X, and with support vectors the duals and
    (except kernel='precomputed') the support rows."""
    x = read_f32(x_word, max(0, n_rows * n_cols))
    if n_support > 0:
        var dp = f32_ptr(dual_word)
        var smp = f32_ptr(support_word)
        dual = read_f32(Int(dp), max(0, n_support))
        if kernel != 4:
            # kernel='precomputed' (4) reads no support rows: X is the
            # cross-kernel and the caller passes no support matrix.
            support = read_f32(Int(smp), max(0, n_support * n_cols))


def svc_fit_store_run(
    res: SvcFitOutputs,
    dp: MutPointer[Float32, MutUntrackedOrigin],
    sip: MutPointer[Int32, MutUntrackedOrigin],
    smp: MutPointer[Float32, MutUntrackedOrigin],
    ip: MutPointer[Float64, MutUntrackedOrigin],
    n_cols: Int,
    device_form: Bool,
) raises -> Int:
    """The SVC fit's model into the caller's buffers and its info words;
    refuses a flagged X in the short forms. Returns `n_support`."""
    copy_f32(res.dual_coefs.unsafe_ptr(), dp, res.n_support)
    unsafe_memcpy(dest=sip, src=res.support_idx.unsafe_ptr(), count=res.n_support)
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
    return res.n_support


def svr_fit_store_run(
    res: SvrFitOutputs,
    dp: MutPointer[Float32, MutUntrackedOrigin],
    sip: MutPointer[Int32, MutUntrackedOrigin],
    smp: MutPointer[Float32, MutUntrackedOrigin],
    ip: MutPointer[Float64, MutUntrackedOrigin],
    n_cols: Int,
    device_form: Bool,
) raises -> Int:
    """The SVR fit's model into the caller's buffers and its info words;
    refuses a flagged X in the short forms. Returns `n_support`."""
    copy_f32(res.dual_coefs.unsafe_ptr(), dp, res.n_support)
    unsafe_memcpy(dest=sip, src=res.support_idx.unsafe_ptr(), count=res.n_support)
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
    return res.n_support


def iforest_resident_info_run(
    res: IFRunOutputs, ip: MutPointer[Float64, MutUntrackedOrigin]
):
    """The resident run's four info words: offset_, max_samples_,
    n_features_in_ and the forest's token."""
    ip.unsafe_store(0, res.offset_)
    ip.unsafe_store(1, Float64(res.max_samples_))
    ip.unsafe_store(2, Float64(res.n_features_in_))
    ip.unsafe_store(3, Float64(res.token))


def iforest_outputs_run(
    res: IFRunOutputs,
    want: Int,
    out_i32_word: Int,
    out_f32_word: Int,
    n_query: Int,
    ip: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    """The refit-per-call run's output (labels for predict, else values)
    and its three info words."""
    if want == IF_WANT_PREDICT:
        var oi = i32_ptr(out_i32_word)
        unsafe_memcpy(dest=oi, src=res.labels.unsafe_ptr(), count=n_query)
    else:
        var of = f32_ptr(out_f32_word)
        unsafe_memcpy(dest=of, src=res.values.unsafe_ptr(), count=n_query)
    ip.unsafe_store(0, res.offset_)
    ip.unsafe_store(1, Float64(res.max_samples_))
    ip.unsafe_store(2, Float64(res.n_features_in_))
