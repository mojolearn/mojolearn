# SPDX-License-Identifier: Apache-2.0
# PACK of `SVC.fit` (bindings/svc_kernel_native_door.mojo sequences the
# operation): the public `support_vectors_` the one-vs-one gather writes in
# place, the compact public arrays cut from the fit's Mojo-owned buffers,
# and the fitted fields, published together after RUN in the order the fit
# always wrote them (a refused or failed fit leaves the estimator's previous
# fitted state whole).
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains, py_setitem
from bindings.family_common_pack import family_call, family_array
from bindings.svc_common_pack import svc_prefix, svc_scalar_array
from bindings.svc_common_run import svc_round_f32
from bindings.svm_keys_unpack import svm_keys
from svm.svc_kernel_fit_run import SvcFitResult
from bindings.svc_kernel_fit_unpack import SvcFitUnpacked


def svc_fit_vectors(unpacked: SvcFitUnpacked, result: SvcFitResult, context: PythonObject) raises -> PythonObject:
    """One-vs-one with support vectors: the public float32
    (total x n_features) `support_vectors_` RUN gathers into; else None."""
    if unpacked.plan.classes > 2 and result.total > 0:
        return family_array(Python.tuple(PythonObject(result.total), PythonObject(unpacked.plan.cols)), "<f4", context)
    return PythonObject(None)


def _f4(values: List[Float32], count: Int, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    return svc_prefix(Int(values.unsafe_ptr()) if len(values) else 0, count, shape, "<f4", context)


def _i4(values: List[Int32], count: Int, context: PythonObject) raises -> PythonObject:
    return svc_prefix(Int(values.unsafe_ptr()) if len(values) else 0, count, Python.tuple(PythonObject(count)), "<i4", context)


def _f8(values: List[Float64], count: Int, context: PythonObject) raises -> PythonObject:
    return svc_prefix(Int(values.unsafe_ptr()) if len(values) else 0, count, Python.tuple(PythonObject(count)), "<f8", context)


def svc_fit_pack(unpacked: SvcFitUnpacked, result: SvcFitResult, vectors: PythonObject,
                 context: PythonObject) raises:
    """The fitted fields. Every public array is built before the first
    field is written."""
    var fields = unpacked.fields
    var nk = svm_keys()
    var n_cols = unpacked.plan.cols
    var k = unpacked.plan.classes
    var n_pairs = k * (k - 1) // 2
    var prob_ab = Python.list()
    var prob_a = PythonObject(None)
    var prob_b = PythonObject(None)
    if unpacked.plan.probability:
        for q in range(n_pairs):  # driver-loop: over the class pairs (their sigmoid parameters)
            _ = py_call(prob_ab.append, Python.tuple(PythonObject(result.prob_a[q]), PythonObject(result.prob_b[q])))
        prob_a = _f8(result.prob_a, n_pairs, context)
        prob_b = _f8(result.prob_b, n_pairs, context)
    if k > 2:
        var pairs = Python.list()
        for q in range(len(result.pairs)):  # driver-loop: over the class pairs (one fitted machine each)
            var n_sv = result.pairs[q].supports
            var pair = Python.dict()
            py_setitem(pair, nk[].k_i, PythonObject(result.pairs[q].i))
            py_setitem(pair, nk[].k_j, PythonObject(result.pairs[q].j))
            py_setitem(pair, nk[].k_dual, _f4(result.pairs[q].dual, n_sv, Python.tuple(PythonObject(1), PythonObject(n_sv)), context))
            py_setitem(pair, nk[].k_sv, _f4(result.pairs[q].vectors, n_sv * result.pairs[q].vector_cols,
                                              Python.tuple(PythonObject(n_sv), PythonObject(result.pairs[q].vector_cols)), context))
            py_setitem(pair, nk[].k_support, _i4(result.pairs[q].support, n_sv, context))
            py_setitem(pair, nk[].k_b, PythonObject(result.pairs[q].bias))
            py_setitem(pair, nk[].k_n_iter, PythonObject(result.pairs[q].n_iter))
            _ = py_call(pairs.append, pair)
        var total = result.total
        var support: PythonObject
        var support_vectors: PythonObject
        var dual_coef: PythonObject
        if total > 0:
            support = _i4(result.support, total, context)
            support_vectors = vectors
            dual_coef = _f4(result.dual_all, (k - 1) * total, Python.tuple(PythonObject(k - 1), PythonObject(total)), context)
        else:
            support = family_array(Python.tuple(PythonObject(0)), "<i4", context)
            support_vectors = family_array(Python.tuple(PythonObject(0), PythonObject(n_cols)), "<f4", context)
            dual_coef = family_array(Python.tuple(PythonObject(k - 1), PythonObject(0)), "<f4", context)
        var intercept = _f4(result.intercept, n_pairs, Python.tuple(PythonObject(n_pairs)), context)
        var n_support = _i4(result.n_support, k, context)
        var n_iter = _i4(result.n_iter, n_pairs, context)
        py_setitem(fields, nk[].k_input_copied_, PythonObject(unpacked.copied))
        py_setitem(fields, nk[].k_classes_, unpacked.classes)
        py_setitem(fields, nk[].k_n_features_in_, PythonObject(n_cols))
        py_setitem(fields, nk[].k__pairs, pairs)
        py_setitem(fields, nk[].k_support_, support)
        py_setitem(fields, nk[].k_support_vectors_, support_vectors)
        py_setitem(fields, nk[].k_dual_coef_, dual_coef)
        py_setitem(fields, nk[].k_intercept_, intercept)
        py_setitem(fields, nk[].k_n_support_, n_support)
        py_setitem(fields, nk[].k_n_iter_, n_iter)
        py_setitem(fields, nk[].k__gamma, PythonObject(result.gamma))
        py_setitem(fields, nk[].k__label0, PythonObject(0.0))
        py_setitem(fields, nk[].k__label1, PythonObject(1.0))
    else:
        # The binary machine's public attributes: compact C-order Arrays
        # detached from the worst-case buffers.
        var n_support = result.supports
        var intercept = svc_scalar_array(svc_round_f32(result.solved.info[0]), "<f4", context)
        var support = _i4(result.solved.support, n_support, context)
        var support_vectors = _f4(result.solved.vectors, n_support * n_cols,
                                  Python.tuple(PythonObject(n_support), PythonObject(n_cols)), context)
        var dual_coef = _f4(result.solved.dual, n_support, Python.tuple(PythonObject(1), PythonObject(n_support)), context)
        py_setitem(fields, nk[].k_input_copied_, PythonObject(unpacked.copied))
        if py_contains(fields, nk[].k__pairs):
            _ = family_call(fields.pop, PythonObject("_pairs"), PythonObject(None))
        py_setitem(fields, nk[].k_classes_, unpacked.classes)
        py_setitem(fields, nk[].k_n_features_in_, PythonObject(n_cols))
        py_setitem(fields, nk[].k_n_support_, PythonObject(n_support))
        py_setitem(fields, nk[].k_n_iter_, PythonObject(Int(result.solved.info[2])))
        py_setitem(fields, nk[].k_intercept_, intercept)
        py_setitem(fields, nk[].k_support_, support)
        py_setitem(fields, nk[].k_support_vectors_, support_vectors)
        py_setitem(fields, nk[].k_dual_coef_, dual_coef)
        py_setitem(fields, nk[].k__gamma, PythonObject(result.gamma))
        py_setitem(fields, nk[].k__label0, PythonObject(svc_round_f32(result.solved.info[3])))
        py_setitem(fields, nk[].k__label1, PythonObject(svc_round_f32(result.solved.info[4])))
    if unpacked.plan.probability:
        py_setitem(fields, nk[].k__prob_ab, prob_ab)
        py_setitem(fields, nk[].k_probA_, prob_a)
        py_setitem(fields, nk[].k_probB_, prob_b)
    elif py_contains(fields, nk[].k__prob_ab):
        _ = family_call(fields.pop, PythonObject("_prob_ab"), PythonObject(None))
