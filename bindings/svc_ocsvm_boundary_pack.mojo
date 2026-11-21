# SPDX-License-Identifier: Apache-2.0
# PACK of OneClassSVM (the door is bindings/svc_ocsvm_native_door.mojo): the
# public arrays a fit and a query return, and the fitted fields, published
# together once RUN has finished (a fit that fails leaves the estimator as
# it was, `_gamma` included).
from std.python import Python, PythonObject
from std.memory import unsafe_memcpy
from bindings.labels_descriptor_boundary_door import labels_gather
from core.ocsvm_plan import oc_f32_address, oc_i32_address
from bindings.py_balanced import py_setitem
from bindings.classical_arrays_door import classical_address
from bindings.family_common_pack import family_array
from bindings.svc_common_pack import svc_prefix, svc_scalar_array
from svm.svc_ocsvm_native_run import OcFitRun
from bindings.svc_ocsvm_boundary_unpack import OcFitUnpacked
from bindings.svm_keys_unpack import svm_keys


@fieldwise_init
struct OcFitOutputs(Movable):
    """The fit's public support (filled) and support vectors (allocated:
    the door's RUN gathers the rows of X into them), with their addresses."""
    var support: PythonObject
    var vectors: PythonObject
    var support_address: Int
    var vectors_address: Int
    var supports: Int


def oc_fit_outputs(unpacked: OcFitUnpacked, run: OcFitRun, context: PythonObject) raises -> OcFitOutputs:
    var n_sv = run.state.supports
    var support = PythonObject(None)
    if run.weights.selected and n_sv > 0:
        # The former first prefix Array was private. Preserve its copy in a
        # native owner before allocating the final mapped public support.
        var prefix = List[Int32](unsafe_uninit_length=n_sv)
        unsafe_memcpy(dest=prefix.unsafe_ptr(), src=run.state.support_rows.unsafe_ptr(), count=n_sv)
        support = family_array(Python.tuple(PythonObject(n_sv)), "<i4", context)
        labels_gather(context, 0, oc_i32_address(run.weights.rows), run.weights.count, oc_i32_address(prefix), n_sv,
                      Int(py=classical_address(support)))
        _ = prefix
    else:
        support = svc_prefix(oc_i32_address(run.state.support_rows), n_sv, Python.tuple(PythonObject(n_sv)), "<i4", context)
    var width = 0 if unpacked.request.precomputed else unpacked.request.features
    var vectors = family_array(Python.tuple(PythonObject(n_sv), PythonObject(width)), "<f4", context)
    return OcFitOutputs(support, vectors, Int(py=classical_address(support)), Int(py=classical_address(vectors)), n_sv)


def oc_fit_pack(unpacked: OcFitUnpacked, run: OcFitRun, outputs: OcFitOutputs, context: PythonObject) raises:
    """The fitted fields of `OneClassSVM.fit`."""
    var nk = svm_keys()
    var fields = unpacked.fields
    var n_sv = outputs.supports
    py_setitem(fields, nk[].k__gamma, PythonObject(run.gamma))
    py_setitem(fields, nk[].k_support_, outputs.support)
    py_setitem(fields, nk[].k_support_vectors_, outputs.vectors)
    py_setitem(fields, nk[].k_dual_coef_,
        svc_prefix(oc_f32_address(run.state.support_values), n_sv,
            Python.tuple(PythonObject(1), PythonObject(n_sv)), "<f4", context))
    py_setitem(fields, nk[].k_intercept_, svc_scalar_array(-run.rho, "<f4", context))
    py_setitem(fields, nk[].k_offset_, PythonObject(run.rho))
    py_setitem(fields, nk[].k_n_iter_, PythonObject(Int(run.state.iterations[0])))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(unpacked.request.features))
    py_setitem(fields, nk[].k_fit_status_, PythonObject(0))


@fieldwise_init
struct OcPublicVector(Movable):
    """A query's public result vector and its address for RUN."""
    var value: PythonObject
    var address: Int


def oc_public_vector(count: Int, typestr: String, context: PythonObject) raises -> OcPublicVector:
    """`count` zeroed items of `typestr` ("<f4" scores, "<i8" labels)."""
    var value = family_array(Python.tuple(PythonObject(count)), typestr, context)
    return OcPublicVector(value, Int(py=classical_address(value)))
