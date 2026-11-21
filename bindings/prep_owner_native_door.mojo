# SPDX-License-Identifier: Apache-2.0
# The public door of the x_prep family (registered as prep_encode_native_call).
# Every public preprocessing operation is one call here. The classes the
# encode owner serves (bindings/prep_encode_native_door.mojo) go on to
# prep_encode_c unchanged; the label encoders (bindings/prep_labels_native_door.mojo),
# IterativeImputer (bindings/prep_iterimp_native_door.mojo), RFE and the
# mutual-information functions (bindings/prep_select_native_door.mojo) and the
# univariate score functions (below) are dispatched by the model's class
# (`prep_owner_kinds`) or, for a module-level function (model None), by the
# operation name. Each owner builds the x_prep programs the Python built (the
# same stages, operands and order, so the same bits).
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_class_admission_door import class_record, class_admit, class_watch_ready
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.prep_encode_native_door import prep_encode_c, prep_encode_scores_call
from bindings.prep_labels_native_door import prep_labels_call
from bindings.prep_owner_route_unpack import prep_route_context, prep_route_vendor_used
from bindings.prep_iterimp_native_door import prep_iterimp_call
from bindings.prep_select_native_door import prep_rfe_call, prep_mutual_info_call
from bindings.prep_composition_lifecycle_door import composition_before, composition_after_fit
from bindings.parallel_public_result_door import parallel_public_result
from bindings.public_lifecycle_owner_door import lifecycle_begin, lifecycle_end, lifecycle_complete, lifecycle_result
from bindings.prep_keys_unpack import (
    prep_keys,
    PREP_OPN_CHI2, PREP_OPN_FIT, PREP_OPN_FIT_TRANSFORM, PREP_OPN_F_CLASSIF, PREP_OPN_F_REGRESSION,
    PREP_OPN_INVERSE_TRANSFORM, PREP_OPN_MUTUAL_INFO_CLASSIF, PREP_OPN_MUTUAL_INFO_REGRESSION,
    PREP_OPN_R_REGRESSION, PREP_OPN_TRANSFORM,
)
from bindings.py_keys import py_op_code
from bindings.py_balanced import py_item


def _owner_kind(model: PythonObject, context: PythonObject) raises -> String:
    var nk = prep_keys()
    # The owner kind of this model's class (or a base, by its MRO); "" when
    # the encode owner serves it (or model is None: a module function).
    if model is None:
        return ""
    var mods = NativeModules()
    var b = mods.module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native preprocessing refuses Python metaclass hooks")
    var kinds = py_item(context, nk[].k_prep_owner_kinds)
    # The model's own class is the MRO's first row: an exact owner class is
    # answered by that one lookup (perf3-prep-shared; same as the encode
    # owner's `_kind`). Any other class (the encode classes: the common
    # case) used to read its MRO and look up every row per call; the answer
    # depends only on the class rows and the connected kinds table, so it is
    # kept per class state (native_class_admission type watchers) and per
    # kinds-table identity.
    var own = py_get(kinds, cls)
    if own is not None:
        return String(own)
    var record = class_record(mods, _OWNER_KIND_SLOT, cls)
    if record is not None and record[0] is kinds:
        return String(record[1])
    var mro = py_call(py_item(b.type.__dict__, nk[].k___mro__).__get__, cls)
    var found = String("")
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        var kind = py_get(kinds, mro[k])
        if kind is not None:
            found = String(kind)
            break
    # The payload holds the kinds table (import-time public classes), never
    # `cls` itself. Not built where no record can be kept (Python 3.10/3.11).
    if class_watch_ready():
        class_admit(mods, _OWNER_KIND_SLOT, cls, Python.tuple(kinds, PythonObject(found)))
    return found


comptime _OWNER_KIND_SLOT = "prep_owner_kind_v1"


def prep_owner_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        var kind = _owner_kind(model, context)
        if kind == "" and model is not None:
            # SimpleImputer, KBinsDiscretizer, the encoders, SplineTransformer, SelectKBest
            return prep_encode_c(py_self, args_ptr)
        # The code of the operation (one lookup with the caller's str); the
        # name String stays for the lifecycle owner and the refusal texts.
        var op = py_op_code(prep_keys()[].ops, call[1])
        classical_connect_numpy(context)
        var result = PythonObject(None)
        if kind == "label_encoder" or kind == "label_binarizer" or kind == "multilabel_binarizer":
            # The label encoders' public lifecycle (the former `_mode`
            # wrappers): NotFittedError, fit bookkeeping and NumPy results;
            # their input is y, so no X table checks.
            var public = (op == PREP_OPN_FIT or op == PREP_OPN_FIT_TRANSFORM or op == PREP_OPN_TRANSFORM
                          or op == PREP_OPN_INVERSE_TRANSFORM)
            # The connected context of this estimator's device route (one
            # record read and one pointer compare; resolved once per
            # estimator, bindings/prep_owner_route_unpack.mojo). The label
            # owner reads no `device`.
            var run = prep_route_context(model, context)
            if run is not context:
                # the CPU context beside a device set: its own NumPy cells
                classical_connect_numpy(run)
            var depth = lifecycle_begin(model, operation, args[0], False, False, True, run) if public else 0
            try:
                if not public and operation == "vendor_used":
                    # the image of this estimator's device route, read back
                    result = prep_route_vendor_used(model, context)
                else:
                    result = prep_labels_call(model, kind, op, operation, args, run)
            finally:
                if public:
                    lifecycle_end(model, operation, depth)
            if public:
                lifecycle_complete(model, operation, PythonObject(None), False, run)
                result = lifecycle_result(model, operation, result, "", run)
        elif kind == "iterative_imputer":
            if op < 0 and operation == "vendor_used":
                # the image of this estimator's device route, read back (a
                # name outside the operation table: one integer test on a
                # data operation)
                return prep_route_vendor_used(model, context).steal_data()
            # The connected context of this estimator's device route (one
            # record read and one pointer compare; resolved once per
            # estimator). Every program of the operation, the initial
            # SimpleImputer included, selects its image from it; a member
            # estimator runs on its own `device`. Nothing below reads `device`.
            var run = prep_route_context(model, context)
            if run is not context:
                # the CPU context beside a device set: its own NumPy cells
                classical_connect_numpy(run)
            var prepared = composition_before(model, kind, op, operation, args, run)
            result = prep_iterimp_call(model, op, operation, prepared, run)
            if op == PREP_OPN_FIT or op == PREP_OPN_FIT_TRANSFORM:
                composition_after_fit(model, args[0], run)
            return parallel_public_result(result, operation, PythonObject(None), run).steal_data()
        elif kind == "rfe":
            var prepared = composition_before(model, kind, op, operation, args, context)
            result = prep_rfe_call(model, op, operation, prepared, context)
        elif kind != "":
            raise Error("TypeError: the native prep owner does not serve this class")
        elif op == PREP_OPN_MUTUAL_INFO_CLASSIF or op == PREP_OPN_MUTUAL_INFO_REGRESSION:
            result = prep_mutual_info_call(op, args, context)
        elif op == PREP_OPN_F_CLASSIF or op == PREP_OPN_CHI2 or op == PREP_OPN_F_REGRESSION or op == PREP_OPN_R_REGRESSION:
            result = prep_encode_scores_call(operation, args, context)
        else:
            raise Error("unknown native prep operation")
        if kind == "rfe":
            if op == PREP_OPN_FIT or op == PREP_OPN_FIT_TRANSFORM:
                composition_after_fit(model, args[0], context)
            result = parallel_public_result(result, operation, PythonObject(None), context)
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("IndexError:"):
            return raise_python_exception(error_without_class(error, "IndexError"), ExceptionType("PyExc_IndexError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
