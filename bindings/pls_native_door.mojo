# SPDX-License-Identifier: Apache-2.0
# The native owner of python/mojolearn/_expansion_decomp.py's PLS family
# (PLSRegression, PLSCanonical, CCA): fit, fit_transform, transform,
# inverse_transform, predict and score are each one call into pls_c, in
# three parts:
#   UNPACK  bindings/pls_boundary_unpack.mojo: the settings snapshot, the
#           admission of X and the target, the kit operands (resident fitted
#           operands on a query), the fit's option words;
#   RUN     bindings/pls_run.mojo over the typed kit (bindings/decomp_kit_run
#           .mojo): the steps `_PLS` ran (the centering and scaling, the one
#           `x_decomp_pls_fit` component loop, the rotations through `_pinv`,
#           the coefficients, the projections), the GIL released, the same
#           x_decomp entries with the same words in the same order, so the
#           bits are unchanged. fit_transform projects the X and Y the fit
#           already holds (one conversion and upload of each);
#   PACK    bindings/pls_boundary_pack.mojo: the fitted fields, the public
#           arrays, the fit envelope's instance marks.
# This file only sequences them. The x_decomp entries are the ESTIMATOR'S
# (its device route, `decomp_entries_routed`), as is the metrics module of
# `score`: no call here asks which device the process runs on.
#
# The fitted centering state is kept as float32 vectors (`_x_mean`,
# `_x_std`, `_y_mean`, `_y_std`, scikit-learn's names), the values the
# Python shell held as device matrices.
from core.array_metadata import ArrayMetadata
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from std.python._cpython import PyObjectPtr, GILReleased
from bindings.py_balanced import py_type, py_get, py_item
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_error_boundary_pack import classical_python_error
from bindings.parallel_public_result_door import parallel_public_result
from providers.python.array_metadata_pyapi_door import ArrayPin
from bindings.admitted_input_unpack import admit_f32
from bindings.metrics_cls_native_unpack import metrics_cls_operation
from bindings.operation_names_mcls import MCLS_OPN_FAMILY_MCLS_R2
from bindings.decomp_kit_native_door import (
    DecompDoor, decomp_entries_routed, decomp_kit_device_operation, decomp_kit_unpack,
)
from bindings.pls_run import PlsFit, pls_fit_run, pls_project, pls_rebuild, pls_predict_run
from bindings.pls_keys_unpack import (
    pls_keys,
    PLS_OPN_FIT,
    PLS_OPN_FIT_TRANSFORM,
    PLS_OPN_SCORE,
    PLS_OPN_PREDICT,
    PLS_OPN_TRANSFORM,
    PLS_OPN_INVERSE_TRANSFORM,
    PLS_OPN_VENDOR_USED,
)
from bindings.py_keys import py_op_code
from bindings.pls_boundary_unpack import (
    pls_public_unpack, pls_algorithm_check, pls_store, pls_fit_unpack, pls_transform_unpack, pls_inverse_unpack,
    pls_predict_unpack, pls_check_fitted, pls_score_binding,
)
from bindings.pls_boundary_pack import (
    pls_fit_pack, pls_pair_pack, pls_predict_pack, pls_fit_enter, pls_fit_leave, pls_fit_annotate, pls_fit_mark,
)


def _pls_fit(model: PythonObject, x_source: PythonObject, y_source: PythonObject, transform: Bool,
             context: PythonObject, fields: PythonObject, state: PythonObject,
             captured_x: Optional[ArrayMetadata] = None, captured_y: Optional[ArrayMetadata] = None,
             x_deferred: Bool = False, x_pin: Optional[ArrayPin] = None,
             y_pin: Optional[ArrayPin] = None) raises -> PythonObject:
    var nk = pls_keys()
    # `_PLS.fit` reads the captured settings and publishes into separate
    # state: one UNPACK, one RUN (bindings/pls_run.mojo), one PACK.
    var entries = decomp_entries_routed(model, fields, py_item(fields, nk[].k_numeric_mode), context)
    var door = DecompDoor(entries, py_item(context, nk[].k_output_context))
    var kit = decomp_kit_unpack(entries)
    try:
        var unpacked = pls_fit_unpack(door, kit, x_source, y_source, context, fields, state, captured_x, captured_y,
                                      x_deferred, x_pin, y_pin)
        # RUN: the centering, the one component-loop call, the rotations,
        # the coefficients (and fit_transform's projections), the GIL released.
        var fit = PlsFit()
        var py = Python()
        with GILReleased(py):
            fit = pls_fit_run(kit, unpacked.x, unpacked.ym, unpacked.words, transform)
        return pls_fit_pack(model, door, kit, fit, unpacked, transform, fields, state, context)
    finally:
        kit.close()


def pls_operation(model: PythonObject, operation: Int, args: PythonObject, context: PythonObject,
                  fields: PythonObject, state: PythonObject, captured_x: Optional[ArrayMetadata] = None,
                  captured_y: Optional[ArrayMetadata] = None, x_deferred: Bool = False,
                  x_pin: Optional[ArrayPin] = None, y_pin: Optional[ArrayPin] = None) raises -> PythonObject:
    var nk = pls_keys()
    if operation == PLS_OPN_FIT or operation == PLS_OPN_FIT_TRANSFORM:
        # args: (X, y, Y); y wins over Y, as PLSRegression.fit chose.
        var y = args[1] if args[1] is not None else args[2]
        pls_algorithm_check(fields)
        pls_store(fields, state, "numeric_mode_", py_item(fields, nk[].k_numeric_mode))
        return _pls_fit(model, args[0], y, operation == PLS_OPN_FIT_TRANSFORM, context, fields, state, captured_x,
                        captured_y, x_deferred, x_pin, y_pin)
    if operation == PLS_OPN_SCORE:
        # `RegressorScoreMixin.score`: y as float32 (`as_f32_c(y, ndim=None,
        # name="y")`), the native predict, then `metrics.r2_score(y, pred,
        # sample_weight=..., numeric_mode=...)` through the metrics owner
        # (`family_mcls_r2`, the entry r2_score calls, with its arguments);
        # args: (X, y, sample_weight); binding selection remains native.
        var owner = py_get(context, nk[].k_pls_metric_owner)
        if owner is None:
            raise Error("NotImplementedError: mojolearn: the PLS score needs the native metrics owner "
                        + "(family_mcls_r2); this install does not carry it")
        var binding = pls_score_binding(model, fields, owner[1])
        var truth = admit_f32(args[1], -1, "y", context, finite=False)
        var prediction = pls_operation(model, PLS_OPN_PREDICT, Python.tuple(args[0]), context, fields, state, captured_x,
                                       None, x_deferred, x_pin)
        var mode = py_get(fields, nk[].k_numeric_mode)
        var r2 = metrics_cls_operation(PythonObject(None), MCLS_OPN_FAMILY_MCLS_R2,
                         Python.tuple(binding, truth.array, prediction, args[2], PythonObject("uniform_average"),
                                      PythonObject(True), mode), owner[1])
        _ = truth^
        return r2
    pls_check_fitted(model, fields)
    var entries = decomp_entries_routed(model, fields, py_get(fields, nk[].k_numeric_mode_), context)
    var door = DecompDoor(entries, py_item(context, nk[].k_output_context))
    # A query: the fitted operands stay on the device across calls.
    door.owner = model
    var kit = decomp_kit_unpack(entries)
    try:
        var py = Python()
        if operation == PLS_OPN_TRANSFORM:
            # args: (X, y, Y).
            var y = args[1] if args[1] is not None else args[2]
            var o = pls_transform_unpack(door, kit, args, y, captured_x, x_pin, x_deferred, fields, context)
            var xs = -1
            var ys = -1
            with GILReleased(py):
                xs = pls_project(kit, o.a, o.a_mean, o.a_std, o.a_basis)
                if o.b >= 0:
                    ys = pls_project(kit, o.b, o.b_mean, o.b_std, o.b_basis)
            return pls_pair_pack(door, kit, xs, ys, o.b >= 0)
        if operation == PLS_OPN_INVERSE_TRANSFORM:
            var y = args[1] if args[1] is not None else args[2]
            var o = pls_inverse_unpack(door, kit, args, y, captured_x, x_pin, x_deferred, fields, context)
            var xr = -1
            var yr = -1
            with GILReleased(py):
                xr = pls_rebuild(kit, o.a, o.a_basis, o.a_std, o.a_mean)
                if o.b >= 0:
                    yr = pls_rebuild(kit, o.b, o.b_basis, o.b_std, o.b_mean)
            return pls_pair_pack(door, kit, xr, yr, o.b >= 0)
        if operation == PLS_OPN_PREDICT:
            var o = pls_predict_unpack(door, kit, args, captured_x, x_pin, x_deferred, fields, context)
            var prediction = -1
            with GILReleased(py):
                prediction = pls_predict_run(kit, o.a, o.a_mean, o.a_basis, o.b_mean)
            return pls_predict_pack(door, kit, prediction, fields)
        raise Error("unknown native PLS operation")
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


def pls_public_operation(model: PythonObject, operation: Int, operation_name: PythonObject, args: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    # Own the former _track_fit/_needs_fit/_public_output envelope: UNPACK
    # admits the call, the operation runs, PACK marks the instance.
    var admitted = pls_public_unpack(model, operation, operation_name, args, context)
    var result = PythonObject(None)
    if admitted.fitting:
        var depth = pls_fit_enter(admitted.fields, admitted.state)
        try:
            result = pls_operation(model, operation, args, context, admitted.fields, admitted.state, admitted.x.copy(),
                                   admitted.y, admitted.x_deferred, Optional(admitted.x_pin.copy()), admitted.y_pin)
        except error:
            pls_fit_annotate(String(error), admitted.name, admitted.x.shape[0], admitted.x.shape[1])
            raise error^
        finally:
            pls_fit_leave(admitted.state, depth)
        pls_fit_mark(admitted.state, admitted.feature_names)
        if operation == PLS_OPN_FIT:
            return result
    else:
        result = pls_operation(model, operation, args, context, admitted.fields, admitted.state, admitted.x.copy(), None,
                               admitted.x_deferred, Optional(admitted.x_pin.copy()))
    _ = admitted^  # the door's pins held X (and the target) through the operation (the kit holds its own owners)
    return parallel_public_result(result, String(py=operation_name), PythonObject("regressor"), context)


def pls_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: pls_native_call requires model, operation, arguments and connected context")
        var b = native_loaded_module("builtins")
        if py_type(call[1]) is not b.str or py_type(call[2]) is not b.tuple or py_type(call[3]) is not b.dict:
            raise Error("TypeError: native PLS requires an operation, an argument tuple and its context")
        var context = call[3]
        classical_connect_numpy(context)
        var nk = pls_keys()
        var operation = py_op_code(nk[].ops, call[1])
        if operation == PLS_OPN_VENDOR_USED:
            # Metadata, no input: the vendor of the image the estimator's route names.
            return decomp_kit_device_operation(call[0], False, call[2], context).steal_data()
        var result = pls_public_operation(call[0], operation, call[1], call[2], context)
        return result.steal_data()
    except error:
        return classical_python_error(error^, args_ptr)
