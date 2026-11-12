# SPDX-License-Identifier: Apache-2.0
# The native owner of python/mojolearn/_expansion_decomp.py's
# AlternatingLeastSquares: fit, recommend and similar_items are each one call
# into als_c. The owner runs the steps the Python shell ran (the confidence
# scale, the Philox factor draws, the one `x_decomp_als_fit` sweep call, the
# score row and the top-N order of `_topn`) through
# bindings/decomp_kit_native.mojo: the same x_decomp entries with the same
# words in the same order, so the bits are unchanged.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.memory import unsafe_memcpy
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_arrays_door import classical_address, classical_wrap_array
from bindings.decomp_public_lifecycle_unpack import decomp_public_begin
from bindings.decomp_public_lifecycle_pack import decomp_public_finish
from x_decomp.decomp_public_lifecycle_run import decomp_public_method
from bindings.public_lifecycle_owner_door import lifecycle_end
from bindings.decomp_kit_native_door import (
    DecompF32, DecompDoor, decomp_float, decomp_int, decomp_seed, decomp_seed_word, decomp_mode, decomp_entries,
    decomp_input, decomp_resident_drop, decomp_kit_unpack, decomp_kit_publish, decomp_kit_out_shape,
    decomp_door_borrow, decomp_door_input, decomp_door_resident_hit, decomp_door_resident_input,
    decomp_door_resident_derived, decomp_door_resident_keep,
)
from bindings.decomp_kit_run import DKit
from bindings.als_run import AlsFitWords, AlsTop, als_fit_run, als_recommend_run, als_unit_run, als_similar_run
from core.decomp_native_impl import decomp_f32_to_i32
from std.python._cpython import GILReleased
from bindings.py_lists import py_list_f64


#: The resident record name of the unit-normalised item factors (N325).
comptime _ALS_UNIT = "item_factors.unit"


def _fitted(mut door: DecompDoor, mut kit: DKit, source: PythonObject, name: String,
            context: PythonObject) raises -> Int:
    # UNPACK a fitted factor matrix (row N325): on a query its device copy is
    # kept across calls in the model's record (decomp_door_resident_input),
    # so recommend / similar_items do not upload the whole factors per call;
    # a frozen hit skips the re-admission. Same words, same routes (added as
    # a host matrix, as a borrowed input), so no bit change. Any other
    # storage (a caller-assigned ndarray) is admitted and uploaded per call.
    var hit = decomp_door_resident_hit(door, kit, source, name)
    if hit >= 0:
        return hit
    var array = decomp_input(source, name, context)
    var rows = Int(py=array.shape[0])
    var cols = Int(py=array.shape[1])
    var held = decomp_door_resident_input(door, kit, source, name, array, Int(py=classical_address(array)), array,
                                          rows, cols)
    if held >= 0:
        return held
    return decomp_door_borrow(door, kit, array, rows, cols)


def _typed(n: Int, typestr: String, code: String, item: Int, context: PythonObject) raises -> PythonObject:
    var output = context["output_context"]
    var store = py_call(output["array_type"], PythonObject(code), py_call(output["native_bytes_type"], n * item))
    return classical_wrap_array(store, Python.tuple(n), typestr, "C", output)


def _row_check(kit: DKit, a: Int, index: Int) raises:
    # `_M.rows(i, i + 1)`, whose store must hold one row.
    if index < 0 or index >= kit.nrows(a):
        raise Error("x_decomp: matrix store does not match its shape")


def als_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    # New factors replace the old: their device copies go now (N325).
    decomp_resident_drop(model)
    var mode = decomp_mode(fields["numeric_mode"], context)
    py_setitem(fields, "numeric_mode_", mode)
    # UNPACK: the image, user_items and every option, once.
    var entries = decomp_entries(mode, context)
    var door = DecompDoor(entries, context["output_context"])
    var kit = decomp_kit_unpack(entries)
    try:
        var r = decomp_door_input(door, kit, args[0], "user_items", context)
        var alpha = fields["alpha"]
        var unit = Bool(py=py_call(native_loaded_module("operator").eq, alpha, PythonObject(1.0)))
        var scale = 1.0 if unit else decomp_float(alpha)
        var seed = decomp_seed_word(decomp_seed(fields["random_state"]))
        var f = decomp_int(fields["factors"])
        var use_cg = Bool(py=fields["use_cg"])
        var cg_steps = decomp_int(fields["cg_steps"])
        if use_cg and cg_steps < 1:
            raise Error("AlternatingLeastSquares: cg_steps must be >= 1")
        var words = AlsFitWords(unit, scale, seed, f, use_cg, cg_steps, decomp_int(fields["iterations"]),
                                Bool(py=fields["calculate_training_loss"]), decomp_float(fields["regularization"]))
        # RUN: the draws and the one sweep call, the GIL released.
        var losses = List[Float64]()
        var user_m = -1
        var item_m = -1
        var py = Python()
        with GILReleased(py):
            var factors = als_fit_run(kit, r, words, losses)
            user_m = factors[0]
            item_m = factors[1]
        # PACK: the losses as one list, the factors as fresh Arrays, published once.
        if words.loss:
            py_setitem(fields, "training_loss_", py_list_f64(losses))
        var users = decomp_kit_publish(door, kit, user_m)
        var items = decomp_kit_publish(door, kit, item_m)
        py_setitem(fields, "user_factors", users)
        py_setitem(fields, "item_factors", items)
        py_setitem(fields, "user_factors_", users)
        py_setitem(fields, "item_factors_", items)
        return model
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


def _topn_pack(mut door: DecompDoor, mut kit: DKit, top: AlsTop, context: PythonObject) raises -> PythonObject:
    # PACK `_topn`: (ids int32, scores float32).
    var c = top.count
    if top.ids >= 0:
        var labels = _typed(c, "<i4", "i", 4, context)
        decomp_f32_to_i32(kit.ptr(top.ids), MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=labels._addr)), c)
        _ = labels  # keep the Python object alive through the native call above
        return Python.tuple(labels, decomp_kit_out_shape(door, kit, top.values, Python.tuple(c)))
    var labels = _typed(c, "<i4", "i", 4, context)
    var scores = _typed(c, "<f4", "f", 4, context)
    if c:
        unsafe_memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=labels._addr)),
               src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=kit.iaddr(top.order)), count=c)
        _ = labels  # keep the Python object alive through the native call above
        unsafe_memcpy(dest=DecompF32(unsafe_from_address=Int(py=scores._addr)), src=kit.ptr(top.values), count=c)
        _ = scores  # keep the Python object alive through the native call above
    return Python.tuple(labels, scores)


def als_query(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if not py_contains(fields, "user_factors"):
        raise Error("RuntimeError: AlternatingLeastSquares is not fitted; call fit first")
    var entries = decomp_entries(py_get(fields, "numeric_mode_"), context)
    var door = DecompDoor(entries, context["output_context"])
    # The fitted model of this query: its factors stay on the device (N325).
    door.owner = model
    var kit = decomp_kit_unpack(entries)
    try:
        var py = Python()
        if operation == "recommend":
            # args: (userid, user_items, N, filter_already_liked_items)
            var users = _fitted(door, kit, fields["user_factors"], "user_factors", context)
            var items = _fitted(door, kit, fields["item_factors"], "item_factors", context)
            var userid = decomp_int(args[0])
            _row_check(kit, users, userid)
            var count = decomp_int(args[2])
            if count < 0:
                raise Error("recommend: N must be non-negative, got " + String(py=py_call(native_loaded_module("builtins").str, args[2])))
            var r = -1
            var skip_row = 0
            if Bool(py=args[3]) and args[1] is not None:
                r = decomp_door_input(door, kit, args[1], "user_items", context)
                var rows = kit.nrows(r)
                var cols = kit.ncols(r)
                var width = kit.nrows(items)
                if cols != width or not (rows == 1 or (0 <= userid and userid < rows)):
                    raise Error("recommend: user_items of shape (" + String(rows) + ", " + String(cols) + ") has no row "
                                + String(userid) + " over the " + String(width) + " items")
                skip_row = 0 if rows == 1 else userid
            var top = AlsTop(0, -1, -1, -1)
            with GILReleased(py):
                top = als_recommend_run(kit, users, items, userid, count, r, skip_row)
            return _topn_pack(door, kit, top, context)
        if operation == "similar_items":
            # args: (itemid, N)
            var source = fields["item_factors"]
            # The unit-normalised item factors depend on item_factors alone:
            # computed by the first similar_items of a fitted state and kept on
            # the device for the next (decomp_door_resident_keep), the same
            # words the same steps would compute again (N325).
            var unit = decomp_door_resident_derived(door, kit, source, _ALS_UNIT)
            var y = -1
            if unit < 0:
                y = _fitted(door, kit, source, "item_factors", context)
            var itemid = decomp_int(args[0])
            _row_check(kit, unit if unit >= 0 else y, itemid)
            var count = decomp_int(args[1])
            if count < 0:
                raise Error("similar_items: N must be non-negative, got " + String(py=py_call(native_loaded_module("builtins").str, args[1])))
            var computed = unit < 0
            var top = AlsTop(0, -1, -1, -1)
            with GILReleased(py):
                if computed:
                    unit = als_unit_run(kit, y)
                top = als_similar_run(kit, unit, itemid, count)
            if computed:
                _ = decomp_door_resident_keep(door, kit, source, _ALS_UNIT, unit)
            return _topn_pack(door, kit, top, context)
        raise Error("unknown native AlternatingLeastSquares operation")
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


def als_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: als_native_call requires model, operation, arguments and connected context")
        var b = native_loaded_module("builtins")
        if py_type(call[1]) is not b.str or py_type(call[2]) is not b.tuple or py_type(call[3]) is not b.dict:
            raise Error("TypeError: native AlternatingLeastSquares requires an operation, an argument tuple and its context")
        var context = call[3]
        classical_connect_numpy(context)
        var operation = String(py=call[1])
        var model = call[0]
        var args = call[2]
        # user_items is the estimator's own interaction matrix (dense or
        # sparse), not a feature table: no X refusals or width checks
        var method = decomp_public_method(operation)
        var depth = decomp_public_begin(model, method, args, False, True, context)
        var result = PythonObject(None)
        try:
            if operation == "fit":
                result = als_fit(model, args, context)
            else:
                result = als_query(model, operation, args, context)
        finally:
            lifecycle_end(model, method, depth)
        result = decomp_public_finish(model, method, args, result, "", context)
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
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
