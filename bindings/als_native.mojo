# SPDX-License-Identifier: Apache-2.0
# The native owner of python/mojolearn/_expansion_decomp.py's
# AlternatingLeastSquares: fit, recommend and similar_items are each one call
# into als_c. The owner runs the steps the Python shell ran (the confidence
# scale, the Philox factor draws, the one `x_decomp_als_fit` sweep call, the
# score row and the top-N order of `_topn`) through
# bindings/decomp_kit_native.mojo: the same x_decomp entries with the same
# words in the same order, so the bits are unchanged.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.memory import memcpy
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_address, classical_typed, classical_wrap_array
from bindings.decomp_kit_native import (
    DecompKit, DecompF32, decomp_float, decomp_int, decomp_seed, decomp_seed_word, decomp_mode, decomp_entries,
    decomp_input, XD_SCALE, XD_SQ, XD_SQRT, XD_DIV,
)
from core.decomp_native_impl import decomp_f32_to_i32


def _matrix(mut kit: DecompKit, source: PythonObject, name: String, context: PythonObject) raises -> Int:
    var array = decomp_input(source, name, context)
    return kit.input(array, Int(py=array.shape[0]), Int(py=array.shape[1]))


def _typed(n: Int, typestr: String, code: String, item: Int, context: PythonObject) raises -> PythonObject:
    var output = context["output_context"]
    var store = py_call(output["array_type"], PythonObject(code), py_call(output["native_bytes_type"], n * item))
    return classical_wrap_array(store, Python.tuple(n), typestr, "C", output)


def als_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var mode = decomp_mode(fields["numeric_mode"], context)
    py_setitem(fields, "numeric_mode_", mode)
    var kit = DecompKit(decomp_entries(mode, context), context["output_context"])
    try:
        var r = _matrix(kit, args[0], "user_items", context)
        var n = kit.nrows(r)
        var m = kit.ncols(r)
        var alpha = fields["alpha"]
        var unit = Bool(py=py_call(Python.import_module("operator").eq, alpha, PythonObject(1.0)))
        var c = r if unit else kit.ew(XD_SCALE, r, scalar=decomp_float(alpha))
        var seed = decomp_seed_word(decomp_seed(fields["random_state"]))
        var f = decomp_int(fields["factors"])
        var user_draws = kit.rand(n, f, seed, 80, 0)
        var x = kit.ew(XD_SCALE, user_draws, scalar=0.01)
        var item_draws = kit.rand(m, f, seed, 81, 0)
        var y = kit.ew(XD_SCALE, item_draws, scalar=0.01)
        var use_cg = Bool(py=fields["use_cg"])
        if use_cg and decomp_int(fields["cg_steps"]) < 1:
            raise Error("AlternatingLeastSquares: cg_steps must be >= 1")
        # The sweeps (and the losses) are ONE binding call on host copies of
        # the factors, written in place.
        x = kit.copy_host(x)
        y = kit.copy_host(y)
        var iterations = decomp_int(fields["iterations"])
        var slots = max(iterations, 1)
        var losses_store = classical_typed(Python.tuple(slots), "<f8", context["output_context"])
        var loss = Bool(py=fields["calculate_training_loss"])
        var params = Python.list(n, m, f, iterations, Int(use_cg), decomp_int(fields["cg_steps"]), Int(loss))
        var ac = kit.address(c)
        var ax = kit.address(x)
        var ay = kit.address(y)
        var count = decomp_int(py_call(kit.entries["x_decomp_als_fit"], ac, ax, ay, classical_address(losses_store),
                                       params, Python.list(decomp_float(fields["regularization"]))))
        kit.wrote_host(x)
        kit.wrote_host(y)
        if loss:
            var losses = Python.list()
            var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(losses_store)))
            for i in range(min(count, slots)):  # driver-loop: over the fit's recorded iterations
                _ = py_call(losses.append, PythonObject(words[i]))
            py_setitem(fields, "training_loss_", losses)
        var users = kit.publish(x)
        var items = kit.publish(y)
        py_setitem(fields, "user_factors", users)
        py_setitem(fields, "item_factors", items)
        py_setitem(fields, "user_factors_", users)
        py_setitem(fields, "item_factors_", items)
        return model
    finally:
        kit.close()


def _topn(mut kit: DecompKit, s: Int, count: Int, skip: Int, context: PythonObject) raises -> PythonObject:
    # `AlternatingLeastSquares._topn`: (ids int32, scores float32) of the N
    # best entries of the score row by (-score, id), the skip row's nonzero
    # entries left out (skip: a 1 x m host matrix, or -1).
    var m = kit.size(s)
    var n = max(min(count, m), 0)
    if n and kit.use(s):
        var order = kit.order_dev(s, 1, skip)
        var c = min(n, Int(kit.word(order[1])))
        if c == 0:
            return Python.tuple(_typed(0, "<i4", "i", 4, context), _typed(0, "<f4", "f", 4, context))
        var ids = kit.rows_of(order[0], 0, c)
        var row = kit.reshape(s, 1, m)
        var values = kit.take_cols_m(row, ids, c)
        var labels = _typed(c, "<i4", "i", 4, context)
        var source = kit.host(ids)
        decomp_f32_to_i32(DecompF32(unsafe_from_address=Int(py=classical_address(source))),
                          MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=labels._addr)), c)
        _ = source^
        return Python.tuple(labels, kit.out_shape(values, Python.tuple(c)))
    var order = py_call(context["output_context"]["array_type"], PythonObject("i"),
                        py_call(context["output_context"]["native_bytes_type"], 4 * max(n, 1)))
    var order_address = Int(py=py_call(order.buffer_info)[0])
    var skip_address = PythonObject(0) if skip < 0 else kit.address(skip)
    var c = 0
    if n:
        var sa = kit.address(s)
        c = decomp_int(py_call(kit.entries["x_decomp_topn_desc"], sa, Python.list(m, n), skip_address, order_address))
    var values = kit.zeros(max(c, 1), 1)
    if c:
        var sa = kit.address(s)
        _ = py_call(kit.entries["x_decomp_gather"], sa, order_address, c, kit.address(values))
    var labels = _typed(c, "<i4", "i", 4, context)
    var scores = _typed(c, "<f4", "f", 4, context)
    if c:
        memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=labels._addr)),
               src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=order_address), count=c)
        memcpy(dest=DecompF32(unsafe_from_address=Int(py=scores._addr)),
               src=DecompF32(unsafe_from_address=Int(py=kit.address(values))), count=c)
    _ = order^
    return Python.tuple(labels, scores)


def _one_row(mut kit: DecompKit, a: Int, index: Int) raises -> Int:
    # `_M.rows(i, i + 1)`, whose store must hold one row.
    if index < 0 or index >= kit.nrows(a):
        raise Error("x_decomp: matrix store does not match its shape")
    return kit.rows_of(a, index, index + 1)


def als_query(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if not py_contains(fields, "user_factors"):
        raise Error("RuntimeError: AlternatingLeastSquares is not fitted; call fit first")
    var kit = DecompKit(decomp_entries(py_get(fields, "numeric_mode_"), context), context["output_context"])
    try:
        if operation == "recommend":
            # args: (userid, user_items, N, filter_already_liked_items)
            var users = _matrix(kit, fields["user_factors"], "user_factors", context)
            var items = _matrix(kit, fields["item_factors"], "item_factors", context)
            var userid = decomp_int(args[0])
            var u = _one_row(kit, users, userid)
            var s = kit.mm(u, items, tb=True)
            var count = decomp_int(args[2])
            if count < 0:
                raise Error("recommend: N must be non-negative, got " + String(py=py_call(Python.import_module("builtins").str, args[2])))
            var skip = -1
            if Bool(py=args[3]) and args[1] is not None:
                var r = _matrix(kit, args[1], "user_items", context)
                var rows = kit.nrows(r)
                var cols = kit.ncols(r)
                if cols != kit.ncols(s) or not (rows == 1 or (0 <= userid and userid < rows)):
                    raise Error("recommend: user_items of shape (" + String(rows) + ", " + String(cols) + ") has no row "
                                + String(userid) + " over the " + String(kit.ncols(s)) + " items")
                skip = kit.rows_of(r, 0 if rows == 1 else userid, (0 if rows == 1 else userid) + 1)
            return _topn(kit, s, count, skip, context)
        if operation == "similar_items":
            # args: (itemid, N)
            var y = _matrix(kit, fields["item_factors"], "item_factors", context)
            var squares = kit.ew(XD_SQ, y)
            var sums = kit.rowsum(squares)
            var norms = kit.ew(XD_SQRT, sums)
            var unit = kit.ew(XD_DIV, y, norms)
            var row = _one_row(kit, unit, decomp_int(args[0]))
            var s = kit.mm(row, unit, tb=True)
            var count = decomp_int(args[1])
            if count < 0:
                raise Error("similar_items: N must be non-negative, got " + String(py=py_call(Python.import_module("builtins").str, args[1])))
            return _topn(kit, s, count, -1, context)
        raise Error("unknown native AlternatingLeastSquares operation")
    finally:
        kit.close()


def als_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: als_native_call requires model, operation, arguments and connected context")
        var b = Python.import_module("builtins")
        if py_type(call[1]) is not b.str or py_type(call[2]) is not b.tuple or py_type(call[3]) is not b.dict:
            raise Error("TypeError: native AlternatingLeastSquares requires an operation, an argument tuple and its context")
        var context = call[3]
        classical_connect_numpy(context)
        var operation = String(py=call[1])
        var result = PythonObject(None)
        if operation == "fit":
            result = als_fit(call[0], call[2], context)
        else:
            result = als_query(call[0], operation, call[2], context)
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
