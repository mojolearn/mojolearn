# SPDX-License-Identifier: Apache-2.0
# Provider-boundary sequence preparation only; native MLP kernels do not import
# this module. Public callers have already run sequential_admit_value.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem
from bindings.native_loaded_module import native_loaded_module, native_namespace_string_keys
from bindings.native_array_snapshot import native_array_snapshot, native_array_slots
from bindings.native_torch_admission import native_torch_exact, native_torch_admit, NATIVE_TORCH, _torch_type_fields
from bindings.array_native import _as_c, _values, _row, _ints, _tuple, _prod, _both_orders

comptime SMALL_MLP_ITEMS_PENDING = "small-mlp-items: native iterator exception pending"


def _plain_items(value: PythonObject, limit: Int, tuple_input: Bool) raises -> List[PythonObject]:
    var count = len(value)
    if limit >= 0:
        count = min(count, limit)
    var result = List[PythonObject](capacity=count)
    ref cpy = Python().cpython()
    var get_item = cpy.lib.get_function[PyObjectPtr]("PyTuple_GetItem" if tuple_input else "PyList_GetItem")
    for i in range(count):  # driver-loop: admitted exact list/tuple items, no iterator callbacks
        var raw = get_item(value._obj_ptr, i)
        if not raw:
            raise Error(SMALL_MLP_ITEMS_PENDING)
        result.append(PythonObject(from_owned=cpy.Py_NewRef(raw)))
    _ = value  # Pin borrowed item owners through every NewRef.
    return result^


def _string_dict(value: PythonObject, bi: PythonObject) raises:
    if py_type(value) is not bi.dict or not native_namespace_string_keys(value._obj_ptr, bi.str):
        raise Error("TypeError: SmallMLP Array connection requires native string-keyed dictionaries")


def _array_items(value: PythonObject, limit: Int, refusal: String,
                 array_cls: PythonObject, array_type: PythonObject, bi: PythonObject) raises -> List[PythonObject]:
    var context = Python.dict(array_class=array_cls)
    var captured = PythonObject(None)
    var shape = PythonObject(None)
    var count = 0
    var step = 0
    # Match iterator construction's existing refusal boundary; iteration errors
    # below this setup are no longer swallowed. Use admitted passive metadata,
    # not Array.__iter__ or a Python map/partial row callback.
    try:
        captured = native_array_snapshot(value, context, True)
        var ndim = Int(py=captured.ndim)
        if ndim == 0:
            raise Error("TypeError: iteration over a 0-d Array")
        var dtype = String(captured.dtype)
        if ndim == 1:
            if dtype == "<f2":
                var unpack = native_loaded_module("_struct").unpack
                if py_type(unpack) is not py_type(bi.len):
                    raise Error("TypeError: SmallMLP half values require native struct unpack")
                py_setitem(context, "struct_unpack", unpack)
            var values = _values(captured._mv, dtype, context)
            return _plain_items(values, limit, False)
        var codes = Python.dict()
        var sizes = Python.dict()
        py_setitem(codes, dtype, captured._mv.format)
        py_setitem(sizes, dtype, captured.itemsize)
        py_setitem(context, "codes", codes)
        py_setitem(context, "itemsizes", sizes)
        py_setitem(context, "array_type", array_type)
        var dims = _ints(captured.shape)
        if String(captured.order) != "C" and not _both_orders(dims) and Int(py=captured.size) != 0:
            var fields = py_fields(native_loaded_module("mojolearn._array"))
            var connected = py_get(fields, "_ARRAY_CONTEXT")
            _string_dict(connected, bi)
            var helpers = py_get(connected, "helpers")
            _string_dict(helpers, bi)
            var copy = py_get(helpers, "strided_copy_bytes")
            if copy is not None and py_type(copy) is not py_type(bi.len):
                raise Error("TypeError: SmallMLP Array reorder requires its native helper")
            py_setitem(context, "helpers", Python.dict(strided_copy_bytes=copy))
        else:
            py_setitem(context, "helpers", Python.dict())
        captured = _as_c(captured, context)
        # The reorder helper may release the GIL. Re-admit class slots before
        # constructing rows; values and output memory remain owned by captured.
        _ = native_array_slots(context)
        dims = _ints(captured.shape)
        var rest = List[Int](capacity=len(dims) - 1)
        for axis in range(1, len(dims)):  # small-loop(trailing native Array rank)
            rest.append(dims[axis])
        shape = _tuple(rest)
        step = _prod(rest)
        count = dims[0]
    except:
        ref cpy = Python().cpython()
        if cpy.PyErr_Occurred():
            _ = cpy.unsafe_get_error()
        raise Error("TypeError: " + refusal)
    if limit >= 0:
        count = min(count, limit)
    var result = List[PythonObject](capacity=count)
    for i in range(count):  # driver-loop: native row views retain captured storage, never copy scalar words
        try:
            result.append(_row(captured, shape, step, i, context))
        except error:
            if Python().cpython().PyErr_Occurred():
                raise Error(SMALL_MLP_ITEMS_PENDING)
            raise error
    return result^


def small_mlp_items(value: PythonObject, limit: Int, refusal: String,
                    array_cls: PythonObject, array_type: PythonObject, bi: PythonObject) raises -> List[PythonObject]:
    var cls = py_type(value)
    if cls is bi.list or cls is bi.tuple:
        return _plain_items(value, limit, cls is bi.tuple)
    if cls is array_cls:
        return _array_items(value, limit, refusal, array_cls, array_type, bi)
    if native_torch_exact(cls):
        # Type/mode refusal belongs to admission; do not relabel failures from
        # an already admitted tensor operation as a non-iterable input.
        native_torch_admit(value)
        var base_fields = _torch_type_fields(NATIVE_TORCH.get_or_create_ptr()[].base)
        var unbind = py_get(base_fields, "unbind")
        if unbind is None or py_type(unbind) is not py_type(bi.list.append):
            raise Error("TypeError: SmallMLP Tensor iteration requires native unbind")
        # Tensor.__iter__ returns iter(self.unbind(0)). Keep eager unbind before
        # limit, tensor views/dtype/device, and the native exception class and
        # traceback. py_call consumes the indicator into a Mojo Error, so this
        # call hands an untouched error to small_mlp_c's existing pending path.
        var arguments = Python.tuple(value, 0)
        ref cpy = Python().cpython()
        var raw_rows = cpy.PyObject_Call(unbind._obj_ptr, arguments._obj_ptr, PyObjectPtr())
        _ = arguments
        _ = unbind
        _ = value
        if not raw_rows:
            raise Error(SMALL_MLP_ITEMS_PENDING)
        var rows = PythonObject(from_owned=raw_rows)
        if py_type(rows) is not bi.tuple:
            raise Error("TypeError: native Tensor unbind must return a tuple")
        return _plain_items(rows, limit, True)
    ref cpy = Python().cpython()
    var raw_iterator = cpy.lib.get_function[PyObjectPtr]("PyObject_GetIter")(value._obj_ptr)
    if not raw_iterator:
        _ = cpy.unsafe_get_error()
        raise Error("TypeError: " + refusal)
    var iterator = PythonObject(from_owned=raw_iterator)
    var next_item = cpy.lib.get_function[PyObjectPtr]("PyIter_Next")
    var result = List[PythonObject]()
    while limit < 0 or len(result) < limit:  # driver-loop: remaining admitted native iterators
        var raw = next_item(iterator._obj_ptr)
        if not raw:
            if cpy.PyErr_Occurred():
                # PyIter_Next clears StopIteration only. Preserve every other
                # exception at the public C entry, including memoryview errors.
                raise Error(SMALL_MLP_ITEMS_PENDING)
            break
        result.append(PythonObject(from_owned=raw))
    _ = iterator
    _ = value
    return result^
