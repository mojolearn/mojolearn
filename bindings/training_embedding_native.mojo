# SPDX-License-Identifier: Apache-2.0
# Public embedding input preparation and ownership; numerical entries unchanged.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get
from bindings.optimizer_admission import opt_value, opt_numpy
from bindings.optimizer_public_words import stash_python_error
from bindings.optimizer_public_door import TensorView, tensor_view, new_array, contiguous_addr
from bindings.optimizer_device_native import shape_count
from bindings.training_rms_native import rms_host_view
from bindings.training_scalar_native import training_scalar_number
from bindings.native_numpy_metadata import native_numpy_type
from bindings.sequence_dtype_native import sequence_dtype_classes, sequence_dtype_instance, sequence_dtype_create
from gemm.f16_widen import f16_bits_to_f32
from core.training_embedding_inputs import embedding_pack, embedding_cast_refusal


def embedding_native_call(entry: PythonObject, args: PythonObject, list_store: Bool = False) raises -> PythonObject:
    # Call only an admitted native constructor/slot, preserving its exception.
    ref cpy = Python().cpython()
    var result = cpy.PyObject_Call(entry._obj_ptr, args._obj_ptr, PyObjectPtr())
    if not result:
        if list_store:
            # Array._from_flat wrapped native numeric storage construction
            # failures in TypeError. The admitted inputs have no hooks.
            cpy.lib.get_function[NoneType]("PyErr_Clear")()
            raise Error("TypeError: cannot build an index Array from these native values")
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def embedding_array_leaves(value: PythonObject, mut dims: List[Int], depth: Int,
                           mut leaf_depth: Int, leaves: PythonObject) raises:
    var value_owner = value
    var view = TensorView()
    _ = tensor_view(value, view)
    if leaf_depth >= 0 and leaf_depth != depth + len(view.shape):
        raise Error("ValueError: embedding indices must be rectangular")
    leaf_depth = depth + len(view.shape)
    for j in range(len(view.shape)):  # driver-loop: nested Array shape metadata
        if depth + j == len(dims):
            dims.append(view.shape[j])
        elif dims[depth + j] != view.shape[j]:
            raise Error("ValueError: embedding indices must be rectangular")
    if len(dims) != leaf_depth:
        raise Error("ValueError: embedding indices must be rectangular")
    var count = shape_count(view.shape)
    var packed = embedding_pack(view.addr, view.shape, view.strides, view.itemsize, False, 0, count)
    var pointer = packed.unsafe_ptr()
    var dtype = String(value.dtype)
    for i in range(count):  # driver-loop: serialize nested native Array leaves into the admitted index envelope
        var scalar = PythonObject(None)
        if dtype == "<f4":
            scalar = PythonObject(Float64(pointer.bitcast[Float32]()[i]))
        elif dtype == "<f8":
            scalar = PythonObject(pointer.bitcast[Float64]()[i])
        elif dtype == "<f2":
            scalar = PythonObject(Float64(f16_bits_to_f32(pointer.bitcast[UInt16]()[i])))
        elif dtype == "<i8":
            scalar = PythonObject(Int(pointer.bitcast[Int64]()[i]))
        elif dtype == "<i4":
            scalar = PythonObject(Int(pointer.bitcast[Int32]()[i]))
        elif dtype == "<u4":
            scalar = PythonObject(Int(pointer.bitcast[UInt32]()[i]))
        elif dtype == "<u2":
            scalar = PythonObject(Int(pointer.bitcast[UInt16]()[i]))
        elif dtype == "<i1":
            scalar = PythonObject(Int(pointer.bitcast[Int8]()[i]))
        elif dtype == "<u1":
            scalar = PythonObject(Int(pointer[i]))
        else:
            raise Error("TypeError: nested index Array has an unsupported dtype")
        py_call(leaves.append, scalar)
    _ = packed^
    _ = value_owner^


def embedding_leaves(value: PythonObject, mut dims: List[Int], depth: Int,
                     mut leaf_depth: Int, leaves: PythonObject, context: PythonObject) raises:
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    if cls is context["array_class"]:
        embedding_array_leaves(value, dims, depth, leaf_depth, leaves)
        return
    var native_array = False
    if opt_numpy(cls, context):
        # Only exact ndarray has a nonempty native shape; NumPy scalar shape
        # is empty and its scalar class is retained as a leaf below.
        native_array = cls is native_numpy_type("ndarray", context)
    if cls is b.list or cls is b.tuple or native_array:
        if native_array and Int(py=value.ndim) == 0:
            raise Error("TypeError: iteration over a 0-d array")
        if native_array:
            embedding_numpy_dtype(value, context)
        var size = len(value)
        if depth == len(dims):
            dims.append(size)
        elif dims[depth] != size:
            raise Error("ValueError: embedding indices must be rectangular")
        if leaf_depth >= 0 and depth >= leaf_depth:
            raise Error("ValueError: embedding indices must be rectangular")
        for i in range(size):  # driver-loop: native traversal of the caller's index container
            # Exact native ndarray basic integer indexing returns only an
            # owned native view/scalar. Each next leaf is admitted before use.
            embedding_leaves(value[i], dims, depth + 1, leaf_depth, leaves, context)
        return
    if leaf_depth >= 0 and leaf_depth != depth:
        raise Error("ValueError: embedding indices must be rectangular")
    leaf_depth = depth
    if len(dims) != depth:
        raise Error("ValueError: embedding indices must be rectangular")
    opt_value(value, context)
    py_call(leaves.append, value)


def embedding_numpy_dtype(value: PythonObject, context: PythonObject) raises:
    var dtype = value.dtype
    var admitted = sequence_dtype_instance(dtype, sequence_dtype_classes(context))
    for code in ["q", "Q"]:  # driver-loop: native long-long descriptor aliases
        admitted = admitted or py_type(dtype) is py_type(sequence_dtype_create(PythonObject(code), context))
    if not admitted:
        raise Error("TypeError: embedding indices require a built-in native dtype")
    var kind = String(dtype.kind)
    if kind == "c":
        raise Error("ValueError: Complex data not supported for embedding indices")
    if kind != "b" and kind != "i" and kind != "u" and kind != "f" and kind != "O":
        raise Error("TypeError: embedding indices require a supported real numeric buffer")


def embedding_leaf_store(leaves: PythonObject, object_array: Bool, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var floating = False
    for i in range(len(leaves)):  # driver-loop: validate every input scalar before any native conversion slot
        var value = leaves[i]
        var cls = py_type(value)
        var builtin = cls is b.bool or cls is b.int or cls is b.float
        var numpy = opt_numpy(cls, context)
        if not builtin and not numpy:
            raise Error("TypeError: embedding indices require native real numbers without Python conversion hooks")
        if numpy:
            if cls is native_numpy_type("ndarray", context):
                # An ndarray stored inside an object array is not a real
                # scalar. Never invoke its conversion protocol as a leaf.
                raise Error("TypeError: embedding object indices require real scalar values")
            var kind = String(value.dtype.kind)
            if kind != "b" and kind != "i" and kind != "u" and kind != "f":
                raise Error("TypeError: embedding indices require real numbers")
        # The previous list door inferred f64 only from Python float or its
        # native float64 subtype. Object buffers with any NumPy real scalar
        # used f64; retain that distinct promotion before the int32 cast.
        floating = floating or cls is b.float or (numpy and (object_array or cls is sequence_dtype_create(PythonObject("f8"), context).type))
    if not floating:
        for i in range(len(leaves)):  # driver-loop: preserve the list door's integer-leaf admission
            var cls = py_type(leaves[i])
            if cls is not b.int and cls is not b.bool:
                raise Error("TypeError: embedding index lists require native Python integers or a floating-point leaf")
    var constructor = context["array_type"]
    if constructor is not context["buffer_types"][2]:
        raise Error("TypeError: embedding index storage requires the connected native array constructor")
    var store = embedding_native_call(constructor, Python.tuple("d" if floating else "q", leaves), True)
    return Python.tuple(store, 1 if floating else 3)


def embedding_indices(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var value_owner = value
    var numpy = py_get(context["modules"], "numpy")
    if numpy is not None and py_type(numpy) is context["module_type"]:
        var classes = sequence_dtype_classes(context)
        for i in range(len(classes)):  # driver-loop: preserve native complex-scalar refusal before general admission
            if py_type(value) is classes[i][1] and String(value.dtype.kind) == "c":
                raise Error("ValueError: Complex data not supported for embedding indices")
    opt_value(value, context)
    var b = Python.import_module("builtins")
    if opt_numpy(py_type(value), context):
        embedding_numpy_dtype(value, context)
    var view = TensorView()
    var storage = PythonObject(None)
    var source_code = -1
    var bytes = List[UInt8]()
    var count = 0
    var source_addr = 0
    if tensor_view(value, view):
        count = shape_count(view.shape)
        if count == 0:
            raise Error("ValueError: embedding indices must be nonempty")
        var format = view.fmt
        var swapped = format.startswith(">") or format.startswith("!")
        if format.byte_length() > 1 and (swapped or format.startswith("@") or format.startswith("=") or format.startswith("<")):
            var normalized = String(format[byte=1:])
            format = normalized^
        if format == "O":
            # Exact native object ndarray + admitted builtin dtype. Public
            # scalar item access INCREFs one stored object, with no numeric
            # conversion; admit every resulting leaf before calling a slot.
            var leaves = Python.list()
            ref cpy = Python().cpython()
            for i in range(count):  # driver-loop: native index-container item unboxing in logical C order
                var remaining = i
                var words = List[Int](length=len(view.shape), fill=0)
                for j in range(len(view.shape) - 1, -1, -1):  # driver-loop: coordinates of one scalar item
                    words[j] = remaining % view.shape[j]
                    remaining //= view.shape[j]
                var indexes = Python.list()
                for j in range(len(words)):  # driver-loop: assemble the native scalar index tuple
                    py_call(indexes.append, words[j])
                var index = py_call(b.tuple, indexes)
                var item = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(value._obj_ptr, index._obj_ptr)
                _ = index^
                if not item:
                    var failure = stash_python_error()
                    raise failure^
                var scalar = PythonObject(from_owned=item)
                py_call(leaves.append, scalar)
            var packed = embedding_leaf_store(leaves, True, context)
            storage = packed[0]
            source_code = Int(py=packed[1])
            source_addr = Int(py=py_call(storage.buffer_info)[0])
            _ = leaves^
        else:
            if format.startswith("Z"):
                raise Error("ValueError: Complex data not supported for embedding indices")
            if format == "f" and view.itemsize == 4:
                source_code = 0
            elif format == "d" and view.itemsize == 8:
                source_code = 1
            elif (format == "i" or format == "l" or format == "q" or format == "n") and view.itemsize == 4:
                source_code = 2
            elif (format == "i" or format == "l" or format == "q" or format == "n") and view.itemsize == 8:
                source_code = 3
            elif (format == "I" or format == "L" or format == "Q" or format == "N") and view.itemsize == 4:
                source_code = 4
            elif (format == "?" or format == "B") and view.itemsize == 1:
                source_code = 5
            elif format == "H" and view.itemsize == 2:
                source_code = 6
            elif format == "b" and view.itemsize == 1:
                source_code = 7
            elif format == "e" and view.itemsize == 2:
                source_code = 8
            elif format == "h" and view.itemsize == 2:
                source_code = 9
            elif (format == "L" or format == "Q" or format == "N") and view.itemsize == 8:
                source_code = 10
            else:
                raise Error("TypeError: embedding indices require a supported real numeric buffer")
            if source_code == 2 and view.c_contig and not swapped:
                # Existing contiguous int32 indices are already the kernel
                # contract. Keep their owner alive and pass the same bytes.
                return Python.tuple(value, view.addr, count)
            bytes = embedding_pack(view.addr, view.shape, view.strides, view.itemsize, swapped, source_code, count)
            source_code = 2 if source_code == 9 else (3 if source_code == 10 else source_code)
            source_addr = Int(bytes.unsafe_ptr())
    else:
        var dims = List[Int]()
        var depth = -1
        var leaves = Python.list()
        embedding_leaves(value, dims, 0, depth, leaves, context)
        count = len(leaves)
        if count == 0:
            raise Error("ValueError: embedding indices must be nonempty")
        var packed = embedding_leaf_store(leaves, False, context)
        storage = packed[0]
        source_code = Int(py=packed[1])
        source_addr = Int(py=py_call(storage.buffer_info)[0])
        _ = leaves^
    var result = new_array([count], "<i4")
    var entry = context["embedding_index_cast"]
    if py_type(entry) is not context["device_finalizer_types"][2]:
        raise Error("TypeError: embedding requires its connected compiled index cast")
    if Int(py=py_call(entry, source_addr, source_code, result._addr, 2, count)) != 0:
        embedding_cast_refusal(source_addr, source_code, count)
    _ = bytes^
    _ = storage^
    _ = value_owner^
    return Python.tuple(result, result._addr, count)


def training_embedding_operation(operation: String, values: PythonObject, entries: PythonObject,
                                 context: PythonObject) raises -> PythonObject:
    var backward = operation == "embedding_backward"
    if len(values) != (3 if backward else 2):
        raise Error("TypeError: invalid embedding argument envelope")
    var entry = entries[PythonObject(operation)]
    if py_type(entry) is not context["device_finalizer_types"][2]:
        raise Error("TypeError: embedding requires its connected compiled entry")
    var input = values[0]
    var view = TensorView()
    rms_host_view(input, "dy" if backward else "weight", operation, view)
    if len(view.shape) != 2:
        raise Error("ValueError: mojolearn." + operation + ": " + ("dy must be (N, D)" if backward else "weight must be (V, D)"))
    var ids = embedding_indices(values[1], context)
    var count = Int(py=ids[2])
    if backward and count != view.shape[0]:
        raise Error("ValueError: mojolearn.embedding_backward: ids and dy row counts differ")
    var vocab = view.shape[0]
    if backward:
        vocab = Int(py=training_scalar_number(values[2], operation, context, True))
        if vocab < 0:
            raise Error("ValueError: negative embedding vocabulary size")
    _ = shape_count([vocab, view.shape[1]])
    var result = new_array([vocab if backward else count, view.shape[1]], "<f4")
    var keep = List[List[Float32]]()
    var address = contiguous_addr(input, view, keep)
    if backward:
        # The public contract converted vocab for allocation and again for
        # launch metadata; preserve native scalar warnings from both calls.
        vocab = Int(py=training_scalar_number(values[2], operation, context, True))
    _ = py_call(entry, Python.list(result._addr, address, ids[1]), Python.list(count, vocab, view.shape[1]))
    _ = keep^
    _ = input^
    _ = ids^
    return result
