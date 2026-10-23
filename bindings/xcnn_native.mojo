# SPDX-License-Identifier: Apache-2.0
"""`xcnn_native_call(layer, operation, args, context)`: the one native call
of every migrated public operation of the CNN layers
(python/mojolearn/_native_xcnn_bridge.py connects it at import; the
operations are in bindings/xcnn_pool_native.mojo)."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.xcnn_common import xcnn_numpy
from bindings.xcnn_pool_native import xcnn_pool_layers
from bindings.xcnn_bn_native import xcnn_bn_layers
from bindings.xcnn_conv_native import xcnn_conv_layers


def xcnn_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var context = call[3]
        xcnn_numpy(context)
        var operation = String(py=call[1])
        if operation.startswith("conv_forward") or operation.startswith("conv_transform") or operation.startswith("block_"):
            return xcnn_conv_layers(call[0], operation, call[2], context).steal_data()
        if operation.startswith("bn_"):
            return xcnn_bn_layers(call[0], operation, call[2], context).steal_data()
        return xcnn_pool_layers(call[0], operation, call[2], context).steal_data()
    except error:
        var text = String(error)
        if text.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if text.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if text.startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if text.startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if text.startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if text.startswith("IndexError:"):
            return raise_python_exception(error_without_class(error, "IndexError"), ExceptionType("PyExc_IndexError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
