# SPDX-License-Identifier: Apache-2.0
"""`xcnn_native_call(layer, operation, args, context)`: the one native call
of every migrated public operation of the CNN layers
(python/mojolearn/_native_xcnn_bridge.py connects it at import; the
operations are in bindings/xcnn_pool_native.mojo)."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.xcnn_common import xcnn_numpy, xcnn_builtins, xcnn_dev_fields, xcnn_module_entries
from bindings.py_balanced import py_type, py_call, py_setitem
from bindings.py_typed_call import py_call_typed
from bindings.xcnn_tensor_native import xcnn_tensor_operation
from bindings.xcnn_pool_native import xcnn_pool_layers
from bindings.xcnn_bn_native import xcnn_bn_layers
from bindings.xcnn_conv_native import xcnn_conv_layers
from bindings.xcnn_graph_native import xcnn_gcn_forward, xcnn_sage_forward
from bindings.xcnn_classifier_native import cnnclf_fit, cnnclf_predict, cnnclf_predict_proba


def _xcnn_finalize(owner: PythonObject, operation: String, context: PythonObject) -> PythonObject:
    try:
        if operation == "tensor_finalize":
            return xcnn_tensor_operation(owner, operation, Python.tuple(), context)
        if operation == "dev_finalize":
            var fields = xcnn_dev_fields(owner, context)
            var entries = xcnn_module_entries(fields["b"], context)
            var table = fields["h"]
            var names = py_call(xcnn_builtins().list, table.keys())
            py_setitem(fields, "h", Python.dict())
            for i in range(len(names)):  # driver-loop: release the owner's native allocation handles
                _ = py_call_typed(entries["symbols"]["x_cnn_res_free"], table[names[i]][0])
    except:
        pass  # partially initialized owner or interpreter shutdown
    return PythonObject(None)


def xcnn_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        # Import-time native descriptors bind (context, operation); CPython
        # decref then enters this C finalizer directly, without a Python body.
        if len(call) == 2:
            var bound = call[0]
            if py_type(bound) is not py_type(Python.tuple()) or len(bound) != 2:
                raise Error("TypeError: CNN finalizer requires its native bound context")
            return _xcnn_finalize(call[1], String(bound[1]), bound[0]).steal_data()
        var context = call[3]
        var operation = String(py=call[1])
        if operation != "tensor_finalize":
            xcnn_numpy(context)
        if operation.startswith("tensor_"):
            return xcnn_tensor_operation(call[0], operation, call[2], context).steal_data()
        if operation.startswith("conv_") or operation.startswith("block_"):
            return xcnn_conv_layers(call[0], operation, call[2], context).steal_data()
        if operation == "cnnclf_fit":
            return cnnclf_fit(call[0], call[2], context).steal_data()
        if operation == "cnnclf_predict":
            return cnnclf_predict(call[0], call[2], context).steal_data()
        if operation == "cnnclf_predict_proba":
            return cnnclf_predict_proba(call[0], call[2], context).steal_data()
        if operation == "gcn_forward":
            return xcnn_gcn_forward(call[0], call[2], context).steal_data()
        if operation == "sage_forward":
            return xcnn_sage_forward(call[0], call[2], context).steal_data()
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
