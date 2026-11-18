# SPDX-License-Identifier: Apache-2.0
"""`xcnn_native_call(layer, operation, args, context)`: the one native call
of every migrated public operation of the CNN layers
(python/mojolearn/_native_xcnn_bridge.py connects it at import; the
operations are in bindings/xcnn_pool_native.mojo)."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.xcnn_common_door import xcnn_connect_numpy, xcnn_builtins, xcnn_dev_fields, xcnn_module_entries, xcnn_ref, xcnn_word
from bindings.xcnn_common_door import xcnn_at, xcnn_exact, XT_TUPLE
from std.python._cpython import GILReleased
from x_cnn.tensor_run import xcnn_tensor_run_free
from bindings.native_loaded_module_door import native_interpreter_finalizing
from bindings.py_balanced import py_type, py_call, py_setitem
from bindings.xcnn_tensor_native_pack import xcnn_tensor_operation
from bindings.xcnn_pool_native_door import xcnn_pool_layers
from std.sys.compile import is_defined
from bindings.xcnn_pool_typed_door import xcnn_pool_typed_call, xcnn_pool_typed_learn
from bindings.xcnn_bn_native_door import xcnn_bn_layers
from bindings.xcnn_conv_native_door import xcnn_conv_layers
from bindings.xcnn_graph_native_door import xcnn_graph_layers
from bindings.xcnn_classifier_native_door import cnnclf_fit, cnnclf_predict, cnnclf_predict_proba, cnnclf_initialize, cnnclf_weights


def _xcnn_finalize(owner: PythonObject, operation: String, context: PythonObject) -> PythonObject:
    try:
        # A DeviceTensor or resident-handle owner collected during interpreter
        # teardown (a tensor held in `__main__` or a closure at exit) releases
        # nothing: the process is ending, and xcnn_builtins() and the layout
        # check read loaded modules, a fatal error once the interpreter's
        # modules are cleared, which `except` cannot catch. The 0.8.66 AMD
        # measurement smoke's own-host reference child (MOJOLEARN_VENDOR=cpu)
        # exited nonzero with "<no Python frame>" (a fatal error with no
        # Python frame on the main thread: teardown) on every CNN lane whose
        # layer takes DeviceTensors (conv,
        # pools, batch norm, GCN, SAGE, BasicBlock), and those 13 races were
        # refused. Same rule as host_archive_lifetime.mojo and
        # embedding_layer_native.mojo.
        if native_interpreter_finalizing():
            return PythonObject(None)
        if operation == "tensor_finalize":
            return xcnn_tensor_operation(owner, operation, Python.tuple(), context)
        if operation == "dev_finalize":
            var fields = xcnn_dev_fields(owner, context)
            var entries = xcnn_module_entries(fields["b"], context)
            var key = xcnn_ref(entries)
            var table = fields["h"]
            var names = py_call(xcnn_builtins().list, table.keys())
            py_setitem(fields, "h", Python.dict())
            for i in range(len(names)):  # driver-loop: release the owner's native allocation handles
                var word = xcnn_word(table[names[i]][0])
                var py = Python()
                with GILReleased(py):
                    xcnn_tensor_run_free(key, word)
    except:
        pass  # partially initialized owner or interpreter shutdown
    return PythonObject(None)


def xcnn_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        # MOJOLEARN_XCNN_FAST_TYPED_DOOR (lane fx-typed-pilot, 2026-11-18;
        # opt-in arm, default off, A/B owed: docs/apple-fast/EXPERIMENTS.md):
        # a resident forward or backward of an average pool whose layer has
        # a valid typed record is answered from the typed tables
        # (bindings/xcnn_pool_typed_door.mojo); null means "not covered" and
        # the door below runs as it always did. With the define off this
        # block is not compiled and the entry is the door alone.
        if is_defined["MOJOLEARN_XCNN_FAST_TYPED_DOOR"]():
            var typed = xcnn_pool_typed_call(args_ptr)
            if typed:
                return typed
        var call = PythonObject(from_borrowed=args_ptr)
        # fx-cnn-layers (2026-11-18): the argument tuple's items are read as
        # borrowed tuple items (`xcnn_at`; each subscript built an index
        # object first), each once, and the bound finalizer pair is told by
        # an exact-type test against the held tuple type (it built an empty
        # tuple to take its type). Every layer call and every tensor release
        # enters here.
        # Import-time native descriptors bind (context, operation); CPython
        # decref then enters this C finalizer directly, without a Python body.
        if len(call) == 2:
            var bound = xcnn_at(call, 0)
            if not xcnn_exact(bound, XT_TUPLE) or len(bound) != 2:
                raise Error("TypeError: CNN finalizer requires its native bound context")
            return _xcnn_finalize(xcnn_at(call, 1), String(xcnn_at(bound, 1)), xcnn_at(bound, 0)).steal_data()
        var context = xcnn_at(call, 3)
        var operation = String(py=xcnn_at(call, 1))
        if operation != "tensor_finalize":
            xcnn_connect_numpy(context)
        var owner = xcnn_at(call, 0)
        var args = xcnn_at(call, 2)
        if operation.startswith("tensor_"):
            return xcnn_tensor_operation(owner, operation, args, context).steal_data()
        if operation.startswith("conv_") or operation.startswith("block_"):
            return xcnn_conv_layers(owner, operation, args, context).steal_data()
        if operation == "cnnclf_init":
            return cnnclf_initialize(owner, args, context).steal_data()
        if operation == "cnnclf_weights":
            return cnnclf_weights(owner, context).steal_data()
        if operation == "cnnclf_fit":
            return cnnclf_fit(owner, args, context).steal_data()
        if operation == "cnnclf_predict":
            return cnnclf_predict(owner, args, context).steal_data()
        if operation == "cnnclf_predict_proba":
            return cnnclf_predict_proba(owner, args, context).steal_data()
        if operation.startswith("gcn_") or operation.startswith("sage_") or operation.startswith("linear_") or operation.startswith("relu_"):
            return xcnn_graph_layers(owner, operation, args, context).steal_data()
        if operation.startswith("bn_"):
            return xcnn_bn_layers(owner, operation, args, context).steal_data()
        var answered = xcnn_pool_layers(owner, operation, args, context)
        if is_defined["MOJOLEARN_XCNN_FAST_TYPED_DOOR"]():
            # the door answered: keep the layer's typed record for the next call
            xcnn_pool_typed_learn(owner, xcnn_at(call, 1), context)
        return answered.steal_data()
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
