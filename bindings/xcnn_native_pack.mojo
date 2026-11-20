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
from bindings.py_balanced import py_type, py_call, py_setitem, py_item
from bindings.xcnn_tensor_native_pack import xcnn_tensor_operation
from bindings.xcnn_pool_native_door import xcnn_pool_layers
# TOMBSTONE: MOJOLEARN_XCNN_FAST_TYPED_DOOR (NOISE, A/B round 1 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T125554Z-6f7afc, one run per arm, digests equal on 12 cells, 0.92-1.08, ten cells within 1 %) deleted 2026-11-19 by lane/fx-verdicts3; code recoverable at f96f23b7fb.
# Restore: git apply experiments/removed/MOJOLEARN_XCNN_FAST_TYPED_DOOR.patch; record in docs/TOMBSTONES.md.
from bindings.xcnn_bn_native_door import xcnn_bn_layers
from bindings.xcnn_conv_native_door import xcnn_conv_layers
from bindings.xcnn_graph_native_door import xcnn_graph_layers
from bindings.xcnn_classifier_native_door import cnnclf_fit, cnnclf_predict, cnnclf_predict_proba, cnnclf_initialize, cnnclf_weights
from bindings.xcnn_keys_unpack import (
    xcnn_keys, XCNN_OPN_TENSOR_FINALIZE, XCNN_OPN_TENSOR_RESHAPE, XCNN_OPN_DEV_FINALIZE, XCNN_OPN_BLOCK_INIT,
    XCNN_OPN_CONV_TRANSFORM, XCNN_OPN_GCN_INIT, XCNN_OPN_RELU_BACKWARD, XCNN_OPN_BN_INIT, XCNN_OPN_BN_FIT,
    XCNN_OPN_CNNCLF_INIT, XCNN_OPN_CNNCLF_WEIGHTS, XCNN_OPN_CNNCLF_FIT, XCNN_OPN_CNNCLF_PREDICT,
    XCNN_OPN_CNNCLF_PREDICT_PROBA,
)
from bindings.py_keys import py_op_code


def _xcnn_finalize(owner: PythonObject, name: PythonObject, context: PythonObject) -> PythonObject:
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
        # bw1-xcnn (2026-11-19): the bound name is resolved to its code by one
        # lookup of the caller's own str (it was copied into a String and
        # compared); a name that is not an operation is -1 and releases
        # nothing, as before.
        var nk = xcnn_keys()
        var operation = py_op_code(nk[].ops, name)
        if operation == XCNN_OPN_TENSOR_FINALIZE:
            return xcnn_tensor_operation(owner, operation, Python.tuple(), context)
        if operation == XCNN_OPN_DEV_FINALIZE:
            var fields = xcnn_dev_fields(owner, context)
            var entries = xcnn_module_entries(py_item(fields, nk[].k_b), context)
            var key = xcnn_ref(entries)
            var table = py_item(fields, nk[].k_h)
            var names = py_call(xcnn_builtins().list, table.keys())
            py_setitem(fields, nk[].k_h, Python.dict())
            for i in range(len(names)):  # driver-loop: release the owner's native allocation handles
                var word = xcnn_word(table[names[i]][0])
                var py = Python()
                with GILReleased(py):
                    xcnn_tensor_run_free(key, word)
    except:
        pass  # partially initialized owner or interpreter shutdown
    return PythonObject(None)


comptime _XCNN_TENSOR = 0
comptime _XCNN_CONV = 1
comptime _XCNN_GRAPH = 2
comptime _XCNN_BN = 3
comptime _XCNN_OTHER = 4


def _xcnn_owner(operation: Int, name: PythonObject) raises -> Int:
    """The owner of an operation code: a range test, the codes being numbered
    by owner (bindings/xcnn_keys_unpack.mojo). Only a name that is not an
    operation (-1) builds its String, for the prefix route it took before
    (a name that is not a str raises here, where its String copy raised)."""
    if operation >= XCNN_OPN_TENSOR_FINALIZE and operation <= XCNN_OPN_TENSOR_RESHAPE:
        return _XCNN_TENSOR
    if operation >= XCNN_OPN_BLOCK_INIT and operation <= XCNN_OPN_CONV_TRANSFORM:
        return _XCNN_CONV
    if operation >= XCNN_OPN_GCN_INIT and operation <= XCNN_OPN_RELU_BACKWARD:
        return _XCNN_GRAPH
    if operation >= XCNN_OPN_BN_INIT and operation <= XCNN_OPN_BN_FIT:
        return _XCNN_BN
    if operation >= 0:
        return _XCNN_OTHER
    var text = String(py=name)
    if text.startswith("tensor_"):
        return _XCNN_TENSOR
    if text.startswith("conv_") or text.startswith("block_"):
        return _XCNN_CONV
    if text.startswith("gcn_") or text.startswith("sage_") or text.startswith("linear_") or text.startswith("relu_"):
        return _XCNN_GRAPH
    if text.startswith("bn_"):
        return _XCNN_BN
    return _XCNN_OTHER


def xcnn_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        # TOMBSTONE: MOJOLEARN_XCNN_FAST_TYPED_DOOR (NOISE) typed-table answer ahead of the door deleted 2026-11-19 by lane/fx-verdicts3; code recoverable at f96f23b7fb.
        # Restore: git apply experiments/removed/MOJOLEARN_XCNN_FAST_TYPED_DOOR.patch
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
            return _xcnn_finalize(xcnn_at(call, 1), xcnn_at(bound, 1), xcnn_at(bound, 0)).steal_data()
        # bw1-xcnn (2026-11-19): the operation is one lookup of the caller's
        # str in the held operation table (it was copied into a String and
        # routed by up to eleven prefix tests and compares). Codes are
        # numbered by owner, so the owner is a range test; a name that is not
        # an operation (-1) is routed by its prefix, as before, to the owner
        # whose unchanged "unknown operation" error it raised.
        var context = xcnn_at(call, 3)
        var nk = xcnn_keys()
        var name = xcnn_at(call, 1)
        var operation = py_op_code(nk[].ops, name)
        var owner_group = _xcnn_owner(operation, name)
        if operation != XCNN_OPN_TENSOR_FINALIZE:
            xcnn_connect_numpy(context)
        var owner = xcnn_at(call, 0)
        var args = xcnn_at(call, 2)
        if owner_group == _XCNN_TENSOR:
            return xcnn_tensor_operation(owner, operation, args, context).steal_data()
        if owner_group == _XCNN_CONV:
            return xcnn_conv_layers(owner, operation, name, args, context).steal_data()
        if operation == XCNN_OPN_CNNCLF_INIT:
            return cnnclf_initialize(owner, args, context).steal_data()
        if operation == XCNN_OPN_CNNCLF_WEIGHTS:
            return cnnclf_weights(owner, context).steal_data()
        if operation == XCNN_OPN_CNNCLF_FIT:
            return cnnclf_fit(owner, args, context).steal_data()
        if operation == XCNN_OPN_CNNCLF_PREDICT:
            return cnnclf_predict(owner, args, context).steal_data()
        if operation == XCNN_OPN_CNNCLF_PREDICT_PROBA:
            return cnnclf_predict_proba(owner, args, context).steal_data()
        if owner_group == _XCNN_GRAPH:
            return xcnn_graph_layers(owner, operation, name, args, context).steal_data()
        if owner_group == _XCNN_BN:
            return xcnn_bn_layers(owner, operation, args, context).steal_data()
        return xcnn_pool_layers(owner, operation, name, args, context).steal_data()
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
