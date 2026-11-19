# SPDX-License-Identifier: Apache-2.0
# Whole CNN DeviceTensor public operations. Slot identities are import-connected;
# shapes and buffer owners are admitted before conversion or native execution.
# The PACK half of the former `bindings/xcnn_tensor_native.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/xcnn_tensor_native_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.neural_block_admission_door import neural_plain_value
from bindings.sequential_admission_door import sequential_admit_value, sequential_integer
from bindings.sequence_array_native_unpack import sequence_array_cast
from bindings.xcnn_common_door import (
    xcnn_builtins, xcnn_tensor, xcnn_tensor_slot, xcnn_tensor_set, xcnn_tensor_admit,
    xcnn_shape, xcnn_size, xcnn_tuple, xcnn_new, xcnn_out_new, xcnn_wrap,
    xcnn_host_of, xcnn_f32, xcnn_entries, xcnn_module_entries, xcnn_device_io_off,
    xcnn_is_t, xcnn_reshape_t, xcnn_tensor_layout, _xcnn_set_raw, _xcnn_slot_raw, xcnn_ref,
    xcnn_exact, xcnn_item, XT_INT, XK_TENSOR_CLASS,
)
from std.python._cpython import GILReleased
from x_cnn.tensor_run import (
    xcnn_tensor_run_check_dim, xcnn_tensor_run_check_handle, xcnn_tensor_run_handle_live,
    xcnn_tensor_run_reshape_dims, xcnn_tensor_run_free,
)


from bindings.xcnn_tensor_native_unpack import (
    _shape,
)
from bindings.xcnn_keys_unpack import (
    xcnn_keys, XCNN_OPN_TENSOR_ARRAY, XCNN_OPN_TENSOR_FINALIZE, XCNN_OPN_TENSOR_INIT, XCNN_OPN_TENSOR_NDIM,
    XCNN_OPN_TENSOR_NEW, XCNN_OPN_TENSOR_NEW_OUT, XCNN_OPN_TENSOR_NUMPY, XCNN_OPN_TENSOR_RESHAPE,
    XCNN_OPN_TENSOR_SIZE, XCNN_OPN_TENSOR_TO, XCNN_OPN_TENSOR_WRAP,
)
from bindings.py_balanced import py_item


def xcnn_tensor_operation(owner: PythonObject, operation: Int, args: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    if operation == XCNN_OPN_TENSOR_FINALIZE:
        # fx-cnn-layers (2026-11-18): tested first and without the builtins
        # read. Every output tensor a layer call returns is released through
        # here (the caller's reassignment of its result), so a layer call
        # paid this release too: 49k of the 403k instructions of a
        # global-avgpool forward in the L40S call profile. The owner's class
        # is compared through the held name and the handle's type against
        # the held int type; the slot reads and the write are the generic
        # attribute calls (`_xcnn_slot_raw`). A foreign owner is refused with
        # the same text, before the suppressing block, as before.
        if py_type(owner) is not xcnn_item(context, XK_TENSOR_CLASS):
            raise Error("TypeError: CNN tensor operation requires its exact native owner")
        # Partially initialized objects and interpreter shutdown are legal.
        # Suppression lives in native code, with handles cleared before free.
        try:
            # One layout check for the three reads and the write.
            xcnn_tensor_layout(context)
            var handle = _xcnn_slot_raw(owner, "h", context)
            var base = _xcnn_slot_raw(owner, "_base", context)
            if base is None and handle is not None:
                if not xcnn_exact(handle, XT_INT) or not xcnn_tensor_run_handle_live(Int(py=handle)):
                    return PythonObject(None)
                var entries = xcnn_module_entries(_xcnn_slot_raw(owner, "b", context), context)
                _xcnn_set_raw(owner, "h", PythonObject(None), context)
                var key = xcnn_ref(entries)
                var word = Int(py=handle)
                var py = Python()
                with GILReleased(py):
                    xcnn_tensor_run_free(key, word)
        except:
            pass
        return PythonObject(None)
    var b = xcnn_builtins()
    if operation == XCNN_OPN_TENSOR_TO:
        if xcnn_is_t(args[0], context):
            xcnn_tensor_admit(args[0], context)
            return args[0]
        var array = xcnn_f32(args[0], "x", context)
        var settings = Python.dict(numeric_mode=args[1])
        var entries = xcnn_entries(settings, context)
        if xcnn_device_io_off(entries, context):
            return array
        return xcnn_wrap(entries, array, context)
    var nk = xcnn_keys()
    if operation == XCNN_OPN_TENSOR_NEW or operation == XCNN_OPN_TENSOR_NEW_OUT or operation == XCNN_OPN_TENSOR_WRAP:
        if owner is not py_item(context, nk[].k_xcnn_tensor_class):
            raise Error("TypeError: CNN tensor allocation requires the connected class")
        var entries = xcnn_module_entries(args[0], context)
        if operation == XCNN_OPN_TENSOR_WRAP:
            sequential_admit_value(args[1], context)
            return xcnn_wrap(entries, args[1], context)
        var shape = _shape(args[1], context)
        return xcnn_new(entries, shape, False, context) if operation == XCNN_OPN_TENSOR_NEW else xcnn_out_new(entries, shape, context)
    if py_type(owner) is not py_item(context, nk[].k_xcnn_tensor_class):
        raise Error("TypeError: CNN tensor operation requires its exact native owner")
    if operation == XCNN_OPN_TENSOR_INIT:
        var entries = xcnn_module_entries(args[0], context)
        var shape = _shape(args[1], context)
        if args[2] is not None:
            if py_type(args[2]) is not b.int:
                raise Error("TypeError: CNN tensor handle requires a positive native integer")
            xcnn_tensor_run_check_handle(Int(py=args[2]))
        if args[3] is not None:
            sequential_admit_value(args[3], context)
        if args[4] is not None and not xcnn_is_t(args[4], context):
            raise Error("TypeError: CNN tensor base requires an exact native tensor")
        # One layout check for the five slot writes (it ran once per slot).
        xcnn_tensor_layout(context)
        _xcnn_set_raw(owner, "b", py_item(entries, nk[].k_module), context)
        _xcnn_set_raw(owner, "shape", xcnn_tuple(shape), context)
        _xcnn_set_raw(owner, "h", args[2], context)
        _xcnn_set_raw(owner, "_host", args[3], context)
        _xcnn_set_raw(owner, "_base", args[4], context)
        return PythonObject(None)
    xcnn_tensor_admit(owner, context)
    var shape = xcnn_shape(owner)
    if operation == XCNN_OPN_TENSOR_SIZE:
        return PythonObject(xcnn_size(shape))
    if operation == XCNN_OPN_TENSOR_NDIM:
        return PythonObject(len(shape))
    if operation == XCNN_OPN_TENSOR_NUMPY or operation == XCNN_OPN_TENSOR_ARRAY:
        var result = xcnn_host_of(owner, context)
        if operation == XCNN_OPN_TENSOR_NUMPY:
            return result
        return sequence_array_cast(result, args[0], py_item(context, nk[].k_xcnn_array_context))
    if operation == XCNN_OPN_TENSOR_RESHAPE:
        var requested = args[0]
        if py_type(requested) is not b.tuple:
            raise Error("TypeError: CNN reshape requires a native argument tuple")
        if len(requested) == 1 and (py_type(requested[0]) is b.list or py_type(requested[0]) is b.tuple):
            requested = requested[0]
        var dims = xcnn_tensor_run_reshape_dims(_shape(requested, context, True), shape)
        return xcnn_reshape_t(owner, dims, context)
    raise Error("ValueError: unknown CNN tensor operation")
