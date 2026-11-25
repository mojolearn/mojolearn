# SPDX-License-Identifier: Apache-2.0
"""Reject protocol callbacks before the parallel owner touches model/process state."""
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.samba_admission_door import samba_admit, samba_admit_value, _samba_metadata
from bindings.samba_keys_unpack import SAMBA_OPN_PARALLEL_LAYOUT
from bindings.py_balanced import py_call, py_type
from bindings.parallel_keys_unpack import (
    parallel_keys,
    PARALLEL_NEURAL_OPN_CHECKPOINT,
    PARALLEL_NEURAL_OPN_INIT,
    PARALLEL_NEURAL_OPN_INIT_WITH_WORKER,
    PARALLEL_NEURAL_OPN_ORDERED_SUM,
    PARALLEL_NEURAL_OPN_RESTORE,
    PARALLEL_NEURAL_OPN_TRAIN,
    PARALLEL_NEURAL_OPN_WORKER,
)
from bindings.py_balanced import py_item


def _neural_container(value: PythonObject, label: String, range_ok: Bool = False) raises:
    var bi = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is not bi.list and cls is not bi.tuple and not (range_ok and cls is bi.range):
        raise Error("TypeError: parallel " + label + " requires a native list or tuple; Python iteration hooks are not accepted")


def neural_admit(operation: Int, args: PythonObject, owner_cls: PythonObject,
                 mlp_cls: PythonObject, samba_cls: PythonObject, context: PythonObject) raises:
    var bi = native_loaded_module("builtins")
    var nk = parallel_keys()
    var admission = py_item(context, nk[].k_admission)
    # Validate cached class descriptors before reading even an exact owner.
    samba_admit(SAMBA_OPN_PARALLEL_LAYOUT, bi.tuple(), context)
    _neural_container(args, "argument envelope")
    if operation == PARALLEL_NEURAL_OPN_WORKER:
        samba_admit_value(args, admission)
        return
    if operation == PARALLEL_NEURAL_OPN_ORDERED_SUM:
        _neural_container(args[0], "gradient shards")
        for i in range(len(args[0])):  # driver-loop: over logical shard containers before native reduction
            _neural_container(args[0][i], "gradient tensors")
        # args[1] is the `device` word, admitted by its own exact read in the
        # door (neural_ordered_sum_device, `device_unpack`); the shards are
        # admitted here exactly as before (the envelope tuple only recursed).
        samba_admit_value(args[0], admission)
        return
    if operation == PARALLEL_NEURAL_OPN_INIT or operation == PARALLEL_NEURAL_OPN_INIT_WITH_WORKER:
        if py_type(args[0]) is not owner_cls:
            raise Error("TypeError: ParallelNeuralTrainer requires its native public class without subclass hooks")
        if py_type(args[1]) is not mlp_cls and py_type(args[1]) is not samba_cls:
            raise Error("TypeError: ParallelNeuralTrainer requires an exact SmallMLPTrainer or SambaStack; Python trainer hooks are not accepted")
        _neural_container(args[2], "devices", True)
        _samba_metadata(args[2], admission)
        _samba_metadata(args[3], admission)
        samba_admit_value(args, admission)
        return
    if operation == PARALLEL_NEURAL_OPN_RESTORE:
        if args[0] is not owner_cls:
            raise Error("TypeError: ParallelNeuralTrainer restore requires its native public class without subclass hooks")
        _neural_container(args[2], "devices", True)
        _samba_metadata(args[2], admission)
        samba_admit_value(args, admission)
        return
    var owner = args[0]
    if py_type(owner) is not owner_cls:
        raise Error("TypeError: parallel operation requires its native public owner without subclass hooks")
    if operation == PARALLEL_NEURAL_OPN_TRAIN:
        _neural_container(args[1], "training shards")
        for i in range(len(args[1])):  # driver-loop: over logical shard envelopes before any snapshot or child launch
            _neural_container(args[1][i], "training inputs and targets")
        samba_admit_value(args, admission)
        return
    if operation == PARALLEL_NEURAL_OPN_CHECKPOINT:
        samba_admit_value(owner, admission)
        return
    # Cleanup must remain available after rejecting invalid model metadata.
    # These operations never read the model, so validate only their own fields.
    var fields = py_call(py_item(admission, nk[].k_owner_fields), owner)
    var keys = py_call(bi.list, fields)
    for i in range(len(keys)):  # driver-loop: over native owner keys before dictionary lookup
        if py_type(keys[i]) is not bi.str:
            raise Error("TypeError: parallel owner metadata requires native string keys")
    samba_admit_value(py_item(fields, nk[].k__lock), admission)
    samba_admit_value(py_item(fields, nk[].k__pool), admission)
    samba_admit_value(py_item(fields, nk[].k__update_pool), admission)
    # The committed gradient buffers are no instance field: they are the
    # native arrays of the update worker's reply, held by the owner record
    # (bindings/parallel_neural_native_door.mojo), which no caller can replace.
