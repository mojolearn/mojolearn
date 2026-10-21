# SPDX-License-Identifier: Apache-2.0
"""Reject protocol callbacks before the parallel owner touches model/process state."""
from std.python import Python, PythonObject
from bindings.samba_admission import samba_admit, samba_admit_value, _samba_metadata


def _neural_container(value: PythonObject, label: String, range_ok: Bool = False) raises:
    var bi = Python.import_module("builtins")
    var cls = bi.type(value)
    if cls is not bi.list and cls is not bi.tuple and not (range_ok and cls is bi.range):
        raise Error("TypeError: parallel " + label + " requires a native list or tuple; Python iteration hooks are not accepted")


def neural_admit(operation: String, args: PythonObject, owner_cls: PythonObject,
                 mlp_cls: PythonObject, samba_cls: PythonObject, context: PythonObject) raises:
    var bi = Python.import_module("builtins")
    var admission = context["admission"]
    # Validate cached class descriptors before reading even an exact owner.
    samba_admit("parallel_layout", bi.tuple(), context)
    _neural_container(args, "argument envelope")
    if operation == "worker" or operation == "worker_dispatch":
        samba_admit_value(args, admission)
        return
    if operation == "ordered_sum":
        _neural_container(args[0], "gradient shards")
        for i in range(len(args[0])):  # driver-loop: over logical shard containers before native reduction
            _neural_container(args[0][i], "gradient tensors")
        samba_admit_value(args, admission)
        return
    if operation == "init" or operation == "init_with_worker":
        if bi.type(args[0]) is not owner_cls:
            raise Error("TypeError: ParallelNeuralTrainer requires its native public class without subclass hooks")
        if bi.type(args[1]) is not mlp_cls and bi.type(args[1]) is not samba_cls:
            raise Error("TypeError: ParallelNeuralTrainer requires an exact SmallMLPTrainer or SambaStack; Python trainer hooks are not accepted")
        _neural_container(args[2], "devices", True)
        _samba_metadata(args[2], admission)
        _samba_metadata(args[3], admission)
        samba_admit_value(args, admission)
        return
    if operation == "restore":
        if args[0] is not owner_cls:
            raise Error("TypeError: ParallelNeuralTrainer restore requires its native public class without subclass hooks")
        _neural_container(args[2], "devices", True)
        _samba_metadata(args[2], admission)
        samba_admit_value(args, admission)
        return
    var owner = args[0]
    if bi.type(owner) is not owner_cls:
        raise Error("TypeError: parallel operation requires its native public owner without subclass hooks")
    if operation == "train":
        _neural_container(args[1], "training shards")
        for i in range(len(args[1])):  # driver-loop: over logical shard envelopes before any snapshot or child launch
            _neural_container(args[1][i], "training inputs and targets")
        samba_admit_value(args, admission)
        return
    if operation == "checkpoint":
        samba_admit_value(owner, admission)
        return
    # Cleanup must remain available after rejecting invalid model metadata.
    # These operations never read the model, so validate only their own fields.
    var fields = admission["owner_fields"](owner)
    var keys = bi.list(fields)
    for i in range(len(keys)):  # driver-loop: over native owner keys before dictionary lookup
        if bi.type(keys[i]) is not bi.str:
            raise Error("TypeError: parallel owner metadata requires native string keys")
    for key in [String("_lock"), String("_pool"), String("_update_pool")]:
        samba_admit_value(fields[key], admission)
    if operation == "export_gradients":
        samba_admit_value(fields["_gradients"], admission)
