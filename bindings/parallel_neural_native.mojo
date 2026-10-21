# SPDX-License-Identifier: Apache-2.0
"""Native owner and local workers for existing parallel MLP/Samba training."""
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.generation_buffers import GENERATION_BOUNDARY, generation_array, generation_copy
from bindings.parallel_byte_native import _parallel_int, _parallel_view
from bindings.parallel_mlp_native import (
    mlp_snapshot, mlp_from_state, mlp_publish, mlp_gradients_from_state,
    mlp_apply_from_state, mlp_batch, mlp_targets, neural_ordered_sum,
    neural_cooperative_admit,
)
from bindings.native_process_pool import (
    process_pool_create, process_pool_map, process_pool_close,
    process_worker_connect, process_worker_next, process_send,
)
from bindings.native_transport import transport_close
from bindings.parallel_neural_admission import neural_admit


struct ParallelNeuralBoundary(Defaultable, Movable):
    var bi: PythonObject
    var owner_cls: PythonObject
    var mlp_cls: PythonObject
    var samba_cls: PythonObject
    var samba_call: PythonObject
    var samba_context: PythonObject
    var admission_context: PythonObject
    var lock: PythonObject

    def __init__(out self):
        self.bi = PythonObject(None)
        self.owner_cls = PythonObject(None)
        self.mlp_cls = PythonObject(None)
        self.samba_cls = PythonObject(None)
        self.samba_call = PythonObject(None)
        self.samba_context = PythonObject(None)
        self.admission_context = PythonObject(None)
        self.lock = PythonObject(None)


comptime PARALLEL_NEURAL_BOUNDARY = _Global[
    StorageType=ParallelNeuralBoundary, name="MojoParallelNeuralOwner",
    init_fn=ParallelNeuralBoundary.__init__,
]


def parallel_neural_initialize(args: PythonObject) raises:
    var b = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()
    b[].bi = Python.import_module("builtins")
    b[].owner_cls = args[0]
    b[].mlp_cls = args[1]
    b[].samba_cls = args[2]
    b[].samba_call = args[3]
    b[].samba_context = args[4]
    b[].admission_context = args[5]
    b[].lock = Python.import_module("_thread").RLock
    if b[].samba_call is not None and b[].bi.type(b[].samba_call) is not b[].bi.type(b[].bi.len):
        raise Error("parallel neural: Samba operations must use the compiled native dispatcher")


def _neural_samba(operation: String, values: PythonObject) raises -> PythonObject:
    var b = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()
    if b[].samba_call is None:
        raise Error("ImportError: installed native Samba training dispatcher is unavailable")
    return b[].samba_call(operation, b[].bi.tuple(values), b[].samba_context)


def _neural_values(value: PythonObject) raises -> PythonObject:
    var result = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()[].bi.list()
    result.append(value)
    return result


def _neural_snapshot(owner: PythonObject) raises -> PythonObject:
    if String(owner._operation) == "mlp_gradient":
        return mlp_snapshot(owner.model)
    var args = _neural_values(owner.model)
    args.append(True)
    return _neural_samba("snapshot", args)


def _neural_publish(owner: PythonObject, state: PythonObject) raises:
    if String(owner._operation) == "mlp_gradient":
        mlp_publish(owner.model, state)
    else:
        var args = _neural_values(owner.model)
        args.append(state)
        _ = _neural_samba("publish", args)


def _neural_init(owner: PythonObject, model: PythonObject, devices: PythonObject, logical: PythonObject, child_module: String) raises:
    var b = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()
    var shards = _parallel_int(logical, 1, 1024, "logical_shards")
    var operation = String()
    if b[].bi.type(model) is b[].mlp_cls:
        operation = "mlp_gradient"
        _ = mlp_snapshot(model)
    elif b[].bi.type(model) is b[].samba_cls:
        operation = "samba_gradient"
        var fields = b[].bi.object.__getattribute__(model, "__dict__")
        var optimizer = b[].bi.object.__getattribute__(fields["optimizer"], "__dict__")
        if Int(py=optimizer["accumulation_steps"]) != 1:
            raise Error("parallel Samba requires accumulation_steps=1; logical_shards owns accumulation")
        if fields["numeric_mode"] is not None and String(fields["numeric_mode"]) != "identical":
            raise Error("parallel training requires identical numeric mode")
        var values = _neural_values(model)
        values.append(True)
        var snapshot = _neural_samba("snapshot", values)
        if String(snapshot["numeric_mode"]) != "identical":
            raise Error("parallel training requires an identical model")
    else:
        raise Error("TypeError: ParallelNeuralTrainer supports SmallMLPTrainer and SambaStack")
    var pool = process_pool_create(devices, False, child_module)
    if len(pool["devices"]) > shards:
        raise Error("physical device count exceeds logical shard count")
    var update_pool = process_pool_create(pool["devices"], True, child_module)
    owner._operation = operation
    owner._pool = pool
    owner._update_pool = update_pool
    owner.model = model
    owner.logical_shards = shards
    owner._lock = b[].lock()
    owner._closed = False
    owner._gradients = PythonObject(None)


def _neural_ids(value: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var own = g[].bi.type(value) is g[].array_cls
    var view = _parallel_view(value)
    var shape = value.shape if own else view.shape
    if len(shape) != 2:
        raise Error("mojolearn.SambaStack: inputs and targets must be (B,L) integer ids")
    var rows = Int(py=shape[0])
    var columns = Int(py=shape[1])
    var fmt = String(view.format.lstrip("=@<"))
    if fmt != "b" and fmt != "B" and fmt != "h" and fmt != "H" and fmt != "i" and fmt != "I" and fmt != "l" and fmt != "L" and fmt != "q" and fmt != "Q":
        raise Error("mojolearn.SambaStack: inputs and targets must be native integer buffers")
    var packed = g[].arrays(fmt)
    packed.frombytes(view.tobytes("C"))
    var out = generation_array([rows, columns], "i")
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=out._addr))
    var column_major = own and String(value.order) == "F"
    for i in range(rows * columns):
        var source = i % columns * rows + i // columns if column_major else i
        var number = packed[source]
        if Bool(py=number < -2147483648) or Bool(py=number > 2147483647):
            raise Error("OverflowError: Samba token id is outside int32")
        dst[i] = Int32(Int(py=number))
    _ = packed^
    return out


def _neural_request(operation: String, state: PythonObject, args: PythonObject) raises -> PythonObject:
    var bi = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()[].bi
    var request = bi.list()
    request.append(operation)
    request.append(state)
    request.append(args)
    return bi.tuple(request)


def _neural_train(owner: PythonObject, supplied: PythonObject) raises -> PythonObject:
    var b = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()
    if Bool(py=owner._closed):
        raise Error("RuntimeError: parallel trainer is closed")
    var shards = b[].bi.list(supplied)
    if len(shards) != Int(py=owner.logical_shards):
        raise Error("logical shard count mismatch")
    var snapshot = _neural_snapshot(owner)
    owner._gradients = PythonObject(None)
    try:
        var requests = b[].bi.list()
        var stream = PythonObject(None)
        var next_rng = PythonObject(None)
        var samba = String(owner._operation) == "samba_gradient"
        if samba and Float64(py=snapshot["config"]["dropout"]) > 0:
            var counter = Int(py=snapshot["rng"]["counter"])
            if counter > 0xFFFFFFFF:
                raise Error("RuntimeError: mojolearn.Generator stream counter exhausted")
            stream = counter
            next_rng = b[].bi.dict(snapshot["rng"])
            next_rng["counter"] = counter + 1
        var offset = 0
        for i in range(len(shards)):  # driver-loop: over logical training shards in immutable reduction order
            var shard = b[].bi.tuple(shards[i])
            if len(shard) != 2:
                raise Error("parallel shard must contain inputs and targets")
            var args = b[].bi.list()
            if samba:
                var inputs = _neural_ids(shard[0])
                args.append(inputs)
                args.append(_neural_ids(shard[1]))
                args.append(stream)
                args.append(offset)
                offset += Int(py=inputs.size)
            else:
                var inputs = mlp_batch(shard[0])
                args.append(inputs)
                args.append(mlp_targets(shard[1], Int(py=inputs.shape[0])))
            requests.append(_neural_request(String(owner._operation), snapshot, b[].bi.tuple(args)))
        var results = process_pool_map(owner._pool, requests)
        var losses = b[].bi.list()
        var parts = b[].bi.list()
        for i in range(len(results)):  # driver-loop: over returned logical shard loss/gradient records
            losses.append(results[i][0])
            parts.append(results[i][1])
        var update_state = snapshot
        if next_rng is not None:
            update_state = b[].bi.dict(snapshot)
            update_state["rng"] = next_rng
        var update_requests = b[].bi.list()
        update_requests.append(_neural_request("samba_update" if samba else "mlp_update", update_state, parts))
        var updated = process_pool_map(owner._update_pool, update_requests)[0]
        _neural_publish(owner, updated[0])
        owner._gradients = updated[1]
        var result = b[].bi.dict()
        result["losses"] = b[].bi.tuple(losses)
        result["completed_steps"] = updated[2]
        result["logical_shards"] = owner.logical_shards
        result["reduction"] = "ordered_sum"
        return result
    except error:
        # Keep the same complete-model rollback contract, including RNG and
        # schedule state, even if a publication failed after a worker reply.
        try:
            _neural_publish(owner, snapshot)
        except rollback_error:
            raise Error("RuntimeError: parallel operation failed: " + String(error) + "; full-state rollback also failed: " + String(rollback_error))
        raise error


def _neural_checkpoint(owner: PythonObject) raises -> PythonObject:
    var result = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()[].bi.dict()
    result["schema"] = "mojolearn.parallel-neural.v1"
    result["operation"] = owner._operation
    result["logical_shards"] = owner.logical_shards
    result["reduction"] = "ordered_sum"
    result["state"] = _neural_snapshot(owner)
    return result


def _neural_restore(cls: PythonObject, checkpoint: PythonObject, devices: PythonObject, child_module: String) raises -> PythonObject:
    var b = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()
    if String(checkpoint.get("schema", "")) != "mojolearn.parallel-neural.v1" or String(checkpoint.get("reduction", "")) != "ordered_sum":
        raise Error("unsupported parallel neural checkpoint contract")
    var operation = String(checkpoint.get("operation", ""))
    var model = PythonObject(None)
    if operation == "mlp_gradient":
        model = mlp_from_state(checkpoint["state"])
    elif operation == "samba_gradient":
        model = _neural_samba("from_state", _neural_values(checkpoint["state"]))
    else:
        raise Error("unsupported parallel neural checkpoint operation")
    var owner = b[].bi.object.__new__(cls)
    _neural_init(owner, model, devices, checkpoint["logical_shards"], child_module)
    return owner


def _neural_close(owner: PythonObject) raises:
    owner._closed = True
    var failure = String()
    try:
        process_pool_close(owner._pool)
    except error:
        failure = String(error)
    try:
        process_pool_close(owner._update_pool)
    except error:
        if failure == "":
            failure = String(error)
    if failure != "":
        raise Error(failure)


def neural_worker_dispatch(request: PythonObject) raises -> PythonObject:
    var b = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()
    if len(request) != 3:
        raise Error("parallel neural: invalid worker request envelope")
    var operation = String(request[0])
    var state = request[1]
    var args = request[2]
    if operation == "mlp_gradient":
        return mlp_gradients_from_state(state, args[0], args[1])
    if operation == "samba_gradient":
        var values = _neural_values(state)
        for i in range(len(args)):  # driver-loop: over the compiled Samba gradient operation envelope
            values.append(args[i])
        return _neural_samba("gradients", values)
    if operation != "mlp_update" and operation != "samba_update":
        raise Error("parallel neural: unsupported worker operation")
    neural_cooperative_admit()
    var grads = neural_ordered_sum(args)
    var updated = PythonObject(None)
    var step = 0
    if operation == "mlp_update":
        updated = mlp_apply_from_state(state, grads)
        step = Int(py=updated["optimizer"]["step"])
    else:
        var values = _neural_values(state)
        values.append(grads)
        updated = _neural_samba("apply", values)
        step = Int(py=updated["t"])
    var result = b[].bi.list()
    result.append(updated)
    result.append(grads)
    result.append(step)
    return b[].bi.tuple(result)


def neural_worker(argv: PythonObject) raises:
    var socket = process_worker_connect(argv)
    try:
        while True:  # driver-loop: over complete native local model training requests
            var request = process_worker_next(socket)
            if request is None:
                break
            var sequence = Int(py=request[0].get("sequence", -1))
            try:
                var result = neural_worker_dispatch(request[1])
                process_send(socket, "result", sequence, result)
            except error:
                process_send(socket, "error", sequence, PythonObject(String(error)))
    except error:
        transport_close(socket)
        raise error
    transport_close(socket)


def parallel_neural_call(operation: String, args: PythonObject) raises -> PythonObject:
    var b = PARALLEL_NEURAL_BOUNDARY.get_or_create_ptr()
    var child = String("mojolearn._native_parallel_neural_worker")
    if operation == "initialize":
        parallel_neural_initialize(args)
        return PythonObject(None)
    neural_admit(operation, args, b[].owner_cls, b[].mlp_cls, b[].samba_cls, b[].admission_context)
    if operation == "init":
        _neural_init(args[0], args[1], args[2], args[3], child)
        return PythonObject(None)
    if operation == "init_with_worker":
        # Verification entry chooses an independent child audit controller;
        # the public constructor always selects the installed native worker.
        _neural_init(args[0], args[1], args[2], args[3], String(args[4]))
        return PythonObject(None)
    if operation == "restore":
        return _neural_restore(args[0], args[1], args[2], child)
    if operation == "ordered_sum":
        return neural_ordered_sum(args[0])
    if operation == "worker":
        neural_worker(args)
        return PythonObject(None)
    if operation == "worker_dispatch":
        return neural_worker_dispatch(args[0])
    var owner = args[0]
    _ = owner._lock.acquire()
    var result = PythonObject(None)
    try:
        if operation == "train":
            result = _neural_train(owner, args[1])
        elif operation == "export_gradients":
            if owner._gradients is None:
                raise Error("RuntimeError: no committed parallel gradient")
            result = b[].bi.list()
            for i in range(len(owner._gradients)):  # driver-loop: over committed gradient output buffers
                result.append(generation_copy(owner._gradients[i]))
        elif operation == "checkpoint":
            result = _neural_checkpoint(owner)
        elif operation == "close" or operation == "exit":
            _neural_close(owner)
        elif operation == "enter":
            result = owner
        else:
            raise Error("parallel neural: unknown operation " + operation)
    except error:
        _ = owner._lock.release()
        raise error
    _ = owner._lock.release()
    return result


def parallel_neural_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        var result = parallel_neural_call(String(args[0]), args[1])
        return result.steal_data()
    except error:
        var message = String(error)
        if message.startswith("TypeError:"):
            return raise_python_exception(error, ExceptionType.TypeError)
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error, ExceptionType("PyExc_RuntimeError"))
        if message.startswith("ImportError:"):
            return raise_python_exception(error, ExceptionType("PyExc_ImportError"))
        if message.startswith("OverflowError:"):
            return raise_python_exception(error, ExceptionType("PyExc_OverflowError"))
        if message.startswith("NotImplementedError:"):
            return raise_python_exception(error, ExceptionType("PyExc_NotImplementedError"))
        return raise_python_exception(error, ExceptionType.ValueError)
