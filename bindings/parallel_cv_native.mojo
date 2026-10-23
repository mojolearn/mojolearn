# SPDX-License-Identifier: Apache-2.0
# Whole-fold native process coordination. Each worker owns one local native
# estimator at a time; the driver sends a complete wave before reading it in
# fold order. No Python task loop, pickle reducer, estimator or scorer callback.
from std.ffi import external_call
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.native_codec import codec_initialize
from bindings.native_process_pool import (
    process_initialize, process_pool_create, process_pool_start, process_pool_map, process_pool_close,
    process_worker_connect, process_worker_next, process_send,
)
from bindings.native_transport import transport_close
from bindings.native_worker_inventory import native_worker_inventory, native_require_worker_inventory
from bindings.classical_attributes import classical_connect_numpy
from bindings.msel_cv import msel_prepare_folds, MselFoldRows, msel_fit
from bindings.msel_common import msel_score, msel_f64_list
from bindings.msel_parallel_wire import (
    msel_parallel_model_pack, msel_parallel_model_unpack, msel_parallel_scorer_pack,
    msel_parallel_scorer_unpack, msel_wire_needs_numpy, msel_wire_pack, msel_wire_unpack,
)
from bindings.msel_parallel_witness import msel_parallel_capture
from bindings.optimizer_public_words import PENDING_PYTHON_ERROR, PYTHON_ERROR_PENDING


def msel_parallel_initialize(context: PythonObject) raises:
    var metadata = context["family_msel_parallel"]
    # These globals belong to this physical DSO. The tokenizer owner's
    # import-time initialization does not initialize a separate dev CV owner.
    process_initialize(String(metadata["executable"]), metadata["environ"], String(metadata["vendor"]))
    codec_initialize(context["array_class"], PythonObject(None), PythonObject(None))


def _devices(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if py_type(value) is not b.tuple and py_type(value) is not b.list:
        raise Error("TypeError: devices must be an explicit list or tuple of distinct nonnegative integers")
    if len(value) == 0:
        raise Error("ValueError: devices must not be empty")
    var selected = py_call(b.tuple, value)
    var seen = Python.list()
    for i in range(len(selected)):  # driver-loop: exact explicit worker indices, before spawn
        if py_type(selected[i]) is not b.int or Int(py=selected[i]) < 0:
            raise Error("ValueError: devices must contain distinct nonnegative integers")
        if selected[i] in seen:
            raise Error("ValueError: devices must not repeat a worker index")
        _ = py_call(seen.append, selected[i])
    var metadata = context["family_msel_parallel"]
    var vendor = String(metadata["vendor"])
    if vendor == "metal" and (len(selected) != 1 or Int(py=selected[0]) != 0):
        raise Error("NotImplementedError: Metal cross-validation supports devices=(0,) only")
    if vendor != "cuda" and vendor != "hip" and vendor != "metal" and not (vendor == "cpu" and Bool(py=metadata["cpu_only"])):
        raise Error("NotImplementedError: parallel cross-validation requires CUDA, HIP, Metal device zero or a CPU-only install")
    return selected


def _worker_error(error: Error, context: PythonObject) raises -> String:
    var message = String(error)
    if message != PYTHON_ERROR_PENDING:
        return message
    # family_call retained the callee's exception. Transfer its three owned
    # references now, without calling an exception's Python __str__/__repr__.
    ref cpy = Python().cpython()
    var pending = PENDING_PYTHON_ERROR.get_or_create_ptr()
    var words = pending[].words.copy()
    var owned = Python.list()
    pending[].words = List[Int](length=3, fill=0)
    try:
        for i in range(3):  # small-loop(three owned CPython exception references)
            if words[i] != 0:
                var pointer = cpy.lib.get_function[PyObjectPtr]("Py_NewRef")(words[i])
                _ = py_call(owned.append, PythonObject(from_owned=pointer))
            else:
                _ = py_call(owned.append, PythonObject(None))
    finally:
        for i in range(3):  # small-loop(release the three original references even if packaging fails)
            if words[i] != 0:
                cpy.lib.get_function[NoneType]("Py_DecRef")(words[i])
    var b = Python.import_module("builtins")
    var name = String("native worker exception")
    if owned[0] is not None and py_type(owned[0]) is b.type:
        name = String(py_call(b.type.__dict__["__name__"].__get__, owned[0]))
    var value = owned[1]
    if py_type(value) is b.str:
        return name + ": " + String(value)
    if value is not None:
        var args = py_call(context["family_msel_parallel"]["exception_args"], value, py_type(value))
        if py_type(args) is b.tuple and len(args) > 0 and py_type(args[0]) is b.str:
            return name + ": " + String(args[0])
    return name + ": native fold failed (non-string exception payload retained by its native boundary)"


def _fold(request: PythonObject, inventory: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if py_type(request) is not b.tuple or len(request) != 9 or py_type(request[1]) is not b.int or py_type(request[8]) is not b.bool:
        raise Error("ValueError: malformed native CV fold request")
    var estimator = msel_parallel_model_unpack(request[2], context)
    var scorer = msel_parallel_scorer_unpack(request[3], context)
    msel_fit(estimator, request[4], request[5], context)
    var score = msel_score(estimator, request[6], request[7], scorer, context)
    var captured = PythonObject(None)
    if Bool(py=request[8]):
        # The capture operation is a native verification mode of this same
        # fold path. Completed artifacts are consumed by the testing caller.
        captured = msel_parallel_capture(estimator, request[6], request[7], score, inventory, context)
    return Python.tuple(request[1], score, captured)


def msel_parallel_worker(argv: PythonObject, context: PythonObject) raises:
    classical_connect_numpy(context)
    var metadata = context["family_msel_parallel"]
    var socket = process_worker_connect(argv)
    var inventory = PythonObject(None)
    try:
        while True:  # driver-loop: authenticated fixed native CV request dispatch
            var envelope = process_worker_next(socket)
            if envelope is None:
                break
            var sequence = Int(py=py_get(envelope[0], "sequence", -1))
            try:
                var request = envelope[1]
                var b = Python.import_module("builtins")
                if py_type(request) is not b.tuple or len(request) < 1 or py_type(request[0]) is not b.str:
                    raise Error("ValueError: malformed native CV worker request")
                var result = PythonObject(None)
                if String(request[0]) == "identity" and len(request) == 1:
                    if inventory is None:
                        inventory = native_worker_inventory(String(metadata["vendor"]), metadata["environ"])
                    result = inventory
                elif String(request[0]) == "fold":
                    if inventory is None:
                        raise Error("RuntimeError: CV worker identity must be admitted before fitting")
                    result = _fold(request, inventory, context)
                else:
                    raise Error("ValueError: unknown native CV worker request")
                process_send(socket, "result", sequence, result)
            except error:
                process_send(socket, "error", sequence, PythonObject(_worker_error(error, context)))
    finally:
        transport_close(socket)


def msel_parallel_require_batch(result: PythonObject, start: Int, count: Int) raises:
    var b = Python.import_module("builtins")
    if py_type(result) is not b.list or len(result) != count:
        raise Error("ValueError: cross-validation workers returned an incomplete fold batch")
    for i in range(len(result)):  # driver-loop: validate all reply envelopes before consuming any score
        var reply = result[i]
        if py_type(reply) is not b.tuple or len(reply) != 3 or py_type(reply[0]) is not b.int or Int(py=reply[0]) != start + i:
            raise Error("ValueError: cross-validation worker fold order differs")
        if py_type(reply[1]) is not b.float:
            raise Error("TypeError: cross-validation worker score must be a native real scalar")
        if reply[2] is not None and py_type(reply[2]) is not b.dict:
            raise Error("TypeError: cross-validation worker capture must be native metadata")


def msel_parallel_score(args: PythonObject, capture: Bool, context: PythonObject) raises -> PythonObject:
    # All configuration, callback refusal and fold validation precede spawn.
    var trace = Python.dict()
    py_setitem(trace, "stages", Python.list("validate"))
    py_setitem(trace, "workers", Python.list())
    py_setitem(trace, "waves", Python.list())
    py_setitem(trace, "assignments", Python.list())
    py_setitem(trace, "status", "VALIDATING")
    py_setitem(trace, "closed", True)
    py_setitem(context["family_msel_parallel"], "last_run", trace)
    var devices = _devices(args[3], context)
    var prepared = msel_prepare_folds(args[0], args[1], args[2], args[4], args[5], args[6], args[7], context)
    var model = msel_parallel_model_pack(args[0], context)
    var scorer = msel_parallel_scorer_pack(prepared[3], context)
    _ = py_call(trace["stages"].append, "validated")
    if capture:
        var cls = py_type(args[0])
        var accepted = context["family_msel_parallel"]["capture_classes"]
        if cls is not accepted[0] and cls is not accepted[1]:
            raise Error("TypeError: native CV model captures require the connected boosting classifier or regressor")
    var pool = process_pool_create(devices, False, "mojolearn._native_parallel_cv_worker")
    var environment = Python.dict()
    py_setitem(environment, "MOJOLEARN_NATIVE_CV_NUMPY", "1" if msel_wire_needs_numpy(model) or msel_wire_needs_numpy(scorer) else "0")
    py_setitem(pool, "environment", environment)
    var rows = MselFoldRows(prepared[0], prepared[1], prepared[2], False, context)
    var scores = List[Float64]()
    var captures = Python.list()
    try:
        py_setitem(trace, "closed", False)
        py_setitem(trace, "status", "RUNNING")
        _ = py_call(trace["stages"].append, "start")
        var requests = Python.list()
        for i in range(len(devices)):  # driver-loop: one pre-fit identity request per selected worker
            _ = py_call(requests.append, Python.tuple("identity"))
        var identities = process_pool_map(pool, requests)
        var pids = Python.list()
        for i in range(len(pool["workers"])):  # driver-loop: authenticate inventory against actual spawned PIDs
            _ = py_call(pids.append, pool["workers"][i]["pid"])
        py_setitem(trace, "workers", pids)
        native_require_worker_inventory(identities, String(context["family_msel_parallel"]["vendor"]), pids,
                                        Int(external_call["getpid", Int32]()))
        py_setitem(trace, "identities", identities)
        _ = py_call(trace["stages"].append, "identity")
        var folds = prepared[2]
        var width = len(devices)
        for start in range(0, len(folds), width):  # driver-loop: bounded whole-fold waves
            requests = Python.list()
            _ = py_call(trace["waves"].append, min(width, len(folds) - start))
            for i in range(start, min(start + width, len(folds))):  # driver-loop: independent fresh estimator per fold
                var data = rows.take(i, PythonObject(None), context)
                _ = py_call(trace["assignments"].append, Python.tuple(i, devices[i - start], pids[i - start]))
                _ = py_call(requests.append, Python.tuple("fold", i, model, scorer, data[0], data[1], data[2], data[3], capture))
            _ = py_call(trace["stages"].append, "wave")
            var result = process_pool_map(pool, requests)
            msel_parallel_require_batch(result, start, len(requests))
            for i in range(len(result)):  # driver-loop: consume in logical fold order, never completion order
                scores.append(Float64(py=result[i][1]))
                if capture:
                    _ = py_call(captures.append, result[i][2])
    except error:
        var failure = _worker_error(error, context)
        var cleanup = String()
        try:
            rows.close(context)
        except cleanup_error:
            cleanup += "; row cleanup also failed: " + _worker_error(cleanup_error, context)
        try:
            process_pool_close(pool)
        except cleanup_error:
            cleanup += "; process cleanup also failed: " + _worker_error(cleanup_error, context)
        py_setitem(trace, "closed", len(pool["workers"]) == 0)
        py_setitem(trace, "status", "FAILED")
        py_setitem(trace, "error", failure + cleanup)
        _ = py_call(trace["stages"].append, "close")
        raise Error(failure + cleanup)
    try:
        rows.close(context)
    finally:
        process_pool_close(pool)
    py_setitem(trace, "closed", len(pool["workers"]) == 0)
    py_setitem(trace, "status", "COMPLETE")
    _ = py_call(trace["stages"].append, "close")
    var result = msel_f64_list(scores, context)
    return Python.tuple(result, captures) if capture else result


def msel_parallel_operation(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if operation == "family_msel_parallel_initialize":
        msel_parallel_initialize(context)
    elif operation == "family_msel_parallel_worker":
        msel_parallel_worker(args[0], context)
    elif operation == "family_msel_parallel_score" or operation == "family_msel_parallel_capture":
        return msel_parallel_score(args, operation == "family_msel_parallel_capture", context)
    elif operation == "family_msel_parallel_inventory":
        var metadata = context["family_msel_parallel"]
        return native_worker_inventory(String(metadata["vendor"]), metadata["environ"])
    elif operation == "family_msel_parallel_require_inventory":
        if py_type(args[1]) is not py_type(PythonObject("")) or py_type(args[3]) is not py_type(PythonObject(0)):
            raise Error("TypeError: inventory route and parent PID require exact native metadata")
        native_require_worker_inventory(args[0], String(args[1]), args[2], Int(py=args[3]))
    elif operation == "family_msel_parallel_wire":
        return msel_wire_unpack(msel_wire_pack(args[0], context), context)
    elif operation == "family_msel_parallel_model_pack":
        return msel_parallel_model_pack(args[0], context)
    elif operation == "family_msel_parallel_scorer_pack":
        return msel_parallel_scorer_pack(args[0], context)
    elif operation == "family_msel_parallel_last_run":
        return context["family_msel_parallel"]["last_run"]
    elif operation == "family_msel_parallel_require_batch":
        if py_type(args[1]) is not py_type(PythonObject(0)) or py_type(args[2]) is not py_type(PythonObject(0)):
            raise Error("TypeError: native batch bounds must be integers")
        msel_parallel_require_batch(args[0], Int(py=args[1]), Int(py=args[2]))
    else:
        raise Error("ValueError: unknown native parallel CV operation")
    return PythonObject(None)
