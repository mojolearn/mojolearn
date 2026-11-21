# SPDX-License-Identifier: Apache-2.0
# Whole-fold native process coordination. Each worker owns one local native
# estimator at a time; the driver sends a complete wave before reading it in
# fold order. No Python task loop, pickle reducer, estimator or scorer callback.
#
# Rows (lane ne1-b-door, N057): X and y cross the socket ONCE per worker (a
# "data" request after the identity check); every fold request then carries
# only the fold's train and test index arrays, and the worker gathers the
# fold's rows out of its own resident copy (`MselFoldRows.rows`: the same
# device gather the driver ran per fold before, now on the worker's device).
# The same rows reach the same estimator in the same order: no bit changes.
#
# UNPACK and PACK only: the worker-index and route refusals, the request
# codes and their order rule, the wave plan and the reply order are typed in
# bindings/parallel_cv_native_run.mojo. The frames stay the codec's (a packed
# model carries `bytes`, which the typed frame of core/parallel_worker_frame.mojo
# refuses); the member is fitted and scored through msel_fit / msel_score
# (boundary calls) until member families publish entry tables.
from std.ffi import external_call
from bindings.native_loaded_module_door import native_loaded_module
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.native_codec_door import codec_initialize
from bindings.native_process_pool_door import (
    process_initialize, process_pool_create, process_pool_start, process_pool_map_typed, process_pool_close, process_pool_environment_set,
    process_worker_connect, process_worker_next, process_worker_value, process_worker_reply, process_worker_refuse, WorkerTurn,
    process_pool_pids,
)
from bindings.native_transport_door import transport_close
from bindings.native_worker_inventory_door import native_worker_inventory, native_require_worker_inventory
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.msel_cv_door import msel_prepare_folds, MselFoldRows, msel_fit
from bindings.msel_common_door import msel_score, msel_f64_list
from bindings.msel_parallel_wire_unpack import (
    msel_wire_needs_numpy,
)
from bindings.msel_parallel_wire_pack import (
    msel_parallel_model_pack,
    msel_parallel_model_unpack,
    msel_parallel_scorer_pack,
    msel_parallel_scorer_unpack,
    msel_wire_pack,
    msel_wire_unpack,
)
from bindings.msel_parallel_witness_pack import msel_parallel_capture
from bindings.optimizer_public_words_unpack import PENDING_PYTHON_ERROR, PYTHON_ERROR_PENDING
from bindings.parallel_keys_unpack import parallel_keys
from core.parallel_worker_request import WorkerRequest, worker_request_decode
from bindings.py_balanced import py_item
from bindings.parallel_cv_native_run import (
    CV_REQUEST_IDENTITY, CV_REQUEST_DATA, CV_DEVICE_NOT_AN_INDEX,
    cv_devices_admit, cv_vendor_admit, cv_request_admit, cv_wave_count,
    cv_plain_words, cv_fold_request, cv_fold_admit,
    cv_batch_count_admit, cv_reply_order_admit,
    CvTrace, cv_stage_name, cv_status_name, CV_STATUS_RUNNING, CV_STATUS_FAILED, CV_STATUS_COMPLETE,
    CV_STAGE_VALIDATED, CV_STAGE_START, CV_STAGE_IDENTITY, CV_STAGE_DATA, CV_STAGE_WAVE,
)


def msel_parallel_initialize(context: PythonObject) raises:
    var nk = parallel_keys()
    var metadata = py_item(context, nk[].k_family_msel_parallel)
    # These globals belong to this physical DSO. The tokenizer owner's
    # import-time initialization does not initialize a separate dev CV owner.
    process_initialize(String(py_item(metadata, nk[].k_executable)), py_item(metadata, nk[].k_environ), String(py_item(metadata, nk[].k_vendor)))
    codec_initialize(py_item(context, nk[].k_array_class), PythonObject(None), PythonObject(None))


def _devices(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(value) is not b.tuple and py_type(value) is not b.list:
        raise Error("TypeError: devices must be an explicit list or tuple of distinct nonnegative integers")
    var selected = py_call(b.tuple, value)
    var indices = List[Int](capacity=len(selected))
    for i in range(len(selected)):  # driver-loop: exact explicit worker indices, before spawn
        indices.append(Int(py=selected[i]) if py_type(selected[i]) is b.int else CV_DEVICE_NOT_AN_INDEX)
    cv_devices_admit(indices)
    var nk = parallel_keys()
    var metadata = py_item(context, nk[].k_family_msel_parallel)
    var vendor = String(py_item(metadata, nk[].k_vendor))
    # `cpu_only` is read only on the CPU vendor, as before.
    cv_vendor_admit(indices, vendor, vendor == "cpu" and Bool(py=py_item(metadata, nk[].k_cpu_only)))
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
    var b = native_loaded_module("builtins")
    var name = String("native worker exception")
    var nk = parallel_keys()
    if owned[0] is not None and py_type(owned[0]) is b.type:
        name = String(py_call(py_item(b.type.__dict__, nk[].k___name__).__get__, owned[0]))
    var value = owned[1]
    if py_type(value) is b.str:
        return name + ": " + String(value)
    if value is not None:
        var args = py_call(py_item(py_item(context, nk[].k_family_msel_parallel), nk[].k_exception_args), value, py_type(value))
        if py_type(args) is b.tuple and len(args) > 0 and py_type(args[0]) is b.str:
            return name + ": " + String(args[0])
    return name + ": native fold failed (non-string exception payload retained by its native boundary)"


def _fold_rows(typed: WorkerRequest, request: PythonObject, mut rows: MselFoldRows, inventory: PythonObject,
               context: PythonObject) raises -> PythonObject:
    # The typed request holds the fold index (handle) and the capture flag
    # (its word); the payload is (model, scorer, train, test). The fold's
    # rows are gathered here out of the run's X and y the data request left.
    var b = native_loaded_module("builtins")
    cv_fold_admit(typed)
    if py_type(request) is not b.tuple or len(request) != 4:
        raise Error("ValueError: malformed native CV fold request")
    if rows.x is None:
        raise Error("RuntimeError: CV worker fold rows require the run's X and y first")
    var train = request[2]
    var test = request[3]
    var x = rows.x
    var y = rows.y
    var x_train = rows.rows(x, train, context)
    var x_test = rows.rows(x, test, context)
    var y_train = rows.rows(y, train, context)
    var y_test = rows.rows(y, test, context)
    # The fold's index arrays are this request's objects: drop their device
    # copies now (X and y stay resident for the next fold).
    rows.resident_release(train, context)
    rows.resident_release(test, context)
    var estimator = msel_parallel_model_unpack(request[0], context)
    var scorer = msel_parallel_scorer_unpack(request[1], context)
    msel_fit(estimator, x_train, y_train, context)
    var score = msel_score(estimator, x_test, y_test, scorer, context)
    var captured = PythonObject(None)
    if typed.words[0] != 0:
        captured = msel_parallel_capture(estimator, x_test, y_test, score, inventory, context)
    return Python.tuple(PythonObject(typed.handle), score, captured)


def _worker_data(request: PythonObject, mut rows: MselFoldRows, context: PythonObject) raises -> PythonObject:
    # The data request's payload (X, y): this run's X and y, kept for every
    # later fold request.
    var b = native_loaded_module("builtins")
    if py_type(request) is not b.tuple or len(request) != 2 or request[0] is None or request[1] is None:
        raise Error("ValueError: malformed native CV data request")
    rows.close(context)
    rows = MselFoldRows(request[0], request[1], Python.list(), False, context)
    return PythonObject(True)


def _worker_request(socket: PythonObject, turn: WorkerTurn, mut inventory: PythonObject,
                    mut rows: MselFoldRows, metadata: PythonObject, context: PythonObject) raises:
    # One authenticated request owns its result/error response, but never the
    # socket lifetime. A broken envelope or error-response send still raises
    # to the worker's unconditional close. The inventory cache belongs to the
    # worker and is updated in place, including before a result-send failure.
    # The pinned parser rejects this handler as the final nested statement of
    # the worker's try/finally (https://github.com/modular/modular/issues/6718).
    var nk = parallel_keys()
    var sequence = turn.sequence
    try:
        # The typed request (a frame without request words is refused here).
        var typed = worker_request_decode(Int(turn.request.unsafe_ptr()), len(turn.request) * 8)
        var result = PythonObject(None)
        var code = typed.op
        cv_request_admit(code, inventory is not None)
        if code == CV_REQUEST_IDENTITY:
            if inventory is None:
                inventory = native_worker_inventory(String(py_item(metadata, nk[].k_vendor)), py_item(metadata, nk[].k_environ))
            result = inventory
        elif code == CV_REQUEST_DATA:
            result = _worker_data(process_worker_value(turn), rows, context)
        else:
            result = _fold_rows(typed, process_worker_value(turn), rows, inventory, context)
        process_worker_reply(socket, sequence, result)
    except error:
        process_worker_refuse(socket, sequence, _worker_error(error, context))


def msel_parallel_worker(argv: PythonObject, context: PythonObject) raises:
    classical_connect_numpy(context)
    var nk = parallel_keys()
    var metadata = py_item(context, nk[].k_family_msel_parallel)
    var socket = process_worker_connect(argv)
    var inventory = PythonObject(None)
    # The run's X and y once a "data" request arrives (none until then).
    var rows = MselFoldRows(PythonObject(None), PythonObject(None), Python.list(), False, context)
    try:
        while True:  # driver-loop: authenticated fixed native CV request dispatch
            var turn = process_worker_next(socket)
            if not turn.open:
                break
            _worker_request(socket, turn, inventory, rows, metadata, context)
        rows.close(context)
    finally:
        # A worker that stops on an error exits with its process; the device
        # copies of its X and y go with it.
        transport_close(socket)


def msel_parallel_require_batch(result: PythonObject, start: Int, count: Int) raises:
    var b = native_loaded_module("builtins")
    var listed = py_type(result) is b.list
    cv_batch_count_admit(listed, len(result) if listed else 0, count)
    for i in range(len(result)):  # driver-loop: validate all reply envelopes before consuming any score
        var reply = result[i]
        var shaped = py_type(reply) is b.tuple and len(reply) == 3 and py_type(reply[0]) is b.int
        cv_reply_order_admit(shaped, Int(py=reply[0]) if shaped else -1, start + i)
        if py_type(reply[1]) is not b.float:
            raise Error("TypeError: cross-validation worker score must be a native real scalar")
        if reply[2] is not None and py_type(reply[2]) is not b.dict:
            raise Error("TypeError: cross-validation worker capture must be native metadata")


def _trace_publish(published: PythonObject, trace: CvTrace) raises:
    """PACK of the run's record into the `last_run` mapping."""
    var nk = parallel_keys()
    var stages = Python.list()
    for i in range(len(trace.stages)):  # small-loop(the stages one run passed)
        _ = py_call(stages.append, PythonObject(cv_stage_name(trace.stages[i])))
    var workers = Python.list()
    for i in range(len(trace.workers)):  # small-loop(one process id per worker)
        _ = py_call(workers.append, PythonObject(trace.workers[i]))
    var waves = Python.list()
    for i in range(len(trace.waves)):  # small-loop(one size per wave)
        _ = py_call(waves.append, PythonObject(trace.waves[i]))
    var assignments = Python.list()
    for i in range(len(trace.assignments) // 3):  # small-loop(one triple per assigned fold)
        _ = py_call(assignments.append, Python.tuple(PythonObject(trace.assignments[3 * i]),
            PythonObject(trace.assignments[3 * i + 1]), PythonObject(trace.assignments[3 * i + 2])))
    py_setitem(published, nk[].k_stages, stages)
    py_setitem(published, nk[].k_workers, workers)
    py_setitem(published, nk[].k_waves, waves)
    py_setitem(published, nk[].k_assignments, assignments)
    py_setitem(published, nk[].k_status, PythonObject(cv_status_name(trace.status)))
    py_setitem(published, nk[].k_closed, PythonObject(trace.closed))


def msel_parallel_score(args: PythonObject, capture: Bool, context: PythonObject) raises -> PythonObject:
    # All configuration, callback refusal and fold validation precede spawn.
    # The run's record is typed (`CvTrace`); `published` is its PACKed form,
    # the `last_run` mapping, written when validation ends and when the run
    # ends (it was a dictionary appended to stage by stage).
    var trace = CvTrace()
    var published = Python.dict()
    var nk = parallel_keys()
    py_setitem(py_item(context, nk[].k_family_msel_parallel), nk[].k_last_run, published)
    _trace_publish(published, trace)
    var devices = _devices(args[3], context)
    # UNPACK of the admitted worker indices, once: the wave plan and the
    # assignments read these words.
    var device_words = List[Int]()
    for i in range(len(devices)):  # small-loop(explicit worker indices, before spawn)
        device_words.append(Int(py=devices[i]))
    var prepared = msel_prepare_folds(args[0], args[1], args[2], args[4], args[5], args[6], args[7], context)
    var model = msel_parallel_model_pack(args[0], context)
    var scorer = msel_parallel_scorer_pack(prepared[3], context)
    trace.stage(CV_STAGE_VALIDATED)
    _trace_publish(published, trace)
    if capture:
        var cls = py_type(args[0])
        var accepted = py_item(py_item(context, nk[].k_family_msel_parallel), nk[].k_capture_classes)
        if cls is not accepted[0] and cls is not accepted[1]:
            raise Error("TypeError: native CV model captures require the connected boosting classifier or regressor")
    var pool = process_pool_create(devices, False, "mojolearn._native_parallel_cv_worker")
    var environment = Python.dict()
    py_setitem(environment, nk[].k_MOJOLEARN_NATIVE_CV_NUMPY, "1" if msel_wire_needs_numpy(model) or msel_wire_needs_numpy(scorer) else "0")
    process_pool_environment_set(pool, environment)
    var scores = List[Float64]()
    var captures = Python.list()
    try:
        trace.closed = False
        trace.status = CV_STATUS_RUNNING
        trace.stage(CV_STAGE_START)
        var requests = Python.list()
        for i in range(len(devices)):  # driver-loop: one pre-fit identity request per selected worker
            _ = py_call(requests.append, PythonObject(None))
        var identities = process_pool_map_typed(pool, cv_plain_words(CV_REQUEST_IDENTITY, len(devices)), requests)
        # The spawned process ids, from the pool's record.
        trace.workers = process_pool_pids(pool)
        var pids = Python.list()
        for i in range(len(trace.workers)):  # driver-loop: authenticate inventory against actual spawned PIDs
            _ = py_call(pids.append, PythonObject(trace.workers[i]))
        native_require_worker_inventory(identities, String(py_item(py_item(context, nk[].k_family_msel_parallel), nk[].k_vendor)), pids,
                                        Int(external_call["getpid", Int32]()))
        py_setitem(published, nk[].k_identities, identities)
        trace.stage(CV_STAGE_IDENTITY)
        # X and y go to every worker once (N057); the fold requests below
        # carry only index arrays.
        requests = Python.list()
        for i in range(len(devices)):  # driver-loop: one data request per selected worker
            _ = py_call(requests.append, Python.tuple(prepared[0], prepared[1]))
        var loaded = process_pool_map_typed(pool, cv_plain_words(CV_REQUEST_DATA, len(devices)), requests)
        if py_type(loaded) is not native_loaded_module("builtins").list or len(loaded) != len(devices):
            raise Error("ValueError: cross-validation workers did not all receive the run's X and y")
        trace.stage(CV_STAGE_DATA)
        var folds = prepared[2]
        var width = len(device_words)
        for start in range(0, len(folds), width):  # driver-loop: bounded whole-fold waves
            requests = Python.list()
            var words = List[List[Int64]]()
            var wave = cv_wave_count(width, len(folds), start)
            trace.waves.append(wave)
            for i in range(start, start + wave):  # driver-loop: independent fresh estimator per fold
                var pair = folds[i]
                trace.assign(i, device_words[i - start], trace.workers[i - start])
                words.append(cv_fold_request(i, capture))
                _ = py_call(requests.append, Python.tuple(model, scorer, pair[0], pair[1]))
            trace.stage(CV_STAGE_WAVE)
            var result = process_pool_map_typed(pool, words, requests)
            msel_parallel_require_batch(result, start, len(requests))
            for i in range(len(result)):  # driver-loop: consume in logical fold order, never completion order
                scores.append(Float64(py=result[i][1]))
                if capture:
                    _ = py_call(captures.append, result[i][2])
    except error:
        var failure = _worker_error(error, context)
        var cleanup = String()
        try:
            process_pool_close(pool)
        except cleanup_error:
            cleanup += "; process cleanup also failed: " + _worker_error(cleanup_error, context)
        trace.error = failure + cleanup
        trace.finish(CV_STATUS_FAILED, len(process_pool_pids(pool)) == 0)
        _trace_publish(published, trace)
        py_setitem(published, nk[].k_error, PythonObject(failure + cleanup))
        raise Error(failure + cleanup)
    process_pool_close(pool)
    trace.finish(CV_STATUS_COMPLETE, len(process_pool_pids(pool)) == 0)
    _trace_publish(published, trace)
    var result = msel_f64_list(scores, context)
    return Python.tuple(result, captures) if capture else result


def msel_parallel_operation(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = parallel_keys()
    if operation == "family_msel_parallel_initialize":
        msel_parallel_initialize(context)
    elif operation == "family_msel_parallel_worker":
        msel_parallel_worker(args[0], context)
    elif operation == "family_msel_parallel_score" or operation == "family_msel_parallel_capture":
        return msel_parallel_score(args, operation == "family_msel_parallel_capture", context)
    elif operation == "family_msel_parallel_inventory":
        var metadata = py_item(context, nk[].k_family_msel_parallel)
        return native_worker_inventory(String(py_item(metadata, nk[].k_vendor)), py_item(metadata, nk[].k_environ))
    elif operation == "family_msel_parallel_require_inventory":
        if py_type(args[1]) is not py_type(PythonObject("")) or py_type(args[3]) is not py_type(PythonObject(0)):
            raise Error("TypeError: inventory route and parent PID require exact native metadata")
        native_require_worker_inventory(args[0], String(args[1]), args[2], Int(py=args[3]))
    elif operation == "family_msel_parallel_wire":
        return msel_wire_unpack(msel_wire_pack(args[0], context), context)
    elif operation == "family_msel_parallel_model_pack":
        return msel_parallel_model_pack(args[0], context)
    elif operation == "family_msel_parallel_last_run":
        return py_item(py_item(context, nk[].k_family_msel_parallel), nk[].k_last_run)
    elif operation == "family_msel_parallel_require_batch":
        if py_type(args[1]) is not py_type(PythonObject(0)) or py_type(args[2]) is not py_type(PythonObject(0)):
            raise Error("TypeError: native batch bounds must be integers")
        msel_parallel_require_batch(args[0], Int(py=args[1]), Int(py=args[2]))
    else:
        raise Error("ValueError: unknown native parallel CV operation")
    return PythonObject(None)
