# SPDX-License-Identifier: Apache-2.0
# Native normalization for the existing Session API. Public wrappers hand
# over objects unchanged; this boundary owns descriptors, options and outputs.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_item, py_setattr, py_setitem, py_type, py_contains
from providers.python.autograd_python_errors_pyapi_pack import graph_int
from sequence.exec_trait import Exec
from bindings.autograd_keys_unpack import autograd_keys
from bindings.py_keys import py_op_add, py_op_code
from core.shared_slot import SharedSlot
from sequence.ops import FP
from autograd.ops_api import ops_session, ops_session_registry_token
from autograd.optimizer_requests import OpsOptimizerExtras
from autograd.optimizer_requests import (
    OpsOptimizerRequest, OpsOptimizerRestoreRequest, ops_optimizer_codes,
)
from providers.python.autograd_program_pyapi_door import program_input, _integer, ops_program_fit_native_py, ops_program_execute_py, fit_stop_options
from autograd.ops_frontend_impl import session_split, clip_low_unbounded, clip_high_unbounded
from autograd.ops_block_frontend import (
    SESSION_ATTENTION_MIN_PIECES, SESSION_ATTENTION_GATE, session_mamba_code, session_attention_ints,
    session_attention_option_words, session_block_create, session_block_step, session_lm_rows, session_lm_pick,
    session_lm_alive, session_rnn,
)
from sequence.ops import CELL_LSTM, CELL_GRU, CELL_RNN_TANH
from autograd.block_step_ops import OPS_ATTN_STEP
from autograd.program_io import ProgramInput
from core.cast_in import CastIn
from providers.python.autograd_program_arrays_pyapi_door import output_array_unfilled, is_integer_input, uint64_input, program_admit_value, program_numeric_scalar, program_builtin_types
from providers.python.autograd_pyapi_door import program_info, ops_profile_reset_py, ops_profile_report_py
from providers.python.autograd_model_checkpoint_pyapi_unpack import ops_load_program_py
from providers.python.autograd_model_checkpoint_pyapi_pack import ops_save_state_py, ops_load_state_py
from providers.python.autograd_program_save_pyapi_door import ops_program_save_native_py
from providers.python.autograd_program_owner_pyapi_door import program_initialize_owner, program_new_owner, program_owner_fields
from providers.python.autograd_owned_program_public_pyapi_door import ops_owned_program_create_request_py
from providers.python.autograd_program_public_dispatch_pyapi_pack import ops_program_public_execute_py
from bindings.native_json_door import native_path
from providers.python.autograd_weight_frontend_pyapi_pack import ops_weights_native_py
from providers.python.autograd_session_frontend_pyapi_door import ops_session_fields
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_resource_handle, neural_descriptor_resource_close
from providers.python.autograd_execution_profile_pyapi_pack import execution_profile_object
from providers.python.autograd_control_checkpoint_pyapi_pack import control_save_py, control_load_py
from providers.python.autograd_gradient_diagnostics_pyapi_pack import gradient_report_py
# nn-blocks-scan begin: Session.scan (autograd/nn_scan_frontend.mojo)
from autograd.nn_scan_frontend import session_scan
from autograd.autograd_ops_frontend_run import _SOP_SCAN
# nn-blocks-scan end
# nn-blocks-cbw begin
from autograd.program import OpsProgram
# nn-blocks-cbw end


# Session operation codes (row N089, lesson 12) and their one-per-process
# name table: the RUN half of this file, autograd/autograd_ops_frontend_run.mojo
# (wave 8 run split). The dispatch below compares the integer codes.
from autograd.autograd_ops_frontend_run import (
    _SESSION_OP_CODES, _append,
    _SOP_PROGRAM_PUBLIC, _SOP_PROGRAM_INIT, _SOP_ENTER, _SOP_FREE_PROGRAM, _SOP_CLOSE, _SOP_EXIT,
    _SOP_PROFILE_RESET, _SOP_PROFILE_REPORT, _SOP_LOAD_PARTITIONED, _SOP_LOAD_DISTRIBUTED,
    _SOP_PROGRAM_RUN, _SOP_FIT, _SOP_FIT_IF, _SOP_GRADIENT_REPORT, _SOP_EXECUTION_PROFILE_BEGIN,
    _SOP_EXECUTION_PROFILE_END, _SOP_SAVE_CONTROL, _SOP_LOAD_CONTROL, _SOP_SAVE_PROGRAM,
    _SOP_LOAD_WEIGHTS, _SOP_TENSOR, _SOP_IDS, _SOP_SHAPE, _SOP_NUMPY, _SOP_GRAD, _SOP_SCALAR,
    _SOP_CAPTURE, _SOP_LOAD_PROGRAM, _SOP_SAVE_STATE, _SOP_LOAD_STATE, _SOP_RESET, _SOP_FREE,
    _SOP_BACKWARD, _SOP_GRAD_TENSOR, _SOP_VJP, _SOP_HVP, _SOP_JVP, _SOP_ZERO_GRAD, _SOP_OPTIMIZER,
    _SOP_STEP, _SOP_SCHEDULER_STEP, _SOP_LEARNING_RATE, _SOP_OPTIMIZER_STATE,
    _SOP_LOAD_OPTIMIZER_STATE, _SOP_COPY, _SOP_DETACH, _SOP_MATMUL, _SOP_ADD, _SOP_SUB, _SOP_MUL,
    _SOP_DIV, _SOP_RELU, _SOP_SIGMOID, _SOP_TANH, _SOP_EXP, _SOP_LOG, _SOP_SQRT, _SOP_ERF,
    _SOP_SOFTPLUS, _SOP_SILU, _SOP_HARD_SWISH, _SOP_ABS, _SOP_FLOOR, _SOP_CEIL, _SOP_ROUND,
    _SOP_SIN, _SOP_COS, _SOP_SIGN, _SOP_RECIPROCAL, _SOP_POW, _SOP_MINIMUM, _SOP_MAXIMUM,
    _SOP_WHERE, _SOP_FULL, _SOP_EXPAND, _SOP_SCALE, _SOP_LEAKY_RELU, _SOP_ELU, _SOP_POW_SCALAR,
    _SOP_HARD_SIGMOID, _SOP_TRANSPOSE, _SOP_PERMUTE, _SOP_RESHAPE, _SOP_FLATTEN, _SOP_SQUEEZE,
    _SOP_UNSQUEEZE, _SOP_SUM, _SOP_MEAN, _SOP_MAX, _SOP_ANY, _SOP_ALL, _SOP_SOFTMAX,
    _SOP_LOG_SOFTMAX, _SOP_CONCAT, _SOP_SPLIT, _SOP_EMBEDDING, _SOP_TAKE, _SOP_CONV2D, _SOP_LINEAR,
    _SOP_MAX_POOL2D, _SOP_AVG_POOL2D, _SOP_GELU, _SOP_CLIP, _SOP_EQUAL, _SOP_NOT_EQUAL,
    _SOP_GREATER, _SOP_GREATER_EQUAL, _SOP_LESS, _SOP_LESS_EQUAL, _SOP_LOGICAL_NOT,
    _SOP_LOGICAL_AND, _SOP_LOGICAL_OR, _SOP_ARGMAX, _SOP_PAD, _SOP_CUMSUM, _SOP_INDEX,
    _SOP_LAYER_NORM, _SOP_TRANSFORMER_SOFTMAX, _SOP_TRANSFORMER_LINEAR, _SOP_RMS_NORM, _SOP_ROTARY,
    _SOP_DROPOUT, _SOP_PERMUTATION, _SOP_GATHER_ROWS, _SOP_MSE, _SOP_MAE, _SOP_HUBER,
    _SOP_BCE_WITH_LOGITS, _SOP_CROSS_ENTROPY, _SOP_ACCURACY, _SOP_LOG_LOSS, _SOP_ROC_AUC,
    _SOP_STATE, _SOP_MAMBA_BLOCK, _SOP_ATTENTION_BLOCK, _SOP_MAMBA_STEP, _SOP_ATTENTION_STEP, _SOP_SMALL_MLP,
    _SOP_LM_HEAD_LOSS, _SOP_LM_ROWS, _SOP_LM_PICK, _SOP_LM_ALIVE, _SOP_LM_EMBEDDING, _SOP_CONV_BLOCK,
    _SOP_CONV_LINEAR, _SOP_LSTM, _SOP_GRU, _SOP_RNN,
    _SOP_CUSTOM_OP, _SOP_CUSTOM,  # nn-blocks-cbw
)
# nn-blocks-qkv begin
from autograd.autograd_ops_frontend_run import _SOP_ATTENTION_QKV
# nn-blocks-qkv end
# nn-blocks-pg begin: optimizer parameter groups and the shared optimizer UNPACK / PACK
from autograd.autograd_ops_frontend_run import _SOP_PG_FIRST, _SOP_PG_LAST
from providers.python.nn_pg_pyapi_door import (
    pg_session_call_py, ops_optimizer_unpack, ops_optimizer_state_pack, ops_optimizer_state_unpack,
)
# nn-blocks-pg end
# nn-blocks-scatter begin
from autograd.autograd_ops_frontend_run import _SOP_SCATTER_ADD, _SOP_SEGMENT_SUM, _SOP_SEGMENT_MEAN, _SOP_SEGMENT_MAX
from autograd.nn_scatter import OPS_SCATTER_ADD, OPS_SEGMENT_SUM
# nn-blocks-scatter end


struct _SessionOpNames(Defaultable, Movable):
    # The Session operation names as one Python dictionary (interned name ->
    # code), built once per process from the native table of the RUN half.
    var ready: Bool
    var ops: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)


comptime _SESSION_OP_NAMES = SharedSlot[_SessionOpNames, "MojolearnSessionOpNamesV1"]


def _session_op_names() raises -> PythonObject:
    var held = _SESSION_OP_NAMES.get_or_create_ptr()
    if not held[].ready:
        var ops = Python.dict()
        var native = _SESSION_OP_CODES.get_or_create_ptr()
        for item in native[].codes.items():  # driver-loop: over the fixed Session operation names, once per process
            py_op_add(ops, item.key, item.value)
        held[].ops = ops
        held[].ready = True
    return held[].ops


def _nil(x: PythonObject) -> Bool:
    return x is PythonObject(None)


def _int(x: PythonObject, name: String = "integer", low: Int = -9223372036854775807) raises -> Int:
    return _integer(x, name, low)


def _float(x: PythonObject, name: String) raises -> Float32:
    # A real number for the scalar argument `name`; anything else (a str,
    # None, an object without __float__) is refused naming the argument.
    if not program_numeric_scalar(x):
        raise Error("TypeError: " + name + " must be a native real scalar without conversion hooks")
    try:
        return Float32(py=x)
    except error:
        raise Error("TypeError: " + name + " must be a real number")


def _is_int(x: PythonObject) raises -> Bool:
    return is_integer_input(x)


def _ints(x: PythonObject, name: String = "integer", low: Int = -9223372036854775807) raises -> List[Int]:
    var out = List[Int]()
    if _is_int(x):
        out.append(_int(x, name, low))
    else:
        for k in range(len(x)):  # driver-loop: over public handle/shape metadata
            out.append(_int(x[k], name, low))
    return out^


def _axes(x: PythonObject) raises -> List[Int]:
    if _nil(x):
        return List[Int]()
    var axes = _ints(x, "axis")
    if len(axes) > 8:
        raise Error("ValueError: axis lists more than eight axes")
    return axes^


def _keep(x: PythonObject) raises -> Int:
    return -1 if _nil(x) else Int(Bool(py=x))


def _seed_words(x: PythonObject) raises -> List[Int]:
    if not _is_int(x):
        raise Error("TypeError: seed must be an integer")
    var seed = uint64_input(x, "seed")
    var out = List[Int]()
    out.append(Int(seed & 0xffffffff))
    out.append(Int(seed >> 32))
    return out^


def _tuple_ints(xs: List[Int]) raises -> PythonObject:
    var out = Python.list()
    for k in range(len(xs)):  # driver-loop: over returned metadata
        _ = py_call(out.append, xs[k])
    return py_call(native_loaded_module("builtins").tuple, out)



# TOMBSTONE: MOJOLEARN_SESSION_CALL_DESCRIPTOR (unmeasured B arm: every Session call through the descriptor; EXP:2395 RESTORED the direct typed call as default) deleted 2026-11-15 by lane/expdel2-08-base; code recoverable at 8fe6eaaa01.
# Restore: git apply experiments/removed/MOJOLEARN_SESSION_CALL_DESCRIPTOR.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/expdel2-restore-owed: owed B arm (owed-pre0867.json), under four weeks old; the deletion was premature.
# TOMBSTONE: MOJOLEARN_SESSION_CALL_DESCRIPTOR (B arm loses: every same-image Session call through the descriptor, zero-filled read-back arrays; A/B abn9 NVIDIA L40S, ops-session-step 2.63 -> 2.93 ms 1.11, program-run 1.00 arm/base) deleted 2026-11-21 by lane/fx-verdicts7; code recoverable at 519388d23a.
# Restore: git apply experiments/removed/MOJOLEARN_SESSION_CALL_DESCRIPTOR.fx-verdicts7.patch; record in docs/TOMBSTONES.md.


# nn-blocks-cbw begin: UNPACK of Session.custom_op: a Program's description,
# read once from the inference session that captured it (this connection).
def _cbw_program[E: Exec & Movable & Deinitable, KIND: Int](program: PythonObject, which: String) raises -> OpsProgram:
    var nk = autograd_keys()
    var fields = program_owner_fields(program)
    var pid = py_item(fields, nk[].k__id)
    if _nil(pid):
        raise Error("ValueError: custom_op: the " + which + " program was freed")
    var owner = ops_session_fields(py_item(fields, nk[].k__s))
    var owner_id = py_item(owner, nk[].k__id)
    if _nil(owner_id):
        raise Error("ValueError: custom_op: the session that captured the " + which + " program is closed")
    var o = ops_session[E, KIND](_integer(owner_id, "session", 1))
    return o[].program(_integer(pid, which + " program", 1))
# nn-blocks-cbw end


def _program_initialize(mut out: PythonObject, session: PythonObject, info: PythonObject) raises:
    # Retained native initializer entry shared by ImportedProgram construction.
    program_initialize_owner(out, session, info)


def ops_session_call_py[E: Exec & Movable & Deinitable, KIND: Int](
    session: PythonObject, operation: PythonObject, args: PythonObject,
) raises -> PythonObject:
    return ops_session_call_fields_py[E, KIND](session, operation, args, ops_session_fields(session))


def ops_session_call_fields_py[E: Exec & Movable & Deinitable, KIND: Int](
    session: PythonObject, operation: PythonObject, args: PythonObject, fields: PythonObject,
) raises -> PythonObject:
    # The descriptor route's entry (`session_public_native_v1`) and the
    # retained `ops_session_call_py`: admit the public handle against the
    # native registry owner, then run the operation.
    var resource_scope: List[Int] = [ops_session_registry_token(), KIND]
    # Held keys: `_id` was read by name four times here (a Python str built,
    # hashed and freed each time) on every operation of a session owned by
    # another image (the CPU image of `device="cpu"` on a GPU installation).
    var nk = autograd_keys()
    var handle = py_item(fields, nk[].k__id)
    if not _nil(handle):
        var types = program_builtin_types()
        if py_type(handle) is not types[].int_type:
            raise Error("TypeError: Session handle must be a native integer")
        var admitted_id = neural_descriptor_resource_handle(py_item(fields, nk[].k__lifetime), resource_scope)
        if admitted_id == 0 or admitted_id != Int(py=handle):
            raise Error("TypeError: Session handle differs from its native registry owner")
    return ops_session_call_admitted_py[E, KIND](session, operation, args, fields)


# Lane fix-executor-session-rows (2026-11-13): the operation body, for a
# caller that has already admitted `fields["_id"]` against this image's
# registry owner. The same-image public door
# (`autograd_session_public_request_pyapi_pack.mojo`) admits it from the
# resource it already holds, so a Session call no longer reads `_lifetime`,
# builds the scope list and walks the resource words a second time per op
# (about 30 calls per ops-session-step round). A call from another image
# enters through `ops_session_call_fields_py` above, as before.
def ops_session_call_admitted_py[E: Exec & Movable & Deinitable, KIND: Int](
    session: PythonObject, operation: PythonObject, args: PythonObject, fields: PythonObject,
) raises -> PythonObject:
    # The code of the caller's own operation str (bindings/py_keys.mojo): one
    # dictionary read. The former door copied the name into a String and
    # hashed the copy on every Session operation. An unknown name is 0, as
    # before, and is refused at the end of the chain in the same words.
    var op = py_op_code(_session_op_names(), operation)
    if op < 0:
        op = 0
    # Held keys: every operation read `_id` by name twice (two Python strs
    # built, hashed and freed); it is read once, with the held key.
    var nk = autograd_keys()
    if op == _SOP_PROGRAM_PUBLIC:
        return ops_program_public_execute_py[E, KIND](session, fields, args)
    if op == _SOP_PROGRAM_INIT:
        var out = args[0]
        program_initialize_owner(out, session, args[1])
        return PythonObject(None)
    if op == _SOP_ENTER:
        return session
    if op == _SOP_FREE_PROGRAM:
        var open_id = py_item(fields, nk[].k__id)
        if not _nil(open_id) and not _nil(args[0]._id):
            var owner = ops_session[E, KIND](Int(py=open_id))
            owner[].program_free(Int(py=args[0]._id))
        var program = args[0]
        py_setattr(program, "_id", PythonObject(None))
        return PythonObject(None)
    if op == _SOP_CLOSE or op == _SOP_EXIT:
        if not _nil(py_item(fields, nk[].k__id)):
            var resource_scope: List[Int] = [ops_session_registry_token(), KIND]
            neural_descriptor_resource_close(py_item(fields, nk[].k__lifetime), resource_scope)
            py_setitem(fields, nk[].k__id, PythonObject(None))
        return PythonObject(False) if op == _SOP_EXIT else PythonObject(None)
    if op == _SOP_PROFILE_RESET:
        _ = ops_profile_reset_py()
        return PythonObject(None)
    if op == _SOP_PROFILE_REPORT:
        return ops_profile_report_py()
    var sid = py_item(fields, nk[].k__id)
    if _nil(sid):
        raise Error("ValueError: this session is closed")
    if op == _SOP_LOAD_PARTITIONED or op == _SOP_LOAD_DISTRIBUTED:
        return ops_owned_program_create_request_py(fields, 0, String(operation), args)
    if op == _SOP_PROGRAM_RUN:
        if len(args) != 4:
            raise Error("TypeError: Session.run requires a Program, arrays and two flags")
        var program = program_owner_fields(args[0])
        if py_item(program, nk[].k__s) is not session:
            raise Error("ValueError: Program belongs to another Session")
        if not program_numeric_scalar(args[2]) or not program_numeric_scalar(args[3]):
            raise Error("TypeError: Program flags require native boolean or numeric scalars")
        var pid = _integer(py_item(program, nk[].k__id), "program", 1)
        return ops_program_execute_py[E, KIND](sid, PythonObject(pid), sid, args[1], args[2], args[3], PythonObject(False))
    var s = ops_session[E, KIND](Int(py=sid))
    if op == _SOP_FIT or op == _SOP_FIT_IF:
        var program = args[0]
        if program._s is not session:
            raise Error("ValueError: training components must belong to the same session")
        var condition = 0
        var otherwise = 0
        if op == _SOP_FIT_IF:
            if args[1]._s is not session or args[2]._s is not session:
                raise Error("ValueError: training components must belong to the same session")
            condition = _int(args[1]._id, "condition", 1)
            otherwise = _int(args[2]._id, "otherwise", 1)
        var options = args[5]
        var stop = fit_stop_options(options)
        var result = ops_program_fit_native_py[E, KIND](
            sid, program._id, args[3], args[4], options[0], options[1], options[2], options[3],
            options[4], options[5], options[6], options[7], condition, otherwise, stop[0], stop[1])
        py_setattr(program, "loss_curve_", result[0])
        py_setattr(program, "n_iter_", result[1])
        if op == _SOP_FIT_IF:
            py_setattr(program, "branch_counts_", result[2])
        return program
    if op == _SOP_GRADIENT_REPORT:
        return gradient_report_py[E, KIND](Int(py=sid), _ints(args[0], "handle", 1), Bool(py=args[1]))
    if op == _SOP_EXECUTION_PROFILE_BEGIN:
        s[].ex.execution_profile_begin()
        return PythonObject(None)
    if op == _SOP_EXECUTION_PROFILE_END:
        return execution_profile_object(s[].ex.execution_profile_end())
    if op == _SOP_SAVE_CONTROL:
        var opt = 0 if _nil(args[5]) else _int(args[5], "optimizer", 1)
        return control_save_py[E, KIND](sid, args[0], args[2], args[3], args[4], args[1], opt, uint64_input(args[6], "seed"))
    if op == _SOP_LOAD_CONTROL:
        var bundle = control_load_py[E, KIND](sid, args[0])
        py_setitem(bundle, nk[].k_body, program_new_owner(session, py_item(bundle, nk[].k_body)))
        py_setitem(bundle, nk[].k_condition, program_new_owner(session, py_item(bundle, nk[].k_condition)))
        var else_program = py_item(bundle, nk[].k_otherwise)
        if else_program is not PythonObject(None):
            py_setitem(bundle, nk[].k_otherwise, program_new_owner(session, else_program))
        return bundle
    if op == _SOP_SAVE_PROGRAM:
        return ops_program_save_native_py[E, KIND](args[0], args[1])
    if op == _SOP_LOAD_WEIGHTS:
        return ops_weights_native_py[E, KIND](sid, args[0], args[1], args[2])
    if op == _SOP_TENSOR or op == _SOP_IDS or op == _SOP_STATE:
        # _SOP_STATE: args (x, ids); the upload, then declared state
        var as_ids = op == _SOP_IDS or (op == _SOP_STATE and Bool(py=args[1]))
        var input = program_input(args[0], as_ids)
        var handle: Int
        if as_ids:
            handle = s[].upload_ids(input.src, input.shape)
        elif op == _SOP_STATE:
            handle = s[].upload_nd(input.src, input.shape, False)
        else:
            handle = s[].upload_nd(input.src, input.shape, Bool(py=args[1]))
        if op == _SOP_STATE:
            s[].declare_state(handle)
        _ = input^  # keep all scalar-list storage alive across the upload
        _ = args  # retain original external array exporters through upload
        return PythonObject(handle)
    if op == _SOP_SHAPE:
        var sh = s[].shape_nd(_int(args[0], "handle", 1))
        var dims = List[Int]()
        for k in range(sh.ndim):  # small-loop(sh.ndim: tensor rank): shape metadata
            dims.append(sh.d[k])
        return _tuple_ints(dims)
    if op == _SOP_NUMPY or op == _SOP_GRAD:
        var handle = _int(args[0], "handle", 1)
        # TOMBSTONE: MOJOLEARN_SESSION_CALL_DESCRIPTOR (unmeasured B arm: every Session call through the descriptor; EXP:2395 RESTORED the direct typed call as default) deleted 2026-11-15 by lane/expdel2-08-base; code recoverable at 8fe6eaaa01.
        # Restore: git apply experiments/removed/MOJOLEARN_SESSION_CALL_DESCRIPTOR.patch; record in docs/TOMBSTONES.md.
        # RESTORED 2026-11-15 by lane/expdel2-restore-owed: owed B arm (owed-pre0867.json), under four weeks old; the deletion was premature.
        # TOMBSTONE: MOJOLEARN_SESSION_CALL_DESCRIPTOR (B arm loses: every same-image Session call through the descriptor, zero-filled read-back arrays; A/B abn9 NVIDIA L40S, ops-session-step 2.63 -> 2.93 ms 1.11, program-run 1.00 arm/base) deleted 2026-11-21 by lane/fx-verdicts7; code recoverable at 519388d23a.
        # Restore: git apply experiments/removed/MOJOLEARN_SESSION_CALL_DESCRIPTOR.fx-verdicts7.patch; record in docs/TOMBSTONES.md.
        # Lane fix-executor-session-rows (2026-11-13): the download writes
        # every element (rows x cols floats of this tensor), so the result
        # is made over UNINITIALIZED storage with its address in hand, as
        # 0.8.55's `np.empty` + `ops_download` and as Program.run's fixed
        # route (no zero fill of a bytearray, no `__array_interface__` dict
        # per read-back; ops-session-step reads eight arrays per round).
        var made = output_array_unfilled(s[].shape_nd(handle))
        var array = made[0]
        var dst = FP(unsafe_from_address=made[1])
        if op == _SOP_NUMPY:
            s[].download(handle, dst)
        else:
            s[].gradient_into(handle, dst)
        return array
    if op == _SOP_SCALAR:
        return PythonObject(Float64(s[].scalar(_int(args[0], "handle", 1))))
    if op == _SOP_CAPTURE:
        var pid = s[].capture(_ints(args[0], "input", 1), _ints(args[1], "output", 1))
        return program_new_owner(session, program_info(pid, s[].program(pid)))
    if op == _SOP_LOAD_PROGRAM:
        if s[].tape.is_recording():
            raise Error("ValueError: programs run in an inference session (Session(record=False))")
        return program_new_owner(session, ops_load_program_py[E, KIND](sid, native_path(args[0])))
    if op == _SOP_SAVE_STATE:
        var params = _ints(args[1], "handle", 1)
        if len(params) == 0:
            raise Error("ValueError: params must hold at least one tensor handle")
        var opt = 0 if _nil(args[2]) else _int(args[2], "optimizer", 1)
        if opt != 0:
            _ = s[]._opt_index(opt)
        var seed = _seed_words(args[3])
        var counters = Python.list()
        _ = py_call(counters.append, seed[0])
        _ = py_call(counters.append, seed[1])
        _ = py_call(counters.append, _int(args[4], "epoch", 0))
        _ = py_call(counters.append, _int(args[5], "step", 0))
        _ = ops_save_state_py[E, KIND](sid, native_path(args[0]), _tuple_ints(params), PythonObject(opt), counters)
        return PythonObject(None)
    if op == _SOP_LOAD_STATE:
        var opt = 0 if _nil(args[1]) else _int(args[1], "optimizer", 1)
        if opt != 0:
            _ = s[]._opt_index(opt)
        var grad = Bool(py=args[2]) and s[].recording()
        var res = ops_load_state_py[E, KIND](sid, native_path(args[0]), PythonObject(Int(grad)), PythonObject(opt))
        var result = native_loaded_module("builtins").dict()
        py_setitem(result, "params", res[0])
        py_setitem(result, "seed", res[3].__lshift__(32).__or__(res[2]))
        py_setitem(result, "epoch", res[4])
        py_setitem(result, "step", res[5])
        return result
    if op == _SOP_RESET:
        s[].reset()
        return PythonObject(None)
    if op == _SOP_FREE:
        s[].free(_int(args[0], "handle", 1))
        return PythonObject(None)
    if op == _SOP_BACKWARD:
        if Bool(py=args[2]):
            if Bool(py=args[1]):
                raise Error("ValueError: accumulate=True cannot be combined with create_graph=True")
            s[].backward_graph(_int(args[0], "loss", 1))
        else:
            s[].backward(_int(args[0], "loss", 1), Bool(py=args[1]))
        return PythonObject(None)
    if op == _SOP_GRAD_TENSOR:
        return PythonObject(s[].grad_handle(_int(args[0], "handle", 1)))
    if op == _SOP_VJP or op == _SOP_HVP or op == _SOP_JVP:
        var inputs = _ints(args[1], "input", 1)
        var out = List[Int]()
        if op == _SOP_VJP:
            out = s[].vjp(_int(args[0], "output", 1), inputs, _int(args[2], "v", 1), Bool(py=args[3]))
        else:
            var directions = _ints(args[2], "direction", 1)
            if len(directions) != len(inputs):
                raise Error("ValueError: one direction is required per input")
            if op == _SOP_JVP:
                return PythonObject(s[].jvp(_int(args[0], "output", 1), inputs, directions))
            out = s[].hvp(_int(args[0], "loss", 1), inputs, directions)
        var result = Python.list()
        for k in range(len(out)):  # driver-loop: over derivative result handles
            _ = py_call(result.append, out[k])
        return result
    if op == _SOP_ZERO_GRAD:
        s[].zero_grad()
        return PythonObject(None)
    if op == _SOP_OPTIMIZER:
        # UNPACK in providers/python/nn_pg_pyapi_door.mojo (nn-blocks-pg: shared with
        # param_group_optimizer; the same words)
        var unpacked = ops_optimizer_unpack(args)
        return PythonObject(s[].optimizer_with(unpacked[0], unpacked[1]))
    if op == _SOP_STEP or op == _SOP_SCHEDULER_STEP or op == _SOP_LEARNING_RATE or op == _SOP_OPTIMIZER_STATE or op == _SOP_LOAD_OPTIMIZER_STATE:
        var opt = _int(args[0], "optimizer", 1)
        var oi = s[]._opt_index(opt)
        if op == _SOP_STEP:
            s[].step(opt, _ints(args[1], "parameter", 1))
            return PythonObject(None)
        if op == _SOP_SCHEDULER_STEP:
            s[].scheduler_step(opt)
            return PythonObject(None)
        if op == _SOP_LEARNING_RATE:
            return PythonObject(Float64(s[].learning_rate(opt)))
        if op == _SOP_OPTIMIZER_STATE:
            var snapshot = s[].optimizer_state_value(opt, _int(args[1], "param", 1))
            # PACK the native snapshot directly into the documented mapping
            # (providers/python/nn_pg_pyapi_door.mojo, shared with optimizer_state_dict).
            return ops_optimizer_state_pack(snapshot)
        # UNPACK in providers/python/nn_pg_pyapi_door.mojo (shared with load_optimizer_state_dict)
        var request = ops_optimizer_state_unpack(args[2], s[].opts[oi].cfg.kind)
        s[].load_optimizer_state(opt, _int(args[1], "param", 1), request)
        return PythonObject(None)
    if op == _SOP_MAMBA_BLOCK or op == _SOP_ATTENTION_BLOCK:
        # Lane fx-ops-session-blocks (2026-11-18). UNPACK: mamba_block args
        # (weights, version, batch_size, dt_limit, requires_grad);
        # attention_block args (weights, batch_size, n_heads, n_kv_heads,
        # max_positions, window, requires_grad). `weights` is an exact tuple
        # or list of arrays, one per weight piece in the block's order; each
        # array is admitted and described as `tensor()` admits its input.
        var types = program_builtin_types()
        var weights_kind = py_type(args[0])
        if weights_kind is not types[].tuple_type and weights_kind is not types[].list_type:
            raise Error("TypeError: block weights require a native tuple or list of arrays")
        var pieces = len(args[0])
        var inputs = List[ProgramInput]()
        var srcs = List[CastIn]()
        for k in range(pieces):  # driver-loop: over the block's weight arrays (model structure), owners retained through the upload
            var item = program_input(args[0][k], False)
            srcs.append(item.src)
            inputs.append(item^)
        if pieces < 1:
            raise Error("ValueError: block weights must hold the block's weight arrays")
        var block_code = OPS_ATTN_STEP
        var block_ints = List[Int]()
        var dt_lo = Float32(0)
        var dt_hi = Float32(0)
        var trainable = False
        if op == _SOP_MAMBA_BLOCK:
            block_code = session_mamba_code(_int(args[1], "version", 1))
            block_ints = [_int(args[2], "batch_size", 1), srcs[0].count()]
            dt_lo = _float(args[3][0], "dt_limit")
            dt_hi = _float(args[3][1], "dt_limit")
            trainable = Bool(py=args[4])
        else:
            if pieces < SESSION_ATTENTION_MIN_PIECES:
                raise Error("ValueError: attention_block takes at least " + String(SESSION_ATTENTION_MIN_PIECES)
                            + " weight arrays, got " + String(pieces))
            var heads = _int(args[2], "n_heads", 1)
            block_ints = session_attention_ints(
                _int(args[1], "batch_size", 1), srcs[0].count(), heads,
                heads if _nil(args[3]) else _int(args[3], "n_kv_heads", 1),
                srcs[SESSION_ATTENTION_GATE].count(), _int(args[4], "max_positions", 1), _int(args[5], "window", 0))
            trainable = Bool(py=args[6])
            # The transformer estimator's options, args[7..18]: rope_theta,
            # rope_scaling (None or a dict with 'type' or 'rope_type',
            # 'factor' and, for llama3, 'low_freq_factor', 'high_freq_factor',
            # 'original_max_position_embeddings'), rope_dim, qkv_bias, o_bias,
            # norm, norm_eps, norm_bias, mlp, mlp_bias, qk_norm, attn_softcap.
            var scaling_kind = String("")
            var scaling_factor = Float32(1)
            var scaling_low = Float32(1)
            var scaling_high = Float32(4)
            var scaling_original = 8192
            if not _nil(args[8]):
                var scaling = args[8]
                if py_type(scaling) is not types[].dict_type:
                    raise Error("TypeError: rope_scaling must be None or a dict")
                scaling_kind = String(py_item(scaling, nk[].k_type)) if py_contains(scaling, nk[].k_type) else String(py_item(scaling, nk[].k_rope_type))
                scaling_factor = _float(py_item(scaling, nk[].k_factor), "rope_scaling factor")
                if scaling_kind == "llama3":
                    scaling_low = _float(py_item(scaling, nk[].k_low_freq_factor), "low_freq_factor")
                    scaling_high = _float(py_item(scaling, nk[].k_high_freq_factor), "high_freq_factor")
                    scaling_original = _int(py_item(scaling, nk[].k_original_max_position_embeddings),
                                            "original_max_position_embeddings", 1)
            var option_words = session_attention_option_words(
                block_ints[4], block_ints[6], String(args[12]), String(args[15]), _float(args[7], "rope_theta"),
                scaling_kind, scaling_factor, scaling_low, scaling_high, scaling_original,
                0 if _nil(args[9]) else _int(args[9], "rope_dim", 1), Bool(py=args[10]), Bool(py=args[11]),
                _float(args[13], "norm_eps"), Bool(py=args[14]), Bool(py=args[16]), Bool(py=args[17]),
                Float32(0) if _nil(args[18]) else _float(args[18], "attn_softcap"))
            _append(block_ints, option_words)
        var made = session_block_create[E, KIND](Int(py=sid), block_code, block_ints, srcs, trainable)
        _ = len(inputs)  # keep every scalar-list owner alive across the uploads
        _ = args  # retain the original array exporters through the uploads
        # PACK: the block as a plain dict of handles and its step arguments.
        var config = List[Int](capacity=len(block_ints) + 1)
        config.append(block_code)
        _append(config, block_ints)
        var block = Python.dict()
        py_setitem(block, "weights", made[0])
        py_setitem(block, "state", made[1])
        if made[2] != 0:
            py_setitem(block, "words", made[2])
        else:
            py_setitem(block, "words", PythonObject(None))
        py_setitem(block, "config", _tuple_ints(config))
        py_setitem(block, "dt_lo", Float64(dt_lo))
        py_setitem(block, "dt_hi", Float64(dt_hi))
        return block
    # nn-blocks-scan begin
    if op == _SOP_SCAN:
        # UNPACK: args (cell, xs, init_state, params, reverse, lengths, counter). The cell is a Program (or a loaded
        # one) of this session or of another session of this image; its description is read natively
        # (autograd/nn_scan_frontend.mojo). init_state is one handle or a sequence of handles; params None or a
        # sequence of handles; lengths None or a handle.
        var cell = program_owner_fields(args[0])
        var cell_session = py_item(cell, nk[].k__s)
        var cell_sid = Int(py=sid)
        if cell_session is not session:
            var cell_fields = ops_session_fields(cell_session)
            var cell_handle = py_item(cell_fields, nk[].k__id)
            if _nil(cell_handle):
                raise Error("ValueError: scan: the cell's Session is closed")
            var cell_scope: List[Int] = [ops_session_registry_token(), KIND]
            var admitted = neural_descriptor_resource_handle(py_item(cell_fields, nk[].k__lifetime), cell_scope)
            if admitted == 0 or admitted != _int(cell_handle, "cell session", 1):
                raise Error("ValueError: scan: the cell was captured in a Session on another device;"
                            + " capture it on this Session's device")
            cell_sid = admitted
        if _nil(py_item(cell, nk[].k__id)):
            raise Error("ValueError: scan: the cell Program was freed")
        var pid = _int(py_item(cell, nk[].k__id), "cell", 1)
        var one_state = _is_int(args[2])
        var states = _ints(args[2], "init_state", 1)
        var given = not _nil(args[3])
        var params = List[Int]()
        if given:
            params = _ints(args[3], "params", 1)
        var lengths = 0 if _nil(args[5]) else _int(args[5], "lengths", 1)
        if not program_numeric_scalar(args[4]):
            raise Error("TypeError: reverse must be a bool")
        var got = session_scan[E, KIND](Int(py=sid), cell_sid, pid, _int(args[1], "xs", 1), states, lengths,
                                        params, given, Bool(py=args[4]), _int(args[6], "counter", 0))
        # PACK: (ys, final_state); final_state one handle when init_state was one handle, else a tuple
        var finals = List[Int](capacity=len(got) - 1)
        for k in range(1, len(got)):  # driver-loop: over the final state handles
            finals.append(got[k])
        var pair = Python.list()
        _ = py_call(pair.append, PythonObject(got[0]))
        if one_state:
            _ = py_call(pair.append, PythonObject(got[1]))
        else:
            _ = py_call(pair.append, _tuple_ints(finals))
        return py_call(native_loaded_module("builtins").tuple, pair)
    # nn-blocks-scan end
    if op == _SOP_MAMBA_STEP or op == _SOP_ATTENTION_STEP:
        # UNPACK: args (x, block); the block is the exact dict a block maker
        # returned. One native step op (autograd/ops_block_frontend.mojo).
        var block = args[1]
        var builtin_types = program_builtin_types()
        if py_type(block) is not builtin_types[].dict_type:
            raise Error("TypeError: the block must be the dict mamba_block or attention_block returned")
        var config = _ints(py_item(block, nk[].k_config), "block", 0)
        var words = 0
        if not _nil(py_item(block, nk[].k_words)):
            words = _int(py_item(block, nk[].k_words), "words", 1)
        return PythonObject(session_block_step[E, KIND](
            Int(py=sid), _int(args[0], "handle", 1), _int(py_item(block, nk[].k_weights), "weights", 1),
            _int(py_item(block, nk[].k_state), "state", 1), words, config, _float(py_item(block, nk[].k_dt_lo), "dt_limit"),
            _float(py_item(block, nk[].k_dt_hi), "dt_limit"), op == _SOP_ATTENTION_STEP))
    if op == _SOP_LSTM or op == _SOP_GRU or op == _SOP_RNN:
        # args (x, weight_ih, weight_hh, bias_ih, bias_hh): tensor handles
        var cell = CELL_LSTM if op == _SOP_LSTM else (CELL_GRU if op == _SOP_GRU else CELL_RNN_TANH)
        return PythonObject(session_rnn[E, KIND](
            Int(py=sid), cell, _int(args[0], "input", 1), _int(args[1], "weight_ih", 1),
            _int(args[2], "weight_hh", 1), _int(args[3], "bias_ih", 1), _int(args[4], "bias_hh", 1)))
    if op == _SOP_LM_ROWS:
        # args (batch_size, eos_token_ids): an int, or a sequence of ints
        var eos = List[Int]()
        if not _nil(args[1]):
            eos = _ints(args[1], "eos_token_id", 0)
        return PythonObject(session_lm_rows[E, KIND](Int(py=sid), _int(args[0], "batch_size", 1), eos))
    if op == _SOP_LM_PICK:
        # args (logits, rows, pad_token_id, do_sample, temperature, top_k, top_p, seed)
        var seed = _seed_words(args[7])
        return PythonObject(session_lm_pick[E, KIND](
            Int(py=sid), _int(args[0], "logits", 1), _int(args[1], "rows", 1), _int(args[2], "pad_token_id", 0),
            Bool(py=args[3]), _int(args[5], "top_k", 0), seed[0], seed[1], _float(args[4], "temperature"),
            _float(args[6], "top_p")))
    if op == _SOP_LM_ALIVE:
        return PythonObject(session_lm_alive[E, KIND](
            Int(py=sid), _int(args[0], "ids", 1), _int(args[1], "rows", 1)))
    # nn-blocks-cbw begin: Session.custom_op / Session.custom (autograd/nn_cbw.mojo)
    if op == _SOP_CUSTOM_OP:
        # UNPACK: args (forward, backward, saved, outputs); the binding checks
        # run natively once here (OpsSession.cbw_define).
        var forward = _cbw_program[E, KIND](args[0], "forward")
        var backward = _cbw_program[E, KIND](args[1], "backward")
        var saved = List[Int]()
        if not _nil(args[2]):
            saved = _ints(args[2], "saved", 0)
        var n_out = len(forward.outs) if _nil(args[3]) else _int(args[3], "outputs", 1)
        return PythonObject(s[].cbw_define(forward^, backward^, saved, n_out))
    if op == _SOP_CUSTOM:
        # args (op, inputs): one handle per returned output (PACK: an int, or
        # a tuple for several)
        var outs = s[].cbw_call(_int(args[0], "op", 1), _ints(args[1], "input", 1))
        if len(outs) == 1:
            return PythonObject(outs[0])
        return _tuple_ints(outs)
    # nn-blocks-cbw end
    var code = 0
    var hs = List[Int]()
    var ints = List[Int]()
    var fs = List[Float32]()
    # One exclusive chain: each operation name matches at most one branch, so
    # after a match no later name is compared (was ~90 independent `if`s).
    if op == _SOP_COPY:
        code = 31
        hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_DETACH:
        code = 29
        hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_MATMUL:
        code = 1
        hs.append(_int(args[0], "handle", 1))
        hs.append(_int(args[1], "handle", 1))
    elif op == _SOP_ADD:
        code = 2
        hs.append(_int(args[0], "handle", 1))
        hs.append(_int(args[1], "handle", 1))
    elif op == _SOP_SUB:
        code = 3
        hs.append(_int(args[0], "handle", 1))
        hs.append(_int(args[1], "handle", 1))
    elif op == _SOP_MUL:
        code = 4
        hs.append(_int(args[0], "handle", 1))
        hs.append(_int(args[1], "handle", 1))
    elif op == _SOP_DIV:
        code = 5
        hs.append(_int(args[0], "handle", 1))
        hs.append(_int(args[1], "handle", 1))
    elif op == _SOP_RELU:
        code = 6
        hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_SIGMOID:
        code = 7
        hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_TANH:
        code = 8
        hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_EXP:
        code = 9
        hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_LOG:
        code = 10
        hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_SQRT:
        code = 11
        hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_ERF:
        code = 48
        hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_SOFTPLUS:
        code = 50
        hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_SILU:
        code = 51
        hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_HARD_SWISH:
        code = 55
        hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_ABS:
        code = 57
        hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_FLOOR:
        code = 58
        hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_CEIL:
        code = 59
        hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_ROUND:
        code = 60
        hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_SIN:
        code = 61
        hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_COS:
        code = 62
        hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_SIGN:
        code = 63
        hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_RECIPROCAL:
        code = 64
        hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_POW:
        code = 65
        hs.append(_int(args[0], "handle", 1))
        hs.append(_int(args[1], "handle", 1))
    elif op == _SOP_MINIMUM:
        code = 66
        hs.append(_int(args[0], "handle", 1))
        hs.append(_int(args[1], "handle", 1))
    elif op == _SOP_MAXIMUM:
        code = 67
        hs.append(_int(args[0], "handle", 1))
        hs.append(_int(args[1], "handle", 1))
    elif op == _SOP_WHERE:
        code = 68
        hs.append(_int(args[0], "handle", 1))
        hs.append(_int(args[1], "handle", 1))
        hs.append(_int(args[2], "handle", 1))
    elif op == _SOP_FULL or op == _SOP_EXPAND:
        code = 30 if op == _SOP_FULL else 71
        ints = _ints(args[0 if op == _SOP_FULL else 1], "shape", 1)
        if len(ints) < 1 or len(ints) > 8:
            raise Error("ValueError: shape must have one to eight dimensions")
        if op == _SOP_FULL:
            fs.append(_float(args[1], "value"))
        else:
            hs.append(_int(args[0], "handle", 1))
    elif op == _SOP_SCALE or op == _SOP_LEAKY_RELU or op == _SOP_ELU or op == _SOP_POW_SCALAR:
        code = 12 if op == _SOP_SCALE else (52 if op == _SOP_LEAKY_RELU else (53 if op == _SOP_ELU else 65))
        hs.append(_int(args[0], "handle", 1))
        fs.append(_float(args[1], "factor" if op == _SOP_SCALE else ("exponent" if op == _SOP_POW_SCALAR else "alpha")))
    elif op == _SOP_HARD_SIGMOID:
        code = 54
        hs.append(_int(args[0], "handle", 1))
        fs = [_float(args[1], "alpha"), _float(args[2], "beta")]
    elif op == _SOP_TRANSPOSE:
        hs.append(_int(args[0], "handle", 1))
        var a = _int(args[1], "a")
        var b = _int(args[2], "b")
        code = 13 if a == -2 and b == -1 else 34
        if code == 34:
            ints = [1, a, b]
    elif op == _SOP_PERMUTE:
        code = 34
        hs.append(_int(args[0], "handle", 1))
        ints.append(0)
        _append(ints, _axes(args[1]))
    elif op == _SOP_RESHAPE:
        hs.append(_int(args[0], "handle", 1))
        var dims = _ints(args[1], "shape")
        if len(dims) < 1 or len(dims) > 8:
            raise Error("ValueError: shape must have one to eight dimensions")
        code = 33
        if len(dims) == 2 and dims[0] != -1 and dims[1] != -1:
            code = 14
            if dims[0] < 1 or dims[1] < 1:
                raise Error("ValueError: rows and cols must be positive")
        else:
            ints.append(0)
        _append(ints, dims)
    elif op == _SOP_FLATTEN or op == _SOP_SQUEEZE or op == _SOP_UNSQUEEZE:
        code = 33
        hs.append(_int(args[0], "handle", 1))
        if op == _SOP_FLATTEN:
            ints = [1, _int(args[1], "start"), _int(args[2], "end")]
        elif op == _SOP_SQUEEZE:
            ints.append(2)
            _append(ints, _axes(args[1]))
        else:
            ints = [3, _int(args[1], "axis")]
    elif op == _SOP_SUM or op == _SOP_MEAN or op == _SOP_MAX or op == _SOP_ANY or op == _SOP_ALL:
        code = 35 if op == _SOP_SUM else (36 if op == _SOP_MEAN else (37 if op == _SOP_MAX else 74))
        hs.append(_int(args[0], "handle", 1))
        if code == 74:
            ints.append(1 if op == _SOP_ALL else 0)
        ints.append(_keep(args[2]))
        _append(ints, _axes(args[1]))
    elif op == _SOP_SOFTMAX or op == _SOP_LOG_SOFTMAX:
        hs.append(_int(args[0], "handle", 1))
        var axis = _int(args[1], "axis")
        code = 17 if op == _SOP_SOFTMAX else 18
        if axis != -1:
            code = 42 if op == _SOP_SOFTMAX else 43
            ints.append(axis)
    elif op == _SOP_CONCAT:
        code = 39
        hs = _ints(args[0], "handle", 1)
        if len(hs) == 0:
            raise Error("ValueError: concat needs at least one tensor")
        ints.append(_int(args[1], "axis"))
    elif op == _SOP_SPLIT:
        hs.append(_int(args[0], "handle", 1))
        var axis = _int(args[2], "axis")
        var count: Int
        var tail = List[Int]()
        if _is_int(args[1]):
            count = _int(args[1], "sections", 1)
            tail = [1, count]
        else:
            var sizes = _ints(args[1], "sizes", 1)
            count = len(sizes)
            if count == 0:
                raise Error("ValueError: sizes must list at least one length")
            tail.append(0)
            _append(tail, sizes)
        var views = session_split[E, KIND](Int(py=sid), hs, axis, tail, count, fs)
        var out = Python.list()
        for k in range(len(views)):  # small-loop(views: one handle per requested section): result packing, no data
            _ = py_call(out.append, views[k])
        return out
    elif op == _SOP_EMBEDDING or op == _SOP_TAKE:
        code = 32 if op == _SOP_EMBEDDING else 41
        hs.append(_int(args[0], "handle", 1))
        var option = -1 if op == _SOP_EMBEDDING and _nil(args[2]) else _int(args[2], "padding_idx" if op == _SOP_EMBEDDING else "axis")
        if op == _SOP_EMBEDDING and not _nil(args[2]) and option < 0:
            raise Error("ValueError: padding_idx must be >= 0")
        if _is_int(args[1]):
            hs.append(_int(args[1], "handle", 1))
            ints.append(option)
        else:
            var input = program_input(args[1], True)
            # Native CastIn stores signedness/width, so reuse it rather than
            # asking the external object for dtype metadata a second time.
            var dtype_num = 9
            if input.src.kind == 0:  # CAST_KIND_UNSIGNED
                dtype_num = 2 if input.src.itemsize == 1 else (4 if input.src.itemsize == 2 else (6 if input.src.itemsize == 4 else 10))
            else:
                dtype_num = 1 if input.src.itemsize == 1 else (3 if input.src.itemsize == 2 else (5 if input.src.itemsize == 4 else 9))
            ints = [input.src.addr, dtype_num, input.src.itemsize, input.src.rows,
                    input.src.cols, input.src.s0, input.src.s1, option]
            var out = s[].op(code, hs, ints, fs)
            _ = input^  # scalar sequence owner remains alive through upload
            return PythonObject(out)
    elif op == _SOP_CONV2D or op == _SOP_LINEAR:
        code = 44 if op == _SOP_CONV2D else 19
        hs = [_int(args[0], "input", 1), _int(args[1], "weight", 1)]
        if not _nil(args[2]):
            hs.append(_int(args[2], "bias", 1))
        if op == _SOP_CONV2D:
            ints = [_int(args[3], "stride", 1), _int(args[4], "padding", 0), _int(args[5], "groups", 1)]
        else:
            var act = "none" if _nil(args[3]) else String(args[3])
            var activation = 0
            if act == "relu":
                activation = 1
            elif act == "sigmoid":
                activation = 2
            elif act == "tanh":
                activation = 3
            elif act == "softmax":
                activation = 4
            elif act != "none":
                raise Error("ValueError: activation must be None, 'relu', 'sigmoid', 'tanh' or 'softmax'")
            ints.append(activation)
    elif op == _SOP_MAX_POOL2D or op == _SOP_AVG_POOL2D:
        code = 45 if op == _SOP_MAX_POOL2D else 46
        hs.append(_int(args[0], "handle", 1))
        var kernel = _int(args[1], "kernel_size", 1)
        ints = [kernel, kernel if _nil(args[2]) else _int(args[2], "stride", 1)]
    elif op == _SOP_GELU:
        code = 49
        hs.append(_int(args[0], "handle", 1))
        var approx = String(args[1])
        if approx != "none" and approx != "tanh":
            raise Error("ValueError: approximate must be 'none' or 'tanh'")
        ints.append(1 if approx == "tanh" else 0)
    elif op == _SOP_CLIP:
        code = 56
        hs.append(_int(args[0], "handle", 1))
        fs.append(clip_low_unbounded() if _nil(args[1]) else _float(args[1], "min"))
        fs.append(clip_high_unbounded() if _nil(args[2]) else _float(args[2], "max"))
    elif op == _SOP_EQUAL or op == _SOP_NOT_EQUAL or op == _SOP_GREATER or op == _SOP_GREATER_EQUAL or op == _SOP_LESS or op == _SOP_LESS_EQUAL:
        code = 69
        hs = [_int(args[0], "handle", 1), _int(args[1], "handle", 1)]
        var comparison = 0
        if op == _SOP_NOT_EQUAL:
            comparison = 5
        elif op == _SOP_GREATER:
            comparison = 1
        elif op == _SOP_LESS:
            comparison = 2
        elif op == _SOP_GREATER_EQUAL:
            comparison = 3
        elif op == _SOP_LESS_EQUAL:
            comparison = 4
        ints.append(comparison)
    elif op == _SOP_LOGICAL_NOT or op == _SOP_LOGICAL_AND or op == _SOP_LOGICAL_OR:
        code = 70
        hs.append(_int(args[0], "handle", 1))
        if op != _SOP_LOGICAL_NOT:
            hs.append(_int(args[1], "handle", 1))
        ints.append(0 if op == _SOP_LOGICAL_NOT else (1 if op == _SOP_LOGICAL_AND else 2))
    elif op == _SOP_ARGMAX:
        code = 72
        hs.append(_int(args[0], "handle", 1))
        ints = [_int(args[1], "axis"), Int(Bool(py=args[2])), Int(not Bool(py=args[3]))]
    elif op == _SOP_PAD:
        code = 73
        hs.append(_int(args[0], "handle", 1))
        ints = _ints(args[1], "pads")
        if len(ints) < 1 or len(ints) > 16:
            raise Error("ValueError: pads must list two counts for each of one to eight axes")
        fs.append(_float(args[2], "value"))
    elif op == _SOP_CUMSUM:
        code = 75
        hs.append(_int(args[0], "handle", 1))
        ints = [_int(args[1], "axis"), Int(Bool(py=args[2])), Int(Bool(py=args[3]))]
    elif op == _SOP_INDEX:
        code = 76
        hs.append(_int(args[0], "handle", 1))
        var indices = _ints(args[1], "indices", 1)
        if len(indices) < 1 or len(indices) > 8:
            raise Error("ValueError: index takes one to eight index tensors")
        _append(hs, indices)
        if _nil(args[2]):
            for k in range(len(indices)):  # small-loop(indices: at most eight index axes): metadata only
                ints.append(k)
        else:
            ints = _ints(args[2], "axes")
        if len(ints) != len(indices):
            raise Error("ValueError: index takes one axis for each index tensor")
    elif op == _SOP_LAYER_NORM:
        code = 20
        hs = [_int(args[0], "input", 1), _int(args[1], "weight", 1), _int(args[2], "bias", 1)]
        var eps = _float(args[3], "eps")
        if not eps > 0:
            raise Error("ValueError: eps must be positive")
        fs.append(eps)
    elif op == _SOP_TRANSFORMER_SOFTMAX:
        var b = native_loaded_module("builtins")
        if not (py_type(args[0]) is b.int):
            raise Error("TypeError: transformer handles must be exact native integers")
        code = 80
        hs = [_int(args[0], "input", 1)]
    elif op == _SOP_TRANSFORMER_LINEAR:
        var b = native_loaded_module("builtins")
        for j in range(2):  # small-loop(2: projection handles): exact native scalar admission
            if not (py_type(args[j]) is b.int):
                raise Error("TypeError: transformer handles must be exact native integers")
        code = 79
        hs = [_int(args[0], "input", 1), _int(args[1], "weight", 1)]
    elif op == _SOP_RMS_NORM:
        var b = native_loaded_module("builtins")
        for j in range(2):  # small-loop(2: RMSNorm handles): exact native scalar admission
            if not (py_type(args[j]) is b.int):
                raise Error("TypeError: transformer handles must be exact native integers")
        var kind = py_type(args[2])
        if not (kind is b.float or kind is b.int):
            raise Error("TypeError: RMSNorm epsilon must be an exact native scalar")
        code = 77
        hs = [_int(args[0], "input", 1), _int(args[1], "weight", 1)]
        fs.append(Float32(py=args[2]))
    elif op == _SOP_ROTARY:
        var b = native_loaded_module("builtins")
        for j in range(3):  # small-loop(3: rotary handle/position metadata): exact native scalar admission
            if not (py_type(args[j]) is b.int):
                raise Error("TypeError: rotary handles and positions must be exact native integers")
        var kind = py_type(args[3])
        if not (kind is b.float or kind is b.int):
            raise Error("TypeError: rotary theta must be an exact native scalar")
        code = 78
        hs = [_int(args[0], "input", 1)]
        ints = [_int(args[1], "head_dim", 1), _int(args[2], "offset", 0)]
        fs.append(_float(args[3], "theta"))
    elif op == _SOP_DROPOUT:
        code = 21
        hs.append(_int(args[0], "handle", 1))
        var prob = _float(args[1], "p")
        if not (prob >= 0 and prob < 1):
            raise Error("ValueError: p must be in [0, 1)")
        fs.append(prob)
        ints = _seed_words(args[2])
        ints.append(_int(args[3], "counter", 0))
    elif op == _SOP_PERMUTATION:
        code = 23
        ints.append(_int(args[0], "n", 1))
        _append(ints, _seed_words(args[1]))
        ints.append(_int(args[2], "epoch", 0))
    elif op == _SOP_GATHER_ROWS:
        code = 22
        hs = [_int(args[0], "handle", 1), _int(args[1], "handle", 1)]
        ints = [_int(args[2], "start", 0), 0 if _nil(args[3]) else _int(args[3], "count", 1)]
    elif op == _SOP_MSE or op == _SOP_MAE or op == _SOP_HUBER or op == _SOP_BCE_WITH_LOGITS or op == _SOP_CROSS_ENTROPY:
        code = 24 if op == _SOP_MSE else (25 if op == _SOP_MAE else (26 if op == _SOP_HUBER else (27 if op == _SOP_BCE_WITH_LOGITS else 28)))
        hs = [_int(args[0], "handle", 1), _int(args[1], "handle", 1)]
        var reduction = String(args[3 if op == _SOP_HUBER else 2])
        if reduction != "mean" and reduction != "sum":
            raise Error("ValueError: reduction must be 'mean' or 'sum'")
        ints.append(0 if reduction == "mean" else 1)
        if op == _SOP_HUBER:
            var delta = _float(args[2], "delta")
            if not delta > 0:
                raise Error("ValueError: delta must be positive")
            fs.append(delta)
    elif op == _SOP_SMALL_MLP:
        # lane fx-ops-session-blocks: OPS_SMALL_MLP (autograd/small_mlp_op.mojo)
        code = 88
        hs = [_int(args[0], "input", 1), _int(args[1], "weight1", 1), _int(args[2], "bias1", 1),
              _int(args[3], "weight2", 1), _int(args[4], "bias2", 1)]
    elif op == _SOP_CONV_BLOCK:
        # OPS_CNN_CONV_BLOCK (autograd/cnn_block_op.mojo): args (x, weight,
        # bias, padding, pool_size or None); ints [pad, pool 0 / 1, pool size]
        code = 90
        hs = [_int(args[0], "input", 1), _int(args[1], "weight", 1), _int(args[2], "bias", 1)]
        ints = [_int(args[3], "padding", 0), 0 if _nil(args[4]) else 1,
                1 if _nil(args[4]) else _int(args[4], "pool_size", 1)]
    elif op == _SOP_CONV_LINEAR:
        # OPS_CNN_LINEAR: args (x, weight (outputs, inputs), bias)
        code = 91
        hs = [_int(args[0], "input", 1), _int(args[1], "weight", 1), _int(args[2], "bias", 1)]
    elif op == _SOP_LM_EMBEDDING:
        # OPS_LM_EMBEDDING (autograd/lm_embedding_op.mojo): a table and an
        # ids() tensor; ints [padding_idx, -1 for none] as `embedding`
        code = 93
        hs = [_int(args[0], "table", 1), _int(args[1], "ids", 1)]
        if not _nil(args[2]) and _int(args[2], "padding_idx") < 0:
            raise Error("ValueError: padding_idx must be >= 0")
        ints.append(-1 if _nil(args[2]) else _int(args[2], "padding_idx", 0))
    elif op == _SOP_LM_HEAD_LOSS:
        # lane fx-ops-session-blocks: OPS_LM_HEAD_V2 (autograd/lm_head_v2_op.mojo)
        code = 87
        hs = [_int(args[0], "hidden", 1), _int(args[1], "weight", 1), _int(args[2], "codes", 1)]
    elif op == _SOP_ACCURACY or op == _SOP_LOG_LOSS or op == _SOP_ROC_AUC:
        var metric = 1 if op == _SOP_ACCURACY else (2 if op == _SOP_LOG_LOSS else 3)
        hs = [_int(args[0], "handle", 1), _int(args[1], "handle", 1)]
        return PythonObject(Float64(s[].metric(metric, hs)))
    # nn-blocks-qkv begin
    elif op == _SOP_ATTENTION_QKV:
        # OPS_ATTENTION_QKV (autograd/nn_qkv.mojo): args (q, k, v, mask or None, causal, scale or None);
        # ints [causal, scale given, mask kind (0 none, 1 ids words: boolean, 2 float values: additive)],
        # floats [scale]. The kind is read once here from the mask tensor and recorded with the call, so a
        # captured program replays the same reading.
        code = 122
        hs = [_int(args[0], "q", 1), _int(args[1], "k", 1), _int(args[2], "v", 1)]
        var mask_kind = 0
        if not _nil(args[3]):
            var mask_handle = _int(args[3], "mask", 1)
            hs.append(mask_handle)
            mask_kind = 1 if s[].int_words[s[]._slot(mask_handle)] else 2
        ints = [Int(Bool(py=args[4])), 0 if _nil(args[5]) else 1, mask_kind]
        fs.append(Float32(0) if _nil(args[5]) else _float(args[5], "scale"))
    # nn-blocks-qkv end
    # nn-blocks-scatter begin
    elif op == _SOP_SCATTER_ADD:
        # args (x, dim, index, src): torch.Tensor.scatter_add's order; index an ids() tensor
        code = OPS_SCATTER_ADD
        if not _is_int(args[2]):
            raise Error("TypeError: scatter_add: index must be an integer tensor made with Session.ids")
        hs = [_int(args[0], "x", 1), _int(args[2], "index", 1), _int(args[3], "src", 1)]
        ints.append(_int(args[1], "dim"))
    elif op == _SOP_SEGMENT_SUM or op == _SOP_SEGMENT_MEAN or op == _SOP_SEGMENT_MAX:
        # args (data, segment_ids, num_segments); segment_ids an ids() tensor
        code = OPS_SEGMENT_SUM + (0 if op == _SOP_SEGMENT_SUM else (1 if op == _SOP_SEGMENT_MEAN else 2))
        if not _is_int(args[1]):
            raise Error("TypeError: segment_ids must be an integer tensor made with Session.ids")
        hs = [_int(args[0], "data", 1), _int(args[1], "segment_ids", 1)]
        ints.append(_int(args[2], "num_segments"))
    # nn-blocks-scatter end
    # nn-blocks-pg begin
    if op >= _SOP_PG_FIRST and op <= _SOP_PG_LAST:
        return pg_session_call_py[E](s, op, args)
    # nn-blocks-pg end
    if code == 0:
        raise Error("unknown native session operation: " + String(operation))
    return PythonObject(s[].op(code, hs, ints, fs))
