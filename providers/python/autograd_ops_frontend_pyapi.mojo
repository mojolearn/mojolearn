# SPDX-License-Identifier: Apache-2.0
# Native normalization for the existing Session API. Public wrappers hand
# over objects unchanged; this boundary owns descriptors, options and outputs.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_setattr, py_setitem, py_type, py_contains
from providers.python.autograd_python_errors_pyapi import graph_int
from sequence.exec_trait import Exec
from sequence.ops import FP
from autograd.ops_api import ops_session, ops_session_registry_token
from autograd.optimizer_requests import (
    OpsOptimizerRequest, OpsOptimizerRestoreRequest, ops_optimizer_codes,
)
from providers.python.autograd_program_pyapi import program_input, _integer, ops_program_fit_native_py, ops_program_execute_py, fit_stop_options
from autograd.ops_frontend_impl import session_split, clip_low_unbounded, clip_high_unbounded
from providers.python.autograd_program_arrays_pyapi import output_array, is_integer_input, uint64_input, program_admit_value, program_numeric_scalar
from providers.python.autograd_pyapi import program_info, ops_profile_reset_py, ops_profile_report_py
from providers.python.autograd_model_checkpoint_pyapi import ops_load_program_py, ops_save_state_py, ops_load_state_py
from providers.python.autograd_program_save_pyapi import ops_program_save_native_py
from providers.python.autograd_program_owner_pyapi import program_initialize_owner, program_new_owner, program_owner_fields
from providers.python.autograd_owned_program_public_pyapi import ops_owned_program_create_request_py
from providers.python.autograd_program_public_dispatch_pyapi import ops_program_public_execute_py
from bindings.native_json import native_path
from providers.python.autograd_weight_frontend_pyapi import ops_weights_native_py
from providers.python.autograd_session_frontend_pyapi import ops_session_fields
from providers.python.neural_descriptor_pyapi import neural_descriptor_resource_handle, neural_descriptor_resource_close
from providers.python.autograd_execution_profile_pyapi import execution_profile_object
from providers.python.autograd_control_checkpoint_pyapi import control_save_py, control_load_py
from providers.python.autograd_gradient_diagnostics_pyapi import gradient_report_py
from core.shared_slot import SharedSlot


# Session operation codes (row N089, lesson 12): the operation name is looked up
# once per call in one native table built at first use, and the dispatch below
# compares integers. The codes are private to this file (the public entry still
# takes the operation name); 0 = unknown.  Dispatch only, no arithmetic.
comptime _SOP_PROGRAM_PUBLIC = 1
comptime _SOP_PROGRAM_INIT = 2
comptime _SOP_ENTER = 3
comptime _SOP_FREE_PROGRAM = 4
comptime _SOP_CLOSE = 5
comptime _SOP_EXIT = 6
comptime _SOP_PROFILE_RESET = 7
comptime _SOP_PROFILE_REPORT = 8
comptime _SOP_LOAD_PARTITIONED = 9
comptime _SOP_LOAD_DISTRIBUTED = 10
comptime _SOP_PROGRAM_RUN = 11
comptime _SOP_FIT = 12
comptime _SOP_FIT_IF = 13
comptime _SOP_GRADIENT_REPORT = 14
comptime _SOP_EXECUTION_PROFILE_BEGIN = 15
comptime _SOP_EXECUTION_PROFILE_END = 16
comptime _SOP_SAVE_CONTROL = 17
comptime _SOP_LOAD_CONTROL = 18
comptime _SOP_SAVE_PROGRAM = 19
comptime _SOP_LOAD_WEIGHTS = 20
comptime _SOP_TENSOR = 21
comptime _SOP_IDS = 22
comptime _SOP_SHAPE = 23
comptime _SOP_NUMPY = 24
comptime _SOP_GRAD = 25
comptime _SOP_SCALAR = 26
comptime _SOP_CAPTURE = 27
comptime _SOP_LOAD_PROGRAM = 28
comptime _SOP_SAVE_STATE = 29
comptime _SOP_LOAD_STATE = 30
comptime _SOP_RESET = 31
comptime _SOP_FREE = 32
comptime _SOP_BACKWARD = 33
comptime _SOP_GRAD_TENSOR = 34
comptime _SOP_VJP = 35
comptime _SOP_HVP = 36
comptime _SOP_JVP = 37
comptime _SOP_ZERO_GRAD = 38
comptime _SOP_OPTIMIZER = 39
comptime _SOP_STEP = 40
comptime _SOP_SCHEDULER_STEP = 41
comptime _SOP_LEARNING_RATE = 42
comptime _SOP_OPTIMIZER_STATE = 43
comptime _SOP_LOAD_OPTIMIZER_STATE = 44
comptime _SOP_COPY = 45
comptime _SOP_DETACH = 46
comptime _SOP_MATMUL = 47
comptime _SOP_ADD = 48
comptime _SOP_SUB = 49
comptime _SOP_MUL = 50
comptime _SOP_DIV = 51
comptime _SOP_RELU = 52
comptime _SOP_SIGMOID = 53
comptime _SOP_TANH = 54
comptime _SOP_EXP = 55
comptime _SOP_LOG = 56
comptime _SOP_SQRT = 57
comptime _SOP_ERF = 58
comptime _SOP_SOFTPLUS = 59
comptime _SOP_SILU = 60
comptime _SOP_HARD_SWISH = 61
comptime _SOP_ABS = 62
comptime _SOP_FLOOR = 63
comptime _SOP_CEIL = 64
comptime _SOP_ROUND = 65
comptime _SOP_SIN = 66
comptime _SOP_COS = 67
comptime _SOP_SIGN = 68
comptime _SOP_RECIPROCAL = 69
comptime _SOP_POW = 70
comptime _SOP_MINIMUM = 71
comptime _SOP_MAXIMUM = 72
comptime _SOP_WHERE = 73
comptime _SOP_FULL = 74
comptime _SOP_EXPAND = 75
comptime _SOP_SCALE = 76
comptime _SOP_LEAKY_RELU = 77
comptime _SOP_ELU = 78
comptime _SOP_POW_SCALAR = 79
comptime _SOP_HARD_SIGMOID = 80
comptime _SOP_TRANSPOSE = 81
comptime _SOP_PERMUTE = 82
comptime _SOP_RESHAPE = 83
comptime _SOP_FLATTEN = 84
comptime _SOP_SQUEEZE = 85
comptime _SOP_UNSQUEEZE = 86
comptime _SOP_SUM = 87
comptime _SOP_MEAN = 88
comptime _SOP_MAX = 89
comptime _SOP_ANY = 90
comptime _SOP_ALL = 91
comptime _SOP_SOFTMAX = 92
comptime _SOP_LOG_SOFTMAX = 93
comptime _SOP_CONCAT = 94
comptime _SOP_SPLIT = 95
comptime _SOP_EMBEDDING = 96
comptime _SOP_TAKE = 97
comptime _SOP_CONV2D = 98
comptime _SOP_LINEAR = 99
comptime _SOP_MAX_POOL2D = 100
comptime _SOP_AVG_POOL2D = 101
comptime _SOP_GELU = 102
comptime _SOP_CLIP = 103
comptime _SOP_EQUAL = 104
comptime _SOP_NOT_EQUAL = 105
comptime _SOP_GREATER = 106
comptime _SOP_GREATER_EQUAL = 107
comptime _SOP_LESS = 108
comptime _SOP_LESS_EQUAL = 109
comptime _SOP_LOGICAL_NOT = 110
comptime _SOP_LOGICAL_AND = 111
comptime _SOP_LOGICAL_OR = 112
comptime _SOP_ARGMAX = 113
comptime _SOP_PAD = 114
comptime _SOP_CUMSUM = 115
comptime _SOP_INDEX = 116
comptime _SOP_LAYER_NORM = 117
comptime _SOP_TRANSFORMER_SOFTMAX = 118
comptime _SOP_TRANSFORMER_LINEAR = 119
comptime _SOP_RMS_NORM = 120
comptime _SOP_ROTARY = 121
comptime _SOP_DROPOUT = 122
comptime _SOP_PERMUTATION = 123
comptime _SOP_GATHER_ROWS = 124
comptime _SOP_MSE = 125
comptime _SOP_MAE = 126
comptime _SOP_HUBER = 127
comptime _SOP_BCE_WITH_LOGITS = 128
comptime _SOP_CROSS_ENTROPY = 129
comptime _SOP_ACCURACY = 130
comptime _SOP_LOG_LOSS = 131
comptime _SOP_ROC_AUC = 132
# lane x1: an uploaded tensor declared carried state (OpsSession.declare_state)
comptime _SOP_STATE = 133


struct _SessionOpCodes(Defaultable, Movable):
    var codes: Dict[String, Int]

    def __init__(out self):
        self.codes = Dict[String, Int]()
        self.codes["program_public"] = _SOP_PROGRAM_PUBLIC
        self.codes["program_init"] = _SOP_PROGRAM_INIT
        self.codes["enter"] = _SOP_ENTER
        self.codes["free_program"] = _SOP_FREE_PROGRAM
        self.codes["close"] = _SOP_CLOSE
        self.codes["exit"] = _SOP_EXIT
        self.codes["profile_reset"] = _SOP_PROFILE_RESET
        self.codes["profile_report"] = _SOP_PROFILE_REPORT
        self.codes["load_partitioned"] = _SOP_LOAD_PARTITIONED
        self.codes["load_distributed"] = _SOP_LOAD_DISTRIBUTED
        self.codes["program_run"] = _SOP_PROGRAM_RUN
        self.codes["fit"] = _SOP_FIT
        self.codes["fit_if"] = _SOP_FIT_IF
        self.codes["gradient_report"] = _SOP_GRADIENT_REPORT
        self.codes["execution_profile_begin"] = _SOP_EXECUTION_PROFILE_BEGIN
        self.codes["execution_profile_end"] = _SOP_EXECUTION_PROFILE_END
        self.codes["save_control"] = _SOP_SAVE_CONTROL
        self.codes["load_control"] = _SOP_LOAD_CONTROL
        self.codes["save_program"] = _SOP_SAVE_PROGRAM
        self.codes["load_weights"] = _SOP_LOAD_WEIGHTS
        self.codes["tensor"] = _SOP_TENSOR
        self.codes["ids"] = _SOP_IDS
        self.codes["shape"] = _SOP_SHAPE
        self.codes["numpy"] = _SOP_NUMPY
        self.codes["grad"] = _SOP_GRAD
        self.codes["scalar"] = _SOP_SCALAR
        self.codes["capture"] = _SOP_CAPTURE
        self.codes["load_program"] = _SOP_LOAD_PROGRAM
        self.codes["save_state"] = _SOP_SAVE_STATE
        self.codes["load_state"] = _SOP_LOAD_STATE
        self.codes["reset"] = _SOP_RESET
        self.codes["free"] = _SOP_FREE
        self.codes["backward"] = _SOP_BACKWARD
        self.codes["grad_tensor"] = _SOP_GRAD_TENSOR
        self.codes["vjp"] = _SOP_VJP
        self.codes["hvp"] = _SOP_HVP
        self.codes["jvp"] = _SOP_JVP
        self.codes["zero_grad"] = _SOP_ZERO_GRAD
        self.codes["optimizer"] = _SOP_OPTIMIZER
        self.codes["step"] = _SOP_STEP
        self.codes["scheduler_step"] = _SOP_SCHEDULER_STEP
        self.codes["learning_rate"] = _SOP_LEARNING_RATE
        self.codes["optimizer_state"] = _SOP_OPTIMIZER_STATE
        self.codes["load_optimizer_state"] = _SOP_LOAD_OPTIMIZER_STATE
        self.codes["copy"] = _SOP_COPY
        self.codes["detach"] = _SOP_DETACH
        self.codes["matmul"] = _SOP_MATMUL
        self.codes["add"] = _SOP_ADD
        self.codes["sub"] = _SOP_SUB
        self.codes["mul"] = _SOP_MUL
        self.codes["div"] = _SOP_DIV
        self.codes["relu"] = _SOP_RELU
        self.codes["sigmoid"] = _SOP_SIGMOID
        self.codes["tanh"] = _SOP_TANH
        self.codes["exp"] = _SOP_EXP
        self.codes["log"] = _SOP_LOG
        self.codes["sqrt"] = _SOP_SQRT
        self.codes["erf"] = _SOP_ERF
        self.codes["softplus"] = _SOP_SOFTPLUS
        self.codes["silu"] = _SOP_SILU
        self.codes["hard_swish"] = _SOP_HARD_SWISH
        self.codes["abs"] = _SOP_ABS
        self.codes["floor"] = _SOP_FLOOR
        self.codes["ceil"] = _SOP_CEIL
        self.codes["round"] = _SOP_ROUND
        self.codes["sin"] = _SOP_SIN
        self.codes["cos"] = _SOP_COS
        self.codes["sign"] = _SOP_SIGN
        self.codes["reciprocal"] = _SOP_RECIPROCAL
        self.codes["pow"] = _SOP_POW
        self.codes["minimum"] = _SOP_MINIMUM
        self.codes["maximum"] = _SOP_MAXIMUM
        self.codes["where"] = _SOP_WHERE
        self.codes["full"] = _SOP_FULL
        self.codes["expand"] = _SOP_EXPAND
        self.codes["scale"] = _SOP_SCALE
        self.codes["leaky_relu"] = _SOP_LEAKY_RELU
        self.codes["elu"] = _SOP_ELU
        self.codes["pow_scalar"] = _SOP_POW_SCALAR
        self.codes["hard_sigmoid"] = _SOP_HARD_SIGMOID
        self.codes["transpose"] = _SOP_TRANSPOSE
        self.codes["permute"] = _SOP_PERMUTE
        self.codes["reshape"] = _SOP_RESHAPE
        self.codes["flatten"] = _SOP_FLATTEN
        self.codes["squeeze"] = _SOP_SQUEEZE
        self.codes["unsqueeze"] = _SOP_UNSQUEEZE
        self.codes["sum"] = _SOP_SUM
        self.codes["mean"] = _SOP_MEAN
        self.codes["max"] = _SOP_MAX
        self.codes["any"] = _SOP_ANY
        self.codes["all"] = _SOP_ALL
        self.codes["softmax"] = _SOP_SOFTMAX
        self.codes["log_softmax"] = _SOP_LOG_SOFTMAX
        self.codes["concat"] = _SOP_CONCAT
        self.codes["split"] = _SOP_SPLIT
        self.codes["embedding"] = _SOP_EMBEDDING
        self.codes["take"] = _SOP_TAKE
        self.codes["conv2d"] = _SOP_CONV2D
        self.codes["linear"] = _SOP_LINEAR
        self.codes["max_pool2d"] = _SOP_MAX_POOL2D
        self.codes["avg_pool2d"] = _SOP_AVG_POOL2D
        self.codes["gelu"] = _SOP_GELU
        self.codes["clip"] = _SOP_CLIP
        self.codes["equal"] = _SOP_EQUAL
        self.codes["not_equal"] = _SOP_NOT_EQUAL
        self.codes["greater"] = _SOP_GREATER
        self.codes["greater_equal"] = _SOP_GREATER_EQUAL
        self.codes["less"] = _SOP_LESS
        self.codes["less_equal"] = _SOP_LESS_EQUAL
        self.codes["logical_not"] = _SOP_LOGICAL_NOT
        self.codes["logical_and"] = _SOP_LOGICAL_AND
        self.codes["logical_or"] = _SOP_LOGICAL_OR
        self.codes["argmax"] = _SOP_ARGMAX
        self.codes["pad"] = _SOP_PAD
        self.codes["cumsum"] = _SOP_CUMSUM
        self.codes["index"] = _SOP_INDEX
        self.codes["layer_norm"] = _SOP_LAYER_NORM
        self.codes["transformer_softmax"] = _SOP_TRANSFORMER_SOFTMAX
        self.codes["transformer_linear"] = _SOP_TRANSFORMER_LINEAR
        self.codes["rms_norm"] = _SOP_RMS_NORM
        self.codes["rotary"] = _SOP_ROTARY
        self.codes["dropout"] = _SOP_DROPOUT
        self.codes["permutation"] = _SOP_PERMUTATION
        self.codes["gather_rows"] = _SOP_GATHER_ROWS
        self.codes["mse"] = _SOP_MSE
        self.codes["mae"] = _SOP_MAE
        self.codes["huber"] = _SOP_HUBER
        self.codes["bce_with_logits"] = _SOP_BCE_WITH_LOGITS
        self.codes["cross_entropy"] = _SOP_CROSS_ENTROPY
        self.codes["accuracy"] = _SOP_ACCURACY
        self.codes["log_loss"] = _SOP_LOG_LOSS
        self.codes["roc_auc"] = _SOP_ROC_AUC
        self.codes["state"] = _SOP_STATE


# One table per process (identical contents in every extension that links this file).
comptime _SESSION_OP_CODES = SharedSlot[_SessionOpCodes, "MojolearnSessionOpCodesV2"]


def _session_op_code(name: String) raises -> Int:
    var g = _SESSION_OP_CODES.get_or_create_ptr()
    if name in g[].codes:
        return g[].codes[name]
    return 0


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


def _append(mut dest: List[Int], src: List[Int]):
    for k in range(len(src)):  # driver-loop: over operation metadata
        dest.append(src[k])


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
    var name = String(operation)
    var op = _session_op_code(name)
    var resource_scope: List[Int] = [ops_session_registry_token(), KIND]
    if not _nil(fields["_id"]):
        if py_type(fields["_id"]) is not py_type(PythonObject(0)):
            raise Error("TypeError: Session handle must be a native integer")
        var admitted_id = neural_descriptor_resource_handle(fields["_lifetime"], resource_scope)
        if admitted_id == 0 or admitted_id != Int(py=fields["_id"]):
            raise Error("TypeError: Session handle differs from its native registry owner")
    if op == _SOP_PROGRAM_PUBLIC:
        return ops_program_public_execute_py[E, KIND](session, fields, args)
    if op == _SOP_PROGRAM_INIT:
        var out = args[0]
        program_initialize_owner(out, session, args[1])
        return PythonObject(None)
    if op == _SOP_ENTER:
        return session
    if op == _SOP_FREE_PROGRAM:
        if not _nil(fields["_id"]) and not _nil(args[0]._id):
            var owner = ops_session[E, KIND](Int(py=fields["_id"]))
            owner[].program_free(Int(py=args[0]._id))
        var program = args[0]
        py_setattr(program, "_id", PythonObject(None))
        return PythonObject(None)
    if op == _SOP_CLOSE or op == _SOP_EXIT:
        if not _nil(fields["_id"]):
            neural_descriptor_resource_close(fields["_lifetime"], resource_scope)
            py_setitem(fields, "_id", PythonObject(None))
        return PythonObject(False) if op == _SOP_EXIT else PythonObject(None)
    if op == _SOP_PROFILE_RESET:
        _ = ops_profile_reset_py()
        return PythonObject(None)
    if op == _SOP_PROFILE_REPORT:
        return ops_profile_report_py()
    if _nil(fields["_id"]):
        raise Error("ValueError: this session is closed")
    var sid = fields["_id"]
    if op == _SOP_LOAD_PARTITIONED or op == _SOP_LOAD_DISTRIBUTED:
        return ops_owned_program_create_request_py(fields, 0, name, args)
    if op == _SOP_PROGRAM_RUN:
        if len(args) != 4:
            raise Error("TypeError: Session.run requires a Program, arrays and two flags")
        var program = program_owner_fields(args[0])
        if program["_s"] is not session:
            raise Error("ValueError: Program belongs to another Session")
        if not program_numeric_scalar(args[2]) or not program_numeric_scalar(args[3]):
            raise Error("TypeError: Program flags require native boolean or numeric scalars")
        var pid = _integer(program["_id"], "program", 1)
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
        py_setitem(bundle, "body", program_new_owner(session, bundle["body"]))
        py_setitem(bundle, "condition", program_new_owner(session, bundle["condition"]))
        if bundle["otherwise"] is not PythonObject(None):
            py_setitem(bundle, "otherwise", program_new_owner(session, bundle["otherwise"]))
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
        var out = output_array(s[].shape_nd(handle))
        var dst = FP(unsafe_from_address=Int(py=out.__array_interface__["data"][0]))
        if op == _SOP_NUMPY:
            s[].download(handle, dst)
        else:
            s[].gradient_into(handle, dst)
        return out
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
        # UNPACK: exact supported scalar/container admission precedes every
        # conversion; no user conversion hook can enter optimizer execution.
        program_admit_value(args)
        var codes = ops_optimizer_codes(String(args[0]), String(args[8]))
        var request = OpsOptimizerRequest(codes[0], Bool(py=args[3]), codes[1],
            _int(args[9], "step_size", 1), _float(args[1], "lr"), _float(args[2], "momentum"),
            _float(args[4], "beta1"), _float(args[5], "beta2"), _float(args[6], "eps"),
            _float(args[7], "weight_decay"), _float(args[10], "gamma"), Float32(0))
        return PythonObject(s[].optimizer(request))
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
            # PACK the native snapshot directly into the documented mapping.
            var state = Python.dict()
            py_setitem(state, "step", snapshot.step)
            if snapshot.kind != 0:
                py_setitem(state, "beta1_power", Float64(snapshot.beta1_power))
                py_setitem(state, "beta2_power", Float64(snapshot.beta2_power))
                if snapshot.first != 0:
                    py_setitem(state, "exp_avg", snapshot.first)
                if snapshot.second != 0:
                    py_setitem(state, "exp_avg_sq", snapshot.second)
            elif snapshot.first != 0:
                py_setitem(state, "momentum_buffer", snapshot.first)
            return state
        var state = args[2]
        program_admit_value(state)
        var step = _int(state["step"], "step", 0)
        var first = 0
        var second = 0
        var beta1 = Float32(1)
        var beta2 = Float32(1)
        if s[].opts[oi].cfg.kind == 0:
            if py_contains(state, "momentum_buffer"):
                first = _int(state["momentum_buffer"], "momentum_buffer", 1)
        else:
            if py_contains(state, "exp_avg"):
                first = _int(state["exp_avg"], "exp_avg", 1)
            if py_contains(state, "exp_avg_sq"):
                second = _int(state["exp_avg_sq"], "exp_avg_sq", 1)
            beta1 = Float32(py=state["beta1_power"])
            beta2 = Float32(py=state["beta2_power"])
        var request = OpsOptimizerRestoreRequest(step, first, second, 0, beta1, beta2, Float32(1))
        s[].load_optimizer_state(opt, _int(args[1], "param", 1), request)
        return PythonObject(None)
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
    elif op == _SOP_ACCURACY or op == _SOP_LOG_LOSS or op == _SOP_ROC_AUC:
        var metric = 1 if op == _SOP_ACCURACY else (2 if op == _SOP_LOG_LOSS else 3)
        hs = [_int(args[0], "handle", 1), _int(args[1], "handle", 1)]
        return PythonObject(Float64(s[].metric(metric, hs)))
    if code == 0:
        raise Error("unknown native session operation: " + name)
    return PythonObject(s[].op(code, hs, ints, fs))
