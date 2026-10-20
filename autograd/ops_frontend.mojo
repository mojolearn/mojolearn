# SPDX-License-Identifier: Apache-2.0
# Native normalization for the existing Session API. Public wrappers hand
# over objects unchanged; this boundary owns descriptors, options and outputs.
from std.python import Python, PythonObject
from sequence.exec_trait import Exec
from sequence.ops import FP
from autograd.ops_api import ops_session
from autograd.program_pyapi import ProgramInput, _integer
from autograd.program_arrays import output_array, is_integer_input, uint64_input
from autograd.pyapi import program_info, ops_close_py, ops_profile_reset_py, ops_profile_report_py
from autograd.model_checkpoint import ops_load_program_py, ops_save_state_py, ops_load_state_py
from autograd.program_save import ops_program_save_native_py
from bindings.native_json import native_path
from autograd.weight_frontend import ops_weights_native_py
from autograd.session_lifetime import SessionLifetime
from autograd.execution_profile_pyapi import execution_profile_object
from autograd.control_checkpoint import control_save_py, control_load_py
from autograd.gradient_diagnostics import gradient_report_py


def _nil(x: PythonObject) -> Bool:
    return x is PythonObject(None)


def _int(x: PythonObject, name: String = "integer", low: Int = -9223372036854775807) raises -> Int:
    return _integer(x, name, low)


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
        out.append(xs[k])
    return Python.import_module("builtins").tuple(out)


def _program_object(session: PythonObject, info: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var out = b.object.__new__(session._program_class)
    out._s = session
    out._id = info[0]
    out._execute_native = session._program_execute
    out._inspect_native = session._program_inspect
    out._control_native = session._program_control
    out._fit_native = session._program_fit
    out._session_native = session._session_native
    out.params = b.tuple(info[3])
    out.input_kinds = b.tuple(info[4])
    for k in range(2):  # small-loop(2: input/output shape descriptors): metadata only
        var dims = Python.list()
        for j in range(len(info[k + 1])):  # driver-loop: over graph input/output shapes
            dims.append(b.tuple(info[k + 1][j]))
        if k == 0:
            out.input_shapes = b.tuple(dims)
        else:
            out.output_shapes = b.tuple(dims)
    return out


def ops_session_call_py[E: Exec & Movable & Deinitable, KIND: Int](
    session: PythonObject, operation: PythonObject, args: PythonObject,
) raises -> PythonObject:
    var name = String(operation)
    if name == "enter":
        return session
    if name == "free_program":
        if not _nil(session._id) and not _nil(args[0]._id):
            var owner = ops_session[E, KIND](Int(py=session._id))
            owner[].program_free(Int(py=args[0]._id))
        var program = args[0]
        program._id = PythonObject(None)
        return PythonObject(None)
    if name == "close" or name == "exit":
        if not _nil(session._id):
            var owner = session._lifetime.downcast_value_ptr[SessionLifetime[E, KIND]]()
            owner[].close()
            var target = session
            target._id = PythonObject(None)
        return PythonObject(False) if name == "exit" else PythonObject(None)
    if name == "profile_reset":
        _ = ops_profile_reset_py()
        return PythonObject(None)
    if name == "profile_report":
        return ops_profile_report_py()
    if _nil(session._id):
        raise Error("ValueError: this session is closed")
    var sid = session._id
    var s = ops_session[E, KIND](Int(py=sid))
    if name == "gradient_report":
        return gradient_report_py[E, KIND](Int(py=sid), _ints(args[0], "handle", 1), Bool(py=args[1]))
    if name == "execution_profile_begin":
        s[].ex.execution_profile_begin()
        return PythonObject(None)
    if name == "execution_profile_end":
        return execution_profile_object(s[].ex.execution_profile_end())
    if name == "save_control":
        return control_save_py[E, KIND](sid, args[0], args[2], args[3], args[4], args[1])
    if name == "load_control":
        var bundle = control_load_py[E, KIND](sid, args[0])
        bundle["body"] = _program_object(session, bundle["body"])
        bundle["condition"] = _program_object(session, bundle["condition"])
        if bundle["otherwise"] is not PythonObject(None):
            bundle["otherwise"] = _program_object(session, bundle["otherwise"])
        return bundle
    if name == "save_program":
        return ops_program_save_native_py[E, KIND](args[0], args[1])
    if name == "load_weights":
        return ops_weights_native_py[E, KIND](sid, args[0], args[1], args[2])
    if name == "tensor" or name == "ids":
        var input = ProgramInput(args[0], name == "ids")
        var handle: Int
        if name == "ids":
            handle = s[].upload_ids(input.src, input.shape)
        else:
            handle = s[].upload_nd(input.src, input.shape, Bool(py=args[1]))
        _ = input^  # keep all scalar-list storage alive across the upload
        return PythonObject(handle)
    if name == "shape":
        var sh = s[].shape_nd(_int(args[0], "handle", 1))
        var dims = List[Int]()
        for k in range(sh.ndim):  # small-loop(sh.ndim: tensor rank): shape metadata
            dims.append(sh.d[k])
        return _tuple_ints(dims)
    if name == "numpy" or name == "grad":
        var handle = _int(args[0], "handle", 1)
        var out = output_array(s[].shape_nd(handle))
        var dst = FP(unsafe_from_address=Int(py=out.__array_interface__["data"][0]))
        if name == "numpy":
            s[].download(handle, dst)
        else:
            s[].gradient_into(handle, dst)
        return out
    if name == "scalar":
        return PythonObject(Float64(s[].scalar(_int(args[0], "handle", 1))))
    if name == "capture":
        var pid = s[].capture(_ints(args[0], "input", 1), _ints(args[1], "output", 1))
        return _program_object(session, program_info(pid, s[].program(pid)))
    if name == "load_program":
        if s[].tape.is_recording():
            raise Error("ValueError: programs run in an inference session (Session(record=False))")
        return _program_object(session, ops_load_program_py[E, KIND](sid, native_path(args[0])))
    if name == "save_state":
        var params = _ints(args[1], "handle", 1)
        if len(params) == 0:
            raise Error("ValueError: params must hold at least one tensor handle")
        var opt = 0 if _nil(args[2]) else _int(args[2], "optimizer", 1)
        if opt != 0:
            _ = s[]._opt_index(opt)
        var seed = _seed_words(args[3])
        var counters = Python.list()
        counters.append(seed[0])
        counters.append(seed[1])
        counters.append(_int(args[4], "epoch", 0))
        counters.append(_int(args[5], "step", 0))
        _ = ops_save_state_py[E, KIND](sid, native_path(args[0]), _tuple_ints(params), PythonObject(opt), counters)
        return PythonObject(None)
    if name == "load_state":
        var opt = 0 if _nil(args[1]) else _int(args[1], "optimizer", 1)
        if opt != 0:
            _ = s[]._opt_index(opt)
        var grad = Bool(py=args[2]) and s[].recording()
        var res = ops_load_state_py[E, KIND](sid, native_path(args[0]), PythonObject(Int(grad)), PythonObject(opt))
        var result = Python.import_module("builtins").dict()
        result["params"] = res[0]
        result["seed"] = res[3].__lshift__(32).__or__(res[2])
        result["epoch"] = res[4]
        result["step"] = res[5]
        return result
    if name == "reset":
        s[].reset()
        return PythonObject(None)
    if name == "free":
        s[].free(_int(args[0], "handle", 1))
        return PythonObject(None)
    if name == "backward":
        if Bool(py=args[2]):
            if Bool(py=args[1]):
                raise Error("ValueError: accumulate=True cannot be combined with create_graph=True")
            s[].backward_graph(_int(args[0], "loss", 1))
        else:
            s[].backward(_int(args[0], "loss", 1), Bool(py=args[1]))
        return PythonObject(None)
    if name == "grad_tensor":
        return PythonObject(s[].grad_handle(_int(args[0], "handle", 1)))
    if name == "vjp" or name == "hvp" or name == "jvp":
        var inputs = _ints(args[1], "input", 1)
        var out = List[Int]()
        if name == "vjp":
            out = s[].vjp(_int(args[0], "output", 1), inputs, _int(args[2], "v", 1), Bool(py=args[3]))
        else:
            var directions = _ints(args[2], "direction", 1)
            if len(directions) != len(inputs):
                raise Error("ValueError: one direction is required per input")
            if name == "jvp":
                return PythonObject(s[].jvp(_int(args[0], "output", 1), inputs, directions))
            out = s[].hvp(_int(args[0], "loss", 1), inputs, directions)
        var result = Python.list()
        for k in range(len(out)):  # driver-loop: over derivative result handles
            result.append(out[k])
        return result
    if name == "zero_grad":
        s[].zero_grad()
        return PythonObject(None)
    if name == "optimizer":
        var kind = String(args[0])
        var kind_code = 0 if kind == "sgd" else (1 if kind == "adam" else (2 if kind == "adamw" else -1))
        var schedule = String(args[8])
        var schedule_code = 0 if schedule == "constant" else (1 if schedule == "step" else (2 if schedule == "exponential" else -1))
        if kind_code < 0 or schedule_code < 0:
            raise Error("ValueError: invalid optimizer kind or schedule")
        var ip = List[Int]()
        ip = [kind_code, Int(Bool(py=args[3])), schedule_code, _int(args[9], "step_size", 1)]
        var fp = List[Float32]()
        fp = [Float32(py=args[1]), Float32(py=args[2]), Float32(py=args[4]), Float32(py=args[5]),
              Float32(py=args[6]), Float32(py=args[7]), Float32(py=args[10])]
        var opt = s[].optimizer(ip, fp)
        session._opt_kind[opt] = args[0]
        return PythonObject(opt)
    if name == "step" or name == "scheduler_step" or name == "learning_rate" or name == "optimizer_state" or name == "load_optimizer_state":
        var opt = _int(args[0], "optimizer", 1)
        var oi = s[]._opt_index(opt)
        if name == "step":
            s[].step(opt, _ints(args[1], "parameter", 1))
            return PythonObject(None)
        if name == "scheduler_step":
            s[].scheduler_step(opt)
            return PythonObject(None)
        if name == "learning_rate":
            return PythonObject(Float64(s[].learning_rate(opt)))
        var kind = String(session._opt_kind[opt])
        var b = Python.import_module("builtins")
        if name == "optimizer_state":
            var st = s[].optimizer_state(opt, _int(args[1], "param", 1))
            var state = b.dict()
            state["step"] = st[0]
            if kind != "sgd":
                state["beta1_power"] = Float64(st[1])
                state["beta2_power"] = Float64(st[2])
                if st[4] != 0:
                    state["exp_avg"] = st[4]
                if st[5] != 0:
                    state["exp_avg_sq"] = st[5]
            elif st[4] != 0:
                state["momentum_buffer"] = st[4]
            return state
        var state = args[2]
        var ip = List[Int]()
        var fp = List[Float32]()
        ip.append(_int(state["step"], "step", 0))
        fp = [Float32(1), Float32(1), Float32(1)]
        if kind == "sgd":
            ip.append(_int(state["momentum_buffer"], "momentum_buffer", 1) if Bool(py=state.__contains__("momentum_buffer")) else 0)
            ip.append(0)
        else:
            ip.append(_int(state["exp_avg"], "exp_avg", 1) if Bool(py=state.__contains__("exp_avg")) else 0)
            ip.append(_int(state["exp_avg_sq"], "exp_avg_sq", 1) if Bool(py=state.__contains__("exp_avg_sq")) else 0)
            fp[0] = Float32(py=state["beta1_power"])
            fp[1] = Float32(py=state["beta2_power"])
        ip.append(0)
        s[].load_optimizer_state(opt, _int(args[1], "param", 1), ip, fp)
        return PythonObject(None)
    var code = 0
    var hs = List[Int]()
    var ints = List[Int]()
    var fs = List[Float32]()
    if name == "copy":
        code = 31
        hs.append(_int(args[0], "handle", 1))
    if name == "detach":
        code = 29
        hs.append(_int(args[0], "handle", 1))
    if name == "matmul":
        code = 1
        hs.append(_int(args[0], "handle", 1))
        hs.append(_int(args[1], "handle", 1))
    if name == "add":
        code = 2
        hs.append(_int(args[0], "handle", 1))
        hs.append(_int(args[1], "handle", 1))
    if name == "sub":
        code = 3
        hs.append(_int(args[0], "handle", 1))
        hs.append(_int(args[1], "handle", 1))
    if name == "mul":
        code = 4
        hs.append(_int(args[0], "handle", 1))
        hs.append(_int(args[1], "handle", 1))
    if name == "div":
        code = 5
        hs.append(_int(args[0], "handle", 1))
        hs.append(_int(args[1], "handle", 1))
    if name == "relu":
        code = 6
        hs.append(_int(args[0], "handle", 1))
    if name == "sigmoid":
        code = 7
        hs.append(_int(args[0], "handle", 1))
    if name == "tanh":
        code = 8
        hs.append(_int(args[0], "handle", 1))
    if name == "exp":
        code = 9
        hs.append(_int(args[0], "handle", 1))
    if name == "log":
        code = 10
        hs.append(_int(args[0], "handle", 1))
    if name == "sqrt":
        code = 11
        hs.append(_int(args[0], "handle", 1))
    if name == "erf":
        code = 48
        hs.append(_int(args[0], "handle", 1))
    if name == "softplus":
        code = 50
        hs.append(_int(args[0], "handle", 1))
    if name == "silu":
        code = 51
        hs.append(_int(args[0], "handle", 1))
    if name == "hard_swish":
        code = 55
        hs.append(_int(args[0], "handle", 1))
    if name == "abs":
        code = 57
        hs.append(_int(args[0], "handle", 1))
    if name == "floor":
        code = 58
        hs.append(_int(args[0], "handle", 1))
    if name == "ceil":
        code = 59
        hs.append(_int(args[0], "handle", 1))
    if name == "round":
        code = 60
        hs.append(_int(args[0], "handle", 1))
    if name == "sin":
        code = 61
        hs.append(_int(args[0], "handle", 1))
    if name == "cos":
        code = 62
        hs.append(_int(args[0], "handle", 1))
    if name == "sign":
        code = 63
        hs.append(_int(args[0], "handle", 1))
    if name == "reciprocal":
        code = 64
        hs.append(_int(args[0], "handle", 1))
    if name == "pow":
        code = 65
        hs.append(_int(args[0], "handle", 1))
        hs.append(_int(args[1], "handle", 1))
    if name == "minimum":
        code = 66
        hs.append(_int(args[0], "handle", 1))
        hs.append(_int(args[1], "handle", 1))
    if name == "maximum":
        code = 67
        hs.append(_int(args[0], "handle", 1))
        hs.append(_int(args[1], "handle", 1))
    if name == "where":
        code = 68
        hs.append(_int(args[0], "handle", 1))
        hs.append(_int(args[1], "handle", 1))
        hs.append(_int(args[2], "handle", 1))
    if name == "full" or name == "expand":
        code = 30 if name == "full" else 71
        ints = _ints(args[0 if name == "full" else 1], "shape", 1)
        if len(ints) < 1 or len(ints) > 8:
            raise Error("ValueError: shape must have one to eight dimensions")
        if name == "full":
            fs.append(Float32(py=args[1]))
        else:
            hs.append(_int(args[0], "handle", 1))
    if name == "scale" or name == "leaky_relu" or name == "elu" or name == "pow_scalar":
        code = 12 if name == "scale" else (52 if name == "leaky_relu" else (53 if name == "elu" else 65))
        hs.append(_int(args[0], "handle", 1))
        fs.append(Float32(py=args[1]))
    if name == "hard_sigmoid":
        code = 54
        hs.append(_int(args[0], "handle", 1))
        fs = [Float32(py=args[1]), Float32(py=args[2])]
    if name == "transpose":
        hs.append(_int(args[0], "handle", 1))
        var a = _int(args[1], "a")
        var b = _int(args[2], "b")
        code = 13 if a == -2 and b == -1 else 34
        if code == 34:
            ints = [1, a, b]
    if name == "permute":
        code = 34
        hs.append(_int(args[0], "handle", 1))
        ints.append(0)
        _append(ints, _axes(args[1]))
    if name == "reshape":
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
    if name == "flatten" or name == "squeeze" or name == "unsqueeze":
        code = 33
        hs.append(_int(args[0], "handle", 1))
        if name == "flatten":
            ints = [1, _int(args[1], "start"), _int(args[2], "end")]
        elif name == "squeeze":
            ints.append(2)
            _append(ints, _axes(args[1]))
        else:
            ints = [3, _int(args[1], "axis")]
    if name == "sum" or name == "mean" or name == "max" or name == "any" or name == "all":
        code = 35 if name == "sum" else (36 if name == "mean" else (37 if name == "max" else 74))
        hs.append(_int(args[0], "handle", 1))
        if code == 74:
            ints.append(1 if name == "all" else 0)
        ints.append(_keep(args[2]))
        _append(ints, _axes(args[1]))
    if name == "softmax" or name == "log_softmax":
        hs.append(_int(args[0], "handle", 1))
        var axis = _int(args[1], "axis")
        code = 17 if name == "softmax" else 18
        if axis != -1:
            code = 42 if name == "softmax" else 43
            ints.append(axis)
    if name == "concat":
        code = 39
        hs = _ints(args[0], "handle", 1)
        if len(hs) == 0:
            raise Error("ValueError: concat needs at least one tensor")
        ints.append(_int(args[1], "axis"))
    if name == "split":
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
        var out = Python.list()
        for k in range(count):  # driver-loop: over requested split views
            var spec = List[Int]()
            spec.append(axis)
            spec.append(k)
            _append(spec, tail)
            out.append(s[].op(40, hs, spec, fs))
        return out
    if name == "embedding" or name == "take":
        code = 32 if name == "embedding" else 41
        hs.append(_int(args[0], "handle", 1))
        var option = -1 if name == "embedding" and _nil(args[2]) else _int(args[2], "padding_idx" if name == "embedding" else "axis")
        if name == "embedding" and not _nil(args[2]) and option < 0:
            raise Error("ValueError: padding_idx must be >= 0")
        if _is_int(args[1]):
            hs.append(_int(args[1], "handle", 1))
            ints.append(option)
        else:
            var input = ProgramInput(args[1], True)
            var b = Python.import_module("builtins")
            var dtype_num = 9
            if Bool(py=b.hasattr(args[1], "dtype")):
                dtype_num = Int(py=args[1].dtype.num)
            ints = [input.src.addr, dtype_num, input.src.itemsize, input.src.rows,
                    input.src.cols, input.src.s0, input.src.s1, option]
            var out = s[].op(code, hs, ints, fs)
            _ = input^  # scalar sequence owner remains alive through upload
            return PythonObject(out)
    if name == "conv2d" or name == "linear":
        code = 44 if name == "conv2d" else 19
        hs = [_int(args[0], "input", 1), _int(args[1], "weight", 1)]
        if not _nil(args[2]):
            hs.append(_int(args[2], "bias", 1))
        if name == "conv2d":
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
    if name == "max_pool2d" or name == "avg_pool2d":
        code = 45 if name == "max_pool2d" else 46
        hs.append(_int(args[0], "handle", 1))
        var kernel = _int(args[1], "kernel_size", 1)
        ints = [kernel, kernel if _nil(args[2]) else _int(args[2], "stride", 1)]
    if name == "gelu":
        code = 49
        hs.append(_int(args[0], "handle", 1))
        var approx = String(args[1])
        if approx != "none" and approx != "tanh":
            raise Error("ValueError: approximate must be 'none' or 'tanh'")
        ints.append(1 if approx == "tanh" else 0)
    if name == "clip":
        code = 56
        hs.append(_int(args[0], "handle", 1))
        fs.append(Float32(-1.0) / Float32(0.0) if _nil(args[1]) else Float32(py=args[1]))
        fs.append(Float32(1.0) / Float32(0.0) if _nil(args[2]) else Float32(py=args[2]))
    if name == "equal" or name == "not_equal" or name == "greater" or name == "greater_equal" or name == "less" or name == "less_equal":
        code = 69
        hs = [_int(args[0], "handle", 1), _int(args[1], "handle", 1)]
        var comparison = 0
        if name == "not_equal":
            comparison = 5
        elif name == "greater":
            comparison = 1
        elif name == "less":
            comparison = 2
        elif name == "greater_equal":
            comparison = 3
        elif name == "less_equal":
            comparison = 4
        ints.append(comparison)
    if name == "logical_not" or name == "logical_and" or name == "logical_or":
        code = 70
        hs.append(_int(args[0], "handle", 1))
        if name != "logical_not":
            hs.append(_int(args[1], "handle", 1))
        ints.append(0 if name == "logical_not" else (1 if name == "logical_and" else 2))
    if name == "argmax":
        code = 72
        hs.append(_int(args[0], "handle", 1))
        ints = [_int(args[1], "axis"), Int(Bool(py=args[2])), Int(not Bool(py=args[3]))]
    if name == "pad":
        code = 73
        hs.append(_int(args[0], "handle", 1))
        ints = _ints(args[1], "pads")
        if len(ints) < 1 or len(ints) > 16:
            raise Error("ValueError: pads must list two counts for each of one to eight axes")
        fs.append(Float32(py=args[2]))
    if name == "cumsum":
        code = 75
        hs.append(_int(args[0], "handle", 1))
        ints = [_int(args[1], "axis"), Int(Bool(py=args[2])), Int(Bool(py=args[3]))]
    if name == "index":
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
    if name == "layer_norm":
        code = 20
        hs = [_int(args[0], "input", 1), _int(args[1], "weight", 1), _int(args[2], "bias", 1)]
        var eps = Float32(py=args[3])
        if not eps > 0:
            raise Error("ValueError: eps must be positive")
        fs.append(eps)
    if name == "dropout":
        code = 21
        hs.append(_int(args[0], "handle", 1))
        var prob = Float32(py=args[1])
        if not (prob >= 0 and prob < 1):
            raise Error("ValueError: p must be in [0, 1)")
        fs.append(prob)
        ints = _seed_words(args[2])
        ints.append(_int(args[3], "counter", 0))
    if name == "permutation":
        code = 23
        ints.append(_int(args[0], "n", 1))
        _append(ints, _seed_words(args[1]))
        ints.append(_int(args[2], "epoch", 0))
    if name == "gather_rows":
        code = 22
        hs = [_int(args[0], "handle", 1), _int(args[1], "handle", 1)]
        ints = [_int(args[2], "start", 0), 0 if _nil(args[3]) else _int(args[3], "count", 1)]
    if name == "mse" or name == "mae" or name == "huber" or name == "bce_with_logits" or name == "cross_entropy":
        code = 24 if name == "mse" else (25 if name == "mae" else (26 if name == "huber" else (27 if name == "bce_with_logits" else 28)))
        hs = [_int(args[0], "handle", 1), _int(args[1], "handle", 1)]
        var reduction = String(args[3 if name == "huber" else 2])
        if reduction != "mean" and reduction != "sum":
            raise Error("ValueError: reduction must be 'mean' or 'sum'")
        ints.append(0 if reduction == "mean" else 1)
        if name == "huber":
            var delta = Float32(py=args[2])
            if not delta > 0:
                raise Error("ValueError: delta must be positive")
            fs.append(delta)
    if name == "accuracy" or name == "log_loss" or name == "roc_auc":
        var metric = 1 if name == "accuracy" else (2 if name == "log_loss" else 3)
        hs = [_int(args[0], "handle", 1), _int(args[1], "handle", 1)]
        return PythonObject(Float64(s[].metric(metric, hs)))
    if code == 0:
        raise Error("unknown native session operation: " + name)
    return PythonObject(s[].op(code, hs, ints, fs))
