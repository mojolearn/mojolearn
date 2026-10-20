# SPDX-License-Identifier: Apache-2.0
"""Native small-MLP state and gradient operations for parallel owners.

The fixed architecture and arithmetic entries are the existing supported MLP.
This module owns admission, snapshots and kernel sequences; it introduces no
new numerical kernels or topology-dependent reduction order.
"""
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.math import isfinite
from std.memory import memcpy
from bindings.generation_buffers import GENERATION_BOUNDARY, generation_array, generation_copy, py_ints
from bindings.parallel_byte_native import _parallel_array, _parallel_view, _parallel_int, _parallel_float, _parallel_schedule_check
from bindings.native_transport import _transport_json
from bindings.native_json import native_json_parse


struct ParallelMlpBoundary(Defaultable, Movable):
    var bi: PythonObject
    var model_cls: PythonObject
    var opt_cls: PythonObject
    var lock: PythonObject
    var entries: PythonObject
    var environ: PythonObject
    var mode: String

    def __init__(out self):
        self.bi = PythonObject(None)
        self.model_cls = PythonObject(None)
        self.opt_cls = PythonObject(None)
        self.lock = PythonObject(None)
        self.entries = PythonObject(None)
        self.environ = PythonObject(None)
        self.mode = String()


comptime PARALLEL_MLP_BOUNDARY = _Global[
    StorageType=ParallelMlpBoundary, name="MojoParallelMlpBoundary",
    init_fn=ParallelMlpBoundary.__init__,
]


def mlp_initialize(model_cls: PythonObject, opt_cls: PythonObject, binding: PythonObject,
                   environ: PythonObject, mode: String, vendor: String) raises:
    var b = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()
    b[].bi = Python.import_module("builtins")
    b[].model_cls = model_cls
    b[].opt_cls = opt_cls
    b[].lock = Python.import_module("_thread").RLock
    b[].environ = environ
    b[].mode = mode
    b[].entries = b[].bi.dict()
    for name in [String("neural_gemm"), String("mlp_bias_activation"), String("mlp_relu_backward"),
                 String("mlp_sum_rows"), String("ce_loss"), String("accumulate"), String("optimizer_step"),
                 String("training_numeric_mode"), String("training_vendor"), String("mlp_train_step"),
                 String("optimizer_parallel_available"), String("accumulate_parallel_available"),
                 String("clip_parallel_available")]:
        var entry = PythonObject(None)
        try:
            entry = b[].bi.getattr(binding, name)
        except:
            pass
        if entry is not None and b[].bi.type(entry) is not b[].bi.type(b[].bi.len):
            raise Error("parallel MLP: training entries must be compiled native functions")
        b[].entries[name] = entry
    if Int(py=b[].entries["training_numeric_mode"]()) != (1 if mode == "identical" else 0) or String(b[].entries["training_vendor"]()) != vendor:
        raise Error("parallel MLP: compiled mode/vendor admission mismatch")


def _mlp_names() -> List[String]:
    var result = List[String]()
    result.append("weight1")
    result.append("bias1")
    result.append("weight2")
    result.append("bias2")
    return result^


def _mlp_dims(index: Int) -> List[Int]:
    var result = List[Int]()
    if index == 0:
        result.append(16)
        result.append(8)
    elif index == 1:
        result.append(16)
    elif index == 2:
        result.append(3)
        result.append(16)
    else:
        result.append(3)
    return result^


def _mlp_mode() raises:
    if PARALLEL_MLP_BOUNDARY.get_or_create_ptr()[].mode != "identical":
        raise Error("parallel training requires an identical model")


def _mlp_fields(value: PythonObject, fields: String) raises:
    var bi = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    if bi.type(value) is not bi.dict or Bool(py=bi.set(value) != bi.set(PythonObject(fields).split("|"))):
        raise Error("parallel MLP: state has missing or unknown fields")


def _mlp_schedule(value: PythonObject) raises -> PythonObject:
    var bi = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    if bi.type(value) is not bi.dict or len(value) == 0:
        raise Error("SmallMLPTrainer data_schedule must be a nonempty JSON object")
    var budget = 128
    _parallel_schedule_check(value, 0, budget)
    var encoded = _transport_json(value)
    if encoded.byte_length() > 2048:
        raise Error("SmallMLPTrainer data_schedule exceeds 2048 bytes")
    return native_json_parse(PythonObject(encoded))


def _mlp_finite(value: PythonObject, label: String) raises:
    var data = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=value._addr))
    for i in range(Int(py=value.size)):
        if not isfinite(data[i]):
            raise Error("RuntimeError: SmallMLPTrainer nonfinite " + label)


def mlp_validate_state(state: PythonObject) raises -> PythonObject:
    _mlp_mode()
    var bi = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    _mlp_fields(state, "schema|architecture|numeric_mode|parameter_order|weights|optimizer|config|data_schedule")
    var names = _mlp_names()
    var name_list = bi.list()
    for name in names:
        name_list.append(name)
    if String(state["schema"]) != "mojolearn.small-mlp-trainer.v1" or String(state["numeric_mode"]) != "identical" or Bool(py=state["architecture"] != py_ints([8, 16, 3])) or Bool(py=state["parameter_order"] != name_list):
        raise Error("SmallMLPTrainer state schema/architecture/mode/order mismatch")
    _mlp_fields(state["weights"], "weight1|bias1|weight2|bias2")
    _mlp_fields(state["optimizer"], "kind|step|m|v|flags")
    _mlp_fields(state["config"], "lr|beta1|beta2|eps|weight_decay")
    var config = bi.dict()
    for name in [String("lr"), String("beta1"), String("beta2"), String("eps"), String("weight_decay")]:
        config[name] = Float64(_parallel_float(state["config"][name], name))
    if Float64(py=config["lr"]) <= 0 or Float64(py=config["eps"]) <= 0 or Float64(py=config["weight_decay"]) < 0 or Float64(py=config["beta1"]) < 0 or Float64(py=config["beta1"]) >= 1 or Float64(py=config["beta2"]) < 0 or Float64(py=config["beta2"]) >= 1:
        raise Error("SmallMLPTrainer optimizer configuration outside admitted range")
    var old = state["optimizer"]
    if String(old["kind"]) != "AdamW":
        raise Error("SmallMLPTrainer state optimizer mismatch")
    var optimizer = bi.dict()
    optimizer["kind"] = "AdamW"
    optimizer["step"] = _parallel_int(old["step"], 0, 2147483647, "SmallMLPTrainer step", False)
    optimizer["m"] = _parallel_array(old["m"], [195])
    optimizer["v"] = _parallel_array(old["v"], [195])
    optimizer["flags"] = _parallel_array(old["flags"], [4], True)
    var v = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=optimizer["v"]._addr))
    var flags = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=Int(py=optimizer["flags"]._addr))
    for i in range(195):
        if v[i] < 0:
            raise Error("SmallMLPTrainer state requires nonnegative v")
    for i in range(4):
        if flags[i] < 0 or flags[i] > 1:
            raise Error("SmallMLPTrainer state requires binary flags")
    var weights = bi.dict()
    for i in range(4):  # driver-loop: over the existing fixed MLP parameter registry
        weights[names[i]] = _parallel_array(state["weights"][names[i]], _mlp_dims(i))
    var result = bi.dict(state)
    result["architecture"] = py_ints([8, 16, 3])
    result["parameter_order"] = name_list
    result["weights"] = weights
    result["optimizer"] = optimizer
    result["config"] = config
    result["data_schedule"] = _mlp_schedule(state["data_schedule"])
    return result


def mlp_snapshot(model: PythonObject) raises -> PythonObject:
    _mlp_mode()
    var bi = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    # AdamW inherits the Python public-output attribute adapter. These are
    # internal stored fields, so read their native instance dictionary through
    # object.__getattribute__, preserving the Array owners without that adapter.
    var fields = bi.object.__getattribute__(model, "__dict__")
    var opt = bi.object.__getattribute__(fields["_opt"], "__dict__")
    # The supported identical MLP explicitly constructs AdamW resident=False;
    # its only model session is an Apple fast numerical path.
    if fields["_session"] is not None or opt["_res"] is not None:
        raise Error("parallel MLP: unexpected resident owner under identical admission")
    var result = bi.dict()
    result["schema"] = "mojolearn.small-mlp-trainer.v1"
    result["architecture"] = py_ints([8, 16, 3])
    result["numeric_mode"] = "identical"
    var names = _mlp_names()
    var weights = bi.dict()
    var order = bi.list()
    for i in range(4):  # driver-loop: over fixed MLP snapshot parameter owners
        weights[names[i]] = opt["params"][i]
        order.append(names[i])
    result["parameter_order"] = order
    result["weights"] = weights
    var optimizer = bi.dict()
    optimizer["kind"] = "AdamW"
    optimizer["step"] = opt["t"]
    optimizer["m"] = generation_array([195]) if opt["_m"] is None else opt["_m"]
    optimizer["v"] = generation_array([195]) if opt["_v"] is None else opt["_v"]
    optimizer["flags"] = opt["buf_initialized"]
    result["optimizer"] = optimizer
    result["config"] = fields["_config"]
    result["data_schedule"] = fields["_schedule"]
    return mlp_validate_state(result)


def mlp_publish(model: PythonObject, state: PythonObject) raises:
    var b = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()
    var checked = mlp_validate_state(state)
    var opt = b[].bi.object.__new__(b[].opt_cls)
    var config = checked["config"]
    var saved = checked["optimizer"]
    var names = _mlp_names()
    var params = b[].bi.list()
    for i in range(4):  # driver-loop: over replacement MLP parameter owners
        params.append(checked["weights"][names[i]])
    opt._where = "AdamW"
    opt._res = PythonObject(None)
    opt._res_binding = PythonObject(None)
    opt._host_fresh = True
    opt._resident_wanted = False
    opt.params = params
    opt.lr_schedule = PythonObject(None)
    opt.accumulation_steps = 1
    var offsets = generation_array([5], "i")
    var op = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=offsets._addr))
    var registry = [0, 128, 144, 192, 195]
    for i in range(5):
        op[i] = Int32(registry[i])
    opt.offsets = offsets
    opt.n_total = 195
    opt._m = saved["m"]
    opt._v = saved["v"]
    opt._pristine = False
    opt.buf_initialized = saved["flags"]
    opt.t = saved["step"]
    opt.packed_ = PythonObject(None)
    opt.total_norm_ = PythonObject(None)
    opt.clip_coef_ = PythonObject(None)
    opt.lr = config["lr"]
    var betas = b[].bi.list()
    betas.append(config["beta1"])
    betas.append(config["beta2"])
    opt.betas = b[].bi.tuple(betas)
    opt.eps = config["eps"]
    opt.weight_decay = config["weight_decay"]
    opt.maximize = False
    opt.numeric_mode = "identical"
    model._opt = opt
    model._config = config
    model._schedule = checked["data_schedule"]
    model._session = PythonObject(None)


def mlp_from_state(state: PythonObject) raises -> PythonObject:
    var b = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()
    var model = b[].bi.object.__new__(b[].model_cls)
    model._lock = b[].lock()
    mlp_publish(model, state)
    return model


def mlp_batch(value: PythonObject) raises -> PythonObject:
    var bi = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()[].bi
    var view = _parallel_view(value)
    var shape = value.shape if bi.type(value) is GENERATION_BOUNDARY.get_or_create_ptr()[].array_cls else view.shape
    if len(shape) != 2 or Int(py=shape[1]) != 8 or Int(py=shape[0]) < 1 or Int(py=shape[0]) > 256:
        raise Error("SmallMLPTrainer X must have shape (batch, 8), batch in [1,256]")
    return _parallel_array(value, [Int(py=shape[0]), 8])


def mlp_targets(value: PythonObject, rows: Int) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var view = _parallel_view(value)
    var shape = value.shape if g[].bi.type(value) is g[].array_cls else view.shape
    var fmt = String(view.format.lstrip("=@<"))
    if len(shape) != 1 or Int(py=shape[0]) != rows:
        raise Error("SmallMLPTrainer targets must match batch rows")
    if fmt != "b" and fmt != "B" and fmt != "h" and fmt != "H" and fmt != "i" and fmt != "I" and fmt != "l" and fmt != "L" and fmt != "q" and fmt != "Q":
        raise Error("TypeError: SmallMLPTrainer targets must be a native integer array")
    var packed = g[].arrays(fmt)
    packed.frombytes(view.tobytes("C"))
    var output = generation_array([rows], "i")
    var ids = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=output._addr))
    for i in range(rows):
        # Compare before conversion so unsigned 64-bit invalid targets get
        # the documented class-range refusal instead of signed overflow.
        if Bool(py=packed[i] < 0) or Bool(py=packed[i] >= 3):
            raise Error("SmallMLPTrainer targets require classes 0..2")
        ids[i] = Int32(Int(py=packed[i]))
    _ = packed^
    return output


def _mlp_gemm(a: PythonObject, b: PythonObject, m: Int, n: Int, k: Int, transpose: Int) raises -> PythonObject:
    var out = generation_array([m, n])
    var written = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()[].entries["neural_gemm"](a._addr, b._addr, out._addr, py_ints([m, n, k, transpose]))
    if Int(py=written) != m * n:
        raise Error("RuntimeError: SmallMLPTrainer neural GEMM returned incomplete output")
    _mlp_finite(out, "GEMM")
    return out


def _mlp_bias(values: PythonObject, bias: PythonObject, rows: Int, columns: Int, relu: Bool) raises -> PythonObject:
    var out = generation_array([rows, columns])
    var written = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()[].entries["mlp_bias_activation"](values._addr, bias._addr, out._addr, py_ints([rows, columns, Int(relu)]))
    if Int(py=written) != rows * columns:
        raise Error("RuntimeError: SmallMLPTrainer bias returned incomplete output")
    _mlp_finite(out, "bias/activation")
    return out


def _mlp_sum(values: PythonObject, rows: Int, columns: Int) raises -> PythonObject:
    var out = generation_array([columns])
    var written = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()[].entries["mlp_sum_rows"](values._addr, out._addr, py_ints([rows, columns]))
    if Int(py=written) != columns:
        raise Error("RuntimeError: SmallMLPTrainer sum returned incomplete output")
    _mlp_finite(out, "bias gradient")
    return out


def _mlp_fused_enabled() raises -> Bool:
    var b = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()
    return b[].entries["mlp_train_step"] is not None and Bool(py=b[].environ.get(
        PythonObject("MOJOLEARN_MLP_FUSED").encode("ascii"), PythonObject("1").encode("ascii")) != PythonObject("0").encode("ascii")) and Bool(py=b[].environ.get(
        PythonObject("MOJOLEARN_OPTIMIZER_DEVICE_COUNT").encode("ascii"), PythonObject("1").encode("ascii")) == PythonObject("1").encode("ascii"))


def _mlp_fused_gradients(checked: PythonObject, x: PythonObject, y: PythonObject) raises -> PythonObject:
    var b = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()
    var rows = Int(py=x.shape[0])
    var owners = b[].bi.list()
    owners.append(x)
    owners.append(y)
    var names = _mlp_names()
    for i in range(4):  # driver-loop: over fused MLP parameter registry entries
        owners.append(checked["weights"][names[i]])
    owners.append(generation_array([1]))
    owners.append(generation_array([1]))
    owners.append(generation_array([4], "i"))
    owners.append(generation_array([1]))
    owners.append(generation_array([rows, 3]))
    var gradients = b[].bi.list()
    for i in range(4):  # driver-loop: over fused MLP gradient output owners
        var grad = generation_array(_mlp_dims(i))
        owners.append(grad)
        gradients.append(grad)
    owners.append(generation_array([1]))
    owners.append(generation_array([3]))
    var addresses = b[].bi.list()
    for i in range(len(owners)):  # driver-loop: over the existing fused MLP ABI pointer registry
        addresses.append(owners[i]._addr)
    var config = checked["config"]
    var params = py_ints([rows, 1, 1])
    for name in [String("lr"), String("beta1"), String("beta2"), String("eps"), String("weight_decay")]:
        params.append(config[name])
    params.append(0)
    var written = b[].entries["mlp_train_step"](addresses, params)
    if Int(py=written) != rows * 3:
        raise Error("RuntimeError: SmallMLPTrainer fused gradient returned incomplete output")
    _mlp_finite(owners[9], "loss")
    for i in range(4):
        _mlp_finite(gradients[i], "parameter gradient")
    var result = b[].bi.list()
    result.append(Float64(Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=owners[9]._addr))[0]))
    result.append(gradients)
    _ = owners^
    return b[].bi.tuple(result)


def mlp_gradients_from_state(state: PythonObject, inputs: PythonObject, targets: PythonObject) raises -> PythonObject:
    var b = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()
    var checked = mlp_validate_state(state)
    var x = mlp_batch(inputs)
    var rows = Int(py=x.shape[0])
    var y = mlp_targets(targets, rows)
    if _mlp_fused_enabled():
        return _mlp_fused_gradients(checked, x, y)
    var w = checked["weights"]
    var hidden = _mlp_bias(_mlp_gemm(x, w["weight1"], rows, 16, 8, 1), w["bias1"], rows, 16, True)
    var logits = _mlp_bias(_mlp_gemm(hidden, w["weight2"], rows, 3, 16, 1), w["bias2"], rows, 3, False)
    var loss = generation_array([1])
    var row_loss = generation_array([rows])
    var dlogits = generation_array([rows, 3])
    _ = b[].entries["ce_loss"](loss._addr, row_loss._addr, dlogits._addr, logits._addr, y._addr, py_ints([rows, 3, -100, 2, 0, 1, 0]))
    _mlp_finite(loss, "loss")
    _mlp_finite(dlogits, "logit gradient")
    var dw2 = _mlp_gemm(dlogits, hidden, 3, 16, rows, 2)
    var db2 = _mlp_sum(dlogits, rows, 3)
    var incoming = _mlp_gemm(dlogits, w["weight2"], rows, 16, 3, 0)
    var dhidden = generation_array([rows, 16])
    var written = b[].entries["mlp_relu_backward"](hidden._addr, incoming._addr, dhidden._addr, py_ints([rows, 16]))
    if Int(py=written) != rows * 16:
        raise Error("RuntimeError: SmallMLPTrainer backward returned incomplete output")
    _mlp_finite(dhidden, "ReLU gradient")
    var dw1 = _mlp_gemm(dhidden, x, 16, 8, rows, 2)
    var db1 = _mlp_sum(dhidden, rows, 16)
    var grads = b[].bi.list()
    grads.append(dw1)
    grads.append(db1)
    grads.append(dw2)
    grads.append(db2)
    var result = b[].bi.list()
    var scalar = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=loss._addr))[0]
    result.append(Float64(scalar))
    result.append(grads)
    # Integer addresses do not retain their owners across sibling native
    # entries; keep every input/workspace alive through its consuming call.
    _ = checked^
    _ = x^
    _ = y^
    _ = hidden^
    _ = logits^
    _ = loss^
    _ = row_loss^
    _ = dlogits^
    _ = incoming^
    _ = dhidden^
    return b[].bi.tuple(result)


def neural_ordered_sum(parts: PythonObject) raises -> PythonObject:
    _mlp_mode()
    var b = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()
    var shards = b[].bi.list(parts)
    if len(shards) == 0:
        raise Error("ordered reduction needs at least one shard")
    var first = b[].bi.list(shards[0])
    var total = b[].bi.list()
    var dims = List[List[Int]]()
    var counts = List[Int]()
    var count = 0
    for j in range(len(first)):  # driver-loop: over the caller's gradient tensor registry
        var view = _parallel_view(first[j])
        var shape = first[j].shape if b[].bi.type(first[j]) is GENERATION_BOUNDARY.get_or_create_ptr()[].array_cls else view.shape
        var axes = List[Int]()
        var n = 1
        for axis in range(len(shape)):
            var dimension = Int(py=shape[axis])
            if dimension < 0 or (dimension != 0 and n > 2147483647 // dimension):
                raise Error("gradient registry exceeds signed 32-bit kernel indexing")
            axes.append(dimension)
            n *= dimension
        if count > 2147483647 - n:
            raise Error("gradient registry exceeds signed 32-bit kernel indexing")
        count += n
        counts.append(n)
        # Ordered addition historically admits nonfinite float32 inputs; the
        # existing accumulation kernel owns canonicalization. Do not turn a
        # preparation helper's finite scan into a new reduction restriction.
        total.append(_parallel_array(first[j], axes, False, False))
        dims.append(axes^)
    if len(shards) == 1:
        return total
    if len(first) == 0 or count == 0:
        raise Error("ordered reduction needs nonempty gradient tensors")
    var stacked = generation_array([2, count])
    var output = generation_array([count])
    for shard in range(1, len(shards)):  # driver-loop: over fixed ascending logical gradient shard order
        var current = b[].bi.list(shards[shard])
        if len(current) != len(total):
            raise Error("ordered reduction gradient tensor count mismatch")
        var offset = 0
        var stack = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(py=stacked._addr))
        for j in range(len(total)):  # driver-loop: over paired gradient tensor buffers
            var value = _parallel_array(current[j], dims[j], False, False)
            var left = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(py=total[j]._addr))
            var right = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(py=value._addr))
            memcpy(dest=stack + offset, src=left, count=counts[j])
            memcpy(dest=stack + count + offset, src=right, count=counts[j])
            offset += counts[j]
            _ = value^
        var addresses = b[].bi.list()
        addresses.append(output._addr)
        addresses.append(stacked._addr)
        _ = b[].entries["accumulate"](addresses, py_ints([count, 2, -1]))
        var source = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(py=output._addr))
        offset = 0
        for j in range(len(total)):  # driver-loop: over owned folded gradient output tensors
            var destination = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(py=total[j]._addr))
            memcpy(dest=destination, src=source + offset, count=counts[j])
            offset += counts[j]
    _ = stacked^
    _ = output^
    return total


def neural_cooperative_admit() raises:
    var b = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()
    var count = Int(py=b[].bi.int(b[].environ.get(PythonObject("MOJOLEARN_OPTIMIZER_DEVICE_COUNT").encode("ascii"), PythonObject("1").encode("ascii"))))
    if count > 1:
        for name in [String("optimizer_parallel_available"), String("accumulate_parallel_available"), String("clip_parallel_available")]:
            if b[].entries[name] is None:
                raise Error("ImportError: training binding lacks cooperative neural update entry " + name)
            if Int(py=b[].entries[name]()) != 1:
                raise Error("RuntimeError: training binding refused " + name)


def mlp_apply_from_state(state: PythonObject, gradients: PythonObject) raises -> PythonObject:
    neural_cooperative_admit()
    var b = PARALLEL_MLP_BOUNDARY.get_or_create_ptr()
    var next = mlp_validate_state(state)
    var step = Int(py=next["optimizer"]["step"])
    if step == 2147483647:
        raise Error("SmallMLPTrainer step exceeds its supported range")
    if len(gradients) != 4:
        raise Error("SmallMLPTrainer requires exactly four parameter gradients")
    var params = generation_array([195])
    var grads = generation_array([195])
    var offsets = generation_array([5], "i")
    var info = generation_array([3])
    var pp = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(py=params._addr))
    var gp = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(py=grads._addr))
    var op = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=offsets._addr))
    var names = _mlp_names()
    var offset = 0
    for j in range(4):  # driver-loop: over transactional MLP parameter and gradient buffers
        var value = _parallel_array(gradients[j], _mlp_dims(j))
        var n = Int(py=value.size)
        op[j] = Int32(offset)
        var src = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(py=next["weights"][names[j]]._addr))
        memcpy(dest=pp + offset, src=src, count=n)
        src = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(py=value._addr))
        memcpy(dest=gp + offset, src=src, count=n)
        offset += n
        _ = value^
    op[4] = Int32(offset)
    var cfg = next["config"]
    var plan = py_ints([4, 2, step + 1, 0])
    for name in [String("lr"), String("beta1"), String("beta2"), String("eps"), String("weight_decay")]:
        plan.append(cfg[name])
    for i in range(4):
        plan.append(0)
    var saved = next["optimizer"]
    var written = b[].entries["optimizer_step"](params._addr, grads._addr,
        saved["m"]._addr, saved["v"]._addr, offsets._addr, saved["flags"]._addr, info._addr, plan)
    if Int(py=written) != 195:
        raise Error("RuntimeError: SmallMLPTrainer update returned incomplete state")
    _mlp_finite(params, "updated parameter")
    _mlp_finite(saved["m"], "first moment")
    _mlp_finite(saved["v"], "second moment")
    offset = 0
    for j in range(4):  # driver-loop: over completed transactional parameter publication
        var value = next["weights"][names[j]]
        var n = Int(py=value.size)
        var dst = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(py=value._addr))
        memcpy(dest=dst, src=pp + offset, count=n)
        offset += n
    saved["step"] = step + 1
    _ = params^
    _ = grads^
    _ = offsets^
    _ = info^
    return next


def mlp_call(operation: String, args: PythonObject) raises -> PythonObject:
    if operation == "initialize":
        mlp_initialize(args[0], args[1], args[2], args[3], String(args[4]), String(args[5]))
    elif operation == "snapshot":
        return mlp_snapshot(args[0])
    elif operation == "validate":
        return mlp_validate_state(args[0])
    elif operation == "from_state":
        return mlp_from_state(args[0])
    elif operation == "publish":
        mlp_publish(args[0], args[1])
    elif operation == "gradient":
        return mlp_gradients_from_state(args[0], args[1], args[2])
    elif operation == "ordered_sum":
        return neural_ordered_sum(args[0])
    elif operation == "apply":
        return mlp_apply_from_state(args[0], args[1])
    else:
        raise Error("parallel MLP: unknown operation " + operation)
    return PythonObject(None)


def parallel_mlp_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        var result = mlp_call(String(args[0]), args[1])
        return result.steal_data()
    except error:
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error, ExceptionType.TypeError)
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error, ExceptionType("PyExc_RuntimeError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error, ExceptionType("PyExc_ImportError"))
        return raise_python_exception(error, ExceptionType.ValueError)
