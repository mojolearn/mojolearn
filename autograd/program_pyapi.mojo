# SPDX-License-Identifier: Apache-2.0
# Array ownership at the native Python binding boundary. NumPy is used only
# to allocate returned ABI buffers; conversion and every computed value use
# the executor. No user callbacks drive execution, epochs or batches.
from std.python import Python, PythonObject
from std.time import perf_counter_ns
from sequence.exec_trait import Exec
from autograd.execution_profile_pyapi import execution_profile_object
from sequence.ops import FP
from core.cast_in import CastIn, cast_in_kind, CAST_KIND_FLOAT, CAST_KIND_SIGNED, CAST_KIND_UNSIGNED
from autograd.rankn import ShapeN
from autograd.program_arrays_pyapi import output_array, is_integer_input, uint64_input
from autograd.tape import Tape
from autograd.program_io import ProgramInput, HostArray
from autograd.program_entries import program_execute_prepare, program_execute_run, program_fit_parameters, program_fit_native
from autograd.program_arrays_pyapi import host_arrays_py
from autograd.cast_ops import upload_cast, upload_cast_i32
from autograd.ops_api import ops_session
from autograd.program import OpsProgram, PROGRAM_IN_IDS, same_shape, shape_text
from autograd.program_tools import ProgramTrace, program_op_name
from autograd.program_optimize import ProgramOptimization, program_optimize
from autograd.program_runtime import replay_values, parameter_buffers, train_program, train_program_control
from autograd.pyapi import _input_handles, _seed, ival, fval, _dims_out


def array_input(a: PythonObject, what: String) raises -> Tuple[CastIn, ShapeN]:
    var nd = Int(py=a.ndim)
    if nd > 8:
        raise Error("ValueError: tensor rank must be at most 8")
    if Int(py=a.size) == 0:
        raise Error("ValueError: input must not be empty")
    var dims = List[Int]()
    if nd == 0:
        dims.append(1)
        dims.append(1)
    elif nd == 1:
        dims.append(Int(py=a.shape[0]))
        dims.append(1)
    else:
        for k in range(nd):  # small-loop(nd: input tensor rank): shape metadata, not array elements
            dims.append(Int(py=a.shape[k]))
    var sh = ShapeN.of(dims)
    var size = Int(py=a.itemsize)
    if not Bool(py=a.dtype.isnative):
        raise Error("TypeError: " + what + " must be in native byte order")
    var kind: Int
    try:
        kind = cast_in_kind(Int(py=a.dtype.num), size, what)
    except error:
        raise Error("TypeError: " + String(error))
    var s0 = sh.cols() * size
    var s1 = size
    if nd == 1:
        s0 = Int(py=a.strides[0])
    elif nd == 2:
        s0 = Int(py=a.strides[0])
        s1 = Int(py=a.strides[1])
    elif nd > 2 and not Bool(py=a.flags.c_contiguous):
        raise Error("ValueError: " + what + ": tensors above rank two must be C-contiguous")
    var src = CastIn(Int(py=a.__array_interface__["data"][0]), kind, size, sh.rows(), sh.cols(), s0, s1)
    src.check(what)
    return (src, sh)


def _sequence_values(a: PythonObject, depth: Int, mut dims: List[Int],
                     mut values: List[Float64], mut ids: List[Int64], integer: Bool, mut leaf_depth: Int) raises:
    var b = Python.import_module("builtins")
    if Bool(py=b.isinstance(a, b.list)) or Bool(py=b.isinstance(a, b.tuple)):
        var n = len(a)
        if depth >= 8 or n == 0:
            raise Error("ValueError: program input sequences must be nonempty and have rank at most eight")
        if depth == len(dims):
            dims.append(n)
        elif dims[depth] != n:
            raise Error("ValueError: program input sequences must be rectangular")
        for k in range(n):  # native input marshaling: elements supplied as Python scalar objects
            _sequence_values(a[k], depth + 1, dims, values, ids, integer, leaf_depth)
    else:
        if leaf_depth < 0:
            leaf_depth = depth
        if depth != len(dims) or depth != leaf_depth:
            raise Error("ValueError: program input sequences must be rectangular")
        if integer:
            if Bool(py=b.isinstance(a, b.bool)) or not Bool(py=b.hasattr(a, "__index__")):
                raise Error("TypeError: program ids must be integers")
            ids.append(Int64(py=a))
        else:
            values.append(Float64(py=a))


def program_input(a: PythonObject, integer: Bool) raises -> ProgramInput:
    """A typed ProgramInput from an array (its descriptor) or a nested
    sequence of Python scalars (converted once, owned by the result)."""
    var values = List[Float64]()
    var ids = List[Int64]()
    var b = Python.import_module("builtins")
    if Bool(py=b.hasattr(a, "dtype")) and Bool(py=b.hasattr(a, "strides")):
        var parsed = array_input(a, "numeric program input")
        if integer and parsed[0].kind != CAST_KIND_SIGNED and parsed[0].kind != CAST_KIND_UNSIGNED:
            raise Error("TypeError: program input holds ids and must be integers (an integer array)")
        return ProgramInput(parsed[0], parsed[1], values^, ids^)
    var dims = List[Int]()
    var leaf_depth = -1
    _sequence_values(a, 0, dims, values, ids, integer, leaf_depth)
    if len(dims) == 0:
        dims.append(1)
    if len(dims) == 1:
        dims.append(1)
    var shape = ShapeN.of(dims)
    if integer:
        var id_src = CastIn(Int(ids.unsafe_ptr()), CAST_KIND_SIGNED, 8,
                            shape.rows(), shape.cols(), shape.cols() * 8, 8)
        return ProgramInput(id_src, shape, values^, ids^)
    var src = CastIn(Int(values.unsafe_ptr()), CAST_KIND_FLOAT, 8,
                     shape.rows(), shape.cols(), shape.cols() * 8, 8)
    return ProgramInput(src, shape, values^, ids^)


def program_inputs(p: OpsProgram, arrays: PythonObject, dynamic: Bool, variable_rank: Bool = False) raises -> List[ProgramInput]:
    if len(arrays) != p.n_in:
        raise Error("ValueError: this program takes " + String(p.n_in) + " input arrays, got " + String(len(arrays)))
    var out = List[ProgramInput]()
    for k in range(p.n_in):  # driver-loop: over input descriptors, owners retained through replay
        var item = program_input(arrays[k], p.in_kinds[k] == PROGRAM_IN_IDS)
        if (dynamic and not variable_rank and item.shape.ndim != p.in_shapes[k].ndim) or (not dynamic and not same_shape(item.shape, p.in_shapes[k])):
            raise Error("ValueError: input " + String(k) + " must keep its captured "
                        + ("rank" if dynamic else "shape") + " " + shape_text(p.in_shapes[k])
                        + "; got " + shape_text(item.shape))
        out.append(item^)
    return out^


def _dict() raises -> PythonObject:
    return Python.import_module("builtins").dict()


def _result(arrays: PythonObject) raises -> PythonObject:
    if len(arrays) == 1:
        return arrays[0]
    return Python.import_module("builtins").tuple(arrays)


def _optimization_object(plan: ProgramOptimization) raises -> PythonObject:
    var out = _dict()
    out["schema"] = "mojolearn-graph-passes/1"
    out["scope"] = "inference only; original graph and derivative order unchanged"
    out["storage_reuse"] = False
    out["rewritten_relu_elements_before"] = plan.relu_elements_before
    out["rewritten_relu_elements_after"] = plan.relu_elements_after
    out["liveness_scope"] = "logical values only; tape owns aliases and storage"
    var passes = Python.list()
    for k in range(len(plan.pass_names)):  # driver-loop: over graph transformation report
        var row = _dict()
        row["name"] = plan.pass_names[k]
        row["nodes_before"] = plan.before[k]
        row["nodes_after"] = plan.after[k]
        row["rewrites"] = plan.rewrites[k]
        passes.append(row)
    out["passes"] = passes
    var origins = Python.list()
    for k in range(len(plan.source_nodes)):  # driver-loop: over source node provenance
        origins.append(PythonObject(plan.source_nodes[k]))
    out["source_nodes"] = origins
    var liveness = Python.list()
    for k in range(len(plan.last_use)):  # driver-loop: over logical reference use metadata
        var row = _dict()
        row["value"] = k
        row["last_use"] = plan.last_use[k]
        row["uses"] = plan.uses[k]
        liveness.append(row)
    out["liveness"] = liveness
    return out


def ops_program_execute_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, pid: PythonObject, owner_sid: PythonObject, arrays: PythonObject,
    dynamic_shapes: PythonObject, optimize: PythonObject, profile: PythonObject,
) raises -> PythonObject:
    var sid_i = Int(py=sid)
    var profiled = Bool(py=profile)
    var plan = program_execute_prepare[E, KIND](sid_i, Int(py=pid), Int(py=owner_sid), Bool(py=optimize), profiled)
    # Native ABI shape policy: bool preserves the existing fixed-rank
    # dynamic contract; integer 2 selects compatibility checked at each node.
    # Every operation still validates its own rank/axes/parameter constraints.
    var variable_rank = is_integer_input(dynamic_shapes) and Int(py=dynamic_shapes) == 2
    var inputs = program_inputs(plan.program, arrays, Bool(py=dynamic_shapes), variable_rank)
    var r = program_execute_run[E, KIND](sid_i, plan, inputs^, profiled)
    var result = host_arrays_py(r.outputs)
    if not profiled:
        return _result(result)
    var s = ops_session[E, KIND](sid_i)
    var report = _dict()
    report["executor"] = execution_profile_object(r.executor)
    report["optimization"] = _optimization_object(plan.optimization)
    report["timing"] = "host enqueue and synchronized wait; profiling serializes the queue"
    report["original_nodes"] = plan.original_nodes
    report["executed_nodes"] = plan.program.n_nodes()
    report["input_logical_bytes"] = r.upload_bytes
    report["input_device_bytes"] = r.input_bytes
    report["output_bytes"] = r.download_bytes
    report["parameter_bytes"] = r.parameter_bytes
    report["peak_tape_reserved_bytes"] = s[].program_trace.peak_tape_bytes
    report["upload_and_cast_ns"] = r.upload_ns
    report["download_and_allocate_ns"] = r.download_ns
    report["node_sync_count"] = s[].program_trace.sync_count
    report["node_sync_ns"] = s[].program_trace.sync_ns
    var rows = Python.list()
    for j in range(len(s[].program_trace.nodes)):  # driver-loop: over diagnostic rows
        var row = _dict()
        row["node"] = s[].program_trace.nodes[j]
        row["source_node"] = plan.optimization.source_nodes[s[].program_trace.nodes[j]]
        row["op"] = program_op_name(s[].program_trace.codes[j])
        row["shape"] = s[].program_trace.shapes[j]
        row["output_bytes"] = s[].program_trace.output_bytes[j]
        row["enqueue_ns"] = s[].program_trace.enqueue_ns[j]
        row["wait_ns"] = s[].program_trace.wait_ns[j]
        rows.append(row)
    report["nodes"] = rows
    var out = _dict()
    out["outputs"] = _result(result)
    out["profile"] = report
    return out


def ops_program_dynamic_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, pid: PythonObject, arrays: PythonObject,
) raises -> PythonObject:
    # Keep the previously exported entry ABI stable.
    return ops_program_execute_py[E, KIND](sid, pid, sid, arrays, PythonObject(True), PythonObject(False), PythonObject(False))


def ops_program_inspect_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, pid: PythonObject, optimize: PythonObject,
) raises -> PythonObject:
    var s = ops_session[E, KIND](Int(py=sid))
    var p = s[].program(Int(py=pid))
    var original_nodes = p.n_nodes()
    var optimization = program_optimize(p, Bool(py=optimize))
    p = optimization.program.copy()
    var report = _dict()
    report["original_nodes"] = original_nodes
    report["nodes_removed"] = original_nodes - p.n_nodes()
    report["input_shapes"] = _dims_out(p.in_shapes)
    report["output_shapes"] = _dims_out(p.out_shapes)
    report["parameter_shapes"] = _dims_out(p.param_shapes)
    report["optimization"] = "structural_pass_pipeline" if Bool(py=optimize) else "none"
    report["pass_report"] = _optimization_object(optimization)
    var rows = Python.list()
    for k in range(p.n_nodes()):  # driver-loop: over graph node metadata
        var row = _dict()
        row["node"] = k
        row["source_node"] = optimization.source_nodes[k]
        row["value"] = p.n_in + k
        row["op"] = program_op_name(p.code[k])
        row["code"] = p.code[k]
        var args = Python.list()
        var ints = Python.list()
        var floats = Python.list()
        for j in range(p.a_at[k], p.a_at[k + 1]):  # driver-loop: over input refs
            args.append(PythonObject(p.args[j]))
        for j in range(p.i_at[k], p.i_at[k + 1]):  # driver-loop: over integer attributes
            ints.append(PythonObject(p.ints[j]))
        for j in range(p.f_at[k], p.f_at[k + 1]):  # driver-loop: over float attributes
            floats.append(PythonObject(p.fs[j]))
        row["inputs"] = args
        row["integers"] = ints
        row["floats"] = floats
        rows.append(row)
    report["nodes"] = rows
    var outputs = Python.list()
    for k in range(len(p.outs)):  # driver-loop: over output refs
        outputs.append(PythonObject(p.outs[k]))
    report["outputs"] = outputs
    return report


def ops_program_fit_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, pid: PythonObject, arrays: PythonObject, targets: PythonObject,
    parameters: PythonObject, ip: PythonObject, fp: PythonObject,
) raises -> PythonObject:
    # ip: optimizer, loss, epochs, batch, shuffle, seed low, seed high.
    if len(ip) != 7 or len(fp) != 1:
        raise Error("program fit: malformed options")
    var srcs = List[CastIn]()
    var shapes = List[ShapeN]()
    for k in range(len(arrays)):  # small-loop(arrays: one descriptor per program input): argument parsing, no data
        var parsed = array_input(arrays[k], "program input")
        srcs.append(parsed[0])
        shapes.append(parsed[1])
    var target = CastIn.f32_c(0, 1, 1)
    if ival(ip, 1) != 0:
        target = array_input(targets, "program targets")[0]
    var epochs = ival(ip, 2)
    if epochs < 1 or len(srcs) < 1:
        raise Error("program fit: epochs and input count must be positive")
    var curve = output_array(ShapeN.of([epochs]))
    var s = ops_session[E, KIND](Int(py=sid))
    train_program(s[], Int(py=pid), srcs, shapes, target, ival(ip, 1), fval(fp, 0),
                  ival(ip, 0), _input_handles(parameters), epochs, ival(ip, 3), ival(ip, 4) != 0,
                  _seed(ip, 5), FP(unsafe_from_address=Int(py=curve.__array_interface__["data"][0])))
    return curve


def _integer(v: PythonObject, what: String, low: Int) raises -> Int:
    var b = Python.import_module("builtins")
    if not is_integer_input(v):
        raise Error("TypeError: " + what + " must be an integer")
    var value = Int(py=v)
    if value < low:
        raise Error("ValueError: " + what + " must be >= " + String(low))
    return value


def ops_program_fit_native_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, pid: PythonObject, arrays: PythonObject, targets: PythonObject,
    optimizer: PythonObject, parameters: PythonObject, loss: PythonObject,
    epochs: PythonObject, batch_size: PythonObject, shuffle: PythonObject,
    seed: PythonObject, delta: PythonObject,
    condition_id: Int = 0, otherwise_id: Int = 0,
) raises -> PythonObject:
    # One public binding call owns option parsing as well as the entire
    # upload/shuffle/batch/backward/update path. Old packed ABI stays usable.
    var s = ops_session[E, KIND](Int(py=sid))
    var p = s[].program(Int(py=pid))
    var name = String(py=loss)
    var kind = -1
    if name == "output":
        kind = 0
    elif name == "mse":
        kind = 1
    elif name == "mae":
        kind = 2
    elif name == "huber":
        kind = 3
    elif name == "bce_with_logits":
        kind = 4
    elif name == "cross_entropy":
        kind = 5
    if kind < 0:
        raise Error("ValueError: unknown loss " + name)
    var b = Python.import_module("builtins")
    var no_target = String(py=b.type(targets).__name__) == "NoneType"
    if (kind == 0) != no_target:
        raise Error("ValueError: omit targets only when loss='output'")
    var count = _integer(epochs, "epochs", 1)
    var batch = _integer(batch_size, "batch_size", 1)
    var opt = _integer(optimizer, "optimizer", 1)
    var seed_word = uint64_input(seed, "seed")
    var explicit = String(py=b.type(parameters).__name__) != "NoneType"
    var selected = List[Int]()
    if explicit:
        if is_integer_input(parameters):
            selected.append(_integer(parameters, "parameter", 1))
        else:
            for k in range(len(parameters)):  # small-loop(parameters: the selected parameter handles): argument parsing, no data
                selected.append(_integer(parameters[k], "parameter", 1))
    var params = program_fit_parameters[E, KIND](Int(py=sid), Int(py=pid), otherwise_id, explicit, selected)
    var owners = program_inputs(p, arrays, True)
    var target_owners = List[ProgramInput]()
    if kind != 0:
        target_owners.append(program_input(targets, False))
    var curve = output_array(ShapeN.of([count]))
    var branches = program_fit_native[E, KIND](
        Int(py=sid), Int(py=pid), owners^, target_owners^, kind, Float32(py=delta), opt, params, count, batch,
        Bool(py=shuffle), seed_word, FP(unsafe_from_address=Int(py=curve.__array_interface__["data"][0])),
        condition_id, otherwise_id,
    )
    var out = Python.list()
    out.append(curve)
    out.append(PythonObject(count))
    if condition_id > 0:
        var counts = b.dict()
        counts["true"] = branches[0]
        counts["false"] = branches[1]
        out.append(counts)
    return out
