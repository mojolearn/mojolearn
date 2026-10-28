# SPDX-License-Identifier: Apache-2.0
# Array ownership at the native Python binding boundary. NumPy is used only
# to allocate returned ABI buffers; conversion and every computed value use
# the executor. No user callbacks drive execution, epochs or batches.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_setitem, py_type, py_contains
from std.memory import memcpy
from std.time import perf_counter_ns
from sequence.exec_trait import Exec
from providers.python.autograd_execution_profile_pyapi import execution_profile_object
from sequence.ops import FP
from core.cast_in import CastIn, cast_in_kind, CAST_KIND_FLOAT, CAST_KIND_SIGNED, CAST_KIND_UNSIGNED
from autograd.rankn import ShapeN
from providers.python.autograd_program_arrays_pyapi import output_array, output_array_dims, is_ndarray, is_integer_input, uint64_input, program_native_input, program_admit_value, program_numeric_scalar
from autograd.tape import Tape
from autograd.program_io import ProgramInput, HostArray
from autograd.program_entries import program_execute_prepare, program_execute_run, program_fit_parameters, program_fit_native
from providers.python.autograd_program_arrays_pyapi import host_arrays_py
from autograd.cast_ops import upload_cast, upload_cast_i32
from autograd.ops_api import ops_session
from autograd.program import OpsProgram, PROGRAM_IN_IDS, same_shape, shape_text
from autograd.program_tools import ProgramTrace, program_op_name
from autograd.program_optimize import ProgramOptimization, program_optimize
from autograd.program_runtime import replay_values, parameter_buffers, train_program_control
from providers.python.autograd_pyapi import _dims_out
from providers.python.autograd_import_metadata_pyapi import imported_program_snapshot
from providers.python.autograd_host_provider_pyapi import imported_host_provider_request
from autograd.program_host_request import NativeProgramHostRun


def array_input(a: PythonObject, what: String) raises -> Tuple[CastIn, ShapeN]:
    var nd = Int(py=a.ndim)
    if nd > 8:
        raise Error("ValueError: tensor rank must be at most 8")
    if Int(py=a.size) == 0:
        raise Error("ValueError: input must not be empty")
    var shape = a.shape
    var strides = a.strides
    var dtype = a.dtype
    var dims = List[Int]()
    if nd == 0:
        dims.append(1)
        dims.append(1)
    elif nd == 1:
        dims.append(Int(py=shape[0]))
        dims.append(1)
    else:
        for k in range(nd):  # small-loop(nd: input tensor rank): shape metadata, not array elements
            dims.append(Int(py=shape[k]))
    var sh = ShapeN.of(dims)
    var size = Int(py=a.itemsize)
    if not Bool(py=dtype.isnative):
        raise Error("TypeError: " + what + " must be in native byte order")
    var kind: Int
    try:
        kind = cast_in_kind(Int(py=dtype.num), size, what)
    except error:
        raise Error("TypeError: " + String(error))
    var s0 = sh.cols() * size
    var s1 = size
    if nd == 1:
        s0 = Int(py=strides[0])
    elif nd == 2:
        s0 = Int(py=strides[0])
        s1 = Int(py=strides[1])
    elif nd > 2 and not Bool(py=a.flags.c_contiguous):
        raise Error("ValueError: " + what + ": tensors above rank two must be C-contiguous")
    var src = CastIn(Int(py=a.__array_interface__["data"][0]), kind, size, sh.rows(), sh.cols(), s0, s1)
    src.check(what)
    return (src, sh)


def _sequence_values(a: PythonObject, depth: Int, mut dims: List[Int],
                     mut values: List[Float64], mut ids: List[Int64], integer: Bool, mut leaf_depth: Int) raises:
    var b = native_loaded_module("builtins")
    if Bool(py=py_call(b.isinstance, a, b.list)) or Bool(py=py_call(b.isinstance, a, b.tuple)):
        var n = len(a)
        if depth >= 8 or n == 0:
            raise Error("ValueError: program input sequences must be nonempty and have rank at most eight")
        if depth == len(dims):
            dims.append(n)
        elif dims[depth] != n:
            raise Error("ValueError: program input sequences must be rectangular")
        for k in range(n):  # cpu-route: Python objects marshaled one element at a time before the runtime
            _sequence_values(a[k], depth + 1, dims, values, ids, integer, leaf_depth)
    else:
        if not program_numeric_scalar(a):
            raise Error("TypeError: program inputs require native numeric scalars without conversion hooks")
        if leaf_depth < 0:
            leaf_depth = depth
        if depth != len(dims) or depth != leaf_depth:
            raise Error("ValueError: program input sequences must be rectangular")
        if integer:
            if not is_integer_input(a):
                raise Error("TypeError: program ids must be integers")
            ids.append(Int64(py=a))
        else:
            values.append(Float64(py=a))


def program_input(value: PythonObject, integer: Bool) raises -> ProgramInput:
    """A typed ProgramInput from an array (its descriptor) or a nested
    sequence of Python scalars (converted once, owned by the result)."""
    var a = program_native_input(value)
    var values = List[Float64]()
    var ids = List[Int64]()
    var b = native_loaded_module("builtins")
    if is_ndarray(a):
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
    # The public varargs arrive as an exact tuple; raw packed callers must
    # also use an ordinary container, before length/index protocols are read.
    var cls = py_type(arrays)
    if cls is not py_type(Python.tuple()) and cls is not py_type(Python.list()):
        raise Error("TypeError: program inputs require a native tuple or list")
    var count = len(arrays)
    if count != p.n_in:
        raise Error("ValueError: this program takes " + String(p.n_in) + " input arrays, got " + String(count))
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
    return native_loaded_module("builtins").dict()


def _result(arrays: PythonObject) raises -> PythonObject:
    if len(arrays) == 1:
        return arrays[0]
    return py_call(native_loaded_module("builtins").tuple, arrays)


def _optimization_object(plan: ProgramOptimization) raises -> PythonObject:
    var out = _dict()
    py_setitem(out, "schema", "mojolearn-graph-passes/1")
    py_setitem(out, "scope", "inference only; original graph and derivative order unchanged")
    py_setitem(out, "storage_reuse", False)
    py_setitem(out, "rewritten_relu_elements_before", plan.relu_elements_before)
    py_setitem(out, "rewritten_relu_elements_after", plan.relu_elements_after)
    py_setitem(out, "liveness_scope", "logical values only; tape owns aliases and storage")
    var passes = Python.list()
    for k in range(len(plan.pass_names)):  # driver-loop: over graph transformation report
        var row = _dict()
        py_setitem(row, "name", plan.pass_names[k])
        py_setitem(row, "nodes_before", plan.before[k])
        py_setitem(row, "nodes_after", plan.after[k])
        py_setitem(row, "rewrites", plan.rewrites[k])
        _ = py_call(passes.append, row)
    py_setitem(out, "passes", passes)
    var origins = Python.list()
    for k in range(len(plan.source_nodes)):  # driver-loop: over source node provenance
        _ = py_call(origins.append, PythonObject(plan.source_nodes[k]))
    py_setitem(out, "source_nodes", origins)
    var liveness = Python.list()
    for k in range(len(plan.last_use)):  # driver-loop: over logical reference use metadata
        var row = _dict()
        py_setitem(row, "value", k)
        py_setitem(row, "last_use", plan.last_use[k])
        py_setitem(row, "uses", plan.uses[k])
        _ = py_call(liveness.append, row)
    py_setitem(out, "liveness", liveness)
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
    _ = arrays  # retain original external buffer owners until native replay returns
    var result = host_arrays_py(r.outputs)
    if not profiled:
        return _result(result)
    var s = ops_session[E, KIND](sid_i)
    var report = _dict()
    py_setitem(report, "executor", execution_profile_object(r.executor))
    py_setitem(report, "optimization", _optimization_object(plan.optimization))
    py_setitem(report, "timing", "host enqueue and synchronized wait; profiling serializes the queue")
    py_setitem(report, "original_nodes", plan.original_nodes)
    py_setitem(report, "executed_nodes", plan.program.n_nodes())
    py_setitem(report, "input_logical_bytes", r.upload_bytes)
    py_setitem(report, "input_device_bytes", r.input_bytes)
    py_setitem(report, "output_bytes", r.download_bytes)
    py_setitem(report, "parameter_bytes", r.parameter_bytes)
    py_setitem(report, "peak_tape_reserved_bytes", s[].program_trace.peak_tape_bytes)
    py_setitem(report, "upload_and_cast_ns", r.upload_ns)
    py_setitem(report, "download_and_allocate_ns", r.download_ns)
    py_setitem(report, "node_sync_count", s[].program_trace.sync_count)
    py_setitem(report, "node_sync_ns", s[].program_trace.sync_ns)
    var rows = Python.list()
    for j in range(len(s[].program_trace.nodes)):  # driver-loop: over diagnostic rows
        var row = _dict()
        py_setitem(row, "node", s[].program_trace.nodes[j])
        py_setitem(row, "source_node", plan.optimization.source_nodes[s[].program_trace.nodes[j]])
        py_setitem(row, "op", program_op_name(s[].program_trace.codes[j]))
        py_setitem(row, "shape", s[].program_trace.shapes[j])
        py_setitem(row, "output_bytes", s[].program_trace.output_bytes[j])
        py_setitem(row, "enqueue_ns", s[].program_trace.enqueue_ns[j])
        py_setitem(row, "wait_ns", s[].program_trace.wait_ns[j])
        _ = py_call(rows.append, row)
    py_setitem(report, "nodes", rows)
    var out = _dict()
    py_setitem(out, "outputs", _result(result))
    py_setitem(out, "profile", report)
    return out


# TOMBSTONE: the ops_* per-op session doors and lamb_step_py (dead native entries: reached only through their deleted exports) deleted 2026-10-26 by e4236afa3; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_x_sequence.patch


def ops_program_inspect_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, pid: PythonObject, optimize: PythonObject,
) raises -> PythonObject:
    var s = ops_session[E, KIND](Int(py=sid))
    var p = s[].program(Int(py=pid))
    var original_nodes = p.n_nodes()
    var optimization = program_optimize(p, Bool(py=optimize))
    p = optimization.program.copy()
    var report = _dict()
    py_setitem(report, "original_nodes", original_nodes)
    py_setitem(report, "nodes_removed", original_nodes - p.n_nodes())
    py_setitem(report, "input_shapes", _dims_out(p.in_shapes))
    py_setitem(report, "output_shapes", _dims_out(p.out_shapes))
    py_setitem(report, "parameter_shapes", _dims_out(p.param_shapes))
    py_setitem(report, "optimization", "structural_pass_pipeline" if Bool(py=optimize) else "none")
    py_setitem(report, "pass_report", _optimization_object(optimization))
    var rows = Python.list()
    for k in range(p.n_nodes()):  # driver-loop: over graph node metadata
        var row = _dict()
        py_setitem(row, "node", k)
        py_setitem(row, "source_node", optimization.source_nodes[k])
        py_setitem(row, "value", p.n_in + k)
        py_setitem(row, "op", program_op_name(p.code[k]))
        py_setitem(row, "code", p.code[k])
        var args = Python.list()
        var ints = Python.list()
        var floats = Python.list()
        for j in range(p.a_at[k], p.a_at[k + 1]):  # driver-loop: over input refs
            _ = py_call(args.append, PythonObject(p.args[j]))
        for j in range(p.i_at[k], p.i_at[k + 1]):  # driver-loop: over integer attributes
            _ = py_call(ints.append, PythonObject(p.ints[j]))
        for j in range(p.f_at[k], p.f_at[k + 1]):  # driver-loop: over float attributes
            _ = py_call(floats.append, PythonObject(p.fs[j]))
        py_setitem(row, "inputs", args)
        py_setitem(row, "integers", ints)
        py_setitem(row, "floats", floats)
        _ = py_call(rows.append, row)
    py_setitem(report, "nodes", rows)
    var outputs = Python.list()
    for k in range(len(p.outs)):  # driver-loop: over output refs
        _ = py_call(outputs.append, PythonObject(p.outs[k]))
    py_setitem(report, "outputs", outputs)
    return report


def _integer(v: PythonObject, what: String, low: Int) raises -> Int:
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
    var b = native_loaded_module("builtins")
    var no_target = targets is None
    if (kind == 0) != no_target:
        raise Error("ValueError: omit targets only when loss='output'")
    var count = _integer(epochs, "epochs", 1)
    var batch = _integer(batch_size, "batch_size", 1)
    var opt = _integer(optimizer, "optimizer", 1)
    var seed_word = uint64_input(seed, "seed")
    var explicit = parameters is not None
    var selected = List[Int]()
    if explicit:
        if is_integer_input(parameters):
            selected.append(_integer(parameters, "parameter", 1))
        else:
            # Exact native sequences/arrays only; an Array is unwrapped before
            # len/index so its Python convenience methods never run here.
            var selected_values = program_native_input(parameters)
            for k in range(len(selected_values)):  # small-loop(selected_values: selected parameter handles): argument parsing, no data
                selected.append(_integer(selected_values[k], "parameter", 1))
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
    _ = arrays  # descriptors borrow external inputs through the whole native fit
    _ = targets
    _ = curve
    var out = Python.list()
    _ = py_call(out.append, curve)
    _ = py_call(out.append, PythonObject(count))
    if condition_id > 0:
        var counts = b.dict()
        py_setitem(counts, "true", branches[0])
        py_setitem(counts, "false", branches[1])
        _ = py_call(out.append, counts)
    return out


# ------------------------------------------------------------ ImportedProgram.run
# `mojolearn.importers.ImportedProgram.run` as one native call: the inputs by
# position or by name, each checked against the model's declared shape, a
# rank-0 or rank-1 model value held as the session's (1, 1) or (1, n) row
# (the same descriptor the former Python reshape view gave: no copy), the
# program run, and every output returned in its declared shape and dtype
# (an 'int32' output keeps the device's words as int32 items).

def _imported_repr(value: PythonObject) raises -> String:
    return String(py_call(native_loaded_module("builtins").repr, value))


def _imported_tuple(dims: List[Int]) raises -> PythonObject:
    var out = Python.list()
    for k in range(len(dims)):  # small-loop(dims: a value's rank, at most eight): shape metadata, never array elements
        _ = py_call(out.append, PythonObject(dims[k]))
    return py_call(native_loaded_module("builtins").tuple, out)


def _imported_session_dims(dims: List[Int]) -> List[Int]:
    # importers/_table.session_shape: a session tensor has at least two axes,
    # a scalar is (1, 1) and a [n] vector (1, n).
    var out = List[Int]()
    for _ in range(2 - len(dims)):  # small-loop(2: the padded leading axes): at most two shape entries
        out.append(1)
    for k in range(len(dims)):  # small-loop(dims: rank, at most eight): shape metadata, never array elements
        out.append(dims[k])
    return out^


def _imported_capture_input(value: PythonObject, integer: Bool, mut dims: List[Int]) raises -> ProgramInput:
    # Exact admission precedes every protocol read. Capture ndarray shape,
    # strides, dtype and address once, then use only native descriptors.
    var a = program_native_input(value)
    var values = List[Float64]()
    var ids = List[Int64]()
    if is_ndarray(a):
        var shape = a.shape
        var strides = a.strides
        var dtype = a.dtype
        var rank = len(shape)
        if rank > 8:
            raise Error("ValueError: tensor rank must be at most 8")
        for k in range(rank):  # small-loop(rank: admitted ndarray axes): at most eight shape entries
            dims.append(Int(py=shape[k]))
        var sh = ShapeN.of(_imported_session_dims(dims))
        if sh.count() == 0:
            raise Error("ValueError: input must not be empty")
        var size = Int(py=dtype.itemsize)
        if not Bool(py=dtype.isnative):
            raise Error("TypeError: numeric program input must be in native byte order")
        var kind = cast_in_kind(Int(py=dtype.num), size, "numeric program input")
        if integer and kind != CAST_KIND_SIGNED and kind != CAST_KIND_UNSIGNED:
            raise Error("TypeError: program input holds ids and must be integers (an integer array)")
        var s0 = sh.cols() * size
        var s1 = size
        if rank == 1:
            s1 = Int(py=strides[0])
        elif rank == 2:
            s0 = Int(py=strides[0])
            s1 = Int(py=strides[1])
        elif rank > 2 and not Bool(py=a.flags.c_contiguous):
            raise Error("ValueError: tensors above rank two must be C-contiguous")
        var src = CastIn(Int(py=a.__array_interface__["data"][0]), kind, size, sh.rows(), sh.cols(), s0, s1)
        src.check("numeric program input")
        return ProgramInput(src, sh, values^, ids^)
    var leaf_depth = -1
    _sequence_values(a, 0, dims, values, ids, integer, leaf_depth)
    var sh = ShapeN.of(_imported_session_dims(dims))
    if integer:
        var src = CastIn(Int(ids.unsafe_ptr()), CAST_KIND_SIGNED, 8, sh.rows(), sh.cols(), sh.cols() * 8, 8)
        return ProgramInput(src, sh, values^, ids^)
    var src = CastIn(Int(values.unsafe_ptr()), CAST_KIND_FLOAT, 8, sh.rows(), sh.cols(), sh.cols() * 8, 8)
    return ProgramInput(src, sh, values^, ids^)


def ops_imported_run_py(
    program: PythonObject, arrays: PythonObject, named: PythonObject,
) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var metadata = imported_program_snapshot(program)
    var declared = metadata[2]
    var outputs = metadata[3]
    if (py_type(arrays) is not b.tuple and py_type(arrays) is not b.list) or py_type(named) is not b.dict:
        raise Error("TypeError: imported inputs require native positional and keyword containers")
    var keys = py_call(b.list, named)
    for k in range(len(keys)):  # driver-loop: over exact keyword names before hash/comparison
        if py_type(keys[k]) is not b.str:
            raise Error("TypeError: imported input names require exact strings")
    var given = arrays
    if len(named) > 0:
        if len(arrays) > 0:
            raise Error("ValueError: pass the inputs either in order or by name, not both")
        var names = Python.list()
        for k in range(len(declared)):  # driver-loop: over the model's declared inputs
            _ = py_call(names.append, declared[k][0])
        for k in range(len(keys)):  # driver-loop: over the keyword argument names
            if not py_contains(names, keys[k]):
                raise Error("ValueError: the model has no input " + _imported_repr(keys[k]) + " (inputs: "
                            + String(PythonObject(", ").join(names)) + ")")
        var ordered = Python.list()
        for k in range(len(names)):  # driver-loop: over the model's declared inputs
            if not py_contains(named, names[k]):
                raise Error("ValueError: input " + _imported_repr(names[k]) + " is missing")
            _ = py_call(ordered.append, named[names[k]])
        given = ordered
    if len(given) != len(declared):
        raise Error("ValueError: the model takes " + String(len(declared)) + " inputs, got " + String(len(given)))
    var request = NativeProgramHostRun()
    imported_host_provider_request(metadata, request, 0)
    ref plan = request.plan
    if len(declared) != plan.program.n_in:
        raise Error("ValueError: this program takes " + String(plan.program.n_in) + " input arrays, got " + String(len(declared)))
    var inputs = List[ProgramInput]()
    for k in range(len(declared)):  # driver-loop: over input descriptors captured once before replay
        var got = List[Int]()
        var item = _imported_capture_input(given[k], plan.program.in_kinds[k] == PROGRAM_IN_IDS, got)
        var want = declared[k][1]
        var same = len(got) == len(want)
        for j in range(min(len(got), len(want))):  # small-loop(want: declared input rank): at most eight shape entries
            same = same and got[j] == Int(py=want[j])
        if not same:
            raise Error("ValueError: input " + _imported_repr(declared[k][0]) + " must have the shape "
                        + _imported_repr(want) + ", got " + _imported_repr(_imported_tuple(got)))
        if not same_shape(item.shape, plan.program.in_shapes[k]):
            raise Error("ValueError: input " + String(k) + " must keep its captured shape "
                        + shape_text(plan.program.in_shapes[k]) + "; got " + shape_text(item.shape))
        inputs.append(item^)
    request.inputs = inputs^
    imported_host_provider_request(metadata, request, 1)
    ref r = request.result
    var out = Python.list()
    for k in range(len(r.outputs)):  # driver-loop: over the program's outputs (result packing, no data)
        var want = List[Int]()
        var shape = py_call(b.tuple, outputs[k][1])
        var count = 1
        for j in range(len(shape)):  # small-loop(shape: a declared output's rank): shape metadata, never array elements
            want.append(Int(py=shape[j]))
            count *= want[j]
        if count != r.outputs[k].shape.count():
            raise Error("ValueError: cannot reshape output " + String(k) + " of shape " + shape_text(r.outputs[k].shape)
                        + " into shape " + _imported_repr(shape))
        var dtype = String(outputs[k][2])
        var a = output_array_dims(want, "int32" if dtype == "int32" else "float32")
        if count > 0:
            # the downloaded words as they are (an 'int32' output's words are its int32 items)
            memcpy(dest=FP(unsafe_from_address=Int(py=a.__array_interface__["data"][0])),
                   src=r.outputs[k].data.unsafe_ptr(), count=count)
        _ = py_call(out.append, a)
    return _result(out)
