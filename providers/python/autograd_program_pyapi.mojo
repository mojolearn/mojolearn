# SPDX-License-Identifier: Apache-2.0
# Array ownership at the native Python binding boundary. NumPy is used only
# to allocate returned ABI buffers; conversion and every computed value use
# the executor. No user callbacks drive execution, epochs or batches.
from std.python import Python, PythonObject
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
    var b = Python.import_module("builtins")
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
    var b = Python.import_module("builtins")
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
    return Python.import_module("builtins").dict()


def _result(arrays: PythonObject) raises -> PythonObject:
    if len(arrays) == 1:
        return arrays[0]
    return py_call(Python.import_module("builtins").tuple, arrays)


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
    var no_target = String(py=py_type(targets).__name__) == "NoneType"
    if (kind == 0) != no_target:
        raise Error("ValueError: omit targets only when loss='output'")
    var count = _integer(epochs, "epochs", 1)
    var batch = _integer(batch_size, "batch_size", 1)
    var opt = _integer(optimizer, "optimizer", 1)
    var seed_word = uint64_input(seed, "seed")
    var explicit = String(py=py_type(parameters).__name__) != "NoneType"
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
    return String(py_call(Python.import_module("builtins").repr, value))


def _imported_tuple(dims: List[Int]) raises -> PythonObject:
    var out = Python.list()
    for k in range(len(dims)):  # small-loop(dims: a value's rank, at most eight): shape metadata, never array elements
        _ = py_call(out.append, PythonObject(dims[k]))
    return py_call(Python.import_module("builtins").tuple, out)


def _imported_is_array(a: PythonObject) raises -> Bool:
    # a NumPy ndarray (subclasses included): read through its own buffer
    return is_ndarray(a)


def _imported_given_dims(a: PythonObject, name: PythonObject) raises -> List[Int]:
    """The shape the former `np.asarray(x)` had: an ndarray's own; a nested
    list or tuple's down its first items (rectangularity is checked when its
    values are read); a number's (a Python or NumPy scalar) (). Another
    array-like object (a torch tensor, a DataFrame) is refused: the caller
    converts it to a NumPy array first."""
    var b = Python.import_module("builtins")
    var dims = List[Int]()
    if _imported_is_array(a):
        var shape = py_call(b.tuple, a.shape)
        for k in range(len(shape)):  # small-loop(shape: an array's rank): shape metadata, never array elements
            dims.append(Int(py=shape[k]))
        return dims^
    var item = a
    while Bool(py=py_call(b.isinstance, item, b.list)) or Bool(py=py_call(b.isinstance, item, b.tuple)):  # driver-loop: down a nested sequence's first items (its rank)
        var n = len(item)
        dims.append(n)
        if n == 0 or len(dims) > 8:
            return dims^
        item = item[0]
    var numpy_scalar = String(py=py_type(a).__module__) == "numpy"
    if len(dims) == 0 and not numpy_scalar and (Bool(py=py_call(b.hasattr, a, "__array__"))
                                                       or Bool(py=py_call(b.hasattr, a, "__array_interface__"))):
        raise Error("TypeError: input " + _imported_repr(name) + " must be a NumPy array, a nested list of numbers "
                    + "or a number, got " + String(py_type(a).__name__) + " (convert it with numpy.asarray, "
                    + "or .detach().cpu().numpy() for a torch tensor)")
    return dims^


def _imported_session_dims(dims: List[Int]) -> List[Int]:
    # importers/_table.session_shape: a session tensor has at least two axes,
    # a scalar is (1, 1) and a [n] vector (1, n).
    var out = List[Int]()
    for _ in range(2 - len(dims)):  # small-loop(2: the padded leading axes): at most two shape entries
        out.append(1)
    for k in range(len(dims)):  # small-loop(dims: rank, at most eight): shape metadata, never array elements
        out.append(dims[k])
    return out^


def _imported_input(a: PythonObject, dims: List[Int], integer: Bool, what: String) raises -> ProgramInput:
    """One admitted input as the program reads it: an array by its own
    buffer (a rank-0 or rank-1 one with the row descriptor of its former
    reshape view, rank two and above exactly as `array_input`), a nested
    sequence or number converted once into the session shape."""
    var values = List[Float64]()
    var ids = List[Int64]()
    var sdims = _imported_session_dims(dims)
    if _imported_is_array(a):
        if len(dims) >= 2:
            var parsed = array_input(a, what)
            if integer and parsed[0].kind != CAST_KIND_SIGNED and parsed[0].kind != CAST_KIND_UNSIGNED:
                raise Error("TypeError: program input holds ids and must be integers (an integer array)")
            return ProgramInput(parsed[0], parsed[1], values^, ids^)
        if Int(py=a.size) == 0:
            raise Error("ValueError: input must not be empty")
        var size = Int(py=a.itemsize)
        if not Bool(py=a.dtype.isnative):
            raise Error("TypeError: " + what + " must be in native byte order")
        var kind: Int
        try:
            kind = cast_in_kind(Int(py=a.dtype.num), size, what)
        except error:
            raise Error("TypeError: " + String(error))
        if integer and kind != CAST_KIND_SIGNED and kind != CAST_KIND_UNSIGNED:
            raise Error("TypeError: program input holds ids and must be integers (an integer array)")
        var cols = sdims[1]
        var s1 = size if len(dims) == 0 else Int(py=a.strides[0])
        var src = CastIn(Int(py=a.__array_interface__["data"][0]), kind, size, 1, cols, cols * size, s1)
        src.check(what)
        return ProgramInput(src, ShapeN.of(sdims), values^, ids^)
    var got = List[Int]()
    var leaf_depth = -1
    _sequence_values(a, 0, got, values, ids, integer, leaf_depth)
    var shape = ShapeN.of(sdims)
    if integer:
        var id_src = CastIn(Int(ids.unsafe_ptr()), CAST_KIND_SIGNED, 8, shape.rows(), shape.cols(), shape.cols() * 8, 8)
        return ProgramInput(id_src, shape, values^, ids^)
    var src = CastIn(Int(values.unsafe_ptr()), CAST_KIND_FLOAT, 8, shape.rows(), shape.cols(), shape.cols() * 8, 8)
    return ProgramInput(src, shape, values^, ids^)


def ops_imported_run_py[E: Exec & Movable & Deinitable, KIND: Int](
    program: PythonObject, arrays: PythonObject, named: PythonObject,
) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var declared = program.inputs
    var outputs = program.outputs
    var given = arrays
    if len(named) > 0:
        if len(arrays) > 0:
            raise Error("ValueError: pass the inputs either in order or by name, not both")
        var names = Python.list()
        for k in range(len(declared)):  # driver-loop: over the model's declared inputs
            _ = py_call(names.append, declared[k][0])
        var keys = py_call(b.list, named)
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
    # every shape is checked before the program is touched, as before
    var dims = List[List[Int]]()
    for k in range(len(declared)):  # driver-loop: over the model's declared inputs
        var want = List[Int]()
        var shape = py_call(b.tuple, declared[k][1])
        for j in range(len(shape)):  # small-loop(shape: a declared input's rank): shape metadata, never array elements
            want.append(Int(py=shape[j]))
        var got = _imported_given_dims(given[k], declared[k][0])
        var same = len(got) == len(want)
        for j in range(min(len(got), len(want))):  # small-loop(want: a declared input's rank): shape metadata, never array elements
            same = same and got[j] == want[j]
        if not same:
            raise Error("ValueError: input " + _imported_repr(declared[k][0]) + " must have the shape "
                        + _imported_repr(shape) + ", got " + _imported_repr(_imported_tuple(got)))
        dims.append(want^)
    var session = program._s
    if session._id is None:
        raise Error("ValueError: this session is closed")
    var sid = Int(py=session._id)
    var plan = program_execute_prepare[E, KIND](sid, Int(py=program._id), sid, False, False)
    if len(declared) != plan.program.n_in:
        raise Error("ValueError: this program takes " + String(plan.program.n_in) + " input arrays, got " + String(len(declared)))
    var inputs = List[ProgramInput]()
    for k in range(len(declared)):  # driver-loop: over input descriptors, owners retained through replay
        var item = _imported_input(given[k], dims[k], plan.program.in_kinds[k] == PROGRAM_IN_IDS, "numeric program input")
        if not same_shape(item.shape, plan.program.in_shapes[k]):
            raise Error("ValueError: input " + String(k) + " must keep its captured shape "
                        + shape_text(plan.program.in_shapes[k]) + "; got " + shape_text(item.shape))
        inputs.append(item^)
    var r = program_execute_run[E, KIND](sid, plan, inputs^, False)
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
