# SPDX-License-Identifier: Apache-2.0
# Array ownership at the native Python binding boundary. NumPy is used only
# to allocate returned ABI buffers; conversion and every computed value use
# the executor. No user callbacks drive execution, epochs or batches.
from std.python import Python, PythonObject
from std.time import perf_counter_ns
from sequence.exec_trait import Exec
from sequence.ops import FP
from core.cast_in import CastIn, cast_in_kind, CAST_KIND_FLOAT, CAST_KIND_SIGNED, CAST_KIND_UNSIGNED
from autograd.rankn import ShapeN
from autograd.tape import Tape
from autograd.cast_ops import upload_cast, upload_cast_i32
from autograd.ops_api import ops_session
from autograd.program import OpsProgram, PROGRAM_IN_IDS, same_shape, shape_text
from autograd.program_tools import ProgramTrace, program_cse, program_op_name
from autograd.program_runtime import replay_values, parameter_buffers, train_program
from autograd.pyapi import _input_handles, _seed, ival, fval, _dims_out


def array_input(a: PythonObject, what: String) raises -> Tuple[CastIn, ShapeN]:
    var nd = Int(py=a.ndim)
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
    var kind = cast_in_kind(Int(py=a.dtype.num), size, what)
    var s0 = sh.cols() * size
    var s1 = size
    if nd == 1:
        s0 = Int(py=a.strides[0])
    elif nd == 2:
        s0 = Int(py=a.strides[0])
        s1 = Int(py=a.strides[1])
    elif nd > 2 and not Bool(py=a.flags.c_contiguous):
        raise Error(what + ": tensors above rank two must be C-contiguous")
    var src = CastIn(Int(py=a.ctypes.data), kind, size, sh.rows(), sh.cols(), s0, s1)
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
            if not Bool(py=b.isinstance(a, b.int)):
                raise Error("TypeError: program ids must be integers")
            ids.append(Int64(py=a))
        else:
            values.append(Float64(py=a))


struct ProgramInput(Movable):
    var src: CastIn
    var shape: ShapeN
    var values: List[Float64]
    var ids: List[Int64]

    def __init__(out self, a: PythonObject, integer: Bool) raises:
        self.values = List[Float64]()
        self.ids = List[Int64]()
        var b = Python.import_module("builtins")
        if Bool(py=b.hasattr(a, "dtype")) and Bool(py=b.hasattr(a, "strides")):
            var parsed = array_input(a, "program input")
            self.src = parsed[0]
            self.shape = parsed[1]
            if integer and self.src.kind != CAST_KIND_SIGNED and self.src.kind != CAST_KIND_UNSIGNED:
                raise Error("TypeError: program input holds ids and must be an integer array")
        else:
            var dims = List[Int]()
            var leaf_depth = -1
            _sequence_values(a, 0, dims, self.values, self.ids, integer, leaf_depth)
            if len(dims) == 0:
                dims.append(1)
            if len(dims) == 1:
                dims.append(1)
            self.shape = ShapeN.of(dims)
            if integer:
                self.src = CastIn(Int(self.ids.unsafe_ptr()), CAST_KIND_SIGNED, 8,
                                  self.shape.rows(), self.shape.cols(), self.shape.cols() * 8, 8)
            else:
                self.src = CastIn(Int(self.values.unsafe_ptr()), CAST_KIND_FLOAT, 8,
                                  self.shape.rows(), self.shape.cols(), self.shape.cols() * 8, 8)


def program_inputs(p: OpsProgram, arrays: PythonObject, dynamic: Bool) raises -> List[ProgramInput]:
    if len(arrays) != p.n_in:
        raise Error("ValueError: this program takes " + String(p.n_in) + " input arrays, got " + String(len(arrays)))
    var out = List[ProgramInput]()
    for k in range(p.n_in):  # driver-loop: input descriptors, owners retained through replay
        var item = ProgramInput(arrays[k], p.in_kinds[k] == PROGRAM_IN_IDS)
        if (dynamic and item.shape.ndim != p.in_shapes[k].ndim) or (not dynamic and not same_shape(item.shape, p.in_shapes[k])):
            raise Error("ValueError: input " + String(k) + " must keep its captured "
                        + ("rank" if dynamic else "shape") + " " + shape_text(p.in_shapes[k])
                        + "; got " + shape_text(item.shape))
        out.append(item^)
    return out^


def output_array(shape: ShapeN) raises -> PythonObject:
    var dims = Python.list()
    for k in range(shape.ndim):  # small-loop(shape.ndim: tensor rank): at most eight shape words
        dims.append(PythonObject(shape.d[k]))
    # Allocation only: all output values are written by the native executor.
    return Python.import_module("numpy").empty(dims, dtype="float32")


def _dict() raises -> PythonObject:
    return Python.import_module("builtins").dict()


def _result(arrays: PythonObject) raises -> PythonObject:
    if len(arrays) == 1:
        return arrays[0]
    return Python.import_module("builtins").tuple(arrays)


def ops_program_execute_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, pid: PythonObject, arrays: PythonObject,
    dynamic_shapes: PythonObject, optimize: PythonObject, profile: PythonObject,
) raises -> PythonObject:
    var s = ops_session[E, KIND](Int(py=sid))
    var p = s[].program(Int(py=pid))
    if s[].tape.is_recording():
        raise Error("program execution requires an inference session")
    var original_nodes = p.n_nodes()
    if Bool(py=optimize):
        p = program_cse(p)
    var inputs = program_inputs(p, arrays, Bool(py=dynamic_shapes))
    # Validate every descriptor and parameter before allocation. Earlier
    # dynamic replay leaked input buffers on a later input/parameter error.
    var params = parameter_buffers(s[], p)
    var trainable = List[Bool](length=len(params), fill=False)
    var buffers = List[FP]()
    var shapes = List[ShapeN]()
    var upload_bytes = 0
    var input_bytes = 0
    var parameter_bytes = 0
    for j in range(len(p.param_shapes)):  # driver-loop: parameter storage accounting
        parameter_bytes += p.param_shapes[j].count() * 4
    s[].program_trace = ProgramTrace()
    s[].program_trace.enabled = Bool(py=profile)
    var started = Int(perf_counter_ns())
    try:
        for k in range(len(inputs)):  # driver-loop: native uploads
            var sh = inputs[k].shape
            var buf = s[].ex.alloc(sh.count())
            buffers.append(buf)
            shapes.append(sh)
            input_bytes += sh.count() * 4
            upload_bytes += sh.count() * inputs[k].src.itemsize
            if p.in_kinds[k] == PROGRAM_IN_IDS:
                upload_cast_i32(s[].ex, buf, inputs[k].src, "program ids")
            else:
                upload_cast(s[].ex, buf, inputs[k].src)
        if Bool(py=profile):
            s[].ex.sync()
    except err:
        s[].program_trace.enabled = False
        for k in range(len(buffers)):  # driver-loop: owners acquired before a failed upload
            s[].ex.free(buffers[k])
        raise err
    var upload_ns = Int(perf_counter_ns()) - started
    var held = Tape()
    swap(held, s[].prog_tape)
    swap(held, s[].tape)
    var temps = List[Int]()
    var phs = List[Int]()
    var result = Python.list()
    var download_bytes = 0
    var download_ns = 0
    try:
        s[].tape.no_grad()
        s[].tape.reset()
        var outs = replay_values(s[], p, buffers, shapes, params, trainable, temps, phs)
        started = Int(perf_counter_ns())
        for k in range(len(outs)):  # driver-loop: returned arrays
            var sh = s[].shapes[s[]._slot(outs[k])]
            var a = output_array(sh)
            s[].download(outs[k], FP(unsafe_from_address=Int(py=a.ctypes.data)))
            download_bytes += sh.count() * 4
            result.append(a)
        download_ns = Int(perf_counter_ns()) - started
    except err:
        s[].program_trace.enabled = False
        s[]._drop_temps(temps)
        swap(held, s[].tape)
        swap(held, s[].prog_tape)
        for k in range(len(buffers)):  # driver-loop: input owners
            s[].ex.free(buffers[k])
        raise err
    s[].program_trace.enabled = False
    s[]._drop_temps(temps)
    swap(held, s[].tape)
    swap(held, s[].prog_tape)
    for k in range(len(buffers)):  # driver-loop: input owners
        s[].ex.free(buffers[k])
    if not Bool(py=profile):
        return _result(result)
    var report = _dict()
    report.__setitem__("timing", "host enqueue and synchronized wait; profiling serializes the queue")
    report.__setitem__("original_nodes", original_nodes)
    report.__setitem__("executed_nodes", p.n_nodes())
    report.__setitem__("input_logical_bytes", upload_bytes)
    report.__setitem__("input_device_bytes", input_bytes)
    report.__setitem__("output_bytes", download_bytes)
    report.__setitem__("parameter_bytes", parameter_bytes)
    report.__setitem__("peak_tape_reserved_bytes", s[].program_trace.peak_tape_bytes)
    report.__setitem__("upload_and_cast_ns", upload_ns)
    report.__setitem__("download_and_allocate_ns", download_ns)
    report.__setitem__("node_sync_count", s[].program_trace.sync_count)
    report.__setitem__("node_sync_ns", s[].program_trace.sync_ns)
    var rows = Python.list()
    for j in range(len(s[].program_trace.nodes)):  # driver-loop: diagnostic rows
        var row = _dict()
        row.__setitem__("node", s[].program_trace.nodes[j])
        row.__setitem__("op", program_op_name(s[].program_trace.codes[j]))
        row.__setitem__("shape", s[].program_trace.shapes[j])
        row.__setitem__("output_bytes", s[].program_trace.output_bytes[j])
        row.__setitem__("enqueue_ns", s[].program_trace.enqueue_ns[j])
        row.__setitem__("wait_ns", s[].program_trace.wait_ns[j])
        rows.append(row)
    report.__setitem__("nodes", rows)
    var out = _dict()
    out.__setitem__("outputs", _result(result))
    out.__setitem__("profile", report)
    return out


def ops_program_dynamic_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, pid: PythonObject, arrays: PythonObject,
) raises -> PythonObject:
    # Keep the previously exported entry ABI stable.
    return ops_program_execute_py[E, KIND](sid, pid, arrays, PythonObject(True), PythonObject(False), PythonObject(False))


def ops_program_inspect_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, pid: PythonObject, optimize: PythonObject,
) raises -> PythonObject:
    var s = ops_session[E, KIND](Int(py=sid))
    var p = s[].program(Int(py=pid))
    var original_nodes = p.n_nodes()
    if Bool(py=optimize):
        p = program_cse(p)
    var report = _dict()
    report.__setitem__("original_nodes", original_nodes)
    report.__setitem__("nodes_removed", original_nodes - p.n_nodes())
    report.__setitem__("input_shapes", _dims_out(p.in_shapes))
    report.__setitem__("output_shapes", _dims_out(p.out_shapes))
    report.__setitem__("parameter_shapes", _dims_out(p.param_shapes))
    report.__setitem__("optimization", "exact_duplicate_elimination" if Bool(py=optimize) else "none")
    var rows = Python.list()
    for k in range(p.n_nodes()):  # driver-loop: graph node metadata
        var row = _dict()
        row.__setitem__("node", k)
        row.__setitem__("value", p.n_in + k)
        row.__setitem__("op", program_op_name(p.code[k]))
        row.__setitem__("code", p.code[k])
        var args = Python.list()
        var ints = Python.list()
        var floats = Python.list()
        for j in range(p.a_at[k], p.a_at[k + 1]):  # driver-loop: input refs
            args.append(PythonObject(p.args[j]))
        for j in range(p.i_at[k], p.i_at[k + 1]):  # driver-loop: integer attributes
            ints.append(PythonObject(p.ints[j]))
        for j in range(p.f_at[k], p.f_at[k + 1]):  # driver-loop: float attributes
            floats.append(PythonObject(p.fs[j]))
        row.__setitem__("inputs", args)
        row.__setitem__("integers", ints)
        row.__setitem__("floats", floats)
        rows.append(row)
    report.__setitem__("nodes", rows)
    var outputs = Python.list()
    for k in range(len(p.outs)):  # driver-loop: output refs
        outputs.append(PythonObject(p.outs[k]))
    report.__setitem__("outputs", outputs)
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
    for k in range(len(arrays)):  # driver-loop: over program input arrays
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
                  _seed(ip, 5), FP(unsafe_from_address=Int(py=curve.ctypes.data)))
    return curve
