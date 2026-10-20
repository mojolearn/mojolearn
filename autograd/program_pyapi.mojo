# SPDX-License-Identifier: Apache-2.0
# Array ownership at the native Python binding boundary. NumPy is used only
# to allocate returned ABI buffers; conversion and every computed value use
# the executor. No user callbacks drive execution, epochs or batches.
from std.python import Python, PythonObject
from sequence.exec_trait import Exec
from sequence.ops import FP
from core.cast_in import CastIn, cast_in_kind
from autograd.rankn import ShapeN
from autograd.tape import Tape
from autograd.cast_ops import upload_cast, upload_cast_i32
from autograd.ops_api import ops_session
from autograd.program import PROGRAM_IN_IDS
from autograd.program_runtime import replay_values, parameter_buffers, train_program
from autograd.pyapi import _input_handles, _seed, ival, fval


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


def output_array(shape: ShapeN) raises -> PythonObject:
    var dims = Python.list()
    for k in range(shape.ndim):  # small-loop(shape.ndim: tensor rank): at most eight shape words
        dims.append(PythonObject(shape.d[k]))
    # Allocation only: all output values are written by the native executor.
    return Python.import_module("numpy").empty(dims, dtype="float32")


def ops_program_dynamic_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, pid: PythonObject, arrays: PythonObject,
) raises -> PythonObject:
    var s = ops_session[E, KIND](Int(py=sid))
    var p = s[].program(Int(py=pid))
    if s[].tape.is_recording() or len(arrays) != p.n_in:
        raise Error("dynamic program execution requires an inference session and its input arrays")
    var buffers = List[FP]()
    var shapes = List[ShapeN]()
    for k in range(len(arrays)):  # driver-loop: over program input arrays
        var parsed = array_input(arrays[k], "program input")
        var src = parsed[0]
        var sh = parsed[1]
        if sh.ndim != p.in_shapes[k].ndim:
            raise Error("dynamic program input must keep its captured rank")
        var buf = s[].ex.alloc(sh.count())
        if p.in_kinds[k] == PROGRAM_IN_IDS:
            upload_cast_i32(s[].ex, buf, src, "program ids")
        else:
            upload_cast(s[].ex, buf, src)
        buffers.append(buf)
        shapes.append(sh)
    var params = parameter_buffers(s[], p)
    var trainable = List[Bool](length=len(params), fill=False)
    var held = Tape()
    swap(held, s[].prog_tape)
    swap(held, s[].tape)
    var temps = List[Int]()
    var phs = List[Int]()
    var result = Python.list()
    try:
        s[].tape.no_grad()
        s[].tape.reset()
        var outs = replay_values(s[], p, buffers, shapes, params, trainable, temps, phs)
        for k in range(len(outs)):  # driver-loop: over returned arrays
            var sh = s[].shapes[s[]._slot(outs[k])]
            var a = output_array(sh)
            s[].download(outs[k], FP(unsafe_from_address=Int(py=a.ctypes.data)))
            result.append(a)
    except err:
        s[]._drop_temps(temps)
        swap(held, s[].tape)
        swap(held, s[].prog_tape)
        for k in range(len(buffers)):  # driver-loop: over input buffers
            s[].ex.free(buffers[k])
        raise err
    s[]._drop_temps(temps)
    swap(held, s[].tape)
    swap(held, s[].prog_tape)
    for k in range(len(buffers)):  # driver-loop: over input buffers
        s[].ex.free(buffers[k])
    if len(result) == 1:
        return result[0]
    return Python.import_module("builtins").tuple(result)


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
