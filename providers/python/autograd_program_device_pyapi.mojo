# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# `ImportedProgram.run` on caller device buffers (the torch.compile backend's
# device route, python/mojolearn/adapters/_torch_compile.py): each input is a
# C-contiguous buffer already on this binding's GPU, named by its device
# address, shape and dtype; each output is a caller-allocated device buffer
# of the declared output's shape and dtype. Nothing crosses the host.
#
# Same bits as ops_imported_run_py (program_pyapi.mojo) by construction: an
# input reaches the program's input buffer through the same two steps
# upload_cast / upload_cast_i32 (autograd/cast_ops.mojo) take, with the host
# upload replaced by one device-to-device copy of the same bytes:
#   * float32 (or int32 for an ids input) is copied word for word, as the host
#     path uploads it word for word;
#   * any other accepted dtype has its bytes copied into a zero-filled word
#     buffer (the zero-padded tail word upload_bytes builds) and the same
#     OP_CAST_F32 / OP_CAST_I32 launch converts it, with the same CastIn
#     descriptor the host path builds for a C-contiguous array.
# Each output is the program's device words copied as they are, which is
# what the host path downloads and memcpys into its float32 / int32 array.
#
# CUDA and ROCm only. Apple is refused: Metal buffers are objects, not
# addresses (memory: METAL POINTER TRAPS: NO INT->POINTER), and PyTorch
# exposes no supported raw pointer for an MPS tensor, so the backend keeps the
# host route there. Stream order: the caller drains its own stream before
# this call (the inputs are written, the outputs' storage is free); this call
# drains the session's queue before it returns (the outputs are written).
# Device: the session's shared sequence context, the process's GPU 0; the
# caller routes only tensors on that GPU here.
from std.python import Python, PythonObject
from std.sys.info import has_apple_gpu_accelerator
from bindings.py_balanced import py_call
from core.cast_in import CastIn, CAST_KIND_FLOAT, CAST_KIND_SIGNED, CAST_KIND_UNSIGNED, CAST_KIND_BOOL
from autograd.rankn import ShapeN
from autograd.program_device_execution import program_device_execute
from providers.python.autograd_program_pyapi import _imported_repr, _imported_tuple, _imported_session_dims


def _device_kind(dtype: String, what: String) raises -> Tuple[Int, Int]:
    """(CastIn kind, item size) of a dtype name (torch's names, which are
    NumPy's): the same kinds cast_in_kind gives the host path's arrays."""
    if dtype == "float32":
        return (CAST_KIND_FLOAT, 4)
    if dtype == "float16":
        return (CAST_KIND_FLOAT, 2)
    if dtype == "float64":
        return (CAST_KIND_FLOAT, 8)
    if dtype == "int8":
        return (CAST_KIND_SIGNED, 1)
    if dtype == "int16":
        return (CAST_KIND_SIGNED, 2)
    if dtype == "int32":
        return (CAST_KIND_SIGNED, 4)
    if dtype == "int64":
        return (CAST_KIND_SIGNED, 8)
    if dtype == "uint8":
        return (CAST_KIND_UNSIGNED, 1)
    if dtype == "bool":
        return (CAST_KIND_BOOL, 1)
    raise Error("TypeError: " + what + ": dtype " + dtype + " is not supported; pass a float16, float32, "
                + "float64, integer or bool tensor")


def ops_imported_run_device_py[KIND: Int](
    program: PythonObject, inputs: PythonObject, outputs: PythonObject,
) raises -> PythonObject:
    """Run `program` (an ImportedProgram) on device buffers.

    `inputs`: one (device address, shape tuple, dtype name) per declared
    input, in order, each C-contiguous on this binding's GPU. `outputs`: one
    device address per declared output, of its declared shape, float32 (an
    'int32' output: int32). Returns None; the outputs are written when it
    returns."""
    comptime if has_apple_gpu_accelerator():
        raise Error("ValueError: ops_imported_run_device takes CUDA or ROCm device buffers; on Apple GPUs "
                    + "pass host arrays to ops_imported_run")
    var b = Python.import_module("builtins")
    var declared = program.inputs
    var declared_out = program.outputs
    if len(inputs) != len(declared):
        raise Error("ValueError: the model takes " + String(len(declared)) + " inputs, got " + String(len(inputs)))
    if len(outputs) != len(declared_out):
        raise Error("ValueError: the model returns " + String(len(declared_out)) + " outputs, got "
                    + String(len(outputs)) + " output buffers")
    # every shape is checked before the program is touched, as ops_imported_run_py does
    var srcs = List[CastIn]()
    var shapes = List[ShapeN]()
    for k in range(len(declared)):  # driver-loop: over the model's declared inputs
        var want = List[Int]()
        var shape = py_call(b.tuple, declared[k][1])
        for j in range(len(shape)):  # small-loop(shape: a declared input's rank): shape metadata, never array elements
            want.append(Int(py=shape[j]))
        var given = py_call(b.tuple, inputs[k][1])
        var got = List[Int]()
        for j in range(len(given)):  # small-loop(given: an input's rank, at most eight): shape metadata, never array elements
            got.append(Int(py=given[j]))
        var same = len(got) == len(want)
        for j in range(min(len(got), len(want))):  # small-loop(want: a declared input's rank): shape metadata, never array elements
            same = same and got[j] == want[j]
        if not same:
            raise Error("ValueError: input " + _imported_repr(declared[k][0]) + " must have the shape "
                        + _imported_repr(shape) + ", got " + _imported_repr(_imported_tuple(got)))
        var sh = ShapeN.of(_imported_session_dims(want))
        if sh.count() == 0:
            raise Error("ValueError: input must not be empty")
        var kind = _device_kind(String(inputs[k][2]), "numeric program input")
        var size = kind[1]
        # a C-contiguous array's descriptor, exactly as array_input / _imported_input build it
        var src = CastIn(Int(py=inputs[k][0]), kind[0], size, sh.rows(), sh.cols(), sh.cols() * size, size)
        src.check("numeric program input")
        srcs.append(src)
        shapes.append(sh)
    var out_counts = List[Int]()
    var out_addrs = List[Int]()
    var out_texts = List[String]()
    for k in range(len(declared_out)):  # driver-loop: over the model's declared outputs
        var shape = py_call(b.tuple, declared_out[k][1])
        var count = 1
        for j in range(len(shape)):  # small-loop(shape: a declared output's rank): shape metadata, never array elements
            count *= Int(py=shape[j])
        var addr = Int(py=outputs[k])
        if count > 0 and addr == 0:
            raise Error("ValueError: output buffer " + String(k) + " has a null device address")
        out_counts.append(count)
        out_addrs.append(addr)
        out_texts.append(_imported_repr(shape))
    var session = program._s
    if session._id is None:
        raise Error("ValueError: this session is closed")
    var sid = Int(py=session._id)
    program_device_execute[KIND](sid, Int(py=program._id), srcs, shapes, out_counts, out_addrs, out_texts)
    return PythonObject(None)
