# SPDX-License-Identifier: Apache-2.0
# Exact existing ImportedProgram device copies/replay/cleanup, shared by the
# public packed door and the framework provider. This owner has no Python ABI.
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from sequence.ops import FP, Args
from sequence.exec_device import DeviceExec
from core.cast_in import CastIn, CAST_KIND_SIGNED, CAST_KIND_UNSIGNED
from autograd.rankn import ShapeN
from autograd.tape import Tape
from autograd.ops_api import ops_session, ops_session_registry_token, OPS_DEVICE
from autograd.program import PROGRAM_IN_IDS, same_shape, shape_text
from autograd.program_entries import program_execute_prepare
from autograd.program_runtime import replay_values, parameter_buffers
from autograd.cast_ops import OP_CAST_F32, OP_CAST_I32


def _words_view(ex: DeviceExec, p: FP, n: Int) raises -> DeviceBuffer[DType.float32]:
    """The view of n words at p, a buffer this executor allocated."""
    var found = ex._find(p, n)
    return ex._sub(found[0], found[1], n)


def _foreign_words(ctx: DeviceContext, addr: Int, n: Int) -> DeviceBuffer[DType.float32]:
    """A non-owning view of n words at a caller's device address."""
    return DeviceBuffer[DType.float32](ctx, MutPointer[Float32, MutAnyOrigin](unsafe_from_address=addr), n, owning=False)


def _device_input(mut ex: DeviceExec, dst: FP, src: CastIn, ids: Bool) raises:
    """dst = the caller's device array `src` as the program reads it: the
    host path's upload_cast (ids: upload_cast_i32) with the upload as one
    device-to-device copy."""
    var n = src.count()
    if ids:
        if src.kind != CAST_KIND_SIGNED and src.kind != CAST_KIND_UNSIGNED:
            raise Error("TypeError: program input holds ids and must be integers (an integer tensor)")
    var plain = (src.kind == CAST_KIND_SIGNED and src.itemsize == 4) if ids else src.is_f32_c()
    if plain:
        var view = _words_view(ex, dst, n)
        ex.ctx.enqueue_copy(dst_buf=view, src_buf=_foreign_words(ex.ctx, src.addr, n))
        _ = view^
        return
    # the span's bytes into zero-filled words (alloc zero-fills before the copy, in queue order)
    var raw = ex.alloc(src.words())
    try:
        var nbytes = src.nbytes()
        var bytes_dst = DeviceBuffer[DType.uint8](
            ex.ctx, MutPointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(raw)), nbytes, owning=False)
        var bytes_src = DeviceBuffer[DType.uint8](
            ex.ctx, MutPointer[UInt8, MutAnyOrigin](unsafe_from_address=src.start()), nbytes, owning=False)
        ex.ctx.enqueue_copy(dst_buf=bytes_dst, src_buf=bytes_src)
        _ = bytes_dst^
        _ = bytes_src^
        var a = Args()
        a.p0 = raw
        a.p1 = dst
        a.i0 = src.kind
        a.i1 = src.itemsize
        a.i2 = src.cols
        a.i3 = src.s0
        a.i4 = src.s1
        a.i5 = src.origin()
        if ids:
            ex.launch[OP_CAST_I32](a, n)
        else:
            ex.launch[OP_CAST_F32](a, n)
    except error:
        ex.free(raw)
        raise error
    ex.free(raw)


def program_device_execute[KIND: Int](sid: Int, pid: Int, srcs: List[CastIn],
    shapes: List[ShapeN], out_counts: List[Int], out_addrs: List[Int], out_texts: List[String],
) raises:
    comptime if has_apple_gpu_accelerator():
        raise Error("ValueError: ops_imported_run_device takes CUDA or ROCm device buffers; on Apple GPUs "
                    + "pass host arrays to ops_imported_run")
    var plan = program_execute_prepare[DeviceExec, KIND](sid, pid, sid, False, False)
    if len(srcs) != plan.program.n_in:
        raise Error("ValueError: this program takes " + String(plan.program.n_in) + " input arrays, got " + String(len(srcs)))
    for k in range(len(shapes)):  # driver-loop: over the model's declared inputs
        if not same_shape(shapes[k], plan.program.in_shapes[k]):
            raise Error("ValueError: input " + String(k) + " must keep its captured shape "
                        + shape_text(plan.program.in_shapes[k]) + "; got " + shape_text(shapes[k]))
    var s = ops_session[DeviceExec, KIND](sid)
    var p = plan.program.copy()
    var params = parameter_buffers(s[], p)
    var trainable = List[Bool](length=len(params), fill=False)
    var buffers = List[FP]()
    try:
        for k in range(len(srcs)):  # driver-loop: over the model's declared inputs (device copies)
            var buf = s[].ex.alloc(shapes[k].count())
            buffers.append(buf)
            _device_input(s[].ex, buf, srcs[k], p.in_kinds[k] == PROGRAM_IN_IDS)
    except err:
        for k in range(len(buffers)):  # driver-loop: over owners acquired before a failed copy
            s[].ex.free(buffers[k])
        raise err
    var held = Tape()
    swap(held, s[].prog_tape)
    swap(held, s[].tape)
    var temps = List[Int]()
    var phs = List[Int]()
    try:
        s[].tape.no_grad()
        s[].tape.reset()
        var outs = replay_values(s[], p, buffers, shapes, params, trainable, temps, phs)
        if len(outs) != len(out_addrs):
            raise Error("ValueError: the program returned " + String(len(outs)) + " outputs, the model declares "
                        + String(len(out_addrs)))
        for k in range(len(outs)):  # driver-loop: over the program's outputs (device copies, no data on the host)
            var slot = s[]._slot(outs[k])
            var sh = s[].shapes[slot]
            if out_counts[k] != sh.count():
                raise Error("ValueError: cannot reshape output " + String(k) + " of shape " + shape_text(sh)
                            + " into shape " + out_texts[k])
            if out_counts[k] == 0:
                continue
            var src = s[].tape.value_of(s[].ex, s[].tensors[slot])
            var view = _words_view(s[].ex, src, out_counts[k])
            s[].ex.ctx.enqueue_copy(dst_buf=_foreign_words(s[].ex.ctx, out_addrs[k], out_counts[k]), src_buf=view)
            _ = view^
        # the outputs are written when this returns
        s[].ex.sync()
    except err:
        s[]._drop_temps(temps)
        swap(held, s[].tape)
        swap(held, s[].prog_tape)
        for k in range(len(buffers)):  # driver-loop: over input owners
            s[].ex.free(buffers[k])
        raise err
    s[]._drop_temps(temps)
    swap(held, s[].tape)
    swap(held, s[].prog_tape)
    for k in range(len(buffers)):  # driver-loop: over input owners
        s[].ex.free(buffers[k])


def _descriptor_word(words: List[Int], mut cursor: Int) raises -> Int:
    if cursor >= len(words):
        raise Error("native program: truncated device descriptor")
    var result = words[cursor]
    cursor += 1
    return result


def _descriptor_dims(words: List[Int], mut cursor: Int) raises -> List[Int]:
    var rank = _descriptor_word(words, cursor)
    if rank < 0 or rank > 8:
        raise Error("native program: descriptor rank must be between zero and eight")
    var dims = List[Int]()
    for _ in range(rank):  # driver-loop: over admitted tensor rank metadata, at most eight axes
        dims.append(_descriptor_word(words, cursor))
    return dims^


def program_device_request(addresses: List[Int], words: List[Int]) raises -> Int:
    if len(words) < 6:
        raise Error("native program: missing device descriptor header")
    if words[0] != ops_session_registry_token() or words[1] != OPS_DEVICE:
        raise Error("native program: resource belongs to another registry or executor")
    var sid = words[2]
    var pid = words[3]
    var inputs = words[4]
    var outputs = words[5]
    if inputs < 0 or outputs < 0 or inputs > len(addresses) or outputs != len(addresses) - inputs:
        raise Error("native program: invalid input/output address counts")
    var cursor = 6
    var srcs = List[CastIn]()
    var shapes = List[ShapeN]()
    for k in range(inputs):  # driver-loop: over admitted input descriptors
        var kind = _descriptor_word(words, cursor)
        var itemsize = _descriptor_word(words, cursor)
        var dims = _descriptor_dims(words, cursor)
        var padded = List[Int]()
        for _ in range(2 - len(dims)):  # driver-loop: over at most two leading shape metadata axes
            padded.append(1)
        for width in dims:  # driver-loop: over admitted shape metadata, at most eight axes
            padded.append(width)
        var shape = ShapeN.of(padded)
        if shape.count() == 0:
            raise Error("ValueError: input must not be empty")
        var src = CastIn(addresses[k], kind, itemsize, shape.rows(), shape.cols(), shape.cols() * itemsize, itemsize)
        src.check("numeric program input")
        srcs.append(src)
        shapes.append(shape)
    var counts = List[Int]()
    var destinations = List[Int]()
    var texts = List[String]()
    for k in range(outputs):  # driver-loop: over admitted output descriptors
        var dims = _descriptor_dims(words, cursor)
        var count = 1
        var text = String("(")
        for axis in range(len(dims)):  # driver-loop: over admitted output shape metadata, at most eight axes
            count *= dims[axis]
            if axis > 0:
                text += ", "
            text += String(dims[axis])
        if len(dims) == 1:
            text += ","
        text += ")"
        var address = addresses[inputs + k]
        if count > 0 and address == 0:
            raise Error("ValueError: output buffer " + String(k) + " has a null device address")
        counts.append(count)
        destinations.append(address)
        texts.append(text^)
    if cursor != len(words):
        raise Error("native program: unexpected trailing device descriptor words")
    program_device_execute[OPS_DEVICE](sid, pid, srcs, shapes, counts, destinations, texts)
    return 0
