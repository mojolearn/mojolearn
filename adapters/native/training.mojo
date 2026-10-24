# SPDX-License-Identifier: Apache-2.0
# Whole-training boundary over the existing Sequential trainer. No new math.
from adapters.native.operation import IP, WORDS, descriptor_shape, copy_words
from autograd.model import parse_model, FitConfig, check_fit, model_fit_state, FIT_STATE_WORDS, FS_EPOCH, FS_STEP, FS_STALL
from autograd.rankn import ShapeN
from autograd.embedding_ops import embedding_refuse_ids, embedding_check_words
from sequence.recurrent import OptConfig
from sequence.ops import FP
from sequence.exec_trait import Exec
from core.cast_in import CastIn
from std.math import isfinite
from std.memory import memcpy

comptime DP = MutPointer[Float64, MutUntrackedOrigin]


def read_words[E: Exec](mut ex: E, src: FP, dst: FP, n: Int, host_buffers: Bool) raises:
    if host_buffers:
        memcpy(dest=dst, src=src, count=n)
    else:
        var tmp = ex.alloc(n)
        copy_words(ex, src, tmp, n)
        ex.download(dst, tmp, n)
        ex.free(tmp)


def write_words[E: Exec](mut ex: E, src: FP, dst: FP, n: Int, host_buffers: Bool) raises:
    if host_buffers:
        memcpy(dest=dst, src=src, count=n)
    else:
        var tmp = ex.alloc(n)
        ex.upload(tmp, src, n)
        copy_words(ex, tmp, dst, n)
        ex.sync()
        ex.free(tmp)


def require_vector(words: IP, slot: Int, length: Int, kind: Int) raises:
    var offset = slot * WORDS
    var shape = descriptor_shape(words, offset)
    if shape.ndim != 1 or shape.d[0] != length or words[offset + 1] != kind:
        raise Error("training adapter: invalid vector shape or dtype at slot " + String(slot))


def run_training[E: Exec](
    mut ex: E, inputs: IP, count: Int, outputs: IP,
    program: IP, n_program: Int, options: IP, n_options: Int,
    scalars: DP, n_scalars: Int, host_buffers: Bool,
) raises:
    if count != 3 and count != 7:
        raise Error("training adapter: pass X, Y, parameters, plus four saved state tensors to resume")
    if Int(inputs) == 0 or Int(outputs) == 0 or Int(program) == 0 or Int(options) == 0 or Int(scalars) == 0:
        raise Error("training adapter: missing descriptors or configuration")
    if n_program < 3 or n_options != 13 or n_scalars < 10:
        raise Error("training adapter: invalid configuration length")
    var prog = List[Int]()
    for j in range(n_program):  # driver-loop: native model metadata
        prog.append(program[j])
    var layer_floats = List[Float32]()
    for j in range(n_scalars):  # driver-loop: native scalar configuration
        if not isfinite(scalars[j]) or (j != 6 and not isfinite(Float32(scalars[j]))):
            raise Error("training adapter: configuration scalars must be finite float32 values")
        if j >= 9:
            layer_floats.append(Float32(scalars[j]))
    var spec = parse_model(prog, layer_floats)
    if spec.n_params < 1:
        raise Error("training adapter: model must have trainable parameters")
    # Same packed optimizer as model_fit_state; reject ignored flags/options.
    if options[3] < 0 or options[3] > 1 or (options[2] != 0 and options[3] != 0):
        raise Error("training adapter: only SGD Nesterov accepts a flag")
    if scalars[4] < 0 or scalars[6] < 0 or scalars[5] < 0 or scalars[5] > 1:
        raise Error("training adapter: invalid weight decay, tolerance or dampening")
    if scalars[1] < 0 or scalars[1] >= 1:
        raise Error("training adapter: momentum/beta1 must be in [0,1)")
    if options[2] != 0 and (scalars[2] < 0 or scalars[2] >= 1 or scalars[3] <= 0 or scalars[5] != 0):
        raise Error("training adapter: invalid Adam beta2, epsilon or dampening")
    if options[3] == 1 and (scalars[1] <= 0 or scalars[5] != 0):
        raise Error("training adapter: Nesterov needs momentum and zero dampening")
    if (options[8] != 0 and options[8] != 1) or (options[12] != 0 and options[12] != 1):
        raise Error("training adapter: shuffle and initialize must be zero or one")
    if options[9] < 0 or options[9] > 0xFFFFFFFF or options[10] < 0 or options[10] > 0xFFFFFFFF:
        raise Error("training adapter: seed halves must be unsigned 32-bit words")
    var opt = OptConfig(options[2], options[3], Float32(scalars[1]), Float32(scalars[2]),
                        Float32(scalars[3]), Float32(scalars[4]), Float32(scalars[5]))
    var cfg = FitConfig(options[0], options[1], opt, Float32(scalars[0]),
                        options[4], options[5], Float32(scalars[7]), options[6], options[7],
                        options[8] != 0, UInt64(options[9]) | (UInt64(options[10]) << 32),
                        options[11], scalars[6], Float32(scalars[8]), options[12] != 0)
    var xs = descriptor_shape(inputs, 0)
    var ys = descriptor_shape(inputs, WORDS)
    if inputs[1] != 0 or xs.ndim < 2:
        raise Error("training adapter: X must be a float32 batch")
    var N = xs.d[0]
    check_fit(spec, cfg, N)
    if spec.version == 1:
        if xs.ndim != 2 or xs.d[1] != spec.n_inputs:
            raise Error("training adapter: X shape differs from the model")
    else:
        if xs.ndim != spec.sample.ndim + 1:
            raise Error("training adapter: X rank differs from the model sample shape")
        for j in range(spec.sample.ndim):  # small-loop(rank<=7): sample metadata
            if xs.d[j + 1] != spec.sample.d[j]:
                raise Error("training adapter: X shape differs from the model")
    if cfg.target == 0:
        if inputs[WORDS + 1] != 0 or ys.ndim != 2 or ys.d[0] != N or ys.d[1] != spec.out_features():
            raise Error("training adapter: dense Y must be float32 (samples, outputs)")
    else:
        require_vector(inputs, 1, N, 1)
    var np_ = spec.n_params
    require_vector(inputs, 2, np_, 0)
    var resume = count == 7
    var previous_capacity = 0
    if resume:
        require_vector(inputs, 3, np_, 0)
        require_vector(inputs, 4, np_, 0)
        require_vector(inputs, 5, FIT_STATE_WORDS * 2, 1)
        var prev = descriptor_shape(inputs, 6 * WORDS)
        previous_capacity = prev.count()
        require_vector(inputs, 6, previous_capacity, 0)
    if cfg.epochs > (1 << 58) - previous_capacity:
        raise Error("training adapter: loss curve capacity overflow")
    var capacity = previous_capacity + cfg.epochs
    for j in range(3):  # small-loop(3): parameters and moments
        require_vector(outputs, j, np_, 0)
    require_vector(outputs, 3, FIT_STATE_WORDS * 2, 1)
    require_vector(outputs, 4, capacity, 0)
    # model_fit_state owns uploads, tape, backward, optimizer and epoch loop.
    # Its public input/output boundary is host storage even for DeviceExec.
    var x = List[Float32](length=xs.count(), fill=0)
    var y = List[Float32](length=ys.count(), fill=0)
    var p = List[Float32](length=np_, fill=0)
    var m = List[Float32](length=np_, fill=0)
    var v = List[Float32](length=np_, fill=0)
    var state = List[Int64](length=FIT_STATE_WORDS, fill=0)
    var curve = List[Float32](length=capacity, fill=0)
    var previous = List[Float32](length=max(1, previous_capacity), fill=0)
    read_words(ex, FP(unsafe_from_address=inputs[0]), x.unsafe_ptr(), len(x), host_buffers)
    read_words(ex, FP(unsafe_from_address=inputs[WORDS]), y.unsafe_ptr(), len(y), host_buffers)
    read_words(ex, FP(unsafe_from_address=inputs[2 * WORDS]), p.unsafe_ptr(), np_, host_buffers)
    if resume:
        read_words(ex, FP(unsafe_from_address=inputs[3 * WORDS]), m.unsafe_ptr(), np_, host_buffers)
        read_words(ex, FP(unsafe_from_address=inputs[4 * WORDS]), v.unsafe_ptr(), np_, host_buffers)
        read_words(ex, FP(unsafe_from_address=inputs[5 * WORDS]), FP(unsafe_from_address=Int(state.unsafe_ptr())), FIT_STATE_WORDS * 2, host_buffers)
        if state[FS_EPOCH] < 0 or state[FS_EPOCH] > previous_capacity or state[FS_STEP] < 0 or state[FS_STALL] < 0:
            raise Error("training adapter: invalid saved counters or loss curve length")
        read_words(ex, FP(unsafe_from_address=inputs[6 * WORDS]), previous.unsafe_ptr(), previous_capacity, host_buffers)
    if cfg.target == 1:
        var ids = ex.alloc(N)
        ex.upload(ids, y.unsafe_ptr(), N)
        var scratch_n = embedding_check_words(N)
        var a = ex.alloc(scratch_n)
        var b = ex.alloc(scratch_n)
        embedding_refuse_ids(ex, ids, N, spec.out_features(), a, b)
        ex.free(ids)
        ex.free(a)
        ex.free(b)
    var xin = CastIn.f32_c(Int(x.unsafe_ptr()), N, spec.n_inputs)
    var yin = CastIn.f32_c(Int(y.unsafe_ptr()), N, spec.out_features())
    if cfg.target != 0:
        yin = CastIn(Int(y.unsafe_ptr()), 1, 4, N, 1, 4, 4)
    _ = model_fit_state(ex, spec, cfg, xin, yin, N, p.unsafe_ptr(), curve.unsafe_ptr(),
                        m.unsafe_ptr(), v.unsafe_ptr(), state.unsafe_ptr(), previous.unsafe_ptr(), resume)
    write_words(ex, p.unsafe_ptr(), FP(unsafe_from_address=outputs[0]), np_, host_buffers)
    write_words(ex, m.unsafe_ptr(), FP(unsafe_from_address=outputs[WORDS]), np_, host_buffers)
    write_words(ex, v.unsafe_ptr(), FP(unsafe_from_address=outputs[2 * WORDS]), np_, host_buffers)
    write_words(ex, FP(unsafe_from_address=Int(state.unsafe_ptr())), FP(unsafe_from_address=outputs[3 * WORDS]), FIT_STATE_WORDS * 2, host_buffers)
    write_words(ex, curve.unsafe_ptr(), FP(unsafe_from_address=outputs[4 * WORDS]), capacity, host_buffers)
    ex.sync()
