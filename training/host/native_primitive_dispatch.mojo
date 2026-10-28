# SPDX-License-Identifier: Apache-2.0
# Shared provider table for CPU training primitives. Existing public bindings
# and this table call the same relocated host helpers, preserving arithmetic.
from std.ffi import _Global
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError, PRIMITIVE_ABI_NAME
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.hostptr import f32_ptr, i32_ptr
from core.mojolearn_training_host_impl import (
    embedding_forward_host_into, embedding_backward_host_into,
    rms_norm_forward_host_into, rms_norm_backward_host_into,
    linear_forward_host_into, linear_backward_host_into,
    accumulate_host_into, neural_rng_host_into, ce_loss_host_into, optimizer_step_host_into,
    chunked_lm_head_v2_loss_host_into, chunked_lm_head_v2_train_host_into,
)
from training.neural_identical_experiments import IDN_CHUNKED_LM_HEAD_V2


def host_primitive_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if op == 24:
        return 0  # CPU has no resident GPU model owner.
    if op == 28:
        return Int(IDN_CHUNKED_LM_HEAD_V2)
    if op == 25:
        return optimizer_step_host_into(f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]), f32_ptr(a[3]),
            i32_ptr(a[4]), i32_ptr(a[5]), f32_ptr(a[6]), p[0], p[1], p[2], p[3],
            Float32(f[0]), Float32(f[1]), Float32(f[2]), Float32(f[3]), Float32(f[4]), Float32(f[5]), Float32(f[6]), Float32(f[7]), p[4] != 0)
    if op == 0:
        return embedding_forward_host_into(a, p[0], p[1], p[2])
    if op == 1:
        return embedding_backward_host_into(a, p[0], p[1], p[2])
    if op == 2:
        return rms_norm_forward_host_into(a, p[0], p[1], Float32(f[0]))
    if op == 3:
        return rms_norm_backward_host_into(a, p[0], p[1], Float32(f[0]))
    if op == 4:
        return linear_forward_host_into(a, p[0], p[1], p[2])
    if op == 5:
        return linear_backward_host_into(a, p[0], p[1], p[2])
    if op == 6:
        return neural_rng_host_into(a[0], a[1], p[0], p[1], p[2], p[3], p[4], p[5], Float32(f[0]), Float32(f[1]))
    if op == 8:
        return ce_loss_host_into(f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]), a[3], a[4],
            p[0], p[1], p[2], p[3], p[4], p[5], Float32(f[0]))
    if op == 9:
        return accumulate_host_into(a, p[0], p[1], p[2])
    if op == 10:
        return chunked_lm_head_v2_loss_host_into(a, p[0], p[1], p[2])
    if op == 11:
        return chunked_lm_head_v2_train_host_into(a, p[0], p[1], p[2])
    raise Error("CPU primitive provider: unavailable operation")


def host_primitive_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                      result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if op < 0 or op >= 29 or op == 7 or (op >= 12 and op != 24 and op != 25 and op != 28) or result == 0:
            raise Error("CPU primitive provider: unavailable operation or null result")
        if op == 24 or op == 28:
            if na != 0 or np != 0 or nf != 0:
                raise Error("CPU primitive provider: capability takes no arguments")
        elif op == 25:
            if na != 7 or np != 5 or nf != 8:
                raise Error("CPU optimizer provider: invalid descriptor arity")
        else:
            var address_counts: List[Int] = [3, 3, 3, 5, 3, 5, 2, 7, 5, 2, 6, 8]
            var integer_counts: List[Int] = [3, 3, 2, 2, 3, 3, 6, 6, 6, 3, 3, 3]
            var float_counts: List[Int] = [0, 0, 1, 1, 0, 0, 2, 1, 1, 0, 0, 0]
            if na != address_counts[op] or np != integer_counts[op] or nf != float_counts[op]:
                raise Error("CPU primitive provider: invalid descriptor arity")
        var a = native_descriptor_words(aa, na)
        # CE originally admits optional output pointers and validates them in
        # ce_loss_host_into; preserve that distinction from the other entries.
        for address in a:  # driver-loop: fixed native primitive buffer vector
            if address == 0 and op != 8 and op != 25:
                raise Error("CPU primitive provider: null buffer address")
        var p = native_descriptor_words(pp, np)
        if nf != 0 and ff == 0:
            raise Error("CPU primitive provider: null floating span")
        var raw = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var f = List[Float64]()
        for i in range(nf):  # small-loop(at most eight optimizer floating scalars)
            f.append(raw[i])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = host_primitive_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct NativeHostTrainingPrimitiveAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(PRIMITIVE_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(host_primitive_v1)


comptime NATIVE_HOST_TRAINING_PRIMITIVE_API = _Global[StorageType=NativeHostTrainingPrimitiveAPI,
    name="MojoHostTrainingPrimitiveAPIv1", init_fn=NativeHostTrainingPrimitiveAPI.__init__]
