# SPDX-License-Identifier: Apache-2.0
# Internal native descriptor interface to the SAME public training primitives.
# No arithmetic is respelled and no precision, cache or synchronization changes.
from std.ffi import _Global
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError, PRIMITIVE_ABI_NAME
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.hostptr import f32_ptr, i32_ptr
from core.neural_context import neural_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.philox_neural import neural_rng_host
from training.estimator import identical_ce_loss_host
from training.accumulate_multi_gpu import parallel_accumulate_host
from training.chunked_lm_head_v2 import chunked_lm_head_v2_loss_host, chunked_lm_head_v2_train_host
from training.neural_identical_experiments import IDN_SAMBA_RESIDENT_STEP, IDN_CHUNKED_LM_HEAD_V2
from training.native_resident_dispatch import native_resident_arity, native_resident_operation
from training.samba_ops import (
    samba_embedding_forward_host, samba_embedding_backward_host,
    samba_rms_norm_forward_host, samba_rms_norm_backward_host,
    samba_linear_forward_host, samba_linear_backward_host, samba_head_loss_host,
)

comptime PRIMITIVE_CONTEXT = "MojoNeuralTrainingContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoNeuralTrainingContextFast"


def native_primitive_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if op == 24:
        return Int(IDN_SAMBA_RESIDENT_STEP)
    if op == 28:
        return Int(IDN_CHUNKED_LM_HEAD_V2)
    var ctx = neural_ctx[PRIMITIVE_CONTEXT]()
    if op >= 12:
        return native_resident_operation(ctx, op, a, p, f)
    if op == 0:
        return samba_embedding_forward_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), i32_ptr(a[2]), p[0], p[1], p[2])
    if op == 1:
        return samba_embedding_backward_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), i32_ptr(a[2]), p[0], p[1], p[2])
    if op == 2:
        return samba_rms_norm_forward_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]), p[0], p[1], Float32(f[0]))
    if op == 3:
        return samba_rms_norm_backward_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]), f32_ptr(a[3]), f32_ptr(a[4]), p[0], p[1], Float32(f[0]))
    if op == 4:
        return samba_linear_forward_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]), p[0], p[1], p[2])
    if op == 5:
        return samba_linear_backward_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]), f32_ptr(a[3]), f32_ptr(a[4]), p[0], p[1], p[2])
    if op == 6:
        return neural_rng_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]),
            p[0], p[1], p[2], p[3], p[4], p[5], Float32(f[0]), Float32(f[1]))
    if op == 7:
        return samba_head_loss_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]), f32_ptr(a[3]),
            f32_ptr(a[4]), f32_ptr(a[5]), i32_ptr(a[6]), p[0], p[1], p[2], p[3], p[4], p[5], Float32(f[0]))
    if op == 8:
        return identical_ce_loss_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]), f32_ptr(a[3]),
            i32_ptr(a[4]), p[0], p[1], p[2], p[3], p[4], p[5], Float32(f[0]))
    if op == 9:
        return parallel_accumulate_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), p[0], p[1], p[2])
    if op == 10:
        return chunked_lm_head_v2_loss_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]),
            f32_ptr(a[3]), f32_ptr(a[4]), i32_ptr(a[5]), p[0], p[1], p[2])
    if op == 11:
        return chunked_lm_head_v2_train_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]), f32_ptr(a[3]),
            f32_ptr(a[4]), f32_ptr(a[5]), f32_ptr(a[6]), i32_ptr(a[7]), p[0], p[1], p[2])
    raise Error("neural primitive ABI: unknown operation")


def native_primitive_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                       result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        var names = [String("embedding_forward"), String("embedding_backward"), String("rms_norm_forward"),
                     String("rms_norm_backward"), String("linear_forward"), String("linear_backward"),
                     String("neural_rng"), String("samba_head_loss"), String("ce_loss"), String("accumulate"),
                     String("chunked_lm_head_v2_loss"), String("chunked_lm_head_v2_train")]
        if op < 0 or op >= 29:
            raise Error("neural primitive ABI: unknown operation")
        var address_counts: List[Int] = [3, 3, 3, 5, 3, 5, 2, 7, 5, 2, 6, 8]
        var integer_counts: List[Int] = [3, 3, 2, 2, 3, 3, 6, 6, 6, 3, 3, 3]
        var float_counts: List[Int] = [0, 0, 1, 1, 0, 0, 2, 1, 1, 0, 0, 0]
        var name = String("resident")
        if op >= 12:
            native_resident_arity(op, na, np, nf)
        else:
            name = names[op]
            if na != address_counts[op] or np != integer_counts[op] or nf != float_counts[op]:
                raise Error(name + ": invalid native descriptor arity")
        var a = native_descriptor_words(aa, na)
        for i in range(na):  # small-loop(the admitted primitive's fixed buffer addresses)
            if a[i] == 0 and op != 22 and op != 23:
                raise Error(name + ": null buffer address at slot " + String(i))
        var p = native_descriptor_words(pp, np)
        var f = List[Float64]()
        if nf != 0 and ff == 0:
            raise Error("neural primitive ABI: null scalar span")
        var fp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        for i in range(nf):  # small-loop(zero, one or two admitted primitive floats)
            f.append(fp[i])
        if result == 0:
            raise Error("neural primitive ABI: null result")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = native_primitive_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct NativePrimitiveAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(PRIMITIVE_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(native_primitive_v1)


comptime NATIVE_PRIMITIVE_API = _Global[StorageType=NativePrimitiveAPI,
    name="MojoNeuralPrimitiveNativeAPIv1", init_fn=NativePrimitiveAPI.__init__]
