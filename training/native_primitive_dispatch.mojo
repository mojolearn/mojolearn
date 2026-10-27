# SPDX-License-Identifier: Apache-2.0
# Internal native descriptor interface to the SAME public training primitives.
# No arithmetic is respelled and no precision, cache or synchronization changes.
from std.ffi import _Global
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError, PRIMITIVE_ABI_NAME
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.hostptr import f32_ptr, i32_ptr
from core.neural_context import neural_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from training.samba_ops import (
    samba_embedding_forward_host, samba_embedding_backward_host,
    samba_rms_norm_forward_host, samba_rms_norm_backward_host,
    samba_linear_forward_host, samba_linear_backward_host,
)

comptime PRIMITIVE_CONTEXT = "MojoNeuralTrainingContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoNeuralTrainingContextFast"


def native_primitive_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    var ctx = neural_ctx[PRIMITIVE_CONTEXT]()
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
    raise Error("neural primitive ABI: unknown operation")


def native_primitive_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                       result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        var names = [String("embedding_forward"), String("embedding_backward"), String("rms_norm_forward"),
                     String("rms_norm_backward"), String("linear_forward"), String("linear_backward")]
        if op < 0 or op >= 6:
            raise Error("neural primitive ABI: unknown operation")
        var name = names[op]
        var want = 5 if op == 3 or op == 5 else 3
        if na != want:
            raise Error(name + ": addresses must contain " + String(want) + " entries, got " + String(na))
        var a = native_descriptor_words(aa, na)
        for i in range(na):  # small-loop(three or five native buffer addresses)
            if a[i] == 0:
                raise Error(name + ": null buffer address at slot " + String(i))
        var norm = op == 2 or op == 3
        if np != (2 if norm else 3) or nf != (1 if norm else 0):
            raise Error(name + ": params must contain 3 values, got " + String(np + nf))
        var p = native_descriptor_words(pp, np)
        var f = List[Float64]()
        if norm:
            if ff == 0:
                raise Error("neural primitive ABI: null scalar span")
            f.append(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)[0])
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
