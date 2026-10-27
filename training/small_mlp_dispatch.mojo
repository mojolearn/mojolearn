# SPDX-License-Identifier: Apache-2.0
# Engine-specific operations over the shared native descriptor transport.
# Every branch calls the established owner with the same precision/order.
from std.ffi import _Global
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.hostptr import f32_ptr, i32_ptr
from core.neural_context import neural_ctx
from training.mlp_ops import mlp_train_step_host, mlp_bias_activation_host, mlp_relu_backward_host, mlp_sum_rows_host, mlp_validate_shape
from training.estimator import identical_ce_loss_host
from training.optimizer_public import per_call_step
from training.neural_gemm_admission import neural_gemm_shape
from training.neural_gemm_native import neural_gemm_native

comptime SMALL_MLP_ABI_NAME = "mojolearn.training.small_mlp.v1"
comptime SMALL_MLP_OPERATION_COUNT = 8
comptime SMALL_MLP_CONTEXT = "MojoNeuralTrainingContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoNeuralTrainingContextFast"


def small_mlp_native_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if op == 0:
        return 1 if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else 0
    var ctx = neural_ctx[SMALL_MLP_CONTEXT]()
    if op == 1:
        return mlp_train_step_host(ctx, f32_ptr(a[0]), i32_ptr(a[1]),
            f32_ptr(a[2]), f32_ptr(a[3]), f32_ptr(a[4]), f32_ptr(a[5]),
            f32_ptr(a[6]), f32_ptr(a[7]), i32_ptr(a[8]), f32_ptr(a[9]), f32_ptr(a[10]),
            f32_ptr(a[11]), f32_ptr(a[12]), f32_ptr(a[13]), f32_ptr(a[14]), f32_ptr(a[15]), f32_ptr(a[16]),
            p[0], p[1], p[2], Float32(f[0]), Float32(f[1]), Float32(f[2]), Float32(f[3]), Float32(f[4]), p[3])
    if op == 2:
        var shape = neural_gemm_shape(p[0], p[1], p[2], p[3], a[0], a[1], a[2])
        return neural_gemm_native(a[0], a[1], a[2], shape)
    if op == 3 or op == 4 or op == 5:
        mlp_validate_shape(p[0], p[1])
    if op == 3:
        if p[2] != 0 and p[2] != 1:
            raise Error("mlp_bias_activation relu_flag must be 0 or 1")
        return mlp_bias_activation_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]), p[0], p[1], p[2])
    if op == 4:
        return mlp_relu_backward_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]), p[0], p[1])
    if op == 5:
        return mlp_sum_rows_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), p[0], p[1])
    if op == 6:
        return identical_ce_loss_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]), f32_ptr(a[3]), i32_ptr(a[4]),
            p[0], p[1], p[2], p[3], p[4], p[5], Float32(f[0]))
    if op == 7:
        return per_call_step(ctx, f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]), f32_ptr(a[3]), i32_ptr(a[4]), i32_ptr(a[5]), f32_ptr(a[6]),
            p[0], p[1], p[2], p[3], Float32(f[0]), Float32(f[1]), Float32(f[2]), Float32(f[3]), Float32(f[4]), Float32(f[5]), Float32(f[6]), Float32(f[7]), p[4] != 0)
    raise Error("small MLP native ABI: unknown operation")


def small_mlp_native_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                        result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        var address_counts: List[Int] = [0, 17, 3, 3, 3, 2, 5, 7]
        var int_counts: List[Int] = [0, 4, 4, 3, 2, 2, 6, 5]
        var float_counts: List[Int] = [0, 5, 0, 0, 0, 0, 1, 8]
        if op < 0 or op >= SMALL_MLP_OPERATION_COUNT:
            raise Error("small MLP native ABI: unknown operation")
        if na != address_counts[op] or np != int_counts[op] or nf != float_counts[op]:
            raise Error("small MLP native ABI: descriptor length mismatch")
        if result == 0 or (nf > 0 and ff == 0):
            raise Error("small MLP native ABI: null output or scalar span")
        var a = native_descriptor_words(aa, na)
        for i in range(na):  # small-loop(the fixed operation buffer descriptor)
            if a[i] == 0:
                raise Error("small MLP native ABI: null buffer address")
        var p = native_descriptor_words(pp, np)
        var f = List[Float64]()
        var fp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        for i in range(nf):  # small-loop(fixed operation scalar descriptor)
            f.append(fp[i])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = small_mlp_native_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct SmallMlpNativeTable(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(SMALL_MLP_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(small_mlp_native_v1)


comptime SMALL_MLP_NATIVE_TABLE = _Global[StorageType=SmallMlpNativeTable,
    name="MojoSmallMlpNativeTableV1", init_fn=SmallMlpNativeTable.__init__]
