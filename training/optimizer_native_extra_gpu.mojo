# SPDX-License-Identifier: Apache-2.0
"""The GPU training image's public training operations of the optimizer
native table (word layouts: training/optimizer_native_extra.mojo). The same
device kernels, on the same contexts, that the image's Python entries of the
same names called."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.hostptr import f32_ptr, i32_ptr
from core.neural_context import neural_ctx
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from training.chunked_lm_head_v2 import chunked_lm_head_v2_loss_host, chunked_lm_head_v2_train_host
from training.dev_tensors import IDN_TRAIN_DEV_TENSORS
from training.neural_session_mlp import nn_mlp_sessions_device
from training.residual_dropout import residual_dropout_device
from training.residual_dropout_common_run import residual_dropout_addresses
from training.residual_dropout_boundary_run import residual_dropout_dev_run, residual_dropout_backward_dev_run
from training.optimizer_native_api import (
    OPT_NATIVE_VENDOR, OPT_NATIVE_RESIDUAL_DROPOUT, OPT_NATIVE_RESIDUAL_DROPOUT_BACKWARD,
    OPT_NATIVE_RESIDUAL_DROPOUT_DEV, OPT_NATIVE_RESIDUAL_DROPOUT_BACKWARD_DEV, OPT_NATIVE_LM_HEAD_LOSS,
    OPT_NATIVE_LM_HEAD_TRAIN, OPT_NATIVE_MLP_SESSIONS,
)
from training.optimizer_native_extra import OptimizerExtraRequest, TRAINING_VENDOR_NONE

# The training binding's device context (bindings/_mojolearn_training.mojo _NEURAL_CTX).
comptime _EXTRA_CTX = "MojolearnNeuralTrainingContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnNeuralTrainingContextFastV3"


def _training_vendor_word() -> Int:
    var vendor = String(COMPILED_VENDOR)
    if vendor == "cpu":
        return 0
    if vendor == "metal":
        return 1
    if vendor == "cuda":
        return 2
    if vendor == "hip":
        return 3
    return TRAINING_VENDOR_NONE


def optimizer_extra_gpu(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        var r = OptimizerExtraRequest(operation, aa, na, pp, np, ff, nf, result)
        ref a = r.addresses
        var value = 0
        if operation == OPT_NATIVE_VENDOR:
            value = _training_vendor_word()
        elif operation == OPT_NATIVE_RESIDUAL_DROPOUT:
            var args = r.residual()
            residual_dropout_addresses[False](a[0], a[1], a[2], a[2], args.n)
            if args.n != 0:
                var yp = f32_ptr(a[2])
                # the residual-dropout layer's context is the identical one (it is an identical-mode layer)
                var ctx = neural_ctx["MojolearnNeuralTrainingContextIdenticalV3"]()
                value = residual_dropout_device[False](ctx, yp, yp, f32_ptr(a[0]), f32_ptr(a[1]), args.n, args.offset,
                                                       args.p, args.seed_lo, args.seed_hi, args.stream)
        elif operation == OPT_NATIVE_RESIDUAL_DROPOUT_BACKWARD:
            var args = r.residual()
            residual_dropout_addresses[True](a[0], a[0], a[1], a[2], args.n)
            if args.n != 0:
                var yp = f32_ptr(a[0])
                var ctx = neural_ctx["MojolearnNeuralTrainingContextIdenticalV3"]()
                value = residual_dropout_device[True](ctx, f32_ptr(a[1]), f32_ptr(a[2]), yp, yp, args.n, args.offset,
                                                      args.p, args.seed_lo, args.seed_hi, args.stream)
        elif operation == OPT_NATIVE_RESIDUAL_DROPOUT_DEV or operation == OPT_NATIVE_RESIDUAL_DROPOUT_BACKWARD_DEV:
            comptime if IDN_TRAIN_DEV_TENSORS:
                var args = r.residual()
                var handles = r.handles()
                if args.n != 0:
                    if operation == OPT_NATIVE_RESIDUAL_DROPOUT_DEV:
                        residual_dropout_dev_run(handles, args)
                    else:
                        residual_dropout_backward_dev_run(handles, args)
                value = args.n
            else:
                raise Error("ImportError: residual dropout on device arrays requires updated training binding: missing residual_dropout_dev")
        elif operation == OPT_NATIVE_LM_HEAD_LOSS:
            var ctx = neural_ctx[_EXTRA_CTX]()
            value = chunked_lm_head_v2_loss_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]), f32_ptr(a[3]),
                                                 f32_ptr(a[4]), i32_ptr(a[5]), r.ints[0], r.ints[1], r.ints[2])
        elif operation == OPT_NATIVE_LM_HEAD_TRAIN:
            var ctx = neural_ctx[_EXTRA_CTX]()
            value = chunked_lm_head_v2_train_host(ctx, f32_ptr(a[0]), f32_ptr(a[1]), f32_ptr(a[2]), f32_ptr(a[3]),
                                                  f32_ptr(a[4]), f32_ptr(a[5]), f32_ptr(a[6]), i32_ptr(a[7]),
                                                  r.ints[0], r.ints[1], r.ints[2])
        elif operation == OPT_NATIVE_MLP_SESSIONS:
            var ctx = neural_ctx[_EXTRA_CTX]()
            value = nn_mlp_sessions_device(ctx, r.session_inputs(), r.session_rows(), f32_ptr(a[1]), f32_ptr(a[2]),
                                           f32_ptr(a[3]), f32_ptr(a[4]), f32_ptr(a[0]), r.ints[0], r.ints[1], r.ints[2])
        else:
            raise Error("native optimizer: unknown operation or missing result storage")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1
