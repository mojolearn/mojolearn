# SPDX-License-Identifier: Apache-2.0
"""The host training image's public training operations of the optimizer
native table (word layouts: training/optimizer_native_extra.mojo). The same
host kernels the image's Python entries of the same names called. Residual
dropout on training device arrays has no host kernel: refused in words."""
from core.hostptr import f32_ptr
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.mojolearn_training_host_impl import chunked_lm_head_v2_loss_host_into, chunked_lm_head_v2_train_host_into
from training.neural_session_mlp_host import nn_mlp_sessions_host
from training.residual_dropout_common_run import residual_dropout_addresses
from training.residual_dropout_host import residual_dropout_host
from training.optimizer_native_api import (
    OPT_NATIVE_VENDOR, OPT_NATIVE_RESIDUAL_DROPOUT, OPT_NATIVE_RESIDUAL_DROPOUT_BACKWARD,
    OPT_NATIVE_LM_HEAD_LOSS, OPT_NATIVE_LM_HEAD_TRAIN, OPT_NATIVE_MLP_SESSIONS,
)
from training.optimizer_native_extra import OptimizerExtraRequest


def optimizer_extra_host(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        var r = OptimizerExtraRequest(operation, aa, na, pp, np, ff, nf, result)
        ref a = r.addresses
        var value = 0
        if operation == OPT_NATIVE_VENDOR:
            # "cpu": the vendor word of a host binding.
            value = 0
        elif operation == OPT_NATIVE_RESIDUAL_DROPOUT:
            var args = r.residual()
            residual_dropout_addresses[False](a[0], a[1], a[2], a[2], args.n)
            if args.n != 0:
                var yp = f32_ptr(a[2])
                value = residual_dropout_host[False](yp, yp, f32_ptr(a[0]), f32_ptr(a[1]), args.n, args.offset,
                                                     args.p, args.seed_lo, args.seed_hi, args.stream)
        elif operation == OPT_NATIVE_RESIDUAL_DROPOUT_BACKWARD:
            var args = r.residual()
            residual_dropout_addresses[True](a[0], a[0], a[1], a[2], args.n)
            if args.n != 0:
                var yp = f32_ptr(a[0])
                value = residual_dropout_host[True](f32_ptr(a[1]), f32_ptr(a[2]), yp, yp, args.n, args.offset,
                                                    args.p, args.seed_lo, args.seed_hi, args.stream)
        elif operation == OPT_NATIVE_LM_HEAD_LOSS:
            value = chunked_lm_head_v2_loss_host_into(a, r.ints[0], r.ints[1], r.ints[2])
        elif operation == OPT_NATIVE_LM_HEAD_TRAIN:
            value = chunked_lm_head_v2_train_host_into(a, r.ints[0], r.ints[1], r.ints[2])
        elif operation == OPT_NATIVE_MLP_SESSIONS:
            value = nn_mlp_sessions_host(r.session_inputs(), r.session_rows(), f32_ptr(a[1]), f32_ptr(a[2]),
                                         f32_ptr(a[3]), f32_ptr(a[4]), f32_ptr(a[0]), r.ints[0], r.ints[1], r.ints[2])
        else:
            raise Error("RuntimeError: this build has no training device arrays (identical GPU builds only)")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1
