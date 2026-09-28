# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST on Apple: GaussianProcessClassifier.fit's kernel matrix left on the
device for the Newton loop (lane neighbors-apple3, 2026-09-28):
gaussian_process/classifier.mojo `_gpc_kernel_self` without its download.
Its own module, imported only where a build selects it
(`-D MOJOLEARN_GPC_RESIDENT_K`)."""
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from gaussian_process.checks.gp_sabotage import GP_SAB_NONE
from gaussian_process.checks.kernels import (
    GP_ELEM_TPB,
    GPKernelSpec,
    gp_kernel_matrix,
    gp_kernel_stack_floats,
)
from gaussian_process.estimator import _length_scale_table, _upload


def _gpc_kernel_self_dev(
    ctx: DeviceContext, x: List[Float32], n_train: Int, n_features: Int, kernel: GPKernelSpec
) raises -> DeviceBuffer[DType.float32]:
    """`_gpc_kernel_self` with K left on the device."""
    var trace = IdentityTrace()
    var dx = _upload(ctx, x)
    var dls = _upload(ctx, _length_scale_table(kernel))
    var dk = ctx.enqueue_create_buffer[DType.float32](n_train * n_train)
    var dstack = ctx.enqueue_create_buffer[DType.float32](
        gp_kernel_stack_floats(n_train, n_train)
    )
    ctx.synchronize()
    gp_kernel_matrix(
        ctx,
        dk,
        dx,
        dx,
        dls,
        dstack,
        n_train,
        n_train,
        n_features,
        kernel,
        True,
        trace,
        "gpc.kernel",
        GP_ELEM_TPB,
        GP_SAB_NONE,
    )
    ctx.synchronize()
    _ = dx^
    _ = dls^
    _ = dstack^
    return dk^
