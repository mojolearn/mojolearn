# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Device-resident fitted state of the Gaussian processes (lane
ne1-a-gp-svm, workstream A; rows N363).

`GaussianProcessRegressor.predict` uploaded `X_train_` (n x d), `alpha_`
(n) and, for `return_std`, the n x n factor `L_` on every call (the pointer
route, `gpr_predict_ptr_device`), or first copied all three into host lists
(the `gpr_predict` route). Here they are uploaded once per fitted model into
the GP binding's store (`core/fitted_resident_store.mojo`) and every later
predict uploads only the query:

    slot 0  X_train_, alpha_      every predict
    slot 1  L_                    the first `return_std=True` predict

The store's key is a lease the public door (bindings/gp_regressor_native.mojo
through bindings/fitted_resident_door.mojo) takes once per fitted state and
keeps in the model's prepared-owner record; `GpResidentLease` is the native
object that holds it, and its destruction (a refit, a replaced fitted array,
the model's death) frees the copies.

WHAT DOES NOT CHANGE. The resident predict runs `gpr_predict_dev`
(gaussian_process/gpr_resident.mojo), the body the pointer route runs after
its own uploads, over copies of the same bytes: the same launches in the
same order, so the bits are the same, in both numeric tiers (the host-list
route `gpr_predict_host` runs the same launches too).
"""
from std.ffi import _Global
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.fitted_resident_store import (
    FittedResidentStore,
    fitted_lease_open,
    fitted_lease_close,
    fitted_resident_view,
    fitted_resident_stats,
)
from gaussian_process.checks.kernels import GPKernelSpec, gp_kernel_diag, gp_validate_kernel
from gaussian_process.estimator import _family_ctx
from gaussian_process.gpc_ovr import _gpc_upload_checked
from gaussian_process.gpr_resident import gpr_predict_dev


# One store per numeric tier, next to that tier's GP context.
comptime GP_RESIDENT_STORE = _Global[
    StorageType=FittedResidentStore,
    name=(
        "MojoGpResidentFittedIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
        else "MojoGpResidentFittedFast"
    ),
    init_fn=FittedResidentStore.__init__,
]


def gp_resident_store() raises -> UnsafePointer[FittedResidentStore, MutAnyOrigin]:
    return GP_RESIDENT_STORE.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


struct GpResidentLease(Movable, Writable):
    """The native owner of one fitted state's lease: dropping it frees the
    device copies (a refit, a replaced fitted array, the model's death)."""

    var lease: Int

    def __init__(out self, lease: Int):
        self.lease = lease

    def write_to(self, mut writer: Some[Writer]):
        writer.write("GpResidentLease(open=", self.lease != 0, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def __deinit__(deinit self):
        if self.lease != 0:
            try:
                fitted_lease_close(gp_resident_store(), self.lease)
            except:
                pass


def gp_resident_lease_open() raises -> Int:
    return fitted_lease_open(gp_resident_store())


def gp_resident_stats_words() raises -> List[Int]:
    return fitted_resident_stats(gp_resident_store())


def gpr_predict_resident_device(
    lease: Int,
    xt_addr: Int,
    l_addr: Int,
    dual_addr: Int,
    xs_addr: Int,
    n_train: Int,
    n_features: Int,
    n_star: Int,
    kernel: GPKernelSpec,
    return_std: Bool,
    info: Int,
    unnorm: Bool,
    y_std: Float32,
    y_mean: Float32,
    mean_addr: Int,
    var_addr: Int,
    std_addr: Int,
    clamped_addr: Int,
) raises -> Int:
    """`gpr_predict_ptr_device` with X_train_, alpha_ and L_ read from the
    lease's device copies (uploaded from these addresses at the first call
    of the lease, and again only when one of them changed). The refusals
    are the pointer route's, in its order. Returns the clamp count."""
    if info != 0:
        raise Error(
            "gpr_predict_host: refusing to predict from a FAILED fit"
            " (info="
            + String(info)
            + "). The factor's columns from "
            + String(info - 1)
            + " onward are unfinished, and solving against them returns"
            " infinities that look like numbers. DEVIATION 1634"
        )
    if n_star <= 0:
        raise Error("gpr_predict_host: n_star must be positive, got " + String(n_star))
    if n_train <= 0:
        raise Error("gpr_predict_host: n_train must be positive, got " + String(n_train))
    if xt_addr == 0 or dual_addr == 0 or mean_addr == 0:
        raise Error("gpr_predict: a model or output address is null")
    if return_std and (l_addr == 0 or var_addr == 0 or std_addr == 0 or clamped_addr == 0):
        raise Error("gpr_predict: return_std needs the factor and the three variance outputs")
    gp_validate_kernel(kernel, n_features)
    var kss = gp_kernel_diag(kernel)

    var store = gp_resident_store()
    var ctx = _family_ctx()
    # The query is checked before any model copy is taken.
    var dxs = _gpc_upload_checked(ctx, xs_addr, n_star, n_features, String("X_star"))
    var model = fitted_resident_view(store, ctx, lease, 0, [xt_addr, dual_addr], [n_train * n_features, n_train])
    var dx = model.bufs[0].copy()
    var ddual = model.bufs[1].copy()
    var dl: DeviceBuffer[DType.float32]
    if return_std:
        var factor = fitted_resident_view(store, ctx, lease, 1, [l_addr], [n_train * n_train])
        dl = factor.bufs[0].copy()
        _ = factor^
    else:
        dl = ctx.enqueue_create_buffer[DType.float32](1)
    var n_clamped = gpr_predict_dev(
        ctx, dx, ddual, dl, dxs, n_train, n_features, n_star, kernel, kss,
        return_std, unnorm, y_std, y_mean, mean_addr, var_addr, std_addr, clamped_addr,
    )
    _ = dx^
    _ = ddual^
    _ = dl^
    _ = dxs^
    _ = model^
    # DEVIATION 1946: the context dies LAST.
    _ = ctx^
    return n_clamped
