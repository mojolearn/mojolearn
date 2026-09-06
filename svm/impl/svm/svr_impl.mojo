# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`svrFit` / `svrPredict`, the epsilon-SVR entry points, dense FP32.

PORT OF `cuml/cpp/src/svm/svr_impl.cuh` + `svr.cu` at cuML v26.08.00,
`svrFitX` (dense) and the `SVR` class's `fit`. NOT ported: `svrFitSparse`
and the CSR arms, which this surface has no shape for at all, and the
PRECOMPUTED kernel and the weighted `InitPenalty` arm behind
`sample_weight`, which `check_rung1_scope` refuses by name.

WHAT `svrFit` IS, AND WHAT IT IS NOT
------------------------------------
It is SHORTER than `svcFit`, and that is upstream's shape rather than a
gap here. `svr_impl.cuh:68` hands `y` straight to `SmoSolver::Solve`, so
there is no `getUniquelabels`, no `getOvrlabels` and no `ovr_labels_kernel`
on this path. The targets ARE the solver's `y`, continuous, and every
difference between the two problems then lives inside the solver where the
gates can see it:

    SvrInit             `f = +-epsilon - y`, and the `+-1` label vector the
                        rest of the solver reads (`smosolver.mojo:129`)
    n_train = 2 * n_rows the alpha+ / alpha- domain (`smosolver.mojo:443`,
                        `workingset.mojo:138`, `results.mojo:133`)
    UpdateF twice        the second gemv at `f + n_rows`
    CombineCoefs         the add arm folding the two alpha halves down to
                        `n_rows` coefficients (`results.mojo:71`)
    GetVecIndices        `ws_idx % n_rows` in `ws_idx_mod_svr`

THE MODEL IS n_rows WIDE, NOT 2 * n_rows WIDE, AND THAT IS THE ONE FACT A
CALLER HAS TO KNOW. The doubled domain is INTERNAL. `Results` folds the two
halves in `combine_coefs` and then selects over `self.n_rows`
(`results.mojo:124-130` writes that out), so `dual_coefs`, `support_idx` and
`support_matrix` come back exactly the shapes `svcFit` produces and the
worst case is `n_rows`, `n_rows` and `n_rows * n_cols`. A caller sizing
output buffers at `2 * n_rows` would be allocating a second copy of nothing.

PREDICTION IS `svcPredict` WITH THE CLASS EPILOGUE OFF, and that is
upstream's arrangement too: `SVR` inherits `SVMBase::predict`, which calls
`svcPredict(..., predict_class = false)`. There is no second decision
kernel to port. `svr_predict` below is a named wrapper so a reader of the
regression path does not have to know that, and so the `predict_class =
true` arm cannot be reached from a regressor by passing a flag.

`model.unique_labels` CARRIES TWO DUMMY VALUES ON THIS PATH. A regressor
has no classes; `n_classes` is set to 0 to say so. But `decision_kernel`
takes `unique_labels[0]` and `[1]` as launch arguments whatever
`predict_class` is, so the list must have two entries or the launch cannot
be spelled. They are `-1` and `+1`, they are never read at
`predict_class = false`, and `svr_predict` is the only caller.
"""

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from svm.checks.device_select import upload_f32
from svm.impl.svm.smosolver import SmoSolver, SmoTrace
from svm.impl.svm.svc_impl import svc_predict
from svm.impl.svm.svm_parameter import (
    EPSILON_SVR,
    KernelParams,
    SvmModel,
    SvmParameter,
    check_finite_list,
    check_rung1_scope,
)


def svr_fit(
    ctx: DeviceContext,
    x_host: List[Float32],
    targets_host: List[Float32],
    n_rows: Int,
    n_cols: Int,
    param: SvmParameter,
    kp: KernelParams,
    mut card: IdentityTrace,
    mut trace: SmoTrace,
    has_sample_weight: Bool = False,
    kernel_tile_byte_limit: Int = 1 << 30,
    block_solve_threads: Int = 0,
    record_iterations: Bool = False,
    scratch_pad: Int = 0,
    scratch_poison: Float32 = 0.0,
) raises -> SvmModel:
    """`svrFit(handle, X, n_rows, n_cols, y, param, kernel_params, model,
    sample_weight)`. Returns the model (host values); `trace` receives the
    iteration record, empty unless `record_iterations`.

    `targets_host` holds CONTINUOUS TARGETS, one per row. Nothing validates
    them as a label pair and nothing sorts them, because upstream does
    neither. What IS checked is DEVIATION 636's family, every cell finite,
    for the same reason the classifier checks its labels: a NaN target
    lands in `svm.init.f` on the first recorded stage with a
    vendor-specific payload.

    The body is `svc_fit` minus the two label kernels and plus nothing.
    `svm/checks/svc_check.mojo::_run_svr_device` is the fit this was lifted
    from, and it is the fit the TWO SVR DEVICE GATES ran against
    (`check_svr_device_matches_oracle` over six regression fixtures and
    `check_svr_device_is_launch_invariant` over five arms). The 24 host
    gates beside them check the same solve through `smo_oracle.mojo` and
    never touch a `DeviceContext`.
    """
    if n_cols <= 0:
        raise Error("Parameter n_cols: number of columns cannot be less than one")
    if n_rows <= 0:
        raise Error("Parameter n_rows: number of rows cannot be less than one")
    if len(x_host) != n_rows * n_cols or len(targets_host) != n_rows:
        raise Error("svr_fit: x / y sizes do not match n_rows x n_cols")
    if param.svmType != EPSILON_SVR:
        raise Error(
            "svr_fit: svmType must be EPSILON_SVR ("
            + String(EPSILON_SVR) + "), got " + String(param.svmType)
        )
    check_rung1_scope(param, kp, has_sample_weight)
    # DEVIATION 636 (row 39, FACT 2): no NaN/inf may enter; see svm_parameter.mojo
    check_finite_list(x_host, "X")
    check_finite_list(targets_host, "labels")

    var model = SvmModel()
    # A REGRESSOR HAS NO CLASSES. The two entries below exist so that
    # `decision_kernel`'s launch can be spelled at all; see the module
    # docstring. `n_classes = 0` is what says the pair means nothing.
    model.n_classes = 0
    var dummy_labels: List[Float32] = [Float32(-1.0), Float32(1.0)]
    model.unique_labels = dummy_labels^
    card.header(
        "svm.svr_fit n_rows=" + String(n_rows) + " n_cols=" + String(n_cols)
        + " kernel=" + String(kp.kernel) + " C=" + String(param.C)
        + " epsilon=" + String(param.epsilon)
        + " tol=" + String(param.tol)
    )

    var x = upload_f32(ctx, x_host)
    var yv = upload_f32(ctx, targets_host)
    ctx.synchronize()
    card.record_device[DType.float32](ctx, "svm.input.x", x, n_rows * n_cols)
    card.record_device[DType.float32](ctx, "svm.input.y", yv, n_rows)

    var smo = SmoSolver(
        ctx, param, kp, n_rows, n_cols, kernel_tile_byte_limit,
        block_solve_threads, record_iterations, scratch_pad, scratch_poison,
    )
    smo.solve(ctx, x, yv, model, card, param.max_iter, param.max_outer_iter)
    model.n_cols = n_cols
    ctx.synchronize()
    trace = smo.trace^
    smo.trace = SmoTrace()
    _ = x^
    _ = yv^
    return model^


def svr_predict(
    ctx: DeviceContext,
    model: SvmModel,
    x_host: List[Float32],
    n_rows: Int,
    n_cols: Int,
    kp: KernelParams,
    buffer_size_mib: Float64,
    mut card: IdentityTrace,
) raises -> List[Float32]:
    """`SVR::predict`, which upstream reaches through `SVMBase::predict` and
    therefore through `svcPredict(..., predict_class = false)`. One value
    per row, `sum_j K(x, sv_j) dual_j + b`, the regression estimate.

    There is no `predict_class` argument HERE on purpose. The class
    epilogue is meaningless on a regressor and a flag would let a caller
    ask for it.
    """
    return svc_predict(
        ctx, model, x_host, n_rows, n_cols, kp, buffer_size_mib, False, card
    )
