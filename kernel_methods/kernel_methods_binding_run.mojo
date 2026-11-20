# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN half of `bindings/_mojolearn_kernel_methods.mojo` (wave 8 run split,
2026-11-12): the typed work of the KernelRidge pointer fit and the Nystroem
transform. The door reads its Python address and parameter lists into ints
and floats, releases the GIL around the device calls exactly where it did
before, and packs the result. Typed Mojo only, no Python import
(tools/python_free_dirs.sh). Moved word for word: same copies, same calls,
same order, so the bits are unchanged. The other doors' typed bodies live in
kernel_methods/km_binding_impl.mojo."""
from core.device_query_x import QueryFold
from core.hostptr import read_f32
from core.identity_trace import IdentityTrace
from kernel_methods.estimator import (
    NystroemModel,
    kernel_ridge_fit_ptr_into,
    nystroem_transform_ptr_into,
)
from svm.impl.svm_parameter import KernelParams


def kernel_ridge_fit_ptr_run(
    xaddr: Int,
    yaddr: Int,
    n: Int,
    d: Int,
    t: Int,
    kp: KernelParams,
    alpha: Float32,
    dp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
    waddr: Int,
    fold: QueryFold,
    xfit_addr: Int,
) raises -> Tuple[Int, Int32, Int32]:
    """The KernelRidge fit from X and y on the device (`kernel_ridge_fit_ptr_into`);
    stores `info` in `sp[0]` and returns [info, finite word, range word]."""
    var sw = List[Float32]()
    var trace = IdentityTrace()
    var result = kernel_ridge_fit_ptr_into(
        xaddr, yaddr, n, d, t, kp, alpha, sw, dp, trace, waddr, fold, xfit_addr
    )
    sp.unsafe_store(0, Float64(result[0]))
    return result


def nystroem_transform_model_run(
    components_addr: Int,
    normalization_addr: Int,
    q: Int,
    d: Int,
    kernel: Int,
    degree: Int,
    gamma: Float64,
    coef0: Float64,
    seed: UInt64,
    sweeps: Int,
) raises -> NystroemModel:
    """The transform's model: the components and the normalization copied
    from their addresses; the indices, eigenvalues and eigenvectors stay
    empty."""
    # The transform reads the components and the normalization only
    # (`_nystroem_transform_dev`); the indices, eigenvalues and the q x q
    # eigenvectors are not copied per call (lane p3-svm-gp-kernel, row 340).
    # Their address slots stay in the argument list for its fixed order.
    var components = read_f32(components_addr, max(0, q * d))
    var indices = List[Int32]()
    var normalization = read_f32(normalization_addr, max(0, q * q))
    var eigenvalues = List[Float32]()
    var eigenvectors = List[Float32]()
    return NystroemModel(
        components^,
        indices^,
        normalization^,
        eigenvalues^,
        eigenvectors^,
        q,
        d,
        kernel,
        degree,
        gamma,
        coef0,
        seed,
        sweeps,
    )


def nystroem_transform_ptr_run(
    model: NystroemModel,
    xaddr: Int,
    m: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """The Nystroem transform with X read from its address on the device."""
    var trace = IdentityTrace()
    nystroem_transform_ptr_into(model, xaddr, m, op, trace)
