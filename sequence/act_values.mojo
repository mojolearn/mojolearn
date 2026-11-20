# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The activation followers and their in-register value forms.

MOVED here word for word from `sequence/mlp.mojo` (lane fx-leftovers-product,
2026-11-19): `op_act`, `op_act_bwd`, `act_value`, `act_bwd_value` and the
activation codes. `sequence/mlp.mojo` imports every name back, so
`from sequence.mlp import act_value` (and every other existing import) still
resolves and no caller changes. `op_bias`, the fourth follower, already
lives in `sequence/ops.mojo`.

WHY A MODULE OF ITS OWN: this file imports `sequence/ops.mojo` and
`checks/numerics.mojo` only, and neither imports `gemm/`. `sequence/mlp.mojo`
imports the autograd and optimizer modules, which import `gemm/`, so the
packed GEMM kernel (`gemm/checks/gemm_identical.mojo`) could not import the
follower bodies it must repeat in its store epilogue. From here it can: the
epilogue kinds are these functions' calls, so there is one spelling of each
follower. Nothing in this file may import `gemm/` or a module that does.
"""
from sequence.ops import Args, ld, mul, st, sub
from checks.numerics import ftz, identical_sigmoid, identical_tanh

comptime ACT_IDENTITY = 0
comptime ACT_LOGISTIC = 1
comptime ACT_TANH = 2
comptime ACT_RELU = 3
comptime ACT_SOFTMAX = 4


def op_act(t: Int, a: Args):
    """p0[t] = act(p0[t]); i0 activation (not softmax)."""
    var z = ld(a.p0, t)
    var k = a.i0
    if k == ACT_LOGISTIC:
        z = ftz(identical_sigmoid(z))
    elif k == ACT_TANH:
        z = ftz(identical_tanh(z))
    elif k == ACT_RELU:
        z = z if z > Float32(0.0) else Float32(0.0)
    st(a.p0, t, z)


def op_act_bwd(t: Int, a: Args):
    """p0[t] (delta) *= act'(p1[t]) written through the activation's output,
    the reference's inplace_*_derivative; i0 activation."""
    var d = ld(a.p0, t)
    var z = ld(a.p1, t)
    var k = a.i0
    if k == ACT_LOGISTIC:
        d = mul(mul(d, z), sub(Float32(1.0), z))
    elif k == ACT_TANH:
        d = mul(d, sub(Float32(1.0), mul(z, z)))
    elif k == ACT_RELU:
        if z == Float32(0.0):
            d = Float32(0.0)
    st(a.p0, t, d)


# The in-register forms (lane ag-pass3-impl; the header is in
# sequence/mlp.mojo, "the tape's in-register epilogue"): the bodies below are
# op_act's and op_act_bwd's verbatim, on values.
@always_inline
def act_value(z_in: Float32, k: Int) -> Float32:
    """op_act's body on a value: act(z), activation code k (not softmax)."""
    var z = z_in
    if k == ACT_LOGISTIC:
        z = ftz(identical_sigmoid(z))
    elif k == ACT_TANH:
        z = ftz(identical_tanh(z))
    elif k == ACT_RELU:
        z = z if z > Float32(0.0) else Float32(0.0)
    return z


@always_inline
def act_bwd_value(d_in: Float32, z: Float32, k: Int) -> Float32:
    """op_act_bwd's body on values: delta times act'(z), z the activation."""
    var d = d_in
    if k == ACT_LOGISTIC:
        d = mul(mul(d, z), sub(Float32(1.0), z))
    elif k == ACT_TANH:
        d = mul(d, sub(Float32(1.0), mul(z, z)))
    elif k == ACT_RELU:
        if z == Float32(0.0):
            d = Float32(0.0)
    return d
