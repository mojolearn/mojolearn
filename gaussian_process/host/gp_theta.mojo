# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the gp GPU binding and the gp CPU host binding; product, not only a check.
"""Kernel hyperparameter optimization (2026-09-15): the host pieces the
device estimator (`gaussian_process/estimator.mojo::gpr_lml_grad_host`) and
the CPU verifier (`gaussian_process/host/gpr_grad_oracle.mojo`) SHARE, so
each exists in exactly one spelling.

The reference is scikit-learn 1.9.0 `sklearn/gaussian_process/_gpr.py`
(`log_marginal_likelihood(theta, eval_gradient=True)`, `fit`'s restarts) and
`kernels.py` (`theta`, `bounds`, the `eval_gradient` arms).

DEVIATION 2880: THE HYPERPARAMETERS, THEIR LOGS AND THE GRADIENT'S FOLD.

  theta      the natural log of every FREE hyperparameter, in the kernel's
             postfix leaf order (a Sum or Product lists its left operand's
             entries first, scikit-learn's `k1.theta` then `k2.theta`); an
             ARD length scale contributes one entry per feature. A leaf whose
             bounds are "fixed" contributes none.
  log        `identical_log64` (Float64, host), `gp_log64`, for the initial
             theta and for the bounds.
  exp        `Float32(identical_exp64(theta))` then `ftz`, `gp_theta_param`:
             the hyperparameter that runs is that float32. So a kernel's
             value is always a float32 the binding can be handed, and two
             hosts computing it agree because both seams are portable.
  dK/dtheta  on the device and in the verifier, `kernel_gradient.mojo` and
             `gpr_grad_oracle.mojo`, their headers pin each formula.
  K^-1       the identical Cholesky's solve against the identity (`n`
             right-hand sides), scikit-learn's `cho_solve(L, eye)`.
  gradient   `gp_lml_gradient_fold` below: for parameter `p`, i ASCENDING,
             then j ASCENDING, `w = fma(alpha_i, alpha_j, -Kinv[i, j])` and
             `acc = fma(w, dK_p[j, i], acc)`, seeded +0.0, every stored value
             through `ftz`; then `0.5 * acc` by `identical_mul`. That is
             scikit-learn's `0.5 * einsum("ijl,jik->kl", alpha alpha^T -
             K_inv, K_gradient)` as ONE serial float32 chain.

DEVIATION 2881 (the optimizer) lives in `python/mojolearn/_gp_optimizer.py`;
its restart draws are `gp_restart_uniform` below: position-mapped Philox,
`philox4x32_10(ctr = (restart r, dimension j, 0, "GPOR"), key =
random_state as (low word, high word))`, `u = ((w0 >> 5) * 2^26 + (w1 >> 6))
* 2^-53`, the 53-bit double every Philox double in the repository uses. Each
draw is exact integer arithmetic followed by one exact scaling, so it is the
same double on every host.

THE SABOTAGE (-D MOJOLEARN_GP_GRAD_SABOTAGE=1, a VALUE arm in the gradient):
the fold's `0.5` becomes `0.625`. The likelihood value is untouched, so every
cell that moves under it moved because the optimizer read the gradient.
"""

from std.memory import bitcast
from std.sys.compile import is_defined

from checks.numerics import (
    ftz,
    identical_exp64,
    identical_log64,
    identical_mul,
    identical_mul_add,
)
from core.philox import philox4x32_10
from gaussian_process.gp_grad_items import gp_grad_blocks, gp_grad_part_item, gp_grad_fin_item, gp_free_count

#: THE NEGATIVE CONTROL. See this file's header.
comptime GP_GRAD_SABOTAGE = is_defined["MOJOLEARN_GP_GRAD_SABOTAGE"]()

#: `ctr[3]` of the restart draws, ASCII "GPOR". Distinct from sample_y's
#: "GPSY" (DEVIATION 2793).
comptime GP_RESTART_TAG: UInt32 = 0x47504F52

#: `2^-53` as float64 bits.
comptime GP_TWO_POW_M53_BITS: UInt64 = 0x3CA0000000000000

# The postfix kinds, `kernels.mojo`'s and `gpr_oracle.mojo`'s.
comptime _K_CONST = 0
comptime _K_WHITE = 1
comptime _K_RBF = 2
comptime _K_MATERN = 3
comptime _K_SUM = 4
comptime _K_PROD = 5


def gp_log64(v: Float64) -> Float64:
    """theta of one hyperparameter value or bound (DEVIATION 2880)."""
    return identical_log64(v)


def gp_theta_param(theta: Float64) -> Float32:
    """The float32 hyperparameter that runs at `theta` (DEVIATION 2880)."""
    return ftz(Float32(identical_exp64(theta)))


def gp_restart_uniform(seed: UInt64, restart: Int, dim: Int) -> Float64:
    """The uniform in [0, 1) of restart `restart`, dimension `dim`
    (DEVIATION 2881's stream, this file's header)."""
    var ctr = SIMD[DType.uint32, 4](
        UInt32(restart & 0xFFFFFFFF),
        UInt32(dim & 0xFFFFFFFF),
        UInt32(0),
        GP_RESTART_TAG,
    )
    var key = SIMD[DType.uint32, 2](
        UInt32(seed & 0xFFFFFFFF), UInt32((seed >> 32) & 0xFFFFFFFF)
    )
    var w = philox4x32_10(ctr, key)
    var hi = UInt64(w[0] >> UInt32(5))
    var lo = UInt64(w[1] >> UInt32(6))
    var bits53 = (hi << 26) | lo
    return Float64(bits53) * bitcast[DType.float64](GP_TWO_POW_M53_BITS)


def gp_lml_gradient_fold(
    dual: List[Float32],
    kinv: List[Float32],
    dk: List[Float32],
    n: Int,
    n_free: Int,
) -> List[Float32]:
    """`0.5 * trace((alpha alpha^T - K^-1) dK_p)` for every `p`, the pinned
    serial chain of this file's header. `dual` is `alpha_` (n), `kinv` is
    `K^-1` row-major (n x n), `dk` is the `n_free` gradient matrices, `p`
    ascending, each row-major."""
    var g = List[Float32](length=n_free, fill=Float32(0.0))
    var half = gp_grad_half()
    var nb = gp_grad_blocks(n)
    var part = List[Float32](length=max(n_free * nb, 1), fill=Float32(0.0))
    var pp = MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(part.unsafe_ptr()))
    for t in range(n_free * nb):
        gp_grad_part_item(
            t, MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(dual.unsafe_ptr())),
            MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(kinv.unsafe_ptr())),
            MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(dk.unsafe_ptr())), n, nb, pp,
        )
    for p in range(n_free):
        gp_grad_fin_item(p, pp, nb, half, MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(g.unsafe_ptr())))
    return g^


@always_inline
def gp_grad_half() -> Float32:
    """0.5 (0.625 under -D MOJOLEARN_GP_GRAD_SABOTAGE)."""
    comptime if GP_GRAD_SABOTAGE:
        return Float32(0.625)
    return Float32(0.5)
