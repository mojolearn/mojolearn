# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`LogisticLoss`: the sigmoid objective and its derivative, per row.

PORT OF `cuml/cpp/src/glm/qn/glm_logistic.cuh` at cuML `00094f7`. Whole
file: `Lz::log_sigmoid`, `Lz::operator()`, `Dlz::operator()`, `gradNorm`.
Do not improve.

THEIR TWO FUNCTORS, copied (`glm_logistic.cuh:33-54`):

    log_sigmoid(x) = x < 0 ?  x - log(1 + exp(x))  :  -log(1 + exp(-x))
    lz(y, z)       = -log_sigmoid((2y - 1) z)
    dlz(y, z)      = (z < 0 ? exp(z) : 1) / (1 + exp(z < 0 ? z : -z)) - y

The two `exp` and the `log` are IDENTITY_PATHS row 12's transcendental
seam -- `exp(x)`'s last bit is a VENDOR CHOICE on Metal / PTX / OCML even
with every fold pinned, and E1 measured it on the GBDT Logloss link
(`checks/numerics.mojo`). Under IDENTICAL they go through
`identical_exp` / `identical_log`, the Cephes-style portable pair that is
ONE arithmetic everywhere; under FAST the stdlib's device path verbatim.
`2 * y - 1` is a multiply-add (row 9) and goes through `identical_mul_add`
-- exact when `y` is 0 or 1, which the caller guarantees, and pinned
anyway so a non-binary target cannot become a per-vendor bit. Every
intermediate is stored through `ftz` (row 10) because an `exp(-|z|)` at a
confident margin IS a denormal, and Metal flushes it where CUDA keeps it.

THE TWO LAUNCHES ARE ONE KERNEL HERE. `GLMBase::getLossAndDZ`
(`glm_base.cuh:129-171`) computes the loss with `mapThenSumReduce` and the
derivative with a second `binaryOp` over the same `Z`, and its own comment
says "would be nice to have a kernel that fuses these two steps". The
elementwise half of both IS fused below -- one read of `y` and `z`, two
stores -- and the SUM is a separate pinned fold (`glm_base.mojo::sum_terms`)
because `mapThenSumReduce`'s fold is a float atomic across blocks
(DEVIATION 547, see `simple_mat/dense.mojo`). The arithmetic per element is
theirs character for character; only the launch count differs.

`gradNorm` is `nrmMax` (`glm_logistic.cuh:61-64`): the L-infinity norm is
this loss's convergence metric, and it is a SELECTION, so no fold pin is
needed for it (row 30's reasoning).
"""

from std.gpu import block_dim, block_idx, thread_idx

from checks.numerics import (
    ftz,
    identical_exp,
    identical_log,
    identical_mul_add,
)


@always_inline
def logistic_lz(y: Float32, z: Float32) -> Float32:
    """`Lz::operator()(y, z)`: `-log_sigmoid((2y - 1) z)`."""
    var ytil = ftz(identical_mul_add(Float32(2.0), y, Float32(-1.0)))
    var x = ftz(ytil * z)
    # log_sigmoid
    var e = ftz(identical_exp(x if x < Float32(0.0) else -x))
    var temp = ftz(identical_log(ftz(Float32(1.0) + e)))
    var ls = ftz(x - temp) if x < Float32(0.0) else -temp
    return -ls


@always_inline
def logistic_dlz(y: Float32, z: Float32) -> Float32:
    """`Dlz::operator()(y, z)`."""
    var ez = ftz(identical_exp(z if z < Float32(0.0) else -z))
    var numerator = ez if z < Float32(0.0) else Float32(1.0)
    var q = ftz(numerator / ftz(Float32(1.0) + ez))
    return ftz(q - y)


def logistic_loss_dz_kernel(
    loss_terms: MutPointer[Float32, MutAnyOrigin],
    z: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    normalization: Float32,
):
    """Per row: `loss_terms[i] = lz(y, z) * normalization` (the map half of
    `mapThenSumReduce`, `glm_base.cuh:156-163`) and `z[i] = dlz(y, z)`
    (`binaryOp`, `:164`). One thread per row; no fold here."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var yi = y.unsafe_load(i)
        var zi = z.unsafe_load(i)
        loss_terms.unsafe_store(i, ftz(logistic_lz(yi, zi) * normalization))
        z.unsafe_store(i, logistic_dlz(yi, zi))
