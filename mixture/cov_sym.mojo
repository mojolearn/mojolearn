# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LANE classical-te-gmm (2026-10-07): the SYMMETRIC M-step covariance of
the Gaussian mixture under IDENTICAL, default OFF, the A/B define
-D MOJOLEARN_IDN_GMM_COV_SYM. A leaf: no accelerator import, so the device
M-step (mixture/checks/mstep.mojo) and the host column
(mixture/host/gmm_host_oracle.mojo) read one switch and one cell map.

Each component's raw covariance `scaled^T . diff` (d x d over n rows) is
symmetric in exact arithmetic. The device computes a two-block COVER of
its upper triangle with the shipped identical GEMM as the tile:

    rows [0, h) x every column      `scaled[:, :h]^T . diff`         h x d
    rows [h, d) x columns [h, d)    `scaled[:, h:]^T . diff[:, h:]`  (d-h)^2

h = d // 2 minimizes h*d + (d-h)^2 (3/4 of d^2 cells, the operands read
2.5 n d words instead of 2 n d, and one GEMM fewer per component than the
three-block cover); the cost holds for every d >= 2 (below it there is
nothing to split, the incumbent path runs). The GEMM contract (gemm/
IDENTICAL_FP32_CONTRACT.md: a cell's word depends on k and the operand
words along k only, never on m, n or the plan) makes every kept cell the
incumbent's word on both GPU vendors. `cov_finish_sym_kernel` then reads
cell (a, b) and cell (b, a) from the one kept cell (min, max):
BITS: the upper triangle (a <= b) is unchanged; the lower triangle becomes
the transposed upper words (the covariance is exactly symmetric). The host
column mirrors the same cell of its full product, so NVIDIA, AMD and the
host change together. Switches that take the plain per-component path
away (the FAST Gram, the paired center kernels MOJOLEARN_IDN_GMM_CENTER_PAIR
/ C53, a sabotage) leave this one off for that fit.
"""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime GMM_COV_SYM = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_IDN_GMM_COV_SYM"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


@always_inline
def gmm_cov_sym_split(d: Int) -> Int:
    """h: rows [0, h) of the product are computed whole, then the
    (d-h) x (d-h) block; d // 2 minimizes h*d + (d-h)^2. 0 below d = 2
    (no split)."""
    if d < 2:
        return 0
    return d // 2


@always_inline
def gmm_cov_sym_cell(a: Int, b: Int, d: Int, h: Int) -> Int:
    """The raw-buffer word of kept cell (a, b), a <= b: row a < h sits in
    the h x d block at [0, h*d) in the product's own row-major layout; a
    row at or past h in the (d-h) x (d-h) block stored from h*d on."""
    if a < h:
        return a * d + b
    return h * d + (a - h) * (d - h) + (b - h)
