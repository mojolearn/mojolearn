# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MOJOLEARN_IDN_SVGP_B_FP32_GATED (lane/svgp-b-gated, 2026-10-11, IDENTICAL):
PROMOTED to the IDENTICAL default (lane i17-decide-nbr-lin-prep, 2026-10-11).
A/B one run per arm, main@b2657cc89, algos svgp (runs v0035/v0036, a1618/a1619):
  istella NV L40S 344.4 -> 206.5 ms (0.60), AMD MI325X 1091.7 -> 620.7 ms (0.57)
  taxi    NV L40S 337.0 -> 334.6 ms (0.99), AMD MI325X 1086.9 -> 716.6 ms (0.66)
  digests NV == AMD on both arms; r2 equal (istella -0.10601568648, taxi -0.19498).
`-D MOJOLEARN_IDN_SVGP_B_FP32_GATED_OFF` (or MOJOLEARN_IDN_ALL_OFF) restores the
float-float B everywhere. FAST builds never take it (IDENTICAL-only, so the
SVGP_FAST_BSPLIT assert in iter_device.mojo cannot fire there).

B = Kuf Kfu through the identical GEMM in float32,
with a float32 Cholesky trust gate decided on the device; the float-float B
accumulation (svgp_ff.mojo) runs only when the gate rejects.

STRUCTURAL REASON. The float-float B (52 GFLOP of ff arithmetic at m = 512,
n = 100k; each ff step is 10-20 fp32 operations) is most of the svgp fit on
both vendors. The identical GEMM computes the same Gram at 2.66 TFLOP/s (NV)
and 4.8 TFLOP/s (AMD): ~20 ms NV / ~11 ms AMD at that shape. Expected: svgp
istella well-conditioned fits drop by most of the ff share (NV 347 ms, AMD
1,092 ms today); ill-conditioned fits (taxi: Sigma's spectrum 1e-6 .. 9e5)
pay the GEMM + one fp32 Cholesky extra and keep today's float-float bits.
No size rule: the gate is numerical (per fit), so every shape takes the same
decision on every column.

THE GATE. Sigma32 = Kuu + jitter I + B32 / noise (the float32 statements of
`svgp_ff_init_item`'s A_s), factored by `chol_serial`'s statements (the
device runs them one column per launch, rows in parallel: the
`gs_chol_step_kernel` pattern of glm/impl/gram_solve.mojo); trusted when no
pivot was non-positive and every pivot square is at least 2^-12 of its
diagonal (`gs_pivot_trusted`, the Ridge Gram route's gate). Trusted: B =
(B32, 0) as the float-float pair, and the rest of the float-float solve runs
unchanged. Rejected: the incumbent float-float accumulation over the same row
tiles, so the bits of a rejected fit equal today's.

BITS. A trusted fit changes bits (B32 replaces the ff B) on NVIDIA, AMD and
the host column together: the device GEMM and `gemm_oracle` are the same
contract (gemm/contract.mojo), and the host runs the same gate cells.

Cells only here (no GPU import): x_neighbors/iter_host.mojo imports them for
the host column, x_neighbors/iter_device.mojo wraps them in kernels.
"""
from std.sys.compile import is_defined
from x_neighbors.items import FP
from checks.numerics import ftz, identical_div, identical_mul_add, GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_decomp.cells import chol_diag, div0, sqrt0
from glm.impl.gram_solve_cells import gs_pivot_trusted

comptime IDN_SVGP_B_FP32_GATED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_IDN_SVGP_B_FP32_GATED_OFF"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


@always_inline
def svgp_b32_sigma_item(t: Int, kuu: FP, b32: FP, a: FP, adiag: FP, m: Int, noise: Float32, jitter: Float32):
    """Sigma32[t] = (Kuu + jitter I)[t] + B32[t] / noise, t = i*m + k; the
    diagonal also goes to adiag (the trust gate's reference and the step
    kernel's untouched pivot start)."""
    var i = t // m
    var k = t - i * m
    var v = ftz(kuu.unsafe_load(t))
    if i == k:
        v = ftz(v + jitter)
    v = ftz(v + ftz(identical_div(ftz(b32.unsafe_load(t)), noise)))
    a.unsafe_store(t, v)
    if i == k:
        adiag.unsafe_store(i, v)


@always_inline
def svgp_b32_chol_item(i: Int, a: FP, adiag: FP, info: FP, j: Int, n: Int):
    """Row i >= j of `chol_serial`'s column step j (glm/impl/gram_solve.mojo
    `gs_chol_step_kernel`'s statements): row j is `chol_diag`; every row
    below re-forms the pivot from adiag[j] in the same order."""
    if i == j:
        chol_diag(a, info, j, n)
        return
    var acc = ftz(adiag.unsafe_load(j))
    for p in range(j):
        var l = ftz(a.unsafe_load(j * n + p))
        acc = ftz(identical_mul_add(-l, l, acc))
    if not (acc > Float32(0)):
        acc = Float32(1)
    var dpiv = sqrt0(acc)
    var s = ftz(a.unsafe_load(i * n + j))
    for p in range(j):
        s = ftz(identical_mul_add(-ftz(a.unsafe_load(i * n + p)), ftz(a.unsafe_load(j * n + p)), s))
    a.unsafe_store(i * n + j, div0(s, dpiv))
    a.unsafe_store(j * n + i, Float32(0))


@always_inline
def svgp_b32_trusted(l: FP, adiag: FP, info: FP, m: Int) -> Bool:
    """No non-positive pivot and every pivot passes `gs_pivot_trusted`
    (m loads, pivots ascending)."""
    var ok = info.unsafe_load(0) == Float32(0)
    for j in range(m):
        if not gs_pivot_trusted(l.unsafe_load(j * m + j), adiag.unsafe_load(j)):
            ok = False
    return ok


@always_inline
def svgp_b32_select_item(t: Int, flag: FP, b32: FP, bh: FP, bl: FP):
    """Trusted fit: the float-float B pair is (B32, 0). Rejected: untouched
    (the gated float-float accumulation wrote it)."""
    if flag.unsafe_load(0) == Float32(1):
        bh.unsafe_store(t, b32.unsafe_load(t))
        bl.unsafe_store(t, Float32(0))
