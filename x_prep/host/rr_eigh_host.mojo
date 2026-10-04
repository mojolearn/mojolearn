# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IDENTICAL: the prep lane's `eigh` stage (op 18) as the round-robin Jacobi
on every vendor AND in the host column (lane fam-prep-metrics, 2026-10-04).

`eigh_unit` (x_prep/eigh.mojo) is ONE THREAD per matrix running the cyclic
Jacobi: at LDA's / QDA's d = 220 that is 24,090 dependent rotations a sweep
on one GPU thread while the rest of the GPU idles. x_prep/rr_eigh.mojo
already holds the whole-GPU round-robin Jacobi (FAST + Apple since
lane/apple-fast-ldaqda); every step of it is pinned arithmetic
(x_decomp/rr.mojo: `rr_cs`, `rr_block`, `rr_vrow`, `rr_off_fold`,
`rr_converged`; every product `identical_mul`, every result flushed), a
round's cells each have one writer and no other reader, and its convergence
test is decided from the folded sums in a fixed order. So the same words
come out of the device kernels on NVIDIA, AMD and Apple and out of the host
walk below, which runs the same rounds one after another in place
(x_decomp/rr.mojo `host_eigh_rr` is the same walk for x_decomp).

BITS. The round-robin order is not the cyclic order: eigenvalues and vectors
move within float32 round-off against `eigh_unit`, on all four columns
together. Both are Jacobi iterations to the same tolerance class; the tail
contract is `eigh_unit`'s (eigenvalues descending, index order on a tie,
each vector's largest-magnitude component positive).

A after the stage: `eigh_unit` documents A as DESTROYED. The device
ping-pongs A between the arena and scratch, so what the arena holds at the
end depends on the round parity; under IDN_RR_EIGH the device and the host
both ZERO the matrix after the solve, so the arena words agree.

A stage keeps `eigh_unit` when q[5] != 0 (x_prep/device.mojo EIGH_CYCLIC_Q)
or when the matrix is smaller than IDN_RR_MIN_N (a shape-only rule, the same
on every column): below it the fixed launch budget of the round-robin costs
more than the one-thread solve.

-D MOJOLEARN_IDN_RR_EIGH_OFF restores `eigh_unit` everywhere.
No accelerator import: the CPU-only host binding compiles this file.
"""
from std.sys.compile import is_defined
from checks.numerics import ftz, GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from decomposition.checks.jacobi_eigh_device import JACOBI_TOL
from decomposition.spectrum_order_device import spectrum_rank_desc
from x_decomp.cells import F32Ptr
from x_decomp.rr import rr_cs, rr_block, rr_vrow, rr_off_fold, rr_converged
from x_prep.common import FP, IP, p
from x_prep.eigh import eigh_unit

comptime IDN_RR_EIGH = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (is_defined["MOJOLEARN_IDN_RR_EIGH_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)
#: the smallest matrix the round-robin takes under IDN_RR_EIGH (shape only)
comptime IDN_RR_MIN_N = 24
#: x_prep/rr_eigh.mojo RRE_SWEEPS: the sweep budget (the device enqueues it whole)
comptime IDN_RR_SWEEPS = 32
#: x_prep/device.mojo EIGH_CYCLIC_Q
comptime _CYCLIC_Q = 5


@always_inline
def eigh_rr_takes(n: Int) -> Bool:
    """Whether a matrix of order n goes to the round-robin solve. Under
    IDN_RR_EIGH: n >= IDN_RR_MIN_N; otherwise (FAST + Apple's RR_EIGH) every
    size, as before."""
    comptime if IDN_RR_EIGH:
        return n >= IDN_RR_MIN_N
    else:
        return True


def eigh_rr_host_unit(t: Int, f: FP, q: IP):
    """q = [A, m, astride, EVAL, EVEC, cyclic]; t = batch index: `eigh_unit`'s
    stage as x_prep/rr_eigh.mojo `rr_eigh_into` computes it, on the host.
    The same rounds in the same order (`rre_round_kernel` = `rr_cs` from the
    round's source, then `rr_block` / `rr_vrow`), the same test before every
    sweep (`rre_part_kernel` + `rre_fold_kernel` = `rr_off_fold`, then
    `rr_converged`, or the budget), the diagonal taken at the last test,
    `rre_sign_kernel` and `rre_order_kernel`'s tail; then A zeroed."""
    var n = p(q, 1)
    if p(q, _CYCLIC_Q) != 0 or not eigh_rr_takes(n):
        eigh_unit(t, f, q)
        return
    if n <= 0:
        return
    var m = n + (n % 2)
    var h = m // 2
    var a = f + (p(q, 0) + t * p(q, 2))
    var W = p(q, 3) + t * n
    var E = p(q, 4) + t * n * n
    var vl = List[Float32](length=n * n, fill=Float32(0.0))
    var dl = List[Float32](length=n, fill=Float32(0.0))
    var csl = List[Float32](length=max(2 * h, 2), fill=Float32(0.0))
    var ap = F32Ptr(unsafe_from_address=Int(a))
    var vp = F32Ptr(unsafe_from_address=Int(vl.unsafe_ptr()))
    var dp = F32Ptr(unsafe_from_address=Int(dl.unsafe_ptr()))
    var cs = F32Ptr(unsafe_from_address=Int(csl.unsafe_ptr()))
    for i in range(n):
        vp.unsafe_store(i * n + i, Float32(1.0))
    for sweep in range(IDN_RR_SWEEPS + 1):
        # the test: the diagonal of its source, then the folded sums
        for k in range(n):
            dp.unsafe_store(k, ftz(ap.unsafe_load(k * n + k)))
        var sums = rr_off_fold(ap, n)
        if rr_converged(sums[0], sums[1], Float32(JACOBI_TOL)):
            break
        if sweep == IDN_RR_SWEEPS:
            break
        for rd in range(m - 1):
            for b in range(h):
                var got = rr_cs(ap, n, m, rd, b)
                cs.unsafe_store(2 * b, got[0])
                cs.unsafe_store(2 * b + 1, got[1])
            for i in range(h):
                for j in range(i, h):
                    rr_block(ap, cs, n, m, rd, i, j)
            for k in range(n):
                for j in range(h):
                    rr_vrow(vp, cs, n, m, rd, k, j)
    # rre_sign_kernel: each column positive at its largest-magnitude component (first on a tie)
    for col in range(n):
        var biggest = Float32(0.0)
        var first = 0
        for r in range(n):
            var mg = abs(vp.unsafe_load(r * n + col))
            if mg > biggest:
                biggest = mg
                first = r
        if vp.unsafe_load(first * n + col) < Float32(0.0):
            for r in range(n):
                vp.unsafe_store(r * n + col, -vp.unsafe_load(r * n + col))
    # rre_order_kernel: eigenpair i to its descending position
    for i in range(n):
        var r = spectrum_rank_desc(dp, n, i)
        f.unsafe_store(W + r, dp.unsafe_load(i))
        for c in range(n):
            f.unsafe_store(E + c * n + r, ftz(vp.unsafe_load(c * n + i)))
    # A: zeroed, as the device's `rre_clear_kernel`
    for c in range(n * n):
        ap.unsafe_store(c, Float32(0.0))
    # the lists live to here (pointers into them above)
    _ = csl^
    _ = dl^
    _ = vl^
