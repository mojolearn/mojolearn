# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`gp_lml_gradient_fold`'s statements as items, ONE definition for the device
(`gaussian_process/estimator.mojo::gpr_lml_grad`) and the host column
(`gaussian_process/host/gp_theta.mojo`), cpu-gpu-cleanup c-gp-kernel
(2026-10-02): the GPU binding downloaded every dK_p and folded on the host.

For parameter p, the rows are cut into GP_GRAD_ROWS-row blocks; block b's
chain is the old chain's statements over its rows (i ascending, then j
ascending, `w = fma(alpha_i, alpha_j, -Kinv[i, j])`, `acc = fma(w,
dK_p[j, i], acc)`, seeded +0.0, every value through `ftz`); the block
partials are then added ascending (`ftz(acc + part)`) and scaled by 0.5
(`identical_mul`). The same words on every column."""
from checks.numerics import ftz, identical_mul, identical_mul_add

comptime _P = MutPointer[Float32, MutAnyOrigin]
comptime GP_GRAD_ROWS = 16


@always_inline
def gp_grad_blocks(n: Int) -> Int:
    return (n + GP_GRAD_ROWS - 1) // GP_GRAD_ROWS if n > 0 else 0


def gp_grad_part_item(t: Int, dual: _P, kinv: _P, dk: _P, n: Int, nb: Int, part: _P):
    """Item t = p * nb + b: parameter p's chain over row block b."""
    var p = t // nb
    var b = t - p * nb
    var base = p * n * n
    var lo = b * GP_GRAD_ROWS
    var hi = min(lo + GP_GRAD_ROWS, n)
    var acc = Float32(0.0)
    for i in range(lo, hi):
        var ai = ftz(dual.unsafe_load(i))
        for j in range(n):
            var w = ftz(identical_mul_add(ai, ftz(dual.unsafe_load(j)), ftz(-kinv.unsafe_load(i * n + j))))
            acc = ftz(identical_mul_add(w, ftz(dk.unsafe_load(base + j * n + i)), acc))
    part.unsafe_store(t, acc)


def gp_grad_fin_item(p: Int, part: _P, nb: Int, half: Float32, dst: _P):
    """Parameter p's partials added ascending, times `half` (0.5)."""
    var acc = Float32(0.0)
    for b in range(nb):
        acc = ftz(acc + part.unsafe_load(p * nb + b))
    dst.unsafe_store(p, ftz(identical_mul(half, acc)))


# the kernel tree's node kinds (gaussian_process/host/gp_theta.mojo's _K_*)
comptime GP_KIND_RBF = 2
comptime GP_KIND_MATERN = 3
comptime GP_KIND_SUM = 4
comptime GP_KIND_PROD = 5


def gp_free_count(
    kinds: List[Int32], ls_len: List[Int32], free: List[Int32]
) raises -> Int:
    """The number of theta entries: 1 per free CONST or WHITE leaf, `ls_len`
    per free RBF or MATERN leaf. Refuses a flag list the two sides of the
    binding disagree about, by name."""
    if len(free) != len(kinds) or len(ls_len) != len(kinds):
        raise Error(
            "gpr_lml_grad: the free-flag list holds "
            + String(len(free))
            + " entries for a kernel of "
            + String(len(kinds))
            + " postfix nodes; one flag per node is required"
        )
    var n = 0
    for t in range(len(kinds)):
        var f = Int(free[t])
        var k = Int(kinds[t])
        if f != 0 and f != 1:
            raise Error(
                "gpr_lml_grad: node " + String(t) + " has free flag "
                + String(f) + "; a flag is 0 (fixed) or 1 (free)"
            )
        if f == 0:
            continue
        if k == GP_KIND_SUM or k == GP_KIND_PROD:
            raise Error(
                "gpr_lml_grad: node " + String(t) + " is a Sum or Product"
                " marked free; only leaves carry hyperparameters"
            )
        if k == GP_KIND_RBF or k == GP_KIND_MATERN:
            n += Int(ls_len[t])
        else:
            n += 1
    return n


