# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The public `randomized_svd` as ONE native operation (lane
a3-admit-decomp-prep-metrics, 2026-11-08).

It was sequenced by bindings/linalg_decomp_native.mojo through the binding
decomp kit (`DecompKit`): one Python-level call into an x_decomp entry per
step (the sketch draw, every product, orthonormalization, fold, move and
the small SVD), about 40 calls for 4 power iterations. The same steps now
run on the randomized PCA kit (`RKit`, x_decomp/pca_randomized.mojo) that
PCA(svd_solver='randomized') and TruncatedSVD(algorithm='randomized')
already run on, through `pca_randomized_core` (`_rsvd_core`: the Philox
test matrix of stream 30, the power iterations, the orth rank guard on the
live columns, the small thin SVD and its paddings), then scikit-learn's
`svd_flip(u_based_decision=True)` on the returned factors.

Bits: one kit for every caller. The sign flip runs in the selected mode's
binding (it ran in the process-default mode's kit before; a +-1 multiply
and an argmax, so only the binding that runs them changed), and every step
follows RKit's residency (on a GPU binding every non-empty step is
resident). Same bits on NVIDIA, AMD and Apple within this version; the host
column runs the same steps on the executor."""
from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.pca_randomized import RDev, RKit, pca_randomized_core

comptime _OP_MUL = 2


def randomized_svd_run[E: Exec, D: RDev, RES: Bool](
    x: F32Ptr, u_out: F32Ptr, s_out: F32Ptr, vt_out: F32Ptr, rows: Int, cols: Int, components: Int,
    oversamples: Int, iterations: Int, auto_iterations: Bool, transpose: Bool, flip_sign: Bool, seed: Int,
    finite_word: Int = 0,
) raises:
    """`randomized_svd(M, components, ...)` of the rows x cols float32 M at
    `x` (read in place, never written): U (rows x components), S
    (components) and Vt (components x cols) written. `transpose`: the sketch
    runs on M^T (the caller resolved 'auto'). `auto_iterations`: n_iter='auto'
    (scikit-learn's size-ratio rule).

    `finite_word` (RSVD_FAST_DEVSCAN, x_decomp/s_linalg_fast.mojo; restored
    2026-11-09 by lane r1-restore-measured-winners): when nonzero, the host
    address of one Int32 that receives M's finite status word (0 = every
    word finite), tested on the device copy at M's one upload
    (RKit.device_tested) instead of a host pass over M; a flagged M stops
    the run (nothing written; the door refuses). Only a resident kit takes
    it (the door passes it on the FAST + Apple build alone)."""
    if not (1 <= components and components <= min(rows, cols)):
        raise Error("n_components must be in [1, min(n_samples, n_features)]")
    var power = iterations
    if auto_iterations:
        # n_iter='auto' by the problem's size ratio (not a dimension threshold)
        power = 7 if Float64(components) < 0.1 * Float64(min(rows, cols)) else 4
    var kit = RKit[E, D, RES]()
    try:
        var original = kit.add(Int(x), -1, rows, cols)
        comptime if RES:
            if finite_word != 0:
                var flagged = kit.device_tested(original)
                MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=finite_word)[] = flagged
                if flagged != 0:
                    kit.close()
                    return
        var decomposed = pca_randomized_core(kit, original, components, oversamples, power, seed, transpose)
        var u = decomposed[0]
        var s = decomposed[1]
        var vt = decomposed[2]
        if flip_sign:
            # svd_flip(u_based_decision=True): each column of U signed so its
            # largest-|.| entry is positive; Vt's rows follow. A column's sign
            # reads that column only, so flipping the kept factors equals
            # flipping before the slice.
            var signs = kit.absmax_signs(u, True)
            u = kit.ew(_OP_MUL, u, signs)
            var signst = kit.transpose(signs)
            vt = kit.ew(_OP_MUL, vt, signst)
        kit.copy_out(u, u_out, rows * components)
        kit.copy_out(s, s_out, components)
        kit.copy_out(vt, vt_out, components * cols)
        kit.close()
    except error:
        kit.close()
        raise error^
