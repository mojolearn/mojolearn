"""MiniBatchKMeans' center update as cells (lane/neural-pass133, 2026-10-02).

`x_cluster/minibatch.mojo::minibatch_step`'s unit-weight `update_center_dense`
for one center (the host ops' loop) and for one (center, feature) word (the
device kernel's thread), the same statements: wsum counts the center's rows
in batch order; a center with rows gets `c * w`, then `+ x` for its rows in
batch order, then `* (1 / (w + wsum))`; its count becomes `w + wsum`."""
from experiments.classical_identical_ideas.graph_controls import C37_ROW_PANELS, C37_PANEL_ROWS
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632

from checks.numerics import ftz, identical_div, identical_mul

comptime FPtr = MutPointer[Float32, MutAnyOrigin]
comptime IPtr = MutPointer[Int32, MutAnyOrigin]


@always_inline
def mb_center_wsum(pl: IPtr, batch: Int, j: Int) -> Float32:
    var wsum = Float32(0)
    for t in range(batch):
        if Int(pl[t]) == j:
            wsum = ftz(wsum + Float32(1))
    return wsum


@always_inline
def mb_center_word(pb: FPtr, batch: Int, pl: IPtr, cjf: Float32, wj: Float32, wsum: Float32, j: Int, f: Int,
                   d: Int) -> Float32:
    """The center word (j, f) after the batch (wsum > 0)."""
    var acc = ftz(identical_mul(cjf, wj))
    comptime if C37_ROW_PANELS:
        for start in range(0,batch,C37_PANEL_ROWS):
            var part=Float32(0)
            for t in range(start,min(start+C37_PANEL_ROWS,batch)):
                if Int(pl[t])==j:
                    part=ftz(part+ftz(pb[t*d+f]))
            acc=ftz(acc+part)
    else:
        for t in range(batch):
            if Int(pl[t]) == j:
                acc = ftz(acc + ftz(pb[t * d + f]))
    var alpha = ftz(identical_div(Float32(1), ftz(wj + wsum)))
    return ftz(identical_mul(acc, alpha))


@always_inline
def mb_center_update(pb: FPtr, batch: Int, pl: IPtr, pc: FPtr, pw: FPtr, j: Int, d: Int):
    var wsum = mb_center_wsum(pl, batch, j)
    if wsum > Float32(0):
        var wj = pw[j]
        for f in range(d):
            pc[j * d + f] = mb_center_word(pb, batch, pl, pc[j * d + f], wj, wsum, j, f, d)
        pw[j] = ftz(wj + wsum)
