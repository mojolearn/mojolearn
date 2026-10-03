# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Label layout units (lane apple-fast-py2mojo-core, 2026-10-03): the one-hot
flags, the repeated row weights and the two-column binary scores that
`python/mojolearn/_expansion_metrics.py` built on the host (`_ovr`,
`_micro_inputs`, `top_k_accuracy_score`) before uploading them. One thread
per output word; integer words and bit copies only, so no float bit can move
and the device and the host binding agree by construction."""
from std.memory import bitcast
from x_metrics.common import FP, IP, p, ldi, sti


def onehot_unit(t: Int, f: FP, q: IP):
    """q = [C, OUT, n, k, layout]; t in [0, n*k). layout 0 (class-major):
    word t = c*n + r; layout 1 (row-major): word t = r*k + c. OUT[t] is the
    Int32 word 1 when the code ldi(C + r) equals c, else 0."""
    var C = p(q, 0)
    var OUT = p(q, 1)
    var n = p(q, 2)
    var k = p(q, 3)
    var r: Int
    var c: Int
    if p(q, 4) == 0:
        c = t // n
        r = t - c * n
    else:
        r = t // k
        c = t - r * k
    if ldi(f, C + r) == c:
        sti(f, OUT + t, 1)
    else:
        sti(f, OUT + t, 0)


def rep_rows_unit(t: Int, f: FP, q: IP):
    """q = [W, OUT, k]; t in [0, n*k). OUT[t] = W[t // k], the word copied
    bit for bit (each row's weight repeated over its k one-hot words)."""
    var W = p(q, 0)
    var OUT = p(q, 1)
    var k = p(q, 2)
    f.unsafe_store(OUT + t, f.unsafe_load(W + t // k))


def pair_cols_unit(t: Int, f: FP, q: IP):
    """q = [S, OUT, THR]; t = row. OUT[2t] = the float32 whose bits are the
    Int32 THR, OUT[2t + 1] = S[t] bit for bit: the two score columns
    [thr, s] of the binary top-1 accuracy."""
    var S = p(q, 0)
    var OUT = p(q, 1)
    f.unsafe_store(OUT + 2 * t, bitcast[DType.float32](Int32(p(q, 2))))
    f.unsafe_store(OUT + 2 * t + 1, f.unsafe_load(S + t))
