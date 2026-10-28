# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Compile-only probe (lane/attn-fold): instantiates the byte-LM host
attention folds (`_softmax_head`, `_value_sum_head`) so the v2 key-axis fold
branch type-checks without the byte_lm host binding entry, whose provider
file is red on main independently of this lane. Never run."""

from training.byte_lm_host_kernels import _softmax_head, _value_sum_head


def main():
    var l = 2
    var s = 2
    var hd = 4
    var cell = List[Float32](length=l * s, fill=Float32(0.0))
    var masks = List[Float32](length=l * s, fill=Float32(0.0))
    var aw = List[Float32](length=l * s, fill=Float32(0.0))
    var vp = List[Float32](length=s * hd, fill=Float32(0.0))
    var ctx = List[Float32](length=l * hd, fill=Float32(0.0))
    _softmax_head(cell, masks, l, s, Float32(1.0), aw, 0)
    _value_sum_head(aw, vp, l, s, hd, hd, 0, ctx, 0)
    print(ctx[0])
