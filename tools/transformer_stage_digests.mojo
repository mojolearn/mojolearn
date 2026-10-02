# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Per-stage digests of the transformer HOST oracle at a chosen shape, for a
cross-host comparison (lane neural-pass21, 2026-10-01): two hosts that print
different digests for a stage disagree there, and the first differing stage
names the helper to read.

    MOJOLEARN_TSD_L=2048 pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . tools/transformer_stage_digests.mojo

Shape: batch 1, length L (env MOJOLEARN_TSD_L, default 2048), d_model 384,
6 heads, 6 kv heads, head_dim 64, intermediate 1024 (the ab check's shape),
every tensor from the fixture generator (`fixture_tensor`, seed 7), a fresh
cache of L slots, no window, no plant. The digest is FNV-1a 64 over the
stage's float32 bytes; the stage's length is printed with it.
"""
from std.memory import bitcast
from std.os import getenv

from transformer.checks.transformer_fixture import (
    ScorePlant,
    TID_NORM1_W,
    TID_NORM2_W,
    TID_W_DOWN,
    TID_W_GATE,
    TID_W_K,
    TID_W_O,
    TID_W_Q,
    TID_W_UP,
    TID_W_V,
    TID_X,
    TransformerDims,
    TransformerWeights,
    fixture_tensor,
)
from transformer.checks.transformer_oracle import (
    TransformerKVCache,
    build_rope_table,
    oracle_dump,
    stage_tag,
    transformer_block_oracle,
)


def _fnv(values: List[Float32]) -> UInt64:
    var h = UInt64(14695981039346656037)
    for i in range(len(values)):
        var u = bitcast[DType.uint32](values[i])
        for k in range(4):
            var byte = UInt64((u >> UInt32(8 * k)) & UInt32(255))
            h = (h ^ byte) * UInt64(1099511628211)
    return h


def _hex(h: UInt64) -> String:
    var digits = "0123456789abcdef"
    var out = String("")
    for k in range(16):
        var nib = Int((h >> UInt64(60 - 4 * k)) & UInt64(15))
        out += String(digits[byte=nib])
    return out


def main() raises:
    var l = 2048
    try:
        l = Int(String(getenv("MOJOLEARN_TSD_L", "2048")))
    except:
        l = 2048
    var b = 1
    var dims = TransformerDims(384, 6, 6, 64, 1024, l)
    dims.validate()
    var dm = dims.d_model
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var inter = dims.intermediate
    var seed = UInt64(7)
    var w = TransformerWeights(dims)
    w.norm1_w = fixture_tensor(seed, TID_NORM1_W, dm, 0.5, 1.5)
    w.norm2_w = fixture_tensor(seed, TID_NORM2_W, dm, 0.5, 1.5)
    w.w_q = fixture_tensor(seed, TID_W_Q, qw * dm, -0.5, 0.5)
    w.w_k = fixture_tensor(seed, TID_W_K, kw * dm, -0.5, 0.5)
    w.w_v = fixture_tensor(seed, TID_W_V, kw * dm, -0.5, 0.5)
    w.w_o = fixture_tensor(seed, TID_W_O, dm * qw, -0.25, 0.25)
    w.w_gate = fixture_tensor(seed, TID_W_GATE, inter * dm, -0.25, 0.25)
    w.w_up = fixture_tensor(seed, TID_W_UP, inter * dm, -0.25, 0.25)
    w.w_down = fixture_tensor(seed, TID_W_DOWN, dm * inter, -0.125, 0.125)
    var x = fixture_tensor(seed, TID_X, b * l * dm, -2.0, 2.0)
    var cache = TransformerKVCache(b, dims, l, 0)
    var rope = build_rope_table(dims)
    var st = transformer_block_oracle(w, x, b, l, cache, rope, ScorePlant.none())
    var stages = oracle_dump(st)
    print("transformer_stage_digests: L", l, "stages", len(stages))
    for i in range(len(stages)):
        print("  ", stage_tag(i), len(stages[i]), _hex(_fnv(stages[i])))
    _ = st^
