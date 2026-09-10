# SPDX-License-Identifier: Apache-2.0
"""Numerical-only RoPE table export; no opponent timing or changed arithmetic."""
from std.memory import bitcast
from std.os import getenv
from max.gpu.host import DeviceContext
from transformer.impl.transformers.models.llama.modeling_llama import (
    LlamaDims, LlamaRopeTable, _download,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


def main() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var ctx = DeviceContext()
    var positions: List[Int] = [0, 1, 127, 511, 1023, 2047, 4095, 8191]
    var full = String(getenv("MOJOLEARN_TRANSFORMER_ROPE_FULL")) != ""
    if full:
        positions.clear()
        for p in range(4096):
            positions.append(p)
    for hd in range(64, 129, 64):
        if full and hd != 128:
            continue
        var dims = LlamaDims(hd, 1, 1, hd, hd * 2)
        var rope = LlamaRopeTable(ctx, dims, Float32(10000.0), 8192)
        var inv = _download(ctx, rope.inv_freq, hd // 2)
        var cos = _download(ctx, rope.cos, 8192 * hd // 2)
        var sin = _download(ctx, rope.sin, 8192 * hd // 2)
        for p in positions:
            for i in range(hd // 2):
                var j = p * (hd // 2) + i
                print("ROPE", hd, p, i, bitcast[DType.uint32](inv[i]), bitcast[DType.uint32](cos[j]), bitcast[DType.uint32](sin[j]))
        _ = rope^
