# SPDX-License-Identifier: Apache-2.0
"""NI20 native host model adapters. No GPU dependency or Python arithmetic."""
from transformer.impl.llama.attention_v2_model_contract import (
    V2Ptr, v2m_forward_row, v2m_forward_diagnostic, v2m_prepare_row,
    v2m_dq_row, v2m_dkdv_key, v2m_backward_diagnostic,
)


def _ptr(values: List[Float32]) -> V2Ptr:
    # Owner lists outlive every synchronous native call below.
    return V2Ptr(unsafe_from_address=Int(values.unsafe_ptr()))


struct AttentionV2HostForward(Movable):
    var output: List[Float32]
    var maxima: List[Float32]
    var denominators: List[Float32]
    var scores: List[Float32]
    var masked: List[Float32]
    var exps: List[Float32]
    var weights: List[Float32]

    def __init__(out self, b: Int, length: Int, heads: Int, keys: Int,
                 depth: Int, diagnostic: Bool):
        var rows = b * length * heads
        var cells = rows * keys if diagnostic else 1
        self.output = List[Float32](length=rows * depth, fill=Float32(0.0))
        self.maxima = List[Float32](length=rows, fill=Float32(0.0))
        self.denominators = List[Float32](length=rows, fill=Float32(0.0))
        self.scores = List[Float32](length=cells, fill=Float32(0.0))
        self.masked = List[Float32](length=cells, fill=Float32(0.0))
        self.exps = List[Float32](length=cells, fill=Float32(0.0))
        self.weights = List[Float32](length=cells, fill=Float32(0.0))


def attention_v2_host_forward(q: List[Float32], k: List[Float32], v: List[Float32],
                              b: Int, length: Int, heads: Int, kv_heads: Int,
                              keys: Int, depth: Int, own0: Int, window: Int,
                              scale: Float32, diagnostic: Bool = True) -> AttentionV2HostForward:
    var out = AttentionV2HostForward(b, length, heads, keys, depth, diagnostic)
    for row in range(b * heads * length):
        v2m_forward_row(_ptr(q), _ptr(k), _ptr(v), _ptr(out.output),
                       _ptr(out.maxima), _ptr(out.denominators), row, length,
                       heads, kv_heads, keys, depth, own0, window, scale)
        if diagnostic:
            v2m_forward_diagnostic(_ptr(q), _ptr(k), _ptr(out.maxima),
                                  _ptr(out.denominators), _ptr(out.scores),
                                  _ptr(out.masked), _ptr(out.exps), _ptr(out.weights),
                                  row, length, heads, kv_heads, keys, depth,
                                  own0, window, scale)
    return out^


struct AttentionV2HostBackward(Movable):
    var dq: List[Float32]
    var dk: List[Float32]
    var dv: List[Float32]
    var zdot: List[Float32]
    var dw: List[Float32]
    var dmasked: List[Float32]
    var dscores: List[Float32]
    var dqk: List[Float32]

    def __init__(out self, b: Int, length: Int, heads: Int, kv_heads: Int,
                 keys: Int, depth: Int):
        var rows = b * length * heads
        var cells = rows * keys
        self.dq = List[Float32](length=rows * depth, fill=Float32(0.0))
        self.dk = List[Float32](length=b * kv_heads * keys * depth, fill=Float32(0.0))
        self.dv = List[Float32](length=b * kv_heads * keys * depth, fill=Float32(0.0))
        self.zdot = List[Float32](length=rows, fill=Float32(0.0))
        self.dw = List[Float32](length=cells, fill=Float32(0.0))
        self.dmasked = List[Float32](length=cells, fill=Float32(0.0))
        self.dscores = List[Float32](length=cells, fill=Float32(0.0))
        self.dqk = List[Float32](length=cells, fill=Float32(0.0))


def attention_v2_host_backward(q: List[Float32], k: List[Float32], v: List[Float32],
                               dy: List[Float32], b: Int, length: Int, heads: Int,
                               kv_heads: Int, keys: Int, depth: Int, own0: Int,
                               window: Int, scale: Float32) -> AttentionV2HostBackward:
    var out = AttentionV2HostBackward(b, length, heads, kv_heads, keys, depth)
    var maxima = List[Float32](length=b * heads * length, fill=Float32(0.0))
    var denominators = List[Float32](length=b * heads * length, fill=Float32(0.0))
    for row in range(b * heads * length):
        v2m_prepare_row(_ptr(q), _ptr(k), _ptr(v), _ptr(dy), _ptr(maxima),
                       _ptr(denominators), _ptr(out.zdot), row, length, heads,
                       kv_heads, keys, depth, own0, window, scale)
        v2m_dq_row(_ptr(q), _ptr(k), _ptr(v), _ptr(dy), _ptr(maxima),
                  _ptr(denominators), _ptr(out.zdot), _ptr(out.dq), row,
                  length, heads, kv_heads, keys, depth, own0, window, scale)
        v2m_backward_diagnostic(_ptr(q), _ptr(k), _ptr(v), _ptr(dy),
                               _ptr(maxima), _ptr(denominators), _ptr(out.zdot),
                               _ptr(out.dw), _ptr(out.dmasked), _ptr(out.dscores),
                               _ptr(out.dqk), row, length, heads, kv_heads, keys,
                               depth, own0, window, scale)
    for idx in range(b * kv_heads * keys):
        v2m_dkdv_key(_ptr(q), _ptr(k), _ptr(v), _ptr(dy), _ptr(maxima),
                    _ptr(denominators), _ptr(out.zdot), _ptr(out.dk), _ptr(out.dv),
                    idx, length, heads, kv_heads, keys, depth, own0, window, scale)
    return out^
