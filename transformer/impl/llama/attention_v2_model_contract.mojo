# SPDX-License-Identifier: Apache-2.0
"""NI20 shared native-host/device attention V2 model arithmetic.

Fixed logical tile32 is part of the numerical profile, never a device tile.
Q/output/dQ/dOutput use token-major [B,L,H,D]; K/V/dK/dV use [B,KV,S,D].
Rows/stats/optional diagnostic cells use [B,H,L,(S)]. No allocation, device
intrinsic, Python work or vendor-dependent order appears in these functions.
The existing standalone V2 contract is retained with a model-layout adapter.
"""
from std.memory import bitcast
from checks.numerics import identical_div, identical_fmax, identical_mul, identical_mul_add, portable_expf

comptime ATTENTION_MODEL_V2_TILE = 32
comptime ATTENTION_MODEL_V2_PROFILE = "attention-online-tile32.fp32.v2"
comptime V2Ptr = MutPointer[Float32, MutAnyOrigin]


@no_inline
def v2m_mul(a: Float32, b: Float32) -> Float32:
    return identical_mul(a, b)


@no_inline
def v2m_add(a: Float32, b: Float32) -> Float32:
    return identical_mul_add(Float32(1.0), a, b)


def v2m_query_base(row: Int, length: Int, heads: Int, depth: Int) -> Int:
    var t = row % length
    var h = (row // length) % heads
    var batch = row // (length * heads)
    return ((batch * length + t) * heads + h) * depth


def v2m_key_base(row: Int, length: Int, heads: Int, kv_heads: Int, keys: Int, depth: Int) -> Int:
    var h = (row // length) % heads
    var batch = row // (length * heads)
    return (batch * kv_heads + h // (heads // kv_heads)) * keys * depth


def v2m_visible(row: Int, length: Int, keys: Int, own0: Int, window: Int) -> Tuple[Int, Int]:
    var absolute_in_span = own0 + row % length
    var lo = max(0, absolute_in_span - window + 1) if window > 0 else 0
    return lo, min(keys, absolute_in_span + 1)


def v2m_score(q: V2Ptr, k: V2Ptr, qb: Int, kb: Int, key: Int, depth: Int, scale: Float32) -> Float32:
    var acc = Float32(0.0)
    for d in range(depth):
        acc = identical_mul_add(q.unsafe_load(qb + d), k.unsafe_load(kb + key * depth + d), acc)
    return v2m_mul(acc, scale)


def v2m_normalizer(q: V2Ptr, k: V2Ptr, qb: Int, kb: Int, lo: Int, hi: Int, depth: Int, scale: Float32) -> Tuple[Float32, Float32]:
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var maximum = neg_inf
    var denominator = Float32(0.0)
    var first = lo
    while first < hi:
        var last = min(first + ATTENTION_MODEL_V2_TILE, hi)
        var tile_max = neg_inf
        for key in range(first, last):
            tile_max = identical_fmax(tile_max, v2m_score(q, k, qb, kb, key, depth, scale))
        var new_max = identical_fmax(maximum, tile_max)
        var correction = Float32(0.0) if maximum == neg_inf else portable_expf(maximum - new_max)
        denominator = v2m_mul(denominator, correction)
        for key in range(first, last):
            denominator = v2m_add(portable_expf(v2m_score(q, k, qb, kb, key, depth, scale) - new_max), denominator)
        maximum = new_max
        first += ATTENTION_MODEL_V2_TILE
    return maximum, denominator


def v2m_forward_row(q: V2Ptr, k: V2Ptr, v: V2Ptr, output: V2Ptr,
                    maxima: V2Ptr, denominators: V2Ptr, row: Int,
                    length: Int, heads: Int, kv_heads: Int, keys: Int,
                    depth: Int, own0: Int, window: Int, scale: Float32):
    var qb = v2m_query_base(row, length, heads, depth)
    var kb = v2m_key_base(row, length, heads, kv_heads, keys, depth)
    var visible = v2m_visible(row, length, keys, own0, window)
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var maximum = neg_inf
    var denominator = Float32(0.0)
    for d in range(depth):
        output.unsafe_store(qb + d, Float32(0.0))
    var first = visible[0]
    while first < visible[1]:
        var last = min(first + ATTENTION_MODEL_V2_TILE, visible[1])
        var tile_max = neg_inf
        for key in range(first, last):
            tile_max = identical_fmax(tile_max, v2m_score(q, k, qb, kb, key, depth, scale))
        var new_max = identical_fmax(maximum, tile_max)
        var correction = Float32(0.0) if maximum == neg_inf else portable_expf(maximum - new_max)
        denominator = v2m_mul(denominator, correction)
        for d in range(depth):
            output.unsafe_store(qb + d, v2m_mul(output.unsafe_load(qb + d), correction))
        for key in range(first, last):
            var weight = portable_expf(v2m_score(q, k, qb, kb, key, depth, scale) - new_max)
            denominator = v2m_add(weight, denominator)
            for d in range(depth):
                output.unsafe_store(qb + d, identical_mul_add(weight, v.unsafe_load(kb + key * depth + d), output.unsafe_load(qb + d)))
        maximum = new_max
        first += ATTENTION_MODEL_V2_TILE
    for d in range(depth):
        output.unsafe_store(qb + d, identical_div(output.unsafe_load(qb + d), denominator))
    maxima.unsafe_store(row, maximum)
    denominators.unsafe_store(row, denominator)


def v2m_dyv(v: V2Ptr, dy: V2Ptr, qb: Int, kb: Int, key: Int, depth: Int) -> Float32:
    var acc = Float32(0.0)
    for d in range(depth):
        acc = identical_mul_add(dy.unsafe_load(qb + d), v.unsafe_load(kb + key * depth + d), acc)
    return acc


def v2m_prepare_row(q: V2Ptr, k: V2Ptr, v: V2Ptr, dy: V2Ptr,
                    maxima: V2Ptr, denominators: V2Ptr, zdots: V2Ptr,
                    row: Int, length: Int, heads: Int, kv_heads: Int,
                    keys: Int, depth: Int, own0: Int, window: Int, scale: Float32):
    var qb = v2m_query_base(row, length, heads, depth)
    var kb = v2m_key_base(row, length, heads, kv_heads, keys, depth)
    var visible = v2m_visible(row, length, keys, own0, window)
    var norm = v2m_normalizer(q, k, qb, kb, visible[0], visible[1], depth, scale)
    var zdot = Float32(0.0)
    for key in range(visible[0], visible[1]):
        var p = identical_div(portable_expf(v2m_score(q, k, qb, kb, key, depth, scale) - norm[0]), norm[1])
        zdot = identical_mul_add(p, v2m_dyv(v, dy, qb, kb, key, depth), zdot)
    maxima.unsafe_store(row, norm[0])
    denominators.unsafe_store(row, norm[1])
    zdots.unsafe_store(row, zdot)


def v2m_dq_row(q: V2Ptr, k: V2Ptr, v: V2Ptr, dy: V2Ptr,
               maxima: V2Ptr, denominators: V2Ptr, zdots: V2Ptr, dq: V2Ptr,
               row: Int, length: Int, heads: Int, kv_heads: Int,
               keys: Int, depth: Int, own0: Int, window: Int, scale: Float32):
    var qb = v2m_query_base(row, length, heads, depth)
    var kb = v2m_key_base(row, length, heads, kv_heads, keys, depth)
    var visible = v2m_visible(row, length, keys, own0, window)
    for d in range(depth):
        dq.unsafe_store(qb + d, Float32(0.0))
    for key in range(visible[0], visible[1]):
        var p = identical_div(portable_expf(v2m_score(q, k, qb, kb, key, depth, scale) - maxima.unsafe_load(row)), denominators.unsafe_load(row))
        var ds = v2m_mul(v2m_mul(p, v2m_dyv(v, dy, qb, kb, key, depth) - zdots.unsafe_load(row)), scale)
        for d in range(depth):
            dq.unsafe_store(qb + d, identical_mul_add(ds, k.unsafe_load(kb + key * depth + d), dq.unsafe_load(qb + d)))


def v2m_dkdv_key(q: V2Ptr, k: V2Ptr, v: V2Ptr, dy: V2Ptr,
                 maxima: V2Ptr, denominators: V2Ptr, zdots: V2Ptr,
                 dk: V2Ptr, dv: V2Ptr, idx: Int, length: Int, heads: Int,
                 kv_heads: Int, keys: Int, depth: Int, own0: Int,
                 window: Int, scale: Float32):
    var key = idx % keys
    var group = idx // keys
    var batch = group // kv_heads
    var kvh = group % kv_heads
    var kb = group * keys * depth
    var outbase = idx * depth
    var repeats = heads // kv_heads
    for d in range(depth):
        dk.unsafe_store(outbase + d, Float32(0.0))
        dv.unsafe_store(outbase + d, Float32(0.0))
    # Canonical GQA gradient fold: head ascending, then query ascending.
    for h in range(kvh * repeats, (kvh + 1) * repeats):
        for t in range(length):
            var row = (batch * heads + h) * length + t
            var visible = v2m_visible(row, length, keys, own0, window)
            if key < visible[0] or key >= visible[1]:
                continue
            var qb = v2m_query_base(row, length, heads, depth)
            var p = identical_div(portable_expf(v2m_score(q, k, qb, kb, key, depth, scale) - maxima.unsafe_load(row)), denominators.unsafe_load(row))
            var ds = v2m_mul(v2m_mul(p, v2m_dyv(v, dy, qb, kb, key, depth) - zdots.unsafe_load(row)), scale)
            for d in range(depth):
                dk.unsafe_store(outbase + d, identical_mul_add(ds, q.unsafe_load(qb + d), dk.unsafe_load(outbase + d)))
                dv.unsafe_store(outbase + d, identical_mul_add(p, dy.unsafe_load(qb + d), dv.unsafe_load(outbase + d)))


def v2m_forward_diagnostic(q: V2Ptr, k: V2Ptr, maxima: V2Ptr,
                           denominators: V2Ptr, scores: V2Ptr, masked: V2Ptr,
                           exps: V2Ptr, weights: V2Ptr, row: Int, length: Int,
                           heads: Int, kv_heads: Int, keys: Int, depth: Int,
                           own0: Int, window: Int, scale: Float32):
    # V2 trace stages describe final-max probabilities; online ctx retains
    # its tile-rescaled rounding and is never reconstructed from this trace.
    var qb = v2m_query_base(row, length, heads, depth)
    var kb = v2m_key_base(row, length, heads, kv_heads, keys, depth)
    var visible = v2m_visible(row, length, keys, own0, window)
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    for key in range(keys):
        var cell = row * keys + key
        var score = v2m_score(q, k, qb, kb, key, depth, scale)
        var present = key >= visible[0] and key < visible[1]
        var exponent = portable_expf(score - maxima.unsafe_load(row)) if present else Float32(0.0)
        scores.unsafe_store(cell, score)
        masked.unsafe_store(cell, score if present else neg_inf)
        exps.unsafe_store(cell, exponent)
        weights.unsafe_store(cell, identical_div(exponent, denominators.unsafe_load(row)) if present else Float32(0.0))


def v2m_backward_diagnostic(q: V2Ptr, k: V2Ptr, v: V2Ptr, dy: V2Ptr,
                            maxima: V2Ptr, denominators: V2Ptr, zdots: V2Ptr,
                            dw: V2Ptr, dmasked: V2Ptr, dscores: V2Ptr, dqk: V2Ptr,
                            row: Int, length: Int, heads: Int, kv_heads: Int,
                            keys: Int, depth: Int, own0: Int, window: Int,
                            scale: Float32):
    var qb = v2m_query_base(row, length, heads, depth)
    var kb = v2m_key_base(row, length, heads, kv_heads, keys, depth)
    var visible = v2m_visible(row, length, keys, own0, window)
    for key in range(keys):
        var cell = row * keys + key
        var dp = v2m_dyv(v, dy, qb, kb, key, depth)
        var ds = Float32(0.0)
        if key >= visible[0] and key < visible[1]:
            var p = identical_div(portable_expf(v2m_score(q, k, qb, kb, key, depth, scale) - maxima.unsafe_load(row)), denominators.unsafe_load(row))
            ds = v2m_mul(p, dp - zdots.unsafe_load(row))
        dw.unsafe_store(cell, dp)
        dmasked.unsafe_store(cell, ds)
        dscores.unsafe_store(cell, ds)
        dqk.unsafe_store(cell, v2m_mul(ds, scale))
