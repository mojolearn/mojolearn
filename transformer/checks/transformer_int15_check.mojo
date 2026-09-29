# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The transformer block under `numeric_profile="fixed15_v1"`, inference
(lane/lowbit-blocks, 2026-09-29). Device against the host oracle, bit for bit.

    pixi run check-transformer-int15              # every phase below
    pixi run check-transformer-int15-sabotage     # the same binary with
                                                  # -D MOJOLEARN_LOWBIT_SABOTAGE=1;
                                                  # MUST report failures

WHAT EACH PHASE SAYS
  default  the SAME fixture cases with no planes: the device card equals the
           fp32 host oracle (the block's clause (a)); the dump digest of each
           case is printed so the run can be held line for line against the
           same program on main's modeling_llama (`int15 default digest`).
  profile  the seven projections and S11 under the profile, P.V on fp32: the
           device's thirty stages equal `transformer_block_oracle(...,
           int15=True)`, stage by stage. The device's weight planes are made
           by the PARALLEL quantizer on the device from the uploaded float32
           weights; the oracle quantizes by `quantize_rows_int15` on the
           host. Two spellings, one set of codes (W-10).
  decode   prefill of `p` tokens then one decode step per token, `p` in
           {1, 7, L-1}: every token's block output equals the full prefill's
           row, bit for bit, and the host oracle's decode.
  batch    B = 3 against each row alone at B = 1: the same rows.
The verdict line is `int15 block check: <n> failures`; the program exits
non-zero when n > 0. Under the sabotage define every profile phase must fail
and the default phase must not.
"""

from std.sys import exit
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from transformer.block_options import BlockOptions
from transformer.checks.transformer_fixture import (
    RMS_EPS,
    ROPE_THETA,
    ScorePlant,
    TransformerDims,
    TransformerWeights,
    fixture_case,
    fixture_dims,
    fixture_weights,
    fixture_window,
    fixture_x,
)
from transformer.checks.transformer_oracle import (
    TransformerKVCache,
    build_rope_table,
    oracle_dump,
    transformer_block_oracle,
)
from transformer.checks.transformer_check import (
    compare_dumps,
    count_moved,
    device_dump,
    first_moved,
    llama_dims_of,
)
from transformer.impl.llama.modeling_llama import (
    PLANT_AT_NONE,
    LlamaDeviceStages,
    LlamaDeviceWeights,
    LlamaDims,
    LlamaKVCache,
    LlamaRopeTable,
    _download,
    _upload,
    _zeros,
    llama_decoder_layer_forward,
    llama_decoder_layer_forward_planted,
)
from transformer.impl.llama.int15_block import (
    Int15Planes,
    LlamaInt15Weights,
    int15_planes_from_f32,
)
from core.identity_trace import IdentityTrace
from gemm.checks.gemm_int15 import Int15QuantWorkspace, int15_sabotage_name


# The single-call fixture cases of `transformer_check.clause_a_cases`,
# without the planted ones (a plant writes bits into the fp32 score stage;
# the profile changes nothing there that the unplanted cases do not show).
def profile_cases() -> List[Int]:
    return [0, 1, 2, 3, 4, 5, 7, 15, 16, 17, 18]


def digest(dump: List[List[Float32]]) -> UInt64:
    """FNV-1a 64 over every stage's float bits, in card order."""
    var h = UInt64(0xCBF29CE484222325)
    for i in range(len(dump)):
        for j in range(len(dump[i])):
            var u = bitcast[DType.uint32](dump[i][j])
            for s in range(4):
                h ^= UInt64((u >> UInt32(8 * s)) & 0xFF)
                h *= UInt64(0x100000001B3)
    return h


def digest_list(v: List[Float32]) -> UInt64:
    var d = List[List[Float32]]()
    d.append(v.copy())
    return digest(d)


def hex64(v: UInt64) -> String:
    var digits = String("0123456789abcdef")
    var out = String("")
    for i in range(16):
        var nib = Int((v >> UInt64(4 * (15 - i))) & 0xF)
        out += String(digits[byte=nib])
    return out


def device_weights(
    ctx: DeviceContext, dims: LlamaDims, w: TransformerWeights, int15: Bool
) raises -> LlamaDeviceWeights:
    """The default constructor's weights, or the profile constructor's: the
    seven float32 weights uploaded and quantized on the device by the
    parallel quantizer, norms uploaded, no optional tensor."""
    if not int15:
        return LlamaDeviceWeights(
            ctx, dims, RMS_EPS, w.norm1_w, w.norm2_w, w.w_q, w.w_k, w.w_v,
            w.w_o, w.w_gate, w.w_up, w.w_down,
        )
    var dm = dims.d_model
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var it = dims.intermediate
    var quant = Int15QuantWorkspace(ctx)
    var fq = _upload(ctx, w.w_q)
    var fk = _upload(ctx, w.w_k)
    var fv = _upload(ctx, w.w_v)
    var fo = _upload(ctx, w.w_o)
    var fg = _upload(ctx, w.w_gate)
    var fu = _upload(ctx, w.w_up)
    var fd = _upload(ctx, w.w_down)
    var planes = LlamaInt15Weights(
        int15_planes_from_f32(ctx, fq, qw, dm, quant),
        int15_planes_from_f32(ctx, fk, kw, dm, quant),
        int15_planes_from_f32(ctx, fv, kw, dm, quant),
        int15_planes_from_f32(ctx, fo, dm, qw, quant),
        int15_planes_from_f32(ctx, fg, it, dm, quant),
        int15_planes_from_f32(ctx, fu, it, dm, quant),
        int15_planes_from_f32(ctx, fd, dm, it, quant),
        True,
    )
    _ = quant^
    return LlamaDeviceWeights(
        ctx, dims, BlockOptions(), _upload(ctx, w.norm1_w),
        _upload(ctx, w.norm2_w), planes^,
        _zeros(ctx, 1), _zeros(ctx, 1), _zeros(ctx, 1), _zeros(ctx, 1),
        _zeros(ctx, 1), _zeros(ctx, 1), _zeros(ctx, 1), _zeros(ctx, 1),
        _zeros(ctx, 1), _zeros(ctx, 1), _zeros(ctx, 1),
    )


def device_card(
    ctx: DeviceContext, k: Int, int15: Bool
) raises -> List[List[Float32]]:
    var c = fixture_case(k)
    var dims = fixture_dims(c)
    var ld = llama_dims_of(dims)
    var w = fixture_weights(c)
    var x = fixture_x(c)
    var window = fixture_window(String(c.name))
    var dw = device_weights(ctx, ld, w, int15)
    var kv = LlamaKVCache(ctx, c.b, ld, c.cache_cap, window)
    var rope = LlamaRopeTable(ctx, ld, ROPE_THETA, dims.rope_positions)
    var stages = LlamaDeviceStages(ctx, c.b, c.l, c.cache_cap, ld, window)
    var dx = _upload(ctx, x)
    var trace = IdentityTrace.disabled()
    llama_decoder_layer_forward_planted(
        ctx, stages, kv, rope, dw, dx, c.b, c.l, 0, PLANT_AT_NONE,
        List[Int](), List[UInt32](), trace, String("int15"),
    )
    var out = device_dump(ctx, stages, rope, dx, c.b, c.l, c.l, dims)
    _ = dw^
    _ = kv^
    _ = rope^
    _ = stages^
    _ = dx^
    return out^


def host_card(k: Int, int15: Bool) raises -> List[List[Float32]]:
    var c = fixture_case(k)
    var dims = fixture_dims(c)
    var w = fixture_weights(c)
    var x = fixture_x(c)
    var cache = TransformerKVCache(c.b, dims, c.cache_cap, fixture_window(String(c.name)))
    var rope = build_rope_table(dims)
    var st = transformer_block_oracle(w, x, c.b, c.l, cache, rope, ScorePlant.none(), int15)
    return oracle_dump(st)


def phase_cards(ctx: DeviceContext, int15: Bool) raises -> Int:
    var fails = 0
    var tag = String("profile") if int15 else String("default")
    var cases = profile_cases()
    for i in range(len(cases)):
        var k = cases[i]
        var c = fixture_case(k)
        var host = host_card(k, int15)
        var dev = device_card(ctx, k, int15)
        var diffs = compare_dumps(host, dev, False)
        var moved = count_moved(diffs)
        var line = (
            "  " + tag + " case " + String(k) + " " + String(c.name)
            + " B=" + String(c.b) + " L=" + String(c.l)
            + " dm=" + String(c.d_model) + " inter=" + String(c.intermediate)
            + " hd=" + String(c.head_dim)
            + "  device " + hex64(digest(dev)) + " host " + hex64(digest(host))
        )
        if moved == 0:
            print(line + "  30/30 stages bit-identical")
        else:
            fails += 1
            print(line + "  " + String(moved) + " of 30 stages MOVED, first at " + first_moved(diffs))
        if not int15:
            print("  int15 default digest case " + String(k) + " " + hex64(digest(dev)))
    return fails


def rows_of(v: List[Float32], bb: Int, t0: Int, t1: Int, l: Int, dm: Int) -> List[Float32]:
    var out = List[Float32]()
    for t in range(t0, t1):
        for j in range(dm):
            out.append(v[(bb * l + t) * dm + j])
    return out^


def device_run_split(
    ctx: DeviceContext, k: Int, p: Int, b_only: Int
) raises -> List[Float32]:
    """The block output of case `k` as `[B, L, dm]`: one prefill when
    `p == L`, else a prefill of `p` tokens and one decode step per later
    token, the cache carried. `b_only >= 0` runs that batch row alone."""
    var c = fixture_case(k)
    var dims = fixture_dims(c)
    var ld = llama_dims_of(dims)
    var w = fixture_weights(c)
    var xall = fixture_x(c)
    var dm = dims.d_model
    var b = c.b
    var l = c.l
    var x = xall.copy()
    if b_only >= 0:
        x = rows_of(xall, b_only, 0, l, l, dm)
        b = 1
    var window = fixture_window(String(c.name))
    var dw = device_weights(ctx, ld, w, True)
    var kv = LlamaKVCache(ctx, b, ld, l, window)
    var rope = LlamaRopeTable(ctx, ld, ROPE_THETA, dims.rope_positions)
    var out = List[Float32](length=b * l * dm, fill=Float32(0.0))
    var t0 = 0
    while t0 < l:
        var n = p if t0 == 0 else 1
        var xs = List[Float32]()
        for bb in range(b):
            var r = rows_of(x, bb, t0, t0 + n, l, dm)
            for j in range(len(r)):
                xs.append(r[j])
        var stages = LlamaDeviceStages(ctx, b, n, l, ld, window)
        var dx = _upload(ctx, xs)
        var trace = IdentityTrace.disabled()
        llama_decoder_layer_forward(ctx, stages, kv, rope, dw, dx, b, n, t0, trace, String("dec"))
        var y = _download(ctx, stages.residual2, b * n * dm)
        for bb in range(b):
            for t in range(n):
                for j in range(dm):
                    out[(bb * l + t0 + t) * dm + j] = y[(bb * n + t) * dm + j]
        _ = stages^
        _ = dx^
        t0 += n
    _ = dw^
    _ = kv^
    _ = rope^
    return out^


def host_run_split(k: Int, p: Int) raises -> List[Float32]:
    var c = fixture_case(k)
    var dims = fixture_dims(c)
    var w = fixture_weights(c)
    var x = fixture_x(c)
    var dm = dims.d_model
    var b = c.b
    var l = c.l
    var cache = TransformerKVCache(b, dims, l, fixture_window(String(c.name)))
    var rope = build_rope_table(dims)
    var out = List[Float32](length=b * l * dm, fill=Float32(0.0))
    var t0 = 0
    while t0 < l:
        var n = p if t0 == 0 else 1
        var xs = List[Float32]()
        for bb in range(b):
            var r = rows_of(x, bb, t0, t0 + n, l, dm)
            for j in range(len(r)):
                xs.append(r[j])
        var st = transformer_block_oracle(w, xs, b, n, cache, rope, ScorePlant.none(), True)
        for bb in range(b):
            for t in range(n):
                for j in range(dm):
                    out[(bb * l + t0 + t) * dm + j] = st.residual2_out[(bb * n + t) * dm + j]
        t0 += n
    return out^


def count_diff(a: List[Float32], b: List[Float32]) -> Int:
    if len(a) != len(b):
        return -1
    var n = 0
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            n += 1
    return n


def phase_decode(ctx: DeviceContext) raises -> Int:
    var fails = 0
    var cases: List[Int] = [3, 15]  # base_b3_l16_nrep2, win4_b1_l16_nrep2
    for ci in range(len(cases)):
        var k = cases[ci]
        var c = fixture_case(k)
        var full = device_run_split(ctx, k, c.l, -1)
        var ps: List[Int] = [1, 7, c.l - 1]
        for pi in range(len(ps)):
            var p = ps[pi]
            var dev = device_run_split(ctx, k, p, -1)
            var host = host_run_split(k, p)
            var d1 = count_diff(full, dev)
            var d2 = count_diff(host, dev)
            var ok = d1 == 0 and d2 == 0
            if not ok:
                fails += 1
            print(
                "  decode case " + String(k) + " " + String(c.name) + " prefix "
                + String(p) + ": device prefill+decode vs device prefill "
                + String(d1) + " cells differ, vs host oracle " + String(d2)
                + "  digest " + hex64(digest_list(dev))
                + ("  EQUAL" if ok else "  MOVED")
            )
    return fails


def phase_batch(ctx: DeviceContext) raises -> Int:
    var k = 3  # base_b3_l16_nrep2
    var c = fixture_case(k)
    var dm = c.d_model
    var full = device_run_split(ctx, k, c.l, -1)
    var fails = 0
    for bb in range(c.b):
        var alone = device_run_split(ctx, k, c.l, bb)
        var d = count_diff(rows_of(full, bb, 0, c.l, c.l, dm), alone)
        if d != 0:
            fails += 1
        print(
            "  batch case " + String(k) + " row " + String(bb) + " of B="
            + String(c.b) + " against B=1: " + String(d) + " cells differ"
            + ("  EQUAL" if d == 0 else "  MOVED")
        )
    return fails


def main() raises:
    print("int15 block check (lane/lowbit-blocks): sabotage=" + int15_sabotage_name())
    var ctx = DeviceContext()
    var fails = 0
    print("phase default (no planes; fp32.v1 against its host oracle)")
    fails += phase_cards(ctx, False)
    print("phase profile (fixed15_v1: projections and S11 int15, P.V fp32)")
    fails += phase_cards(ctx, True)
    print("phase decode (prefill of p tokens then decode, p in 1, 7, L-1)")
    fails += phase_decode(ctx)
    print("phase batch (B=3 against each row alone)")
    fails += phase_batch(ctx)
    print("int15 block check: " + String(fails) + " failures")
    if fails > 0:
        exit(1)
