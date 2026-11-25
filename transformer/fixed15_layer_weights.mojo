# SPDX-License-Identifier: Apache-2.0
"""The fixed15_v1 block weights on the device, shared by every entry that
builds them (lane fixed15-decode, 2026-11-25): the block forward and the
decode session open (transformer/transformer_binding_impl.mojo, op 35 and
decode open op 4) and the CausalLM decode program
(transformer/causal_lm_program.mojo), which adopts them into its attention
step entries. Moved word for word out of transformer_binding_impl so the
program (imported by the Mamba image too) does not import the whole block
binding: the same uploads, the same per-part checks, the same profile
constructor, so a layer has the same weights on every route.

The address layout `a` is the fixed15 block entry's (13 base slots, the
11-address options tail, then 21 plane addresses, three per projection in
the order q, k, v, o, gate, up, down: `hi`, `lo`, `exponents`).
"""
from std.memory import unsafe_memcpy
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr, i8_ptr, i32_ptr
from core.numeric_profile_parts import np_part_name, np_part_on
from core.numeric_profile_parts_sabotage import np_entry_parts
from transformer.block_options import BLOCK_OPTION_ADDRS, BlockOptions
from transformer.lean_stages import _lean_for
from transformer.impl.llama.modeling_llama import LlamaDims, LlamaDeviceWeights
from transformer.impl.llama.modeling_llama import _zeros as _llama_zeros
from transformer.impl.llama.int15_block import (
    Int15Planes,
    LLAMA_PROJ_COUNT,
    LLAMA_PROJ_GATE,
    LlamaInt15Weights,
    int15_planes_placeholder,
)


def _optional_upload(
    ctx: DeviceContext, addr: Int, n: Int, on: Bool, name: String, what: String,
) raises -> DeviceBuffer[DType.float32]:
    """One optional tensor: uploaded at its length when its flag is on,
    a one-element placeholder otherwise; a presence/flag mismatch is refused
    by the tensor's name HERE, before any device work."""
    if on:
        if addr == 0:
            raise Error(
                what + ": " + name + " is required by its option and its"
                " address is null"
            )
        return _upload_addr(ctx, addr, n)
    if addr != 0:
        raise Error(
            what + ": " + name + " was passed but its option is off; pass"
            " the option or drop the tensor"
        )
    return _llama_zeros[False](ctx, 1)


def _upload_addr[wait: Bool = True](
    ctx: DeviceContext, addr: Int, n: Int
) raises -> DeviceBuffer[DType.float32]:
    """Copy a live caller buffer, synchronizing before its borrow ends.

    IDENTICAL uses DeviceContext's ordinary host-pointer transfer directly.
    Other modes retain pinned staging.
    `wait=False` (lane/neural-apple, 2026-09-28; the direct path only) leaves
    the copy in flight: the caller's array outlives the call, every reader
    is enqueued after it on the same in-order `ctx`, and the caller waits
    before it returns.
    """
    var p = f32_ptr(addr)
    var n_buf = n
    if n_buf < 1:
        n_buf = 1
    var dev = ctx.enqueue_create_buffer[DType.float32](n_buf)
    # TOMBSTONE: MOJOLEARN_TRANSFORMER_LEGACY_CALLER_TRANSFER (DELETE-OLD-PATH: no A/B recorded) deleted 2026-11-15 by lane/expdel2-33-neural; code recoverable at c3d1c7865f.
    # Restore: git apply experiments/removed/MOJOLEARN_TRANSFORMER_LEGACY_CALLER_TRANSFER.patch
    comptime if GLOBAL_NUMERIC_MODE <= NUMERIC_IDENTICAL:
        if n > 0:
            ctx.enqueue_copy(dst_buf=dev, src_ptr=p)
            comptime if wait:
                ctx.synchronize()
            return dev^
    var host = ctx.enqueue_create_host_buffer[DType.float32](n_buf)
    ctx.synchronize()
    if n > 0:
        unsafe_memcpy(dest=host.unsafe_ptr(), src=p, count=n)
    ctx.enqueue_copy(dst_buf=dev, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return dev^


def _tail_of(a: List[Int], base: Int) -> List[Int]:
    """The eleven optional addresses out of a base + tail list (zeros when
    the list is the base alone)."""
    var t = List[Int]()
    for i in range(BLOCK_OPTION_ADDRS):
        if base + i < len(a):
            t.append(a[base + i])
        else:
            t.append(0)
    return t^


def _lean_for_w(hd: Int, w: LlamaDeviceWeights) -> Bool:
    """`_lean_for` for weights that may carry the profile's planes: never
    lean when the parts mask sets attn_scores (the eager path is forced and
    would grow every stage; lane/lowbit-default: the resident sessions under
    the profile). A mask without attn_scores keeps fp32_v1's choice (lane
    fixed15-parts-native)."""
    if w.int15 and w.int15.value().scores_on():
        return False
    return _lean_for(hd, w.opts)


def _upload_i8(ctx: DeviceContext, addr: Int, n: Int) raises -> DeviceBuffer[DType.int8]:
    var dev = ctx.enqueue_create_buffer[DType.int8](n)
    ctx.enqueue_copy(dst_buf=dev, src_ptr=i8_ptr(addr))
    return dev^


def _upload_i32(ctx: DeviceContext, addr: Int, n: Int) raises -> DeviceBuffer[DType.int32]:
    var dev = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=dev, src_ptr=i32_ptr(addr))
    return dev^


def _upload_planes_one(
    ctx: DeviceContext, a: List[Int], at: Int, rows: Int, cols: Int, name: String
) raises -> Int15Planes:
    if a[at] == 0 or a[at + 1] == 0 or a[at + 2] == 0:
        raise Error("transformer_session_forward_int15: " + name + " planes have a null address")
    return Int15Planes(
        _upload_i8(ctx, a[at], rows * cols), _upload_i8(ctx, a[at + 1], rows * cols),
        _upload_i32(ctx, a[at + 2], rows), rows, cols,
    )


def _int15_entry_parts(
    a: List[Int], pbase: Int, parts_word: Int, gated: Bool, what: String
) raises -> Int:
    """The parts mask of a fixed15 block entry, checked against its address
    list projection by projection (the layout above): set bit = nonzero
    plane triple and zero float32 slot, clear bit = zero triple and nonzero
    float32 slot; an ungated gate sends neither. Refused by the projection's
    name. Returns the block's bits."""
    var parts = np_entry_parts(parts_word, what)
    for i in range(LLAMA_PROJ_COUNT):  # small-loop(7: the projection slots): address checks, no data
        var name = np_part_name(i)
        var slot = a[3 + i]
        var t = pbase + 3 * i
        var any_plane = a[t] != 0 or a[t + 1] != 0 or a[t + 2] != 0
        if i == LLAMA_PROJ_GATE and not gated:
            if slot != 0 or any_plane:
                raise Error(what + ": gate_proj weights or planes were passed but the MLP is ungated")
            continue
        if np_part_on(parts, i):
            if slot != 0:
                raise Error(what + ": " + name + " computes under fixed15_v1 (its planes are the weight); "
                            + "its float32 slot " + String(3 + i) + " must be 0")
            if a[t] == 0 or a[t + 1] == 0 or a[t + 2] == 0:
                raise Error(what + ": " + name + " computes under fixed15_v1 and its planes have a null address")
        else:
            if any_plane:
                raise Error(what + ": " + name + " computes under fp32_v1 and was handed fixed15 planes")
            if slot == 0:
                raise Error(what + ": " + name + " computes under fp32_v1 and its float32 weight has a null address")
    return parts


def _upload_planes(
    ctx: DeviceContext, dims: LlamaDims, opts: BlockOptions, a: List[Int], base: Int, parts: Int
) raises -> LlamaInt15Weights:
    """The planes of each projection the parts mask sets (one-row
    placeholders for the others, never read). `_int15_entry_parts` has
    checked the addresses."""
    var dm = dims.d_model
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var it = dims.intermediate
    var gated = opts.gated()
    var rows: List[Int] = [qw, kw, kw, dm, it, it, dm]
    var cols: List[Int] = [dm, dm, dm, qw, dm, dm, it]
    var got = List[Int15Planes](capacity=LLAMA_PROJ_COUNT)
    for i in range(LLAMA_PROJ_COUNT):  # small-loop(7: the projections): one upload each, no data loop
        if np_part_on(parts, i) and (i != LLAMA_PROJ_GATE or gated):
            got.append(_upload_planes_one(ctx, a, base + 3 * i, rows[i], cols[i], np_part_name(i)))
        else:
            got.append(int15_planes_placeholder(ctx))
    var out = LlamaInt15Weights(
        got[0].copy(), got[1].copy(), got[2].copy(), got[3].copy(),
        got[4].copy(), got[5].copy(), got[6].copy(), gated, parts,
    )
    ctx.synchronize()
    out.check_shapes(dm, qw, kw, it)
    _ = got^
    return out^


def _proj_f32(
    ctx: DeviceContext, a: List[Int], w15: LlamaInt15Weights, which: Int, n: Int
) raises -> DeviceBuffer[DType.float32]:
    """The float32 weight of a projection the mask leaves on fp32, uploaded
    from its slot; a one-element placeholder for a projection with planes
    (the profile constructor keeps a zero placeholder for it)."""
    if w15.on(which) or (which == LLAMA_PROJ_GATE and not w15.gated):
        return ctx.enqueue_create_buffer[DType.float32](1)
    return _upload_addr(ctx, a[3 + which], n)


def _profile_weights(
    ctx: DeviceContext, dims: LlamaDims, opts: BlockOptions, a: List[Int], var planes: LlamaInt15Weights,
    what: String,
) raises -> LlamaDeviceWeights:
    """The profile constructor over an entry's addresses: norms, the
    projections per part (planes, or the float32 weight of an fp32 part),
    the options tail. Shared by the block forward and the decode open."""
    var dm = dims.d_model
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var it = dims.intermediate
    var hd = dims.head_dim
    var t = _tail_of(a, 13)
    var f_q = _proj_f32(ctx, a, planes, 0, qw * dm)
    var f_k = _proj_f32(ctx, a, planes, 1, kw * dm)
    var f_v = _proj_f32(ctx, a, planes, 2, kw * dm)
    var f_o = _proj_f32(ctx, a, planes, 3, dm * qw)
    var f_gate = _proj_f32(ctx, a, planes, 4, it * dm)
    var f_up = _proj_f32(ctx, a, planes, 5, it * dm)
    var f_down = _proj_f32(ctx, a, planes, 6, dm * it)
    return LlamaDeviceWeights(
        ctx, dims, opts,
        _upload_addr(ctx, a[1], dm),
        _upload_addr(ctx, a[2], dm),
        planes^,
        f_q^, f_k^, f_v^, f_o^, f_gate^, f_up^, f_down^,
        _optional_upload(ctx, t[0], qw, opts.qkv_bias, "q_proj.bias", what),
        _optional_upload(ctx, t[1], kw, opts.qkv_bias, "k_proj.bias", what),
        _optional_upload(ctx, t[2], kw, opts.qkv_bias, "v_proj.bias", what),
        _optional_upload(ctx, t[3], dm, opts.o_bias, "o_proj.bias", what),
        _optional_upload(ctx, t[4], dm, opts.norm_bias, "input_layernorm.bias", what),
        _optional_upload(ctx, t[5], dm, opts.norm_bias, "post_attention_layernorm.bias", what),
        _optional_upload(ctx, t[6], it, opts.mlp_bias, "up_proj.bias", what),
        _optional_upload(ctx, t[7], dm, opts.mlp_bias, "down_proj.bias", what),
        _optional_upload(ctx, t[8], it, opts.has_gate_bias(), "gate_proj.bias", what),
        _optional_upload(ctx, t[9], hd, opts.qk_norm, "q_norm.weight", what),
        _optional_upload(ctx, t[10], hd, opts.qk_norm, "k_norm.weight", what),
    )
