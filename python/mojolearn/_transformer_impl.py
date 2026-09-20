# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The transformer (Llama-shaped decoder) block on the GPU, for a Python
caller.

PRIVATE MODULE. `TransformerBlock` and `TransformerState` are re-exported
from `mojolearn/__init__.py` and from `mojolearn.transformer`.

WRITTEN 2026-09-02, giving profile
`mojolearn.identical.transformer.fp32.v1` its first Python symbol --
until this file, the certified block (forward clauses (a) and (d)
recorded on THREE columns 2026-08-28, the same 30-record card bytes on
Apple, NVIDIA and AMD; `transformer/README.md`'s status block is the
authority and its OWED list is real) exported nothing an outside
consumer could import. The contract
(`transformer/IDENTICAL_TRANSFORMER_CONTRACT.md`) is the scope
authority: its section 11 NOT-CLAIMED list is not here, each item for
the reason written there.

THE BINDING THIS FILE CALLS IS `_mojolearn_transformer` (the
FIFTEENTH), and its ABI is the two-list fold (DEVIATION 791, adopted):
each entry point takes `addrs` (every buffer address, order written in
the binding docstring and at the call site below in the same words) and
`params` (the scalars). EVERY ARRAY WHOSE ADDRESS GOES INTO `addrs` IS
BOUND TO A LOCAL for the duration of the call: an address inside a list
keeps nothing alive (`_buffer.py`).

WHAT A BLOCK IS HERE. ONE Llama-shaped decoder layer -- input norm,
eager self-attention with RoPE and a KV cache, residual, post-attention
norm, SiLU-gated MLP, residual (contract section 2) -- taking
`(B, L, d_model)` float32 in and handing `(B, L, d_model)` float32
back, with the recurrent state EXPLICIT and CALLER-OWNED (DEVIATION
792's rule; DEVIATION 795(ii) for this lane's twist): a
`TransformerState` is the KV cache -- two flat float32 capacity buffers
plus `cached_tokens` -- that a consumer can serialize, inspect and
round-trip byte for byte. Prefill, chunked-prefill continuation and
single-token decode are ALL the same certified entry point on the Mojo
side (`llama_decoder_layer_forward` at `pos0 = cached_tokens`; contract
section 7.2's ONE SPELLING); `step` exists as a name because a decode
step is what a reader expects, and it forwards to the same spelling at
L = 1.

THE CACHE IS PACKED AT THE USED STRIDE, AND THE STATE'S SHAPE SAYS SO
(DEVIATION 795(ii)). `LlamaKVCache` packs `[B, n_kv, S, head_dim]` at
stride S = cached_tokens, NOT at the capacity (the stage card records
the packed used region; modeling_llama.mojo DEVIATION 1022). So
`TransformerState.k_cache` / `.v_cache` are deliberately FLAT 1-D
buffers of `B * n_kv_heads * max_tokens * head_dim` floats -- a 4-D
capacity shape would place the used elements at the WRONG indices --
and `state.keys()` / `state.values()` hand back the packed used region
as `(B, n_kv_heads, cached_tokens, head_dim)` views.

FLOAT32 ONLY, AND LOUDLY (DEVIATION 793's split, applied). This surface
REFUSES any non-float32 dtype BY NAME -- bfloat16 and float16 because
the reference's reduced-precision runs are a MIXTURE of cast boundaries
that is not this profile (contract section 11: not BF16, FP16, FP8,
TF32), and float64 because a silent downcast would make the bits that
ran bits you did not make. Non-finite VALUES are not judged here -- they
are refused BY NAME on the device path (`llama_refuse_bad_call` per
call for x/rope/cache; the weights once at upload, DEVIATION 1875) --
and a Python-side copy would make those refusals unreachable.

THE ARRAYS ARE BUFFERS, NOT NUMPY (numpy-free-0.7, DEVIATIONS 2411-2414).
`x`, every weight and both cache buffers are read through the buffer
protocol (`_buffer.view`): a NumPy array, an `array.array('f')`, a
`mojolearn.Array` or any other float32 exporter is accepted and NumPy is
not imported. What this module ALLOCATES -- `allocate_state`'s two cache
buffers and the block output -- is a `mojolearn.Array`, on which
`numpy.asarray` is zero-copy. The cache is updated IN PLACE through its
own address, exactly as before, whatever object the caller allocated it
as. ONE VISIBLE CHANGE: `TransformerState.keys()` / `.values()` hand
back a COPY of the packed used region rather than a view into the cache
(DEVIATION 2413), because the `Array` contract has no strided views.

THIS LANE IS IDENTICAL-ONLY (2026-09-10). `numeric_mode=` accepts
'identical' or nothing; anything else raises. It used to build all three
tiers, and the lane's own record is the argument against that: clause (d)
FAILED under FAST by construction and this was documented as correct
(transformer/README.md), while `attention_path_choice` returned the EAGER
kernels for any build below identical -- so the tier sold as "fast" was
the unfused, slower one. What remains is identical (also the same bits across vendors where the lane's
record says so: the FORWARD card is byte-identical on Apple, NVIDIA and
AMD as of 2026-08-28 -- clauses (b), (c), (e) and the sabotage ladder
remain OWED there and this class claims nothing wider) -- and
`_extension()` cross-checks the binary's own compile-time answer
against the tier the package resolved, `_gp_impl.py`'s pattern, so a
wrong-arm measurement cannot be correctly labelled by accident.

SLIDING WINDOW AND BACKWARD (2026-09-09). `TransformerBlock(window=W)`
runs sliding-window causal attention with the KV cache as a RING of
W slots (`TransformerState` says how the buffers are laid out);
`window=0` is the full causal block above, bit for bit.
`TransformerBlock.backward(x, grad_output)` is the zero-state prefill
VJP under the IDENTICAL tier, from the lane's own backward chains.

ON A CPU-ONLY INSTALL (2026-09-15). `_backend._HOST_MODULES` routes
`_mojolearn_transformer` to `_mojolearn_transformer_host` when that host
binding is built (`bindings/build_transformer_host.sh`), which exports
the same four entries under the same address and params contract over
the lane's host oracles (`transformer/host/transformer_block_host.mojo`),
so this class runs unchanged there. The `transformer` and
`transformer-window` lanes of `tools/identity_break.py` are covered CPU
training lanes (`python/mojolearn/host_surface.py`): the CPU identity
gate diffs their forward, prefill, step, backward, held-out and batch
cells against the Apple, NVIDIA and AMD columns.

RUN LEDGER. THIS PATH RAN 2026-09-02, the day the binding first
compiled (rc 0 on the first attempt, 15 AIR blobs, transformer 7 and
gemm 8). `python/mojolearn/tests/test_transformer_surface.py` printed
green in ALL THREE TIERS, 44 checks 0 failed each, with the binding
rebuilt for every tier -- fast (bitwise rows REPORTED), deterministic
(the repeat-call row ASSERTED), identical (decode, resumption and
determinism all bitwise ASSERTED). ONE BOX, ONE VENDOR -- an Apple M4;
no NVIDIA or AMD box has built or run this path and both columns are
OWED. Still owed with it: the corpus cross-check, because
`transformer/corpus/` holds a generator and no case data, so the gate's
float64 reference is its own transcription of the contract rather than
an independent artifact. The build is `bash bindings/build_transformer.sh`
per tier, REBUILT before any run is believed.
"""

from . import _portable_math as math
import os
import struct
import threading

from . import _buffer as _buffers, _bufcheck as _checks
from ._array import Array as _Array
from ._arrays import _addr, _addr_ro
from . import _backend
from . import lowbit as _lowbit
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, empty, zeros
from ._bufcheck import dtype_name, is_native_f32, memcopy, probe
from ._mode import NumericModeMixin
from . import _ragged

#: `checks/numerics.mojo` codes, duplicated from `_backend._MODE_CODE` on
#: purpose, for `_arima_impl.py`'s reason: the read-back must not share a
#: table with the thing it checks.
_MODE_CODE = {"fast": 0, "identical": 1, "deterministic": 2}

#: The absolute-position ceiling, mirrored from the contract (section 3 /
#: modeling_llama.mojo's MAX_ABS_POSITION; DEVIATION 812: the Cody-Waite
#: domain of _cephes_sincosf_core). NOT a parameter -- documented here,
#: REFUSED BY NAME in Mojo (LlamaKVCache), where the authority stays.
_MAX_ABS_POSITION = 8192

# lane/block-options (2026-09-17): the block options record, mirrored from
# `transformer/block_options.mojo` (the order's authority) and from the
# bindings' tail sections, word for word. A default block sends NEITHER
# tail and reaches the old lists and the old code path; a non-default block
# appends both.
#
#   params tail, 17 ints (floats as IEEE-754 float32 bit patterns):
#     +0  rope_theta_bits  +1 rope_scaling (0 none, 1 linear, 2 llama3)
#     +2  rope_factor_bits +3 rope_low_freq_factor_bits
#     +4  rope_high_freq_factor_bits  +5 rope_original_max_positions
#     +6  rope_dim (0 = head_dim)     +7 max_positions
#     +8  qkv_bias         +9 o_bias
#     +10 norm_kind (0 rmsnorm, 1 layernorm, 2 rmsnorm_offset)
#     +11 norm_eps_bits    +12 norm_bias
#     +13 mlp_kind (0 swiglu, 1 gelu, 2 gelu_tanh, 3 geglu, 4 geglu_tanh)
#     +14 mlp_bias         +15 qk_norm      +16 attn_softcap_bits (0 = none)
#   addrs tail, 11 addresses (0 = absent), the names in `_OPT_NAMES` order.
_OPT_NAMES = (
    "q_proj.bias", "k_proj.bias", "v_proj.bias", "o_proj.bias",
    "input_layernorm.bias", "post_attention_layernorm.bias",
    "up_proj.bias", "down_proj.bias", "gate_proj.bias",
    "q_norm.weight", "k_norm.weight",
)
_NORM_KINDS = {"rmsnorm": 0, "layernorm": 1, "rmsnorm_offset": 2}
_MLP_KINDS = {"swiglu": 0, "gelu": 1, "gelu_tanh": 2, "geglu": 3, "geglu_tanh": 4}
_GATED_MLPS = ("swiglu", "geglu", "geglu_tanh")
_ROPE_SCALINGS = {"linear": 1, "llama3": 2}
_ROPE_THETA_BITS = 0x461C4000  # 10000.0, contract section 3
_NORM_EPS_BITS = 0x358637BD  # 1e-6, LlamaConfig.rms_norm_eps: TODAY's eps
_DEFAULT_TAIL_LEN = 17


def _f32_bits(value, what, name, positive=True):
    """`value` rounded to float32 and returned as its bit pattern (the
    params spelling), refusing a non-finite or, when `positive`, a
    non-positive value BY NAME with the value in the message."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise TypeError(
            f"mojolearn {what}: {name} must be a float, got {value!r}"
        ) from None
    if not math.isfinite(f) or (positive and f <= 0.0):
        raise ValueError(
            f"mojolearn {what}: {name} must be a finite"
            + (" positive" if positive else "")
            + f" float, got {value!r}"
        )
    return struct.unpack("<I", struct.pack("<f", f))[0]


def _block_options(what, head_dim, rope_theta, rope_scaling, rope_dim,
                   max_positions, qkv_bias, o_bias, norm, norm_eps,
                   norm_bias, mlp, mlp_bias, qk_norm, attn_softcap):
    """The constructor's option keywords as (tail, flags): the 17-int
    params tail in the order above, and the flags the weight-dict check
    reads. Every unsupported value or combination is refused here BY
    NAME with the exact value; Mojo repeats the same refusals
    (`BlockOptions.validate`) for raw-binding callers."""
    tail = [0] * _DEFAULT_TAIL_LEN
    tail[0] = _f32_bits(rope_theta, what, "rope_theta")
    if rope_scaling is None:
        tail[1] = 0
        tail[2] = _f32_bits(1.0, what, "rope_factor")
        tail[3] = _f32_bits(1.0, what, "low_freq_factor")
        tail[4] = _f32_bits(4.0, what, "high_freq_factor")
        tail[5] = _MAX_ABS_POSITION
    else:
        if not hasattr(rope_scaling, "get"):
            raise TypeError(
                f"mojolearn {what}: rope_scaling must be None or a dict with "
                f"a 'type' key ('linear' or 'llama3'), got {rope_scaling!r}"
            )
        kind = rope_scaling.get("type", rope_scaling.get("rope_type"))
        if kind not in _ROPE_SCALINGS:
            raise ValueError(
                f"mojolearn {what}: rope_scaling type {kind!r} is not "
                "supported; 'linear' and 'llama3' are (dynamic, yarn and "
                "longrope carry an attention_factor this profile does not "
                "spell)"
            )
        tail[1] = _ROPE_SCALINGS[kind]
        if "factor" not in rope_scaling:
            raise ValueError(
                f"mojolearn {what}: rope_scaling {kind!r} needs 'factor'"
            )
        tail[2] = _f32_bits(rope_scaling["factor"], what, "rope_scaling factor")
        if kind == "llama3":
            for key in ("low_freq_factor", "high_freq_factor",
                        "original_max_position_embeddings"):
                if key not in rope_scaling:
                    raise ValueError(
                        f"mojolearn {what}: rope_scaling 'llama3' needs "
                        f"{key!r}"
                    )
            lo = float(rope_scaling["low_freq_factor"])
            hi = float(rope_scaling["high_freq_factor"])
            tail[3] = _f32_bits(lo, what, "rope_scaling low_freq_factor")
            tail[4] = _f32_bits(hi, what, "rope_scaling high_freq_factor")
            if hi <= lo:
                raise ValueError(
                    f"mojolearn {what}: rope_scaling 'llama3' needs "
                    f"high_freq_factor > low_freq_factor, got {hi!r} <= {lo!r}"
                )
            old = int(rope_scaling["original_max_position_embeddings"])
            if old <= 0:
                raise ValueError(
                    f"mojolearn {what}: rope_scaling 'llama3' "
                    "original_max_position_embeddings must be positive, "
                    f"got {old!r}"
                )
            tail[5] = old
        else:
            tail[3] = _f32_bits(1.0, what, "low_freq_factor")
            tail[4] = _f32_bits(4.0, what, "high_freq_factor")
            tail[5] = _MAX_ABS_POSITION
    if rope_dim is None:
        tail[6] = 0
    else:
        rd = int(rope_dim)
        if rd <= 0 or rd > head_dim or rd % 2 != 0:
            raise ValueError(
                f"mojolearn {what}: rope_dim must be an even integer in "
                f"(0, head_dim={head_dim}], got {rope_dim!r}"
            )
        tail[6] = 0 if rd == head_dim else rd
    mp = int(max_positions)
    if mp <= 0:
        raise ValueError(
            f"mojolearn {what}: max_positions must be positive, got "
            f"{max_positions!r}"
        )
    tail[7] = mp
    tail[8] = 1 if qkv_bias else 0
    tail[9] = 1 if o_bias else 0
    if norm not in _NORM_KINDS:
        raise ValueError(
            f"mojolearn {what}: norm={norm!r} is not supported; one of "
            f"{sorted(_NORM_KINDS)}"
        )
    tail[10] = _NORM_KINDS[norm]
    tail[11] = _f32_bits(norm_eps, what, "norm_eps")
    if norm_bias and norm != "layernorm":
        raise ValueError(
            f"mojolearn {what}: norm_bias=True needs norm='layernorm', got "
            f"norm={norm!r} (an RMSNorm carries no bias)"
        )
    tail[12] = 1 if norm_bias else 0
    if mlp not in _MLP_KINDS:
        raise ValueError(
            f"mojolearn {what}: mlp={mlp!r} is not supported; one of "
            f"{sorted(_MLP_KINDS)}"
        )
    tail[13] = _MLP_KINDS[mlp]
    tail[14] = 1 if mlp_bias else 0
    if qk_norm and norm == "layernorm":
        raise ValueError(
            f"mojolearn {what}: qk_norm=True with norm='layernorm' is not "
            "supported (no reference family normalizes q and k with "
            "LayerNorm; use norm='rmsnorm' or 'rmsnorm_offset')"
        )
    tail[15] = 1 if qk_norm else 0
    if attn_softcap is None:
        tail[16] = 0
    else:
        tail[16] = _f32_bits(attn_softcap, what, "attn_softcap")
    gated = mlp in _GATED_MLPS
    flags = {
        "q_proj.bias": bool(qkv_bias), "k_proj.bias": bool(qkv_bias),
        "v_proj.bias": bool(qkv_bias), "o_proj.bias": bool(o_bias),
        "input_layernorm.bias": bool(norm_bias),
        "post_attention_layernorm.bias": bool(norm_bias),
        "up_proj.bias": bool(mlp_bias), "down_proj.bias": bool(mlp_bias),
        "gate_proj.bias": bool(mlp_bias) and gated,
        "q_norm.weight": bool(qk_norm), "k_norm.weight": bool(qk_norm),
    }
    return tail, flags, gated


def _default_tail():
    tail = [0] * _DEFAULT_TAIL_LEN
    tail[0] = _ROPE_THETA_BITS
    tail[2] = _f32_bits(1.0, "TransformerBlock", "rope_factor")
    tail[3] = _f32_bits(1.0, "TransformerBlock", "low_freq_factor")
    tail[4] = _f32_bits(4.0, "TransformerBlock", "high_freq_factor")
    tail[5] = _MAX_ABS_POSITION
    tail[7] = _MAX_ABS_POSITION
    tail[11] = _NORM_EPS_BITS
    return tail


def _f32_strict(a, what, name):
    """`a` as a C-contiguous float32 `Array`, REFUSING every other dtype
    BY NAME (module header: this surface certifies bits, so there is no
    convenience downcast). Layout may be fixed silently -- a copy to C
    order moves bytes untouched, so no output bit can move -- but the
    dtype may not. DEVIATION 2411: the dtype is the buffer's FORMAT,
    read through `_buffer.view`; `as_f32_c` is reached only by a float32
    buffer, where its one job is the layout (a zero-copy borrow when the
    buffer is already C-contiguous)."""
    try:
        pb = probe(a)
    except TypeError:
        raise TypeError(
            f"mojolearn {what}: {name} is a {type(a).__name__}, which does "
            "not support the buffer protocol; this surface is float32 ONLY "
            "and takes a float32 array (a numpy array, an array.array('f') "
            "or a mojolearn.Array). Convert yourself and pass float32."
        ) from None
    if not is_native_f32(pb.format):
        raise TypeError(
            f"mojolearn {what}: {name} has dtype {dtype_name(a, pb)}; this "
            "surface is float32 ONLY (the fp32 profile, "
            "transformer/IDENTICAL_TRANSFORMER_CONTRACT.md section 3). "
            "bfloat16 and float16 are refused BY NAME: the reference's "
            "reduced-precision runs are a mixture of cast boundaries "
            "that is not this profile (contract section 11), and a "
            "certified reduced-precision profile would ship under its "
            "own version name. float64 is refused rather than downcast "
            "so the bits that ran are bits you made. Convert yourself "
            "and pass float32."
        )
    arr, _copied = as_f32_c(a, ndim=pb.ndim, name=name)
    return arr


def _want_shape(a, what, name, shape):
    """Exact-shape check with the expected spelling in the message, so a
    transposed projection cannot cross as a plausible buffer (a weight at
    [out, in] has exactly as many elements as one at [in, out] -- the
    OP_NT trap's Python face)."""
    if a.shape == shape:
        return a
    raise ValueError(
        f"mojolearn {what}: {name} has shape {a.shape}, want {shape}"
    )


def _state_buf(a, what, name, shape):
    """A state buffer: float32, C-contiguous, WRITABLE, exactly `shape`.
    No silent fixups at all -- the state is read AND written in place
    (DEVIATION 792's rule), so a copy here would update the copy and the
    caller's next call would carry a stale state, which is a wrong
    answer with no diagnostic. DEVIATION 2412: "a buffer" is ANY object
    supporting the buffer protocol -- the `mojolearn.Array` that
    `allocate_state` hands out, a NumPy array, an `array.array('f')` --
    judged through `_buffer.view`; the object itself is returned and its
    address taken with `_buffer.addr` (writable required), so the
    caller's own memory is what the kernel updates."""
    try:
        pb = probe(a)
    except TypeError:
        raise TypeError(
            f"mojolearn {what}: state buffer {name} must be a numpy "
            "ndarray or a mojolearn Array -- any writable float32 buffer "
            f"-- (it is read and written in place), got {type(a)!r}"
        ) from None
    if not is_native_f32(pb.format):
        raise TypeError(
            f"mojolearn {what}: state buffer {name} has dtype "
            f"{dtype_name(a, pb)}, want float32 (the state is part of the "
            "fp32 profile and round-trips byte for byte)"
        )
    if pb.shape != shape:
        raise ValueError(
            f"mojolearn {what}: state buffer {name} has shape {pb.shape},"
            f" want {shape}"
        )
    if not pb.c_contiguous or pb.readonly:
        raise ValueError(
            f"mojolearn {what}: state buffer {name} must be C-contiguous"
            " and writable; it is updated IN PLACE so the caller's array"
            " always holds the post-call state (allocate_state() makes"
            " conforming buffers)"
        )
    return a


def _batch_tokens(x, what, d_model, step):
    """`x` as (B, L, d_model) float32. A step call also admits
    (B, d_model) -- one token per row, same bytes."""
    x = _f32_strict(x, what, "x")
    if step and x.ndim == 2:
        x = x.reshape((x.shape[0], 1, x.shape[1]))
    if x.ndim != 3:
        raise ValueError(
            f"mojolearn {what}: x must be (B, L, d_model)"
            + (" -- or (B, d_model) for a step --" if step else "")
            + f", got {x.ndim}-D shape {x.shape}"
        )
    if x.shape[2] != d_model:
        raise ValueError(
            f"mojolearn {what}: x has d_model {x.shape[2]}, the weights "
            f"have {d_model}"
        )
    if step and x.shape[1] != 1:
        raise ValueError(
            f"mojolearn {what}: a decode step takes exactly one token "
            f"per batch row, got L = {x.shape[1]}; use forward() for "
            "prefill"
        )
    return x


def _take(weights, what, names):
    """The weight dict's arrays, in a fixed order, refusing missing and
    unknown names so a typo cannot become a silently-untrained weight."""
    if not hasattr(weights, "keys"):
        raise TypeError(
            f"mojolearn {what}: weights must be a dict of float32 arrays "
            "(numpy arrays or mojolearn Arrays) keyed by the upstream "
            f"parameter names {names}"
        )
    missing = [n for n in names if n not in weights]
    extra = [n for n in weights if n not in names]
    if missing or extra:
        raise ValueError(
            f"mojolearn {what}: weight dict mismatch"
            + (f"; missing {missing}" if missing else "")
            + (f"; unknown {extra}" if extra else "")
            + f". The exact key set is {list(names)} -- the upstream "
            "parameter names (modeling_llama.py; the same names "
            "llama_refuse_bad_inputs uses)"
        )
    return [weights[n] for n in names]


class TransformerState:
    """The transformer block's recurrent state -- the KV cache, contract
    section 7.2 -- caller-owned (DEVIATION 792's rule, DEVIATION 795(ii)
    for the packing). With B = batch_size, KV = n_kv_heads, SMAX =
    max_tokens and HD = head_dim:

        k_cache       : (B * KV * SMAX * HD,) float32, FLAT -- the key
                        cache, PACKED at stride `cached_tokens`: the
                        first B*KV*cached_tokens*HD elements are the
                        `[B, KV, cached_tokens, HD]` cache, the rest is
                        capacity. FLAT ON PURPOSE: a 4-D capacity shape
                        would put the used elements at the wrong indices
                        (the pack stride is the USED length, not SMAX --
                        modeling_llama.mojo DEVIATION 1022)
        v_cache       : same shape and packing, the value cache
        cached_tokens : int in [0, max_tokens] -- how many absolute
                        positions the cache holds. Cache slot j IS
                        absolute position j (DEVIATION 1028); the next
                        call's tokens continue at position cached_tokens

    Zeros (and 0) before the first token. Both buffers are read at entry
    and written back whole by `forward` and `step` (`cached_tokens`
    reassigned); serialize them however you like, the bytes round-trip
    exactly. `keys()` / `values()` hand back the packed used region in
    its natural shape.

    WITH A SLIDING WINDOW (`TransformerBlock(window=W)`, W > 0) the two
    buffers are a RING of W slots per (batch, kv head), always at stride
    W: `B * KV * W * HD` floats, slot = absolute position % W, and
    `cached_tokens` keeps counting positions past W. `keys()` /
    `values()` then return the `min(cached_tokens, W)` most recent
    positions in ascending position order as a COPY."""

    def __init__(self, batch_size, n_kv_heads, head_dim, max_tokens,
                 k_cache, v_cache, cached_tokens=0, window=0):
        self.batch_size = int(batch_size)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = int(head_dim)
        self.max_tokens = int(max_tokens)
        self.window = int(window)
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.cached_tokens = int(cached_tokens)

    @property
    def capacity(self):
        """Slots per (batch, kv head) in each buffer: `max_tokens` for the
        linear cache, `window` for the ring."""
        return self.window if self.window > 0 else self.max_tokens

    def _packed(self, buf):
        b, kv, hd = self.batch_size, self.n_kv_heads, self.head_dim
        source = _checks.flat_view(buf, 'f')
        if self.window > 0:
            w = self.window
            s = self.cached_tokens
            held = min(s, w)
            out = _buffers.empty((b, kv, held, hd), '<f4')
            dest = _checks.flat_view(out, 'f')
            # Copy contiguous head vectors in chronological order, preserving
            # ring wrap and head/batch strides without advanced-index arrays.
            for head in range(b * kv):
                for j in range(held):
                    src = (head*w + (s-held+j) % w)*hd
                    dst = (head*held+j)*hd
                    dest[dst:dst+hd] = source[src:src+hd]
            return out
        n = b * kv * self.cached_tokens * hd
        return _Array.from_buffer(source[:n]).reshape((b, kv, self.cached_tokens, hd))

    def keys(self):
        """The key cache's packed used region, as a
        `(B, n_kv_heads, cached_tokens, head_dim)` VIEW into `k_cache`
        (valid until the next call updates the state); under a window,
        the held positions in order, as a copy."""
        return self._packed(self.k_cache)

    def values(self):
        """The value cache's packed used region, the `keys()` shape, a
        copy likewise."""
        return self._packed(self.v_cache)


class TransformerBlock(NumericModeMixin):
    """One Llama-shaped decoder block on the GPU -- input RMSNorm, eager
    self-attention with RoPE, GQA and a KV cache, residual,
    post-attention RMSNorm, SiLU-gated MLP, residual
    (`LlamaDecoderLayer.forward`, transformers modeling_llama.py:295-324
    at the pinned `d56c55b`) -- under profile
    `mojolearn.identical.transformer.fp32.v1`
    (`transformer/IDENTICAL_TRANSFORMER_CONTRACT.md`). The LANE's
    forward is recorded bit-identical on Apple, NVIDIA and AMD for
    clauses (a) and (d) as of 2026-08-28 (transformer/README.md's status
    block; clauses (b), (c), (e) and the sabotage ladder are OWED on the
    cross-vendor legs and this class claims nothing wider) -- and this
    Python path itself printed green in all three tiers on 2026-09-02
    (`test_transformer_surface.py`, 44 checks 0 failed each) on ONE
    APPLE M4, with its NVIDIA and AMD columns OWED.

    WEIGHTS IN, AS GIVEN BITS. The constructor takes a dict keyed by the
    reference parameter names (modeling_llama.py's; the same names the
    lane's refusals use); there is no initializer, deliberately --
    bit-reproducing torch's RNG is a refused validation target
    (the mamba parity table's standing reason), and a cross-check hands
    both sides the SAME weights. With QW = n_heads*head_dim (= d_model),
    KW = n_kv_heads*head_dim, IT = intermediate_size (read from
    gate_proj.weight):

        input_layernorm.weight           (d_model,)
        post_attention_layernorm.weight  (d_model,)
        q_proj.weight                    (QW, d_model)   torch [out, in]
        k_proj.weight                    (KW, d_model)
        v_proj.weight                    (KW, d_model)
        o_proj.weight                    (d_model, QW)
        gate_proj.weight                 (IT, d_model)
        up_proj.weight                   (IT, d_model)
        down_proj.weight                 (d_model, IT)

    THE SHAPE ARGUMENTS: `n_heads` is required; `n_kv_heads` defaults to
    `n_heads` (GQA is ADMITTED, contract DEVIATION 813 -- `repeat_kv` is
    an index map, head h reads kv head h // n_rep); `head_dim` defaults
    to `d_model // n_heads` (`LlamaConfig.__post_init__`'s rule). None
    of the three can be read off the weight shapes alone, which is why
    they are arguments and the weights are not.

    WHAT IS HONORED, WHAT IS FIXED, WHAT IS REFUSED -- the contract's
    sections 3 and 11 are normative; one line each:

        d_model/n_heads/  honored  free shapes, subject to the
          n_kv_heads/               divisibility rules (d_model ==
          head_dim/                 n_heads*head_dim; n_heads %
          intermediate              n_kv_heads == 0; head_dim EVEN --
                                    LlamaDims.validate refuses each by
                                    name; this side repeats only the
                                    first, because it cannot size the
                                    buffers without it)
        B, L, max_tokens  honored  launch shape, not arithmetic;
                                    absolute positions must stay under
                                    8192 (DEVIATION 812, refused by
                                    name in Mojo); B and L through
                                    this class are asked by the batch
                                    part of tools/identity_break.py
                                    since 4230ab5b0, first record owed
        norm_eps          honored  1e-6 (bits 0x358637BD) is today's bits
          rope_theta               and the default; rope_theta 10000.0
                                    (0x461C4000) likewise. Any other
                                    value is DEVIATIONS 2937 / 2930
                                    (lane/block-options): the same
                                    seams with the parameter in place of
                                    the frozen constant
        rope_scaling      honored  None (default), {"type": "linear",
          rope_dim                  "factor"} or {"type": "llama3",
          max_positions             "factor", "low_freq_factor",
                                    "high_freq_factor",
                                    "original_max_position_embeddings"};
                                    rope_dim (partial rotary; None =
                                    head_dim); max_positions (the
                                    declared ceiling, 8192 default; the
                                    rotary reduction's ANGLE domain of
                                    8192.0 is refused by name in Mojo
                                    whatever the ceiling)
        norm, norm_bias   honored  "rmsnorm" (default), "layernorm"
                                    (+ norm_bias, the *.bias tensors),
                                    "rmsnorm_offset" (Gemma's 1 + w)
        mlp, mlp_bias     honored  "swiglu" (default), "gelu",
                                    "gelu_tanh" (ungated, no gate_proj),
                                    "geglu", "geglu_tanh"; mlp_bias adds
                                    up_proj.bias, down_proj.bias and,
                                    when gated, gate_proj.bias
        qkv_bias, o_bias  honored  the q/k/v (Qwen2) and o biases
        qk_norm           honored  q_norm.weight / k_norm.weight over
                                    head_dim before RoPE (Qwen3)
        attn_softcap      honored  Gemma2's tanh softcap of the scores
                                    (forces the eager attention path)
        attention impl    FIXED    the EAGER path only (contract section
                                    6): FlashAttention, SDPA, paged
                                    attention and chunked prefill are
                                    out of scope BY CONTRACT, not
                                    missing
        dropout           refused  attention_dropout is 0.0 and has no
                                    inference meaning; nothing spells it
        window            honored  `window=0` (default) is full causal
                                    attention, today's bits exactly;
                                    `window=W > 0` is sliding-window
                                    causal attention (query at position
                                    p sees keys [max(0, p-W+1), p]) with
                                    the KV cache a RING of W slots; the
                                    same one spelling serves prefill,
                                    split prefill and decode bit for bit
        masks beyond      refused  contract section 11's list, by
          causal/window             absence
        backward          honored  `backward(x, grad_output)`: the
                                    zero-state prefill VJP for x and the
                                    nine weights, IDENTICAL tier only
                                    (the lane's `transformer_backward
                                    .mojo` chains, window included);
                                    DEFAULT OPTIONS ONLY, refused by
                                    name otherwise
        dtype             refused  float32 ONLY; bf16/fp16/float64 by
                                    name

    STATE IS EXPLICIT (`TransformerState`): `allocate_state(B,
    max_tokens)` makes the zero cache, `forward`/`step` update it, and
    decode == prefill is the contract's construction (section 7.2: one
    spelling serves both paths; the lane's clause (d) verifies it
    bitwise under IDENTICAL and records that FAST fails it by design).

    Non-finite inputs are refused BY NAME IN MOJO: the weights once at
    upload (DEVIATION 1875), x, the rotary table and the carried cache
    on every call (`llama_refuse_bad_call`), before any stage runs."""

    _BINDING = "_mojolearn_transformer"

    _W_NAMES = (
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "gate_proj.weight", "up_proj.weight", "down_proj.weight",
    )

    def __init__(self, weights, *, n_heads, n_kv_heads=None, head_dim=None,
                 window=0, rope_theta=10000.0, rope_scaling=None,
                 rope_dim=None, max_positions=8192, qkv_bias=False,
                 o_bias=False, norm="rmsnorm", norm_eps=1e-6,
                 norm_bias=False, mlp="swiglu", mlp_bias=False,
                 qk_norm=False, attn_softcap=None):
        self._runtime_lock = threading.RLock()
        self._native_session = None
        self._session_binding = None
        what = "TransformerBlock"
        # lane/identical-lowbit-inference (2026-09-17): packed bf16 or int8
        # projection weights (mojolearn.lowbit) are materialized exactly here
        # and the block runs its certified fp32 path on the result.
        weights, self.weight_format = _lowbit.unpack(weights, what)
        # lane/block-options (2026-09-17): the option keywords. `head_dim`
        # is needed for rope_dim's bound and the q/k norm widths, so the
        # shape is read first from the norm weight and the head arguments.
        if not hasattr(weights, "keys") or "input_layernorm.weight" not in weights:
            _take(weights, what, self._W_NAMES)  # raises with the key set
        dm0 = int(_f32_strict(weights["input_layernorm.weight"], what,
                              "input_layernorm.weight").shape[0])
        nh0 = int(n_heads)
        if nh0 < 1:
            raise ValueError(
                f"mojolearn {what}: n_heads must be positive, got "
                f"{n_heads!r}"
            )
        hd0 = int(head_dim) if head_dim is not None else dm0 // nh0
        tail, flags, gated = _block_options(
            what, hd0, rope_theta, rope_scaling, rope_dim, max_positions,
            qkv_bias, o_bias, norm, norm_eps, norm_bias, mlp, mlp_bias,
            qk_norm, attn_softcap)
        self._opts_tail = tail
        self._opts_flags = flags
        self._gated = gated
        self._extended = tail != _default_tail()
        self.rope_theta = float(rope_theta)
        self.rope_scaling = None if rope_scaling is None else dict(rope_scaling)
        self.rope_dim = hd0 if rope_dim is None else int(rope_dim)
        self.max_positions = int(max_positions)
        self.qkv_bias = bool(qkv_bias)
        self.o_bias = bool(o_bias)
        self.norm = norm
        self.norm_eps = float(norm_eps)
        self.norm_bias = bool(norm_bias)
        self.mlp = mlp
        self.mlp_bias = bool(mlp_bias)
        self.qk_norm = bool(qk_norm)
        self.attn_softcap = None if attn_softcap is None else float(attn_softcap)
        # The exact key set: the nine (minus gate_proj.weight under an
        # ungated MLP) plus every optional tensor whose flag is on. A bias
        # present with its flag off, or missing with its flag on, is
        # refused by ITS name before the generic key-set message.
        if hasattr(weights, "keys"):
            for name in _OPT_NAMES:
                present = name in weights
                if present and not flags[name]:
                    raise ValueError(
                        f"mojolearn {what}: {name} is present in the weight "
                        "dict but its option is off (pass the option that "
                        "carries it, or drop the tensor; a silently ignored "
                        "tensor is a wrong model that looks right)"
                    )
                if flags[name] and not present:
                    raise ValueError(
                        f"mojolearn {what}: {name} is required by the "
                        "options passed and is missing from the weight dict"
                    )
            if not gated and "gate_proj.weight" in weights:
                raise ValueError(
                    f"mojolearn {what}: gate_proj.weight is present but "
                    f"mlp={mlp!r} is ungated (down_proj(act(up_proj(x)))); "
                    "drop it or pick a gated form"
                )
        base_names = tuple(n for n in self._W_NAMES
                           if gated or n != "gate_proj.weight")
        opt_names = tuple(n for n in _OPT_NAMES if flags[n])
        arrs_all = _take(weights, what, base_names + opt_names)
        arrs = list(arrs_all[:len(base_names)])
        if not gated:
            arrs.insert(6, None)  # the gate slot, absent
        opt_arrs = list(arrs_all[len(base_names):])
        win = int(window)
        if win < 0:
            raise ValueError(
                f"mojolearn {what}: window must be 0 (full causal) or a "
                f"positive sliding-window width, got {window!r}"
            )
        self.window = win
        norm1_w = _f32_strict(arrs[0], what, "input_layernorm.weight")
        if norm1_w.ndim != 1 or norm1_w.shape[0] < 1:
            raise ValueError(
                f"mojolearn {what}: input_layernorm.weight must be 1-D "
                f"(d_model,), got shape {norm1_w.shape}"
            )
        dm = int(norm1_w.shape[0])
        nh = int(n_heads)
        if nh < 1:
            raise ValueError(
                f"mojolearn {what}: n_heads must be positive, got "
                f"{n_heads!r}"
            )
        hd = int(head_dim) if head_dim is not None else dm // nh
        nkv = int(n_kv_heads) if n_kv_heads is not None else nh
        if dm != nh * hd:
            # LlamaDims.validate's rule, quoted; the Mojo constructor
            # remains the authority (raw-binding callers hit it), this
            # copy exists because the buffers below cannot be sized
            # without it. The OTHER divisibility rules (n_heads %
            # n_kv_heads, head_dim even) are NOT repeated: sizing does
            # not need them, so they stay Mojo's alone, reachable.
            raise ValueError(
                f"mojolearn {what}: d_model must equal n_heads*head_dim "
                f"(LlamaDims.validate carries the same refusal), got "
                f"d_model {dm} vs n_heads {nh} * head_dim {hd}"
            )
        if nkv < 1:
            raise ValueError(
                f"mojolearn {what}: n_kv_heads must be positive, got "
                f"{n_kv_heads!r}"
            )
        # `intermediate` is read from gate_proj.weight (gated) or from
        # up_proj.weight (ungated); the other is checked against it below.
        it_src = "gate_proj.weight" if gated else "up_proj.weight"
        it_w = _f32_strict(arrs[6] if gated else arrs[7], what, it_src)
        if it_w.ndim != 2 or it_w.shape[0] < 1 or it_w.shape[1] != dm:
            raise ValueError(
                f"mojolearn {what}: {it_src} must be "
                f"(intermediate, d_model={dm}), got shape {it_w.shape}"
            )
        it = int(it_w.shape[0])
        gate_w = it_w if gated else None
        qw = nh * hd
        kw = nkv * hd
        self.d_model = dm
        self.n_heads = nh
        self.n_kv_heads = nkv
        self.head_dim = hd
        self.intermediate = it
        # Every weight to float32 C-order ONCE, at construction, shapes
        # checked against the profile's derivation rules so a transposed
        # projection cannot cross as a plausible buffer.
        self._w = [
            norm1_w,
            _want_shape(
                _f32_strict(arrs[1], what,
                            "post_attention_layernorm.weight"),
                what, "post_attention_layernorm.weight", (dm,)),
            _want_shape(_f32_strict(arrs[2], what, "q_proj.weight"),
                        what, "q_proj.weight", (qw, dm)),
            _want_shape(_f32_strict(arrs[3], what, "k_proj.weight"),
                        what, "k_proj.weight", (kw, dm)),
            _want_shape(_f32_strict(arrs[4], what, "v_proj.weight"),
                        what, "v_proj.weight", (kw, dm)),
            _want_shape(_f32_strict(arrs[5], what, "o_proj.weight"),
                        what, "o_proj.weight", (dm, qw)),
            (_want_shape(gate_w, what, "gate_proj.weight", (it, dm))
             if gated else None),
            _want_shape(_f32_strict(arrs[7], what, "up_proj.weight"),
                        what, "up_proj.weight", (it, dm)),
            _want_shape(_f32_strict(arrs[8], what, "down_proj.weight"),
                        what, "down_proj.weight", (dm, it)),
        ]
        # The optional tensors, in `_OPT_NAMES` order, None where absent;
        # each present one float32 and exactly its shape.
        opt_shapes = {
            "q_proj.bias": (qw,), "k_proj.bias": (kw,), "v_proj.bias": (kw,),
            "o_proj.bias": (dm,), "input_layernorm.bias": (dm,),
            "post_attention_layernorm.bias": (dm,), "up_proj.bias": (it,),
            "down_proj.bias": (dm,), "gate_proj.bias": (it,),
            "q_norm.weight": (hd,), "k_norm.weight": (hd,),
        }
        present = iter(opt_arrs)
        self._wopt = []
        for name in _OPT_NAMES:
            if flags[name]:
                a = next(present)
                self._wopt.append(_want_shape(_f32_strict(a, what, name),
                                              what, name, opt_shapes[name]))
            else:
                self._wopt.append(None)

    def _weight_addrs(self):
        """The nine base weight addresses in the binding's order; the gate
        slot is 0 under an ungated MLP. The arrays are alive on `self`."""
        return [0 if a is None else _addr_ro(a) for a in self._w]

    def _tail_addrs(self):
        """The eleven optional addresses in `_OPT_NAMES` order, 0 where
        absent -- the addrs tail, sent only when `_extended`."""
        return [0 if a is None else _addr_ro(a) for a in self._wopt]

    def _with_tails(self, addrs, params):
        """The two lists as the binding wants them: the old lists for a
        default block (the old code path exactly), the tails appended
        otherwise."""
        if not self._extended:
            return addrs, params
        return addrs + self._tail_addrs(), params + list(self._opts_tail)

    def __getstate__(self):
        # Device contexts and thread locks cannot cross serialization/copy.
        # Only ordinary model state is saved; GPU ownership is recreated lazily.
        state = self.__dict__.copy()
        for name in ("_runtime_lock", "_native_session", "_session_binding"):
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._runtime_lock = threading.RLock()
        self._native_session = None
        self._session_binding = None

    def allocate_state(self, batch_size, max_tokens):
        """The zero KV cache for `batch_size` sequences of up to
        `max_tokens` total positions each (prefill plus every later
        decode step; the capacity is fixed at allocation because the
        pack stride and every growth refusal are functions of it,
        DEVIATION 795(ii)). `max_tokens` above 8192 is refused BY NAME
        in Mojo at the first call (DEVIATION 812's absolute-position
        ceiling), not here. Under a sliding window the buffers are a
        ring of `window` slots and `max_tokens` bounds positions only."""
        b = int(batch_size)
        if b < 1:
            raise ValueError(
                f"mojolearn TransformerBlock: batch_size must be "
                f"positive, got {batch_size!r}"
            )
        smax = int(max_tokens)
        if smax < 1:
            raise ValueError(
                f"mojolearn TransformerBlock: max_tokens must be "
                f"positive, got {max_tokens!r}"
            )
        cap = self.window if self.window > 0 else smax
        n = b * self.n_kv_heads * cap * self.head_dim
        return TransformerState(
            b, self.n_kv_heads, self.head_dim, smax,
            _buffers.zeros((n,), '<f4'),
            _buffers.zeros((n,), '<f4'),
            0, self.window,
        )

    def _call_fresh(self, x, ext):
        """Stateless prefill with the original zero cache owned only on device."""
        b, l = int(x.shape[0]), int(x.shape[1])
        y = _buffers.empty((b, l, self.d_model), '<f4')
        # All pointer owners remain live until the synchronous native return.
        w = self._w
        wopt = self._wopt  # noqa: F841  (keeps the optional arrays alive)
        addrs, params = self._with_tails(
            [_addr_ro(x)] + self._weight_addrs() + [_addr(y)],
            [b, l, self.d_model, self.n_heads, self.n_kv_heads,
             self.head_dim, self.intermediate, self.window])
        ext.transformer_forward_fresh(addrs, params)
        return y

    def _call(self, x, state, step):
        what = "TransformerBlock.step" if step else "TransformerBlock.forward"
        x = _batch_tokens(x, what, self.d_model, step)
        ext = self._extension()
        reuse = (_exports(ext, "transformer_session_forward")
                 and os.environ.get("MOJOLEARN_TRANSFORMER_LEGACY_SETUP") != "1")
        if not reuse and self._native_session is None:
            # CPU and older extensions keep the existing path, without a lock.
            return self._call_impl(x, state, step, ext, False)
        with self._runtime_lock:
            if self._native_session is not None and (
                    not reuse or self._session_binding is not ext):
                self._session_binding.transformer_session_close(self._native_session)
                self._native_session = None
                self._session_binding = None
            return self._call_impl(x, state, step, ext, reuse)

    def _call_impl(self, x, state, step, ext, reuse):
        what = ("TransformerBlock.step" if step
                else "TransformerBlock.forward")
        b, l = int(x.shape[0]), int(x.shape[1])
        fresh_ext = None
        mode = getattr(self, "numeric_mode", None) or _backend.default_mode()
        if (state is None and not step and b > 0 and l > 0
                and self.window >= 0 and mode == "identical"):
            fresh_ext = ext
            if hasattr(fresh_ext, "transformer_forward_fresh"):
                return self._call_fresh(x, fresh_ext)
        if state is None:
            # A self-contained prefill: the cache exists for exactly this
            # call and is discarded, so its capacity is the call's length.
            state = self.allocate_state(b, l)
        _refuse_resident(state, what)
        if state.batch_size != b:
            raise ValueError(
                f"mojolearn {what}: the state was allocated for "
                f"batch_size {state.batch_size} but x has B = {b} "
                "(allocate_state(B, max_tokens) makes a matching one)"
            )
        if int(getattr(state, "window", 0)) != self.window:
            raise ValueError(
                f"mojolearn {what}: the state was allocated for window "
                f"{getattr(state, 'window', 0)} but this block has window "
                f"{self.window} (allocate_state on THIS block makes a "
                "matching one)"
            )
        smax = int(state.max_tokens)
        n = b * self.n_kv_heads * state.capacity * self.head_dim
        kc = _state_buf(state.k_cache, what, "k_cache", (n,))
        vc = _state_buf(state.v_cache, what, "v_cache", (n,))
        s0 = int(state.cached_tokens)
        # cached_tokens outside [0, max_tokens] is a boundary
        # disagreement the binding refuses by name; growth past the
        # capacity and the 8192 ceiling are refused by name in Mojo
        # (DEVIATION 795(iv)); all three go down unjudged.
        y = empty((b, l, self.d_model), "<f4")
        # TWO LISTS, NOT THIRTEEN ARGUMENTS (DEVIATION 791). Every array
        # addressed below is bound in this frame -- x, the nine entries
        # of self._w (alive on self), kc, vc, y -- which is what keeps
        # the addresses alive (_buffer.py). `addr` (writable) for the
        # cache and the output, `addr_ro` for x and the weights.
        w = self._w
        wopt = self._wopt  # noqa: F841  (keeps the optional arrays alive)
        ext = fresh_ext if fresh_ext is not None else ext
        addrs = (
            # ORDER MATCHES bindings/_mojolearn_transformer.mojo::
            # transformer_forward_binding: x, input_layernorm.weight,
            # post_attention_layernorm.weight, q_proj.weight,
            # k_proj.weight, v_proj.weight, o_proj.weight,
            # gate_proj.weight (0 under an ungated MLP), up_proj.weight,
            # down_proj.weight, k_cache, v_cache, y_out -- then, for a
            # non-default block only, the 11-address options tail
            # (`_OPT_NAMES` order) and the 17-int params tail.
            [addr_ro(x, name="x")]
            + [0 if a is None else addr_ro(a, name="weight") for a in w]
            + [addr(kc, name="k_cache"), addr(vc, name="v_cache"),
               addr(y, name="y")]
        )
        if reuse:
            if self._native_session is None:
                self._native_session = ext.transformer_session_create()
                self._session_binding = ext
            a2, p2 = self._with_tails(
                addrs,
                [b, l, self.d_model, self.n_heads, self.n_kv_heads,
                 self.head_dim, self.intermediate, smax, s0, self.window])
            new_len = ext.transformer_session_forward(
                self._native_session, a2, p2)
        elif step:
            # B, d_model, n_heads, n_kv_heads, head_dim, intermediate,
            # max_tokens, cached_tokens, window.
            a2, p2 = self._with_tails(
                addrs,
                [b, self.d_model, self.n_heads, self.n_kv_heads,
                 self.head_dim, self.intermediate, smax, s0, self.window])
            new_len = ext.transformer_decode_step(a2, p2)
        else:
            # B, L, then the same seven.
            a2, p2 = self._with_tails(
                addrs,
                [b, l, self.d_model, self.n_heads, self.n_kv_heads,
                 self.head_dim, self.intermediate, smax, s0, self.window])
            new_len = ext.transformer_forward(a2, p2)
        state.cached_tokens = int(new_len)
        return y

    def _extension(self):
        """The `_mojolearn_transformer` binding for THIS block's tier,
        with the binary's own compile-time answer cross-checked against
        it -- `_gp_impl.py`'s pattern, for its reason: a wrong-arm
        measurement that is correctly labelled by accident is the
        failure the three-tier design exists to prevent."""
        mod = self._bind()
        want = (getattr(self, "numeric_mode", None)
                or _backend.default_mode())
        fn = getattr(mod, "transformer_numeric_mode", None)
        if fn is not None:
            got = int(fn())
            if got != _MODE_CODE.get(want):
                raise RuntimeError(
                    f"mojolearn {type(self).__name__}: numeric_mode="
                    f"{want!r} was requested but {mod.__name__} reports "
                    f"compile-time mode code {got}; the binary and the "
                    "directory it sits in disagree, rebuild it with "
                    "bash bindings/build_transformer.sh"
                )
        return mod

    def forward(self, x, state=None, *, lengths=None):
        """One block call: `(B, L, d_model)` float32 in, the block
        output (both residual adds included) back, any B and L that fit
        the state's capacity.

        `lengths` (2026-09-15) makes the batch RAGGED: `B` integers in
        `[1, L]`, row `i` real at positions `[0, lengths[i])` and padding
        after. Every real position's output is byte for byte the row run
        alone at its own length (causal attention: contract 7.1, 7.3 and
        7.4, no arithmetic changes; `_ragged.py` says why) and every padding
        position's output is exactly `+0.0`, whatever the input held there.
        Refused with a carried `state`.

        `state=None` runs a self-contained prefill from a zero cache
        sized to exactly this call and DISCARDS the final state. Pass a
        `TransformerState` to carry it: the cache is read at entry and
        updated, so a later `forward` or `step` on that state continues
        the sequence -- bit-for-bit the prefill that ran the whole
        sequence at once under the IDENTICAL tier (contract section
        7.2's construction; the lane's clause (d) verifies it, and FAST
        deliberately promises none of it)."""
        if lengths is None:
            return self._call(x, state, step=False)
        what = "TransformerBlock.forward"
        x = _batch_tokens(x, what, self.d_model, False)
        return _ragged.ragged_forward(lambda xp: self._call(xp, None, step=False),
                                      x, state, lengths, "<f4", what)[0]

    def step(self, x, state):
        """One decode token: the profile's spelling -- the SAME entry as
        `forward` at L = 1 with the cache carried and the position taken
        from `cached_tokens`, no arithmetic of its own (contract section
        7.2: two spellings that agree today are two spellings that can
        drift tomorrow). `x` is `(B, 1, d_model)` or `(B, d_model)`;
        `state` is REQUIRED, because a stateless decode step has no
        meaning."""
        if state is None:
            raise ValueError(
                "mojolearn TransformerBlock.step: state is required (a "
                "decode step continues a sequence; allocate_state(B, "
                "max_tokens) makes the fresh one)"
            )
        return self._call(x, state, step=True)

    def backward(self, x, grad_output):
        """The zero-state prefill VJP: `x` `(B, L, d_model)` float32 and
        `grad_output` of the same shape (d loss / d block output) in; a
        dict of float32 gradients out, keyed `"x"` plus the nine weight
        names, each in its weight's shape. The forward is recomputed
        from a zero cache at positions `[0, L)` under this block's
        `window`, then the lane's backward (`transformer/checks/
        transformer_backward.mojo`) runs on the saved stages: the
        activation gradients are its pinned serial chains, every weight
        gradient is a sum over this call's `B*L` tokens. IDENTICAL tier
        only; no incoming-cache or carried-state cotangent."""
        what = "TransformerBlock.backward"
        if getattr(self, "_extended", False):
            # lane/block-options: the backward chains spell the frozen
            # profile's seams and none of the options'.
            raise NotImplementedError(
                f"mojolearn {what}: the backward is implemented for the "
                "default block options only (rope_theta 10000.0, no "
                "rope_scaling, full rotary, max_positions 8192, no bias, "
                "norm='rmsnorm' at eps 1e-6, mlp='swiglu', no qk_norm, no "
                f"attn_softcap); this block has norm={self.norm!r}, "
                f"mlp={self.mlp!r}, rope_theta={self.rope_theta!r}, "
                f"rope_scaling={self.rope_scaling!r}, rope_dim={self.rope_dim!r}, "
                f"max_positions={self.max_positions!r}, qkv_bias={self.qkv_bias!r}, "
                f"o_bias={self.o_bias!r}, norm_eps={self.norm_eps!r}, "
                f"norm_bias={self.norm_bias!r}, mlp_bias={self.mlp_bias!r}, "
                f"qk_norm={self.qk_norm!r}, attn_softcap={self.attn_softcap!r}"
            )
        mode = getattr(self, "numeric_mode", None) or _backend.default_mode()
        if mode != "identical":
            raise NotImplementedError(
                f"mojolearn {what}: only the IDENTICAL zero-state prefill "
                f"backward is implemented; got numeric_mode={mode!r}"
            )
        x = _batch_tokens(x, what, self.d_model, False)
        b, l = int(x.shape[0]), int(x.shape[1])
        dy = _want_shape(_f32_strict(grad_output, what, "grad_output"),
                         what, "grad_output", x.shape)
        ext = self._extension()
        native = getattr(ext, "transformer_backward", None)
        if native is None:
            raise RuntimeError(
                f"mojolearn {what}: the loaded transformer extension lacks "
                "transformer_backward; rebuild bindings/build_transformer.sh"
                " in IDENTICAL mode"
            )
        w = self._w
        grads = [_buffers.empty(x.shape, "<f4")] + [_buffers.empty(a.shape, "<f4") for a in w]
        # ORDER MATCHES bindings/_mojolearn_transformer.mojo::
        # transformer_backward_binding: x, the nine weights, grad_output,
        # grad_x, the nine weight gradients.
        addrs = ([_addr_ro(x)] + [_addr_ro(a) for a in w] + [_addr_ro(dy)]
                 + [_addr(g) for g in grads])
        native(addrs, [b, l, self.d_model, self.n_heads, self.n_kv_heads,
                       self.head_dim, self.intermediate, self.window])
        return dict(zip(("x",) + self._W_NAMES, grads))

    def decode_session(self, state):
        """A `TransformerDecodeSession` on this block and `state`: the
        weights, the KV cache, the rotary table and the L = 1 stages
        RESIDENT on the device across decode steps (DEVIATION 2940,
        lane/infer-speed-neural, 2026-09-17), so a token costs one input
        upload, the block and one output download instead of the per-call
        entry's context, weight and cache round trip. `step` there is
        `TransformerBlock.step`'s bytes: the same certified entry point at
        L = 1 on the same structs, built once instead of per call.

        OWNERSHIP. The session COPIES this block's weights and the state's
        cache at open. Until `close()` (or `sync_state()`) the state's
        `k_cache`/`v_cache` are STALE and `forward`/`step` on this block
        refuse the state by name; `cached_tokens` on the state is kept
        current. Edits to the weights after open are NOT observed: close
        and open a new session. Edits to the cache arrays after open are
        not observed either: `load_state()` re-uploads them.

        THE CPU HOST ROUTE TAKES THIS DOOR TOO (lane/cpu-routes-gpu-only-four,
        2026-09-20). A host binding exports no `transformer_decode_session_*`
        entry and never will -- there is no device context to hold anything
        resident in -- but the session's ARITHMETIC is not the residency:
        `step` and `forward` are `transformer_decode_step` and
        `transformer_forward`, the certified per-call entries this block's own
        `step`/`forward` take, which `bindings/_mojolearn_transformer_host.mojo`
        exports over `transformer/host/transformer_block_host.mojo`. So on the
        host route this object holds its own copies of the weights and of the
        KV cache and calls those entries, in the same order and with the same
        addresses `TransformerBlock._call_impl` builds. It is the ownership
        wrapper, not new arithmetic, and it earns no speed claim: on the host
        every call re-reads the weights exactly as the per-call entry does.

        This is NOT the block's retained per-model context (`_native_session`,
        the binding's `transformer_session_*` entry points, docs/
        TRANSFORMER_SESSION_REUSE.md), which the per-call `forward`/`step`
        use and which rereads the weights and the caller's cache on every
        call. The decode session owns its own context and its entry points
        are `transformer_decode_session_*`; the two coexist on one block."""
        return TransformerDecodeSession(self, state)

    __call__ = forward


def _exports(ext, name):
    """Whether the loaded binding exports `name`. On a CPU-only install the
    stand-in for a GPU binding RAISES ImportError by name from `__getattr__`
    (`_backend.py::_HostBinding`), and `hasattr` swallows only
    AttributeError, so a bare `hasattr` probe took the whole CPU host route
    down (seen 2026-09-17 on the merged tree: every transformer and samba
    CPU cell REFUSED at `transformer_session_forward`). A probe is not a
    use; the repo's own guard in `_backend.py` does the same."""
    try:
        return hasattr(ext, name)
    except ImportError:
        return False


def _private_copy(a):
    """A fresh float32 C-order buffer holding `a`'s bytes, or None for None.

    The resident decode session's OWNERSHIP clause: the session COPIES the
    weights and the cache at open, so a caller's later edit to either is not
    observed. On the GPU arm the copy lands in device memory; on the host arm
    it lands here, in a buffer of this process, which is the same promise
    with the same visibility and the same refresh door (`load_state`)."""
    if a is None:
        return None
    pb = probe(a)
    out = empty(tuple(int(d) for d in pb.shape), "<f4")
    memcopy(addr(out, name="copy"), addr_ro(a, name="source"), 4 * int(out.size))
    return out


def _refuse_resident(state, what):
    owner = getattr(state, "_resident_session", None)
    if owner is not None:
        raise ValueError(
            f"mojolearn {what}: the state is owned by an open resident "
            "TransformerDecodeSession (its cache lives in the session, on the "
            "device or on the host, and the caller's buffers are stale); "
            "call sync_state() or close() on the session first"
        )


class TransformerDecodeSession:
    """Resident decode on one `TransformerBlock` and one `TransformerState`
    (DEVIATION 2940). Made by `TransformerBlock.decode_session(state)`.

    `step(x)`      one decode token per row, `(B, 1, d_model)` or
                   `(B, d_model)` float32 in, `(B, 1, d_model)` out;
                   `state.cached_tokens` advances
    `forward(x)`   `L` tokens per row on the resident cache (a prefill
                   or a chunked continuation), `(B, L, d_model)` in and out
    `sync_state()` copy the resident cache into the state's buffers; the
                   state stays owned by the session
    `load_state()` re-upload the state's buffers and `cached_tokens`
                   (the explicit refresh of the state half)
    `close()`      sync, release every device buffer and hand the state
                   back; also the context manager exit

    Every output is BYTE FOR BYTE the per-call `step`/`forward` on the
    same block and state: the binding runs the one certified entry point
    on structs that hold the same bytes. On a GPU binding the session holds
    its own device context; it is not thread-safe and refuses re-entrant
    use.

    TWO ARMS, ONE CLASS AND ONE SET OF BYTES. On a binding that exports
    `transformer_decode_session_create` the weights, the KV cache, the
    rotary table and the L = 1 stages are RESIDENT on the device across
    steps. On the host route
    (`bindings/_mojolearn_transformer_host.mojo`, which exports no session
    entry) the session owns its copies HERE and each call is the host
    binding's `transformer_decode_step` / `transformer_forward` on them:
    the same certified entries, the same addresses in the same order, so
    the byte-for-byte claim above holds on both arms and is what the
    `transformer-decode-session` lane hashes. The host arm is the OWNERSHIP
    wrapper only and makes NO speed claim."""

    def __init__(self, block, state):
        what = "TransformerDecodeSession"
        ext = block._extension()
        create = (getattr(ext, "transformer_decode_session_create", None)
                  if _exports(ext, "transformer_decode_session_create") else None)
        # THE HOST ARM. No device session entry, but the two per-call entries
        # the session would have run are right there under their own names.
        host = create is None and _exports(ext, "transformer_decode_step")
        if create is None and not host:
            raise NotImplementedError(
                f"mojolearn {what}: the loaded {type(block).__name__} binding "
                "exports neither a resident decode session nor the per-call "
                "transformer_decode_step entry the host arm runs; rebuild "
                "bindings/build_transformer.sh (or build_transformer_host.sh) "
                "in IDENTICAL mode"
            )
        if state is None:
            raise ValueError(f"mojolearn {what}: state is required (allocate_state)")
        _refuse_resident(state, what)
        if state.batch_size < 1:
            raise ValueError(f"mojolearn {what}: the state holds no rows")
        if int(getattr(state, "window", 0)) != block.window:
            raise ValueError(
                f"mojolearn {what}: the state was allocated for window "
                f"{getattr(state, 'window', 0)} but this block has window "
                f"{block.window}"
            )
        b = int(state.batch_size)
        n = b * block.n_kv_heads * state.capacity * block.head_dim
        kc = _state_buf(state.k_cache, what, "k_cache", (n,))
        vc = _state_buf(state.v_cache, what, "v_cache", (n,))
        self._block = block
        self._state = state
        self._ext = ext
        self._open = False
        w = block._w
        wopt = block._wopt  # noqa: F841  (keeps the optional arrays alive)
        if host:
            # The weights (the optional tail included) and the KV cache
            # COPIED, which is the ownership clause the device arm satisfies
            # with an upload. `cached_tokens` is unchanged by opening.
            self._native = None
            self._hw = [_private_copy(a) for a in w]
            self._hwopt = [_private_copy(a) for a in block._wopt]
            self._kc = _private_copy(kc)
            self._vc = _private_copy(vc)
            s0 = int(state.cached_tokens)
        else:
            self._native = create()
            addrs, params = block._with_tails(
                [0 if a is None else addr_ro(a, name="weight") for a in w]
                + [addr(kc, name="k_cache"), addr(vc, name="v_cache")],
                [b, block.d_model, block.n_heads, block.n_kv_heads, block.head_dim,
                 block.intermediate, int(state.max_tokens), int(state.cached_tokens),
                 block.window])
            s0 = int(ext.transformer_decode_session_open(self._native, addrs, params))
        self._open = True
        state.cached_tokens = s0
        state._resident_session = self

    def _host_call(self, entry, x, y, params):
        """One host entry on the session's OWN weights and cache: the address
        list `TransformerBlock._call_impl` builds, in the same order it
        documents (x, the nine weights, k_cache, v_cache, y_out, then the
        eleven-address options tail for a non-default block), with this
        session's buffers standing where the caller's would."""
        blk = self._block
        addrs = ([addr_ro(x, name="x")]
                 + [0 if a is None else addr_ro(a, name="weight") for a in self._hw]
                 + [addr(self._kc, name="k_cache"), addr(self._vc, name="v_cache"),
                    addr(y, name="y")])
        if blk._extended:
            addrs = addrs + [0 if a is None else addr_ro(a, name="weight")
                             for a in self._hwopt]
            params = params + list(blk._opts_tail)
        return int(entry(addrs, params))

    @property
    def state(self):
        return self._state

    @property
    def is_open(self):
        return self._open

    def _require_open(self, what):
        if not self._open:
            raise ValueError(f"mojolearn {what}: the session is closed")

    def step(self, x):
        what = "TransformerDecodeSession.step"
        self._require_open(what)
        blk = self._block
        x = _batch_tokens(x, what, blk.d_model, True)
        b = int(x.shape[0])
        st = self._state
        if b != st.batch_size:
            raise ValueError(
                f"mojolearn {what}: the session holds {st.batch_size} rows, x has B = {b}")
        y = empty((b, 1, blk.d_model), "<f4")
        if self._native is None:
            # B, d_model, n_heads, n_kv_heads, head_dim, intermediate,
            # max_tokens, cached_tokens, window -- `_call_impl`'s step params.
            new_len = self._host_call(
                self._ext.transformer_decode_step, x, y,
                [b, blk.d_model, blk.n_heads, blk.n_kv_heads, blk.head_dim,
                 blk.intermediate, int(st.max_tokens), int(st.cached_tokens), blk.window])
        else:
            new_len = self._ext.transformer_decode_session_step(
                self._native, [addr_ro(x, name="x"), addr(y, name="y")], [int(st.cached_tokens)])
        st.cached_tokens = int(new_len)
        return y

    def forward(self, x):
        what = "TransformerDecodeSession.forward"
        self._require_open(what)
        blk = self._block
        x = _batch_tokens(x, what, blk.d_model, False)
        b, l = int(x.shape[0]), int(x.shape[1])
        st = self._state
        if b != st.batch_size:
            raise ValueError(
                f"mojolearn {what}: the session holds {st.batch_size} rows, x has B = {b}")
        y = empty((b, l, blk.d_model), "<f4")
        if self._native is None:
            # B, L, then the same seven -- `_call_impl`'s forward params.
            new_len = self._host_call(
                self._ext.transformer_forward, x, y,
                [b, l, blk.d_model, blk.n_heads, blk.n_kv_heads, blk.head_dim,
                 blk.intermediate, int(st.max_tokens), int(st.cached_tokens), blk.window])
        else:
            new_len = self._ext.transformer_decode_session_forward(
                self._native, [addr_ro(x, name="x"), addr(y, name="y")], [l, int(st.cached_tokens)])
        st.cached_tokens = int(new_len)
        return y

    def sync_state(self):
        what = "TransformerDecodeSession.sync_state"
        self._require_open(what)
        st = self._state
        blk = self._block
        n = st.batch_size * blk.n_kv_heads * st.capacity * blk.head_dim
        kc = _state_buf(st.k_cache, what, "k_cache", (n,))
        vc = _state_buf(st.v_cache, what, "v_cache", (n,))
        if self._native is None:
            memcopy(addr(kc, name="k_cache"), addr_ro(self._kc, name="resident"), 4 * n)
            memcopy(addr(vc, name="v_cache"), addr_ro(self._vc, name="resident"), 4 * n)
        else:
            st.cached_tokens = int(self._ext.transformer_decode_session_export_state(
                self._native, [addr(kc, name="k_cache"), addr(vc, name="v_cache")]))
        return st

    def load_state(self):
        what = "TransformerDecodeSession.load_state"
        self._require_open(what)
        st = self._state
        blk = self._block
        n = st.batch_size * blk.n_kv_heads * st.capacity * blk.head_dim
        kc = _state_buf(st.k_cache, what, "k_cache", (n,))
        vc = _state_buf(st.v_cache, what, "v_cache", (n,))
        if self._native is None:
            memcopy(addr(self._kc, name="resident"), addr_ro(kc, name="k_cache"), 4 * n)
            memcopy(addr(self._vc, name="resident"), addr_ro(vc, name="v_cache"), 4 * n)
        else:
            st.cached_tokens = int(self._ext.transformer_decode_session_load_state(
                self._native, [addr(kc, name="k_cache"), addr(vc, name="v_cache")],
                [int(st.cached_tokens)]))
        return st

    def _release(self):
        if self._native is None:
            self._hw = self._hwopt = self._kc = self._vc = None
        else:
            self._ext.transformer_decode_session_close(self._native)

    def close(self):
        if not self._open:
            return
        try:
            self.sync_state()
        finally:
            self._open = False
            self._state._resident_session = None
            self._release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            if getattr(self, "_open", False):
                self._open = False
                self._state._resident_session = None
                self._release()
        except Exception:
            pass

    def __repr__(self):
        return "TransformerDecodeSession(open=%r, cached_tokens=%d)" % (
            self._open, int(self._state.cached_tokens))
