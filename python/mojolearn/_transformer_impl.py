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
keeps nothing alive (`_arrays.py`).

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

THE NUMERIC TIER IS THE LOADED BINARY'S, SELECTED AT BUILD TIME OF THE
.so. `numeric_mode=` on the class (or the process default) picks which
compiled set answers -- fast (no promise; the lane's own clause (d)
FAILS under FAST by construction and that is recorded as correct,
transformer/README.md), deterministic (same bits run to run on one
device), identical (also the same bits across vendors where the lane's
record says so: the FORWARD card is byte-identical on Apple, NVIDIA and
AMD as of 2026-08-28 -- clauses (b), (c), (e) and the sabotage ladder
remain OWED there and this class claims nothing wider) -- and
`_extension()` cross-checks the binary's own compile-time answer
against the tier the package resolved, `_gp_impl.py`'s pattern, so a
wrong-arm measurement cannot be correctly labelled by accident.

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

import numpy as np

from . import _backend
from ._arrays import _addr, _addr_ro
from ._mode import NumericModeMixin

#: `checks/numerics.mojo` codes, duplicated from `_backend._MODE_CODE` on
#: purpose, for `_arima_impl.py`'s reason: the read-back must not share a
#: table with the thing it checks.
_MODE_CODE = {"fast": 0, "identical": 1, "deterministic": 2}

#: The absolute-position ceiling, mirrored from the contract (section 3 /
#: modeling_llama.mojo's MAX_ABS_POSITION; DEVIATION 812: the Cody-Waite
#: domain of _cephes_sincosf_core). NOT a parameter -- documented here,
#: REFUSED BY NAME in Mojo (LlamaKVCache), where the authority stays.
_MAX_ABS_POSITION = 8192


def _f32_strict(a, what, name):
    """`a` as a C-contiguous float32 ndarray, REFUSING every other dtype
    BY NAME (module header: this surface certifies bits, so there is no
    convenience downcast). Layout may be fixed silently -- a copy to C
    order moves bytes untouched, so no output bit can move -- but the
    dtype may not."""
    a = np.asarray(a)
    if a.dtype != np.float32:
        raise TypeError(
            f"mojolearn {what}: {name} has dtype {a.dtype.name}; this "
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
    return np.ascontiguousarray(a)


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
    answer with no diagnostic."""
    if not isinstance(a, np.ndarray):
        raise TypeError(
            f"mojolearn {what}: state buffer {name} must be a numpy "
            f"ndarray (it is read and written in place), got {type(a)!r}"
        )
    if a.dtype != np.float32:
        raise TypeError(
            f"mojolearn {what}: state buffer {name} has dtype "
            f"{a.dtype.name}, want float32 (the state is part of the "
            "fp32 profile and round-trips byte for byte)"
        )
    if a.shape != shape:
        raise ValueError(
            f"mojolearn {what}: state buffer {name} has shape {a.shape},"
            f" want {shape}"
        )
    if not a.flags["C_CONTIGUOUS"] or not a.flags["WRITEABLE"]:
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
        x = x.reshape(x.shape[0], 1, x.shape[1])
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
            f"mojolearn {what}: weights must be a dict of numpy arrays "
            f"keyed by the upstream parameter names {names}"
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
    its natural shape."""

    def __init__(self, batch_size, n_kv_heads, head_dim, max_tokens,
                 k_cache, v_cache, cached_tokens=0):
        self.batch_size = int(batch_size)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = int(head_dim)
        self.max_tokens = int(max_tokens)
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.cached_tokens = int(cached_tokens)

    def _packed(self, buf):
        n = (self.batch_size * self.n_kv_heads * self.cached_tokens
             * self.head_dim)
        return buf[:n].reshape(
            self.batch_size, self.n_kv_heads, self.cached_tokens,
            self.head_dim,
        )

    def keys(self):
        """The key cache's packed used region, as a
        `(B, n_kv_heads, cached_tokens, head_dim)` VIEW into `k_cache`
        (valid until the next call updates the state)."""
        return self._packed(self.k_cache)

    def values(self):
        """The value cache's packed used region, the `keys()` shape."""
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
    upstream parameter names (modeling_llama.py's; the same names the
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
                                    name in Mojo)
        rms eps           FIXED    1e-6 (bits 0x358637BD); rope theta
          rope theta               10000.0 (0x461C4000); rope type
                                    "default" only -- frozen profile
                                    constants, a different value is a v2
        attention impl    FIXED    the EAGER path only (contract section
                                    6): FlashAttention, SDPA, paged
                                    attention and chunked prefill are
                                    out of scope BY CONTRACT, not
                                    missing
        biases/dropout    refused  by absence: attention_bias, mlp_bias
                                    and attention_dropout are the config
                                    defaults (False/False/0.0) and the
                                    profile refuses a nonzero value
                                    rather than specifying where it
                                    would round; no bias key exists
        rope_scaling,     refused  contract section 11's list, by
          sliding window,           absence
          masks beyond
          causal
        backward/training refused  the backward profile exists in the
                                    lane (`transformer_backward_check
                                    .mojo`) with NO recorded run and is
                                    deliberately NOT on this surface
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

    def __init__(self, weights, *, n_heads, n_kv_heads=None, head_dim=None):
        what = "TransformerBlock"
        arrs = _take(weights, what, self._W_NAMES)
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
        gate_w = _f32_strict(arrs[6], what, "gate_proj.weight")
        if gate_w.ndim != 2 or gate_w.shape[0] < 1 or gate_w.shape[1] != dm:
            raise ValueError(
                f"mojolearn {what}: gate_proj.weight must be "
                f"(intermediate, d_model={dm}), got shape {gate_w.shape}"
            )
        it = int(gate_w.shape[0])
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
            _want_shape(gate_w, what, "gate_proj.weight", (it, dm)),
            _want_shape(_f32_strict(arrs[7], what, "up_proj.weight"),
                        what, "up_proj.weight", (it, dm)),
            _want_shape(_f32_strict(arrs[8], what, "down_proj.weight"),
                        what, "down_proj.weight", (dm, it)),
        ]

    def allocate_state(self, batch_size, max_tokens):
        """The zero KV cache for `batch_size` sequences of up to
        `max_tokens` total positions each (prefill plus every later
        decode step; the capacity is fixed at allocation because the
        pack stride and every growth refusal are functions of it,
        DEVIATION 795(ii)). `max_tokens` above 8192 is refused BY NAME
        in Mojo at the first call (DEVIATION 812's absolute-position
        ceiling), not here."""
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
        n = b * self.n_kv_heads * smax * self.head_dim
        return TransformerState(
            b, self.n_kv_heads, self.head_dim, smax,
            np.zeros((n,), dtype=np.float32),
            np.zeros((n,), dtype=np.float32),
            0,
        )

    def _call(self, x, state, step):
        what = ("TransformerBlock.step" if step
                else "TransformerBlock.forward")
        x = _batch_tokens(x, what, self.d_model, step)
        b, l = int(x.shape[0]), int(x.shape[1])
        if state is None:
            # A self-contained prefill: the cache exists for exactly this
            # call and is discarded, so its capacity is the call's length.
            state = self.allocate_state(b, l)
        if state.batch_size != b:
            raise ValueError(
                f"mojolearn {what}: the state was allocated for "
                f"batch_size {state.batch_size} but x has B = {b} "
                "(allocate_state(B, max_tokens) makes a matching one)"
            )
        smax = int(state.max_tokens)
        n = b * self.n_kv_heads * smax * self.head_dim
        kc = _state_buf(state.k_cache, what, "k_cache", (n,))
        vc = _state_buf(state.v_cache, what, "v_cache", (n,))
        s0 = int(state.cached_tokens)
        # cached_tokens outside [0, max_tokens] is a boundary
        # disagreement the binding refuses by name; growth past the
        # capacity and the 8192 ceiling are refused by name in Mojo
        # (DEVIATION 795(iv)); all three go down unjudged.
        y = np.empty((b, l, self.d_model), dtype=np.float32)
        # TWO LISTS, NOT THIRTEEN ARGUMENTS (DEVIATION 791). Every array
        # addressed below is bound in this frame -- x, the nine entries
        # of self._w (alive on self), kc, vc, y -- which is what keeps
        # the addresses alive (_arrays.py).
        w = self._w
        ext = self._extension()
        addrs = (
            # ORDER MATCHES bindings/_mojolearn_transformer.mojo::
            # transformer_forward_binding: x, input_layernorm.weight,
            # post_attention_layernorm.weight, q_proj.weight,
            # k_proj.weight, v_proj.weight, o_proj.weight,
            # gate_proj.weight, up_proj.weight, down_proj.weight,
            # k_cache, v_cache, y_out
            [_addr_ro(x)]
            + [_addr_ro(a) for a in w]
            + [_addr(kc), _addr(vc), _addr(y)]
        )
        if step:
            # B, d_model, n_heads, n_kv_heads, head_dim, intermediate,
            # max_tokens, cached_tokens.
            new_len = ext.transformer_decode_step(
                addrs,
                [b, self.d_model, self.n_heads, self.n_kv_heads,
                 self.head_dim, self.intermediate, smax, s0],
            )
        else:
            # B, L, then the same six.
            new_len = ext.transformer_forward(
                addrs,
                [b, l, self.d_model, self.n_heads, self.n_kv_heads,
                 self.head_dim, self.intermediate, smax, s0],
            )
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

    def forward(self, x, state=None):
        """One block call: `(B, L, d_model)` float32 in, the block
        output (both residual adds included) back, any B and L that fit
        the state's capacity.

        `state=None` runs a self-contained prefill from a zero cache
        sized to exactly this call and DISCARDS the final state. Pass a
        `TransformerState` to carry it: the cache is read at entry and
        updated, so a later `forward` or `step` on that state continues
        the sequence -- bit-for-bit the prefill that ran the whole
        sequence at once under the IDENTICAL tier (contract section
        7.2's construction; the lane's clause (d) verifies it, and FAST
        deliberately promises none of it)."""
        return self._call(x, state, step=False)

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

    __call__ = forward
