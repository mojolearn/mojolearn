# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the transformer (Llama-shaped decoder) block lane.

A FIFTEENTH extension module, and a separate one on purpose, for the
reason `bindings/_mojolearn_estimators.mojo`'s header gives: an
independently changing binding must not become a merge point. This file
gives the certified transformer block (profile
`mojolearn.identical.transformer.fp32.v1`, forward clause (a) and clause
(d) recorded on THREE columns 2026-08-28, the same card bytes on Apple,
NVIDIA and AMD -- `transformer/README.md`'s status block) its first
Python symbol: before it the lane exported nothing an outside consumer
could import.

THE BINDING CALLS THE CERTIFIED ENTRY POINT AND NOTHING ELSE. Every call
below goes through `llama_decoder_layer_forward`
(`transformer/impl/transformers/models/llama/modeling_llama.mojo`) --
the SAME function `transformer/checks/transformer_check.mojo` gates
(through its planted twin, with the plant OFF). NO arithmetic is
respelled in this file: a binding-side copy of any seam would be a
second spelling of a pinned rounding, which is the drift the whole lane
exists to forbid. The decode entry point is the contract's own
construction -- the block forward at `l = 1` with the KV cache carried
and `pos0 = cached_tokens` (contract section 7.2, ONE SPELLING FOR BOTH
PATHS) -- with no arithmetic of its own.

THE TWO-LIST ABI IS DEVIATION 791'S, ADOPTED, NOT RE-DECIDED. Each entry
point takes exactly TWO Python lists, `addrs` (every NumPy buffer
address, in an exact order written in each docstring and mirrored at the
`_transformer_impl.py` call site) and `params` (the scalars), because
`PythonModuleBuilder.def_function` stops elaborating above roughly nine
arguments (MEASURED 2026-09-01, `bindings/_mojolearn_gp.mojo`'s header)
and `transformer_forward` carries thirteen addresses. Every array whose
address goes into `addrs` MUST be bound to a Python local for the
duration of the call: an address inside a list keeps nothing alive
(`python/mojolearn/_arrays.py`).

DEVIATION 795 -- THE TRANSFORMER SURFACE'S OWN DEPARTURES, IN ONE BLOCK.

  (i) A POINTER-ABI SURFACE WHERE UPSTREAM HAS ONLY TORCH MODULES.
  Upstream's only Python face for this block is `LlamaDecoderLayer`, an
  `nn.Module` over torch tensors, autograd and a `DynamicCache` object.
  This surface is bare float32 buffer addresses plus packed scalars into
  the certified Mojo entry point, torch-free by design (the package's
  standing shape, mamba DEVIATIONS 791/792 applied to this lane): the
  consumer's job is cross-checking bits, and a torch dependency on the
  checking side would put the reference's own arithmetic inside the
  instrument.

  (ii) THE STATE IS THE KV CACHE, CROSSING AS TWO CALLER-OWNED
  FULL-CAPACITY BUFFERS PACKED AT THE USED STRIDE, PLUS ONE INTEGER.
  DEVIATION 792's caller-owned-buffers rule extends here with a twist
  none of the mamba states has: `LlamaKVCache` is PACKED at stride `s`
  (the USED length), not at stride `s_max` (DEVIATION 1022 -- the stage
  card records `[B, n_kv, S, head_dim]` and the packed used region IS
  that array), so the caller's `k_cache`/`v_cache` buffers are FLAT
  capacity buffers of `B * n_kv * max_tokens * head_dim` floats whose
  first `B * n_kv * cached_tokens * head_dim` elements are the cache at
  stride `cached_tokens`. Both are READ at entry and WRITTEN BACK whole
  before return, so the caller's bytes round-trip exactly; zeros in,
  with `cached_tokens = 0`, is a fresh sequence. `cached_tokens` is the
  one integer piece of state and crosses as a params scalar IN and as
  the return value OUT (mamba2 `buf_len`'s shape). `max_tokens` (the
  capacity, `s_max`) is a params scalar too, because the packing stride
  and every refusal about growth are functions of it.

  (iii) THE ROTARY TABLE AND THE ATTENTION SCALE ARE REBUILT PER CALL
  FROM FROZEN CONSTANTS, NEVER PARAMETERS. rms eps (1e-6, bits
  0x358637BD) and rope theta (10000.0, bits 0x461C4000) are contract
  section 3 FROZEN constants, imported from
  `transformer/checks/transformer_fixture.mojo` (the fixture is their
  bit authority); changing either is a v2 profile, not a knob. The
  rotary table is `LlamaRopeTable(ctx, dims, ROPE_THETA, max_tokens)`,
  computed on-device per call exactly as the lane's own check driver
  builds it -- a caching layer would be state this surface deliberately
  does not hold (upstream computes it once per config; recomputing is
  bit-inert because S6-S8 are pure functions of (theta, head_dim,
  position)).

  (iv) WHAT IS REFUSED HERE, AND WHAT GOES DOWN UNJUDGED (DEVIATION
  793's split, applied). Refused HERE: a null address, an
  `addrs`/`params` list of the wrong length, and a `cached_tokens`
  outside `[0, max_tokens]` -- each means the two sides of THIS boundary
  disagree and no lane refusal exists for it. Everything else goes down
  UNJUDGED so the lane's own refusals stay reachable from Python: the
  five divisibility rules are `LlamaDims.validate`'s (refused by name in
  Mojo), capacity growth past `max_tokens` and the 8192 absolute-
  position ceiling (DEVIATION 812) are `llama_decoder_layer_forward`'s
  and `LlamaKVCache`'s, and non-finite inputs are refused BY NAME on the
  device path (`llama_refuse_bad_call` for x/rope/cache per call;
  weights once at upload, DEVIATION 1875). The dtype refusals (float32
  ONLY) live in `_transformer_impl.py`, because dtype is a NumPy
  property this side cannot see.

  (v) BLAST RADIUS. This file, `python/mojolearn/_transformer_impl.py`,
  `python/mojolearn/transformer.py` and
  `python/mojolearn/tests/test_transformer_surface.py`. The lane's
  kernels, oracle, gates and contract are untouched; the BACKWARD
  profile (`transformer_backward_check.mojo`, present, NO recorded run)
  is deliberately NOT on this surface.

THE THREE-TIER SEMANTICS ARE THE BUILD'S, NOT THIS FILE'S. The numeric
mode (fast / deterministic / identical) is a compile-time define
(`checks/numerics.mojo`), one binary per tier under
`python/mojolearn/[<tier>/]_mojolearn_transformer.so`;
`transformer_numeric_mode` below is the read-back
`_transformer_impl.py` cross-checks so a wrong-arm measurement cannot be
correctly labelled by accident.

THE GIL is released around every device call, and nothing inside a
`GILReleased` block touches a `PythonObject`.

RUN LEDGER. NOTHING IN THIS FILE HAS EVER BEEN COMPILED. It is
UNVERIFIED, RUN OWED, per tier. Build:
`bash bindings/build_transformer.sh` (per tier via
MOJOLEARN_NUMERIC_MODE); gate:
`cd python && python3 -m mojolearn.tests.test_transformer_surface`.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR

# The two FROZEN arithmetic constants, from their bit authority
# (`transformer_fixture.mojo` cross-checks both against their hex bits in
# `profile_constants_are_intact`). The binding importing from
# `transformer/checks/` is the mamba binding's own precedent (it imports
# `mamba_fixture`'s constants); it is the IMPL file that may not
# (`modeling_llama.mojo`'s "no fixture type crosses this boundary").
from transformer.checks.transformer_fixture import RMS_EPS, ROPE_THETA

# `_upload`/`_download` by their underscore names is DEVIATION 1112's
# settled pattern (`transformer_check.mojo` imports the same pair).
from transformer.impl.transformers.models.llama.modeling_llama import (
    BLOCK_ANY_SABOTAGE,
    LlamaDeviceStages,
    LlamaDeviceWeights,
    LlamaDims,
    LlamaKVCache,
    LlamaRopeTable,
    _download,
    _upload,
    llama_decoder_layer_forward,
)


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _read_f32(addr: Int, n: Int) raises -> List[Float32]:
    """The first `n` float32 of a borrowed NumPy buffer, as a host list.
    A COPY, deliberately: the device helpers below take host lists, and
    the borrow ends when this returns, so no pointer is retained."""
    var p = _f32_ptr(addr)
    var out = List[Float32]()
    for i in range(n):
        out.append(p.unsafe_load(i))
    return out^


def _write_f32(addr: Int, values: List[Float32]) raises:
    """A host list into a borrowed NumPy buffer, element for element."""
    var p = _f32_ptr(addr)
    for i in range(len(values)):
        p.unsafe_store(i, values[i])


def transformer_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER, as the `NUMERIC_*` code itself: 0 FAST, 1
    IDENTICAL, 2 DETERMINISTIC. The same shape as `mamba_numeric_mode`
    and for the same reason: `_transformer_impl.py` reads it once and
    refuses to run if the binary it loaded disagrees with the mode the
    package asked for."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def transformer_vendor_binding() raises -> PythonObject:
    """'metal', 'cuda', 'hip' or 'none' -- the accelerator API this
    binary was compiled for, folded in from `checks/vendor.mojo`. The
    answer comes from the binary that actually loaded, never from the
    directory it sat in."""
    return PythonObject(String(COMPILED_VENDOR))


# ===========================================================================
# The block: profile mojolearn.identical.transformer.fp32.v1
# ===========================================================================


def _transformer_run(
    a: List[Int],
    b: Int,
    l: Int,
    dm: Int,
    nh: Int,
    nkv: Int,
    hd: Int,
    it: Int,
    smax: Int,
    s0: Int,
) raises -> Int:
    """The GIL-free half of the two entry points: everything after the
    `PythonObject`s have been read. Builds the device weights, uploads
    the caller's cache state, runs THE certified entry point once, and
    writes the output and the post-call state back into the caller's
    buffers. Returns the post-call `cached_tokens` (DEVIATION 795(ii):
    the one integer piece of the state).

    `[[mojo-buffer-freed-at-last-use]]`: every device buffer below is
    still alive when `llama_decoder_layer_forward` returns because that
    function synchronizes before it does, and the explicit transfers at
    the end hold them past the last download anyway."""
    # LlamaDims.validate REFUSES the five divisibility rules by name
    # (d_model == n_heads*head_dim, n_heads % n_kv == 0, head_dim even,
    # positivity) -- reached BEFORE any buffer size below is computed
    # from them, which is why nothing here pre-judges the shape.
    var dims = LlamaDims(dm, nh, nkv, hd, it)
    dims.validate()
    var qw = dims.q_width()
    var kw = dims.kv_width()
    if s0 < 0 or s0 > smax:
        raise Error(
            String("transformer: cached_tokens must be in [0, ")
            + String(smax)
            + "] (the cache capacity, max_tokens), got "
            + String(s0)
            + "; the two sides of this boundary disagree about the state"
        )

    var cache_n = b * nkv * smax * hd

    var ctx = DeviceContext()
    # Host weights THROUGH the lane's own struct (its length table is the
    # upstream shape authority, and its constructor is where the weights
    # are refused non-finite ONCE, DEVIATION 1875); values arrive as
    # given bits, unjudged.
    var w = LlamaDeviceWeights(
        ctx,
        dims,
        RMS_EPS,
        _read_f32(a[1], dm),
        _read_f32(a[2], dm),
        _read_f32(a[3], qw * dm),
        _read_f32(a[4], kw * dm),
        _read_f32(a[5], kw * dm),
        _read_f32(a[6], dm * qw),
        _read_f32(a[7], it * dm),
        _read_f32(a[8], it * dm),
        _read_f32(a[9], dm * it),
    )
    # The caller's cache over the fresh zeros. LlamaKVCache's own
    # constructor refuses b <= 0, smax <= 0 and smax > 8192 (the
    # absolute-position ceiling, DEVIATION 812) BY NAME before the
    # uploads below. Zeros in with s0 == 0 IS a fresh sequence; anything
    # else is a carried one, packed at stride s0 (DEVIATION 795(ii)).
    var kv = LlamaKVCache(ctx, b, dims, smax)
    kv.k = _upload(ctx, _read_f32(a[10], cache_n))
    kv.v = _upload(ctx, _read_f32(a[11], cache_n))
    kv.s = s0
    # Per call, from the FROZEN theta (DEVIATION 795(iii)). p_max is the
    # cache capacity: pos0 + l <= kv.s_max <= p_max holds for every legal
    # call, so the table always covers the absolute positions used.
    var rope = LlamaRopeTable(ctx, dims, ROPE_THETA, smax)
    var stages = LlamaDeviceStages(ctx, b, l, smax, dims)
    var dx = _upload(ctx, _read_f32(a[0], b * l * dm))

    var trace = IdentityTrace.disabled()
    # pos0 = s0: DEVIATION 1028 (cache slot j IS absolute position j)
    # refuses any other value, and this surface has no other value to
    # offer. Prefill, chunked continuation and decode are all this one
    # call; capacity growth past smax is refused by name downstream.
    llama_decoder_layer_forward(
        ctx, stages, kv, rope, w, dx, b, l, s0, trace, String("py")
    )

    # The block output is residual2 (contract section 2's
    # `residual2 + mlp(...)`, stage `residual2.out`), and the cache goes
    # back to its owner whole -- the full capacity buffer, so the bytes
    # round-trip exactly whatever the used stride is.
    _write_f32(a[12], _download(ctx, stages.residual2, b * l * dm))
    _write_f32(a[10], _download(ctx, kv.k, cache_n))
    _write_f32(a[11], _download(ctx, kv.v, cache_n))
    var out_len = kv.s
    _ = w^
    _ = kv^
    _ = rope^
    _ = stages^
    _ = dx^
    return out_len


def _transformer_addrs(addrs: PythonObject, what: String) raises -> List[Int]:
    if len(addrs) != 13:
        raise Error(
            what
            + ": addrs must contain 13 addresses (x,"
            " input_layernorm.weight, post_attention_layernorm.weight,"
            " q_proj.weight, k_proj.weight, v_proj.weight, o_proj.weight,"
            " gate_proj.weight, up_proj.weight, down_proj.weight, k_cache,"
            " v_cache, y_out), got "
            + String(len(addrs))
        )
    var a = List[Int]()
    for i in range(13):
        a.append(Int(py=addrs[i]))
    return a^


def transformer_forward_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """One transformer block call: prefill from the state the caller
    hands in -- fresh (zero caches, cached_tokens 0) or resumed (a
    carried cache: chunked-prefill continuation) -- any B and L that fit
    the capacity. Returns the post-call `cached_tokens`.

    `addrs` is the THIRTEEN buffer addresses, in this exact order
    (mirrored in `python/mojolearn/_transformer_impl.py`; QW =
    n_heads*head_dim = d_model, KW = n_kv_heads*head_dim, IT =
    intermediate_size, SMAX = max_tokens):

        0  x                                B * L * d_model float32, read
        1  input_layernorm.weight           d_model, read
        2  post_attention_layernorm.weight  d_model, read
        3  q_proj.weight                    QW * d_model, read (torch
                                             [out, in]; OP_NT reads it)
        4  k_proj.weight                    KW * d_model, read
        5  v_proj.weight                    KW * d_model, read
        6  o_proj.weight                    d_model * QW, read
        7  gate_proj.weight                 IT * d_model, read
        8  up_proj.weight                   IT * d_model, read
        9  down_proj.weight                 d_model * IT, read
       10  k_cache                          B * n_kv * SMAX * head_dim,
                                             READ AND WRITTEN (state;
                                             PACKED at stride
                                             cached_tokens, DEVIATION
                                             795(ii))
       11  v_cache                          B * n_kv * SMAX * head_dim,
                                             READ AND WRITTEN (state)
       12  y_out                            B * L * d_model, WRITTEN (the
                                             block output, both residual
                                             adds included -- contract
                                             section 2)

    `params` is, in this exact order:

        0  B
        1  L
        2  d_model        must equal n_heads*head_dim; refused otherwise
                           BY NAME in Mojo (LlamaDims.validate)
        3  n_heads
        4  n_kv_heads     n_heads % n_kv_heads must be 0 (GQA admitted,
                           contract DEVIATION 813)
        5  head_dim       must be even (RoPE pairs halves)
        6  intermediate   intermediate_size, > 0
        7  max_tokens     the cache capacity s_max; cached_tokens + L
                           past it, and any value over 8192 (DEVIATION
                           812's ceiling), are refused by name in Mojo
        8  cached_tokens  the carried cache's used length; 0 for a
                           fresh sequence

    Profile constants (rms eps 1e-6, rope theta 10000.0, eager attention,
    the causal mask value, no biases, silu -- contract section 3) are NOT
    parameters: changing one is a v2 profile, not a knob. There is no
    bias address and no dropout: the profile REFUSES both by absence."""
    var a = _transformer_addrs(addrs, String("transformer_forward"))
    if len(params) != 9:
        raise Error(
            "transformer_forward: params must contain 9 values (B, L,"
            " d_model, n_heads, n_kv_heads, head_dim, intermediate,"
            " max_tokens, cached_tokens), got "
            + String(len(params))
        )
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    var nh = Int(py=params[3])
    var nkv = Int(py=params[4])
    var hd = Int(py=params[5])
    var it = Int(py=params[6])
    var smax = Int(py=params[7])
    var s0 = Int(py=params[8])
    var out_len = 0
    with GILReleased(Python()):
        out_len = _transformer_run(a, b, l, dm, nh, nkv, hd, it, smax, s0)
    return PythonObject(out_len)


def transformer_decode_step_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """One decode token: the profile's spelling -- the block forward at
    L = 1 with the KV cache carried and `pos0 = cached_tokens` (contract
    section 7.2: ONE SPELLING FOR BOTH PATHS is what makes clause (d),
    decode == prefill bitwise, a theorem the gate verifies rather than a
    coincidence it hopes for), with NO arithmetic of its own. There is
    deliberately no second entry path to drift.

    `addrs`: the same THIRTEEN as `transformer_forward` with L = 1
    shapes (x and y_out are B * d_model). `params`: 0 B, 1 d_model,
    2 n_heads, 3 n_kv_heads, 4 head_dim, 5 intermediate, 6 max_tokens,
    7 cached_tokens. Returns the post-call cached_tokens."""
    var a = _transformer_addrs(addrs, String("transformer_decode_step"))
    if len(params) != 8:
        raise Error(
            "transformer_decode_step: params must contain 8 values (B,"
            " d_model, n_heads, n_kv_heads, head_dim, intermediate,"
            " max_tokens, cached_tokens), got "
            + String(len(params))
        )
    var b = Int(py=params[0])
    var dm = Int(py=params[1])
    var nh = Int(py=params[2])
    var nkv = Int(py=params[3])
    var hd = Int(py=params[4])
    var it = Int(py=params[5])
    var smax = Int(py=params[6])
    var s0 = Int(py=params[7])
    var out_len = 0
    with GILReleased(Python()):
        out_len = _transformer_run(a, b, 1, dm, nh, nkv, hd, it, smax, s0)
    return PythonObject(out_len)


@export
def PyInit__mojolearn_transformer() abi("C") -> PythonObject:
    # DEVIATION 793's last clause, applied here: a sabotage arm exists to
    # be run by a gate and to FAIL; a Python surface that quietly served
    # one would be a wrong answer wearing a green label. Refuse to exist
    # instead.
    comptime if BLOCK_ANY_SABOTAGE:
        abort(
            String(
                "_mojolearn_transformer: refusing to initialize -- a"
                " sabotage arm is compiled into this binary. Sabotage"
                " defines are for the lane gates (transformer/checks/),"
                " never for a shipped binding; rebuild with"
                " bash bindings/build_transformer.sh and no"
                " MOJOLEARN_TRANSFORMER_SABOTAGE_* or"
                " MOJOLEARN_BATCHINV_SABOTAGE_* define."
            )
        )
    try:
        var m = PythonModuleBuilder("_mojolearn_transformer")
        m.def_function[transformer_vendor_binding]("transformer_vendor")
        m.def_function[transformer_numeric_mode_binding](
            "transformer_numeric_mode"
        )
        m.def_function[transformer_forward_binding]("transformer_forward")
        m.def_function[transformer_decode_step_binding](
            "transformer_decode_step"
        )
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_transformer: ", e))
