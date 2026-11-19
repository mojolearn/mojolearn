# SPDX-License-Identifier: Apache-2.0
"""Typed pieces of the Session door's block operations (lane
fx-ops-session-blocks, 2026-11-18): `Session.mamba_block`,
`Session.attention_block`, `Session.mamba_step`, `Session.attention_step`.
The door (providers/python/autograd_ops_frontend_pyapi_door.mojo) converts
the Python arguments and calls these; no Python object reaches this file.

A block is three session tensors, exactly the ones the model programs make
(mamba/impl/mamba_block_program.mojo `MambaBlockProgram.__init__`,
transformer/causal_lm_program.mojo, training/samba_program.mojo): the packed
weights (`OpsSession.upload_block_grad`, BLOCK_PART_WEIGHTS), the zero
carried state (BLOCK_PART_STATE) and the state words (BLOCK_PART_WORDS; none
for Mamba-1), with a full-causal attention cache declared append-only behind
its position word (`declare_append`), as the causal LM program declares it.
A step is one `OpsSession.op` of the block's step code
(autograd/block_step_ops.mojo OPS_M1_STEP 81, OPS_M2_STEP 82, OPS_M3_STEP 83,
OPS_ATTN_STEP 84) on those tensors: the same op, the same operands and the
same integer and float words as the model programs', so the same launches in
the same order and the same output words on every column. In a recording
session the step is the zero-state prefill recorded as one node over
[x, weights] (`OpsSession._block_step`), which `backward` differentiates
through StepOpRules.

The attention block takes the transformer estimator's options under its
constructor's names (`session_attention_option_words`: the block options
record, transformer/block_options.mojo). The default record sends the op's
eight-integer form (today's block, the old code path); any other record
appends the seventeen option words, and the weights then include the
optional pieces that record turns on, in `block_weight_sizes` order.

Nothing here runs per element: the loops are over a block's pieces (model
structure). Packing a block is serial by nature: each piece is one cast
launch (or a host cast on the host executor, which splits its rows across
cores itself) queued in order on the session's executor.
"""
from sequence.exec_trait import Exec
from core.cast_in import CastIn
from autograd.ops_api import ops_session
from autograd.block_step_ops import (
    OPS_M1_STEP, OPS_M2_STEP, OPS_M3_STEP, OPS_ATTN_STEP, BLOCK_PART_WEIGHTS, BLOCK_PART_STATE, BLOCK_PART_WORDS,
    block_weight_sizes, block_state_sizes, block_int_words,
)
from transformer.block_options import BlockOptions
from autograd.generation_ops import OPS_LM_PICK, OPS_LM_ALIVE, lm_rows_words, LM_PICK_GREEDY, LM_PICK_SAMPLE
from autograd.rankn import ShapeN
from core.cast_in import CAST_KIND_SIGNED
from autograd.rules_recurrent import OPS_RNN_SCAN

#: The weight arrays `Session.attention_block` takes (the default options'
#: present pieces: two norms, q, k, v, o, gate, up, down), and the index of
#: `gate_proj.weight` among them (its size gives the intermediate width).
comptime SESSION_ATTENTION_PIECES = 9
comptime SESSION_ATTENTION_GATE = 6
#: The fewest arrays an attention block has (an ungated MLP has no gate
#: projection); array SESSION_ATTENTION_GATE is then `up_proj.weight`,
#: which has the same (intermediate, d_model) size.
comptime SESSION_ATTENTION_MIN_PIECES = 8


def session_mamba_code(version: Int) raises -> Int:
    """The step op of `Session.mamba_block(version=...)`."""
    if version == 1:
        return OPS_M1_STEP
    if version == 2:
        return OPS_M2_STEP
    if version == 3:
        return OPS_M3_STEP
    raise Error("ValueError: mamba_block: version must be 1, 2 or 3")


def session_attention_ints(
    batch: Int, d_model: Int, n_heads: Int, n_kv_heads: Int, gate_values: Int, max_positions: Int, window: Int,
) raises -> List[Int]:
    """OPS_ATTN_STEP's eight integers (b, d_model, n_heads, n_kv, head_dim,
    intermediate, s_max, window) from the public arguments: head_dim is
    d_model / n_heads and the intermediate width is read from
    `gate_proj.weight`'s size (`gate_values` = intermediate * d_model). The
    op validates the rest (`LlamaDims.validate`)."""
    if batch < 1 or d_model < 1:
        raise Error("ValueError: attention_block: batch_size and d_model must be positive")
    if n_heads < 1 or d_model % n_heads != 0:
        raise Error("ValueError: attention_block: d_model must be a multiple of n_heads")
    if n_kv_heads < 1 or n_heads % n_kv_heads != 0:
        raise Error("ValueError: attention_block: n_heads must be a multiple of n_kv_heads")
    if gate_values < d_model or gate_values % d_model != 0:
        raise Error("ValueError: attention_block: the seventh weight array (gate_proj.weight, or up_proj.weight "
                    + "of an ungated MLP) must be (intermediate, d_model)")
    if max_positions < 1:
        raise Error("ValueError: attention_block: max_positions must be at least 1")
    if window < 0:
        raise Error("ValueError: attention_block: window must be >= 0")
    var ints: List[Int] = [
        batch, d_model, n_heads, n_kv_heads, d_model // n_heads, gate_values // d_model, max_positions, window,
    ]
    return ints^


def session_attention_option_words(
    head_dim: Int, max_positions: Int, norm: String, mlp: String, rope_theta: Float32, rope_scaling: String,
    rope_factor: Float32, rope_low_freq_factor: Float32, rope_high_freq_factor: Float32,
    rope_original_max_positions: Int, rope_dim: Int, qkv_bias: Bool, o_bias: Bool, norm_eps: Float32,
    norm_bias: Bool, mlp_bias: Bool, qk_norm: Bool, attn_softcap: Float32,
) raises -> List[Int]:
    """The block options tail of OPS_ATTN_STEP from the transformer
    estimator's constructor arguments (`mojolearn.TransformerBlock`):
    `rope_scaling` is "" (none), "linear" or "llama3" with its factors;
    `rope_dim` 0 means head_dim; `attn_softcap` 0 means none. Returns an
    EMPTY list for the default record (the op's eight-integer form), else
    the seventeen words. The refusals are the model's
    (bindings/causal_lm_construct_door.mojo), then `BlockOptions.validate`."""
    var nk = 0 if norm == "rmsnorm" else (1 if norm == "layernorm" else (2 if norm == "rmsnorm_offset" else -1))
    var mk = -1
    if mlp == "swiglu":
        mk = 0
    elif mlp == "gelu":
        mk = 1
    elif mlp == "gelu_tanh":
        mk = 2
    elif mlp == "geglu":
        mk = 3
    elif mlp == "geglu_tanh":
        mk = 4
    if nk < 0:
        raise Error("ValueError: attention_block: unsupported norm='" + norm
                    + "'; norm is 'rmsnorm', 'layernorm' or 'rmsnorm_offset'")
    if mk < 0:
        raise Error("ValueError: attention_block: unsupported mlp='" + mlp
                    + "'; mlp is 'swiglu', 'gelu', 'gelu_tanh', 'geglu' or 'geglu_tanh'")
    var sk = 0
    if rope_scaling == "linear":
        sk = 1
    elif rope_scaling == "llama3":
        sk = 2
    elif rope_scaling != "":
        raise Error("ValueError: attention_block: unsupported rope_scaling type '" + rope_scaling
                    + "'; the types are 'linear' and 'llama3'")
    if sk == 2 and (not (rope_high_freq_factor > rope_low_freq_factor) or rope_original_max_positions < 1):
        raise Error("ValueError: attention_block: llama3 rope_scaling needs high_freq_factor > low_freq_factor "
                    + "and a positive original_max_position_embeddings")
    var rd = head_dim if rope_dim == 0 else rope_dim
    if rd < 1 or rd > head_dim or rd % 2 != 0:
        raise Error("ValueError: attention_block: rope_dim must be positive even and no larger than head_dim")
    if norm_bias and nk != 1:
        raise Error("ValueError: attention_block: norm_bias=True needs norm='layernorm'")
    if qk_norm and nk == 1:
        raise Error("ValueError: attention_block: qk_norm=True with norm='layernorm' is not supported "
                    + "(qk_norm is an RMSNorm)")
    var o = BlockOptions()
    o.rope_theta = rope_theta
    o.rope_scaling = sk
    if sk != 0:
        o.rope_factor = rope_factor
    if sk == 2:
        o.rope_low_freq_factor = rope_low_freq_factor
        o.rope_high_freq_factor = rope_high_freq_factor
        o.rope_original_max_positions = rope_original_max_positions
    o.rope_dim = 0 if rd == head_dim else rd
    # the absolute-position ceiling: the record's default unless this
    # block's cache is longer
    if max_positions > o.max_positions:
        o.max_positions = max_positions
    o.qkv_bias = qkv_bias
    o.o_bias = o_bias
    o.norm_kind = nk
    o.norm_eps = norm_eps
    o.norm_bias = norm_bias
    o.mlp_kind = mk
    o.mlp_bias = mlp_bias
    o.qk_norm = qk_norm
    o.attn_softcap = attn_softcap
    o.validate(head_dim)
    if o.is_default():
        return List[Int]()
    return o.to_params()


def session_block_create[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, code: Int, ints: List[Int], given: List[CastIn], requires_grad: Bool,
) raises -> List[Int]:
    """Pack one block for step op `code`: `given` holds one array per weight
    piece PRESENT under the block's options, in `block_weight_sizes` order
    (an absent optional piece takes no array). Returns [weights handle,
    state handle, words handle (0: the block carries none)]. The state is
    zero and the words 0: a fresh block. A failure frees what was made."""
    var s = ops_session[E, KIND](sid)
    var sizes = block_weight_sizes(code, ints)
    var present = 0
    for i in range(len(sizes)):  # small-loop(sizes: a block's weight pieces): layout count, no data
        if sizes[i] > 0:
            present += 1
    if len(given) != present:
        raise Error("ValueError: ops: this block takes " + String(present) + " weight arrays, got "
                    + String(len(given)))
    var srcs = List[CastIn](capacity=len(sizes))
    var at = 0
    for i in range(len(sizes)):  # small-loop(sizes: a block's weight pieces): one descriptor each, no data
        if sizes[i] > 0:
            srcs.append(given[at])
            at += 1
        else:
            srcs.append(CastIn.f32_c(0, 0, 1))
    var wh = s[].upload_block_grad(code, ints, BLOCK_PART_WEIGHTS, srcs, List[Int](), requires_grad)
    var sh = 0
    var qh = 0
    try:
        # The certified zero state (the model programs' own construction).
        var ssizes = block_state_sizes(code, ints)
        var most = 1
        for k in range(len(ssizes)):  # small-loop(ssizes: a block's state pieces): layout arithmetic, no data
            most = max(most, ssizes[k])
        var zero = List[Float32](length=most, fill=Float32(0))
        var zsrc = List[CastIn](capacity=len(ssizes))
        for k in range(len(ssizes)):  # small-loop(ssizes: a block's state pieces): one source each
            if ssizes[k] > 0:
                zsrc.append(CastIn.f32_c(Int(zero.unsafe_ptr()), ssizes[k], 1))
            else:
                zsrc.append(CastIn.f32_c(0, 0, 1))
        sh = s[].upload_block(code, ints, BLOCK_PART_STATE, zsrc, List[Int]())
        _ = len(zero)  # keep-alive past the staging copies
        var nk = block_int_words(code)
        if nk > 0:
            qh = s[].upload_block(code, ints, BLOCK_PART_WORDS, List[CastIn](), List[Int](length=nk, fill=0))
        if code == OPS_ATTN_STEP and ints[7] == 0:
            # A full-causal cache is append-only behind its position word: a
            # bounded loop's gate restores the word, not the cache.
            s[].declare_append(sh, qh)
    except err:
        if qh != 0:
            s[].free(qh)
        if sh != 0:
            s[].free(sh)
        s[].free(wh)
        raise err
    var out: List[Int] = [wh, sh, qh]
    return out^


def session_block_step[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, x: Int, weights: Int, state: Int, words: Int, config: List[Int], dt_lo: Float32, dt_hi: Float32,
    attention: Bool,
) raises -> Int:
    """One step op on a block made by `session_block_create`: `config` is
    [step code, then the op's integers]; `dt_lo` / `dt_hi` are Mamba-2's dt
    clamp (its two float words), read for that block only. Returns the new
    tensor's handle (x's shape)."""
    if len(config) < 3:
        raise Error("ValueError: ops: the block must come from mamba_block or attention_block")
    var code = config[0]
    if attention:
        if code != OPS_ATTN_STEP:
            raise Error("ValueError: ops: attention_step takes a block made by attention_block")
    elif code != OPS_M1_STEP and code != OPS_M2_STEP and code != OPS_M3_STEP:
        raise Error("ValueError: ops: mamba_step takes a block made by mamba_block")
    var ints = List[Int](capacity=len(config) - 1)
    for k in range(1, len(config)):  # small-loop(config: the op's two or eight integers): argument words, no data
        ints.append(config[k])
    var hs: List[Int] = [x, weights, state]
    if code != OPS_M1_STEP:
        if words == 0:
            raise Error("ValueError: ops: this block's state words are missing")
        hs.append(words)
    var fs = List[Float32]()
    if code == OPS_M2_STEP:
        fs.append(dt_lo)
        fs.append(dt_hi)
    var s = ops_session[E, KIND](sid)
    return s[].op(code, hs, ints, fs)


# ---------------------------------------------------------------- generation
# `Session.lm_rows`, `Session.lm_pick`, `Session.lm_alive` (lane
# fx-ops-session-blocks, 2026-11-19): the generation ops with the arguments
# the causal LM decode program passes (transformer/causal_lm_program.mojo
# `load_rows`, `set_policy`, `step`).
def session_lm_rows[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, b: Int, eos: List[Int]) raises -> Int:
    """The rows state of a fresh generation of `b` rows (autograd/
    generation_ops.mojo's layout): every row active with length 0, no
    non-finite token seen, alive 1, the sampler's position 0, then the eos
    ids. One int32-word tensor declared state; returns its handle."""
    if b < 1:
        raise Error("ValueError: lm_rows: batch_size must be positive")
    var n = lm_rows_words(b, len(eos))
    var w = List[Int32](length=n, fill=Int32(0))
    for r in range(b):  # driver-loop: over batch rows (state words, no data)
        w[2 * b + r] = Int32(-1)
    w[3 * b] = Int32(1)
    w[3 * b + 1] = Int32(-1)
    w[3 * b + 3] = Int32(len(eos))
    for i in range(len(eos)):  # small-loop(eos: the end ids): state words, no data
        if eos[i] < 0 or eos[i] >= (1 << 31):
            raise Error("ValueError: lm_rows: an eos token id must be a non-negative int32")
        w[3 * b + 4 + i] = Int32(eos[i])
    var s = ops_session[E, KIND](sid)
    var h = s[].upload_ids(CastIn(Int(w.unsafe_ptr()), CAST_KIND_SIGNED, 4, 1, n, n * 4, 4), ShapeN.matrix(1, n))
    _ = len(w)  # keep-alive past the staging copy
    s[].declare_state(h)
    return h


def session_lm_pick[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, logits: Int, rows: Int, pad: Int, sample: Bool, top_k: Int, seed_lo: Int, seed_hi: Int,
    temperature: Float32, top_p: Float32,
) raises -> Int:
    """OPS_LM_PICK on `logits` (b rows first, the vocabulary last) with the
    decode program's policy words: greedy sends top_k 0, temperature 1 and
    top_p 1, whatever was passed. Returns the (b, 1) int32 ids tensor."""
    var s = ops_session[E, KIND](sid)
    var sh = s[].shape_nd(logits)
    var b = sh.d[0]
    var vocab = sh.d[sh.ndim - 1]
    var ints: List[Int] = [b, vocab, pad, LM_PICK_SAMPLE if sample else LM_PICK_GREEDY, top_k if sample else 0,
                           seed_lo, seed_hi]
    var fs: List[Float32] = [temperature if sample else Float32(1.0), top_p if sample else Float32(1.0)]
    var hs: List[Int] = [logits, rows]
    return s[].op(OPS_LM_PICK, hs, ints, fs)


def session_lm_alive[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, ids: Int, rows: Int) raises -> Int:
    """OPS_LM_ALIVE: the rows state's alive word as a (1, 1) tensor (1.0
    while some row is unfinished and none failed); `ids` holds b rows."""
    var s = ops_session[E, KIND](sid)
    var b = s[].shape_nd(ids).d[0]
    var hs: List[Int] = [ids, rows]
    var bl: List[Int] = [b]
    return s[].op(OPS_LM_ALIVE, hs, bl, List[Float32]())


# ---------------------------------------------------------------- recurrent layers
def session_rnn[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, cell: Int, x: Int, w_ih: Int, w_hh: Int, b_ih: Int, b_hh: Int,
) raises -> Int:
    """`Session.lstm` / `Session.gru` / `Session.rnn`: one layer over a
    time-major (steps, batch, inputs) tensor, as torch.nn.LSTM / GRU / RNN
    with batch_first=False and a zero initial state. Returns the
    (steps, batch, hidden) outputs: OPS_RNN_SCAN, then the rank-3 view
    (operation 33, `reshape`) of its (steps * batch, hidden) result."""
    var s = ops_session[E, KIND](sid)
    var sh = s[].shape_nd(x)
    if sh.ndim != 3:
        raise Error("ValueError: ops: a recurrent layer takes a (steps, batch, inputs) tensor")
    var steps = sh.d[0]
    var batch = sh.d[1]
    var hidden = s[].shape_nd(w_hh).d[1]
    var hs: List[Int] = [x, w_ih, w_hh, b_ih, b_hh]
    var ints: List[Int] = [cell, steps, batch]
    var flat = s[].op(OPS_RNN_SCAN, hs, ints, List[Float32]())
    var fh: List[Int] = [flat]
    var view: List[Int] = [0, steps, batch, hidden]
    return s[].op(33, fh, view, List[Float32]())
