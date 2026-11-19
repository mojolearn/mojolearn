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

The attention block here is the default block (transformer/block_options.mojo
`BlockOptions()`: RMSNorm without bias, SiLU-gated MLP, no projection
biases), the op's eight-integer form. A block with other options needs the
seventeen option words and its optional pieces; that form has no public
name yet.

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

#: The weight arrays `Session.attention_block` takes (the default options'
#: present pieces: two norms, q, k, v, o, gate, up, down), and the index of
#: `gate_proj.weight` among them (its size gives the intermediate width).
comptime SESSION_ATTENTION_PIECES = 9
comptime SESSION_ATTENTION_GATE = 6


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
        raise Error("ValueError: attention_block: gate_proj.weight must be (intermediate, d_model)")
    if max_positions < 1:
        raise Error("ValueError: attention_block: max_positions must be at least 1")
    if window < 0:
        raise Error("ValueError: attention_block: window must be >= 0")
    var ints: List[Int] = [
        batch, d_model, n_heads, n_kv_heads, d_model // n_heads, gate_values // d_model, max_positions, window,
    ]
    return ints^


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
