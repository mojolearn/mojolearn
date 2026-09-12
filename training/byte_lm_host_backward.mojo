# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One byte LM training step on the host, forward and backward (DEVIATION 2680).

COMPOSITION, NOT NEW ARITHMETIC. Every floating point operation below happens
inside an oracle that is already the normative answer of its own profile:

    embedding forward      emb_forward_oracle
    each block forward     transformer_block_oracle
    head                   gemm_oracle, OP_NT
    loss and its gradient  ce_forward_oracle, ce_backward_oracle
    head backward          _gemm_bwd_a / _gemm_bwd_b (host, over gemm_oracle)
    each block backward    transformer_block_backward_oracle
    embedding gradient     emb_backward_oracle
    the update             optimizer_step_oracle

Nothing here adds a seam, a fold or a flush. What this file contributes is the
ORDER, which is `training/byte_lm.mojo::_byte_step_device`'s order, and the
byte LM's own parameter layout, taken from `ByteConfig.offsets()` rather than
spelled a fourth time (`train_offsets`, `pack_grads` and
`train_step_check.mojo` are the three that already exist, and that lane's own
plan lists a single shared table as owed).

WHAT IS ALREADY PROVEN, AND WHERE THE GAP WAS. `train_step_check.mojo` has
compared a composed host step against the device step bitwise, thirteen
stages, with four negative controls firing, but on ONE device and at a ONE
BLOCK fixture with its own registry. The byte LM profile has two blocks and a
34944 element registry, and its recorded capture covers three vendors. So this
file is the byte LM shaped composition, and `tools/byte_lm_cpu_train_gate.py`
is what judges it against all three.

TWO BLOCKS IS THE ONE STRUCTURAL DIFFERENCE from that check. Each layer's
backward needs THAT LAYER'S saved forward stages and THAT LAYER'S input, which
is the previous layer's `residual2_out` and, for layer 0, the embedding output.
The forward here therefore retains both per layer instead of overwriting one
running activation, exactly as the device step keeps `tr.forward[layer]` and
passes `tr.forward[layer - 1].residual2` down.

NO THREADING, NO FAST PATH. This is the reference path's shape: one thread,
the oracles as written. The threaded host forward (DEVIATION 2640) has no
backward twin, and it must not grow one by accident: a weight gradient sums
over every row of the batch, so unlike the forward it crosses every thread
boundary, and a threaded version needs a fixed cross thread fold rather than
letting threads accumulate as they finish.

BATCH COMPOSITION IS PART OF THE CLAIM. Nine of the gradients contract over
the token count, so `gemm_backward.mojo`'s own statement applies: the gradient
at 1024 tokens is not the bits of the gradient at 512 tokens accumulated
twice. A pass here is a statement at the shape it ran at.
"""

from gemm.checks.gemm_oracle import OP_NT, gemm_oracle
from embedding.checks.embedding_oracle import (
    EmbConfig,
    emb_backward_oracle,
    emb_forward_oracle,
)
from training.byte_lm_config import ByteConfig
from training.byte_lm_host import (
    byte_host_block_weights,
    byte_host_dims,
)
from training.checks.loss_oracle import (
    CeConfig,
    ce_backward_oracle,
    ce_forward_oracle,
)
from training.checks.optimizer_oracle import (
    OPT_ADAMW,
    OptimizerConfig,
    optimizer_step_oracle,
)
from transformer.checks.transformer_fixture import ScorePlant
from transformer.checks.transformer_oracle import (
    RopeTable,
    TransformerKVCache,
    TransformerStages,
    build_rope_table,
    transformer_block_oracle,
)
from transformer.checks.transformer_backward_oracle import (
    _gemm_bwd_a,
    _gemm_bwd_b,
    transformer_block_backward_oracle,
)


def byte_host_adamw(lr: Float32, beta1: Float32, beta2: Float32,
                    eps: Float32, weight_decay: Float32) -> OptimizerConfig:
    """The byte LM's optimizer, spelled the one way its own validator allows.

    `byte_validate_optimizer` (training/byte_lm.mojo) refuses anything but
    positive-lr AdamW with finite legal betas and eps, no clipping, no
    momentum, no dampening and no nesterov, so the remaining fields are zero
    because that is the only value they may take here, not because zero is a
    convenient default."""
    return OptimizerConfig(OPT_ADAMW, lr, beta1, beta2, eps, weight_decay,
                           Float32(0.0), Float32(0.0), False, Float32(0.0))


struct ByteHostStep(Movable):
    """One step's outputs, in the order a gate reads them.

    `grad` is the flat gradient in the registry's order, which is the order
    `post_p` and the capture's `grad.f32` use: embed, then each block's nine
    tensors, then lm_head. `loss` is the mean next byte cross entropy of the
    step's own batch, before the update."""

    var loss: Float32
    var grad: List[Float32]
    var param: List[Float32]
    var m_state: List[Float32]
    var v_state: List[Float32]

    def __init__(out self, loss: Float32, var grad: List[Float32],
                 var param: List[Float32], var m_state: List[Float32],
                 var v_state: List[Float32]):
        self.loss = loss
        self.grad = grad^
        self.param = param^
        self.m_state = m_state^
        self.v_state = v_state^


def byte_host_split_ids(ids: List[Int32], config: ByteConfig) raises
        -> Tuple[List[Int32], List[Int32]]:
    """`(inputs, targets)` from ids `[batch, length + 1]`.

    The same split `byte_host_loss` and `_byte_forward_loss` perform: a row's
    inputs are its first `length` bytes and its targets are that row shifted
    by one. Spelled once here so the step cannot drift from the loss."""
    var b = config.batch
    var l = config.length
    if len(ids) != b * (l + 1):
        raise Error("byte LM host step: ids must be [batch, length + 1]")
    var inputs = List[Int32](capacity=b * l)
    var targets = List[Int32](capacity=b * l)
    for bi in range(b):
        for li in range(l):
            inputs.append(ids[bi * (l + 1) + li])
            targets.append(ids[bi * (l + 1) + li + 1])
    return (inputs^, targets^)


def byte_host_gradient(params: List[Float32], ids: List[Int32],
                       config: ByteConfig) raises
        -> Tuple[Float32, List[Float32]]:
    """`(loss, grad)` for one batch, the whole backward pass, no update.

    The order is `_byte_step_device`'s: forward and loss, the loss gradient,
    the head's two GEMM backwards, the blocks in REVERSE, the embedding
    gradient, then the pack. No parameter, moment or step is touched."""
    config.validate()
    var offsets = config.offsets()
    var dims = byte_host_dims(config)
    var rope = build_rope_table(dims)
    var b = config.batch
    var l = config.length
    var m = b * l
    var dm = config.d_model
    var v = config.vocab_size
    var emb_cfg = EmbConfig.llama(v, dm)
    var ce_cfg = CeConfig.causal_lm(v)
    # `List` is not `ImplicitlyCopyable`, so reading a list out of a tuple is
    # an explicit copy or a transfer. Copies here, because both are read again
    # below (`inputs` by the embedding gradient, `targets` by the loss).
    var split = byte_host_split_ids(ids, config)
    var inputs = split[0].copy()
    var targets = split[1].copy()

    # ---- forward, retaining every layer's stages AND its input ------------
    # `inputs_of[layer]` is what that layer's backward needs as `x`: the
    # embedding output for layer 0 and the previous layer's residual2
    # otherwise. The device step reads the same two things.
    var x = emb_forward_oracle(_byte_host_slice(params, offsets, 0), inputs, emb_cfg)
    var inputs_of = List[List[Float32]]()
    var saved = List[TransformerStages]()
    for layer in range(config.n_layers):
        var w = byte_host_block_weights(params, offsets, layer, dims)
        var cache = TransformerKVCache(b, dims, l, 0)
        inputs_of.append(x.copy())
        var st = transformer_block_oracle(w, x, b, l, cache, rope, ScorePlant.none())
        x = st.residual2_out.copy()
        saved.append(st^)

    # ---- head, loss, and the loss gradient --------------------------------
    var head_id = config.n_tensors() - 1
    var lm_w = _byte_host_slice(params, offsets, head_id)
    var logits = gemm_oracle(x, lm_w, OP_NT, m, v, dm)
    var ce = ce_forward_oracle(logits, targets, ce_cfg)
    ce_backward_oracle(ce, targets, ce_cfg)
    var loss = ce.loss[0]

    # ---- the head's two GEMM backwards ------------------------------------
    # `x` is still the LAST layer's residual2, which is the head's forward `A`
    # operand, so dB reads it unchanged. dA arrives as the last layer's
    # incoming cotangent.
    var d_h = _gemm_bwd_a(ce.dlogits, lm_w, OP_NT, m, v, dm)
    var dw_lm = _gemm_bwd_b(ce.dlogits, x, OP_NT, m, v, dm)

    # ---- the blocks, in reverse -------------------------------------------
    var per_layer = List[List[Float32]]()
    for _ in range(config.n_layers):
        per_layer.append(List[Float32]())
    var d_out = d_h.copy()
    for layer in range(config.n_layers - 1, -1, -1):
        var w = byte_host_block_weights(params, offsets, layer, dims)
        var bwd = transformer_block_backward_oracle(
            w, saved[layer], d_out, b, l, 0, rope
        )
        var packed = List[Float32]()
        _extend(packed, bwd.dw_norm1)
        _extend(packed, bwd.dw_q)
        _extend(packed, bwd.dw_k)
        _extend(packed, bwd.dw_v)
        _extend(packed, bwd.dw_o)
        _extend(packed, bwd.dw_norm2)
        _extend(packed, bwd.dw_gate)
        _extend(packed, bwd.dw_up)
        _extend(packed, bwd.dw_down)
        per_layer[layer] = packed^
        d_out = bwd.d_x.copy()

    # ---- the embedding gradient -------------------------------------------
    # `accumulate` is off, so `dw_prev` is unread and an empty list is the
    # honest argument; `emb_backward_seed` is the oracle's own decision.
    var dw_emb = emb_backward_oracle(d_out, inputs, emb_cfg, List[Float32]())

    # ---- pack, in the registry's order ------------------------------------
    var grad = List[Float32](capacity=config.n_total())
    _extend(grad, dw_emb)
    for layer in range(config.n_layers):
        _extend(grad, per_layer[layer])
    _extend(grad, dw_lm)
    if len(grad) != config.n_total():
        raise Error(
            String("byte LM host step: packed gradient is ")
            + String(len(grad)) + " floats, the registry says "
            + String(config.n_total())
        )
    return (loss, grad^)


def byte_host_train_step(params: List[Float32], m_state: List[Float32],
                         v_state: List[Float32], ids: List[Int32],
                         config: ByteConfig, opt: OptimizerConfig,
                         completed_steps: Int) raises -> ByteHostStep:
    """One whole step: the gradient, then the update.

    `completed_steps` is the number of steps ALREADY taken, so the optimizer's
    `t` is `completed_steps + 1`, which is what the bias correction of a first
    step needs and what the capture's `initial_*` arrays are the state for."""
    config.validate()
    var n = config.n_total()
    if len(params) != n or len(m_state) != n or len(v_state) != n:
        raise Error("byte LM host step: parameters and moments must be n_total")
    if completed_steps < 0:
        raise Error("byte LM host step: completed_steps must not be negative")
    # `loss` is a Float32 and copies implicitly; `grad` is a List and does not.
    var got = byte_host_gradient(params, ids, config)
    var loss = got[0]
    var grad = got[1].copy()

    # `optimizer_step_oracle` takes five `mut` arguments and clips `grad` in
    # place, so the locals are transferred into it rather than assigned.
    var p_out = params.copy()
    var g_out = grad.copy()
    var m_out = m_state.copy()
    var v_out = v_state.copy()
    var initialized = List[Bool]()
    for _ in range(config.n_tensors()):
        initialized.append(True)
    var offsets = config.offsets()
    _ = optimizer_step_oracle(p_out, g_out, m_out, v_out, initialized,
                              offsets, opt, completed_steps + 1)
    return ByteHostStep(loss, grad^, p_out^, m_out^, v_out^)


def _byte_host_slice(values: List[Float32], offsets: List[Int], j: Int)
        -> List[Float32]:
    """Registry tensor `j`. A copy, because the oracles take owned lists."""
    var out = List[Float32](capacity=offsets[j + 1] - offsets[j])
    for i in range(offsets[j], offsets[j + 1]):
        out.append(values[i])
    return out^


def _extend(mut into: List[Float32], values: List[Float32]):
    for i in range(len(values)):
        into.append(values[i])
