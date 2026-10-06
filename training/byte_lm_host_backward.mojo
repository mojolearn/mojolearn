# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One byte LM training step on the host, forward and backward (DEVIATION 2680).

COMPOSITION, NOT NEW ARITHMETIC. Every floating point operation below happens
inside an oracle that is already the normative answer of its own profile:

    embedding forward      emb_forward_oracle
    each block forward     transformer_block_oracle
    head                   gemm_oracle, OP_NT (through gemm_host_rows, its bits at
                           CPU speed)
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

NO FAST PATH in this file: the oracles as written, in the oracles' order.
WHAT RUNS ON MORE THAN ONE CORE, AND WHY THE BITS ARE THE SAME (lane
neural-cpu 2026-09-28, lane neural-pass6 2026-09-30). Every cross-row fold
in the step is a gemm v1 call (`gemm/host/gemm_host_rows.mojo` computes
`gemm_oracle`'s bits, its fixed leaf-and-tree chain over the contraction
axis, whatever the task count): the projections, the attention products,
the head, every weight gradient (which is where the sum over the batch
lives), the norm weight gradients and the loss's denominators. Everything
else is a per-row or per-element chain, and a chain runs the same
statements in the same order whichever task runs it, in the calling
thread's floating-point environment (`core/host_parallel.mojo`, DEVIATION
5900). So the block oracles' per-row stages, the loss's rows
(`training/loss_host_rows.mojo`) and the optimizer's elements
(`training/optimizer_host_rows.mojo`) run over host tasks. The loss and the
optimizer keep the serial oracle calls one environment value away
(`MOJOLEARN_BYTE_LM_HOST_STEP_ROWS=0`); `tools/byte_lm_host_step_check.py`
compares the two, every byte of every step, and
`tools/byte_lm_cpu_train_gate.py cpu` compares either against the recorded
three-vendor bytes. Nothing lets a thread accumulate as it finishes.

BATCH COMPOSITION IS PART OF THE CLAIM. Nine of the gradients contract over
the token count, so `gemm_backward.mojo`'s own statement applies: the gradient
at 1024 tokens is not the bits of the gradient at 512 tokens accumulated
twice. A pass here is a statement at the shape it ran at.
"""

from std.memory import unsafe_memcpy
from std.os import getenv

from core.host_lanes import host_f32_uninit
from gemm.contract import OP_NT
from gemm.host.identical_gemm import gemm_oracle
from gemm.host.gemm_host_rows import gemm_host_rows
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
from training.checks.loss_contract import CeConfig
from training.checks.loss_oracle import ce_backward_oracle, ce_forward_oracle
from training.checks.optimizer_contract import OPT_ADAMW, OptimizerConfig
from training.checks.optimizer_oracle import optimizer_step_oracle
from training.loss_host_rows import ce_host_rows
from training.chunked_lm_head_host import chunked_lm_head_gemm_host_train
from training.optimizer_host_rows import adam_host_rows
from transformer.checks.transformer_fixture import ScorePlant, TransformerWeights
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


def byte_host_step_rows() -> Bool:
    """True unless MOJOLEARN_BYTE_LM_HOST_STEP_ROWS=0: the loss's rows and the
    optimizer's elements over host tasks (module note); 0 restores the serial
    oracle calls for both, which is what the check tool compares against."""
    return String(getenv("MOJOLEARN_BYTE_LM_HOST_STEP_ROWS")) != "0"


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
    # Each layer's weights are sliced ONCE a step (lane neural-pass8) and the
    # backward reads the same lists; the stages keep each layer's input
    # (`input_x`), so no separate copy of it is held.
    var weights = List[TransformerWeights]()
    var saved = List[TransformerStages]()
    for layer in range(config.n_layers):
        weights.append(byte_host_block_weights(params, offsets, layer, dims))
        var cache = TransformerKVCache(b, dims, l, 0)
        var st = transformer_block_oracle(weights[layer], x, b, l, cache, rope, ScorePlant.none())
        x = st.residual2_out.copy()
        saved.append(st^)

    # ---- head, loss, and the loss gradient --------------------------------
    var head_id = config.n_tensors() - 1
    var lm_w = _byte_host_slice(params, offsets, head_id)
    var loss: Float32
    var d_h: List[Float32]
    var dw_lm: List[Float32]
    if config.chunked_lm_head_v2:
        # NI34: match ByteTrainer's GEMM-logit chunked profile, including its
        # serial vocabulary/dWeight folds; no rows*vocab host activation.
        var head = chunked_lm_head_gemm_host_train(x, lm_w, targets, m, v, dm)
        loss = head.loss
        d_h = head.d_hidden.copy()
        dw_lm = head.d_weight.copy()
    else:
        var logits = gemm_host_rows(x, lm_w, OP_NT, m, v, dm)
        var dlogits: List[Float32]
        if byte_host_step_rows():
            var got = ce_host_rows(logits, targets, ce_cfg)
            loss = got[0]
            dlogits = got[1].copy()
        else:
            var ce = ce_forward_oracle(logits, targets, ce_cfg)
            ce_backward_oracle(ce, targets, ce_cfg)
            loss = ce.loss[0]
            dlogits = ce.dlogits.copy()
        _ = logits^

        # ---- the head's two GEMM backwards ------------------------------------
        # `x` is still the LAST layer's residual2, which is the head's forward `A`
        # operand, so dB reads it unchanged. dA arrives as the last layer's
        # incoming cotangent.
        d_h = _gemm_bwd_a(dlogits, lm_w, OP_NT, m, v, dm)
        dw_lm = _gemm_bwd_b(dlogits, x, OP_NT, m, v, dm)
        _ = dlogits^

    # ---- the blocks, in reverse -------------------------------------------
    # Each gradient tensor lands at its registry offset directly (lane
    # neural-pass8): the same bytes in the same order as the per-layer pack
    # and the final pack it replaces, written once. `placed` counts floats
    # so the whole registry is proven covered before the gradient is handed
    # out (the shape check every tensor passes on its way in).
    var grad = host_f32_uninit(config.n_total())
    var placed = 0
    var d_out = d_h.copy()
    placed += _place(grad, offsets, head_id, dw_lm)
    for layer in range(config.n_layers - 1, -1, -1):
        var bwd = transformer_block_backward_oracle(
            weights[layer], saved[layer], d_out, b, l, 0, rope
        )
        var base = 1 + layer * 9
        placed += _place(grad, offsets, base, bwd.dw_norm1)
        placed += _place(grad, offsets, base + 1, bwd.dw_q)
        placed += _place(grad, offsets, base + 2, bwd.dw_k)
        placed += _place(grad, offsets, base + 3, bwd.dw_v)
        placed += _place(grad, offsets, base + 4, bwd.dw_o)
        placed += _place(grad, offsets, base + 5, bwd.dw_norm2)
        placed += _place(grad, offsets, base + 6, bwd.dw_gate)
        placed += _place(grad, offsets, base + 7, bwd.dw_up)
        placed += _place(grad, offsets, base + 8, bwd.dw_down)
        d_out = bwd.d_x.copy()

    # ---- the embedding gradient -------------------------------------------
    # `accumulate` is off, so `dw_prev` is unread and an empty list is the
    # honest argument; `emb_backward_seed` is the oracle's own decision.
    var dw_emb = emb_backward_oracle(d_out, inputs, emb_cfg, List[Float32]())
    placed += _place(grad, offsets, 0, dw_emb)
    if placed != config.n_total():
        raise Error(
            String("byte LM host step: placed gradient is ")
            + String(placed) + " floats, the registry says "
            + String(config.n_total())
        )
    return (loss, grad^)


def byte_host_train_step(var params: List[Float32], var m_state: List[Float32],
                         var v_state: List[Float32], ids: List[Int32],
                         config: ByteConfig, opt: OptimizerConfig,
                         completed_steps: Int) raises -> ByteHostStep:
    """One whole step: the gradient, then the update.

    `completed_steps` is the number of steps ALREADY taken, so the optimizer's
    `t` is `completed_steps + 1`, which is what the bias correction of a first
    step needs and what the capture's `initial_*` arrays are the state for.

    The parameters and both moments arrive OWNED (lane neural-pass8) and come
    back updated in the same lists: the three 80 MB copies a step used to make
    here, on one thread, were a tenth of its wall on a 64-core host. A caller
    that still needs its inputs passes copies."""
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

    if byte_host_step_rows():
        # No clipping (`byte_validate_optimizer`), so the gradient is read,
        # not scaled: the elements over host tasks, written in place.
        adam_host_rows(params, grad, m_state, v_state, opt, completed_steps + 1)
        return ByteHostStep(loss, grad^, params^, m_state^, v_state^)
    # `optimizer_step_oracle` takes five `mut` arguments and clips `grad` in
    # place, so the gradient is copied for it.
    var g_out = grad.copy()
    var initialized = List[Bool]()
    for _ in range(config.n_tensors()):
        initialized.append(True)
    var offsets = config.offsets()
    _ = optimizer_step_oracle(params, g_out, m_state, v_state, initialized,
                              offsets, opt, completed_steps + 1)
    return ByteHostStep(loss, grad^, params^, m_state^, v_state^)


def _byte_host_slice(values: List[Float32], offsets: List[Int], j: Int)
        -> List[Float32]:
    """Registry tensor `j`. A copy, because the oracles take owned lists; one
    block copy (lane neural-pass8)."""
    var n = offsets[j + 1] - offsets[j]
    var out = host_f32_uninit(n)
    if n > 0:
        unsafe_memcpy(dest=out.unsafe_ptr(), src=values.unsafe_ptr().unsafe_offset(offsets[j]), count=n)
    return out^


def _place(mut grad: List[Float32], offsets: List[Int], j: Int,
           values: List[Float32]) raises -> Int:
    """Registry tensor `j` of the gradient := `values`, one block copy at the
    tensor's offset; refuses a length other than the tensor's. Returns the
    floats written."""
    var n = offsets[j + 1] - offsets[j]
    if len(values) != n:
        raise Error(
            String("byte LM host step: gradient tensor ") + String(j) + " holds "
            + String(len(values)) + " floats, the registry says " + String(n)
        )
    if n > 0:
        unsafe_memcpy(dest=grad.unsafe_ptr().unsafe_offset(offsets[j]), src=values.unsafe_ptr(), count=n)
    return n
