# SPDX-License-Identifier: Apache-2.0
"""Optimizers, schedules, losses and training primitives over float32 arrays.

These functions work on explicit float32 buffers: `mojolearn.Array`, NumPy arrays,
or any buffer-protocol object. NumPy is not required.

`SGD`, `Adam` and `AdamW` update the parameter arrays you pass in place, from gradients
you supply in the same order. `ConstantLR`, `WarmupLinearLR` and `WarmupCosineLR` give
the learning rate of each step. `clip_grad_norm_` scales gradients in place (L2 norm
only). `cross_entropy` takes float32 logits and integer class targets and can return
the logits gradient in the same call. `embedding_forward`, `rms_norm_forward`,
`linear_forward`, `chunked_lm_head_loss` and their backward functions are the layer
operations of a small language-model stack, and `Generator` draws reproducible
initializations and dropout masks.

These are building blocks, not an autograd framework or a drop-in replacement for
PyTorch: you compute and pass the gradients. `SmallMLPTrainer` and the byte language
model trainers provide complete training loops for their fixed model shapes.

`numeric_mode` selects the compiled tier where a function takes it. No compiled
extension is loaded until a function or optimizer step needs it.

Examples
--------
>>> import numpy as np
>>> from mojolearn.training import AdamW, WarmupCosineLR
>>> w = np.zeros((64, 8), dtype=np.float32)
>>> opt = AdamW([w], lr=3e-4, weight_decay=0.01,
...             lr_schedule=WarmupCosineLR(3e-4, warmup_steps=10, total_steps=100))
>>> opt.step([np.ones_like(w)])
"""
from ._training_impl import (
    SGD,
    Adam,
    AdamW,
    ConstantLR,
    Generator,
    WarmupCosineLR,
    WarmupLinearLR,
    accumulate_grads,
    accumulation_is_aligned,
    chunked_lm_head_loss,
    clip_grad_norm_,
    cross_entropy,
    embedding_backward,
    embedding_forward,
    linear_backward,
    linear_forward,
    mlp_inference_sessions,
    numeric_mode_used,
    rms_norm_backward,
    rms_norm_forward,
    vendor_used,
)
from ._samba_impl import SambaConfig, SambaStack
from ._residual_dropout_impl import residual_dropout, residual_dropout_backward

# THE SIX TRAINING PRIMITIVES (workstream D, 2026-09-14). `embedding_forward`
# and `embedding_backward` (the gather and the run-sorted ascending fold,
# `embedding/checks/embedding_identical.mojo`), `rms_norm_forward` and
# `rms_norm_backward`, `linear_forward` and `linear_backward` (torch's
# Linear without bias through the certified GEMM at OP_NT, the clause 9.2
# contraction for the weight gradient). They were exported by the shipped
# training binding and implemented in `_training_impl.py` since the
# transformer training lane and were omitted from this module and from
# `__all__`, so no public path reached them (the claim-surface census). The
# functions themselves are unchanged; this module now names them. The
# same rule as the rest of this file: bounded primitives over explicit
# float32 buffers, not an autograd framework.

# THE CHUNKED LM HEAD, V2 (lane/exposure-leftovers, 2026-09-23).
# `chunked_lm_head_loss` is the Python door to the v2 profile both training
# bindings register (training/CHUNKED_LM_HEAD_V2.md): the mean cross entropy
# of `hidden . weight^T` and its two gradients without the (rows, vocab)
# logits. It is laneless until an identity lane carries it; see
# tools/lane_accounting.py::DECLARED_LANELESS for what is owed.

__all__ = ['SGD', 'Adam', 'AdamW', 'clip_grad_norm_', 'cross_entropy',
           'numeric_mode_used', 'vendor_used', 'ConstantLR',
           'WarmupLinearLR', 'WarmupCosineLR', 'Generator',
           'accumulate_grads', 'accumulation_is_aligned', 'SambaConfig',
           'SambaStack',
           'embedding_forward', 'embedding_backward',
           'rms_norm_forward', 'rms_norm_backward',
           'linear_forward', 'linear_backward', 'chunked_lm_head_loss', 'mlp_inference_sessions', 'residual_dropout', 'residual_dropout_backward']
