# SPDX-License-Identifier: Apache-2.0
"""Alpha GPU optimizer and loss primitives over explicit buffer-protocol
arrays (`mojolearn.Array`, or NumPy arrays when the caller has NumPy;
DEVIATION 2461: NumPy is no longer a dependency of this module).

SGD, Adam and AdamW update supplied parameter arrays in place. Callers supply
gradients in the same stable tensor order. ``clip_grad_norm_`` changes supplied
gradients in place and supports L2 norm only. ``cross_entropy`` accepts FP32
logits and integer class targets, with an optional returned logits gradient.
The underlying functions document supported settings and explicit refusals;
these are bounded primitives, not a general autograd framework or drop-in
PyTorch replacement. Optimizer failure rollback is not a public contract of
these primitives. SmallMLPTrainer and SmallByteLanguageModelTrainer separately
provide transactional complete-state updates for their fixed model shapes.

``numeric_mode`` selects an available compiled tier. Alpha API exposure adds
no new bitwise validation: retained evidence must identify the exact profile,
source, binding, device, input bytes and state. No extension is loaded by these
re-exports until a function or optimizer operation needs it.
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
    clip_grad_norm_,
    cross_entropy,
    embedding_backward,
    embedding_forward,
    linear_backward,
    linear_forward,
    numeric_mode_used,
    rms_norm_backward,
    rms_norm_forward,
    vendor_used,
)
from ._samba_impl import SambaConfig, SambaStack

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

__all__ = ['SGD', 'Adam', 'AdamW', 'clip_grad_norm_', 'cross_entropy',
           'numeric_mode_used', 'vendor_used', 'ConstantLR',
           'WarmupLinearLR', 'WarmupCosineLR', 'Generator',
           'accumulate_grads', 'accumulation_is_aligned', 'SambaConfig',
           'SambaStack',
           'embedding_forward', 'embedding_backward',
           'rms_norm_forward', 'rms_norm_backward',
           'linear_forward', 'linear_backward']
