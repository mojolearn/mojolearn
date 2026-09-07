# SPDX-License-Identifier: Apache-2.0
"""Alpha GPU optimizer and loss primitives over explicit NumPy arrays.

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
    clip_grad_norm_,
    cross_entropy,
    numeric_mode_used,
    vendor_used,
)

__all__ = ['SGD', 'Adam', 'AdamW', 'clip_grad_norm_', 'cross_entropy',
           'numeric_mode_used', 'vendor_used']
