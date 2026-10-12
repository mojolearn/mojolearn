# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Llama-style transformer decoder block.

`TransformerBlock` is one Llama-shaped decoder layer (input RMSNorm, causal
self-attention with RoPE, grouped-query attention and a KV cache, residual,
post-attention RMSNorm, SiLU-gated MLP, residual), float32 in and out. It is a
building block, not an estimator: it has no `fit`. You pass in the weights, and
the KV cache is an explicit, caller-owned `TransformerState`. Prefill,
continuation from a carried cache and single-token decode all run through the
same computation. `TransformerDecodeSession` keeps the weights and cache on the
device across decode steps.

NumPy is not required: inputs are read through the buffer protocol and outputs
are `mojolearn.Array` objects, which `numpy.asarray` wraps without a copy. All
three names are also exported from `mojolearn`.

Examples
--------
>>> from mojolearn.transformer import TransformerBlock
>>> blk = TransformerBlock(weights, n_heads=2)      # dict of float32 buffers
>>> y = blk.forward(x)                              # (B, L, d_model) -> same
>>> st = blk.allocate_state(batch_size=1, max_tokens=64)
>>> y = blk.forward(x, st)                          # prefill, cache carried
>>> y_t = blk.step(x_t, st)                         # one decode token
"""

from ._transformer_impl import (TransformerBlock, TransformerDecodeSession,
                                TransformerState)

__all__ = [
    "TransformerBlock",
    "TransformerDecodeSession",
    "TransformerState",
]
