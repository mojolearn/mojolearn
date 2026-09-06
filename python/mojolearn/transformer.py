# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mojolearn.transformer`: the Llama-shaped decoder block, for
cross-checking.

The public face of `_transformer_impl.py`, which carries every contract
detail on its classes; this module exists because the block is not an
estimator -- it has no `fit`, so it lives beside the sklearn-shaped
names rather than among them (it IS also re-exported from `mojolearn`
itself), exactly as `mojolearn.mamba` does for the Mamba blocks.

What is here, in one paragraph. `TransformerBlock` is ONE
reference-pinned Llama-shaped decoder layer -- input RMSNorm, eager
self-attention with RoPE, GQA and a KV cache, residual, post-attention
RMSNorm, SiLU-gated MLP, residual -- float32 in and out, weights handed
in as given bits, with the recurrent state EXPLICIT and caller-owned
(`TransformerState`: the KV cache as plain NumPy buffers that
round-trip byte for byte, plus `cached_tokens`). Prefill, continuation
from a carried cache and single-token decode all run through the
certified Mojo entry point the lane gates run (`transformer/checks/`);
`numeric_mode=` selects the fast / deterministic / identical tier at
call time, per instance. The identity claims are the LANE's: the
forward card is recorded byte-identical on Apple, NVIDIA and AMD
(clauses (a) and (d), 2026-08-28; the rest of the clause set is OWED
cross-vendor and `transformer/README.md` says exactly what). This
Python path itself printed green on 2026-09-02 in all three tiers,
44 checks 0 failed each (`tests/test_transformer_surface.py`), on ONE
APPLE M4 and no other vendor -- its NVIDIA and AMD columns are OWED.

    import numpy as np
    from mojolearn.transformer import TransformerBlock

    blk = TransformerBlock(weights, n_heads=2)  # dict of float32 arrays,
                                                # upstream parameter names
    y = blk.forward(x)                          # (B, L, d_model) -> same
    st = blk.allocate_state(batch_size=1, max_tokens=64)
    y = blk.forward(x, st)                      # prefill, cache carried
    for t in range(steps):                      # decode == prefill/token
        y_t = blk.step(x_t, st)                 # st updated in place
"""

from ._transformer_impl import TransformerBlock, TransformerState

__all__ = [
    "TransformerBlock",
    "TransformerState",
]
