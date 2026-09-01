# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mojolearn.mamba`: the Mamba-1 and Mamba-2 blocks, for cross-checking.

The public face of `_mamba_impl.py`, which carries every contract detail
on its classes; this module exists because `mamba/FEATURE_PARITY.md`'s
consumer table names `mojolearn.mamba` as the surface a downstream user
imports, and because the block classes are not estimators -- they have no
`fit`, so they live beside the sklearn-shaped names rather than among
them (they ARE also re-exported from `mojolearn` itself).

What is here, in one paragraph. `Mamba1Block` and `Mamba2Block` are ONE
reference-pinned block each -- norm, mixer, residual -- float32 in and
out, weights handed in as given bits, with the recurrent state EXPLICIT
and caller-owned (`Mamba1State`, `Mamba2State`: plain NumPy arrays that
round-trip byte for byte). Prefill, continuation from a carried state,
`initial_states` and single-token decode all run through the certified
Mojo entry points the lane gates run; `numeric_mode=` selects the
fast / deterministic / identical tier at call time, per instance. The
identity claims are the LANES': three vendors for Mamba-1, one gated
vendor (Apple) for Mamba-2 as of 2026-09-01 -- and this Python path is
UNVERIFIED, RUN OWED until `tests/test_mamba_surface.py` prints.

    import numpy as np
    from mojolearn.mamba import Mamba1Block

    blk = Mamba1Block(weights)            # dict of float32 arrays,
                                          # upstream parameter names
    y = blk.forward(x)                    # (B, L, d_model) -> same shape
    st = blk.allocate_state(batch_size=1)
    for t in range(x.shape[1]):           # decode == prefill, per token
        y_t = blk.step(x[:, t:t+1], st)   # st updated in place
"""

from ._mamba_impl import (
    Mamba1Block,
    Mamba1State,
    Mamba2Block,
    Mamba2State,
)

__all__ = ["Mamba1Block", "Mamba1State", "Mamba2Block", "Mamba2State"]
