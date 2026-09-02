# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mojolearn.mamba`: the Mamba-1, Mamba-2 and Mamba-3 blocks, for
cross-checking.

The public face of `_mamba_impl.py`, which carries every contract detail
on its classes; this module exists because `mamba/FEATURE_PARITY.md`'s
consumer table names `mojolearn.mamba` as the surface a downstream user
imports, and because the block classes are not estimators -- they have no
`fit`, so they live beside the sklearn-shaped names rather than among
them (they ARE also re-exported from `mojolearn` itself).

What is here, in one paragraph. `Mamba1Block`, `Mamba2Block` and
`Mamba3Block` are ONE reference-pinned block each -- norm, mixer,
residual -- float32 in and out, weights handed in as given bits, with
the recurrent state EXPLICIT and caller-owned (`Mamba1State`,
`Mamba2State`, `Mamba3State`: plain NumPy arrays that round-trip byte
for byte). Prefill, continuation from a carried state, `initial_states`
(Mamba-2) / `Input_States` (Mamba-3) and single-token decode all run
through the certified Mojo entry points the lane gates run;
`numeric_mode=` selects the fast / deterministic / identical tier at
call time, per instance. The identity claims are the LANES': three
vendors for Mamba-1; for Mamba-2, Apple + NVIDIA bit-identical and now
Apple + AMD too, the Apple-vs-AMD hash diff having RUN 2026-09-02 and
come back IDENTICAL on all 26 recorded stages (Apple M4 at cd56e8ce
against an AMD MI325X, gfx942, at cb8ea360, with the compiled path
unchanged across those commits) -- All three vendors now agree pairwise
against Apple, but the three columns are NOT all at one commit -- the
NVIDIA column is the 2026-09-01 Apple-vs-RTX-4090 result at that day's
commit and was NOT re-run at cd56e8ce, so a single-commit three-vendor
card is still OWED, and an H100 leg at cd56e8ce is IN FLIGHT to close
exactly that; and for Mamba-3, Apple PASS with the AMD column RED as of 2026-09-02 --
gate (a) failed on an MI325X (gfx942) by one ULP on 1,179 cells over
four stages, NVIDIA owed
(`mamba/IDENTICAL_MAMBA3_CONTRACT.md`'s 2026-09-02 RUN RECORD). The Mamba-1/2
Python path printed green in all three tiers on 2026-09-01 (one box,
one vendor); the Mamba-3 path is UNVERIFIED, RUN OWED until
`tests/test_mamba_surface.py` prints with its arms in.

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
    Mamba3Block,
    Mamba3State,
)

__all__ = [
    "Mamba1Block",
    "Mamba1State",
    "Mamba2Block",
    "Mamba2State",
    "Mamba3Block",
    "Mamba3State",
]
