# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mojolearn.mamba`: the Mamba-1, Mamba-2 and Mamba-3 blocks, for
cross-checking.

The public face of `_mamba_impl.py`, which carries every contract detail
on its classes; this module exists because `archive/evidence/mamba/FEATURE_PARITY.md`'s
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
call time, per instance. Python backward is not exposed.

Native and Python evidence have separate scopes. At `718495cd`, Apple,
NVIDIA and AMD matched 54 native backward gradient tensors across five
Mamba-1/2/3 cases; see
`bench/results/e1g/2026-09-05_042552-amd-mamba/cross-device.json`.
At `b715b124`, NVIDIA retained five native cases and 102 Python API
checks; see `bench/results/e1g/2026-09-05_065820-nvidia-mamba/classification.json`.
DigitalOcean AMD MI325X matched those 54 NVIDIA native tensors at the
same source; see
`bench/results/e1/2026-09-05_111524-mojolearn-e2-amd/comparisons.json`.
The baseline AMD Python path faulted (exit 134). The state-allocation fix
`6dc93269` passed 102 Apple IDENTICAL API checks, retained in
`bench/results/mamba/2026-09-05-state-allocation-fix/metadata.json`;
durable NVIDIA/AMD qualification of that fix is pending. A supplemental
AMD pass whose artifacts were not retained does not close that gate.
These are fixture-scoped source checks, not universal identity or Linux
wheel certification. The released macOS 0.5.0 wheel predates the fix.


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
