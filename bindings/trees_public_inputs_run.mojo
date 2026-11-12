# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN half of the trees input door (bindings/trees_public_inputs_door.mojo):
the one host finite scan of float32 words and the forest's refusal wording.
No Python import (tools/python_free_dirs.sh). Moved word for word (lane
admitted-x-view, 2026-11-12); the scan splits its words across the host's
cores (core/host_finite_scan.mojo)."""

from core.host_finite_scan import host_all_finite_f32


def trees_finite_f32(address: Int, count: Int) -> Bool:
    """True when the `count` float32 words at `address` are all finite: the
    one host scan (core/host_finite_scan.mojo), SIMD tasks across cores."""
    if count <= 0:
        return True
    return host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address), count)


#: The forests' refusal of a non-finite X (randomforest.py / extratrees.py
#: `_NONFINITE_REFUSAL`).
comptime TREES_NONFINITE = "X contains NaN or infinity; the forest has no missing-value arm"
