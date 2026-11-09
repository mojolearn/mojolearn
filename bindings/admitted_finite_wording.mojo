# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The one wording of a non-finite input refusal, in a module with no
imports, so a provider door can raise it without importing the whole
admission path (bindings/admitted_input.mojo re-exports it; lane
m3-linalg-and-housekeeping, 2026-11-09)."""


def admitted_finite_wording(est: String, name: String = "X") -> String:
    """scikit-learn's wording of the non-finite refusal, the ONE text every
    door raises (admitted_finite_refuse, and a family door's cast verdict
    through `admit_family_meta(..., refusal=...)`)."""
    return "Input " + name + " contains NaN or infinity (" + est + ")."
