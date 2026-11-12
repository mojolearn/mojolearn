# SPDX-License-Identifier: Apache-2.0
"""RUN pieces of the decomposition owners' shared helpers
(bindings/decomp_owner_common.mojo): the typed refusals on extents. Typed
Mojo only, no Python import (tools/python_free_dirs.sh); moved unchanged
(lane rs-decomposition, 2026-11-12)."""


def decomp_min_samples(n: Int, d: Int, who: String, minimum: Int = 2) raises:
    """`_min_samples`: scikit-learn's check_array(ensure_min_samples=...) wording."""
    if n < minimum:
        raise Error("Found array with " + String(n) + " sample(s) (shape=(" + String(n) + ", " + String(d)
                    + ")) while a minimum of " + String(minimum) + " is required by " + who + ".")


def decomp_gemm_check(k: Int, k2: Int) raises:
    """`_Kit.mm`'s inner-dimension refusal."""
    if k != k2:
        raise Error("x_decomp: gemm inner dimensions " + String(k) + " and " + String(k2) + " differ")
