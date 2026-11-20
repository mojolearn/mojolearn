# SPDX-License-Identifier: Apache-2.0
"""RUN pieces of bindings/tsvd_native.mojo: the minimum-shape refusal text. Typed Mojo only, no Python
import (tools/python_free_dirs.sh); moved unchanged (lane rs-decomposition,
2026-11-12)."""


def _too_small(rows: Int, cols: Int) -> String:
    # scikit-learn's check_array wording for a block below 2 x 2.
    if rows < 2:
        return ("Found array with " + String(rows) + " sample(s) (shape=(" + String(rows) + ", " + String(cols)
                + ")) while a minimum of 2 is required by TruncatedSVD.")
    return ("Found array with " + String(cols) + " feature(s) (shape=(" + String(rows) + ", " + String(cols)
            + ")) while a minimum of 2 is required by TruncatedSVD.")
