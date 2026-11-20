# SPDX-License-Identifier: Apache-2.0
"""RUN pieces of bindings/ipca_native.mojo: the fitted state field names. Typed Mojo only, no Python
import (tools/python_free_dirs.sh); moved unchanged (lane rs-decomposition,
2026-11-12)."""


def _state_name(index: Int) -> String:
    if index == 0:
        return "components_"
    if index == 1:
        return "singular_values_"
    if index == 2:
        return "mean_"
    if index == 3:
        return "var_"
    return "explained_variance_"
