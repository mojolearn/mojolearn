# SPDX-License-Identifier: Apache-2.0
"""RUN pieces of bindings/ipca_owner_boundary.mojo: the prepared-owner slot names. Typed Mojo only, no Python
import (tools/python_free_dirs.sh); moved unchanged (lane rs-decomposition,
2026-11-12)."""


def _slot(index: Int) -> String:
    if index == 0:
        return "ipca_components_native_v1"
    if index == 1:
        return "ipca_singular_native_v1"
    if index == 2:
        return "ipca_mean_native_v1"
    if index == 3:
        return "ipca_variance_native_v1"
    return "ipca_explained_native_v1"
