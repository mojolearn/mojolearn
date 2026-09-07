# SPDX-License-Identifier: Apache-2.0
"""Public dense Euclidean UMAP fit, embedding and held-out transform API.

The fitted neighbor graph uses CSR storage. Input remains a dense array;
supervised UMAP, alternate metrics and alternate initializers are unsupported.
Numeric-mode contracts and qualification apply to the specific fitted and
transform paths, not to all features of another UMAP implementation.
"""
from ._umap_impl import UMAP

__all__ = ['UMAP']
