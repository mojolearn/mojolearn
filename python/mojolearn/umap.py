# SPDX-License-Identifier: Apache-2.0
"""UMAP: Uniform Manifold Approximation and Projection.

Provides :class:`UMAP`, which fits a low-dimensional embedding of dense data
and embeds new rows with ``transform``. densMAP, callable metrics and
non-euclidean output metrics are not supported.
"""
from ._umap_impl import UMAP

__all__ = ['UMAP']
