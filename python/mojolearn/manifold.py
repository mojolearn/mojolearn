# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Manifold learning: spectral embedding (Laplacian eigenmaps) and UMAP."""
from ._spectral_impl import SpectralEmbedding, spectral_embedding
from ._umap_impl import UMAP

__all__ = ["SpectralEmbedding", "UMAP", "spectral_embedding"]


# scikit-learn keeps more estimators in this module than it defines; they resolve from
# the package at first use (`_sklearn_modules`), so `from mojolearn.manifold import ...`
# takes the same names as scikit-learn's module.
from ._sklearn_modules import LAYOUT as _SKLEARN_LAYOUT, lazy_exports as _lazy_exports  # noqa: E402
__getattr__ = _lazy_exports("manifold")
__all__ += _SKLEARN_LAYOUT["manifold"]
