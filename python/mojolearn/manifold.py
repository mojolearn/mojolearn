# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Manifold learning: spectral embedding (Laplacian eigenmaps) and UMAP."""
from ._spectral_impl import SpectralEmbedding, spectral_embedding
from ._umap_impl import UMAP

__all__ = ["SpectralEmbedding", "UMAP", "spectral_embedding"]
