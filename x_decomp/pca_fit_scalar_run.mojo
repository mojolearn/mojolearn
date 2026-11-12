# SPDX-License-Identifier: Apache-2.0
"""RUN piece of providers/python/pca_fit_scalar_pyapi.mojo: the typed record
of one captured scalar array (rank, count, dtype kind and char). Typed Mojo
only, no Python import (tools/python_free_dirs.sh); moved unchanged (lane
rs-decomposition, 2026-11-12)."""


struct PCAFitScalarArray(Copyable, Movable):
    var ndim: Int
    var count: Int
    var kind: String
    var char: String

    def __init__(out self, ndim: Int, count: Int, kind: String, char: String):
        self.ndim = ndim
        self.count = count
        self.kind = kind
        self.char = char
