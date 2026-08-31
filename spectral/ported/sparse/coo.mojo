# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""`raft/sparse/coo.hpp`'s `COO<T, IdxT, nnz_t>`: rows, cols, vals, n.

Host-resident here (theirs is device-resident, `rmm::device_uvector`s), because
every consumer in this lane reads it on the host before uploading the sorted
arrays once. Nothing else of the struct is ported (`allocate`, `validate_mem`,
the stream plumbing): those are memory management, not algorithm.
"""


@fieldwise_init
struct CooGraph(Copyable, Movable):
    """A COO matrix on the host: `rows`, `cols`, `vals`, all of length
    `nnz`, over `n x n`. Not sorted unless the producer says so."""

    var n: Int
    var rows: List[Int32]
    var cols: List[Int32]
    var vals: List[Float32]

    def nnz(self) -> Int:
        return len(self.vals)
