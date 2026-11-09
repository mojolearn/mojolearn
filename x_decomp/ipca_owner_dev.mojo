# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GPU binding's resident eigendecomposition for IncrementalPCA
(x_decomp/ipca_owner.mojo), present when the build registers
`x_decomp_dev_eigh` (IDN_EIGH_RESIDENT), as `_Kit.eigh` chose.

Lane n9-decomp-eigh-families (2026-11-09): the Gram's solve is the executor
op (`DevExec.eigh_enq`, x_decomp/eigh_ops.mojo `eigh_op_sym`, `DevExec.eigh`'s
route and words) into device buffers, and the descending order
(`_reversed_columns`' copies) is one kernel into the kit's device matrices:
w and V never come home. One wait, for the op's status word."""
from x_decomp.ipca_owner import IpcaDev
from x_decomp.resident import IDN_EIGH_RESIDENT, dev_eigh_desc_run


struct DevIpcaDev(IpcaDev):
    @staticmethod
    def has_eigh() -> Bool:
        return IDN_EIGH_RESIDENT

    @staticmethod
    def eigh_desc(a: Int, w: Int, v: Int, n: Int) raises:
        # the shared body (x_decomp/resident.mojo), FastICA / FactorAnalysis's
        # owners reach it too
        dev_eigh_desc_run(a, w, v, n)
