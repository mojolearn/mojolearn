# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GPU binding's resident eigendecomposition for IncrementalPCA
(x_decomp/ipca_owner.mojo): `x_decomp_dev_eigh`'s typed body, present when
the build registers that entry (IDN_EIGH_RESIDENT), as `_Kit.eigh` chose."""
from x_decomp.cells import F32Ptr
from x_decomp.ipca_owner import IpcaDev
from x_decomp.resident import IDN_EIGH_RESIDENT, dev_eigh_run


struct DevIpcaDev(IpcaDev):
    @staticmethod
    def has_eigh() -> Bool:
        return IDN_EIGH_RESIDENT

    @staticmethod
    def eigh(a: Int, w: F32Ptr, v: F32Ptr, n: Int, uplo: Int) raises:
        _ = dev_eigh_run(a, w, v, n, uplo)
