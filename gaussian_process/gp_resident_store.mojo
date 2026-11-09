# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GP binding's store of device-resident fitted arrays (lane
ne1-a-gp-svm; core/fitted_resident_store.mojo). Its own module so that the
regressor (gaussian_process/gp_resident_state.mojo) and the one-vs-rest
classifier (gaussian_process/gpc_ovr.mojo) share one store without an import
cycle. One store per numeric tier, next to that tier's GP context."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.fitted_resident_store import FittedResidentStore
from core.shared_slot import SharedSlot

comptime GP_RESIDENT_STORE = SharedSlot[FittedResidentStore, ( "MojolearnGpResidentFittedIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnGpResidentFittedFastV1" )]


def gp_resident_store() raises -> UnsafePointer[FittedResidentStore, MutAnyOrigin]:
    return GP_RESIDENT_STORE.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()
