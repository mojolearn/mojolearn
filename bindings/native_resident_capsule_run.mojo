# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# RUN part of bindings/native_resident_capsule.mojo: typed Mojo only (no
# Python imports). The capsule name owner and the word/byte pointer aliases.
from core.shared_slot import SharedSlot


struct _ResidentNames(Defaultable, Movable):
    var capsule_name: String

    def __init__(out self):
        # PyCapsule keeps this name pointer, not a copy: the DSO-global owner
        # is never reassigned and outlives every capsule.
        self.capsule_name = String("mojolearn.decomp_kit.resident.v1")


comptime _RESIDENT_NAMES = SharedSlot[_ResidentNames, "MojolearnDecompResidentNamesV2"]

comptime _ResidentWords = MutPointer[Int, MutUntrackedOrigin]
comptime _ResidentBytes = MutPointer[UInt8, MutUntrackedOrigin]
