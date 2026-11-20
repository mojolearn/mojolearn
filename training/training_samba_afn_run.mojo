# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN half of `providers/python/training_samba_afn_pyapi.mojo` (wave 8 run
split): the typed address casts its doors hand to `training/samba_afn.mojo`.
Typed Mojo only: no Python import."""


def _afn_f32(addr: Int) -> MutPointer[Float32, MutUntrackedOrigin]:
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _afn_i32(addr: Int) -> MutPointer[Int32, MutUntrackedOrigin]:
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)
