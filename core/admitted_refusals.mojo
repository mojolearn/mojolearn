# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The one wording of a non-finite input refusal, in a module with no
imports, so a provider door can raise it without importing the whole
admission path (lane m3-linalg-and-housekeeping, 2026-11-09). The refusals
of a deferred finite status word and of a device fold's status words live
here too, so a Python-free RUN module raises them with the same text. Moved
from bindings/admitted_finite_wording.mojo to core/ (lane doorsplit-linear,
2026-11-11); that file, bindings/admitted_input_unpack.mojo and
bindings/device_rows_door.mojo re-export these names."""


def admitted_finite_wording(est: String, name: String = "X") -> String:
    """scikit-learn's wording of the non-finite refusal, the ONE text every
    door raises (admitted_finite_refuse, and a family door's cast verdict
    through `admit_family_meta(..., refusal=...)`)."""
    return "Input " + name + " contains NaN or infinity (" + est + ")."


def admitted_finite_refuse(status: Int, est: String, name: String = "X") raises:
    """The refusal of a deferred finite test: `status` is the word the
    kernel entry returned (core/admitted_finite_status.mojo; 0 = every word
    finite). Raises scikit-learn's wording, as admit_x would have."""
    if status != 0:
        raise Error(admitted_finite_wording(est, name))


def door_rows_refuse(finite_word: Int, range_word: Int, est: String, base_rows: Int) raises:
    """The refusal of a fold a device entry gathered (core/msel_device_x.mojo
    status words, 0 = clear): an index out of range (IndexError) or a NaN /
    infinity, in admit_x's wording."""
    if range_word != 0:
        raise Error("IndexError: DeviceRows row index out of range for a base of " + String(base_rows) + " rows")
    if finite_word != 0:
        raise Error(admitted_finite_wording(est))
