# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host column of x_neighbors/kapprox_dev.mojo: the CPU binding exports the
same name. HOST ONLY. The Python layer never takes the resident KernelPCA
path on this binding (`x_neighbors_kpca_resident()` is 0 here)."""
from std.python import PythonObject


def kpca_resident_binding() raises -> PythonObject:
    return PythonObject(0)
