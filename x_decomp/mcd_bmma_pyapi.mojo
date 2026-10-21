# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/mcd_bmma.mojo: the PythonObject <-> typed-value
door that `bindings/_mojolearn_x_decomp.mojo` registers under the same name.
It unpacks one integer and calls one typed entry; no data work happens here."""
from std.python import PythonObject

from x_decomp.mcd_bmma import mcd_cov_reach_run


def mcd_cov_reach_py(which: PythonObject) raises -> PythonObject:
    return PythonObject(mcd_cov_reach_run(Int(py=which)))
