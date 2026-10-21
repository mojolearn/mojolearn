# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of the FAST resident dictionary update
(x_decomp/dict_fast.mojo): the door the GPU binding registers. It unpacks ids
and integers and calls one typed entry in x_decomp/dict_fast.mojo; no data
work happens here."""
from std.python import PythonObject

from x_decomp.dict_fast import dict_update_dev
from x_decomp.resident_pyapi import _id, _n


def dev_dict_update_py(
    src: PythonObject, a: PythonObject, b: PythonObject, dst: PythonObject, p: PythonObject
) raises -> PythonObject:
    """dst (nc x m) = src updated atom by atom (see x_decomp/dict_fast.mojo);
    p = [nc, m]. Every A[j, j] must be > 1e-6 (the caller checks)."""
    return PythonObject(dict_update_dev(_id(src), _id(a), _id(b), _id(dst), _n(p, 0), _n(p, 1)))
