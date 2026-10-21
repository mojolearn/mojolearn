# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python entry of the MoE forward on resident tensors
(`sequence/moe_dev.mojo`): PythonObject arguments to typed lists and the
typed result back. Nothing else."""
from std.python import PythonObject

from sequence.moe_dev import moe_forward_dev
from sequence.pyapi import py_ints


def moe_forward_dev_py(handles: PythonObject, ip: PythonObject) raises -> PythonObject:
    """`moe_forward_dev`'s arguments as Python lists. Returns T * D."""
    return PythonObject(moe_forward_dev(py_ints(handles), py_ints(ip)))
