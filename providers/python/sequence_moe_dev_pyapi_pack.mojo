# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python entry of the MoE forward on resident tensors
(`sequence/moe_dev.mojo`): PythonObject arguments to typed lists and the
typed result back. Nothing else."""
# The PACK half of the former `providers/python/sequence_moe_dev_pyapi.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `providers/python/sequence_moe_dev_pyapi_unpack.mojo`.
from std.python import PythonObject

from sequence.moe_dev import moe_forward_dev
from providers.python.sequence_pyapi_unpack import py_ints
from providers.python.sequence_seq_tensor_pyapi_pack import seq_tensor_handles_list
from sequence.sequence_seq_tensor_run import seq_tensor_alloc_slots, seq_tensor_free_made


def moe_forward_dev_py(handles: PythonObject, ip: PythonObject) raises -> PythonObject:
    """`moe_forward_dev`'s arguments as Python lists. Returns T * D."""
    return PythonObject(moe_forward_dev(py_ints(handles), py_ints(ip)))
