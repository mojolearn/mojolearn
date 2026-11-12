# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python entry of LayerNorm on resident tensors
(`sequence/layernorm_dev.mojo`): PythonObject arguments to typed lists and
the typed result back. Nothing else."""
# The PACK half of the former `providers/python/sequence_layernorm_dev_pyapi.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `providers/python/sequence_layernorm_dev_pyapi_unpack.mojo`.
from std.python import PythonObject

from sequence.layernorm_dev import layer_norm_dev
from providers.python.sequence_pyapi_unpack import py_f64s, py_ints
from providers.python.sequence_seq_tensor_pyapi_pack import seq_tensor_handles_list
from sequence.sequence_seq_tensor_run import seq_tensor_alloc_slots, seq_tensor_free_made


def layer_norm_dev_py(handles: PythonObject, addrs: PythonObject, ip: PythonObject,
                      fp: PythonObject) raises -> PythonObject:
    """`layer_norm_dev`'s arguments as Python lists. Returns M * D."""
    return PythonObject(layer_norm_dev(py_ints(handles), py_ints(addrs), py_ints(ip), py_f64s(fp)))
