# SPDX-License-Identifier: Apache-2.0
"""Installed ABI connection for the private native local-worker protocol."""
import os
import sys

from . import _backend
from . import lowbit as _lowbit
from ._array import Array
from .tokenizer import _binding

_runtime = _binding()._module
# A spawned worker imports only this module (`python -m mojolearn._native_process`),
# never `models.causal_lm`, which is where the parent connects the state codec's
# storage classes. Connect them here, once, so the worker can unpack what the
# parent packed ("codec: native storage classes are not connected", smoke 91d5).
# The connection is idempotent for the same classes (bindings/native_codec_door.mojo).
_runtime.native_codec_call("initialize", (Array, _lowbit.BF16Weight, _lowbit.Int8Weight))
_native_process = _runtime.native_process_call
_native_process("initialize", (sys.executable, os.environ._data, _backend.vendor()))

if __name__ == "__main__":
    _native_process("worker_echo", sys.argv)
