# SPDX-License-Identifier: Apache-2.0
"""Installed ABI connection for the private native local-worker protocol."""
import os
import sys

from . import _backend
from .tokenizer import _binding

_native_process = _binding()._module.native_process_call
_native_process("initialize", (sys.executable, os.environ._data, _backend.vendor()))

if __name__ == "__main__":
    _native_process("worker_echo", sys.argv)
