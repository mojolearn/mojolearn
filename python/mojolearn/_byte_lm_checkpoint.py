# SPDX-License-Identifier: Apache-2.0
"""Stream and verify byte-LM checkpoints with complete optimizer state."""
from ._backend import load_host_module as _load_host_module
from ._array import Array as _Array

MAGIC = b'MOJOLEARN-BYTE-LM\x01\n'
SCHEMA = 'mojolearn.byte-lm-stream.v1'
HEADER_LIMIT = 1024 * 1024
CHUNK = 1024 * 1024
ARRAYS = (('parameters', '<f4'), ('m', '<f4'), ('v', '<f4'), ('flags', '<i4'))
_native_checkpoint = _load_host_module('_mojolearn_tokenizer_host').cross_vendor_call
_native_checkpoint('byte_checkpoint_initialize', (_Array,))


def save(path, state):
    """Atomically write raw array words and metadata; return the file SHA-256."""
    return _native_checkpoint('byte_checkpoint_save', (path, state))


def load(path):
    """Verify and restore a v1 archive before constructing a model."""
    return _native_checkpoint('byte_checkpoint_load', (path,))
