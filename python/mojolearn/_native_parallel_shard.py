# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the Python-sharded multi-GPU drivers' native
owner (bindings/parallel_shard_native.mojo, on the native local process
pool). Each driver module registers its fixed classes, entries and contexts
here at its import; each public driver is then one call into it."""
from . import _native_process  # noqa: F401  (connects the native process pool)
from . import _backend
from .tokenizer import _binding

_native_parallel_shard = _binding()._module.parallel_shard_call


def register(**fixed):
    """Add one driver module's fixed tables to the owner, once, at import."""
    # glue: import-time connection of compiled entries, classes and contexts
    _native_parallel_shard('register', dict(fixed, binding_getter=_backend.binding))
