# SPDX-License-Identifier: Apache-2.0
"""Import-time installation of the shared native operation-lock protocol."""
from types import MethodType
import _thread


class _OperationLock:
    __slots__ = ('_owner',)


_connected_runtime = None


def connect(runtime):
    # Only connection work runs here. Installed methods are compiled instance
    # descriptors, with no Python acquire/release/context-manager wrappers.
    global _connected_runtime
    if _connected_runtime is runtime:
        return _OperationLock
    if _connected_runtime is not None:
        raise RuntimeError('native operation lock already connected to another runtime')
    owner = vars(_OperationLock)['_owner']
    context = {'class': _OperationLock, 'get_owner': owner.__get__,
               'set_owner': owner.__set__, 'legacy_class': _thread.RLock}
    runtime.operation_lock_connect(context)
    # glue: import-time registration of six native protocol descriptors
    for name, entry in (  # glue: fixed import-time native protocol registration
                        ('__init__', 'operation_lock_init'),
                        ('acquire', 'operation_lock_acquire'),
                        ('release', 'operation_lock_release'),
                        ('__enter__', 'operation_lock_enter'),
                        ('__exit__', 'operation_lock_exit'),
                        ('_is_owned', 'operation_lock_owned')):  # glue: fixed native protocol registration
        method = runtime.classical_attribute_descriptor(MethodType(getattr(runtime, entry), context))
        setattr(_OperationLock, name, method)
    _connected_runtime = runtime
    return _OperationLock
