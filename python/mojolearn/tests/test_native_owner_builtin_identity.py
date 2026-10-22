# SPDX-License-Identifier: Apache-2.0
"""Whole public owner replacement retains native builtin identities."""
import builtins
from types import MappingProxyType
from unittest.mock import patch

import numpy as np
import pytest

from mojolearn._array import Array
from mojolearn._arena_io import DeviceRows
from mojolearn._verify_native_runtime import audit_native_call


@pytest.mark.parametrize('kind', ['dictionary', 'dtype_metadata', 'memoryview', 'complex'])
def test_device_rows_reinitialization_uses_connected_owner_identities(kind):
    original_list = list
    dictionary_items = type({}.items())
    calls = []
    old = Array.from_list([[1., 2.]], '<f4')
    if kind == 'dictionary':
        old._pin = {'values': [1, 2]}
    elif kind == 'dtype_metadata':
        old = np.zeros((1, 2), dtype=np.dtype('f4', metadata={'values': (1, 2)}))
    elif kind == 'memoryview':
        old._pin = memoryview(b'old owner')
    else:
        old._pin = 1 + 2j
    indices = Array.from_list([0], '<i8')
    new = Array.from_list([[3., 4.]], '<f4')
    rows = DeviceRows(old, indices, None)

    if kind in ('dictionary', 'dtype_metadata'):
        def foreign_list(value=()):
            # Auditor setup also builds lists, outside the operation. Only
            # retained-owner metadata is recorded in this explicit hook log;
            # every product Python call remains visible to the auditor.
            if type(value) in (dictionary_items, MappingProxyType):
                calls.append(type(value).__name__)
            return original_list(value)
        replacement = foreign_list
        name = 'list'
    else:
        def foreign(*args, **kwargs):
            calls.append('foreign')
            raise AssertionError('mutable builtin reached owner replacement')
        replacement = foreign
        name = kind
    with patch.object(builtins, name, replacement):
        audit = audit_native_call(rows.__init__, new, indices, None)
    assert audit.assert_native() is None
    assert calls == []
    assert rows.base is new and rows.indices is indices
    assert rows._host is None and rows.shape == (1, 2)
