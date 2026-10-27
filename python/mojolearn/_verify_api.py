"""Bounded public endpoint verification. Test control, never product execution."""
__mojolearn_user_boundary__ = True

import hashlib
import importlib
import json
import struct
import traceback
from pathlib import Path


def resolve(public):
    module, _, name = public.rpartition('.')
    return getattr(importlib.import_module(module), name)


def encode(value, seen=None):
    """Exact numerical bytes, structural metadata, no object addresses."""
    import numpy as np
    seen = set() if seen is None else set(seen)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return {'float64': struct.pack('<d', value).hex()}
    if type(value) is bytes:
        return {'bytes': value.hex()}
    if id(value) in seen:
        return {'cycle': type(value).__name__}
    seen.add(id(value))
    if isinstance(value, (np.ndarray, np.generic)) or type(value).__name__ == 'Array':
        array = np.asarray(value)
        if array.dtype.kind == 'O':
            return {'shape': list(array.shape), 'items': encode(array.tolist(), seen)}
        raw = array.tobytes(order="C")
        return {'shape': list(array.shape), 'dtype': array.dtype.str, 'bytes': raw.hex(),
                'sha256': hashlib.sha256(raw).hexdigest()}
    if isinstance(value, dict):
        return {str(k): encode(v, seen) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [encode(v, seen) for v in value]
    if isinstance(value, slice):
        return {'slice': encode((value.start, value.stop, value.step), seen)}
    if isinstance(value, type) or callable(value):
        return {'callable': value.__module__ + '.' + getattr(value, '__qualname__', type(value).__name__)}
    # Parameter objects are compared through their public configuration, not
    # their address or private vendor-specific handles.
    if hasattr(value, 'get_params'):
        return {'type': type(value).__name__, 'params': encode(value.get_params(deep=False), seen)}
    raise TypeError('no explicit consumed-output encoding for ' + type(value).__name__)


class Recorder:
    def __init__(self, path, targets):
        self.path = Path(path)
        self.targets = set(targets)
        self.rows = {}

    def check(self, endpoint, operation, *, kind='numerical'):
        if endpoint not in self.targets:
            return None
        if endpoint in self.rows:
            raise ValueError('duplicate endpoint execution would overwrite evidence: ' + endpoint)
        try:
            result = operation()
            if isinstance(result,dict) and result.get('numerical_execution') is False:
                kind='availability contract'
            self.rows[endpoint] = dict(status='captured', kind=kind, output=encode(result))
        except Exception as error:
            self.rows[endpoint] = dict(status='failed', kind=kind, error=type(error).__name__ + ': ' + str(error),
                                       traceback=traceback.format_exc())
        with self.path.with_suffix('.jsonl').open('a') as stream:
            stream.write(json.dumps({'endpoint': endpoint, **self.rows[endpoint]}, sort_keys=True) + '\n')
        return self.rows[endpoint]

    def save(self):
        self.path.write_text(json.dumps(dict(format='mojolearn.api-endpoints.v1',
            expected=sorted(self.targets), endpoints=self.rows,
            pending=sorted(self.targets-self.rows.keys())), indent=2, sort_keys=True) + '\n')
