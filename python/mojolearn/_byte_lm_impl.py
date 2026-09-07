# SPDX-License-Identifier: Apache-2.0
"""Authored, unqualified public wrapper for the fixed two-block byte LM.

This module contains no forward/backward/update arithmetic. All numerical
work belongs to _mojolearn_byte_lm. No learning, performance, mathematical
correctness or cross-vendor identity claim follows from this source alone.

NUMPY-FREE (numpy-free-0.7, DEVIATIONS 2428-2432). Inputs are read through
the buffer protocol (`_buffer.view`): a NumPy array, an `array.array`, a
`mojolearn.Array`. Every array this class hands back -- parameters,
gradients, the state snapshot -- is a `mojolearn.Array` (`numpy.asarray`
on it is zero-copy). THE CHECKPOINT BYTES ARE UNCHANGED: the JSON/hex
envelope, its canonical serialization and its SHA-256 are produced from
the same little-endian `<f4` / `<i4` bytes the NumPy spelling emitted
(`_bufcheck.le_bytes`), so a file written by 0.6.x loads here and a file
written here loads there, byte for byte, and the resume/compare tooling
under tools/ reads the same file (see `save_checkpoint`).
"""
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import struct
import tempfile
import threading

from . import _backend
from ._buffer import (
    addr, addr_ro, all_finite, as_f32_c, as_i32_c, empty, frombytes, full,
    zeros,
)
from ._bufcheck import (
    flat_view, is_int32, is_native_f32, le_bytes, memcopy, probe,
)

PROFILE = 'mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1'
_BLOCK_NAMES = ('norm1_w', 'w_q', 'w_k', 'w_v', 'w_o', 'norm2_w', 'w_gate', 'w_up', 'w_down')
_BLOCK_SHAPES = ((32,), (32, 32), (16, 32), (16, 32), (32, 32), (32,), (64, 32), (64, 32), (32, 64))
PARAMETER_NAMES = ('embed', *('block%d.%s' % (block, name)
                             for block in range(2) for name in _BLOCK_NAMES), 'lm_head')
PARAMETER_SHAPES = ((256, 32), *_BLOCK_SHAPES, *_BLOCK_SHAPES, (256, 32))
_OFFSETS = [0]
for _shape in PARAMETER_SHAPES:
    _OFFSETS.append(_OFFSETS[-1] + math.prod(_shape))
_OFFSETS = tuple(_OFFSETS)
_N = 34944
_EXTENSION = '_mojolearn_byte_lm'
_SCHEMA = 'mojolearn.small-byte-lm-state.v1'
_CHECKPOINT_SCHEMA = 'mojolearn.small-byte-lm-json-checkpoint.v1'
_CHECKPOINT_LIMIT = 2 * 1024 * 1024
#: `numpy.finfo(numpy.float32).max`, as a Python float.
_F32_MAX = 3.4028234663852886e+38


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def _schedule(value):
    if not isinstance(value, dict) or not value:
        raise ValueError('SmallByteLanguageModelTrainer data_schedule must be a nonempty JSON object')
    budget = [256]
    def check(item, depth=0):
        budget[0] -= 1
        if budget[0] < 0 or depth > 8:
            raise ValueError('Byte-LM schedule is too complex')
        if type(item) is dict:
            if len(item) > 128 or any(type(key) is not str or len(key) > 2048 for key in item):
                raise ValueError('Byte-LM schedule keys/count exceed their bounds')
            for child in item.values():
                check(child, depth + 1)
        elif type(item) is list:
            if len(item) > 128:
                raise ValueError('Byte-LM schedule list exceeds its bound')
            for child in item:
                check(child, depth + 1)
        elif type(item) is str:
            if len(item) > 2048:
                raise ValueError('Byte-LM schedule string exceeds its bound')
        elif type(item) is int:
            if item.bit_length() > 4096:
                raise ValueError('Byte-LM schedule integer exceeds its bound')
        elif item is not None and type(item) not in (float, bool):
            raise ValueError('Byte-LM schedule requires JSON values')
    check(value)
    try:
        encoded = _canonical(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError('Byte-LM schedule must be finite JSON') from exc
    if len(encoded) > 16384:
        raise ValueError('Byte-LM schedule exceeds 16384 bytes')
    return json.loads(encoded)


def _is_bool(value):
    # `bool` and NumPy's `bool_` (not a `bool` subclass), judged by name so
    # that numpy need not be importable here.
    return isinstance(value, bool) or type(value).__name__ == 'bool_'


def _round_f32(value):
    # DEVIATION 2428: `float(np.float32(value))` is one round-to-nearest-even
    # to binary32 and back, which is exactly what `struct` does.
    return struct.unpack('<f', struct.pack('<f', value))[0]


def _float32(value, name):
    if _is_bool(value):
        raise ValueError('Byte-LM ' + name + ' must be a finite float32 scalar')
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Byte-LM ' + name + ' must be a scalar') from exc
    if not math.isfinite(result) or abs(result) > _F32_MAX:
        raise ValueError('Byte-LM ' + name + ' must be finite float32')
    return _round_f32(result)


def _configuration(lr, betas, eps, weight_decay):
    try:
        beta1, beta2 = betas
    except (ValueError, TypeError) as exc:
        raise ValueError('Byte-LM betas must contain two scalars') from exc
    values = {name: _float32(value, name) for name, value in
              (('lr', lr), ('beta1', beta1), ('beta2', beta2),
               ('eps', eps), ('weight_decay', weight_decay))}
    if values['lr'] <= 0 or values['eps'] <= 0 or values['weight_decay'] < 0:
        raise ValueError('Byte-LM requires lr > 0, eps > 0 and weight_decay >= 0')
    if not 0 <= values['beta1'] < 1 or not 0 <= values['beta2'] < 1:
        raise ValueError('Byte-LM betas must remain in [0, 1) in float32')
    return dict(values, kind=2, momentum=0.0, dampening=0.0, nesterov=False, max_norm=0.0)


def _array(value, shape, name, dtype='<f4'):
    """An OWNED, C-contiguous, exactly-shaped, finite copy of `value` as a
    `mojolearn.Array` (DEVIATION 2429). The dtype is judged from the
    buffer's format and REFUSED rather than converted, as the NumPy
    spelling refused a dtype mismatch; `_buffer.as_f32_c` / `as_i32_c`
    then supply the layout, and a borrowed buffer is copied so that the
    trainer never aliases caller memory."""
    kind = 'float32' if dtype == '<f4' else 'int32'
    try:
        pb = probe(value)
    except TypeError:
        raise TypeError('Byte-LM ' + name + ' must be a ' + kind
                        + ' array (a NumPy array, an array.array or a mojolearn Array)') from None
    ok = is_native_f32(pb.format) if dtype == '<f4' else is_int32(pb.format, pb.itemsize)
    if not ok:
        raise TypeError('Byte-LM ' + name + ' must be a ' + kind
                        + ' array (a NumPy array, an array.array or a mojolearn Array)')
    if pb.shape != shape:
        raise ValueError('Byte-LM ' + name + ' requires finite shape ' + repr(shape))
    convert = as_f32_c if dtype == '<f4' else as_i32_c
    arr, copied = convert(value, ndim=len(shape), name=name)
    if not copied:
        arr = arr.copy()
    if dtype == '<f4' and not all_finite(arr):
        raise ValueError('Byte-LM ' + name + ' requires finite shape ' + repr(shape))
    return arr


def _parameters(value):
    if isinstance(value, dict):
        if set(value) != set(PARAMETER_NAMES):
            raise ValueError('Byte-LM named parameters must contain the exact 20-tensor registry')
        arrays = [_array(value[name], shape, name) for name, shape in zip(PARAMETER_NAMES, PARAMETER_SHAPES)]
        # `np.concatenate` of the 20 tensors: one memmove each into the
        # flat registry, at the registry's own offsets (DEVIATION 2429).
        flat = empty((_N,), '<f4')
        base = addr(flat, name='parameters')
        for index, array in enumerate(arrays):
            memcopy(base + 4 * _OFFSETS[index], addr_ro(array, name='parameter'), array.nbytes)
        return flat
    return _array(value, (_N,), 'parameters')


def _step(value):
    if _is_bool(value):
        raise ValueError('Byte-LM completed_steps must be an integer')
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError('Byte-LM completed_steps must be an integer') from exc
    if not 0 <= value <= 999999:
        raise ValueError('Byte-LM completed_steps must be in [0, 999999]')
    return value


def _mode():
    if _backend.default_mode() != 'identical' or _backend.numeric_mode() != 'identical':
        raise RuntimeError('SmallByteLanguageModelTrainer requires process-selected IDENTICAL mode')


def _load():
    _mode()
    binding = _backend.binding(_EXTENSION, 'identical')
    if (int(binding.byte_lm_numeric_mode()) != 1
            or str(binding.byte_lm_profile()) != PROFILE
            or str(binding.byte_lm_vendor()) not in ('cuda', 'hip', 'metal')):
        raise RuntimeError('Byte-LM requires the exact native profile, IDENTICAL mode and CUDA/HIP/Metal vendor')
    if not callable(getattr(binding, 'byte_lm_run', None)):
        raise ImportError('Byte-LM binding is missing byte_lm_run; rebuild bindings/build_byte_lm.sh')
    return binding


def _snapshot(state):
    return dict(state, parameters=state['parameters'].copy(), m=state['m'].copy(),
                v=state['v'].copy(), flags=state['flags'].copy(), config=dict(state['config']),
                data_schedule=_schedule(state['data_schedule']),
                parameter_names=list(PARAMETER_NAMES),
                parameter_shapes=[list(shape) for shape in PARAMETER_SHAPES],
                parameter_offsets=list(_OFFSETS))


def _validate_state(value):
    keys = {'schema', 'profile', 'numeric_mode', 'parameter_names', 'parameter_shapes',
            'parameter_offsets', 'parameters', 'm', 'v', 'flags', 'completed_steps',
            'next_batch_index', 'config', 'data_schedule'}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('Byte-LM state has missing or unknown fields')
    if (value['schema'] != _SCHEMA or value['profile'] != PROFILE or value['numeric_mode'] != 'identical'
            or value['parameter_names'] != list(PARAMETER_NAMES)
            or value['parameter_shapes'] != [list(shape) for shape in PARAMETER_SHAPES]
            or value['parameter_offsets'] != list(_OFFSETS)):
        raise ValueError('Byte-LM state profile/registry/mode mismatch')
    cfg = value['config']
    if not isinstance(cfg, dict) or set(cfg) != set(_configuration(1, (.9, .999), 1e-8, .01)):
        raise ValueError('Byte-LM state optimizer configuration mismatch')
    config = _configuration(cfg['lr'], (cfg['beta1'], cfg['beta2']), cfg['eps'], cfg['weight_decay'])
    if (type(cfg['kind']) is not int or cfg['kind'] != 2 or type(cfg['nesterov']) is not bool
            or cfg['nesterov'] or any(_float32(cfg[key], key) != 0 for key in ('momentum', 'dampening', 'max_norm'))):
        raise ValueError('Byte-LM first profile supports AdamW without clipping or SGD options')
    completed = _step(value['completed_steps'])
    if _step(value['next_batch_index']) != completed:
        raise ValueError('Byte-LM schedule cursor must match completed_steps')
    parameters = _array(value['parameters'], (_N,), 'parameters')
    m = _array(value['m'], (_N,), 'm')
    v = _array(value['v'], (_N,), 'v')
    flags = _array(value['flags'], (20,), 'flags', '<i4')
    # DEVIATION 2430: `np.any(v < 0)` and the binary-flags test are C-level
    # min/max scans over the 34,944 second moments and 20 flags.
    flag_view = flat_view(flags, 'i')
    if min(flat_view(v, 'f')) < 0 or min(flag_view) < 0 or max(flag_view) > 1:
        raise ValueError('Byte-LM requires nonnegative second moments and binary flags')
    return dict(value, parameters=parameters, m=m, v=v, flags=flags,
                completed_steps=completed, next_batch_index=completed, config=config,
                data_schedule=_schedule(value['data_schedule']),
                parameter_names=list(PARAMETER_NAMES),
                parameter_shapes=[list(shape) for shape in PARAMETER_SHAPES], parameter_offsets=list(_OFFSETS))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _binding_metadata(binding):
    root = Path(__file__).resolve().parents[2]
    names = ('python/mojolearn/_byte_lm_impl.py', 'python/mojolearn/language_model.py',
             'bindings/_mojolearn_byte_lm.mojo', 'training/byte_lm.mojo')
    inventory = {name: _sha(root / name) for name in names if (root / name).is_file()}
    # Installed wheels may lack native sources; the Python source and exact
    # loaded binding are still identified. Do not claim a full source audit.
    inventory['loaded_python_wrapper'] = _sha(__file__)
    return dict(binding_file=binding.__file__, binding_sha256=_sha(binding.__file__),
                native_profile=str(binding.byte_lm_profile()),
                native_vendor=str(binding.byte_lm_vendor()),
                native_numeric_mode=int(binding.byte_lm_numeric_mode()),
                source_sha256=inventory,
                source_scope='available direct source files; binding SHA identifies the compiled artifact')


class SmallByteLanguageModelTrainer:
    """Authored, unqualified FP32 two-block byte-language-model trainer.

    Fixed B2/L32/DM32/H4/KV2/FF64/V256, no biases/dropout/final norm, untied
    embedding/head. Supply a copied flat FP32[34944] parameter array or the
    exact 20 named row-major tensors exposed by parameter_registry(), as
    float32 buffers (NumPy arrays or mojolearn Arrays). There is no hidden
    initialization, tokenizer, padding, truncation or RNG. Every array
    returned is a `mojolearn.Array`.

    train_step/evaluate require actual int32[2,33] IDs in [0,256). The first
    32 positions predict the next 32; loss is mean cross-entropy over 64
    targets. All arithmetic runs in the native CUDA/HIP/Metal IDENTICAL
    profile. The process must already select IDENTICAL mode.

    data_schedule is a bounded caller-supplied JSON descriptor. Retain corpus
    and actual token-order SHA256, batch offsets, and planned steps there.
    next_batch_index equals completed_steps; callers must provide the matching
    next batch after restoration. This class does not fetch/reorder a corpus.

    Complete state is copied and updates commit only after success. Evaluation
    requires byte-unchanged parameters, moments, flags and counter. Public
    checkpoints use an explicitly separate bounded JSON/hex schema; they are
    not training/checkpoint.mojo's native binary v1 files. Learning, numerical
    correctness and cross-vendor identity remain unqualified in this slice.
    """

    def __init__(self, parameters, *, data_schedule, lr=1e-3, betas=(.9, .999),
                 eps=1e-8, weight_decay=.01):
        flat = _parameters(parameters)
        config = _configuration(lr, betas, eps, weight_decay)
        descriptor = _schedule(data_schedule)
        _mode()
        self._lock = threading.RLock()
        self._runtime = None
        self._runtime_binding = None
        self._state = dict(schema=_SCHEMA, profile=PROFILE, numeric_mode='identical',
                           parameter_names=list(PARAMETER_NAMES),
                           parameter_shapes=[list(shape) for shape in PARAMETER_SHAPES],
                           parameter_offsets=list(_OFFSETS), parameters=flat,
                           m=zeros((_N,), '<f4'), v=zeros((_N,), '<f4'),
                           flags=zeros((20,), '<i4'), completed_steps=0, next_batch_index=0,
                           config=config, data_schedule=descriptor)

    @staticmethod
    def parameter_registry():
        return [{'name': name, 'shape': shape, 'offset': _OFFSETS[index],
                 'size': _OFFSETS[index + 1] - _OFFSETS[index]}
                for index, (name, shape) in enumerate(zip(PARAMETER_NAMES, PARAMETER_SHAPES))]

    @property
    def step_(self):
        with self._lock:
            return self._state['completed_steps']

    @property
    def parameters_(self):
        with self._lock:
            return self._state['parameters'].copy()

    def state_dict(self):
        with self._lock:
            return _snapshot(self._state)

    def load_state_dict(self, state):
        with self._lock:
            replacement = _validate_state(state)
            _mode()
            self._state = replacement
        return self

    def _binding(self):
        binding = _load()
        if binding is not self._runtime_binding:
            self._runtime = _binding_metadata(binding)
            self._runtime_binding = binding
        return binding

    def run_metadata(self):
        """Exact current profile/config/schedule/source/binary runtime witness.

        Reads binding constants only, without a model execution. Call before
        a capture and retain the returned object alongside its raw arrays.
        """
        with self._lock:
            self._binding()
            return dict(json.loads(_canonical(self._runtime)), schema='small-byte-lm.run-metadata.v1',
                        qualification='authored/unqualified', profile=PROFILE,
                        config=dict(self._state['config']), data_schedule=_schedule(self._state['data_schedule']),
                        completed_steps=self.step_, next_batch_index=self.step_)

    def _run(self, ids, train):
        tokens = _array(ids, (2, 33), 'ids', '<i4')
        token_view = flat_view(tokens, 'i')
        if min(token_view) < 0 or max(token_view) >= 256:
            raise ValueError('Byte-LM IDs must be in [0, 256)')
        working = _validate_state(self._state)
        if train and working['completed_steps'] >= 999999:
            raise ValueError('Byte-LM native call step counter is exhausted')
        binding = self._binding()
        inputs = [working['parameters'], working['m'], working['v'], working['flags'], tokens]
        before = tuple(array.tobytes() for array in inputs)
        # DEVIATION 2431: NaN-filled / -1-filled output buffers, so an
        # unwritten cell is detectable, exactly as `np.full` made them.
        nan = float('nan')
        out_p = full((_N,), nan, '<f4')
        out_m = full((_N,), nan, '<f4')
        out_v = full((_N,), nan, '<f4')
        out_flags = full((20,), -1, '<i4')
        out_loss = full((1,), nan, '<f4')
        out_grad = full((_N,), nan, '<f4') if train else None
        cfg = working['config']
        addresses = [*(addr_ro(value, name='input') for value in inputs),
                     addr(out_p, name='out_p'), addr(out_m, name='out_m'),
                     addr(out_v, name='out_v'),
                     addr(out_grad, name='out_grad') if train else 0,
                     addr(out_flags, name='out_flags'), addr(out_loss, name='out_loss')]
        parameters = [int(train), working['completed_steps'], cfg['kind'], cfg['lr'],
                      cfg['beta1'], cfg['beta2'], cfg['eps'], cfg['weight_decay'],
                      cfg['momentum'], cfg['dampening'], int(cfg['nesterov']), cfg['max_norm']]
        completed = binding.byte_lm_run(addresses, parameters)
        expected = working['completed_steps'] + int(train)
        if _is_bool(completed) or not isinstance(completed, int) or completed != expected:
            raise RuntimeError('Byte-LM returned an invalid completed-step counter')
        if before != tuple(array.tobytes() for array in inputs):
            raise RuntimeError('Byte-LM native call changed an input state/token buffer')
        if not all_finite(out_loss):
            raise RuntimeError('Byte-LM returned a nonfinite/unwritten loss')
        candidate = _validate_state(dict(working, parameters=out_p, m=out_m, v=out_v, flags=out_flags,
                                         completed_steps=expected, next_batch_index=expected))
        _mode()
        if not train:
            if any(candidate[key].tobytes() != working[key].tobytes()
                   for key in ('parameters', 'm', 'v', 'flags')):
                raise RuntimeError('Byte-LM evaluation changed full training state')
            return float(out_loss[0])
        gradients = _array(out_grad, (_N,), 'pre-update gradients')
        # An Array slice COPIES (the Array contract), so each named gradient
        # already owns its memory; `reshape` is a C-order view of that copy.
        result = dict(loss=float(out_loss[0]), step=expected, completed_steps=expected,
                      next_batch_index=expected, flat_gradients=gradients,
                      gradients={name: gradients[_OFFSETS[index]:_OFFSETS[index + 1]].reshape(shape)
                                 for index, (name, shape) in enumerate(zip(PARAMETER_NAMES, PARAMETER_SHAPES))})
        self._state = candidate
        return result

    def train_step(self, ids):
        """One mean-CE/AdamW update, with all 20 pre-update gradients returned."""
        with self._lock:
            return self._run(ids, True)

    def evaluate(self, ids):
        """Return mean loss, requiring full training-state byte invariance."""
        with self._lock:
            return self._run(ids, False)

    def save_checkpoint(self, path):
        """Atomically write at most 2 MiB of canonical no-pickle JSON/hex.

        DEVIATION 2432, BYTE COMPATIBILITY: each tensor descriptor is
        `{dtype, shape, hex}` with `hex` the C-order LITTLE-ENDIAN bytes --
        `_bufcheck.le_bytes`, the same bytes `np.asarray(value, dtype='<f4',
        order='C').tobytes()` produced on every host -- so the canonical
        JSON, the payload SHA-256 and the file are identical to 0.6.x's,
        and `tools/byte_lm_state_compare.py` and the resume tooling read
        the same file.
        """
        payload = self.state_dict()
        for key in ('parameters', 'm', 'v', 'flags'):
            value = payload[key]
            integer = key == 'flags'
            dtype = '<i4' if integer else '<f4'
            payload[key] = dict(dtype=dtype, shape=list(value.shape),
                                hex=le_bytes(value, 'i' if integer else 'f').hex())
        envelope = dict(schema=_CHECKPOINT_SCHEMA, payload=payload,
                        payload_sha256=hashlib.sha256(_canonical(payload)).hexdigest())
        encoded = _canonical(envelope) + b'\n'
        if len(encoded) > _CHECKPOINT_LIMIT:
            raise ValueError('Byte-LM checkpoint exceeds 2 MiB')
        path = Path(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.', delete=False) as stream:
                temporary = stream.name
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                os.unlink(temporary)

    @classmethod
    def from_checkpoint(cls, path):
        """Restore this explicit JSON checkpoint schema, never native binary v1."""
        with Path(path).open('rb') as stream:
            encoded = stream.read(_CHECKPOINT_LIMIT + 1)
        return cls.from_checkpoint_bytes(encoded)

    @classmethod
    def from_checkpoint_bytes(cls, encoded):
        """Restore one bounded immutable capture without opening any path.

        Only exact ``bytes`` is accepted: callers must first capture mutable
        buffers themselves, then hash and supply that same immutable object.
        This loader does not establish file provenance or vendor identity.
        It shares every schema, integrity, tensor, optimizer and cursor check
        with from_checkpoint; neither API launches native model operations.
        """
        if type(encoded) is not bytes:
            raise TypeError('Byte-LM checkpoint capture must be immutable bytes')
        if len(encoded) > _CHECKPOINT_LIMIT:
            raise ValueError('Byte-LM checkpoint exceeds 2 MiB')
        try:
            envelope = json.loads(encoded, object_pairs_hook=_unique_object)
        except (ValueError, UnicodeDecodeError, RecursionError) as exc:
            raise ValueError('Byte-LM checkpoint is not valid bounded JSON') from exc
        if (not isinstance(envelope, dict) or set(envelope) != {'schema', 'payload', 'payload_sha256'}
                or envelope['schema'] != _CHECKPOINT_SCHEMA):
            raise ValueError('Byte-LM checkpoint schema mismatch')
        payload = envelope['payload']
        if hashlib.sha256(_canonical(payload)).hexdigest() != envelope['payload_sha256']:
            raise ValueError('Byte-LM checkpoint integrity mismatch')
        if not isinstance(payload, dict):
            raise ValueError('Byte-LM checkpoint payload must be an object')
        for key in ('parameters', 'm', 'v', 'flags'):
            value = payload.get(key)
            cells, dtype = (20, '<i4') if key == 'flags' else (_N, '<f4')
            if (not isinstance(value, dict) or set(value) != {'dtype', 'shape', 'hex'}
                    or value['dtype'] != dtype or value['shape'] != [cells]
                    or not isinstance(value['hex'], str) or len(value['hex']) != cells * 8):
                raise ValueError('Byte-LM checkpoint tensor descriptor mismatch')
            try:
                raw = bytes.fromhex(value['hex'])
            except ValueError as exc:
                raise ValueError('Byte-LM checkpoint tensor is not hexadecimal') from exc
            if len(raw) != cells * 4:
                raise ValueError('Byte-LM checkpoint tensor byte count mismatch')
            # `np.frombuffer(raw, dtype).astype(native, copy=True)`:
            # `_buffer.frombytes` reads the little-endian bytes into a
            # fresh native Array (DEVIATION 2432).
            payload[key] = frombytes(raw, dtype, (cells,))
        state = _validate_state(payload)
        cfg = state['config']
        result = cls(state['parameters'], data_schedule=state['data_schedule'], lr=cfg['lr'],
                     betas=(cfg['beta1'], cfg['beta2']), eps=cfg['eps'], weight_decay=cfg['weight_decay'])
        return result.load_state_dict(state)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate byte-LM checkpoint key')
        result[key] = value
    return result
