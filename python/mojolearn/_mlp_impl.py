# SPDX-License-Identifier: Apache-2.0
"""Bounded 8→16→3 FP32 training composition; host code only packs/validates.

All forward, backward, loss and update arithmetic executes in the existing
IDENTICAL GPU GEMM/training bindings. This is a small fixed architecture,
not autograd, a general neural-network trainer, or a Llama checkpoint format.
Composed cross-vendor identity requires its own retained qualification.

NUMPY-FREE (numpy-free-0.7, DEVIATIONS 2423-2427). Inputs are read through
the buffer protocol (`_buffer.view`): a NumPy array, an `array.array`, a
`mojolearn.Array`. Every array this class hands back -- logits, gradients,
the state snapshot -- is a `mojolearn.Array` (`numpy.asarray` on it is
zero-copy). THE CHECKPOINT BYTES ARE UNCHANGED: the JSON/hex envelope,
its canonical serialization and its SHA-256 are produced from the same
little-endian `<f4` / `<i4` bytes the NumPy spelling emitted
(`_bufcheck.le_bytes`), so a file written by 0.6.x loads here and a file
written here loads there, byte for byte (see `_encode_array`).
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

from . import _backend, _linalg_impl, _training_impl
from ._buffer import (
    addr, addr_ro, all_finite, as_f32_c, as_i32_c, empty, frombytes,
)
from ._bufcheck import (
    flat_view, is_int32, is_integer, is_native_f32, le_bytes, probe,
)

_NAMES = ('weight1', 'bias1', 'weight2', 'bias2')
_SHAPES = ((16, 8), (16,), (3, 16), (3,))
_TOTAL = 195
_MAX_STEP = (1 << 31) - 1
_STATE_SCHEMA = 'mojolearn.small-mlp-trainer.v1'
_FILE_SCHEMA = 'mojolearn.small-mlp-checkpoint.v1'
_FILE_LIMIT = 32768
#: `numpy.finfo(numpy.float32).max`, as a Python float.
_F32_MAX = 3.4028234663852886e+38


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def _schedule(value):
    if not isinstance(value, dict) or not value:
        raise ValueError('SmallMLPTrainer data_schedule must be a nonempty JSON object')
    remaining = [128]

    def visit(item, depth=0):
        remaining[0] -= 1
        if depth > 8 or remaining[0] < 0:
            raise ValueError('SmallMLPTrainer data_schedule is too complex')
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError('SmallMLPTrainer data_schedule keys must be strings')
            if len(item) > 128 or any(len(key) > 2048 for key in item):
                raise ValueError('SmallMLPTrainer data_schedule object is too large')
            for child in item.values():
                visit(child, depth + 1)
        elif type(item) is list:
            if len(item) > 128:
                raise ValueError('SmallMLPTrainer data_schedule list is too large')
            for child in item:
                visit(child, depth + 1)
        elif type(item) is str and len(item) > 2048:
            raise ValueError('SmallMLPTrainer data_schedule string is too large')
        elif type(item) is int and item.bit_length() > 4096:
            raise ValueError('SmallMLPTrainer data_schedule integer is too large')
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ValueError('SmallMLPTrainer data_schedule must contain JSON values')
    visit(value)
    try:
        encoded = _canonical(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError('SmallMLPTrainer data_schedule must be finite JSON') from exc
    if len(encoded) > 2048:
        raise ValueError('SmallMLPTrainer data_schedule exceeds 2048 bytes')
    return json.loads(encoded)


def _is_bool(value):
    # `bool` and NumPy's `bool_` (which is not a `bool` subclass), judged by
    # name so that numpy need not be importable here.
    return isinstance(value, bool) or type(value).__name__ == 'bool_'


def _round_f32(value):
    # DEVIATION 2423: `float(np.float32(value))` is one round-to-nearest-even
    # to binary32 and back, which is exactly what `struct` does.
    return struct.unpack('<f', struct.pack('<f', value))[0]


def _scalar(value, name):
    if _is_bool(value):
        raise ValueError('SmallMLPTrainer ' + name + ' must be a finite float32 scalar')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('SmallMLPTrainer ' + name + ' must be a scalar') from exc
    if not math.isfinite(value) or abs(value) > _F32_MAX:
        raise ValueError('SmallMLPTrainer ' + name + ' must be finite float32')
    return _round_f32(value)


def _config(lr, betas, eps, weight_decay):
    try:
        beta1, beta2 = betas
    except (TypeError, ValueError) as exc:
        raise ValueError('SmallMLPTrainer betas must contain two scalars') from exc
    result = {name: _scalar(value, name) for name, value in
              (('lr', lr), ('beta1', beta1), ('beta2', beta2),
               ('eps', eps), ('weight_decay', weight_decay))}
    if result['lr'] <= 0 or result['eps'] <= 0 or result['weight_decay'] < 0:
        raise ValueError('SmallMLPTrainer requires lr > 0, eps > 0 and weight_decay >= 0')
    if not 0 <= result['beta1'] < 1 or not 0 <= result['beta2'] < 1:
        raise ValueError('SmallMLPTrainer betas must remain in [0, 1) in float32')
    return result


def _array(value, shape, name, dtype='<f4'):
    """An OWNED, C-contiguous, exactly-shaped copy of `value` as a
    `mojolearn.Array` (DEVIATION 2424). The dtype is judged from the
    buffer's format and REFUSED rather than converted, as the NumPy
    spelling refused a dtype mismatch; `_buffer.as_f32_c` / `as_i32_c`
    then supply the layout, and a borrowed buffer is copied so that the
    trainer never aliases caller memory."""
    kind = 'float32' if dtype == '<f4' else 'int32'
    try:
        pb = probe(value)
    except TypeError:
        raise TypeError('SmallMLPTrainer ' + name + ' must be a ' + kind
                        + ' array (a NumPy array, an array.array or a mojolearn Array)') from None
    ok = is_native_f32(pb.format) if dtype == '<f4' else is_int32(pb.format, pb.itemsize)
    if not ok:
        raise TypeError('SmallMLPTrainer ' + name + ' must be a ' + kind
                        + ' array (a NumPy array, an array.array or a mojolearn Array)')
    if pb.shape != shape:
        raise ValueError('SmallMLPTrainer ' + name + ' must have shape ' + repr(shape))
    convert = as_f32_c if dtype == '<f4' else as_i32_c
    arr, copied = convert(value, ndim=len(shape), name=name)
    if not copied:
        arr = arr.copy()
    if dtype == '<f4' and not all_finite(arr):
        raise ValueError('SmallMLPTrainer ' + name + ' must be finite')
    return arr


def _batch(value):
    try:
        pb = probe(value)
    except TypeError:
        raise ValueError('SmallMLPTrainer X must have shape (batch, 8)') from None
    if pb.ndim != 2 or pb.shape[1] != 8:
        raise ValueError('SmallMLPTrainer X must have shape (batch, 8)')
    if not 1 <= pb.shape[0] <= 256:
        raise ValueError('SmallMLPTrainer batch must be in [1, 256]')
    return _array(value, pb.shape, 'X')


def _targets(value, rows):
    try:
        pb = probe(value)
    except TypeError:
        raise TypeError('SmallMLPTrainer targets must be an integer array') from None
    if not is_integer(pb.format):
        raise TypeError('SmallMLPTrainer targets must be an integer array')
    if pb.shape != (rows,):
        raise ValueError('SmallMLPTrainer targets must have shape (batch,) with classes 0..2')
    try:
        arr, copied = as_i32_c(value, ndim=1, name='targets')
    except OverflowError:
        # DEVIATION 2467: a label outside int32 is a bad class, not a
        # conversion accident; the documented refusal is this ValueError.
        raise ValueError('SmallMLPTrainer targets must have shape (batch,) with classes 0..2') from None
    if not copied:
        arr = arr.copy()
    # A C-level scan of at most 256 labels (the permitted O(rows) loop).
    labels = flat_view(arr, 'i')
    if min(labels) < 0 or max(labels) >= 3:
        raise ValueError('SmallMLPTrainer targets must have shape (batch,) with classes 0..2')
    return arr


def _require_mode():
    if _backend.default_mode() != 'identical' or _backend.numeric_mode() != 'identical':
        raise RuntimeError('SmallMLPTrainer requires process-selected IDENTICAL mode; select it before use')


def _optimizer(parameters, config, state=None):
    opt = _training_impl.AdamW(
        parameters, lr=config['lr'], betas=(config['beta1'], config['beta2']),
        eps=config['eps'], weight_decay=config['weight_decay'], numeric_mode='identical')
    if state is not None:
        opt.load_state_dict({'t': state['step'], 'exp_avg': state['m'].copy(),
                             'exp_avg_sq': state['v'].copy(),
                             'buf_initialized': state['flags'].copy()})
    return opt


def _state(weights, opt, config, schedule):
    return dict(schema=_STATE_SCHEMA, architecture=[8, 16, 3], numeric_mode='identical',
                parameter_order=list(_NAMES),
                weights={name: array.copy() for name, array in zip(_NAMES, weights)},
                optimizer=dict(kind='AdamW', step=int(opt.t), m=opt.exp_avg.copy(),
                               v=opt.exp_avg_sq.copy(), flags=opt.buf_initialized.copy()),
                config=dict(config), data_schedule=_schedule(schedule))


def _validate_state(state):
    required = {'schema', 'architecture', 'numeric_mode', 'parameter_order',
                'weights', 'optimizer', 'config', 'data_schedule'}
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError('SmallMLPTrainer state has missing or unknown fields')
    if (state['schema'] != _STATE_SCHEMA or state['architecture'] != [8, 16, 3]
            or state['numeric_mode'] != 'identical' or state['parameter_order'] != list(_NAMES)):
        raise ValueError('SmallMLPTrainer state schema/architecture/mode/order mismatch')
    if not isinstance(state['weights'], dict) or set(state['weights']) != set(_NAMES):
        raise ValueError('SmallMLPTrainer state requires all four named parameters')
    weights = [_array(state['weights'][name], shape, name) for name, shape in zip(_NAMES, _SHAPES)]
    cfg = state['config']
    if not isinstance(cfg, dict) or set(cfg) != {'lr', 'beta1', 'beta2', 'eps', 'weight_decay'}:
        raise ValueError('SmallMLPTrainer state optimizer configuration mismatch')
    config = _config(cfg['lr'], (cfg['beta1'], cfg['beta2']), cfg['eps'], cfg['weight_decay'])
    opt = state['optimizer']
    if not isinstance(opt, dict) or set(opt) != {'kind', 'step', 'm', 'v', 'flags'} or opt['kind'] != 'AdamW':
        raise ValueError('SmallMLPTrainer state optimizer mismatch')
    if _is_bool(opt['step']):
        raise ValueError('SmallMLPTrainer step must be an integer')
    try:
        step = operator.index(opt['step'])
    except TypeError as exc:
        raise ValueError('SmallMLPTrainer step must be an integer') from exc
    if not 0 <= step <= _MAX_STEP:
        raise ValueError('SmallMLPTrainer step is outside its supported range')
    moments = dict(step=step, m=_array(opt['m'], (_TOTAL,), 'm'),
                   v=_array(opt['v'], (_TOTAL,), 'v'),
                   flags=_array(opt['flags'], (4,), 'flags', '<i4'))
    # DEVIATION 2425: `np.any(v < 0)` and the binary-flags test are C-level
    # min/max scans over 195 floats and 4 ints.
    flags = flat_view(moments['flags'], 'i')
    if min(flat_view(moments['v'], 'f')) < 0 or min(flags) < 0 or max(flags) > 1:
        raise ValueError('SmallMLPTrainer state requires nonnegative v and binary flags')
    return weights, moments, config, _schedule(state['data_schedule'])


class SmallMLPTrainer:
    """Fixed FP32 8→16→3 ReLU, mean cross-entropy, AdamW trainer.

    Supply weight1 (16,8), bias1 (16,), weight2 (3,16), bias2 (3,) as
    float32 buffers (NumPy arrays or mojolearn Arrays). Parameters and
    inputs are copied. Batches contain 1..256 rows and labels 0..2. All
    neural arithmetic runs on the GPU, in process-selected IDENTICAL mode;
    this class never changes that mode. There is no CPU fallback. Every
    array returned is a `mojolearn.Array`.

    data_schedule is a bounded JSON descriptor supplied by the caller (for
    example dataset hash, sample order and batch size). It is retained exactly
    with the step counter, but this trainer does not fetch or reorder data.
    The caller must feed the corresponding next batch after restoring state.

    train_step returns pre-update loss/logits and all four parameter gradients,
    plus input_grad when requested. No clipping, dropout, mixed precision,
    automatic differentiation or configurable layer shapes are supported.
    State changes are serialized and published only after the whole step
    succeeds. A checkpoint is a separate bounded MLP format, not a Llama file.
    """

    def __init__(self, weight1, bias1, weight2, bias2, *, data_schedule,
                 lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        weights = [_array(value, shape, name) for value, shape, name in
                   zip((weight1, bias1, weight2, bias2), _SHAPES, _NAMES)]
        config = _config(lr, betas, eps, weight_decay)
        schedule = _schedule(data_schedule)
        _require_mode()
        self._lock = threading.RLock()
        self._config = config
        self._schedule = schedule
        self._opt = _optimizer(weights, config)

    @property
    def step_(self):
        with self._lock:
            return int(self._opt.t)

    @property
    def weights_(self):
        return self.state_dict()['weights']

    def state_dict(self):
        """Return an independent snapshot of every resumable state cell."""
        with self._lock:
            return _state(self._opt.params, self._opt, self._config, self._schedule)

    def load_state_dict(self, state):
        """Validate/copy the complete snapshot before changing any live state."""
        with self._lock:
            weights, moments, config, schedule = _validate_state(state)
            _require_mode()
            replacement = _optimizer(weights, config, moments)
            self._opt, self._config, self._schedule = replacement, config, schedule
        return self

    @staticmethod
    def _binding():
        _require_mode()
        _linalg_impl.require_identical()
        binding = _training_impl._load('identical')
        for name in ('mlp_bias_activation', 'mlp_relu_backward', 'mlp_sum_rows'):
            if not callable(getattr(binding, name, None)):
                raise ImportError('SmallMLPTrainer requires updated training binding: missing ' + name)
        return binding

    @staticmethod
    def _matmul(a, b, **kwargs):
        _require_mode()
        result = _linalg_impl.matmul(a, b, identical=True, **kwargs)
        # DEVIATION 2426: the product is a mojolearn.Array; finiteness via
        # the native `all_finite` helper.
        if result.dtype != '<f4' or not all_finite(result):
            raise RuntimeError('SmallMLPTrainer GEMM returned an invalid result')
        return result

    @staticmethod
    def _bias(binding, values, bias, relu):
        result = empty(values.shape, '<f4')
        written = binding.mlp_bias_activation(
            addr_ro(values, name='values'), addr_ro(bias, name='bias'),
            addr(result, name='result'),
            [values.shape[0], values.shape[1], int(relu)])
        if written != result.size or not all_finite(result):
            raise RuntimeError('SmallMLPTrainer bias/activation returned an invalid result')
        return result

    @staticmethod
    def _sum(binding, values):
        result = empty((values.shape[1],), '<f4')
        written = binding.mlp_sum_rows(addr_ro(values, name='values'),
                                       addr(result, name='result'), list(values.shape))
        if written != result.size or not all_finite(result):
            raise RuntimeError('SmallMLPTrainer bias gradient returned an invalid result')
        return result

    def _forward(self, x, weights, binding):
        w1, b1, w2, b2 = weights
        activation = self._bias(binding, self._matmul(x, w1, transpose_b=True), b1, True)
        logits = self._bias(binding, self._matmul(activation, w2, transpose_b=True), b2, False)
        return activation, logits

    def predict_logits(self, X):
        """Return independent FP32 logits from the current weights."""
        with self._lock:
            x = _batch(X)
            binding = self._binding()
            _, result = self._forward(x, self._opt.params, binding)
            _require_mode()
            return result

    def train_step(self, X, targets, *, return_input_grad=False):
        with self._lock:
            x = _batch(X)
            y = _targets(targets, len(x))
            if type(return_input_grad) is not bool:
                raise TypeError('SmallMLPTrainer return_input_grad must be a bool')
            weights, moments, config, schedule = _validate_state(self.state_dict())
            if moments['step'] == _MAX_STEP:
                raise ValueError('SmallMLPTrainer step counter is exhausted')
            binding = self._binding()
            working = _optimizer(weights, config, moments)
            activation, logits = self._forward(x, weights, binding)
            loss, dlogits = _training_impl.cross_entropy(
                logits, y, reduction='mean', return_grad=True, numeric_mode='identical')
            if not math.isfinite(loss):
                raise RuntimeError('SmallMLPTrainer loss is not finite')
            dlogits = _array(dlogits, (len(x), 3), 'logit gradient')
            dw2 = self._matmul(dlogits, activation, transpose_a=True)
            db2 = self._sum(binding, dlogits)
            incoming = self._matmul(dlogits, weights[2])
            dhidden = empty(activation.shape, '<f4')
            written = binding.mlp_relu_backward(
                addr_ro(activation, name='activation'), addr_ro(incoming, name='incoming'),
                addr(dhidden, name='dhidden'), list(activation.shape))
            if written != dhidden.size or not all_finite(dhidden):
                raise RuntimeError('SmallMLPTrainer ReLU gradient returned an invalid result')
            dw1 = self._matmul(dhidden, x, transpose_a=True)
            db1 = self._sum(binding, dhidden)
            grads = [_array(value, shape, name + ' gradient') for value, shape, name in
                     zip((dw1, db1, dw2, db2), _SHAPES, _NAMES)]
            input_grad = self._matmul(dhidden, weights[0]) if return_input_grad else None
            if input_grad is not None and input_grad.shape != x.shape:
                raise RuntimeError('SmallMLPTrainer input gradient shape mismatch')
            working.step(grads)
            # Validation and all potentially allocating result construction
            # precede the single live optimizer-pointer publication.
            _validate_state(_state(weights, working, config, schedule))
            if working.t != moments['step'] + 1:
                raise RuntimeError('SmallMLPTrainer optimizer did not advance exactly one step')
            _require_mode()
            result = dict(step=int(working.t), loss=float(loss), logits=logits.copy(),
                          gradients={name: value.copy() for name, value in zip(_NAMES, grads)})
            if return_input_grad:
                result['input_grad'] = input_grad.copy()
            self._opt = working
            return result

    def save_checkpoint(self, path):
        """Atomically save a canonical, bounded, no-pickle MLP checkpoint."""
        state = self.state_dict()
        payload = _encode_state(state)
        envelope = dict(schema=_FILE_SCHEMA, payload=payload,
                        payload_sha256=hashlib.sha256(_canonical(payload)).hexdigest())
        encoded = _canonical(envelope) + b'\n'
        if len(encoded) > _FILE_LIMIT:
            raise ValueError('SmallMLPTrainer checkpoint exceeds its size bound')
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
        """Restore an MLP snapshot after validating its schema, size and hash."""
        with Path(path).open('rb') as stream:
            encoded = stream.read(_FILE_LIMIT + 1)
        if len(encoded) > _FILE_LIMIT:
            raise ValueError('SmallMLPTrainer checkpoint exceeds its size bound')
        try:
            envelope = json.loads(encoded, object_pairs_hook=_unique_object)
        except (ValueError, UnicodeDecodeError, RecursionError) as exc:
            raise ValueError('SmallMLPTrainer checkpoint is not valid bounded JSON') from exc
        if not isinstance(envelope, dict) or set(envelope) != {'schema', 'payload', 'payload_sha256'}:
            raise ValueError('SmallMLPTrainer checkpoint envelope mismatch')
        if envelope['schema'] != _FILE_SCHEMA:
            raise ValueError('SmallMLPTrainer checkpoint schema mismatch')
        if hashlib.sha256(_canonical(envelope['payload'])).hexdigest() != envelope['payload_sha256']:
            raise ValueError('SmallMLPTrainer checkpoint integrity mismatch')
        state = _decode_state(envelope['payload'])
        weights, _, config, schedule = _validate_state(state)
        result = cls(*weights, data_schedule=schedule, lr=config['lr'],
                     betas=(config['beta1'], config['beta2']), eps=config['eps'],
                     weight_decay=config['weight_decay'])
        return result.load_state_dict(state)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate checkpoint key')
        result[key] = value
    return result


def _encode_array(value):
    """The on-disk tensor descriptor, BYTE-IDENTICAL to the NumPy spelling
    (DEVIATION 2427). `np.asarray(value, dtype='<f4', order='C').tobytes()`
    emitted the C-order LITTLE-ENDIAN bytes whatever the host;
    `_bufcheck.le_bytes` emits the same bytes (a `byteswap` on a big-endian
    host, a no-op elsewhere), so `hex`, the canonical JSON around it and
    the SHA-256 over that JSON are unchanged. `value` is one of this
    module's own Arrays (`<f4` or `<i4`), which is all the NumPy version
    ever saw here too."""
    integer = value.dtype == '<i4'
    dtype = '<i4' if integer else '<f4'
    return dict(dtype=dtype, shape=list(value.shape),
                hex=le_bytes(value, 'i' if integer else 'f').hex())


def _encode_state(state):
    state['weights'] = {name: _encode_array(value) for name, value in state['weights'].items()}
    for key in ('m', 'v', 'flags'):
        state['optimizer'][key] = _encode_array(state['optimizer'][key])
    return state


def _decode_array(value, shape, dtype):
    if (not isinstance(value, dict) or set(value) != {'dtype', 'shape', 'hex'}
            or value['dtype'] != dtype or value['shape'] != list(shape)
            or not isinstance(value['hex'], str)):
        raise ValueError('SmallMLPTrainer checkpoint tensor descriptor mismatch')
    cells = math.prod(shape)
    if len(value['hex']) != cells * 8:
        raise ValueError('SmallMLPTrainer checkpoint tensor length mismatch')
    try:
        raw = bytes.fromhex(value['hex'])
    except ValueError as exc:
        raise ValueError('SmallMLPTrainer checkpoint tensor is not hexadecimal') from exc
    if len(raw) != cells * 4:
        raise ValueError('SmallMLPTrainer checkpoint tensor byte count mismatch')
    # `np.frombuffer(raw, dtype='<f4').astype(native, copy=True)`:
    # `_buffer.frombytes` reads the little-endian bytes into a fresh
    # native Array (DEVIATION 2427).
    return frombytes(raw, dtype, tuple(shape))


def _decode_state(payload):
    if not isinstance(payload, dict):
        raise ValueError('SmallMLPTrainer checkpoint payload must be an object')
    # Bound expected descriptors before looking at any caller-controlled shape.
    try:
        if not isinstance(payload['weights'], dict) or set(payload['weights']) != set(_NAMES):
            raise ValueError('SmallMLPTrainer checkpoint parameter registry mismatch')
        payload['weights'] = {name: _decode_array(payload['weights'][name], shape, '<f4')
                              for name, shape in zip(_NAMES, _SHAPES)}
        for key in ('m', 'v', 'flags'):
            payload['optimizer'][key] = _decode_array(
                payload['optimizer'][key], (4,) if key == 'flags' else (_TOTAL,),
                '<i4' if key == 'flags' else '<f4')
    except (KeyError, TypeError) as exc:
        raise ValueError('SmallMLPTrainer checkpoint state is incomplete') from exc
    return payload
