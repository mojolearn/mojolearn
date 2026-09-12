# SPDX-License-Identifier: Apache-2.0
"""CPU inference for the byte-level decoder language model (DEVIATION 2610).

`LanguageModelInference.from_checkpoint(path)` loads the parameters of a
checkpoint written by `SmallByteLanguageModelTrainer.export_checkpoint` and
runs the forward pass on the CPU through `_mojolearn_byte_lm_host`, which
composes the host FP32 oracles the GPU kernels are gated against.

This module holds no arithmetic. What it promises is what the gate measured:
docs/BYTE_LM_CPU_INFERENCE.md lists the CPUs on which the loss bytes of the
retained Metal, CUDA and HIP captures were reproduced. A CPU not listed there
is not certified, whatever this code returns on it.

The binary is loaded from `mojolearn/host/` by path and not through
`_backend.load_set`, because that selector refuses a binary whose vendor
read-back is not a GPU API, and this one reads back `cpu` by design.
`MOJOLEARN_BYTE_LM_HOST_BINARY` names a different file, which is how the gate
loads its sabotage build; a sabotage build is refused unless
`MOJOLEARN_BYTE_LM_HOST_ALLOW_SABOTAGE=1`.
"""
import hashlib
import importlib.machinery
import importlib.util
import os
import struct
import sys
from pathlib import Path

from ._buffer import addr, addr_ro, all_finite, as_f32_c, as_i32_c, frombytes, zeros
from ._bufcheck import flat_view, le_bytes
from ._byte_lm_config import ByteLanguageModelConfig

_EXTENSION = '_mojolearn_byte_lm_host'
_MODULE_NAME = 'mojolearn._host.' + _EXTENSION
_IDENTICAL_CODE = 1
_MODULE = None


def binary_path():
    """The binary this process loads, or would load."""
    override = os.environ.get('MOJOLEARN_BYTE_LM_HOST_BINARY', '').strip()
    if override:
        return os.path.abspath(override)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'host', _EXTENSION + '.so')


def _load():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    path = binary_path()
    if not os.path.exists(path):
        raise ImportError(
            f"mojolearn: {path} is not built. Build it with "
            "bindings/build_byte_lm_host.sh")
    module = sys.modules.get(_MODULE_NAME)
    if module is None:
        loader = importlib.machinery.ExtensionFileLoader(_MODULE_NAME, path)
        spec = importlib.util.spec_from_loader(_MODULE_NAME, loader, origin=path)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        sys.modules[_MODULE_NAME] = module
    if int(module.byte_lm_host_numeric_mode()) != _IDENTICAL_CODE:
        raise RuntimeError(f"mojolearn: {path} was not compiled IDENTICAL; rebuild it")
    if str(module.byte_lm_host_vendor()) != 'cpu':
        raise RuntimeError(f"mojolearn: {path} does not read back as the CPU binding")
    if bool(module.byte_lm_host_sabotage()) and os.environ.get('MOJOLEARN_BYTE_LM_HOST_ALLOW_SABOTAGE') != '1':
        raise RuntimeError(
            f"mojolearn: {path} is the gate's SABOTAGE build (DEVIATION 2612) and "
            "computes wrong answers on purpose; it is refused outside the gate")
    _MODULE = module
    return module


def _native_shape(shape):
    return [shape.batch, shape.length, shape.d_model, shape.n_heads, shape.n_kv,
            shape.head_dim, shape.intermediate, shape.n_layers, shape.vocab_size]


_MAX_THREADS = 1024
#: torch's `ignore_index`, which the loss oracle skips as a target.
_IGNORE_INDEX = -100


def _refuse_ids(tokens, vocab, batch, width, target_column):
    """ValueError unless every id is a byte value in [0, vocab). When
    `target_column` is set, that column of each row feeds only the loss as a
    target and may also hold `_IGNORE_INDEX`, exactly what the native loss
    admits; every other position is a model input."""
    flat = flat_view(tokens, 'i')
    for r in range(batch):
        base = r * width
        for c in range(width):
            v = flat[base + c]
            if 0 <= v < vocab:
                continue
            if target_column is not None and c == target_column and v == _IGNORE_INDEX:
                continue
            raise ValueError(f'ids must be byte values in [0, {vocab}); got {v} at row {r}, position {c}')


def _logits_ids(ids, shape):
    """`(tokens, copied)` for a logits call, int32 ids `[batch, length]` with
    `batch >= 1`, `1 <= length <= shape.length` and every id a byte value,
    refused with ValueError otherwise. Shared with the trainer's GPU logits
    (DEVIATION 2658) so both surfaces admit the same ids."""
    tokens, copied = as_i32_c(ids, ndim=2, name='ids')
    batch, length = tokens.shape
    if batch <= 0 or not 0 < length <= shape.length:
        raise ValueError(f'ids must be [batch, 1..{shape.length}]')
    _refuse_ids(tokens, shape.vocab_size, batch, length, None)
    return tokens, copied


def _greedy_next_bytes(logits):
    """The greedy next byte after each row of float32 logits
    `[batch, length, vocab]`, read at the last position; ties go to the
    lowest byte value. Shared with the trainer's GPU logits (DEVIATION
    2658), so equal logits bytes pick equal bytes."""
    batch, length, vocab = logits.shape
    flat = flat_view(logits, 'f')
    result = []
    for b in range(batch):
        base = ((b * length) + length - 1) * vocab
        best = 0
        for v in range(1, vocab):
            if flat[base + v] > flat[base + best]:
                best = v
        result.append(best)
    return result


def _thread_count(value):
    """None is one thread per physical core, sent to the binding as 0."""
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError('threads must be an int or None')
    if not 1 <= value <= _MAX_THREADS:
        raise ValueError(f'threads must be in [1, {_MAX_THREADS}]')
    return value


class LanguageModelInference:
    """Forward-only byte LM on the CPU. The parameters are copied at
    construction and never change afterward.

    By default (`threaded=True`) calls run the threaded path (DEVIATIONS
    2616, 2640): host kernels that spend the oracles' arithmetic in the
    oracles' order without their per-cell allocation, advanced as SIMD lanes
    and split across at most `threads` threads (None: one per physical core)
    along axes the contracts make independent. `threaded=False` runs the
    reference path, the oracles as written on one thread. Both give the same
    bits, which the gate and tools/byte_lm_host_path_sweep.py check on every
    certified CPU. Each call may override both instance defaults."""

    def __init__(self, parameters, *, shape=None, threaded=True, threads=None):
        shape = ByteLanguageModelConfig() if shape is None else shape
        if not isinstance(shape, ByteLanguageModelConfig):
            raise TypeError('shape must be a ByteLanguageModelConfig')
        if not isinstance(threaded, bool):
            raise TypeError('threaded must be a bool')
        _thread_count(threads)
        self._threaded = threaded
        self._threads = threads
        array, _ = as_f32_c(parameters, ndim=1, name='parameters')
        if tuple(array.shape) != (shape.n_total,):
            raise ValueError(f'parameters must be float32 [{shape.n_total}]')
        if not all_finite(array):
            raise ValueError('parameters must be finite')
        self._parameters = frombytes(le_bytes(array, 'f'), '<f4', (shape.n_total,))
        self._shape = shape
        self._native = _native_shape(shape)
        self._binding = _load()
        compiled = str(self._binding.byte_lm_host_profile(self._native))
        if compiled != shape.profile:
            raise RuntimeError(f'byte LM host profile mismatch: {compiled} != {shape.profile}')

    @classmethod
    def from_checkpoint(cls, path, *, threaded=True, threads=None):
        """Parameters from a `mojolearn.small-byte-lm-json-checkpoint.v1`
        file, through the trainer's own decoder and integrity checks."""
        from ._byte_lm_impl import _CHECKPOINT_LIMIT, _decode_checkpoint
        with Path(path).open('rb') as stream:
            encoded = stream.read(_CHECKPOINT_LIMIT + 1)
        state, shape = _decode_checkpoint(encoded)
        return cls(state['parameters'], shape=shape, threaded=threaded, threads=threads)

    def _threads_flag(self, threaded):
        value = self._threaded if threaded is None else threaded
        if not isinstance(value, bool):
            raise TypeError('threaded must be a bool or None')
        return 1 if value else 0

    def _threads_arg(self, threads):
        return _thread_count(self._threads if threads is None else threads)

    @property
    def shape(self):
        return self._shape

    @property
    def profile(self):
        return self._shape.profile

    def parameters_sha256(self):
        return hashlib.sha256(le_bytes(self._parameters, 'f')).hexdigest()

    def logits(self, ids, *, threaded=None, threads=None):
        """Float32 logits `[batch, length, vocab]` for int32 ids
        `[batch, length]`, positions from 0, length at most `shape.length`."""
        flag = self._threads_flag(threaded)
        count = self._threads_arg(threads)
        tokens, _ = _logits_ids(ids, self._shape)
        batch, length = tokens.shape
        out = zeros((batch, length, self._shape.vocab_size), '<f4')
        written = self._binding.byte_lm_host_logits(
            [addr_ro(self._parameters, name='parameters'), addr_ro(tokens, name='ids'),
             addr(out, name='logits')],
            [batch, length], self._native, flag, count)
        if int(written) != batch * length * self._shape.vocab_size:
            raise RuntimeError('byte LM host wrote an unexpected number of logits')
        return out

    def loss_bits(self, ids, *, threaded=None, threads=None):
        """IEEE-754 bits of the mean next-byte loss of int32 ids
        `[shape.batch, shape.length + 1]`, the training batch layout. The last
        column is only ever a target, and -100 there is ignored as the loss
        oracle ignores it."""
        flag = self._threads_flag(threaded)
        count = self._threads_arg(threads)
        tokens, _ = as_i32_c(ids, ndim=2, name='ids')
        if tuple(tokens.shape) != (self._shape.batch, self._shape.length + 1):
            raise ValueError(f'ids must be [{self._shape.batch}, {self._shape.length + 1}]')
        _refuse_ids(tokens, self._shape.vocab_size, self._shape.batch, self._shape.length + 1,
                    self._shape.length)
        return int(self._binding.byte_lm_host_loss(
            [addr_ro(self._parameters, name='parameters'), addr_ro(tokens, name='ids')],
            self._native, flag, count))

    def loss(self, ids, *, threaded=None, threads=None):
        bits = self.loss_bits(ids, threaded=threaded, threads=threads)
        return struct.unpack('<f', struct.pack('<I', bits))[0]

    def next_bytes(self, ids, *, threaded=None, threads=None):
        """Greedy next byte after each row of ids `[batch, length]`; ties go
        to the lowest byte value."""
        return _greedy_next_bytes(self.logits(ids, threaded=threaded, threads=threads))


class LanguageModelHostTrainer:
    """One byte LM training step on the CPU (DEVIATION 2680).

    Forward, backward and the AdamW update, on the reference path, through
    `byte_lm_host_train_step`. This module still holds no arithmetic; the step
    is a composition of host oracles that are each the normative answer of
    their own profile.

    NOT CERTIFIED YET, AND NOTHING HERE SAYS OTHERWISE. The CPU inference
    surface above is certified on the CPUs listed in
    docs/BYTE_LM_CPU_INFERENCE.md. This one has a gate,
    `tools/byte_lm_cpu_train_gate.py`, which compares a step's gradient, both
    Adam moments and the loss against the retained Apple, NVIDIA and AMD
    captures byte for byte, and docs/BYTE_LM_CPU_TRAINING.md records what that
    gate has actually shown. Read it before believing a number from this class.

    NO THREADED PATH. The threaded forward has no backward twin, because a
    weight gradient sums over every row of the batch and so crosses every
    thread boundary; that needs a fixed cross-thread fold, not threads
    accumulating as they finish.

    Batch composition is part of any claim made with this. Nine of the
    gradients contract over the token count, so the gradient at one batch shape
    is not the bits of the same tokens presented as two smaller batches.
    """

    def __init__(self, parameters, *, m=None, v=None, shape=None,
                 completed_steps=0, lr=1e-3, betas=(.9, .999), eps=1e-8,
                 weight_decay=0.0):
        shape = ByteLanguageModelConfig() if shape is None else shape
        self._shape = shape
        self._native = _native_shape(shape)
        self._binding = _load()
        n = shape.n_total
        array, _ = as_f32_c(parameters, ndim=1, name='parameters')
        if array.size != n:
            raise ValueError(f'parameters must hold {n} float32 values')
        if not all_finite(array):
            raise ValueError('parameters must be finite')
        self._parameters = frombytes(le_bytes(array, 'f'), '<f4', (n,))
        self._m = self._moment(m, 'm', n)
        self._v = self._moment(v, 'v', n)
        if not isinstance(completed_steps, int) or isinstance(completed_steps, bool):
            raise ValueError('completed_steps must be an int')
        if completed_steps < 0:
            raise ValueError('completed_steps must not be negative')
        self._completed = completed_steps
        beta1, beta2 = betas
        # The byte LM trainer's own validator admits positive-lr AdamW only,
        # with no clipping and no SGD options, so this refuses the same set
        # rather than passing a configuration the native side would reject.
        for name, value in (('lr', lr), ('beta1', beta1), ('beta2', beta2),
                            ('eps', eps), ('weight_decay', weight_decay)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'{name} must be a real number')
        if not lr > 0 or not eps > 0:
            raise ValueError('lr and eps must be positive')
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError('betas must be in [0, 1)')
        if weight_decay < 0:
            raise ValueError('weight_decay must not be negative')
        self._scalars = [float(lr), float(beta1), float(beta2), float(eps),
                         float(weight_decay)]
        self._gradient = None

    def _moment(self, given, name, n):
        if given is None:
            return zeros((n,), '<f4')
        array, _ = as_f32_c(given, ndim=1, name=name)
        if array.size != n:
            raise ValueError(f'{name} must hold {n} float32 values')
        if not all_finite(array):
            raise ValueError(f'{name} must be finite')
        return frombytes(le_bytes(array, 'f'), '<f4', (n,))

    @classmethod
    def from_state(cls, parameters, m, v, *, completed_steps=0, **kwargs):
        """A trainer resuming a recorded step's starting state, which is how
        the gate replays one step of the retained capture in isolation."""
        return cls(parameters, m=m, v=v, completed_steps=completed_steps, **kwargs)

    @property
    def shape(self):
        return self._shape

    @property
    def profile(self):
        return self._shape.profile

    @property
    def completed_steps(self):
        return self._completed

    @property
    def parameters_(self):
        return self._parameters

    @property
    def m_(self):
        return self._m

    @property
    def v_(self):
        return self._v

    @property
    def gradient_(self):
        """The last step's gradient, or None before any step. A failed step
        leaves this None rather than a stale value."""
        return self._gradient

    def parameters_sha256(self):
        return hashlib.sha256(le_bytes(self._parameters, 'f')).hexdigest()

    def train_step(self, ids):
        """One step on int32 ids `[shape.batch, shape.length + 1]`, the
        training batch layout, returning the loss's IEEE-754 bits.

        The last column is only ever a target and may hold the ignore index,
        exactly as `loss_bits` admits it. On success the parameters and both
        moments are replaced and the step count advances; if the native call
        raises, none of them move and the gradient is cleared, because a
        half-applied step is worse than a refused one."""
        tokens, _ = as_i32_c(ids, ndim=2, name='ids')
        if tuple(tokens.shape) != (self._shape.batch, self._shape.length + 1):
            raise ValueError(f'ids must be [{self._shape.batch}, {self._shape.length + 1}]')
        _refuse_ids(tokens, self._shape.vocab_size, self._shape.batch,
                    self._shape.length + 1, self._shape.length)
        n = self._shape.n_total
        grad = zeros((n,), '<f4')
        post_p = zeros((n,), '<f4')
        post_m = zeros((n,), '<f4')
        post_v = zeros((n,), '<f4')
        self._gradient = None
        bits = int(self._binding.byte_lm_host_train_step(
            [addr_ro(self._parameters, name='parameters'),
             addr_ro(self._m, name='m'), addr_ro(self._v, name='v'),
             addr_ro(tokens, name='ids'),
             addr(grad, name='grad'), addr(post_p, name='post_p'),
             addr(post_m, name='post_m'), addr(post_v, name='post_v')],
            self._native, self._scalars, self._completed))
        self._gradient = grad
        self._parameters = post_p
        self._m = post_m
        self._v = post_v
        self._completed += 1
        return bits

    def loss(self, ids):
        """`train_step`'s loss as a float. The bits are the comparable value;
        this is for reading."""
        return struct.unpack('<f', struct.pack('<I', self.train_step(ids)))[0]
