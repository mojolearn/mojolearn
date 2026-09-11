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


class LanguageModelInference:
    """Forward-only byte LM on the CPU. The parameters are copied at
    construction and never change afterward.

    `threaded=True` splits the work across cores along axes the contracts
    make independent (DEVIATION 2616): the same bits as the one-thread
    reference path, which the gate checks on every certified CPU. Each call
    may override the instance default."""

    def __init__(self, parameters, *, shape=None, threaded=False):
        shape = ByteLanguageModelConfig() if shape is None else shape
        if not isinstance(shape, ByteLanguageModelConfig):
            raise TypeError('shape must be a ByteLanguageModelConfig')
        if not isinstance(threaded, bool):
            raise TypeError('threaded must be a bool')
        self._threaded = threaded
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
    def from_checkpoint(cls, path, *, threaded=False):
        """Parameters from a `mojolearn.small-byte-lm-json-checkpoint.v1`
        file, through the trainer's own decoder and integrity checks."""
        from ._byte_lm_impl import _CHECKPOINT_LIMIT, _decode_checkpoint
        with Path(path).open('rb') as stream:
            encoded = stream.read(_CHECKPOINT_LIMIT + 1)
        state, shape = _decode_checkpoint(encoded)
        return cls(state['parameters'], shape=shape, threaded=threaded)

    def _threads_flag(self, threaded):
        value = self._threaded if threaded is None else threaded
        if not isinstance(value, bool):
            raise TypeError('threaded must be a bool or None')
        return 1 if value else 0

    @property
    def shape(self):
        return self._shape

    @property
    def profile(self):
        return self._shape.profile

    def parameters_sha256(self):
        return hashlib.sha256(le_bytes(self._parameters, 'f')).hexdigest()

    def logits(self, ids, *, threaded=None):
        """Float32 logits `[batch, length, vocab]` for int32 ids
        `[batch, length]`, positions from 0, length at most `shape.length`."""
        flag = self._threads_flag(threaded)
        tokens, _ = as_i32_c(ids, ndim=2, name='ids')
        batch, length = tokens.shape
        if batch <= 0 or not 0 < length <= self._shape.length:
            raise ValueError(f'ids must be [batch, 1..{self._shape.length}]')
        out = zeros((batch, length, self._shape.vocab_size), '<f4')
        written = self._binding.byte_lm_host_logits(
            [addr_ro(self._parameters, name='parameters'), addr_ro(tokens, name='ids'),
             addr(out, name='logits')],
            [batch, length], self._native, flag)
        if int(written) != batch * length * self._shape.vocab_size:
            raise RuntimeError('byte LM host wrote an unexpected number of logits')
        return out

    def loss_bits(self, ids, *, threaded=None):
        """IEEE-754 bits of the mean next-byte loss of int32 ids
        `[shape.batch, shape.length + 1]`, the training batch layout."""
        flag = self._threads_flag(threaded)
        tokens, _ = as_i32_c(ids, ndim=2, name='ids')
        if tuple(tokens.shape) != (self._shape.batch, self._shape.length + 1):
            raise ValueError(f'ids must be [{self._shape.batch}, {self._shape.length + 1}]')
        return int(self._binding.byte_lm_host_loss(
            [addr_ro(self._parameters, name='parameters'), addr_ro(tokens, name='ids')],
            self._native, flag))

    def loss(self, ids, *, threaded=None):
        return struct.unpack('<f', struct.pack('<I', self.loss_bits(ids, threaded=threaded)))[0]

    def next_bytes(self, ids, *, threaded=None):
        """Greedy next byte after each row of ids `[batch, length]`; ties go
        to the lowest byte value."""
        out = self.logits(ids, threaded=threaded)
        batch, length, vocab = out.shape
        flat = flat_view(out, 'f')
        result = []
        for b in range(batch):
            base = ((b * length) + length - 1) * vocab
            best = 0
            for v in range(1, vocab):
                if flat[base + v] > flat[base + best]:
                    best = v
            result.append(best)
        return result
