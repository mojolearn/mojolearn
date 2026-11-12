# SPDX-License-Identifier: Apache-2.0
"""SmallMLPTrainer: the fixed 8-16-3 FP32 trainer, one native call per public method.

Every public method hands its arguments once to the native owner
`bindings/small_mlp_native.mojo` (`small_mlp_call` in the tokenizer host
binding), which validates them, selects the numeric mode and route, runs the
compiled training entries and builds the result. The trainer object holds only
its native state handle (`_handle`): parameters, AdamW moments, step counter,
settings and data schedule live in `core/small_mlp_state_impl.mojo`.

The checkpoint bytes are unchanged: the same canonical JSON/hex envelope and
SHA-256 as before, so files written by earlier releases load here and files
written here load there.

State, checkpoint and schedule validation are native; the Python copies
(`_validate_state`, `_decode_state`, `_schedule`, ...) were deleted
(pyout-neural-lm, 2026-11-11). The helpers left here (`_array`, `_batch`,
`_encode_state`, `_canonical`) are read by tests only.
"""
import json
import os

from . import _backend, _linalg_impl, _training_impl
from ._array import Array
from ._buffer import as_f32_c, as_i32_c, all_finite
from ._bufcheck import is_int32, is_native_f32, le_bytes, probe

_NAMES = ('weight1', 'bias1', 'weight2', 'bias2')
_SHAPES = ((16, 8), (16,), (3, 16), (3,))
_TOTAL = 195
_STATE_SCHEMA = 'mojolearn.small-mlp-trainer.v1'
_FILE_SCHEMA = 'mojolearn.small-mlp-checkpoint.v1'
_FILE_LIMIT = 32768


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


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
        # A nested list of the right shape used to be told its SHAPE was
        # wrong. The fault is the type: this surface reads float32 buffers
        # only (the same refusal a float64 array gets from `_array`).
        raise TypeError('SmallMLPTrainer X must be a float32 array of shape (batch, 8) '
                        '(a NumPy array, an array.array or a mojolearn Array), got '
                        + type(value).__name__) from None
    if pb.ndim != 2 or pb.shape[1] != 8:
        raise ValueError('SmallMLPTrainer X must have shape (batch, 8)')
    if not 1 <= pb.shape[0] <= 256:
        raise ValueError('SmallMLPTrainer batch must be in [1, 256]')
    return _array(value, pb.shape, 'X')


def _require_mode():
    """The process-selected tier, IDENTICAL or FAST (lane neural, 2026-09-27:
    FAST runs the same kernels with the pins on the free schedule and
    promises quality, never bits). Returns it."""
    mode = _backend.default_mode()
    if mode not in ('identical', 'fast') or _backend.numeric_mode() != mode:
        raise RuntimeError('SmallMLPTrainer requires process-selected IDENTICAL or FAST mode; select it before use')
    return mode


class SmallMLPTrainer:
    """Trainer for a fixed 8-16-3 ReLU network with mean cross-entropy and AdamW.

    The model is ``Linear(8, 16) -> ReLU -> Linear(16, 3)``. Pass `weight1` (16, 8),
    `bias1` (16,), `weight2` (3, 16) and `bias2` (3,) as float32 buffers (NumPy arrays or
    `mojolearn.Array`); parameters and inputs are copied. Each batch has 1 to 256 rows
    of 8 features, with labels 0, 1 or 2. All arithmetic runs on the GPU in the numeric
    mode selected for the process (identical by default); there is no CPU fallback, and
    a state saved under one mode loads only under the same mode. Every array returned is
    a `mojolearn.Array`.

    A step either completes or leaves the state unchanged. No clipping, dropout, mixed
    precision, automatic differentiation or other layer shapes are supported.

    Parameters
    ----------
    weight1, bias1, weight2, bias2 : buffers
        Initial parameters, shapes as above.
    data_schedule : dict
        A small JSON-serializable description of your data order (for example a dataset
        hash, sample order and batch size). It is stored with the step counter and in
        checkpoints, so a resumed run can continue with the right next batch. It is not
        validated against the batches you pass; the trainer never fetches or reorders
        data.
    lr : float, default 0.001
        AdamW learning rate.
    betas : tuple of float, default (0.9, 0.999)
        AdamW moment decay rates.
    eps : float, default 1e-8
        AdamW epsilon.
    weight_decay : float, default 0.01
        AdamW decoupled weight decay.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> w1 = rng.standard_normal((16, 8)).astype(np.float32) * 0.1
    >>> w2 = rng.standard_normal((3, 16)).astype(np.float32) * 0.1
    >>> trainer = mojolearn.SmallMLPTrainer(w1, np.zeros(16, np.float32), w2, np.zeros(3, np.float32),
    ...                                     data_schedule={"batch_size": 32})
    >>> X = rng.random((32, 8), dtype=np.float32)
    >>> y = rng.integers(0, 3, 32)
    >>> out = trainer.train_step(X, y)
    """

    # Every method below is the API shell: one call into the native owner
    # (bindings/small_mlp_native.mojo), which checks the arguments, chooses the
    # route and builds the result. The object holds only `_handle`; native admission reads it from passive
    # instance storage and balanced calls retain no extra owner references.

    def __init__(self, weight1, bias1, weight2, bias2, *, data_schedule,
                 lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        return _NATIVE('owner_create', (self, (weight1, bias1, weight2, bias2, data_schedule,
                                              lr, betas, eps, weight_decay)))

    def __del__(self):
        return _NATIVE('owner_close', (self,))

    @property
    def step_(self):
        """Number of optimizer steps taken."""
        return _NATIVE('owner_step', (self,))

    @property
    def weights_(self):
        """Current parameters as a dict: `weight1`, `bias1`, `weight2`, `bias2`."""
        return _NATIVE('owner_weights', (self,))

    def state_dict(self):
        """Return an independent snapshot of the full resumable state (weights, optimizer moments, settings, data schedule)."""
        return _NATIVE('owner_state_dict', (self,))

    def load_state_dict(self, state):
        """Restore a snapshot from `state_dict`. The snapshot is validated and copied before any live state changes."""
        return _NATIVE('owner_load_state_dict', (self, state))

    def train_steps(self, X, targets, k):
        """Run `k` consecutive training steps in one call.

        `X` has shape (k * batch, 8) and `targets` (k * batch,); minibatch ``i`` is
        rows ``i * batch`` to ``(i + 1) * batch``, each 1 to 256 rows. Returns a dict
        with `step`, `losses` (one per minibatch), and `loss`, `logits` and `gradients`
        of the last minibatch.
        """
        return _NATIVE('owner_train_steps', (self, X, targets, k))

    @staticmethod
    def _binding():
        # The selected training binding, for tools and tests that call its
        # entries directly; the trainer's own methods resolve it natively.
        mode = _require_mode()
        if mode == 'identical':
            _linalg_impl.require_identical()
        return _training_impl._load(mode)

    def predict_logits_sessions(self, inputs):
        """Logits of several independent batches with the current model, concatenated in the
        order given.

        Parameters
        ----------
        inputs : list of buffers
            float32 batches, each (rows_i, 8).
        """
        return _NATIVE('owner_predict_logits_sessions', (self, inputs))

    def predict_logits(self, X):
        """Logits (n_rows, 3) of `X` (n_rows, 8) with the current weights."""
        return _NATIVE('owner_predict_logits', (self, X))

    def loss_and_grads(self, X, targets):
        """Mean cross-entropy loss and parameter gradients of a batch, without a step.

        Use it with `apply_gradients` to accumulate microbatches yourself.

        Returns
        -------
        loss : float
        gradients : sequence of Arrays
            Gradients of `weight1`, `bias1`, `weight2`, `bias2`, in that order.
        """
        return _NATIVE('owner_loss_and_grads', (self, X, targets))

    def apply_gradients(self, gradients):
        """Apply one already-combined gradient with AdamW; returns the new step count.

        `gradients` must hold exactly the four parameter gradients, in the order `weight1`,
        `bias1`, `weight2`, `bias2`, otherwise `ValueError`.
        The update either completes or leaves the state unchanged.
        """
        return _NATIVE('owner_apply_gradients', (self, gradients))

    def train_step(self, X, targets, *, return_input_grad=False):
        """One AdamW training step on a batch.

        Parameters
        ----------
        X : buffer, shape (n_rows, 8)
            float32 batch, 1 to 256 rows.
        targets : buffer, shape (n_rows,)
            Labels 0, 1 or 2.
        return_input_grad : bool, default False
            Also return the gradient with respect to `X`.

        Returns
        -------
        dict
            `step` (the new step count), `loss` and `logits` from before the update,
            `gradients` (the four parameter gradients), and `input_grad` when requested.
        """
        return _NATIVE('owner_train_step', (self, X, targets, return_input_grad))

    def save_checkpoint(self, path):
        """Atomically save the full state to `path` in a bounded, pickle-free MLP checkpoint format."""
        return _NATIVE('owner_save_checkpoint', (self, path))

    @classmethod
    def from_checkpoint(cls, path):
        """Restore a trainer from `save_checkpoint` output, after checking its schema, size and hash."""
        return _NATIVE('from_checkpoint', (cls, path))


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
    state['weights'] = {name: _encode_array(value) for name, value in state['weights'].items()}  # glue: encodes each named weight
    for key in ('m', 'v', 'flags'):  # glue: encodes three optimizer arrays
        state['optimizer'][key] = _encode_array(state['optimizer'][key])
    return state


#: The compiled training entries the native owner calls, resolved once at import.
_TRAINING_ENTRIES = (
    'training_numeric_mode', 'training_vendor', 'mlp_sessions', 'mlp_sessions_rows', 'neural_arithmetic_profile',
)


def _connect():
    # ABI connection at module import only; no trainer operation calls it.
    from types import BuiltinFunctionType
    from ._native_classical_bridge import _state_context
    _backend.default_mode()  # fixes the process mode the native owner reads
    modes = {}
    for mode in ('identical', 'fast'):  # glue: connect native modes at package import
        try:
            training = _backend.binding('_mojolearn_training', mode)
            entries = {'mode': mode}
            for name in _TRAINING_ENTRIES:  # glue: resolve existing compiled ABI symbols at import
                try:
                    entry = getattr(training, name)
                except Exception:
                    continue
                if isinstance(entry, BuiltinFunctionType):
                    entries[name] = entry
            if 'training_numeric_mode' not in entries:
                raise ImportError('SmallMLPTrainer requires a compiled training binding')
            try:
                # the trainer's captured program (autograd/small_mlp_program.mojo)
                program = _backend.binding('_mojolearn_x_sequence', mode).small_mlp_program
                if isinstance(program, BuiltinFunctionType):
                    entries['small_mlp_program'] = program
            except Exception:
                pass
            modes[mode] = entries
        except Exception as error:
            modes[mode] = type(error).__name__ + ': ' + str(error)
    return dict(trainer=SmallMLPTrainer, trainer_finalizer=vars(SmallMLPTrainer)["__del__"], array=Array, modes=modes,
                backend=vars(_backend), environ=os.environ._data, admission=_state_context())


from .tokenizer import _binding as _small_mlp_tokenizer_binding
_SMALL_MLP_MODULE = _small_mlp_tokenizer_binding()._module
_NATIVE = _SMALL_MLP_MODULE.small_mlp_call
# glue: native state cleanup must also stay native when a constructed owner is
# decref'd inside another public operation or while unwinding an error.
from types import MethodType as _MethodType
SmallMLPTrainer.__del__ = _SMALL_MLP_MODULE.classical_attribute_descriptor(
    _MethodType(_SMALL_MLP_MODULE.small_mlp_finalize, {}))

_NATIVE('initialize', (_connect(),))
