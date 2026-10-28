# SPDX-License-Identifier: Apache-2.0
"""Public wrapper for a runtime-shaped decoder language model.

This module contains no forward/backward/update arithmetic. All numerical
work belongs to _mojolearn_byte_lm, or, on a process that loaded no GPU set,
to the CPU byte LM binding through `_byte_lm_trainer_host` (2026-09-15). No
learning, performance, mathematical correctness or cross-vendor identity
claim follows from this source alone.

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

DEVICE-OWNED STEP (DEVIATION 2514). With `resident=True`
the native session's device buffers are the ONLY copy of the parameters,
the optimizer moments and the last gradient while it is open: this object's
`_state['parameters']`, `['m']` and `['v']` are `None` from the call that
opened the session until `close()`, `load_state_dict()` or a lost session.
State crosses the boundary at admission (the first call after
construction, `load_state_dict()` or `close()` hands the host state to the
native call, which validates and uploads it once), at export
(`export_state`, `export_gradients`, `export_checkpoint`, `close`) and never
per step; a step moves the ids in and the loss and the flags out. The
stateless path (`resident=False`) hands the state in and takes the new
state out on every call. Since cpu2-l11-neural (2026-10-04) `resident`
defaults to True on a GPU install (the stateless path is an explicit
opt-out and the route of a binding without session entries); a defaulted
session keeps `step_result='full'`.

ONE NATIVE CALL PER METHOD (lane neural-byte-lm-trainer, 2026-10-21).
`train_step`, `evaluate`, `logits`, `next_bytes`, `export_state` (also
`state_dict`, `parameters_`, `close`) and `export_gradients` each make one
call into the byte LM binding (`byte_lm_trainer_*`,
training/byte_lm_session_ops.mojo `byte_owner_*`), which admits, runs,
exports, splits the gradient by name and rolls back on failure. Python
checks the arguments, allocates the result arrays, passes the buffers and
the native session handle, and assigns what the call returned.

GPU LOGITS (DEVIATION 2658). `logits(ids)` and `next_bytes(ids)` run the
forward alone, through the native `byte_lm_logits` (stateless) and
`byte_lm_session_logits` (resident) entries, and return the numbers the
trainer's loss is computed from. They write no state on either path.
tools/byte_lm_gpu_logits_sweep.py compares them byte for byte with the CPU
reference path of DEVIATION 2610, `LanguageModelInference(threaded=False)`.
"""
from . import _buffer as _buffers, _bufcheck as _checks
from ._array import Array as _Array
import json
from . import _portable_math as math
from . import _numeric_profile
import operator
import os
import time as _time


class _LogitsPool:
    """Host storage for `logits()` outputs, reused across calls of one
    trainer (lane/neural-pass61, 2026-10-01). A fresh 64 MiB output per
    call cost ~30 of the L40S's ~50 ms lm-forward (zero fill plus first
    touch), the native call being ~19 ms. A store comes back here when the
    caller drops the Array it was handed, so no two live arrays ever share
    memory; the next call takes it instead of allocating. At most `keep`
    stores of the current size are held; `close()` empties it."""

    def __init__(self, keep=2):
        self.keep = keep
        self.free = []

    def take(self, size):
        for i in range(len(self.free)):  # glue: walks the small free logits pool
            if len(self.free[i]) == size:
                return self.free.pop(i)
        return None

    def give(self, store):
        if store is None:
            return
        if any(st is store for st in self.free):  # glue: walks the small free logits pool
            return
        if len(self.free) >= self.keep:
            self.free.pop(0)
        self.free.append(store)


class _PooledLogits(_Array):
    """An owned logits Array whose storage returns to `_pool` when the
    last reference to the Array goes away. Every value in it is written by
    the download that produced it (`_require_written`)."""

    def __del__(self):
        pool = getattr(self, "_pool", None)
        store = getattr(self, "_store", None)
        if pool is not None and store is not None:
            pool.give(store)
from pathlib import Path
import struct
import tempfile
import threading

from . import _backend
from ._arrays import _addr, _addr_ro
from ._buffer import addr, addr_ro, all_finite, as_f32_c, as_i32_c, empty, frombytes, full, zeros
from ._bufcheck import flat_view, is_int32, is_native_f32, le_bytes, memcopy, probe
from ._byte_lm_config import ByteLanguageModelConfig, require_shape, state_shape
from . import _byte_lm_checkpoint
from . import _ragged

PROFILE = 'mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1'
# Compatibility constants describe only the original regression fixture.
PARAMETER_NAMES = ByteLanguageModelConfig().parameter_names
PARAMETER_SHAPES = ByteLanguageModelConfig().parameter_shapes
_OFFSETS = ByteLanguageModelConfig().offsets
_N = ByteLanguageModelConfig().n_total
_EXTENSION = '_mojolearn_byte_lm'
_SCHEMA = 'mojolearn.small-byte-lm-state.v1'
_CHECKPOINT_SCHEMA = 'mojolearn.small-byte-lm-json-checkpoint.v1'
_CHECKPOINT_LIMIT = 2 * 1024 * 1024
#: `numpy.finfo(numpy.float32).max`, as a Python float.
_F32_MAX = 3.4028234663852886e+38
#: DEVIATION 2514: the native entries a resident session needs. A binding
#: missing any of them refuses with ImportError.
#: The native refusal of an id outside [0, vocab) (training/byte_lm.mojo
#: `byte_validate_tokens`), mapped back to the public ValueError.
_TOKEN_REFUSAL = 'token ID outside configured vocabulary'

_SESSION_ENTRIES = ('byte_lm_session_create', 'byte_lm_session_close', 'byte_lm_session_info',
                    'byte_lm_trainer_step', 'byte_lm_trainer_export_state',
                    'byte_lm_trainer_export_gradients')
#: The stateless trainer step: `byte_lm_run_configured` plus the gradient
#: split by name, in one call.
_RUN_ENTRY = 'byte_lm_trainer_run'
#: DEVIATION 2658 names the native logits entries and the limits they
#: admit; `logits` refuses anything outside the limits here first.
_LOGITS_ENTRY = 'byte_lm_logits'
_SESSION_LOGITS_ENTRY = 'byte_lm_trainer_logits'
#: cpu3-seq (2026-10-04): the resident greedy next byte, argmax on the device.
_SESSION_NEXT_ENTRY = 'byte_lm_trainer_next_bytes'
_LOGITS_MAX_BATCH = 1024
_LOGITS_MAX_CELLS = 268435456


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def _require_schedule_vocabulary(descriptor, shape):
    """The ONE field of `data_schedule` that is checked (2026-09-18,
    lane/tokenized-corpus): a schedule that names its vocabulary
    (`mojolearn.lm_corpus.TokenBatches.data_schedule`) must name one of this
    model's size, or the ids index rows of a different table. A schedule
    with no `vocabulary` key (every byte run) is not affected."""
    v = descriptor.get('vocabulary')
    if v is None:
        return
    if not isinstance(v, dict) or type(v.get('n_vocab')) is not int or type(v.get('sha256')) is not str:
        raise ValueError("SmallByteLanguageModelTrainer data_schedule['vocabulary'] must carry "
                         "an int n_vocab and a str sha256 (mojolearn.lm_corpus)")
    if v['n_vocab'] != shape.vocab_size:
        raise ValueError(f"vocabulary mismatch: data_schedule names vocabulary {v['sha256']} with n_vocab "
                         f"{v['n_vocab']}, and this model's vocab_size is {shape.vocab_size}")


def _schedule(value):
    if not isinstance(value, dict) or not value:
        raise ValueError('SmallByteLanguageModelTrainer data_schedule must be a nonempty JSON object')
    budget = [256]
    def check(item, depth=0):
        budget[0] -= 1
        if budget[0] < 0 or depth > 8:
            raise ValueError('Byte-LM schedule is too complex')
        if type(item) is dict:
            if len(item) > 128 or any(type(key) is not str or len(key) > 2048 for key in item):  # glue: validates the schedule argument keys
                raise ValueError('Byte-LM schedule keys/count exceed their bounds')
            for child in item.values():  # glue: validates the schedule argument values
                check(child, depth + 1)
        elif type(item) is list:
            if len(item) > 128:
                raise ValueError('Byte-LM schedule list exceeds its bound')
            for child in item:  # glue: validates the schedule argument items
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
    values = {name: _float32(value, name) for name, value in  # glue: converts the configuration arguments
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


def _parameters(value, shape=None):
    shape = require_shape(shape)
    if isinstance(value, dict):
        if set(value) != set(shape.parameter_names):
            raise ValueError('Byte-LM named parameters must contain the exact configured tensor registry')
        arrays = [_array(value[name], shape, name) for name, shape in zip(shape.parameter_names, shape.parameter_shapes)]  # glue: one array per named model parameter
        flat = empty((shape.n_total,), '<f4')
        at = 0
        for value in arrays:  # glue: walks the named model parameter arrays
            memcopy(addr(flat, name='parameters') + at, addr_ro(value, name='tensor'), value.nbytes)
            at += value.nbytes
        return flat
    return _array(value, (shape.n_total,), 'parameters')


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


#: The tiers the byte LM ships and the code its binding reads back
#: (`byte_lm_numeric_mode`). IDENTICAL is the default; FAST (afn-lm,
#: 2026-10-03) runs the same device step with the pins on the free schedule
#: and promises quality, never bits.
_NATIVE_MODE_CODE = {'identical': 1, 'fast': 0}


def _mode():
    """The process-selected tier ('identical' or 'fast'), or raise."""
    mode = _backend.default_mode()
    if mode not in _NATIVE_MODE_CODE or _backend.numeric_mode() != mode:
        raise RuntimeError('SmallByteLanguageModelTrainer requires process-selected IDENTICAL or FAST mode')
    return mode


def _load(shape=None):
    shape = require_shape(shape)
    mode = _mode()
    binding = _backend.binding(_EXTENSION, mode)
    # On a process that loaded no GPU set, `_backend.binding` serves the
    # single-device entries from the CPU byte LM binding
    # (_byte_lm_trainer_host, lane/cpu-training-embedding-ivf, 2026-09-15),
    # which reads back "cpu"; a box with a GPU never admits that vendor.
    # A CPU-only install admits only the "cpu" vendor and a GPU install only
    # the GPU vendors; the install decides, so this module never imports the
    # CPU trainer adapter (cpu-gpu-cleanup n-pyneural).
    vendors = ('cpu',) if _backend._CPU_ONLY is not None else ('cuda', 'hip', 'metal')
    # The binding declares its own arithmetic suffix (every -D switch tag that
    # ByteConfig.profile() appends: neural, CE token-tree, GEMM leaf,
    # attention; training/byte_lm_config.byte_lm_arithmetic_suffix), so an
    # A/B arm build loads with its arm tags; base PROFILE (shape + version),
    # numeric mode and vendor still refuse a real mismatch. Before
    # 2026-10-10 the suffix carried only the neural part and grid arms
    # attn_softmax / ce_token_fold were refused here (bytelm-profile-harness).
    arithmetic_suffix = str(binding.byte_lm_arithmetic_suffix())
    if (int(binding.byte_lm_numeric_mode()) != _NATIVE_MODE_CODE[mode]
            or str(binding.byte_lm_profile()) != PROFILE + arithmetic_suffix
            or str(binding.byte_lm_vendor()) not in vendors):
        raise RuntimeError('Byte-LM requires the exact native profile, the selected IDENTICAL/FAST mode and CUDA/HIP/Metal vendor '
                           '(or, on a CPU-only install, the CPU byte LM binding)')
    if not callable(getattr(binding, 'byte_lm_run', None)):
        raise ImportError('Byte-LM binding is missing byte_lm_run; rebuild bindings/build_byte_lm.sh')
    if shape.profile != PROFILE:
        if not callable(getattr(binding, 'byte_lm_run_configured', None)) or not callable(getattr(binding, 'byte_lm_config_profile', None)):
            raise ImportError('Byte-LM binding lacks runtime shapes; rebuild bindings/build_byte_lm.sh')
        if str(binding.byte_lm_config_profile(list(shape.native_shape))) != shape.profile + arithmetic_suffix:
            raise RuntimeError('Byte-LM native runtime shape/profile mismatch')
    return binding


def _runtime_profile(shape, mode=None):
    """Bind shape metadata to the arithmetic of the actual loaded binary."""
    binding = _backend.binding(_EXTENSION, mode or _mode())
    return shape.profile + str(binding.byte_lm_arithmetic_suffix())


def _snapshot(state, copy=True):
    """The `state_dict()` shape. `copy=False` hands the arrays through as
    they are, for a caller that already holds fresh owned copies (an
    export, DEVIATION 2514) and would otherwise copy 3n floats twice."""
    shape = state_shape(state)
    state = dict(state)
    if 'model_shape' in state:
        state['model_shape'] = shape.to_dict()
    arrays = {key: (state[key].copy() if copy else state[key]) for key in ('parameters', 'm', 'v', 'flags')}  # glue: four named state arrays of a snapshot
    return dict(state, config=dict(state['config']), **arrays,
                data_schedule=_schedule(state['data_schedule']),
                parameter_names=list(shape.parameter_names),
                parameter_shapes=[list(shape) for shape in shape.parameter_shapes],  # glue: lists the model parameter shapes
                parameter_offsets=list(shape.offsets))


def _validate_state(value):
    keys = {'schema', 'profile', 'numeric_mode', 'parameter_names', 'parameter_shapes',
            'parameter_offsets', 'parameters', 'm', 'v', 'flags', 'completed_steps',
            'next_batch_index', 'config', 'data_schedule'}
    if isinstance(value, dict) and 'model_shape' in value:
        keys.add('model_shape')
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('Byte-LM state has missing or unknown fields')
    shape = state_shape(value)
    if (value['schema'] != _SCHEMA or value['numeric_mode'] not in _NATIVE_MODE_CODE or value['profile'] != _runtime_profile(shape, value['numeric_mode'])
            or value['parameter_names'] != list(shape.parameter_names)
            or value['parameter_shapes'] != [list(shape) for shape in shape.parameter_shapes]  # glue: compares the model parameter shapes
            or value['parameter_offsets'] != list(shape.offsets)):
        raise ValueError('Byte-LM state profile/registry/mode mismatch')
    cfg = value['config']
    if not isinstance(cfg, dict) or set(cfg) != set(_configuration(1, (.9, .999), 1e-8, .01)):
        raise ValueError('Byte-LM state optimizer configuration mismatch')
    config = _configuration(cfg['lr'], (cfg['beta1'], cfg['beta2']), cfg['eps'], cfg['weight_decay'])
    if (type(cfg['kind']) is not int or cfg['kind'] != 2 or type(cfg['nesterov']) is not bool
            or cfg['nesterov'] or any(_float32(cfg[key], key) != 0 for key in ('momentum', 'dampening', 'max_norm'))):  # glue: validates three optimizer config keys
        raise ValueError('Byte-LM first profile supports AdamW without clipping or SGD options')
    completed = _step(value['completed_steps'])
    if _step(value['next_batch_index']) != completed:
        raise ValueError('Byte-LM schedule cursor must match completed_steps')
    parameters = _array(value['parameters'], (shape.n_total,), 'parameters')
    m = _array(value['m'], (shape.n_total,), 'm')
    v = _array(value['v'], (shape.n_total,), 'v')
    flags = _array(value['flags'], (shape.n_tensors,), 'flags', '<i4')
    # Native min/max (reduce_stat, DEVIATION 3101) over buffers `_array`
    # already proved finite: `min < 0` is `any(x < 0)` exactly, and an int32
    # flag vector is binary iff min >= 0 and max <= 1.
    if v.min() < 0 or flags.min() < 0 or flags.max() > 1:
        raise ValueError('Byte-LM requires nonnegative second moments and binary flags')
    value = dict(value)
    if 'model_shape' in value:
        value['model_shape'] = shape.to_dict()
    return dict(value, parameters=parameters, m=m, v=v, flags=flags,
                completed_steps=completed, next_batch_index=completed, config=config,
                data_schedule=_schedule(value['data_schedule']),
                parameter_names=list(shape.parameter_names),
                parameter_shapes=[list(shape) for shape in shape.parameter_shapes], parameter_offsets=list(shape.offsets))  # glue: lists the model parameter shapes


def _sha256_hex(data):
    """The SHA-256 of the bytes `data` as 64 hex digits, computed by the Mojo
    SHA-256 (core/sha256.mojo) behind the loaded byte LM binding's
    `sha256_hex`: the same digest as before, over the same bytes."""
    return str(_backend.binding(_EXTENSION, _mode()).sha256_hex(data))


def _sha(binding, path):
    """The SHA-256 of the file at `path`, streamed by the binding's Mojo
    file digest (`byte_lm_file_sha256`); the file never passes through
    Python."""
    return str(binding.byte_lm_file_sha256(str(path)))  # cpu-route: file path str read once by byte_lm_file_sha256 (the file never enters Python)


def _binding_metadata(binding, shape):
    root = Path(__file__).resolve().parents[2]
    names = ('python/mojolearn/_byte_lm_impl.py', 'python/mojolearn/language_model.py',
             'bindings/_mojolearn_byte_lm.mojo', 'training/byte_lm.mojo',
             'training/byte_lm_config.mojo', 'python/mojolearn/_byte_lm_config.py')
    inventory = {name: _sha(binding, root / name) for name in names if (root / name).is_file()}  # glue: one native file digest per named source file
    # Installed wheels may lack native sources; the Python source and exact
    # loaded binding are still identified. Do not claim a full source audit.
    inventory['loaded_python_wrapper'] = _sha(binding, __file__)
    # DEVIATION 2534: the attention arm the native launchers run, the
    # column's default and whether the binding is a trial build, so a run
    # names its arm instead of an environment variable that may be unset.
    # None for a binding built before the read-back existed.
    attention = None
    if hasattr(binding, 'byte_lm_attention_arm'):
        arm, default, trial, resolved = binding.byte_lm_attention_arm()
        attention = dict(arm=str(arm), default=str(default), trial_build=bool(int(trial)),
                         resolved_hd64=str(resolved))
        if hasattr(binding, 'byte_lm_attention_memory_profile'):
            memory_profile, retained_bytes = binding.byte_lm_attention_memory_profile(list(shape.native_shape))
            attention['memory_profile'] = str(memory_profile)
            attention['retained_exp_bytes_per_layer'] = int(retained_bytes)
    # DEVIATION 2648: the step glue arm the native step runs and whether the
    # binding is a glue trial build. None for a binding without the read-back.
    step_glue = None
    if hasattr(binding, 'byte_lm_step_glue_arm'):
        glue_arm, glue_trial = binding.byte_lm_step_glue_arm()
        step_glue = dict(arm=str(glue_arm), trial_build=bool(int(glue_trial)))
    return dict(binding_file=binding.__file__, binding_sha256=_sha(binding, binding.__file__),
                native_profile=str(binding.byte_lm_profile()),
                native_vendor=str(binding.byte_lm_vendor()),
                native_numeric_mode=int(binding.byte_lm_numeric_mode()),
                native_attention_arm=attention,
                native_step_glue_arm=step_glue,
                source_sha256=inventory,
                source_scope='available direct source files; binding SHA identifies the compiled artifact')


# THE IDS ADMISSION AND THE GREEDY PICK, shared by the GPU trainer's
# `logits`/`next_bytes` and the CPU `LanguageModelInference` (DEVIATION
# 2658), so both surfaces admit the same ids and equal logits bytes pick
# equal bytes. They live here, in the GPU-path module, so this module does
# not import the CPU-side `_byte_lm_host` (cpu-gpu-cleanup n-pyneural,
# 2026-10-02); `_byte_lm_host` keeps the same spelling.


def _logits_ids(ids, shape):
    """`(tokens, copied)` for a logits call, int32 ids `[batch, length]` with
    `batch >= 1`, `1 <= length <= shape.length` and every id a byte value in
    `[0, vocab)`, refused with ValueError otherwise."""
    tokens, copied = as_i32_c(ids, ndim=2, name='ids')
    batch, length = tokens.shape
    if batch <= 0 or not 0 < length <= shape.length:
        raise ValueError(f'ids must be [batch, 1..{shape.length}]')
    vocab = shape.vocab_size
    # Native min/max admits the common case in one pass. On a refusal the
    # first offending id is located natively too (pyglue-text-io, Oct 3): two
    # `threshold_labels_i64` masks (id < 0, id >= vocab) and a native
    # first-max argmax of each; Python reads only the one offending value.
    if tokens.size and tokens.min() >= 0 and tokens.max() < vocab:
        return tokens, copied
    from ._labels import threshold_codes
    flat = tokens.reshape((batch * length,))
    first = batch * length
    for mask in (threshold_codes(flat, -0.5, strict=True, below=1, above=0),  # glue: two native masks on the refusal path only
                 threshold_codes(flat, vocab - 0.5, strict=True, below=0, above=1)):
        if mask.max() == 1:
            first = min(first, int(mask.argmax()))  # glue: native argmax on the refusal path only
    r, c = divmod(first, length)
    raise ValueError(f'ids must be byte values in [0, {vocab}); got {int(flat[first])} at row {r}, position {c}')


def _argmax_last_positions(logits, batch, length, vocab):
    """int64 `(batch,)`: the first-max argmax of each row's LAST position of
    float32 logits `[batch, length, vocab]`, by the base binding's
    `argmax_last_rows_f32` (the `argmax_rows_f32` rule; lane cpu4-python:
    the last-position rows are read on the device, the host byte gather
    `gather_rows_bytes` is gone)."""
    out = empty((batch,), '<i8')
    _buffers._native('argmax_last_rows_f32')(addr_ro(logits, name='logits'), [batch, length, vocab],
                                             addr(out, name='next bytes'))
    return out


def _greedy_next_bytes(logits):
    """The greedy next byte after each row of float32 logits
    `[batch, length, vocab]`, read at the last position; ties go to the
    lowest byte value (the base binding's `argmax_rows_f32`: strict `>` from
    index 0, so ties keep the lowest byte and a NaN never replaces). The
    last-position gather and the scan are both native (pyglue-text-io)."""
    logits, _ = as_f32_c(logits, ndim=3, name='logits')
    batch, length, vocab = logits.shape
    if not batch:
        return []
    if not vocab:
        return [0] * batch
    return _argmax_last_positions(logits, batch, length, vocab).tolist()


def _gpu_logits_ids(ids, shape):
    """`(tokens, batch, length)` for `logits` (DEVIATION 2658). The CPU
    class's admission (`_logits_ids` above) plus the native batch
    and cell limits, returned as an owned copy so the trainer never aliases
    caller memory. An empty 2-D buffer is refused by its reported shape
    before any conversion."""
    try:
        pb = probe(ids)
    except TypeError:
        pb = None
    if pb is not None and pb.ndim == 2 and 0 in pb.shape:
        raise ValueError(f'ids must be [batch, 1..{shape.length}]')
    tokens, copied = _logits_ids(ids, shape)
    batch, length = tokens.shape
    if batch > _LOGITS_MAX_BATCH or batch * length * shape.vocab_size > _LOGITS_MAX_CELLS:
        raise ValueError(f'Byte-LM logits require batch <= {_LOGITS_MAX_BATCH} and '
                         f'batch * length * vocab <= {_LOGITS_MAX_CELLS}')
    return (tokens if copied else tokens.copy()), batch, length


def _gradient_outputs(shape, named):
    """Fresh output arrays for a gradient the native call writes: the flat
    vector and, with `named`, one array per parameter tensor (the native
    call splits the gradient into them)."""
    flat = empty((shape.n_total,), '<f4')
    by_name = {}
    if named:
        for name, tensor_shape in zip(shape.parameter_names, shape.parameter_shapes):  # glue: one output array per model parameter tensor
            by_name[name] = empty(tuple(tensor_shape), '<f4')
    return flat, by_name


def _require_written(written, expected):
    if _is_bool(written) or not isinstance(written, int) or written != expected:
        raise RuntimeError('Byte-LM native logits wrote an unexpected number of logits')


# FP32 decoder language-model trainer with runtime dimensions.
#
# Defaults B2/L32/DM32/H4/KV2/FF64/V256; pass ByteLanguageModelConfig
# as shape to configure dimensions, layer count and vocabulary.
# no biases/dropout/final norm, untied embedding/head. Supply a flat FP32
# array or the configured named tensors exposed by parameter_registry(shape). There
# is no hidden initialization, tokenizer, padding, truncation or RNG.
#
# logits(ids)/next_bytes(ids) (DEVIATION 2658) run the same device forward
# alone on int32[batch, length] IDs and change no state; the logits are the
# numbers the loss is computed from, and next_bytes picks greedily with
# ties going to the lowest byte value.
#
# train_step/evaluate require actual int32[B,L+1] IDs in [0, shape.vocab_size).
# The first L positions predict the next L; loss averages B*L targets. All arithmetic runs in the native CUDA/HIP/Metal
# profile of the process-selected tier: IDENTICAL (default, same bits on every vendor) or FAST (quality, never bits).
#
# data_schedule is a bounded caller-supplied JSON descriptor. Retain corpus
# and actual token-order SHA256, batch offsets, and planned steps there.
# next_batch_index equals completed_steps; callers must provide the matching
# next batch after restoration. This class does not fetch/reorder a corpus.
#
# NOTHING VALIDATES ITS CONTENTS, and a reader of a published run has to
# know that (lane/data-ordering-determinism, 2026-09-16). The descriptor is
# checked for size, depth, key count and JSON round-trip and for nothing
# else: no key is required, no value is compared against the IDs handed to
# train_step, and {'dataset': 'test'} trains. What it buys is that once a
# checkpoint is written the descriptor is covered by the envelope's
# payload_sha256, so it cannot be swapped without detection. That makes it
# TAMPER-EVIDENT, not TRUE. The run is reproducible from it only to the
# extent the caller made it so, and this class cannot tell the difference.
#
# resident=True retains an owned native context/model/optimizer across
# calls, and (DEVIATION 2514) the device buffers are then the ONLY copy of
# the parameters, moments and last gradient while the session is open:
# nothing is mirrored to the host per step. The state is validated once at
# admission (the first call after construction, `load_state_dict()` or
# `close()`), validated on the device after every update, and validated
# again when it is exported. `export_state()` (also `state_dict()` and
# `parameters_`), `export_gradients()` and `export_checkpoint()` copy it
# out; `close()` exports first and then releases the device, so the next
# call re-admits from host state. A step that fails after its update is
# rolled back on the device; a failure the Python layer detects after
# native success is rolled back through `byte_lm_session_rollback`. What
# is weaker than the stateless path: a LOST context (a rollback whose
# re-scan does not answer) loses the steps since the last export; the
# object then raises "session lost at step k; last exported state is step
# j" on every call until `load_state_dict()`.
#
# step_result selects what `train_step` returns. 'full' returns the
# complete dict (loss, gradients, state); on a resident session it is the
# lean step plus `export_gradients()`, so the bytes are the stateless
# path's. 'lean' (resident only) returns `loss, step, completed_steps,
# next_batch_index, flags` and leaves the gradient on the device until
# `export_gradients()`. The default (None) is 'lean' for a resident
# trainer and 'full' for a stateless one: DEVIATION 2514 gate G5 on the
# H100 (2026-09-11) measured a lean target-shape step at 0.565 s against
# 8.6 s full and 45 s before the device-owned step, with every lean step
# bit-equal to the full path's and to the pre-change trainer's.
#
# Complete state is copied and updates commit only after success. On the
# stateless path evaluation requires byte-unchanged parameters, moments,
# flags and counter; on a resident session that invariance is a native
# gate (design G1), not a per-call download. Public checkpoints use an
# explicitly separate bounded JSON/hex schema; they are not
# training/checkpoint.mojo's native binary v1 files. The JSON codec
# retains its 2 MiB bound; larger models export `export_state()` arrays.
# Learning, numerical correctness and cross-vendor identity remain
# unqualified in this slice.
class SmallByteLanguageModelTrainer:
    """FP32 decoder language model trainer (AdamW, mean cross-entropy).

    Trains a decoder-only language model whose dimensions come from
    :class:`~mojolearn.LanguageModelConfig` (``shape=``); the defaults are a
    small byte-level model (batch 2, length 32, width 32, 4 heads, 2 key/value
    heads, MLP 64, vocabulary 256). The model has no biases, no dropout and no
    final norm, and the embedding and output head are separate (untied).
    ``LanguageModelTrainer`` is the same class.

    You supply everything: the initial parameters (there is no hidden
    initialization), the token ids of every step (there is no tokenizer,
    padding, truncation or random number generation inside), and a
    ``data_schedule`` describing them. All arithmetic runs on the GPU in the
    process numeric mode: ``"identical"`` (the default) gives the same bits on
    Apple, NVIDIA and AMD.

    Resident sessions. On a GPU install ``resident`` defaults to True: the
    parameters, AdamW moments and last gradient live on the device between
    steps, and only the ids go in and the loss comes out. They are copied to
    the host only when you ask (:meth:`export_state`, :meth:`export_gradients`,
    :meth:`export_checkpoint`, :meth:`close`). A step that fails is rolled back.
    If the device context is lost, the steps since the last export are lost
    and every call raises RuntimeError until :meth:`load_state_dict`.
    ``resident=False`` runs each step statelessly with the state on the host.

    The ``data_schedule`` is a small JSON-serializable dict you provide
    (for example from :meth:`mojolearn.lm_corpus.TokenBatches.data_schedule`):
    record the corpus, token order, batch offsets and planned steps there. It
    is checked only for size and JSON round-trip, not against the ids you
    pass, but once a checkpoint is written it is covered by the checkpoint's
    SHA-256, so it cannot be changed undetected. ``next_batch_index`` equals
    the completed step count; after a restore, pass the matching next batch.

    Parameters
    ----------
    parameters : array-like or dict
        Initial parameters: one flat float32 array of ``shape.n_total``
        values, or a dict of named tensors as listed by
        :meth:`parameter_registry`.
    data_schedule : dict
        The JSON descriptor described above. When it records a vocabulary,
        its ``n_vocab`` must equal ``shape.vocab_size``.
    lr : float, default 1e-3
        AdamW learning rate.
    betas : tuple of float, default (0.9, 0.999)
        AdamW moment decay rates.
    eps : float, default 1e-8
        AdamW epsilon.
    weight_decay : float, default 0.01
        AdamW decoupled weight decay.
    shape : LanguageModelConfig or None, default None
        Model dimensions; None uses the defaults.
    resident : bool or None, default None
        Keep the training state on the device between steps. None chooses the
        resident session whenever the loaded GPU binaries support it.
    step_result : {"full", "lean"} or None, default None
        What :meth:`train_step` returns. ``"full"`` includes the gradients;
        ``"lean"`` (requires ``resident=True``) returns the loss, step counters
        and flags and leaves the gradient on the device. None means ``"lean"``
        when ``resident=True`` is given and ``"full"`` otherwise.

    Attributes
    ----------
    step_ : int
        Completed training steps.
    parameters_ : Array
        A copy of the current flat parameters.
    data_schedule : dict
        A copy of the ``data_schedule`` descriptor.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> shape = mojolearn.LanguageModelConfig()
    >>> rng = np.random.default_rng(0)
    >>> weights = (rng.standard_normal(shape.n_total) * 0.02).astype(np.float32)
    >>> trainer = mojolearn.LanguageModelTrainer(weights, data_schedule={"corpus": "demo"})
    >>> ids = rng.integers(0, 256, (shape.batch, shape.length + 1), dtype=np.int32)
    >>> loss = trainer.train_step(ids)["loss"]
    """

    def __init__(self, parameters, *, data_schedule, lr=1e-3, betas=(.9, .999),
                 eps=1e-8, weight_decay=.01, shape=None, resident=None, step_result=None):
        _LM_TRAINER(self, "construct", (parameters, data_schedule, lr, betas, eps,
                    weight_decay, shape, resident, step_result), _LM_TRAINER_CONTEXT)

    @property
    def data_schedule(self):
        """A copy of the ``data_schedule`` descriptor, as checkpoints keep it."""
        return _LM_TRAINER(self, "data_schedule", (), _LM_TRAINER_CONTEXT)

    @staticmethod
    def parameter_registry(shape=None):
        """Name, shape, offset and size of every parameter tensor.

        Parameters
        ----------
        shape : LanguageModelConfig or None, default None
            Model dimensions; None uses the defaults.

        Returns
        -------
        list of dict
            One ``{"name", "shape", "offset", "size"}`` record per tensor, in
            the order of the flat parameter vector.
        """
        return _LM_TRAINER(None, "parameter_registry", (shape,), _LM_TRAINER_CONTEXT)

    @property
    def step_(self):
        """Number of completed training steps."""
        return _LM_TRAINER(self, "step", (), _LM_TRAINER_CONTEXT)

    @property
    def parameters_(self):
        """A copy of the current flat float32 parameters."""
        return _LM_TRAINER(self, "parameters", (), _LM_TRAINER_CONTEXT)

    def state_dict(self):
        """The complete training state; same as :meth:`export_state`."""
        return _LM_TRAINER(self, "export_state", (), _LM_TRAINER_CONTEXT)

    def _require_not_lost(self):
        if self._lost_at is not None:
            raise RuntimeError('Byte-LM session lost at step %d; last exported state is step %d'
                               % self._lost_at)

    def _mark_lost(self):
        if self._lost_at is None:
            self._lost_at = (self._state['completed_steps'], self._last_export_step)

    # The complete state as fresh `mojolearn.Array` copies plus metadata
    # (the `state_dict()` shape). On an open resident session: the device
    # state is re-scanned natively, downloaded once and validated here by
    # the same `_validate_state` that admits an import; on a closed or
    # stateless trainer the host arrays are copied. Mutating the returned
    # arrays cannot reach the device (design section 3).
    def export_state(self):
        """The complete training state as fresh copies.

        On a resident session this downloads the device state once and
        validates it. Changing the returned arrays does not affect the trainer.

        Returns
        -------
        dict
            Parameters, AdamW moments ``m`` and ``v``, flags, step counters,
            configuration, ``data_schedule`` and metadata, as accepted by
            :meth:`load_state_dict`.
        """
        return _LM_TRAINER(self, "export_state", (), _LM_TRAINER_CONTEXT)

    def export_gradients(self, named=True):
        """The pre-update gradient of the last completed step of a resident session.

        Valid until the next :meth:`train_step` starts. The stateless path
        returns its gradients from :meth:`train_step` instead.

        Parameters
        ----------
        named : bool, default True
            Also return the gradient split into named tensors.

        Returns
        -------
        dict
            ``{"flat_gradients": Array}`` and, when ``named``,
            ``{"gradients": {name: Array}}``.

        Raises
        ------
        RuntimeError
            If no resident session is open, or no step has completed since it was
            opened, restored or rolled back.
        """
        return _LM_TRAINER(self, "export_gradients", (named,), _LM_TRAINER_CONTEXT)

    # DEVIATION 3010: what the OPEN resident session's quadratic
    # attention stages are holding RIGHT NOW. `None` when no session is
    # open, or when the loaded binding predates this report.
    #
    # Keys: `forward_eager_cells`, `forward_aexp_cells`,
    # `backward_eager_cells` and the matching `*_bytes` at four bytes a
    # cell; `layers_grown_forward`, `layers_grown_backward`,
    # `layers_full_aexp`, `layers`; `eager_bytes` (forward eager +
    # backward eager, the two that only the eager fallback grows) and
    # `total_bytes` (those plus `aexp`). New bindings also report per-layer
    # `forward_status`, `backward_status`, their named `*_counts`, and
    # `attn_materialized` after the latest step. Status -1 means not
    # attempted; 0 ran, 1 refused regime, 2 corner, 3 DEVIATION 3110's
    # latch (this layer refused before, so nothing was launched at all and
    # the eager path ran alone). Materialization may
    # have occurred during backward recomputation. Retained capacity does
    # not say which path ran this step.
    #
    # WHAT IT IS FOR. The ten `[B, n_heads, L, S]` attention arrays are
    # allocated at ONE element for the fused path and GROW ON DEMAND the
    # first time a layer takes the eager path -- a refused regime, or a
    # `FUSED_CORNER` hit, both of which depend on the DATA and so on the
    # step. With the legacy retention policy they stay for the session.
    # `release_eager` builds release dead forward scratch after forward
    # and remaining eager stages after each layer backward; the report's
    # `released_eager_bytes` witnesses that work separately from capacity.
    # `sticky_eager` and `layers_prefer_eager` report the policy which
    # chooses eager before launch after a layer's first corner refusal.
    # `repair_masked_tail` identifies the replay build; per-layer
    # `backward_repair_sites` is a bitmask of estash repairs actually executed:
    # 1 for zdot, 2 for dQ, 3 for both, 0 for neither.
    # A legacy run therefore has no single device footprint: it can step up
    # once, at a step nobody chose, and every capacity figure taken
    # before that step is wrong afterwards. A three-step probe cannot
    # see it. Read this between steps and the step where it moved is
    # named rather than inferred.
    #
    # `forward_aexp_cells` is reported APART from the other forward
    # arrays because two unrelated mechanisms grow `aexp`: the eager
    # fallback grows all four together, while a build carrying the
    # fused exp stash grows `aexp` alone on a call that refused
    # nothing. `eager_bytes` excludes it for exactly that reason.
    #
    # Host metadata only: buffer lengths and saved launcher statuses. Nothing is launched, downloaded or
    # synchronized, and no step path calls this.
    def attention_stage_report(self):
        """Memory held by the open session's attention stages.

        The attention work arrays start small and grow the first time a layer
        needs the unfused attention path, which depends on the data. Read this
        between steps to see the current device footprint of those arrays.
        Nothing is launched or downloaded.

        Returns
        -------
        dict or None
            Cell and byte counts per stage (``forward_eager_cells``,
            ``forward_aexp_cells``, ``backward_eager_cells`` and their ``*_bytes``,
            ``eager_bytes``, ``total_bytes``), layer counts and per-layer status.
            None when no resident session is open or the loaded binaries do not
            report it.
        """
        with self._lock:
            if not self._session_open or self._native_session is None:
                return None
            info = self._session_binding.byte_lm_session_info(self._native_session)
            if len(info) < 10:
                return None
            fwd, aexp, bwd = int(info[4]), int(info[5]), int(info[6])
            report = dict(forward_eager_cells=fwd, forward_aexp_cells=aexp,
                        backward_eager_cells=bwd,
                        forward_eager_bytes=fwd * 4, forward_aexp_bytes=aexp * 4,
                        backward_eager_bytes=bwd * 4,
                        eager_bytes=(fwd + bwd) * 4,
                        total_bytes=(fwd + aexp + bwd) * 4,
                        layers_grown_forward=int(info[7]),
                        layers_grown_backward=int(info[8]),
                        layers_full_aexp=int(info[9]),
                        layers=state_shape(self._state).n_layers)
            n_layers = report['layers']
            if len(info) >= 12:
                report['stage_lists'] = (int(info[10]), int(info[11]))
                n_layers = min(n_layers, int(info[10]), int(info[11]))
            if len(info) >= 12 + 3 * n_layers:
                names = {-1: 'NOT_ATTEMPTED', 0: 'FUSED_RAN',
                         1: 'FUSED_REFUSED_REGIME', 2: 'FUSED_CORNER', 3: 'FUSED_SKIPPED_STICKY'}
                for offset, key in ((0, 'forward_status'), (1, 'backward_status')):  # glue: two status keys of the stage report
                    values = [int(info[12 + 3 * layer + offset]) for layer in range(n_layers)]  # glue: reads per-layer status codes from the binding (n_layers-sized: model layers)
                    report[key] = values
                    report[key + '_counts'] = {name: values.count(code)
                                              for code, name in names.items()}  # glue: maps status code names
                report['attn_materialized'] = [bool(info[14 + 3 * layer])
                                             for layer in range(n_layers)]  # glue: per-layer status names (n_layers-sized: model layers)
            if len(info) > 12 + 3 * n_layers:
                report['exact_tail_guard'] = bool(info[12 + 3 * n_layers])
            if len(info) > 14 + 3 * n_layers:
                report['release_eager'] = bool(info[13 + 3 * n_layers])
                report['released_eager_bytes'] = int(info[14 + 3 * n_layers]) * 4
            if len(info) >= 16 + 4 * n_layers:
                report['sticky_eager'] = bool(info[15 + 3 * n_layers])
                report['layers_prefer_eager'] = [bool(info[16 + 3 * n_layers + layer])
                                               for layer in range(n_layers)]  # glue: per-layer status names (n_layers-sized: model layers)
            if len(info) >= 17 + 5 * n_layers:
                report['repair_masked_tail'] = bool(info[16 + 4 * n_layers])
                report['backward_repair_sites'] = [int(info[17 + 4 * n_layers + layer])
                                                  for layer in range(n_layers)]  # glue: per-layer status names (n_layers-sized: model layers)
            return report

    def _export_state_impl(self, release=False):
        """Lock held, session open. ONE native call
        (`byte_lm_trainer_export_state`, training/byte_lm_session_ops.mojo
        `byte_owner_export_state`): the device state is re-scanned,
        downloaded into fresh owned arrays and checked against the committed
        step and flags; with `release` the session is closed after it.
        Returns the state dict holding those arrays."""
        shape = state_shape(self._state)
        binding = self._session_binding
        out_p = empty((shape.n_total,), '<f4')
        out_m = empty((shape.n_total,), '<f4')
        out_v = empty((shape.n_total,), '<f4')
        out_flags = empty((shape.n_tensors,), '<i4')
        completed = self._state['completed_steps']
        addresses = [addr_ro(self._state['flags'], name='flags'), addr(out_p, name='out_p'),
                     addr(out_m, name='out_m'), addr(out_v, name='out_v'), addr(out_flags, name='out_flags')]
        returned = binding.byte_lm_trainer_export_state(self._native_session, addresses,
                                                        list(shape.native_shape), completed, bool(release))
        if _is_bool(returned) or not isinstance(returned, int) or returned != completed:
            raise RuntimeError('Byte-LM state export returned a different completed step')
        self._last_export_step = completed
        return dict(self._state, parameters=out_p, m=out_m, v=out_v, flags=out_flags)

    def _export_gradients_impl(self, step, named):
        """ONE native call (`byte_lm_trainer_export_gradients`): the last
        step's gradient, scanned on the device, into a fresh flat array and,
        with `named`, also split into one fresh array per tensor."""
        shape = state_shape(self._state)
        flat, by_name = _gradient_outputs(shape, named)
        addresses = [addr(flat, name='out_grad')] + [addr(by_name[name], name='out_grad') for name in by_name]  # glue: one address per model parameter tensor
        returned = self._session_binding.byte_lm_trainer_export_gradients(
            self._native_session, addresses, list(shape.native_shape), step)
        if _is_bool(returned) or not isinstance(returned, int) or returned != step:
            raise RuntimeError('Byte-LM gradient export returned a different step')
        result = dict(flat_gradients=flat)
        if named:
            result['gradients'] = by_name
        return result

    def close(self):
        """Copy the device state to the host and release the device.

        A later call re-opens the session from that host state. On a lost
        session nothing can be exported: the device is released and the trainer
        stays lost until :meth:`load_state_dict`.
        """
        return _LM_TRAINER(self, "close", (), _LM_TRAINER_CONTEXT)

    def _suspend_session(self):
        if self._session_open and self._lost_at is None:
            try:
                exported = self._export_state_impl(release=True)
            except BaseException:
                # The native export closed the session on its way out.
                self._mark_lost()
                self._forget_session()
                raise
            self._state = exported
            self._forget_session()
            return
        self._release_session()

    def _forget_session(self):
        """Drop the Python references to a session the native side closed."""
        self._native_session = None
        self._session_binding = None
        self._session_open = False
        self._logits_pool = _LogitsPool()
        self._grad_step = -1

    def _release_session(self):
        session, binding = self._native_session, self._session_binding
        self._forget_session()
        if session is not None:
            binding.byte_lm_session_close(session)

    def _session_entries(self, extra=()):
        """Lock held. The binding, refused by name (ImportError) when it lacks
        an entry the resident session needs; there is no fallback."""
        binding = self._binding()
        missing = [name for name in _SESSION_ENTRIES + tuple(extra)  # glue: checks the session binding entry names
                   if not callable(getattr(binding, name, None))]
        if missing:
            raise ImportError('Byte-LM binding lacks owned sessions (%s); rebuild bindings/build_byte_lm.sh'
                              % ', '.join(missing))
        return binding

    def _session_handle(self, binding):
        """Lock held. The native session handle, created on first use (a
        handle only: no device work until the first call admits the state)."""
        if self._native_session is None:
            self._native_session = binding.byte_lm_session_create()
            self._session_binding = binding
            self._session_open = False
            self._grad_step = -1
        return self._native_session

    def _admission_addresses(self):
        """The host state the native call admits and uploads when the session
        is not open; zeros when it is (the device holds the only copy)."""
        if self._session_open:
            return [0, 0, 0]
        return [addr_ro(self._state[key], name='state') for key in ('parameters', 'm', 'v')]  # glue: three named state arrays

    def _adopt_open_session(self):
        """After a native call admitted the host state: the device buffers are
        now the only copy (DEVIATION 2514), so the host arrays are dropped."""
        if not self._session_open:
            self._session_open = True
            self._grad_step = -1
            self._last_export_step = self._state['completed_steps']
            self._state = dict(self._state, parameters=None, m=None, v=None)

    def _after_native_failure(self):
        """The failure path of a resident call. The native call already
        rolled the device back and judged the session (design 4.2 item 6,
        `byte_owner_recover`); this reads its verdict back. A session the
        call opened is adopted, and one that is no longer usable at the
        committed step is lost. The last gradient is no longer exportable."""
        self._grad_step = -1
        session, binding = self._native_session, self._session_binding
        if session is None or self._lost_at is not None:
            return
        try:
            info = binding.byte_lm_session_info(session)
            is_open, usable, completed = bool(info[3]), bool(info[2]), info[0]
        except Exception:
            is_open, usable, completed = self._session_open, False, None
        if is_open:
            self._adopt_open_session()
        if self._session_open and (not usable or completed != self._state['completed_steps']):
            self._mark_lost()

    def _parameters(self, train):
        """The twelve scalars every native step call reads: action, committed
        step and the optimizer configuration."""
        cfg = self._state['config']
        return [int(train), self._state['completed_steps'], cfg['kind'], cfg['lr'],
                cfg['beta1'], cfg['beta2'], cfg['eps'], cfg['weight_decay'],
                cfg['momentum'], cfg['dampening'], int(cfg['nesterov']), cfg['max_norm']]

    def load_state_dict(self, state):
        """Replace the complete training state.

        The replacement is validated before anything changes; a refused state
        leaves the trainer untouched. Any resident session is released without
        export and the next call opens one from the replacement.

        Parameters
        ----------
        state : dict
            A state from :meth:`export_state` (or :meth:`state_dict`).
        """
        with self._lock:
            replacement = _validate_state(state)
            _mode()
            self._release_session()
            self._state = replacement
            self._lost_at = None
            self._last_export_step = replacement['completed_steps']
        return self

    def _binding(self):
        binding = _load(state_shape(self._state))
        if binding is not self._runtime_binding:
            self._suspend_session()
            self._runtime = None
            self._runtime_binding = binding
        if self._runtime is None:
            # the run metadata of the binding in use; the native train_step
            # resolves the binding without it and leaves it for this read
            self._runtime = _binding_metadata(binding, state_shape(self._state))
        return binding

    def _is_resident(self):
        """Lock held. Resolve AUTO (`resident=None`, cpu2-l11-neural): the
        resident session when the binding carries every session entry, the
        stateless path otherwise. An explicit bool is returned as given."""
        # F03 M3 2026-10-06 confirms this already-enabled default: explicit
        # resident vs stateless 12-step trajectories B/A=0.5125 and0.5565,
        # vocab257/131, shape[2,7,24,3,1,8,40,2,V], seed738. Heldout loss
        # and checkpoint resume error equal; failed-update recovery captured.
        # One warmup+one scored trajectory; caller67d0efb29 and reused binary
        # provenance: ~/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/
        # results/F03/default. No compile/identity retest, no new API default.
        if self._resident is None:
            binding = self._binding()
            self._resident = all(callable(getattr(binding, name, None)) for name in _SESSION_ENTRIES)  # glue: binding entry names
        return self._resident

    def run_metadata(self):
        """Profile, configuration, schedule and binary details of this run.

        Reads constants only; runs no model computation. Keep it with any results
        you record.

        Returns
        -------
        dict
        """
        return _LM_TRAINER(self, "run_metadata", (), _LM_TRAINER_CONTEXT)

    def _run(self, ids, train):
        """`train_step` / `evaluate`: the ids are checked and copied here, and
        ONE native call does the rest (admission, step or evaluation,
        gradient export, rollback on failure)."""
        shape = state_shape(self._state)
        tokens = _array(ids, (shape.batch, shape.length + 1), 'ids', '<i4')
        try:
            if self._is_resident():
                return self._run_resident(tokens, train, shape)
            return self._run_stateless(tokens, train, shape)
        except BaseException as exc:
            if not isinstance(exc, ValueError) and _TOKEN_REFUSAL in str(exc):
                raise ValueError(f'Byte-LM IDs must be in [0, {shape.vocab_size})') from exc
            raise

    def _run_stateless(self, tokens, train, shape):
        """ONE native call (`byte_lm_trainer_run`): the binding admits the
        host state and the ids, builds and destroys one device context,
        validates the new state on the device, requires an evaluation to
        leave the state bit-identical and the inputs unchanged, and writes
        the new state, the loss and the gradient (flat and by name). The
        new state is committed only after the call returned."""
        state = self._state
        if train and state['completed_steps'] >= 999999:
            raise ValueError('Byte-LM native call step counter is exhausted')
        binding = self._binding()
        if not callable(getattr(binding, _RUN_ENTRY, None)):
            raise ImportError('Byte-LM binding lacks %s; rebuild bindings/build_byte_lm.sh' % _RUN_ENTRY)
        out_p = empty((shape.n_total,), '<f4')
        out_m = empty((shape.n_total,), '<f4')
        out_v = empty((shape.n_total,), '<f4')
        out_flags = empty((shape.n_tensors,), '<i4')
        out_loss = empty((1,), '<f4')
        flat, by_name = _gradient_outputs(shape, True) if train else (None, {})
        addresses = [*(addr_ro(state[key], name='state') for key in ('parameters', 'm', 'v', 'flags')),  # glue: four named state arrays
                     addr_ro(tokens, name='ids'), addr(out_p, name='out_p'), addr(out_m, name='out_m'),
                     addr(out_v, name='out_v'), addr(flat, name='out_grad') if train else 0,
                     addr(out_flags, name='out_flags'), addr(out_loss, name='out_loss')]
        named = [addr(by_name[name], name='out_grad') for name in by_name]  # glue: one address per model parameter tensor
        completed = getattr(binding, _RUN_ENTRY)(addresses, named, self._parameters(train),
                                                 list(shape.native_shape))
        expected = state['completed_steps'] + int(train)
        if _is_bool(completed) or not isinstance(completed, int) or completed != expected:
            raise RuntimeError('Byte-LM returned an invalid completed-step counter')
        if not train:
            return float(out_loss[0])
        self._state = dict(state, parameters=out_p, m=out_m, v=out_v, flags=out_flags,
                           completed_steps=expected, next_batch_index=expected)
        return dict(loss=float(out_loss[0]), step=expected, completed_steps=expected,
                    next_batch_index=expected, flat_gradients=flat, gradients=by_name)

    def _run_resident(self, tokens, train, shape):
        """The device-owned step (DEVIATION 2514) in ONE native call
        (`byte_lm_trainer_step`, training/byte_lm_session_ops.mojo
        `byte_owner_step`): the ids are admitted, the host state is admitted
        and uploaded when no session is open, the step or evaluation runs,
        and for 'full' the gradient is exported and split by name. Nothing
        but the ids crosses inward once the session is open; the loss, the
        flags and (for 'full') the gradient cross outward. A failure is
        rolled back on the device before it propagates; the host counter
        and flags commit only after the call returned."""
        self._require_not_lost()
        binding = self._session_entries()
        completed = self._state['completed_steps']
        if train and completed >= 999999:
            raise ValueError('Byte-LM native call step counter is exhausted')
        session = self._session_handle(binding)
        full = train and self._step_result == 'full'
        out_loss = empty((1,), '<f4')
        out_flags = empty((shape.n_tensors,), '<i4') if train else None
        flat, by_name = _gradient_outputs(shape, True) if full else (None, {})
        addresses = [addr_ro(tokens, name='ids'), addr_ro(self._state['flags'], name='flags'),
                     *self._admission_addresses(), addr(out_loss, name='out_loss'),
                     addr(out_flags, name='out_flags') if train else 0,
                     addr(flat, name='out_grad') if full else 0,
                     *(addr(by_name[name], name='out_grad') for name in by_name)]  # glue: one address per model parameter tensor
        if train:
            self._grad_step = -1
        try:
            returned = binding.byte_lm_trainer_step(session, addresses, self._parameters(train),
                                                    list(shape.native_shape))
        except BaseException:
            self._after_native_failure()
            raise
        self._adopt_open_session()
        expected = completed + int(train)
        if _is_bool(returned) or not isinstance(returned, int) or returned != expected:
            self._mark_lost()
            raise RuntimeError('Byte-LM returned an invalid completed-step counter')
        _mode()
        if not train:
            return float(out_loss[0])
        result = dict(loss=float(out_loss[0]), step=expected, completed_steps=expected,
                      next_batch_index=expected)
        if full:
            result.update(flat_gradients=flat, gradients=by_name)
        else:
            result['flags'] = out_flags.copy()
        self._state['flags'] = out_flags
        self._state['completed_steps'] = expected
        self._state['next_batch_index'] = expected
        self._grad_step = expected
        return result

    def train_step(self, ids):
        """Run one training step: mean cross-entropy loss and an AdamW update.

        Parameters
        ----------
        ids : array-like of shape (batch, length + 1), int32
            Token ids in ``[0, shape.vocab_size)``. The first ``length``
            positions predict the next ``length``; the loss averages
            ``batch * length`` targets.

        Returns
        -------
        dict
            ``loss``, ``step``, ``completed_steps`` and ``next_batch_index``. With
            ``step_result="full"`` also ``flat_gradients`` and ``gradients`` (by
            tensor name), the pre-update gradient; with ``"lean"`` also ``flags``.
        """
        # the lock, the ids, the binding and the session lifecycle and the
        # step in one native call (bindings/byte_lm_trainer_native.mojo)
        return _LM_TRAINER(self, "train_step", (ids,), _LM_TRAINER_CONTEXT)

    def evaluate(self, ids):
        """Mean loss on ``ids`` without changing any state.

        Parameters
        ----------
        ids : array-like of shape (batch, length + 1), int32
            Token ids, as for :meth:`train_step`.

        Returns
        -------
        float
        """
        return _LM_TRAINER(self, "evaluate", (ids,), _LM_TRAINER_CONTEXT)

    # Float32 logits `[batch, length, vocab]` for int32 ids `[batch,
    # length]` (DEVIATION 2658), for positions 0 to length - 1 of each row,
    # prefilled from absolute position 0.
    #
    # `lengths` (2026-09-15) makes the batch RAGGED: `batch` integers in
    # `[1, length]`, row `i` real at positions `[0, lengths[i])` and
    # padding after, whatever int32 values the padding holds. Real
    # positions' logits are byte for byte the row alone at its own length
    # (causal attention, no arithmetic changes, `_ragged.py` says why) and
    # padding positions' logits are exactly `+0.0`. `batch` must be in [1, 1024],
    # `length` in [1, shape.length], `batch * length * vocab` at most
    # 268435456 and every id in [0, shape.vocab_size); anything else is
    # refused with ValueError before any native call.
    #
    # The arithmetic is the IDENTICAL device forward the trainer's loss
    # uses, on Metal, CUDA or HIP, and these are the numbers it computes
    # before the loss. tools/byte_lm_gpu_logits_sweep.py compares them
    # byte for byte with the CPU reference path,
    # `LanguageModelInference(threaded=False)`. The native entries refuse
    # a non-finite logit with an error, because NaN payloads are shaped by
    # the vendor and cannot be part of an identity claim; the CPU class
    # returns them.
    #
    # No state changes on either path. A resident trainer runs on its
    # session (opening it first if needed, as `train_step` does) and the
    # parameters, moments, flags and step counter do not move. A failed
    # call has nothing to roll back; the last gradient is no longer
    # exportable, as after any failure, and a session that no longer
    # reports itself usable is lost. A stateless trainer validates its
    # state as `evaluate` does, builds and destroys one device context,
    # and requires the parameter and id bytes it handed over unchanged
    # afterward.
    def logits(self, ids, *, lengths=None):
        """Float32 logits for every position, without changing any state.

        Runs the same GPU forward pass the training loss uses, starting at
        position 0 of each row.

        Parameters
        ----------
        ids : array-like of shape (batch, length), int32
            Token ids in ``[0, shape.vocab_size)``. ``batch`` in [1, 1024],
            ``length`` in [1, shape.length], and ``batch * length * vocab_size`` at
            most 268435456.
        lengths : sequence of int or None, default None
            Makes the batch ragged: row ``i`` is real at positions
            ``[0, lengths[i])`` and padding after, whatever the padding ids are.
            Real positions get exactly the logits of the row alone at its own
            length; padding positions are ``+0.0``.

        Returns
        -------
        Array of shape (batch, length, vocab_size), float32

        Raises
        ------
        ValueError
            For ids or lengths outside the limits above.
        RuntimeError
            If a logit is not finite.
        """
        return _LM_TRAINER(self, "logits", (ids, lengths), _LM_TRAINER_CONTEXT)

    # The greedy next byte after each row of ids `[batch, length]`, ties
    # to the lowest byte value, picked from `logits(ids)` by the helper
    # `LanguageModelInference.next_bytes` uses (DEVIATION 2658). With
    # `lengths` (a ragged batch, see `logits`) the byte after each row's
    # LAST REAL position.
    def next_bytes(self, ids, *, lengths=None):
        """Greedy next token after each row.

        Picks the highest logit at each row's last position (with ``lengths``,
        its last real position), ties going to the lowest id.

        Parameters
        ----------
        ids : array-like of shape (batch, length), int32
            Token ids, as for :meth:`logits`.
        lengths : sequence of int or None, default None
            Ragged row lengths, as for :meth:`logits`.

        Returns
        -------
        list of int
            One token id per row.
        """
        return _LM_TRAINER(self, "next_bytes", (ids, lengths), _LM_TRAINER_CONTEXT)

    def _logits_call(self, entry_name, tokens, out, batch, length, shape):
        """Lock held, resident. ONE native call (`byte_lm_trainer_logits` or
        `byte_lm_trainer_next_bytes`, training/byte_lm_session_ops.mojo
        `byte_owner_logits`): the ids are admitted, the host state is
        admitted and uploaded when no session is open, and the forward runs
        on the session. No state is written, so a failure rolls nothing back;
        the last gradient is no longer exportable, and a session that no
        longer reports itself usable at the committed step is lost."""
        self._require_not_lost()
        binding = self._session_entries((entry_name,))
        session = self._session_handle(binding)
        addresses = [addr_ro(tokens, name='ids'), addr(out, name='out'),
                     addr_ro(self._state['flags'], name='flags'), *self._admission_addresses()]
        try:
            written = getattr(binding, entry_name)(session, addresses, [batch, length],
                                                   self._parameters(False), list(shape.native_shape))
        except BaseException:
            self._after_native_failure()
            raise
        self._adopt_open_session()
        _require_written(written, batch if entry_name == _SESSION_NEXT_ENTRY else batch * length * shape.vocab_size)
        _mode()
        return out

    def _next_bytes_resident(self, tokens, batch, length, shape):
        out = zeros((batch,), '<i4')
        return self._logits_call(_SESSION_NEXT_ENTRY, tokens, out, batch, length, shape).tolist()

    def _logits_stateless(self, tokens, batch, length, shape):
        """ONE native call (`byte_lm_logits`): the binding copies and admits
        the parameters and the ids before any device work, builds and
        destroys one device context and writes the logits."""
        binding = self._binding()
        entry = getattr(binding, _LOGITS_ENTRY, None)
        if not callable(entry):
            raise ImportError('Byte-LM binding lacks GPU logits (%s); rebuild bindings/build_byte_lm.sh'
                              % _LOGITS_ENTRY)
        out = zeros((batch, length, shape.vocab_size), '<f4')
        written = entry([addr_ro(self._state['parameters'], name='parameters'), addr_ro(tokens, name='ids'),
                         addr(out, name='logits')], [batch, length], list(shape.native_shape))
        _require_written(written, batch * length * shape.vocab_size)
        _mode()
        return out

    def _logits_resident(self, tokens, batch, length, shape):
        timing = bool(os.environ.get('MOJOLEARN_TRANSFORMER_TIMING'))
        t0 = _time.perf_counter() if timing else 0.0
        pool = getattr(self, '_logits_pool', None)
        if pool is None:
            pool = self._logits_pool = _LogitsPool()
        n_out = batch * length * shape.vocab_size
        store = pool.take(n_out)
        if store is None:
            out = _PooledLogits((batch, length, shape.vocab_size), '<f4')
            reused = 'fresh'
        else:
            out = _PooledLogits._owned(store, (batch, length, shape.vocab_size), '<f4', 'C')
            reused = 'pooled'
        out._pool = pool
        if timing:
            t1 = _time.perf_counter()
            print('timing python.logits_out %.3f ms %s' % ((t1 - t0) * 1000.0, reused), flush=True)
        self._logits_call(_SESSION_LOGITS_ENTRY, tokens, out, batch, length, shape)
        if timing:
            print('timing python.logits_binding %.3f ms' % ((_time.perf_counter() - t1) * 1000.0), flush=True)
        return out

    def save_checkpoint(self, path):
        """Same as :meth:`export_checkpoint`."""
        return self.export_checkpoint(path)

    def export_checkpoint(self, path):
        """Write the complete state to ``path`` as a JSON checkpoint.

        The file is written atomically, holds no pickle, and is limited to 2 MiB
        (about 87,000 parameters). For larger models use
        :meth:`export_checkpoint_binary`.

        Parameters
        ----------
        path : str or path-like
            Destination file.
        """
        with self._lock:
            # Refuse guaranteed-oversize saves before copying/hex-encoding state.
            if state_shape(self._state).n_total * 24 + state_shape(self._state).n_tensors * 8 > _CHECKPOINT_LIMIT:
                raise ValueError('Byte-LM checkpoint exceeds 2 MiB; export state_dict arrays')
            payload = self.export_state()
        for key in ('parameters', 'm', 'v', 'flags'):  # glue: four named checkpoint arrays
            value = payload[key]
            integer = key == 'flags'
            dtype = '<i4' if integer else '<f4'
            payload[key] = dict(dtype=dtype, shape=list(value.shape),
                                hex=le_bytes(value, 'i' if integer else 'f').hex())
        envelope = dict(schema=_CHECKPOINT_SCHEMA, payload=payload,
                        payload_sha256=_sha256_hex(_canonical(payload)))
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

    # Atomically write the complete state as a streamed BINARY archive
    # (`_byte_lm_checkpoint`, schema `mojolearn.byte-lm-stream.v1`), from
    # `export_state()`. Returns the file's SHA-256.
    #
    # THIS IS THE PATH AT SCALE, AND IT IS A SECOND FORMAT, NOT A WIDER
    # FIRST ONE. `export_checkpoint` keeps its JSON/hex envelope, its
    # bytes and its 2 MiB bound untouched; hex doubles the payload, so
    # that bound stops at about 87,381 parameters and the
    # 162,147,840-parameter shape has always been refused by it. Here the
    # four arrays are streamed as their little-endian `<f4`/`<i4` bytes
    # after a bounded canonical JSON header, so the file is about 1.95 GB
    # at that shape instead of impossible, and a round trip is bit-exact
    # by construction: nothing passes through decimal text, so `-0.0`,
    # NaN payloads and subnormals all survive.
    #
    # THE OPTIMIZER MOMENTS ARE IN THE FILE AND THAT IS THE POINT. A
    # restore that dropped `m` and `v` would produce the same loss and
    # the same gradient on its first step and a different update, so the
    # divergence would surface a step later looking like
    # nondeterminism.
    #
    # No device work beyond the one `export_state()` download a resident
    # session already owes, and no arithmetic.
    def export_checkpoint_binary(self, path):
        """Write the complete state to ``path`` as a binary checkpoint.

        For models of any size. The parameters and the AdamW moments are stored
        as raw little-endian bytes after a JSON header, so a round trip is
        bit-exact (``-0.0``, NaN payloads and subnormals survive) and a resumed
        run continues exactly. The file is written atomically.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Returns
        -------
        str
            The file's SHA-256.
        """
        with self._lock:
            payload = self.export_state()
        return _byte_lm_checkpoint.save(path, payload)

    @classmethod
    def from_checkpoint_binary(cls, path, *, resident=None):
        """Restore a trainer from :meth:`export_checkpoint_binary`.

        The whole file is verified (header and per-array digests, exact size)
        before anything is allocated, and the state is validated as
        :meth:`load_state_dict` validates it.

        Parameters
        ----------
        path : str or path-like
            The checkpoint file.
        resident : bool or None, default None
            As for the constructor.

        Returns
        -------
        LanguageModelTrainer
        """
        state = _byte_lm_checkpoint.load(path)
        shape = state_shape(state)
        cfg = state['config']
        result = cls(state['parameters'], data_schedule=state['data_schedule'], lr=cfg['lr'],
                     betas=(cfg['beta1'], cfg['beta2']), eps=cfg['eps'], weight_decay=cfg['weight_decay'],
                     shape=shape, resident=resident)
        return result.load_state_dict(state)

    @classmethod
    def from_checkpoint(cls, path, *, resident=None):
        """Restore a trainer from a JSON checkpoint written by :meth:`export_checkpoint`.

        Parameters
        ----------
        path : str or path-like
            The checkpoint file.
        resident : bool or None, default None
            As for the constructor.

        Returns
        -------
        LanguageModelTrainer
        """
        with Path(path).open('rb') as stream:
            encoded = stream.read(_CHECKPOINT_LIMIT + 1)
        return cls.from_checkpoint_bytes(encoded, resident=resident)

    @classmethod
    def from_checkpoint_bytes(cls, encoded, *, resident=None):
        """Restore a trainer from the bytes of a JSON checkpoint.

        Runs the same checks as :meth:`from_checkpoint` without opening a file.

        Parameters
        ----------
        encoded : bytes
            The checkpoint contents (exactly ``bytes``; copy mutable buffers
            first).
        resident : bool or None, default None
            As for the constructor.

        Returns
        -------
        LanguageModelTrainer
        """
        state, shape = _decode_checkpoint(encoded)
        cfg = state['config']
        result = cls(state['parameters'], data_schedule=state['data_schedule'], lr=cfg['lr'],
                     betas=(cfg['beta1'], cfg['beta2']), eps=cfg['eps'], weight_decay=cfg['weight_decay'],
                     shape=shape, resident=resident)
        return result.load_state_dict(state)


def _decode_checkpoint(encoded):
    """`(state, shape)` from checkpoint bytes, every schema, integrity,
    tensor, optimizer and cursor check applied, no native call. Shared by
    `from_checkpoint_bytes` and the CPU inference loader (DEVIATION 2610),
    so there is one decoder."""
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
    if _sha256_hex(_canonical(payload)) != envelope['payload_sha256']:
        raise ValueError('Byte-LM checkpoint integrity mismatch')
    if not isinstance(payload, dict):
        raise ValueError('Byte-LM checkpoint payload must be an object')
    shape = state_shape(payload)
    for key in ('parameters', 'm', 'v', 'flags'):  # glue: four named checkpoint arrays
        value = payload.get(key)
        cells, dtype = (shape.n_tensors, '<i4') if key == 'flags' else (shape.n_total, '<f4')
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
    return _validate_state(payload), shape


def _unique_object(pairs):
    result = {}
    for key, value in pairs:  # glue: walks the checkpoint key value pairs
        if key in result:
            raise ValueError('Duplicate byte-LM checkpoint key')
        result[key] = value
    return result


def _connect_trainer():
    # glue: connect the GPU trainer's native step owner and its fixed
    # metadata once, at package import
    from .tokenizer import _binding as _tokenizer_binding
    from ._native_classical_bridge import _state_context
    from types import MethodType
    context = _state_context()
    _numeric_profile.default_profile("training")  # glue: connect the training profile before public construction
    context.update(byte_lm_source_root=str(Path(__file__).resolve().parents[2]), byte_lm_wrapper_file=__file__,
                   byte_lm_source_names=('python/mojolearn/_byte_lm_impl.py', 'python/mojolearn/language_model.py',
                                        'bindings/_mojolearn_byte_lm.mojo', 'training/byte_lm.mojo',
                                        'training/byte_lm_config.mojo', 'python/mojolearn/_byte_lm_config.py'))
    context.update(byte_lm_trainer_class=SmallByteLanguageModelTrainer, byte_lm_config_class=ByteLanguageModelConfig,
                   byte_lm_profile_defaults=_numeric_profile._defaults, byte_lm_profiles=_numeric_profile.PROFILES,
                   byte_lm_lock=type(threading.RLock()))
    _backend.default_mode()  # glue: resolve mode before any native product operation
    bindings = {}
    for mode in _NATIVE_MODE_CODE:  # glue: the two numeric modes, connected once
        try:
            bindings[mode] = _backend.binding(_EXTENSION, mode)
        except (ImportError, AttributeError, ValueError, RuntimeError, TypeError, OSError) as error:
            bindings[mode] = type(error).__name__ + ": " + str(error)
    context.update(byte_lm_bindings=bindings,
                   byte_lm_vendors=('cpu',) if _backend._CPU_ONLY is not None else ('cuda', 'hip', 'metal'),
                   byte_lm_profile=PROFILE, byte_lm_session_entries=_SESSION_ENTRIES,
                   byte_lm_run_entry=_RUN_ENTRY, byte_lm_logits_pool=_LogitsPool, backend=vars(_backend))
    module = _tokenizer_binding()._module
    pooled_context = dict(context, array_class=_PooledLogits)
    context.update(byte_lm_pooled_class=_PooledLogits, byte_lm_pooled_context=pooled_context,
                   byte_lm_store_getter=vars(_Array)["_store"].__get__)
    _PooledLogits.__del__ = module.classical_attribute_descriptor(
        MethodType(module.byte_lm_pooled_finalize_call, context))
    # glue: pin the existing native slot and exact pooled storage hooks once.
    # Native logits checks replacements before allocation, then reads this slot
    # once; Array.__array_interface__ is never an output-address fallback.
    from types import MemberDescriptorType
    address_slot = vars(_Array)["_addr"]
    if type(address_slot) is not MemberDescriptorType:
        raise TypeError("Byte-LM pooled logits require the native Array address slot")
    mro = _PooledLogits.__mro__
    hooks = ('__getattribute__', '__getattr__', '__setattr__', '__del__', '__dict__',
             '__array_interface__', '_pool', '_store', '_base', '_pin', '_mv', '_addr',
             '_readonly', 'shape', 'dtype', 'ndim', 'size', 'nbytes', 'itemsize', 'strides', 'order')
    layout = tuple((base, tuple((name, vars(base).get(name)) for name in hooks))  # glue: fixed native storage hooks
                   for base in mro)  # glue: exact connected pooled Array ancestry
    context.update(byte_lm_addr_getter=address_slot.__get__, byte_lm_pooled_mro=mro,
                   byte_lm_pooled_layout=layout,
                   byte_lm_type_mro_getter=type.__dict__['__mro__'].__get__,
                   byte_lm_type_dict_getter=type.__dict__['__dict__'].__get__)
    return module.byte_lm_trainer_call, context


_LM_TRAINER, _LM_TRAINER_CONTEXT = _connect_trainer()
