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
`_state['parameters']`, `['m']` and `['v']` are `None` from `_open_session`
until `close()`, `load_state_dict()` or a lost session. State crosses the
boundary at admission (`_open_session`: `_validate_state` once, one
upload), at export (`export_state`, `export_gradients`, `export_checkpoint`,
`close`) and never per step; a step moves the ids in and the loss and the
flags out. The stateless path (`resident=False`) is unchanged: mirror in,
mirror out, validated candidate committed or nothing. Since cpu2-l11-neural
(2026-10-04) `resident` defaults to True on a GPU install (the stateless
path is an explicit opt-out and the route of a binding without session
entries); a defaulted session keeps `step_result='full'`.

GPU LOGITS (DEVIATION 2658). `logits(ids)` and `next_bytes(ids)` run the
forward alone, through the native `byte_lm_logits` (stateless) and
`byte_lm_session_logits` (resident) entries, and return the numbers the
trainer's loss is computed from. They write no state on either path.
tools/byte_lm_gpu_logits_sweep.py compares them byte for byte with the CPU
reference path of DEVIATION 2610, `LanguageModelInference(threaded=False)`.
"""
from . import _buffer as _buffers, _bufcheck as _checks
from ._array import Array as _Array
import hashlib
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
import time

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
#: missing any of them refuses with ImportError; there is no fallback to
#: the mirror-in/mirror-out `byte_lm_session_run`.
#: The native refusal of an id outside [0, vocab) (training/byte_lm.mojo
#: `byte_validate_tokens`), mapped back to the public ValueError.
_TOKEN_REFUSAL = 'token ID outside configured vocabulary'

_SESSION_ENTRIES = ('byte_lm_session_create', 'byte_lm_session_close',
                    'byte_lm_session_open', 'byte_lm_session_step',
                    'byte_lm_session_eval', 'byte_lm_session_export_state',
                    'byte_lm_session_export_gradients', 'byte_lm_session_rollback',
                    'byte_lm_session_info')
#: DEVIATION 2658 names the native logits entries and the limits they
#: admit; `logits` refuses anything outside the limits here first.
_LOGITS_ENTRY = 'byte_lm_logits'
_SESSION_LOGITS_ENTRY = 'byte_lm_session_logits'
#: cpu3-seq (2026-10-04): the resident greedy next byte, argmax on the device.
_SESSION_NEXT_ENTRY = 'byte_lm_session_next_bytes'
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
#: and promises quality, never bits. DETERMINISTIC is the tree lanes' tier.
_NATIVE_MODE_CODE = {'identical': 1, 'fast': 0}


def _mode():
    """The process-selected tier ('identical' or 'fast'), or raise."""
    mode = _backend.default_mode()
    if mode not in _NATIVE_MODE_CODE or _backend.numeric_mode() != mode:
        raise RuntimeError('SmallByteLanguageModelTrainer requires process-selected IDENTICAL or FAST mode')
    return mode


def _timing_on():
    """DEVIATION 2499: the Python side of the step-phase timers, behind the
    SAME switch as the native block and step timers
    (MOJOLEARN_TRANSFORMER_TIMING=1). One environment lookup per call when
    off; nothing else."""
    return bool(os.environ.get('MOJOLEARN_TRANSFORMER_TIMING'))


def _tick(on, clock, name, n_bytes=None):
    """Print `timing <name> <ms> ms` (the native line shape) for the time
    since `clock[0]`, then advance it; with `n_bytes`, also print
    `timing <name>_bytes <n> bytes`. Every phase this brackets is host
    work; the native call is bounded by the binding's own final wait."""
    if not on:
        return
    now = time.perf_counter()
    print('timing %s %s ms' % (name, (now - clock[0]) * 1000.0), flush=True)
    if n_bytes is not None:
        print('timing %s_bytes %d bytes' % (name, n_bytes), flush=True)
    clock[0] = now


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
    if (int(binding.byte_lm_numeric_mode()) != _NATIVE_MODE_CODE[mode]
            or str(binding.byte_lm_profile()) != PROFILE
            or str(binding.byte_lm_vendor()) not in vendors):
        raise RuntimeError('Byte-LM requires the exact native profile, the selected IDENTICAL/FAST mode and CUDA/HIP/Metal vendor '
                           '(or, on a CPU-only install, the CPU byte LM binding)')
    if not callable(getattr(binding, 'byte_lm_run', None)):
        raise ImportError('Byte-LM binding is missing byte_lm_run; rebuild bindings/build_byte_lm.sh')
    if shape.profile != PROFILE:
        if not callable(getattr(binding, 'byte_lm_run_configured', None)) or not callable(getattr(binding, 'byte_lm_config_profile', None)):
            raise ImportError('Byte-LM binding lacks runtime shapes; rebuild bindings/build_byte_lm.sh')
        if str(binding.byte_lm_config_profile(list(shape.native_shape))) != shape.profile:
            raise RuntimeError('Byte-LM native runtime shape/profile mismatch')
    return binding


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
    if (value['schema'] != _SCHEMA or value['profile'] != shape.profile or value['numeric_mode'] not in _NATIVE_MODE_CODE
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


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):  # cpu-route: reads a binding file in chunks to hash it (file input)
            digest.update(chunk)
    return digest.hexdigest()


def _binding_metadata(binding, shape):
    root = Path(__file__).resolve().parents[2]
    names = ('python/mojolearn/_byte_lm_impl.py', 'python/mojolearn/language_model.py',
             'bindings/_mojolearn_byte_lm.mojo', 'training/byte_lm.mojo',
             'training/byte_lm_config.mojo', 'python/mojolearn/_byte_lm_config.py')
    inventory = {name: _sha(root / name) for name in names if (root / name).is_file()}  # cpu-route: hashes the binding files (file input)
    # Installed wheels may lack native sources; the Python source and exact
    # loaded binding are still identified. Do not claim a full source audit.
    inventory['loaded_python_wrapper'] = _sha(__file__)
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
    return dict(binding_file=binding.__file__, binding_sha256=_sha(binding.__file__),
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


def _require_written(written, expected):
    if _is_bool(written) or not isinstance(written, int) or written != expected:
        raise RuntimeError('Byte-LM native logits wrote an unexpected number of logits')


class SmallByteLanguageModelTrainer:
    """FP32 decoder language-model trainer with runtime dimensions.

    Defaults B2/L32/DM32/H4/KV2/FF64/V256; pass ByteLanguageModelConfig
    as shape to configure dimensions, layer count and vocabulary.
    no biases/dropout/final norm, untied embedding/head. Supply a flat FP32
    array or the configured named tensors exposed by parameter_registry(shape). There
    is no hidden initialization, tokenizer, padding, truncation or RNG.

    logits(ids)/next_bytes(ids) (DEVIATION 2658) run the same device forward
    alone on int32[batch, length] IDs and change no state; the logits are the
    numbers the loss is computed from, and next_bytes picks greedily with
    ties going to the lowest byte value.

    train_step/evaluate require actual int32[B,L+1] IDs in [0, shape.vocab_size).
    The first L positions predict the next L; loss averages B*L targets. All arithmetic runs in the native CUDA/HIP/Metal
    profile of the process-selected tier: IDENTICAL (default, same bits on every vendor) or FAST (quality, never bits).

    data_schedule is a bounded caller-supplied JSON descriptor. Retain corpus
    and actual token-order SHA256, batch offsets, and planned steps there.
    next_batch_index equals completed_steps; callers must provide the matching
    next batch after restoration. This class does not fetch/reorder a corpus.

    NOTHING VALIDATES ITS CONTENTS, and a reader of a published run has to
    know that (lane/data-ordering-determinism, 2026-09-16). The descriptor is
    checked for size, depth, key count and JSON round-trip and for nothing
    else: no key is required, no value is compared against the IDs handed to
    train_step, and {'dataset': 'test'} trains. What it buys is that once a
    checkpoint is written the descriptor is covered by the envelope's
    payload_sha256, so it cannot be swapped without detection. That makes it
    TAMPER-EVIDENT, not TRUE. The run is reproducible from it only to the
    extent the caller made it so, and this class cannot tell the difference.

    resident=True retains an owned native context/model/optimizer across
    calls, and (DEVIATION 2514) the device buffers are then the ONLY copy of
    the parameters, moments and last gradient while the session is open:
    nothing is mirrored to the host per step. The state is validated once at
    admission (the first call after construction, `load_state_dict()` or
    `close()`), validated on the device after every update, and validated
    again when it is exported. `export_state()` (also `state_dict()` and
    `parameters_`), `export_gradients()` and `export_checkpoint()` copy it
    out; `close()` exports first and then releases the device, so the next
    call re-admits from host state. A step that fails after its update is
    rolled back on the device; a failure the Python layer detects after
    native success is rolled back through `byte_lm_session_rollback`. What
    is weaker than the stateless path: a LOST context (a rollback whose
    re-scan does not answer) loses the steps since the last export; the
    object then raises "session lost at step k; last exported state is step
    j" on every call until `load_state_dict()`.

    step_result selects what `train_step` returns. 'full' returns the
    complete dict (loss, gradients, state); on a resident session it is the
    lean step plus `export_gradients()`, so the bytes are the stateless
    path's. 'lean' (resident only) returns `loss, step, completed_steps,
    next_batch_index, flags` and leaves the gradient on the device until
    `export_gradients()`. The default (None) is 'lean' for a resident
    trainer and 'full' for a stateless one: DEVIATION 2514 gate G5 on the
    H100 (2026-09-11) measured a lean target-shape step at 0.565 s against
    8.6 s full and 45 s before the device-owned step, with every lean step
    bit-equal to the full path's and to the pre-change trainer's.

    Complete state is copied and updates commit only after success. On the
    stateless path evaluation requires byte-unchanged parameters, moments,
    flags and counter; on a resident session that invariance is a native
    gate (design G1), not a per-call download. Public checkpoints use an
    explicitly separate bounded JSON/hex schema; they are not
    training/checkpoint.mojo's native binary v1 files. The JSON codec
    retains its 2 MiB bound; larger models export `export_state()` arrays.
    Learning, numerical correctness and cross-vendor identity remain
    unqualified in this slice.
    """

    def __init__(self, parameters, *, data_schedule, lr=1e-3, betas=(.9, .999),
                 eps=1e-8, weight_decay=.01, shape=None, resident=None, step_result=None):
        # A trainer runs only under a numeric profile whose TRAINING gates have passed.
        _numeric_profile.require_training("mojolearn.SmallByteLanguageModelTrainer")
        # cpu2-l11-neural (2026-10-04): `resident=None` (the default) is
        # AUTO: the device-owned session whenever the loaded binding carries
        # the session entries (every current GPU build), so no step mirrors,
        # copies or re-scans the parameters and moments on the host. It is
        # resolved at the first call that needs the binding (`_is_resident`).
        # `resident=False` keeps the stateless path as an explicit opt-out.
        if resident is not None and type(resident) is not bool:
            raise TypeError("resident must be a bool")
        if step_result is None:
            # AUTO keeps the stateless return shape ('full').
            step_result = 'lean' if resident is True else 'full'
        if type(step_result) is not str or step_result not in ('full', 'lean'):
            raise ValueError("step_result must be 'full' or 'lean'")
        if step_result == 'lean' and resident is not True:
            raise ValueError("step_result='lean' requires resident=True: the stateless path "
                             "destroys its context before returning, so there is nothing to export from")
        self._resident = resident
        self._step_result = step_result
        self._native_session = None
        self._session_binding = None
        self._session_open = False
        self._logits_pool = _LogitsPool()
        self._grad_step = -1
        self._last_export_step = 0
        self._lost_at = None
        shape = require_shape(shape)
        flat = _parameters(parameters, shape)
        config = _configuration(lr, betas, eps, weight_decay)
        descriptor = _schedule(data_schedule)
        _require_schedule_vocabulary(descriptor, shape)
        mode = _mode()
        self._lock = threading.RLock()
        self._runtime = None
        self._runtime_binding = None
        self._state = dict(schema=_SCHEMA, profile=shape.profile, numeric_mode=mode,
                           parameter_names=list(shape.parameter_names),
                           parameter_shapes=[list(shape) for shape in shape.parameter_shapes],  # glue: lists the model parameter shapes
                           parameter_offsets=list(shape.offsets), parameters=flat,
                           m=_buffers.zeros(shape.n_total, '<f4'), v=_buffers.zeros(shape.n_total, '<f4'),
                           flags=_buffers.zeros(shape.n_tensors, '<i4'), completed_steps=0, next_batch_index=0,
                           config=config, data_schedule=descriptor)
        if shape.profile != PROFILE:
            self._state['model_shape'] = shape.to_dict()

    @property
    def data_schedule(self):
        """A copy of the caller's `data_schedule` descriptor, as every
        checkpoint of this trainer keeps it (`mojolearn.lm_corpus.
        tokenizer_for` reads its `vocabulary`)."""
        return json.loads(json.dumps(self._state['data_schedule']))

    @staticmethod
    def parameter_registry(shape=None):
        shape = require_shape(shape)
        return [{'name': name, 'shape': tensor_shape, 'offset': shape.offsets[index],
                 'size': shape.offsets[index + 1] - shape.offsets[index]}
                for index, (name, tensor_shape) in enumerate(zip(shape.parameter_names, shape.parameter_shapes))]  # glue: one registry entry per named parameter

    @property
    def step_(self):
        with self._lock:
            return self._state['completed_steps']

    @property
    def parameters_(self):
        with self._lock:
            if self._session_open:
                return self.export_state()['parameters']
            self._require_not_lost()
            return self._state['parameters'].copy()

    def state_dict(self):
        """`export_state()`: fresh copies of the complete state. On an open
        resident session this downloads the device state (3n floats)."""
        return self.export_state()

    def _require_not_lost(self):
        if self._lost_at is not None:
            raise RuntimeError('Byte-LM session lost at step %d; last exported state is step %d'
                               % self._lost_at)

    def _mark_lost(self):
        if self._lost_at is None:
            self._lost_at = (self._state['completed_steps'], self._last_export_step)

    def export_state(self):
        """The complete state as fresh `mojolearn.Array` copies plus metadata
        (the `state_dict()` shape). On an open resident session: the device
        state is re-scanned natively, downloaded once and validated here by
        the same `_validate_state` that admits an import; on a closed or
        stateless trainer the host arrays are copied. Mutating the returned
        arrays cannot reach the device (design section 3)."""
        with self._lock:
            self._require_not_lost()
            if self._session_open:
                return _snapshot(self._export_state_impl(), copy=False)
            return _snapshot(self._state)

    def export_gradients(self, named=True):
        """`dict(flat_gradients=Array[, gradients={name: Array}])`: the
        pre-update gradient of the LAST completed step of an open resident
        session, valid until the next `train_step` starts (the device
        gradient is scratch for the next backward). RuntimeError if no step
        has completed since open, restore, import or rollback. `named=False`
        skips the per-tensor slicing. The stateless path returns its
        gradients from `train_step` and has nothing to export."""
        with self._lock:
            self._require_not_lost()
            if not self._session_open:
                raise RuntimeError('Byte-LM export_gradients requires an open resident session; '
                                   'the stateless train_step returns its gradients')
            completed = self._state['completed_steps']
            if self._grad_step != completed:
                raise RuntimeError('Byte-LM has no gradient to export: no step has completed '
                                   'since open, restore, import or rollback')
            return self._export_gradients_impl(completed, named)

    def attention_stage_report(self):
        """DEVIATION 3010: what the OPEN resident session's quadratic
        attention stages are holding RIGHT NOW. `None` when no session is
        open, or when the loaded binding predates this report.

        Keys: `forward_eager_cells`, `forward_aexp_cells`,
        `backward_eager_cells` and the matching `*_bytes` at four bytes a
        cell; `layers_grown_forward`, `layers_grown_backward`,
        `layers_full_aexp`, `layers`; `eager_bytes` (forward eager +
        backward eager, the two that only the eager fallback grows) and
        `total_bytes` (those plus `aexp`). New bindings also report per-layer
        `forward_status`, `backward_status`, their named `*_counts`, and
        `attn_materialized` after the latest step. Status -1 means not
        attempted; 0 ran, 1 refused regime, 2 corner, 3 DEVIATION 3110's
        latch (this layer refused before, so nothing was launched at all and
        the eager path ran alone). Materialization may
        have occurred during backward recomputation. Retained capacity does
        not say which path ran this step.

        WHAT IT IS FOR. The ten `[B, n_heads, L, S]` attention arrays are
        allocated at ONE element for the fused path and GROW ON DEMAND the
        first time a layer takes the eager path -- a refused regime, or a
        `FUSED_CORNER` hit, both of which depend on the DATA and so on the
        step. With the legacy retention policy they stay for the session.
        `release_eager` builds release dead forward scratch after forward
        and remaining eager stages after each layer backward; the report's
        `released_eager_bytes` witnesses that work separately from capacity.
        `sticky_eager` and `layers_prefer_eager` report the policy which
        chooses eager before launch after a layer's first corner refusal.
        `repair_masked_tail` identifies the replay build; per-layer
        `backward_repair_sites` is a bitmask of estash repairs actually executed:
        1 for zdot, 2 for dQ, 3 for both, 0 for neither.
        A legacy run therefore has no single device footprint: it can step up
        once, at a step nobody chose, and every capacity figure taken
        before that step is wrong afterwards. A three-step probe cannot
        see it. Read this between steps and the step where it moved is
        named rather than inferred.

        `forward_aexp_cells` is reported APART from the other forward
        arrays because two unrelated mechanisms grow `aexp`: the eager
        fallback grows all four together, while a build carrying the
        fused exp stash grows `aexp` alone on a call that refused
        nothing. `eager_bytes` excludes it for exactly that reason.

        Host metadata only: buffer lengths and saved launcher statuses. Nothing is launched, downloaded or
        synchronized, and no step path calls this."""
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

    def _export_state_impl(self):
        """Lock held, session open. Returns a validated state dict whose
        arrays are fresh owned copies (the `_validate_state` copies of the
        arrays the native export wrote)."""
        shape = state_shape(self._state)
        binding = self._session_binding
        out_p = _buffers.full(shape.n_total, float("nan"), '<f4')
        out_m = _buffers.full(shape.n_total, float("nan"), '<f4')
        out_v = _buffers.full(shape.n_total, float("nan"), '<f4')
        out_flags = _buffers.full(shape.n_tensors, -1, '<i4')
        addresses = [addr(out_p, name='out_p'), addr(out_m, name='out_m'),
                     addr(out_v, name='out_v'), addr(out_flags, name='out_flags')]
        returned = binding.byte_lm_session_export_state(self._native_session, addresses,
                                                        list(shape.native_shape))
        completed = self._state['completed_steps']
        if _is_bool(returned) or not isinstance(returned, int) or returned != completed:
            raise RuntimeError('Byte-LM state export returned a different completed step')
        exported = _validate_state(dict(self._state, parameters=out_p, m=out_m, v=out_v, flags=out_flags))
        if exported['flags'].tobytes() != self._state['flags'].tobytes():
            raise RuntimeError('Byte-LM exported flags differ from the admitted flags')
        _mode()
        self._last_export_step = completed
        return exported

    def _export_gradients_impl(self, step, named):
        shape = state_shape(self._state)
        binding = self._session_binding
        out_grad = _buffers.full(shape.n_total, float("nan"), '<f4')
        returned = binding.byte_lm_session_export_gradients(self._native_session,
                                                            [addr(out_grad, name='out_grad')],
                                                            list(shape.native_shape))
        if _is_bool(returned) or not isinstance(returned, int) or returned != step:
            raise RuntimeError('Byte-LM gradient export returned a different step')
        # cpu2-l11-neural: the native export already ran the finite scan on
        # the device (`first_nonfinite`) and wrote a fresh owned buffer, so
        # no host re-scan or second copy.
        gradients = out_grad
        result = dict(flat_gradients=gradients)
        if named:
            result['gradients'] = {
                name: gradients[shape.offsets[index]:shape.offsets[index + 1]].reshape(tensor_shape).copy()
                for index, (name, tensor_shape) in enumerate(zip(shape.parameter_names, shape.parameter_shapes))}  # glue: one gradient view per named parameter
        return result

    def close(self):
        """Export the device state to the host (one download, validated as an
        import is), then release the device; a later call re-admits from
        that host state. On a lost session nothing can be exported: the
        device is released and the object stays lost until
        `load_state_dict()`."""
        with self._lock:
            self._suspend_session()

    def _suspend_session(self):
        if self._session_open and self._lost_at is None:
            try:
                exported = self._export_state_impl()
            except BaseException:
                self._mark_lost()
                self._release_session()
                raise
            self._state = exported
        self._release_session()

    def _release_session(self):
        session, binding = self._native_session, self._session_binding
        self._native_session = None
        self._session_binding = None
        self._session_open = False
        self._logits_pool = _LogitsPool()
        self._grad_step = -1
        if session is not None:
            binding.byte_lm_session_close(session)

    def _open_session(self, binding, shape):
        """Admission (design 1.1 item 1): `_validate_state` once, one upload.
        On success the host arrays are dropped; on any failure the live
        host state is untouched and no session is retained."""
        working = _validate_state(self._state)
        cfg = working['config']
        inputs = [working['parameters'], working['m'], working['v'], working['flags']]
        addresses = [addr_ro(value, name='state') for value in inputs]  # glue: addresses of the session input buffers
        parameters = [0, working['completed_steps'], cfg['kind'], cfg['lr'],
                      cfg['beta1'], cfg['beta2'], cfg['eps'], cfg['weight_decay'],
                      cfg['momentum'], cfg['dampening'], int(cfg['nesterov']), cfg['max_norm']]
        session = binding.byte_lm_session_create()
        try:
            returned = binding.byte_lm_session_open(session, addresses, parameters, list(shape.native_shape))
            if _is_bool(returned) or not isinstance(returned, int) or returned != working['completed_steps']:
                raise RuntimeError('Byte-LM session admission returned a different completed step')
        except BaseException:
            try:
                binding.byte_lm_session_close(session)
            except Exception:
                pass
            raise
        self._native_session = session
        self._session_binding = binding
        self._session_open = True
        self._grad_step = -1
        self._last_export_step = working['completed_steps']
        self._state = dict(working, parameters=None, m=None, v=None)

    def _recover_session(self):
        """The `_run` failure path on a resident session (design 4.2 item 6):
        roll the device back to the last committed step. The native
        rollback is a no-op when the trainer already rolled back inside the
        step or the failure preceded its update. A session that reports
        itself unusable, or whose step after rollback is not the committed
        one, is lost."""
        if not self._session_open or self._lost_at is not None:
            return
        binding = self._session_binding
        self._grad_step = -1
        try:
            completed = binding.byte_lm_session_rollback(self._native_session)
            info = binding.byte_lm_session_info(self._native_session)
            usable = bool(info[2])
        except Exception:
            completed, usable = None, False
        if not usable or completed != self._state['completed_steps']:
            self._mark_lost()

    def load_state_dict(self, state):
        """Replace full state and its validated model shape atomically. The
        replacement is validated BEFORE the live session is released; a
        refused restore leaves the session untouched. Any resident state is
        released without export (the replacement supersedes it) and the
        next call admits the replacement."""
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
            self._runtime = _binding_metadata(binding, state_shape(self._state))
            self._runtime_binding = binding
        return binding

    def _is_resident(self):
        """Lock held. Resolve AUTO (`resident=None`, cpu2-l11-neural): the
        resident session when the binding carries every session entry, the
        stateless path otherwise. An explicit bool is returned as given."""
        if self._resident is None:
            binding = self._binding()
            self._resident = all(callable(getattr(binding, name, None)) for name in _SESSION_ENTRIES)  # glue: binding entry names
        return self._resident

    def run_metadata(self):
        """Exact current profile/config/schedule/source/binary runtime witness.

        Reads binding constants only, without a model execution. Call before
        a capture and retain the returned object alongside its raw arrays.
        """
        with self._lock:
            self._binding()
            return dict(json.loads(_canonical(self._runtime)), schema='small-byte-lm.run-metadata.v1',
                        qualification='authored/unqualified', profile=self._state['profile'],
                        device_state_lifetime='resident' if self._is_resident() else 'call',
                        step_result=self._step_result, last_export_step=self._last_export_step,
                        config=dict(self._state['config']), data_schedule=_schedule(self._state['data_schedule']),
                        completed_steps=self.step_, next_batch_index=self.step_)

    def _run(self, ids, train):
        try:
            return self._run_impl(ids, train)
        except BaseException as exc:
            # Native success followed by Python validation failure, or a
            # native failure after the update: roll the device back to the
            # last committed step and keep the session. Host state has not
            # committed. (The stateless path has nothing to recover.)
            try:
                self._recover_session()
            except Exception:
                pass  # Preserve the original failure; the session is marked lost.
            if not isinstance(exc, ValueError) and _TOKEN_REFUSAL in str(exc):
                shape = state_shape(self._state)
                raise ValueError(f'Byte-LM IDs must be in [0, {shape.vocab_size})') from exc
            raise

    def _run_impl(self, ids, train):
        shape = state_shape(self._state)
        ton = _timing_on()
        clock = [time.perf_counter()]
        n4 = shape.n_total * 4
        tokens = _array(ids, (shape.batch, shape.length + 1), 'ids', '<i4')
        # cpu2-l11-neural: no Python min/max over the ids. The native entry
        # admits them (`byte_validate_tokens`, the same [0, vocab) rule)
        # before any upload; `_run` maps its refusal to this ValueError.
        _tick(ton, clock, 'step.py_tokens', tokens.nbytes)
        if self._is_resident():
            return self._run_resident(tokens, train, shape, ton, clock)
        working = _validate_state(self._state)
        if train and working['completed_steps'] >= 999999:
            raise ValueError('Byte-LM native call step counter is exhausted')
        # Three n-float copies plus the v >= 0 and flags scans.
        _tick(ton, clock, 'step.py_validate_state', 3 * n4)
        binding = self._binding()
        inputs = [working['parameters'], working['m'], working['v'], working['flags'], tokens]
        before = tuple(array.tobytes() for array in inputs)
        _tick(ton, clock, 'step.py_before_bytes', 3 * n4)
        out_p = _buffers.full(shape.n_total, float("nan"), '<f4')
        out_m = _buffers.full(shape.n_total, float("nan"), '<f4')
        out_v = _buffers.full(shape.n_total, float("nan"), '<f4')
        out_flags = _buffers.full(shape.n_tensors, -1, '<i4')
        out_loss = _buffers.full(1, float("nan"), '<f4')
        out_grad = _buffers.full(shape.n_total, float("nan"), '<f4') if train else None
        _tick(ton, clock, 'step.py_alloc_outputs', (3 + int(train)) * n4)
        cfg = working['config']
        addresses = [*(addr_ro(value, name='input') for value in inputs),  # glue: addresses of the step input buffers
                     addr(out_p, name='out_p'), addr(out_m, name='out_m'),
                     addr(out_v, name='out_v'),
                     addr(out_grad, name='out_grad') if train else 0,
                     addr(out_flags, name='out_flags'), addr(out_loss, name='out_loss')]
        parameters = [int(train), working['completed_steps'], cfg['kind'], cfg['lr'],
                      cfg['beta1'], cfg['beta2'], cfg['eps'], cfg['weight_decay'],
                      cfg['momentum'], cfg['dampening'], int(cfg['nesterov']), cfg['max_norm']]
        if shape.profile == PROFILE:
            completed = binding.byte_lm_run(addresses, parameters)
        else:
            completed = binding.byte_lm_run_configured(addresses, parameters, list(shape.native_shape))
        # The whole native call, itemized by the `step.bind_*`, `step.*`,
        # `block.*`, `attn.*` and `bwd.*` lines the binding printed; an
        # envelope, not a phase, so the probe keeps it out of the sum.
        _tick(ton, clock, 'envelope.native_call')
        expected = working['completed_steps'] + int(train)
        if _is_bool(completed) or not isinstance(completed, int) or completed != expected:
            raise RuntimeError('Byte-LM returned an invalid completed-step counter')
        # Compare one snapshot at a time: constructing a second tuple keeps
        # all three parameter-sized byte copies alive simultaneously.
        if any(saved != array.tobytes() for saved, array in zip(before, inputs)):
            raise RuntimeError('Byte-LM native call changed an input state/token buffer')
        _tick(ton, clock, 'step.py_input_unchanged', 3 * n4)
        if not all_finite(out_loss):
            raise RuntimeError('Byte-LM returned a nonfinite/unwritten loss')
        candidate = _validate_state(dict(working, parameters=out_p, m=out_m, v=out_v, flags=out_flags,
                                         completed_steps=expected, next_batch_index=expected))
        _mode()
        # Three n-float copies of the outputs plus their scans.
        _tick(ton, clock, 'step.py_candidate_state', 3 * n4)
        if not train:
            if any(candidate[key].tobytes() != working[key].tobytes()
                   for key in ('parameters', 'm', 'v', 'flags')):  # glue: four named state arrays
                raise RuntimeError('Byte-LM evaluation changed full training state')
            _tick(ton, clock, 'step.py_eval_unchanged', 3 * n4)
            return float(out_loss[0])
        gradients = _array(out_grad, (shape.n_total,), 'pre-update gradients')
        # One n-float copy plus all_finite.
        _tick(ton, clock, 'step.py_gradients_array', n4)
        result = dict(loss=float(out_loss[0]), step=expected, completed_steps=expected,
                      next_batch_index=expected, flat_gradients=gradients,
                      gradients={name: gradients[shape.offsets[index]:shape.offsets[index + 1]].reshape(tensor_shape).copy()
                                 for index, (name, tensor_shape) in enumerate(zip(shape.parameter_names, shape.parameter_shapes))})  # glue: one view per named parameter
        self._state = candidate
        # The per-tensor gradient dict: one more n-float copy in slices.
        _tick(ton, clock, 'step.py_gradients_dict', n4)
        return result

    def _run_resident(self, tokens, train, shape, ton, clock):
        """The device-owned step (DEVIATION 2514). Nothing but the ids
        crosses inward; the loss and the flags cross outward. The scalar
        admission (profile, step, optimizer bits, flags) is native and per
        step. Commit order: every post-check, then (for 'full') the gradient
        export, then the host counter and flags; any raise before the commit
        reaches `_run`'s rollback with nothing committed."""
        self._require_not_lost()
        binding = self._binding()
        missing = [name for name in _SESSION_ENTRIES if not callable(getattr(binding, name, None))]  # glue: checks the session binding entries
        if missing:
            raise ImportError('Byte-LM binding lacks owned sessions (%s); rebuild bindings/build_byte_lm.sh'
                              % ', '.join(missing))
        if train and self._state['completed_steps'] >= 999999:
            raise ValueError('Byte-LM native call step counter is exhausted')
        if not self._session_open:
            # Admission: `_validate_state` once, one upload (3n floats).
            self._open_session(binding, shape)
            _tick(ton, clock, 'step.py_open_session', 3 * shape.n_total * 4)
        cfg = self._state['config']
        completed = self._state['completed_steps']
        flags = self._state['flags']
        expected = completed + int(train)
        out_loss = _buffers.full(1, float("nan"), '<f4')
        parameters = [int(train), completed, cfg['kind'], cfg['lr'],
                      cfg['beta1'], cfg['beta2'], cfg['eps'], cfg['weight_decay'],
                      cfg['momentum'], cfg['dampening'], int(cfg['nesterov']), cfg['max_norm']]
        native_shape = list(shape.native_shape)
        if train:
            out_flags = _buffers.full(shape.n_tensors, -1, '<i4')
            addresses = [addr_ro(tokens, name='ids'), addr_ro(flags, name='flags'),
                         addr(out_loss, name='out_loss'), addr(out_flags, name='out_flags')]
            self._grad_step = -1
            returned = binding.byte_lm_session_step(self._native_session, addresses, parameters, native_shape)
        else:
            addresses = [addr_ro(tokens, name='ids'), addr_ro(flags, name='flags'),
                         addr(out_loss, name='out_loss')]
            returned = binding.byte_lm_session_eval(self._native_session, addresses, parameters, native_shape)
        # The whole native call, itemized by the `step.bind_*`, `step.*`,
        # `block.*`, `attn.*` and `bwd.*` lines the binding printed.
        _tick(ton, clock, 'envelope.native_call')
        if _is_bool(returned) or not isinstance(returned, int) or returned != expected:
            raise RuntimeError('Byte-LM returned an invalid completed-step counter')
        if not all_finite(out_loss):
            raise RuntimeError('Byte-LM returned a nonfinite/unwritten loss')
        _mode()
        if not train:
            return float(out_loss[0])
        new_flags = _array(out_flags, (shape.n_tensors,), 'flags', '<i4')
        if new_flags.min() < 0 or new_flags.max() > 1:
            raise RuntimeError('Byte-LM returned non-binary momentum flags')
        _tick(ton, clock, 'step.py_flags', new_flags.nbytes)
        result = dict(loss=float(out_loss[0]), step=expected, completed_steps=expected,
                      next_batch_index=expected)
        if self._step_result == 'lean':
            result['flags'] = new_flags.copy()
        else:
            # 'full': the lean step plus the gradient export, before commit,
            # so a refused export rolls the step back like any other failure.
            result.update(self._export_gradients_impl(expected, True))
            _tick(ton, clock, 'step.py_export_gradients', 2 * shape.n_total * 4)
        self._state['flags'] = new_flags
        self._state['completed_steps'] = expected
        self._state['next_batch_index'] = expected
        self._grad_step = expected
        return result

    def train_step(self, ids):
        """One mean-CE/AdamW update. step_result='full': every configured
        pre-update gradient is returned; 'lean' (resident only): the loss,
        the step and the flags, with the gradient left on the device for
        `export_gradients()`."""
        with self._lock:
            return self._run(ids, True)

    def evaluate(self, ids):
        """Return the mean loss. The stateless path requires full
        training-state byte invariance per call; a resident session
        changes no state by construction (native gate G1)."""
        with self._lock:
            return self._run(ids, False)

    def logits(self, ids, *, lengths=None):
        """Float32 logits `[batch, length, vocab]` for int32 ids `[batch,
        length]` (DEVIATION 2658), for positions 0 to length - 1 of each row,
        prefilled from absolute position 0.

        `lengths` (2026-09-15) makes the batch RAGGED: `batch` integers in
        `[1, length]`, row `i` real at positions `[0, lengths[i])` and
        padding after, whatever int32 values the padding holds. Real
        positions' logits are byte for byte the row alone at its own length
        (causal attention, no arithmetic changes, `_ragged.py` says why) and
        padding positions' logits are exactly `+0.0`. `batch` must be in [1, 1024],
        `length` in [1, shape.length], `batch * length * vocab` at most
        268435456 and every id in [0, shape.vocab_size); anything else is
        refused with ValueError before any native call.

        The arithmetic is the IDENTICAL device forward the trainer's loss
        uses, on Metal, CUDA or HIP, and these are the numbers it computes
        before the loss. tools/byte_lm_gpu_logits_sweep.py compares them
        byte for byte with the CPU reference path,
        `LanguageModelInference(threaded=False)`. The native entries refuse
        a non-finite logit with an error, because NaN payloads are shaped by
        the vendor and cannot be part of an identity claim; the CPU class
        returns them.

        No state changes on either path. A resident trainer runs on its
        session (opening it first if needed, as `train_step` does) and the
        parameters, moments, flags and step counter do not move. A failed
        call has nothing to roll back; the last gradient is no longer
        exportable, as after any failure, and a session that no longer
        reports itself usable is lost. A stateless trainer validates its
        state as `evaluate` does, builds and destroys one device context,
        and requires the parameter and id bytes it handed over unchanged
        afterward."""
        if lengths is not None:
            raw, _ = as_i32_c(ids, ndim=2, name='ids')
            return _ragged.ragged_forward(self.logits, raw, None, lengths, '<i4',
                                          'SmallByteLanguageModelTrainer.logits')[0]
        with self._lock:
            shape = state_shape(self._state)
            tokens, batch, length = _gpu_logits_ids(ids, shape)
            if self._is_resident():
                return self._logits_resident(tokens, batch, length, shape)
            return self._logits_stateless(tokens, batch, length, shape)

    def next_bytes(self, ids, *, lengths=None):
        """The greedy next byte after each row of ids `[batch, length]`, ties
        to the lowest byte value, picked from `logits(ids)` by the helper
        `LanguageModelInference.next_bytes` uses (DEVIATION 2658). With
        `lengths` (a ragged batch, see `logits`) the byte after each row's
        LAST REAL position."""
        if lengths is not None:
            raw, _ = as_i32_c(ids, ndim=2, name='ids')
            logits, lens = _ragged.ragged_forward(self.logits, raw, None, lengths, '<i4',
                                                  'SmallByteLanguageModelTrainer.next_bytes')
            return _greedy_next_bytes(_ragged.last_real_rows(logits, lens,
                                                             'SmallByteLanguageModelTrainer.next_bytes'))
        with self._lock:
            if self._is_resident():
                # cpu3-seq: on a resident session the forward, the refusal
                # and the per-row argmax run on the device and only `batch`
                # int32 come back (the logits never cross the bus).
                shape = state_shape(self._state)
                tokens, batch, length = _gpu_logits_ids(ids, shape)
                return self._next_bytes_resident(tokens, batch, length, shape)
        return _greedy_next_bytes(self.logits(ids))

    def _next_bytes_resident(self, tokens, batch, length, shape):
        self._require_not_lost()
        binding = self._binding()
        missing = [name for name in _SESSION_ENTRIES + (_SESSION_NEXT_ENTRY,)  # glue: ten binding entry names
                   if not callable(getattr(binding, name, None))]
        if missing:
            raise ImportError('Byte-LM binding lacks resident GPU next bytes (%s); rebuild bindings/build_byte_lm.sh'
                              % ', '.join(missing))
        if not self._session_open:
            self._open_session(binding, shape)
        out = zeros((batch,), '<i4')
        try:
            written = getattr(binding, _SESSION_NEXT_ENTRY)(
                self._native_session, [addr_ro(tokens, name='ids'), addr(out, name='next_bytes')],
                [batch, length], list(shape.native_shape), self._state['completed_steps'])
            _require_written(written, batch)
        except BaseException:
            try:
                self._logits_failed()
            except Exception:
                pass  # Preserve the original failure.
            raise
        _mode()
        return out.tolist()

    def _logits_stateless(self, tokens, batch, length, shape):
        working = _validate_state(self._state)
        binding = self._binding()
        entry = getattr(binding, _LOGITS_ENTRY, None)
        if not callable(entry):
            raise ImportError('Byte-LM binding lacks GPU logits (%s); rebuild bindings/build_byte_lm.sh'
                              % _LOGITS_ENTRY)
        parameters = working['parameters']
        saved_parameters = parameters.tobytes()
        saved_tokens = tokens.tobytes()
        out = zeros((batch, length, shape.vocab_size), '<f4')
        written = entry([addr_ro(parameters, name='parameters'), addr_ro(tokens, name='ids'),
                         addr(out, name='logits')], [batch, length], list(shape.native_shape))
        _require_written(written, batch * length * shape.vocab_size)
        if saved_parameters != parameters.tobytes() or saved_tokens != tokens.tobytes():
            raise RuntimeError('Byte-LM native logits changed an input parameter/token buffer')
        _mode()
        return out

    def _logits_resident(self, tokens, batch, length, shape):
        self._require_not_lost()
        binding = self._binding()
        missing = [name for name in _SESSION_ENTRIES + (_SESSION_LOGITS_ENTRY,)  # glue: checks the session binding entries
                   if not callable(getattr(binding, name, None))]
        if missing:
            raise ImportError('Byte-LM binding lacks resident GPU logits (%s); rebuild bindings/build_byte_lm.sh'
                              % ', '.join(missing))
        if not self._session_open:
            # Admission: `_validate_state` once, one upload (3n floats).
            self._open_session(binding, shape)
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
        try:
            written = binding.byte_lm_session_logits(
                self._native_session, [addr_ro(tokens, name='ids'), addr(out, name='logits')],
                [batch, length], list(shape.native_shape), self._state['completed_steps'])
            if timing:
                print('timing python.logits_binding %.3f ms' % ((_time.perf_counter() - t1) * 1000.0), flush=True)
            _require_written(written, batch * length * shape.vocab_size)
        except BaseException:
            try:
                self._logits_failed()
            except Exception:
                pass  # Preserve the original failure.
            raise
        _mode()
        return out

    def _logits_failed(self):
        """The failure path of a resident `logits` call. Logits write no
        state, so nothing is rolled back (a rollback would restore the shadow
        of the last step, which is already committed). The gradient is no
        longer exportable, as after any failure, and a session that does not
        report itself usable at the committed step is lost."""
        if not self._session_open or self._lost_at is not None:
            return
        self._grad_step = -1
        try:
            info = self._session_binding.byte_lm_session_info(self._native_session)
            usable = bool(info[2]) and info[0] == self._state['completed_steps']
        except Exception:
            usable = False
        if not usable:
            self._mark_lost()

    def save_checkpoint(self, path):
        """`export_checkpoint(path)`."""
        return self.export_checkpoint(path)

    def export_checkpoint(self, path):
        """Atomically write at most 2 MiB of canonical no-pickle JSON/hex,
        from `export_state()`. Refused above the bound exactly as before;
        at larger shapes `export_state()` arrays are the checkpoint."""
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

    def export_checkpoint_binary(self, path):
        """Atomically write the complete state as a streamed BINARY archive
        (`_byte_lm_checkpoint`, schema `mojolearn.byte-lm-stream.v1`), from
        `export_state()`. Returns the file's SHA-256.

        THIS IS THE PATH AT SCALE, AND IT IS A SECOND FORMAT, NOT A WIDER
        FIRST ONE. `export_checkpoint` keeps its JSON/hex envelope, its
        bytes and its 2 MiB bound untouched; hex doubles the payload, so
        that bound stops at about 87,381 parameters and the
        162,147,840-parameter shape has always been refused by it. Here the
        four arrays are streamed as their little-endian `<f4`/`<i4` bytes
        after a bounded canonical JSON header, so the file is about 1.95 GB
        at that shape instead of impossible, and a round trip is bit-exact
        by construction: nothing passes through decimal text, so `-0.0`,
        NaN payloads and subnormals all survive.

        THE OPTIMIZER MOMENTS ARE IN THE FILE AND THAT IS THE POINT. A
        restore that dropped `m` and `v` would produce the same loss and
        the same gradient on its first step and a different update, so the
        divergence would surface a step later looking like
        nondeterminism.

        No device work beyond the one `export_state()` download a resident
        session already owes, and no arithmetic."""
        with self._lock:
            payload = self.export_state()
        return _byte_lm_checkpoint.save(path, payload)

    @classmethod
    def from_checkpoint_binary(cls, path, *, resident=None):
        """Restore a `mojolearn.byte-lm-stream.v1` archive written by
        `export_checkpoint_binary`, never the JSON/hex envelope and never
        `training/checkpoint.mojo`'s unreachable native binary v1.

        The archive is verified whole -- magic, header digest, every array
        digest and the exact file size -- and every array length is taken
        from the admitted registry rather than from the file, before this
        allocates anything. The restored state then passes the SAME
        `_validate_state` that admits `load_state_dict`, so a file cannot
        enter through a weaker door than an in-memory dict.

        Provenance is not established here: this says the bytes are the
        bytes that were written, not who wrote them or on what device."""
        state = _byte_lm_checkpoint.load(path)
        shape = state_shape(state)
        cfg = state['config']
        result = cls(state['parameters'], data_schedule=state['data_schedule'], lr=cfg['lr'],
                     betas=(cfg['beta1'], cfg['beta2']), eps=cfg['eps'], weight_decay=cfg['weight_decay'],
                     shape=shape, resident=resident)
        return result.load_state_dict(state)

    @classmethod
    def from_checkpoint(cls, path, *, resident=None):
        """Restore this explicit JSON checkpoint schema, never native binary v1."""
        with Path(path).open('rb') as stream:
            encoded = stream.read(_CHECKPOINT_LIMIT + 1)
        return cls.from_checkpoint_bytes(encoded, resident=resident)

    @classmethod
    def from_checkpoint_bytes(cls, encoded, *, resident=None):
        """Restore one bounded immutable capture without opening any path.

        Only exact ``bytes`` is accepted: callers must first capture mutable
        buffers themselves, then hash and supply that same immutable object.
        This loader does not establish file provenance or vendor identity.
        It shares every schema, integrity, tensor, optimizer and cursor check
        with from_checkpoint; neither API launches native model operations.
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
    if hashlib.sha256(_canonical(payload)).hexdigest() != envelope['payload_sha256']:
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
