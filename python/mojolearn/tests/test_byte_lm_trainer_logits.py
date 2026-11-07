# SPDX-License-Identifier: Apache-2.0
"""The trainer's logits and next_bytes against the installed compiled binding.

The native trainer owner admits only the exact compiled module, so these use
the real binding: they assert shapes, refusals and state invariance, not
model quality. tools/byte_lm_gpu_logits_sweep.py compares the device logits
with the CPU reference path byte for byte.

DELETED (2026-11-07), because each needed a Python stand-in binding that the
native owner now refuses before any entry is read:
- test_missing_native_entries_raise_import_error_by_name: an entry missing
  from the compiled module is refused by name at connection
  (test_byte_lm_host_trainer::test_a_binding_without_the_training_entry_is_refused_by_name).
- test_a_wrong_returned_count_raises_runtime_error: the count check is native
  (bindings/byte_lm_trainer_native.mojo "unexpected number of logits"); the
  compiled entry cannot be made to miscount.
- test_a_resident_native_failure_keeps_a_usable_session_and_loses_an_unusable_one:
  injected device failures; the compiled binding has no fault hook, so the
  lost-session refusal has no real-binding test (reported as a coverage gap).
- test_the_native_cell_limit_is_refused_first: it lowered the Python
  `_byte_lm_impl._LOGITS_MAX_CELLS`, which the native logits path no longer
  reads; the native bound (batch * length * vocab <= 2**28,
  training/byte_lm_host_trainer_dispatch.mojo) is above any default-shape
  request, and reaching it needs a multi-GB request.
- test_a_runtime_shape_reaches_the_native_entry_and_bounds_the_length is
  rewritten below without the call log.
"""
import hashlib
import json
import math
import struct

import numpy as np
import pytest

from mojolearn import ByteLanguageModelConfig as Shape, SmallByteLanguageModelTrainer as Trainer
from mojolearn import _byte_lm_host as host_mod
from mojolearn import _byte_lm_impl as impl
from mojolearn._buffer import frombytes

SCHEDULE = {'dataset': 'logits-test'}


def initial(n=34944):
    return (np.random.default_rng(17).standard_normal(n) * .02).astype(np.float32)


def step_ids():
    return (np.arange(66, dtype=np.int32) % 256).reshape(2, 33)


def tokens(batch, length):
    return (np.arange(batch * length, dtype=np.int32) % 256).reshape(batch, length)


def state_digest(model):
    state = model.state_dict()
    for key in ('parameters', 'm', 'v', 'flags'):
        value = np.asarray(state[key])
        state[key] = value.tobytes().hex()
    return hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()


@pytest.mark.parametrize('resident', [False, True])
def test_logits_shape_dtype_and_finiteness(resident):
    model = Trainer(initial(), data_schedule=SCHEDULE, resident=resident)
    x = tokens(2, 5)
    out = np.asarray(model.logits(x))
    assert out.shape == (2, 5, 256) and out.dtype == np.float32
    assert np.isfinite(out).all()
    model.close()


@pytest.mark.parametrize('resident', [False, True])
def test_logits_change_no_state_and_do_not_advance_the_step(resident):
    model = Trainer(initial(), data_schedule=SCHEDULE, resident=resident)
    model.train_step(step_ids())
    before = state_digest(model)
    x = tokens(3, 32)
    x_bytes = x.tobytes()
    model.logits(x)
    model.next_bytes(x)
    assert state_digest(model) == before and model.step_ == 1
    assert x.tobytes() == x_bytes
    assert model.export_state()['completed_steps'] == 1
    model.close()


@pytest.mark.parametrize('resident', [False, True])
def test_bad_ids_and_shapes_raise_value_error(resident):
    model = Trainer(initial(), data_schedule=SCHEDULE, resident=resident)
    before = state_digest(model)
    for bad in (-1, 256, 2 ** 30):
        x = tokens(2, 4)
        x[1, 2] = bad
        with pytest.raises(ValueError, match=r'token id outside \[0, vocab\)'):
            model.logits(x)
        with pytest.raises(ValueError, match=r'token id outside \[0, vocab\)'):
            model.next_bytes(x)
    for x in (np.zeros((1, 0), np.int32), np.zeros((1, 33), np.int32), np.zeros((0, 4), np.int32),
              np.zeros(4, np.int32), np.zeros((1025, 1), np.int32)):
        with pytest.raises(ValueError):
            model.logits(x)
    assert model.step_ == 0 and state_digest(model) == before
    model.close()


@pytest.mark.parametrize('resident', [False, True])
def test_next_bytes_is_the_greedy_pick_of_the_last_position(resident):
    model = Trainer(initial(), data_schedule=SCHEDULE, resident=resident)
    x = tokens(3, 5)
    logits = np.asarray(model.logits(x))
    assert model.next_bytes(x) == host_mod._greedy_next_bytes(logits)
    assert len(model.next_bytes(tokens(1, 1))) == 1
    assert model.step_ == 0
    model.close()


def test_the_greedy_helper_reads_the_last_position_and_ties_go_low():
    values = [0.0] * (2 * 3 * 5)
    values[0 * 15 + 0 * 5 + 2] = 9.0   # row 0, first position, ignored
    values[0 * 15 + 2 * 5 + 1] = 4.0   # row 0, last position, tie at 1 and 4
    values[0 * 15 + 2 * 5 + 4] = 4.0
    values[1 * 15 + 2 * 5 + 0] = -0.0  # row 1, last position, all equal
    logits = frombytes(struct.pack('<30f', *values), '<f4', (2, 3, 5))
    assert host_mod._greedy_next_bytes(logits) == [1, 0]


def test_a_runtime_shape_bounds_the_length():
    cfg = Shape(batch=1, length=17, d_model=16, n_heads=2, n_kv=1, head_dim=8, intermediate=24)
    model = Trainer(initial(cfg.n_total), shape=cfg, data_schedule=SCHEDULE)
    out = np.asarray(model.logits(tokens(4, 17)))
    assert out.shape == (4, 17, 256) and np.isfinite(out).all()
    with pytest.raises(ValueError):
        model.logits(tokens(1, 18))
    assert model.step_ == 0
    model.close()
