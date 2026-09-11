# SPDX-License-Identifier: Apache-2.0
"""Host plumbing for the trainer's GPU logits (DEVIATION 2658).

The fake binding of test_byte_lm_surface stands in for `_mojolearn_byte_lm`,
so these tests prove wiring, refusals and state invariance, never
arithmetic. tools/byte_lm_gpu_logits_sweep.py compares the device logits
with the CPU reference path byte for byte.
"""
import struct

import numpy as np
import pytest

from mojolearn import ByteLanguageModelConfig as Shape, SmallByteLanguageModelTrainer as Trainer
from mojolearn import _byte_lm_host as host_mod
from mojolearn import _byte_lm_impl as impl
from mojolearn._buffer import frombytes
from mojolearn.tests.test_byte_lm_surface import buffer, host, initial, ids as step_ids, state_digest

SCHEDULE = {'dataset': 'logits-test'}


def tokens(batch, length):
    return (np.arange(batch * length, dtype=np.int32) % 256).reshape(batch, length)


def paint(address, batch, length, vocab):
    """Sentinel logits, not a model. -1 everywhere except a tie at bytes 3
    and 7 of each row's last position, so a greedy pick that goes low
    reads 3."""
    out = buffer(address, batch * length * vocab)
    out[:] = -1
    for b in range(batch):
        last = (b * length + length - 1) * vocab
        out[last + 3] = out[last + 7] = 2.0
    return batch * length * vocab


def stateless_entry(fake):
    """Install `byte_lm_logits` on the fake and return its call log."""
    calls = []
    fake.logits_extra = 0
    fake.logits_mutate_parameters = False

    def logits(addresses, dims, native):
        cfg = Shape(*native)
        batch, length = dims
        assert all(type(value) is int for value in dims)
        calls.append(dict(addresses=len(addresses), dims=list(dims), shape=list(native),
                          parameters=buffer(addresses[0], cfg.n_total).copy(),
                          ids=buffer(addresses[1], batch * length, True).copy()))
        if fake.logits_mutate_parameters:
            buffer(addresses[0], cfg.n_total)[0] = 77
        return paint(addresses[2], batch, length, cfg.vocab_size) + fake.logits_extra

    fake.byte_lm_logits = logits
    return calls


def resident_entry(fake):
    """`enable_sessions()` plus `byte_lm_session_logits`, which reads the
    session and writes nothing to its fake device state. `logits_fail`
    None succeeds, 'usable' raises and keeps the session usable, 'lost'
    raises and marks it unusable as the native `_mark_if_lost` would."""
    created, closed, _ = fake.enable_sessions()
    calls = []
    fake.logits_extra = 0
    fake.logits_fail = None

    def logits(session, addresses, dims, native, completed):
        assert session.open and session.usable and session not in closed
        cfg = Shape(*native)
        batch, length = dims
        assert all(type(value) is int for value in dims)
        assert type(completed) is int
        calls.append(dict(session=session, addresses=len(addresses), dims=list(dims),
                          shape=list(native), completed=completed,
                          ids=buffer(addresses[0], batch * length, True).copy()))
        if fake.logits_fail == 'lost':
            session.usable = False
        if fake.logits_fail is not None:
            raise RuntimeError('injected byte-LM logits failure')
        return paint(addresses[1], batch, length, cfg.vocab_size) + fake.logits_extra

    fake.byte_lm_session_logits = logits
    return created, closed, calls


def test_stateless_logits_reach_byte_lm_logits_with_dims_and_three_addresses(host):
    calls = stateless_entry(host)
    model = Trainer(initial(), data_schedule=SCHEDULE)
    x = tokens(2, 5)
    out = model.logits(x)
    assert tuple(out.shape) == (2, 5, 256)
    assert [(c['addresses'], c['dims'], c['shape']) for c in calls] == [(3, [2, 5], list(Shape().native_shape))]
    np.testing.assert_array_equal(calls[0]['ids'], x.ravel())
    np.testing.assert_array_equal(calls[0]['parameters'], initial())
    view = np.asarray(out)
    assert view.dtype == np.float32
    assert view[1, 4, 3] == view[1, 4, 7] == 2.0 and view[0, 0, 0] == -1
    assert host.calls == []


def test_stateless_logits_change_no_state_and_do_not_advance_the_step(host):
    stateless_entry(host)
    model = Trainer(initial(), data_schedule=SCHEDULE)
    model.train_step(step_ids())
    before = state_digest(model)
    x = tokens(3, 32)
    x_bytes = x.tobytes()
    model.logits(x)
    model.next_bytes(x)
    assert state_digest(model) == before and model.step_ == 1
    np.testing.assert_array_equal(model.parameters_, np.ones(34944, np.float32))
    assert x.tobytes() == x_bytes
    host.logits_mutate_parameters = True
    with pytest.raises(RuntimeError, match='changed an input'):
        model.logits(x)
    assert state_digest(model) == before and model.step_ == 1


def test_resident_logits_open_the_session_then_reach_byte_lm_session_logits(host):
    created, closed, calls = resident_entry(host)
    model = Trainer(initial(), data_schedule=SCHEDULE, resident=True)
    assert created == []
    x = tokens(2, 7)
    out = model.logits(x)
    assert tuple(out.shape) == (2, 7, 256)
    assert len(created) == 1 and created[0].open and closed == []
    assert [(c['session'], c['addresses'], c['dims'], c['shape']) for c in calls] == [
        (created[0], 2, [2, 7], list(Shape().native_shape))]
    np.testing.assert_array_equal(calls[0]['ids'], x.ravel())
    assert model.step_ == 0 and host.calls == []
    model.train_step(step_ids())
    model.logits(x)
    assert len(created) == 1 and calls[-1]['session'] is created[0]
    assert model.step_ == 1
    # The committed step crosses with the call, so a resident logits call
    # cannot silently read a session at another step (the scalar half of
    # byte_lm_session_eval's admission).
    assert [c['completed'] for c in calls] == [0, 1]
    state = model.export_state()
    np.testing.assert_array_equal(state['parameters'], np.ones(34944, np.float32))
    assert state['completed_steps'] == 1


def test_missing_native_entries_raise_import_error_by_name(host):
    stateless = Trainer(initial(), data_schedule=SCHEDULE)
    with pytest.raises(ImportError, match='byte_lm_logits') as info:
        stateless.logits(tokens(1, 4))
    assert 'bindings/build_byte_lm.sh' in str(info.value)
    created, closed, _ = host.enable_sessions()
    resident = Trainer(initial(), data_schedule=SCHEDULE, resident=True)
    with pytest.raises(ImportError, match='byte_lm_session_logits') as info:
        resident.logits(tokens(1, 4))
    assert 'bindings/build_byte_lm.sh' in str(info.value)
    assert created == [] and host.calls == []


@pytest.mark.parametrize('resident', [False, True])
def test_bad_ids_and_shapes_raise_value_error_before_any_native_call(host, resident):
    stateless_calls = stateless_entry(host)
    created, closed, resident_calls = resident_entry(host)
    model = Trainer(initial(), data_schedule=SCHEDULE, resident=resident)
    for bad in (-1, 256, 2 ** 30):
        x = tokens(2, 4)
        x[1, 2] = bad
        with pytest.raises(ValueError, match='byte values'):
            model.logits(x)
        with pytest.raises(ValueError, match='byte values'):
            model.next_bytes(x)
    for x in (np.zeros((1, 0), np.int32), np.zeros((1, 33), np.int32), np.zeros((0, 4), np.int32),
              np.zeros(4, np.int32), np.zeros((1025, 1), np.int32)):
        with pytest.raises(ValueError):
            model.logits(x)
    assert stateless_calls == [] and resident_calls == [] and created == []
    assert model.step_ == 0


@pytest.mark.parametrize('resident', [False, True])
def test_the_native_cell_limit_is_refused_first(host, monkeypatch, resident):
    stateless_calls = stateless_entry(host)
    created, closed, resident_calls = resident_entry(host)
    model = Trainer(initial(), data_schedule=SCHEDULE, resident=resident)
    monkeypatch.setattr(impl, '_LOGITS_MAX_CELLS', 2 * 4 * 256 - 1)
    with pytest.raises(ValueError, match='batch \\* length \\* vocab'):
        model.logits(tokens(2, 4))
    assert stateless_calls == [] and resident_calls == [] and created == []
    assert tuple(model.logits(tokens(1, 4)).shape) == (1, 4, 256)


@pytest.mark.parametrize('resident', [False, True])
def test_a_wrong_returned_count_raises_runtime_error(host, resident):
    stateless_entry(host)
    created, closed, _ = resident_entry(host)
    model = Trainer(initial(), data_schedule=SCHEDULE, resident=resident)
    before = state_digest(model)
    for extra in (1, -1):
        host.logits_extra = extra
        with pytest.raises(RuntimeError, match='number of logits'):
            model.logits(tokens(2, 3))
    host.logits_extra = 0
    assert tuple(model.logits(tokens(2, 3)).shape) == (2, 3, 256)
    assert closed == [] and model.step_ == 0
    assert state_digest(model) == before


def test_a_resident_native_failure_keeps_a_usable_session_and_loses_an_unusable_one(host):
    created, closed, calls = resident_entry(host)
    model = Trainer(initial(), data_schedule=SCHEDULE, resident=True)
    model.train_step(step_ids())
    host.logits_fail = 'usable'
    with pytest.raises(RuntimeError, match='injected'):
        model.logits(tokens(1, 4))
    # As after any failure, the last gradient is no longer trusted.
    with pytest.raises(RuntimeError, match='no gradient'):
        model.export_gradients()
    host.logits_fail = None
    assert tuple(model.logits(tokens(1, 4)).shape) == (1, 4, 256)
    assert model.step_ == 1 and len(created) == 1 and closed == []
    host.logits_fail = 'lost'
    with pytest.raises(RuntimeError, match='injected'):
        model.logits(tokens(1, 4))
    host.logits_fail = None
    for call in (lambda: model.logits(tokens(1, 4)), lambda: model.next_bytes(tokens(1, 4)),
                 lambda: model.evaluate(step_ids())):
        with pytest.raises(RuntimeError, match='session lost at step 1; last exported state is step 0'):
            call()
    assert closed == [] and len(created) == 1


@pytest.mark.parametrize('resident', [False, True])
def test_next_bytes_ties_go_to_the_lowest_byte(host, resident):
    stateless_entry(host)
    resident_entry(host)
    model = Trainer(initial(), data_schedule=SCHEDULE, resident=resident)
    assert model.next_bytes(tokens(3, 5)) == [3, 3, 3]
    assert model.next_bytes(tokens(1, 1)) == [3]
    assert model.step_ == 0


def test_the_greedy_helper_reads_the_last_position_and_ties_go_low():
    values = [0.0] * (2 * 3 * 5)
    values[0 * 15 + 0 * 5 + 2] = 9.0   # row 0, first position, ignored
    values[0 * 15 + 2 * 5 + 1] = 4.0   # row 0, last position, tie at 1 and 4
    values[0 * 15 + 2 * 5 + 4] = 4.0
    values[1 * 15 + 2 * 5 + 0] = -0.0  # row 1, last position, all equal
    logits = frombytes(struct.pack('<30f', *values), '<f4', (2, 3, 5))
    assert host_mod._greedy_next_bytes(logits) == [1, 0]


def test_a_runtime_shape_reaches_the_native_entry_and_bounds_the_length(host):
    cfg = Shape(batch=1, length=17, d_model=16, n_heads=2, n_kv=1, head_dim=8, intermediate=24)
    host.byte_lm_config_profile = lambda dimensions: Shape(*dimensions).profile
    host.byte_lm_run_configured = lambda *args: pytest.fail('logits reached the training entry')
    calls = stateless_entry(host)
    model = Trainer(np.zeros(cfg.n_total, np.float32), shape=cfg, data_schedule=SCHEDULE)
    assert tuple(model.logits(tokens(4, 17)).shape) == (4, 17, 256)
    assert calls[-1]['shape'] == list(cfg.native_shape) and calls[-1]['dims'] == [4, 17]
    with pytest.raises(ValueError):
        model.logits(tokens(1, 18))
    assert len(calls) == 1
