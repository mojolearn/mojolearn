# SPDX-License-Identifier: Apache-2.0
"""Host plumbing for CPU training (DEVIATION 2680).

The same fake binding `test_byte_lm_host.py` uses stands in for
`_mojolearn_byte_lm_host`, so these tests prove WIRING, REFUSALS and state
invariance, never arithmetic. The arithmetic is qualified by
tools/byte_lm_cpu_train_gate.py, which replays steps of the retained capture
and requires the gradient, the loss bits and the post-step parameters and both
Adam moments to equal the recorded GPU bytes, with a wrong-gradient build
required to fail it.

The division matters. A fake binding cannot tell you a gradient is right; it
can tell you that a refused input never reached a kernel, and that a failed
step left nothing half-applied. Those are the two things a gate against
recorded bytes cannot see.
"""
import struct

import numpy as np
import pytest

from mojolearn import ByteLanguageModelConfig as Shape
from mojolearn import _byte_lm_host as host_mod
from mojolearn.tests.test_byte_lm_host import fake_host  # noqa: F401  (fixture)
from mojolearn.tests.test_byte_lm_surface import buffer

SHAPE = Shape()
N = SHAPE.n_total


def ids(batch=None, width=None):
    """A legal training batch, `[batch, length + 1]` of byte values."""
    batch = SHAPE.batch if batch is None else batch
    width = SHAPE.length + 1 if width is None else width
    return (np.arange(batch * width, dtype=np.int32) % 256).reshape(batch, width)


def install(fake, *, fail=False, extra=0.0):
    """Put `byte_lm_host_train_step` on the fake and return its call log.

    The fake writes sentinels rather than arithmetic: grad is filled with 1.0,
    post_p with 2.0, post_m with 3.0 and post_v with 4.0, so a test can tell
    which output buffer the surface read back into which attribute."""
    calls = []

    def train_step(addresses, native, scalars, completed):
        assert len(addresses) == 8
        assert all(type(a) is int for a in addresses)
        assert list(native) == host_mod._native_shape(SHAPE)
        assert len(scalars) == 5 and all(type(s) is float for s in scalars)
        assert type(completed) is int
        calls.append(dict(scalars=list(scalars), completed=completed,
                          params=buffer(addresses[0], N).copy(),
                          m=buffer(addresses[1], N).copy(),
                          v=buffer(addresses[2], N).copy(),
                          ids=buffer(addresses[3], SHAPE.batch * (SHAPE.length + 1), True).copy()))
        if fail:
            raise RuntimeError('injected native training failure')
        for slot, value in ((4, 1.0 + extra), (5, 2.0), (6, 3.0), (7, 4.0)):
            buffer(addresses[slot], N)[:] = value
        return 0x3F800000  # 1.0

    fake.byte_lm_host_train_step = train_step
    return calls


def trainer(**kwargs):
    return host_mod.LanguageModelHostTrainer(np.ones(N, np.float32), **kwargs)


def test_a_binding_without_the_training_entry_is_refused_by_name(fake_host):
    """An older host binding runs inference fine and has no training entry.
    The refusal names the entry and the build script, at construction, rather
    than failing somewhere inside the first step."""
    assert not hasattr(fake_host, 'byte_lm_host_train_step')
    with pytest.raises(ImportError, match='byte_lm_host_train_step') as info:
        trainer()
    assert 'bindings/build_byte_lm_host.sh' in str(info.value)


def test_a_step_reaches_the_binding_and_advances_the_state(fake_host):
    calls = install(fake_host)
    model = trainer()
    assert model.completed_steps == 0 and model.gradient_ is None
    x = ids()
    bits = model.train_step(x)
    assert bits == 0x3F800000
    assert len(calls) == 1 and calls[0]['completed'] == 0
    np.testing.assert_array_equal(calls[0]['ids'], x.ravel())
    np.testing.assert_array_equal(calls[0]['params'], np.ones(N, np.float32))
    # The moments default to zero and are handed to the binding as read-only.
    np.testing.assert_array_equal(calls[0]['m'], np.zeros(N, np.float32))
    # Each output buffer lands on its own attribute.
    np.testing.assert_array_equal(np.asarray(model.gradient_), np.full(N, 1.0, np.float32))
    np.testing.assert_array_equal(np.asarray(model.parameters_), np.full(N, 2.0, np.float32))
    np.testing.assert_array_equal(np.asarray(model.m_), np.full(N, 3.0, np.float32))
    np.testing.assert_array_equal(np.asarray(model.v_), np.full(N, 4.0, np.float32))
    assert model.completed_steps == 1
    # The next step carries the advanced count, which is what the optimizer's
    # bias correction reads, and the previous step's outputs as its inputs.
    model.train_step(x)
    assert calls[1]['completed'] == 1
    np.testing.assert_array_equal(calls[1]['params'], np.full(N, 2.0, np.float32))
    np.testing.assert_array_equal(calls[1]['m'], np.full(N, 3.0, np.float32))
    np.testing.assert_array_equal(calls[1]['v'], np.full(N, 4.0, np.float32))


def test_a_failed_step_leaves_nothing_half_applied(fake_host):
    install(fake_host, fail=True)
    model = trainer()
    before = (np.asarray(model.parameters_).copy(), np.asarray(model.m_).copy(),
              np.asarray(model.v_).copy())
    with pytest.raises(RuntimeError, match='injected'):
        model.train_step(ids())
    assert model.completed_steps == 0
    assert model.gradient_ is None
    for got, want in zip((model.parameters_, model.m_, model.v_), before):
        np.testing.assert_array_equal(np.asarray(got), want)


def test_the_gradient_is_cleared_before_the_call_not_left_stale(fake_host):
    calls = install(fake_host)
    model = trainer()
    model.train_step(ids())
    assert model.gradient_ is not None
    install(fake_host, fail=True)
    with pytest.raises(RuntimeError):
        model.train_step(ids())
    # A stale gradient from the previous step would read as this step's.
    assert model.gradient_ is None
    assert len(calls) == 1


def test_ids_must_be_the_training_batch_layout(fake_host):
    calls = install(fake_host)
    model = trainer()
    for bad in (ids(batch=1), ids(width=SHAPE.length), ids(width=SHAPE.length + 2),
                np.zeros(SHAPE.length + 1, np.int32)):
        with pytest.raises(ValueError):
            model.train_step(bad)
    assert calls == [] and model.completed_steps == 0


def test_byte_range_is_refused_and_the_ignore_index_only_in_the_last_column(fake_host):
    calls = install(fake_host)
    model = trainer()
    for bad in (-1, 256, 2 ** 20):
        x = ids()
        x[0, 3] = bad
        with pytest.raises(ValueError, match='byte values'):
            model.train_step(x)
    # -100 is a target-only sentinel: legal in the last column, refused before.
    x = ids()
    x[0, SHAPE.length] = host_mod._IGNORE_INDEX
    model.train_step(x)
    assert len(calls) == 1
    x = ids()
    x[0, SHAPE.length - 1] = host_mod._IGNORE_INDEX
    with pytest.raises(ValueError, match='byte values'):
        model.train_step(x)
    assert len(calls) == 1


@pytest.mark.parametrize('kwargs', [
    dict(lr=0.0), dict(lr=-1e-3), dict(eps=0.0), dict(eps=-1e-8),
    dict(betas=(1.0, 0.999)), dict(betas=(0.9, 1.0)), dict(betas=(-0.1, 0.999)),
    dict(weight_decay=-1e-4), dict(lr=True), dict(eps='small'),
])
def test_illegal_optimizer_configurations_are_refused_at_construction(fake_host, kwargs):
    install(fake_host)
    with pytest.raises(ValueError):
        trainer(**kwargs)


def test_the_admitted_optimizer_reaches_the_binding_as_five_floats(fake_host):
    calls = install(fake_host)
    model = trainer(lr=3e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    model.train_step(ids())
    assert calls[0]['scalars'] == pytest.approx([3e-3, 0.9, 0.999, 1e-8, 1e-2])


@pytest.mark.parametrize('bad', [np.ones(N - 1, np.float32), np.ones(N + 1, np.float32)])
def test_parameters_and_moments_must_be_the_registry_size(fake_host, bad):
    install(fake_host)
    with pytest.raises(ValueError):
        host_mod.LanguageModelHostTrainer(bad)
    with pytest.raises(ValueError):
        host_mod.LanguageModelHostTrainer(np.ones(N, np.float32), m=bad)
    with pytest.raises(ValueError):
        host_mod.LanguageModelHostTrainer(np.ones(N, np.float32), v=bad)


def test_nonfinite_parameters_and_moments_are_refused(fake_host):
    install(fake_host)
    for spoil in ('parameters', 'm', 'v'):
        values = np.ones(N, np.float32)
        values[17] = np.nan
        kwargs = {} if spoil == 'parameters' else {spoil: values}
        first = values if spoil == 'parameters' else np.ones(N, np.float32)
        with pytest.raises(ValueError):
            host_mod.LanguageModelHostTrainer(first, **kwargs)


@pytest.mark.parametrize('bad', [-1, True, 1.5, '3'])
def test_completed_steps_must_be_a_nonnegative_int(fake_host, bad):
    install(fake_host)
    with pytest.raises(ValueError):
        trainer(completed_steps=bad)


def test_from_state_carries_the_moments_and_the_step_count(fake_host):
    calls = install(fake_host)
    m = np.full(N, 0.25, np.float32)
    v = np.full(N, 0.75, np.float32)
    model = host_mod.LanguageModelHostTrainer.from_state(
        np.ones(N, np.float32), m, v, completed_steps=63, lr=3e-3)
    assert model.completed_steps == 63
    model.train_step(ids())
    assert calls[0]['completed'] == 63
    np.testing.assert_array_equal(calls[0]['m'], m)
    np.testing.assert_array_equal(calls[0]['v'], v)
    assert model.completed_steps == 64


def test_loss_returns_the_float_of_the_bits(fake_host):
    install(fake_host)
    model = trainer()
    assert model.loss(ids()) == struct.unpack('<f', struct.pack('<I', 0x3F800000))[0]
    assert model.completed_steps == 1


def test_the_surface_does_not_mutate_the_caller_s_ids(fake_host):
    install(fake_host)
    model = trainer()
    x = ids()
    raw = x.tobytes()
    model.train_step(x)
    assert x.tobytes() == raw


def test_profile_and_sha256_are_readable_without_a_step(fake_host):
    install(fake_host)
    model = trainer()
    assert model.profile == SHAPE.profile
    assert model.shape is SHAPE or model.shape.profile == SHAPE.profile
    assert len(model.parameters_sha256()) == 64
