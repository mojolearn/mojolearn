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
        if (fail() if callable(fail) else fail):
            raise RuntimeError('injected native training failure')
        for slot, value in ((4, 1.0 + extra), (5, 2.0), (6, 3.0), (7, 4.0)):
            buffer(addresses[slot], N)[:] = value
        return 0x3F800000  # 1.0

    fake.byte_lm_host_train_step = train_step
    return calls


def trainer(**kwargs):
    return host_mod.LanguageModelHostTrainer(np.ones(N, np.float32), **kwargs)


# The trainer's native owner (bindings/byte_lm_host_trainer_native.mojo) reads
# the host binding connected at import into `_HOST_TRAIN_CONTEXT` and admits
# only compiled entries owned by that module
# (providers/python/neural_descriptor_pyapi.mojo, `neural_descriptor_entry`).
# A Python stand-in planted in `sys.modules` is therefore never called, so the
# wiring tests below run one real step of the shipped host binding and read
# its public state; the arithmetic itself is still the gate's business.
def test_a_binding_without_the_training_entry_is_refused_by_name(monkeypatch):
    """An older host binding runs inference fine and has no training entry.
    The refusal names the entry, at construction, rather than failing
    somewhere inside the first step; a binding that failed to import is
    refused with its import error."""
    import types
    older = types.ModuleType(host_mod._MODULE_NAME)
    monkeypatch.setitem(host_mod._HOST_TRAIN_CONTEXT, 'byte_lm_host_binding', older)
    with pytest.raises(TypeError, match='compiled entry byte_lm_host_train_step'):
        trainer()
    reason = ('ImportError: mojolearn: no host binding with byte_lm_host_train_step; '
              'build it with bindings/build_byte_lm_host.sh')
    monkeypatch.setitem(host_mod._HOST_TRAIN_CONTEXT, 'byte_lm_host_binding', reason)
    with pytest.raises(ImportError, match='byte_lm_host_train_step') as info:
        trainer()
    assert 'bindings/build_byte_lm_host.sh' in str(info.value)


def test_a_step_reaches_the_binding_and_advances_the_state():
    model = trainer()
    assert model.completed_steps == 0 and model.gradient_ is None
    np.testing.assert_array_equal(np.asarray(model.m_), np.zeros(N, np.float32))
    np.testing.assert_array_equal(np.asarray(model.v_), np.zeros(N, np.float32))
    x = ids()
    bits = model.train_step(x)
    assert type(bits) is int and 0 <= bits < 1 << 32
    assert np.isfinite(struct.unpack('<f', struct.pack('<I', bits))[0])
    assert model.completed_steps == 1
    gradient = np.asarray(model.gradient_)
    assert gradient.shape == (N,) and gradient.dtype == np.float32 and gradient.any()
    # The update replaced the parameters and both moments.
    first = [np.asarray(a).copy() for a in (model.parameters_, model.m_, model.v_)]
    assert not np.array_equal(first[0], np.ones(N, np.float32))
    assert first[1].any() and first[2].any()
    # The next step carries the advanced count and starts from the previous
    # step's outputs.
    model.train_step(x)
    assert model.completed_steps == 2
    for got, before in zip((model.parameters_, model.m_, model.v_), first):
        assert not np.array_equal(np.asarray(got), before)


# DELETED 2026-11-07 (lane ne1-k-fail-a): test_a_failed_step_leaves_nothing_half_applied
# and test_the_gradient_is_cleared_before_the_call_not_left_stale injected the
# failure through a Python stand-in entry, which the native owner no longer
# admits. The current contract (a refused step leaves parameters, moments and
# the step count unchanged and the gradient None) is tested by
# test_byte_lm_host_native_provider.py::test_host_native_provider_reuse_and_transactional_outputs.


def test_ids_must_be_the_training_batch_layout(fake_host):
    calls = install(fake_host)
    model = trainer()
    for bad in (ids(batch=1), ids(width=SHAPE.length), ids(width=SHAPE.length + 2),
                np.zeros(SHAPE.length + 1, np.int32)):
        with pytest.raises(ValueError):
            model.train_step(bad)
    assert calls == [] and model.completed_steps == 0


def test_byte_range_is_refused_and_the_ignore_index_only_in_the_last_column():
    model = trainer()
    for bad in (-1, 256, 2 ** 20):
        x = ids()
        x[0, 3] = bad
        with pytest.raises(ValueError, match='byte values'):
            model.train_step(x)
    assert model.completed_steps == 0
    # -100 is a target-only sentinel: legal in the last column, refused before.
    x = ids()
    x[0, SHAPE.length] = host_mod._IGNORE_INDEX
    model.train_step(x)
    assert model.completed_steps == 1
    x = ids()
    x[0, SHAPE.length - 1] = host_mod._IGNORE_INDEX
    with pytest.raises(ValueError, match='byte values'):
        model.train_step(x)
    assert model.completed_steps == 1


@pytest.mark.parametrize('kwargs', [
    dict(lr=0.0), dict(lr=-1e-3), dict(eps=0.0), dict(eps=-1e-8),
    dict(betas=(1.0, 0.999)), dict(betas=(0.9, 1.0)), dict(betas=(-0.1, 0.999)),
    dict(weight_decay=-1e-4), dict(lr=True), dict(eps='small'),
])
def test_illegal_optimizer_configurations_are_refused_at_construction(fake_host, kwargs):
    install(fake_host)
    with pytest.raises(ValueError):
        trainer(**kwargs)


def test_the_admitted_optimizer_reaches_the_binding_as_five_floats():
    # The owner keeps the five admitted floats it hands every step.
    model = trainer(lr=3e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    scalars = vars(model)['_scalars']
    assert all(type(v) is float for v in scalars)
    assert scalars == pytest.approx([3e-3, 0.9, 0.999, 1e-8, 1e-2])
    model.train_step(ids())
    assert model.completed_steps == 1


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


def test_from_state_carries_the_moments_and_the_step_count():
    m = np.full(N, 0.25, np.float32)
    v = np.full(N, 0.75, np.float32)
    model = host_mod.LanguageModelHostTrainer.from_state(
        np.ones(N, np.float32), m, v, completed_steps=63, lr=3e-3)
    assert model.completed_steps == 63
    np.testing.assert_array_equal(np.asarray(model.m_), m)
    np.testing.assert_array_equal(np.asarray(model.v_), v)
    model.train_step(ids())
    assert model.completed_steps == 64
    # A resumed step is not a fresh one: the same parameters from zero
    # moments at step 0 end elsewhere.
    fresh = trainer(lr=3e-3)
    fresh.train_step(ids())
    assert not np.array_equal(np.asarray(model.parameters_), np.asarray(fresh.parameters_))


def test_loss_returns_the_float_of_the_bits():
    bits = trainer().train_step(ids())
    model = trainer()
    assert model.loss(ids()) == struct.unpack('<f', struct.pack('<I', bits))[0]
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


#: DEVIATION 2682. The four-row shape, whose capture is a separate certificate
#: because nine weight gradients contract over the token count.
FOUR = Shape(batch=4)


def test_a_second_shape_reaches_the_binding_as_that_shape():
    """The surface has only ever been exercised at the certified shape. A
    capture at another shape is worth nothing if the replay silently runs the
    default, so this checks the four-row batch is the one the step takes."""
    model = host_mod.LanguageModelHostTrainer(np.ones(FOUR.n_total, np.float32), shape=FOUR)
    x = (np.arange(FOUR.batch * (FOUR.length + 1), dtype=np.int32) % 256
         ).reshape(FOUR.batch, FOUR.length + 1)
    assert x.size == FOUR.batch * (FOUR.length + 1) == 132
    model.train_step(x)
    # Same parameter count at both shapes, which is the point: the arrays are
    # the same size and the sums behind them are not.
    assert FOUR.n_total == SHAPE.n_total
    assert model.profile == FOUR.profile != SHAPE.profile
    assert model.completed_steps == 1
    # Four rows and two rows are different sums over the same parameters.
    two = trainer()
    two.train_step(ids())
    assert not np.array_equal(np.asarray(model.gradient_), np.asarray(two.gradient_))


def test_the_certified_batch_layout_is_refused_at_the_second_shape(fake_host):
    """A two-row batch is the right layout for the other shape and must not be
    accepted here, or a replay would compare a different sum than it recorded."""
    install(fake_host)
    model = host_mod.LanguageModelHostTrainer(np.ones(FOUR.n_total, np.float32), shape=FOUR)
    with pytest.raises(ValueError, match=r'ids must be \[4, 33\]'):
        model.train_step(ids())
    assert model.completed_steps == 0
