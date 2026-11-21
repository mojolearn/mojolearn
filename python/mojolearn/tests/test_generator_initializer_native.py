# SPDX-License-Identifier: Apache-2.0
"""Initializer scalar rounding and complete draw-boundary regression oracles."""
import pytest

from mojolearn import _native_samba_bridge as bridge
from mojolearn import _portable_math as _pm
from mojolearn import _training_impl as training
from mojolearn._verify_native_runtime import audit_native_call
from .test_generator_draw_native import LegacyDraws, bits, checked


class LegacyInitializers(LegacyDraws):
    # Test-only copies of the pre-migration public formulas and their order.
    def kaiming_uniform(self, shape, fan_in, a=None):
        if a is None:
            a = _pm.sqrt(5.)
        gain = _pm.sqrt(2. / (1. + float(a)*float(a)))
        bound = gain * _pm.sqrt(3. / float(int(fan_in)))
        return self.uniform(shape, -bound, bound)

    def xavier_uniform(self, shape, fan_in, fan_out, gain=1.):
        bound = float(gain) * _pm.sqrt(6. / float(int(fan_in)+int(fan_out)))
        return self.uniform(shape, -bound, bound)

    def kaiming_normal(self, shape, fan_in, gain=None):
        if gain is None:
            gain = _pm.sqrt(2.)
        return self.normal(shape, 0., float(gain)/_pm.sqrt(float(int(fan_in))))

    def xavier_normal(self, shape, fan_in, fan_out, gain=1.):
        return self.normal(shape, 0., float(gain)*_pm.sqrt(2. / float(int(fan_in)+int(fan_out))))


CASES = [
    ('kaiming_uniform', dict(fan_in=13)),
    ('kaiming_uniform', dict(fan_in=17, a=.231456789)),
    ('xavier_uniform', dict(fan_in=11, fan_out=7)),
    ('xavier_uniform', dict(fan_in=13, fan_out=19, gain=.71234567)),
    ('kaiming_normal', dict(fan_in=13)),
    ('kaiming_normal', dict(fan_in=17, gain=.231456789)),
    ('xavier_normal', dict(fan_in=11, fan_out=7)),
    ('xavier_normal', dict(fan_in=13, fan_out=19, gain=.71234567)),
]


def refusal(function, expected_type, *args, **kwargs):
    audit = audit_native_call(function, *args, **kwargs)
    assert type(audit.exception) is expected_type, audit.report()
    for key in ('python_frames', 'forbidden_native_calls', 'wrapper_work', 'unobserved_existing_threads'):
        assert not audit.report()[key], audit.report()


@pytest.mark.parametrize('shape', [(1,), (3, 7), (2, 3, 5)])
@pytest.mark.parametrize('operation,options', CASES)
def test_initializer_preserves_scalar_rounding_abi_and_seeded_draw(shape, operation, options):
    gen, old = training.Generator(0x12345678abcdef01), LegacyInitializers(0x12345678abcdef01)
    for _ in range(2):
        expected = getattr(old, operation)(shape, **options)
        bits(checked(getattr(gen, operation), shape, **options), expected)
    assert gen.state_dict() == dict(seed=old.seed, counter=old.counter)


@pytest.mark.parametrize('operation,options,error,advance', [
    ('kaiming_uniform', dict(fan_in=0), ZeroDivisionError, 0),
    ('kaiming_uniform', dict(fan_in=-2), ValueError, 0),
    ('xavier_uniform', dict(fan_in=2, fan_out=-2), ZeroDivisionError, 0),
    ('xavier_uniform', dict(fan_in=2, fan_out=3, gain=-1), ValueError, 0),
    ('kaiming_normal', dict(fan_in=0), ZeroDivisionError, 0),
    ('kaiming_normal', dict(fan_in=-2), ValueError, 0),
    ('xavier_normal', dict(fan_in=2, fan_out=3, gain=-1), ValueError, 0),
    ('xavier_normal', dict(fan_in=2, fan_out=3, gain=float('inf')), Exception, 1),
])
def test_initializer_refusals_preserve_order_and_stream_consumption(operation, options, error, advance):
    gen, old = training.Generator(3), LegacyInitializers(3)
    with pytest.raises(error):
        getattr(old, operation)((3, 7), **options)
    refusal(getattr(gen, operation), error, (3, 7), **options)
    assert gen.counter == old.counter == advance


def test_initializer_runs_without_python_math_or_loader(monkeypatch):
    shape, options = (3, 7), dict(fan_in=13)
    expected = LegacyInitializers(5).kaiming_uniform(shape, **options)
    gen = training.Generator(5)

    def poison(*args, **kwargs):
        raise AssertionError('Python portable-math wrapper or loader executed')

    with monkeypatch.context() as patch:
        patch.setattr(_pm, 'sqrt', poison)  # _training_impl no longer imports _portable_math (b53aac9ac4)
        patch.setattr(_pm, '_load', poison)
        bits(checked(gen.kaiming_uniform, shape, **options), expected)


@pytest.mark.parametrize('operation,options', CASES[::2])
def test_initializer_needs_no_scalar_library_entry(operation, options):
    # The square roots are native binary64 in the compiled door: the bridge
    # holds no scalar entry and no RNG symbol for it to call.
    assert 'generator_sqrt' not in bridge.CONTEXT and 'generator_sqrt_type' not in bridge.CONTEXT
    assert all(value is True or isinstance(value, str) for value in bridge.CONTEXT['generator_bindings'].values())
    gen, old = training.Generator(7), LegacyInitializers(7)
    bits(checked(getattr(gen, operation), (3, 7), **options), getattr(old, operation)((3, 7), **options))
    assert gen.counter == old.counter == 1
