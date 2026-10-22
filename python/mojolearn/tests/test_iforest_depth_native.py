# SPDX-License-Identifier: Apache-2.0
"""Whole public fit preserves its consumed native depth-state lifecycle."""
import platform
import warnings

import numpy as np
import pytest

from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests import _legacy_host_iforest_fit as legacy
from mojolearn.tests.test_host_archive_native import _state
from mojolearn.tests.test_host_iforest_fit_native import _model
from mojolearn.tests.test_host_kde_fit_native import _x


def _marker(value):
    if isinstance(value, (np.ndarray, np.generic)):
        raw, dtype = value.tobytes(), value.dtype
        # SysV x87's six padding bytes are not part of the floating value;
        # NumPy does not promise their contents. Keep every significand,
        # sign and exponent byte, including each complex component.
        component = dtype.itemsize // (2 if dtype.kind == 'c' else 1)
        if dtype.char in ('g', 'G') and component == 16 and platform.machine() == 'x86_64':
            raw = b''.join(raw[i:i+10] for i in range(0, len(raw), 16))
        return type(value), dtype.str, dtype.char, value.shape, raw
    return _state(value)


def _pair(depth, *, action='always', error_mode='warn', body_fails=False):
    old, model = _model(n_estimators=1), _model(n_estimators=1)
    for item in (old, model):
        vars(item)['_mojolearn_fitting'] = depth
        if body_fails:
            vars(item)['max_samples'] = 0
    with np.errstate(over=error_mode), warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter(action)
        try:
            legacy.fit(old, _x())
            expected_error = None
        except Exception as error:
            expected_error = error
    with np.errstate(over=error_mode), warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter(action)
        audit = audit_native_call(model.fit, _x())
    if expected_error is None:
        assert audit.assert_native() is model
    else:
        assert type(audit.exception) is type(expected_error), audit.report()
        assert str(audit.exception) == str(expected_error)
        assert not audit.report()['python_frames'], audit.report()
        assert not audit.report()['forbidden_native_calls'], audit.report()
    assert [(w.category, str(w.message)) for w in actual_warnings] == [
        (w.category, str(w.message)) for w in expected_warnings]
    actual, expected = dict(vars(model)), dict(vars(old))
    assert _marker(actual.pop('_mojolearn_fitting', None)) == _marker(expected.pop('_mojolearn_fitting', None))
    assert _state(actual) == _state(expected)
    return model


def _form(scalar, form):
    return scalar if form == 'scalar' else np.asarray(scalar) if form == 'zero_d' else np.asarray([[scalar]])


@pytest.mark.parametrize('dtype', ['i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'i8', 'u8', 'q', 'Q'])
@pytest.mark.parametrize('form', ['scalar', 'zero_d', 'vector'])
def test_fit_depth_preserves_integer_promotion_wrap_and_warning_phases(dtype, form):
    _pair(_form(np.dtype(dtype).type(np.iinfo(dtype).max), form))


@pytest.mark.parametrize('form', ['scalar', 'zero_d', 'vector'])
def test_fit_depth_preserves_boolean_promotion(form):
    _pair(_form(np.bool_(True), form))


@pytest.mark.parametrize('dtype', ['f2', 'f4', 'f8', 'g'])
@pytest.mark.parametrize('form', ['scalar', 'zero_d', 'vector'])
def test_fit_depth_preserves_each_rounded_floating_operation(dtype, form):
    _pair(_form(np.dtype(dtype).type('.1'), form))


@pytest.mark.parametrize('dtype', ['f2', 'f4', 'f8', 'g'])
@pytest.mark.parametrize('form', ['scalar', 'vector'])
def test_fit_depth_preserves_round_to_nearest_even_boundary(dtype, form):
    _pair(_form(np.dtype(dtype).type(2 ** (np.finfo(dtype).nmant + 1)), form))


@pytest.mark.parametrize('dtype', ['f2', 'f4', 'f8', 'g'])
@pytest.mark.parametrize('form', ['scalar', 'vector'])
@pytest.mark.parametrize('negative', [False, True])
def test_fit_depth_preserves_maximum_finite_rounding(dtype, form, negative):
    value = np.finfo(dtype).max
    _pair(_form(-value if negative else value, form))


@pytest.mark.parametrize('dtype,value', [('f2', 'tiny'), ('g', 'tiny'), ('f8', 'nan'),
                                        ('g', 'nan'), ('f8', 'inf'), ('g', '-inf')])
def test_fit_depth_preserves_subnormal_and_special_state(dtype, value):
    scalar = (np.nextafter(np.dtype(dtype).type(0), np.dtype(dtype).type(1))
              if value == 'tiny' else np.dtype(dtype).type(value))
    _pair(np.asarray(scalar))


@pytest.mark.parametrize('dtype,word', [('f4', 0x7f800123), ('f8', 0x7ff0000000000123)])
@pytest.mark.parametrize('form', ['scalar', 'zero_d', 'vector'])
def test_fit_depth_preserves_signaling_nan_payload_and_warning_phase(dtype, word, form):
    scalar = np.frombuffer(word.to_bytes(np.dtype(dtype).itemsize, 'little'), dtype=dtype)[0]
    _pair(_form(scalar, form))


@pytest.mark.parametrize('component', ['real', 'imag'])
@pytest.mark.parametrize('form', ['scalar', 'zero_d'])
def test_fit_depth_preserves_complex_signaling_nan_promotion(component, form):
    word = (0x7f800123).to_bytes(4, 'little')
    raw = word + bytes(4) if component == 'real' else bytes(4) + word
    _pair(_form(np.frombuffer(raw, dtype='F')[0], form))


@pytest.mark.parametrize('form', ['scalar', 'zero_d'])
def test_fit_depth_signaling_nan_error_precedes_training_and_publication(form):
    value = np.frombuffer((0x7f800123).to_bytes(4, 'little'), dtype='f4')[0]
    _pair(_form(value, form), action='error')


@pytest.mark.parametrize('dtype,value', [('g', '.125'), ('G', 1.125+.5j)])
@pytest.mark.parametrize('shape', [(), (1, 1)])
def test_fit_depth_preserves_extended_non_native_byteorder(dtype, value, shape):
    _pair(np.asarray(value, dtype=np.dtype(dtype).newbyteorder('>')).reshape(shape))


@pytest.mark.parametrize('depth', [complex(1, 2), complex(0, -0.)])
def test_fit_depth_preserves_builtin_complex_finally_error(depth):
    _pair(depth)


@pytest.mark.parametrize('dtype', ['F', 'D', 'G'])
@pytest.mark.parametrize('form', ['scalar', 'vector'])
@pytest.mark.parametrize('value', [1j, complex(1, np.nan), complex(np.nan, 1), complex(-0., -0.)])
def test_fit_depth_preserves_native_complex_state_and_ordering(dtype, form, value):
    _pair(_form(np.dtype(dtype).type(value), form))


@pytest.mark.parametrize('form', ['scalar', 'zero_d'])
@pytest.mark.parametrize('dtype', ['i1', 'i8'])
@pytest.mark.parametrize('action,error_mode', [('error', 'warn'), ('always', 'raise'), ('error', 'ignore')])
def test_fit_depth_overflow_error_timing_preserves_consumed_state(form, dtype, action, error_mode):
    _pair(_form(np.dtype(dtype).type(np.iinfo(dtype).max), form), action=action, error_mode=error_mode)


@pytest.mark.parametrize('depth', [np.float16('.1'), np.asarray([[np.float32('.1')]]),
                                   np.longdouble('.1'), np.longlong(1)])
def test_fit_depth_finally_preserves_state_after_body_failure(depth):
    _pair(depth, body_fails=True)


def test_fit_depth_does_not_dispatch_numpy_numeric_operations():
    callback = getattr(np, 'set_numeric_ops', None)
    calls, saved = [], None
    def forbidden(*args, **kwargs):
        calls.append('numpy_numeric')
        raise AssertionError('native fit dispatched NumPy arithmetic')
    depth = np.asarray([[np.float32('.1')]])
    old, model = _model(n_estimators=1), _model(n_estimators=1)
    vars(old)['_mojolearn_fitting'] = depth
    vars(model)['_mojolearn_fitting'] = depth
    legacy.fit(old, _x())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        if callback is not None:
            saved = callback(add=forbidden, subtract=forbidden, greater=forbidden)
        try:
            audit = audit_native_call(model.fit, _x())
        finally:
            if saved is not None:
                callback(**saved)
    assert audit.assert_native() is model
    assert calls == []
    assert _marker(vars(model)['_mojolearn_fitting']) == _marker(vars(old)['_mojolearn_fitting'])


@pytest.mark.parametrize('action', ['call', 'log'])
def test_fit_depth_refuses_numpy_error_callbacks_before_execution(action):
    calls = []
    class Log:
        def write(self, text):
            calls.append(text)
    def callback(*args):
        calls.append(args)
    model = _model(n_estimators=1)
    vars(model)['_mojolearn_fitting'] = np.int64(np.iinfo('i8').max)
    with np.errstate(over=action, call=callback if action == 'call' else Log()):
        audit = audit_native_call(model.fit, _x())
    assert isinstance(audit.exception, TypeError), audit.report()
    assert calls == []
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']
    assert '_x' not in vars(model)


@pytest.mark.parametrize('form', ['object', 'object_array', 'integer_subclass', 'numpy_subclass'])
def test_fit_depth_refuses_application_truth_and_arithmetic_hooks(form):
    calls = []
    def forbidden(*args):
        calls.append(form)
        raise AssertionError('application depth hook ran')
    base = int if form == 'integer_subclass' else np.float32 if form == 'numpy_subclass' else object
    cls = type('ForeignDepth', (base,), {name: forbidden for name in ('__bool__', '__add__', '__sub__', '__gt__')})
    value = cls(1) if base is not object else cls()
    if form == 'object_array':
        value = np.asarray(value, dtype=object)
    model = _model(n_estimators=1)
    vars(model)['_mojolearn_fitting'] = value
    audit = audit_native_call(model.fit, _x())
    assert isinstance(audit.exception, TypeError), audit.report()
    assert calls == []
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']
    assert vars(model)['_mojolearn_fitting'] is value
    assert '_x' not in vars(model)
