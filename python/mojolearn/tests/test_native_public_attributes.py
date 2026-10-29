# SPDX-License-Identifier: Apache-2.0
"""Installed public attribute slots, descriptor hooks, and zero-copy outputs."""
import subprocess
import sys
import textwrap

import pytest

from mojolearn import KMeans, Cholesky
from mojolearn._array import Array
from mojolearn import _native_public_attributes as owner
from mojolearn._verify_native_runtime import audit_native_call


def clean(operation, *args, error=None):
    audit = audit_native_call(operation, *args)
    report = audit.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls',
                                          'unresolved_native_calls', 'wrapper_work',
                                          'unobserved_existing_threads')), report
    if error is not None:
        assert isinstance(audit.exception, error), report
        return audit.exception
    return audit.assert_native()


def reader(module='attribute_user', **markers):
    # Testing controller creates one attribute-read boundary. All production
    # lookup/output work executes through the installed model's native slot.
    namespace = {'__name__': module, **markers}
    exec('def read(model, name):\n    return getattr(model, name)\n', namespace)
    return namespace['read']


def model_with_output(cls=KMeans):
    model = object.__new__(cls)
    value = Array((2, 3), '<f4')
    fields = object.__getattribute__(model, '__dict__')
    fields.update(payload_=value, _payload_=value, payload=value, user_note=object())
    return model, value, fields


def test_actual_public_attribute_slot_and_ordinary_method_binding():
    model, value, fields = model_with_output()
    assert vars(KMeans)['__getattribute__'] is owner._CONTEXT['descriptor']
    assert clean(reader(), model, 'user_note') is fields['user_note']
    bound = clean(reader(), model, 'get_params')
    assert bound.__self__ is model
    assert clean(reader(), model, '__dict__') is fields
    assert clean(reader(), model, '__class__') is KMeans
    clean(reader(), model, 'absent', error=AttributeError)
    clean(reader(), model, '', error=AttributeError)


def test_zero_copy_fitted_array_and_private_or_unfitted_values():
    np = pytest.importorskip('numpy')
    model, value, _ = model_with_output()
    output = clean(reader(), model, 'payload_')
    assert type(output) is np.ndarray
    assert output.shape == value.shape and output.dtype.str == value.dtype
    assert np.shares_memory(output, np.asarray(value))
    output[0, 0] = 7
    assert np.asarray(value)[0, 0] == 7
    assert clean(reader(), model, '_payload_') is value
    assert clean(reader(), model, 'payload') is value


@pytest.mark.parametrize('module,markers,outside', [
    ('mojolearn', {}, False), ('mojolearn.internal', {}, False),
    ('mojolearn_verify_all_probe', {}, False), ('application', {}, True),
    ('application', {'__mojolearn_internal__': True}, False),
    ('mojolearn.internal', {'__mojolearn_internal__': True, '__mojolearn_user_boundary__': True}, True),
])
def test_caller_boundary_and_marker_precedence(module, markers, outside):
    np = pytest.importorskip('numpy')
    model, value, _ = model_with_output()
    result = clean(reader(module, **markers), model, 'payload_')
    assert (type(result) is np.ndarray) is outside
    if not outside:
        assert result is value


def test_only_mode_wrapper_names_skip_their_frame(monkeypatch):
    np = pytest.importorskip('numpy')
    import mojolearn._verify_native_runtime as verifier
    monkeypatch.setattr(verifier, '__mojolearn_user_boundary__', True, raising=False)
    model, value, _ = model_with_output()
    namespace = {'__name__': 'mojolearn._mode'}
    for name in ('public', 'guarded', 'tracked', 'load', 'other'):
        exec(f'def {name}(model, name):\n    return getattr(model, name)\n', namespace)
        result = clean(namespace[name], model, 'payload_')
        assert (type(result) is np.ndarray) is (name != 'other')
        if name == 'other':
            assert result is value


def test_passive_subclass_fields_and_bound_functions():
    class Child(KMeans):
        label = 'plain'
        def own_method(self):
            raise AssertionError('binding must not execute a Python method')
    model, _, _ = model_with_output(Child)
    assert clean(reader(), model, 'label') == 'plain'
    bound = clean(reader(), model, 'own_method')
    assert bound.__self__ is model and bound.__func__ is Child.own_method


def test_function_get_override_is_not_executed_during_binding():
    calls = []
    class Child(KMeans):
        def own_method(self):
            raise AssertionError('binding must not execute a Python method')
    Child.own_method.__get__ = lambda *args: calls.append('callback')
    model, _, _ = model_with_output(Child)
    bound = clean(reader(), model, 'own_method')
    assert bound.__self__ is model and bound.__func__ is Child.own_method
    assert calls == []


def test_python_property_is_refused_before_its_hook():
    calls = []
    class Child(KMeans):
        @property
        def data(self):
            calls.append('property')
            return 1
    model, _, _ = model_with_output(Child)
    clean(reader(), model, 'data', error=TypeError)
    assert not calls


def test_python_getattr_is_refused_before_fallback():
    calls = []
    class Child(KMeans):
        def __getattr__(self, name):
            calls.append(name)
            return 1
    model, _, _ = model_with_output(Child)
    clean(reader(), model, 'missing', error=TypeError)
    assert not calls


def test_descriptor_precedence_and_mutated_descriptor_refusal(monkeypatch):
    calls = []
    model, _, fields = model_with_output()
    fields['user_note'] = 'instance'
    monkeypatch.setattr(KMeans, 'user_note', property(lambda self: calls.append('called')), raising=False)
    clean(reader(), model, 'user_note', error=TypeError)
    assert not calls


def test_cholesky_logdet_preserves_not_fitted_failed_and_success_states():
    model = object.__new__(Cholesky)
    fields = object.__getattribute__(model, '__dict__')
    clean(reader(), model, 'logdet_', error=AttributeError)
    fields.update(info_=3, _logdet=1.25)
    error = clean(reader(), model, 'logdet_', error=ValueError)
    assert 'info=3' in str(error)
    fields['info_'] = 0
    assert clean(reader(), model, 'logdet_') == 1.25


def test_stl_config_uses_native_field_reads_after_lazy_import():
    pytest.importorskip('numpy')
    from mojolearn import STL
    model = object.__new__(STL)
    fields = object.__getattribute__(model, '__dict__')
    names = ('period', 'seasonal', 'seasonal_deg', 'seasonal_jump', 'trend', 'trend_deg', 'trend_jump',
             'low_pass', 'low_pass_deg', 'low_pass_jump', 'robust')
    fields.update((name, index) for index, name in enumerate(names))
    result = clean(reader(), model, 'config')
    assert result == {name: fields[name] for name in names}
    assert result is not fields


def run_isolated(source):
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(source)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_numpy_loaded_after_mojolearn_changes_only_the_output_view():
    pytest.importorskip('numpy')
    run_isolated('''
        import sys
        import mojolearn
        from mojolearn._array import Array
        assert 'numpy' not in sys.modules
        model = object.__new__(mojolearn.KMeans)
        value = Array((2,), '<f4')
        object.__getattribute__(model, '__dict__')['payload_'] = value
        assert model.payload_ is value
        import numpy as np
        output = model.payload_
        assert type(output) is np.ndarray
        assert np.shares_memory(output, np.asarray(value))
    ''')


def test_no_frame_or_array_protocol_audit_callbacks():
    pytest.importorskip('numpy')
    run_isolated('''
        import sys
        import numpy as np
        from mojolearn import KMeans
        from mojolearn._array import Array
        model = object.__new__(KMeans)
        value = Array((2,), '<f4')
        object.__getattribute__(model, '__dict__')['payload_'] = value
        events = []
        def audit(event, args):
            if event == 'sys._getframe' or event == 'object.__getattr__':
                events.append(event)
        sys.addaudithook(audit)
        output = model.payload_
        assert type(output) is np.ndarray
        assert not events, events
    ''')


def test_pandas_import_after_mojolearn_keeps_opaque_get_params_leaf():
    pytest.importorskip('pandas')
    run_isolated('''
        import sys
        from mojolearn import BaggingRegressor
        from mojolearn._verify_native_runtime import audit_native_call
        assert 'pandas' not in sys.modules
        model = BaggingRegressor(numeric_mode='identical')
        getter = object.__getattribute__(model, 'get_params')
        import pandas as pd
        value = pd.DataFrame({'get_params': [1], 'x': [2]})
        object.__getattribute__(model, '__dict__')['estimator'] = value
        result = audit_native_call(getter).assert_native()
        assert result['estimator'] is value
        assert not any(name.startswith('estimator__') for name in result)
    ''')


def test_installed_subset_and_unported_paths_are_both_explicit():
    from mojolearn import BaggingRegressor, Ridge
    from mojolearn import _mode
    context = owner._CONTEXT
    for cls in (KMeans, Cholesky):
        assert cls in context['plans']
        assert vars(cls)['__getattribute__'] is context['descriptor']
    assert BaggingRegressor in context['pending']
    assert vars(BaggingRegressor)['__getattribute__'] is vars(_mode.PublicOutputsMixin)['__getattribute__']
    # An existing native attribute owner retains its own descriptor/context.
    assert next(vars(base)['__getattribute__'] for base in Ridge.__mro__ if '__getattribute__' in vars(base)) is not context['descriptor']
    assert Ridge not in context['plans']


@pytest.mark.parametrize('order,dtype,readonly', [('C', 'f4', False), ('F', 'f4', False),
                                                 ('C', 'f2', False), ('F', 'f8', True)])
def test_external_storage_view_survives_model_and_array_lifetime(order, dtype, readonly):
    import gc
    import weakref
    np = pytest.importorskip('numpy')
    storage = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype, order=order)
    if readonly:
        storage.flags.writeable = False
    reference = weakref.ref(storage)
    value = Array.from_buffer(storage)
    model = object.__new__(KMeans)
    object.__getattribute__(model, '__dict__')['payload_'] = value
    output = clean(reader(), model, 'payload_')
    assert np.shares_memory(output, storage)
    assert output.strides == storage.strides
    assert output.flags.writeable is (not readonly)
    del model, value, storage
    gc.collect()
    assert reference() is not None
    np.testing.assert_array_equal(output, [[1, 2, 3], [4, 5, 6]])
    del output
    gc.collect()
    assert reference() is None


def test_array_row_offset_and_empty_storage_views():
    np = pytest.importorskip('numpy')
    value = Array.from_buffer(np.array([[1, 2], [3, 4]], dtype='f4'))
    row = value._row_view(1, (2,), 2)
    model = object.__new__(KMeans)
    fields = object.__getattribute__(model, '__dict__')
    fields['payload_'] = row
    result = clean(reader(), model, 'payload_')
    np.testing.assert_array_equal(result, [3, 4])
    assert np.shares_memory(result, np.asarray(value))
    fields['payload_'] = Array((0, 3), '<f4')
    result = clean(reader(), model, 'payload_')
    assert result.shape == (0, 3) and result.size == 0


def test_output_storage_python_buffer_hook_refuses_before_invocation():
    pytest.importorskip('numpy')
    calls = []
    class Exporter:
        def __buffer__(self, flags):
            calls.append('buffer')
            return memoryview(b'1234')
    model, value, _ = model_with_output()
    value._store = Exporter()
    clean(reader(), model, 'payload_', error=TypeError)
    assert not calls
