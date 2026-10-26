"""Public constructor probes for the 77 full-reset owners and the 52 declared
prep/decomp owners; run only against the integrated native build
(bindings/full_reset_constructor_native.mojo).

The oracle is the original constructor body retained at install
(`_CONTEXT['originals']`), executed here by the test on a fresh object. It is
never called by the product. A subclass's original body reaches its base
through `super().__init__`, which is the base's native constructor; the
schema proof (tools/native_full_reset_constructor_schemas.json) pins the
complete original write sequence independently.
"""
import array
import importlib
import inspect
import json
from pathlib import Path

import pytest

import mojolearn as ml
from mojolearn._native_full_reset_params import _CLASSES as _EAGER, _CONTEXT, _DEFERRED_MODULES
from mojolearn import _native_declared_constructors as _declared
from mojolearn._verify_native_runtime import audit_native_call

# The NumPy-backed owners connect when their module loads (connect_module);
# importing them here covers all 77.
_CLASSES = _EAGER + tuple(getattr(importlib.import_module('mojolearn.' + module), name)
                          for module, names in _DEFERRED_MODULES.items() for name in names)
PROOF = Path(__file__).resolve().parents[3] / 'tools/native_full_reset_constructor_schemas.json'
CONTEXTS = {cls: _CONTEXT for cls in _CLASSES} | {cls: _declared._CONTEXT for cls in _declared._CLASSES}
OWNERS = tuple(CONTEXTS)


def clean(operation, *args, error=None, **kwargs):
    audit = audit_native_call(operation, *args, **kwargs)
    report = audit.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls',
        'unresolved_native_calls', 'wrapper_work', 'unobserved_existing_threads')), report
    if error is not None:
        assert isinstance(audit.exception, error), report
        return audit.exception
    return audit.assert_native()


def construct(cls, *args, **kwargs):
    model = object.__new__(cls)
    clean(model.__init__, *args, **kwargs)
    return model


def oracle(cls, *args, owner=None, **kwargs):
    model = object.__new__(cls)
    CONTEXTS[owner or cls]['originals'][owner or cls](model, *args, **kwargs)
    return model


def oracle_error(cls, *args, **kwargs):
    with pytest.raises(TypeError) as caught:
        oracle(cls, *args, **kwargs)
    return str(caught.value)


def assert_same(actual, expected):
    assert tuple(vars(actual)) == tuple(vars(expected))
    for name, value in vars(expected).items():
        current = vars(actual)[name]
        if type(value) in (str, int, float, bool, type(None)):
            assert current == value and type(current) is type(value)
        else:
            assert current is value


def sentinels(cls):
    return {name: object() for name in inspect.signature(cls).parameters}


def required(cls):
    return {name: object() for name, p in inspect.signature(cls).parameters.items() if p.default is p.empty}


@pytest.mark.parametrize('cls', OWNERS, ids=lambda cls: cls.__name__)
def test_installed_constructor_is_the_native_call_with_the_public_signature(cls):
    context = CONTEXTS[cls]
    original = context['originals'][cls]
    assert cls.__init__ is not original
    assert inspect.signature(cls.__init__) is context['plans'][cls]['signature']
    assert tuple(inspect.signature(cls).parameters) == context['plans'][cls]['names']


@pytest.mark.parametrize('cls', OWNERS, ids=lambda cls: cls.__name__)
def test_every_keyword_matches_the_original_order_and_reference_identity(cls):
    values = sentinels(cls)
    assert_same(construct(cls, **values), oracle(cls, **values))


@pytest.mark.parametrize('cls', _CLASSES, ids=lambda cls: cls.__name__)
def test_schema_proof_final_fields_and_order(cls):
    rows = {row['declared_class']: row for row in json.loads(PROOF.read_text())['classes']}
    row = rows[cls.__name__]
    values = sentinels(cls)
    model = construct(cls, **values)
    assert list(vars(model)) == row['field_order']
    for name, source in row['final_fields'].items():
        if 'parameter' in source:
            assert vars(model)[name] is values[source['parameter']]
        else:
            assert vars(model)[name] == source['literal']


@pytest.mark.parametrize('cls', OWNERS, ids=lambda cls: cls.__name__)
def test_positional_binding_and_defaults_match_the_original(cls):
    plan = CONTEXTS[cls]['plans'][cls]
    values = sentinels(cls)
    args = tuple(values[name] for name in plan['positional'])
    assert_same(construct(cls, *args), oracle(cls, *args))
    needed = required(cls)
    assert_same(construct(cls, **needed), oracle(cls, **needed))


@pytest.mark.parametrize('cls', OWNERS, ids=lambda cls: cls.__name__)
def test_argument_errors_are_the_original_words_without_mutation(cls):
    plan = CONTEXTS[cls]['plans'][cls]
    needed = required(cls)
    first = plan['names'][0]
    cases = [((), dict(needed, not_a_parameter=1)),
             ((object(),) * (len(plan['positional']) + 2), {}),
             ((object(),), dict(needed, **{first: object()})) if plan['positional'] else ((), needed | {'zz': 1})]
    if needed:
        cases.append(((), {}))
    for args, kwargs in cases:
        expected = oracle_error(cls, *args, **kwargs)
        model = object.__new__(cls)
        error = clean(model.__init__, *args, error=TypeError, **kwargs)
        assert str(error) == expected
        assert vars(model) == {}


@pytest.mark.parametrize('cls', OWNERS, ids=lambda cls: cls.__name__)
def test_reinitialization_keeps_unrelated_fields_in_place(cls):
    values = sentinels(cls)
    model = construct(cls, **values)
    marker = array.array('f', [1.0])
    vars(model)['fitted_'] = marker
    replacement = dict(values, **{next(iter(values)): object()})
    clean(model.__init__, **replacement)
    expected = oracle(cls, **values)
    vars(expected)['fitted_'] = marker
    CONTEXTS[cls]['originals'][cls](expected, **replacement)
    assert_same(model, expected)


def test_numeric_mode_rule_matches_the_wrapper():
    for mode in (None, 'identical'):
        assert_same(construct(ml.KMeans, numeric_mode=mode), oracle(ml.KMeans, numeric_mode=mode))
    model = construct(ml.KMeans, numeric_mode='identical')
    clean(model.__init__, n_clusters=3)
    expected = oracle(ml.KMeans, numeric_mode='identical')
    _CONTEXT['originals'][ml.KMeans](expected, n_clusters=3)
    assert_same(model, expected)
    assert vars(model)['numeric_mode'] == 'identical'


def test_passive_subclass_class_mode_and_user_initializer():
    class Passive(ml.KernelRidge):
        numeric_mode = 'identical'
    model = Passive(alpha=2.0)
    assert 'numeric_mode' not in vars(model) and model.numeric_mode == 'identical'

    class User(ml.KMeans):
        def __init__(self, n_clusters=3, note=None):
            super().__init__(n_clusters=n_clusters)
            self.note = note
    model = User(note='kept')
    expected = object.__new__(User)
    _CONTEXT['originals'][ml.KMeans](expected, n_clusters=3)
    vars(expected)['note'] = 'kept'
    assert list(vars(model)) == list(vars(expected)) and vars(model) == vars(expected)
    assert User().get_params(deep=False)['n_clusters'] == 3


def test_sklearn_style_clone_round_trip():
    model = ml.KMeans(n_clusters=4, random_state=7, numeric_mode='identical')
    params = model.get_params(deep=False)
    clone = type(model)(**params)
    assert clone.get_params(deep=False) == params


@pytest.mark.parametrize('hook', ['__setattr__', '__getattr__', '__getattribute__'])
def test_python_attribute_protocols_are_refused_before_any_write(hook):
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('callback executed')
    Changed = type('Changed', (ml.KernelRidge,), {hook: forbidden})
    model = object.__new__(Changed)
    clean(ml.KernelRidge.__init__, model, alpha=2.0, error=TypeError)
    assert calls == [] and object.__getattribute__(model, '__dict__') == {}


@pytest.mark.parametrize('field', ['alpha', 'numeric_mode'])
def test_field_descriptors_are_refused_without_invocation(field):
    calls = []
    class Descriptor:
        def __get__(self, obj, cls):
            calls.append('get')
        def __set__(self, obj, value):
            calls.append('set')
    Changed = type('Changed', (ml.KernelRidge,), {field: Descriptor()})
    model = object.__new__(Changed)
    clean(ml.KernelRidge.__init__, model, alpha=2.0, error=TypeError)
    assert calls == [] and object.__getattribute__(model, '__dict__') == {}


def test_reinitialization_refuses_an_old_python_finalizer_before_writing():
    calls = []
    class Owner:
        def __del__(self):
            calls.append('del')
    model = ml.KernelRidge(alpha=Owner())
    before = dict(vars(model))
    clean(model.__init__, alpha=1.0, error=TypeError)
    assert vars(model) == before and calls == []
    vars(model)['alpha'] = 1.0


def test_foreign_instance_is_refused():
    clean(ml.KMeans.__init__, ml.KernelRidge(), error=TypeError)


def test_declared_reinitialization_uses_the_declared_release_admission():
    calls = []
    class Owner:
        def __del__(self):
            calls.append('del')
    model = ml.SimpleImputer(fill_value=Owner())
    before = dict(vars(model))
    clean(model.__init__, fill_value=0.0, error=TypeError)
    assert vars(model) == before and calls == []
    vars(model)['fill_value'] = 0.0


def test_declared_super_chain_matches_the_original():
    for cls, kwargs in ((ml.EllipticEnvelope, dict(contamination=0.2, random_state=3)),
                        (ml.PLSCanonical, dict(algorithm='svd', n_components=1))):
        assert_same(construct(cls, **kwargs), oracle(cls, **kwargs))
