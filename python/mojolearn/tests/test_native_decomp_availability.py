"""Native attribute and availability behavior for non-transforming owners."""
import pytest

from mojolearn import _expansion_decomp as decomp
from mojolearn import _native_public_attributes as connection
from mojolearn._verify_native_runtime import audit_native_call


NAMES = ('MinCovDet', 'EllipticEnvelope', 'AlternatingLeastSquares')


def clean(operation, *args, error=None):
    audit = audit_native_call(operation, *args)
    report = audit.report()
    assert not any(report[key] for key in ('python_frames', 'wrapper_work',
        'forbidden_native_calls', 'unresolved_native_calls',
        'unobserved_existing_threads')), report
    if error:
        assert isinstance(audit.exception, error), report
        return audit.exception
    return audit.assert_native()


@pytest.mark.parametrize('name', NAMES)
def test_public_attribute_slot_preserves_instance_references(name):
    cls = getattr(decomp, name)
    model = cls()
    assert vars(cls)['__getattribute__'] is connection._CONTEXT['descriptor']
    value = object()
    vars(model)['note'] = value
    assert clean(getattr, model, 'note') is value


@pytest.mark.parametrize('name', NAMES)
def test_unavailable_transform_preserves_attribute_error_without_python(name):
    model = getattr(decomp, name)()
    error = clean(getattr, model, 'fit_transform', error=AttributeError)
    assert "has no attribute 'fit_transform'" in str(error)
    assert clean(hasattr, model, 'fit_transform') is False


def test_changed_predicate_is_refused_without_running_it():
    model = decomp.MinCovDet()
    descriptor = vars(decomp.MinCovDet)['fit_transform']
    fields = vars(descriptor)
    original = fields['check']
    calls = []
    fields['check'] = lambda self: calls.append('check')
    try:
        clean(getattr, model, 'fit_transform', error=TypeError)
    finally:
        fields['check'] = original
    assert not calls


def test_same_predicate_identity_with_changed_code_is_refused():
    model = decomp.MinCovDet()
    check = vars(vars(decomp.MinCovDet)['fit_transform'])['check']
    original = check.__code__
    check.__code__ = (lambda self: True).__code__
    try:
        clean(getattr, model, 'fit_transform', error=TypeError)
    finally:
        check.__code__ = original
