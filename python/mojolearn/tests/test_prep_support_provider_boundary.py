"""Selector support boundary controls; native runtime cases need integration.

The source test exercises only the Python forwarding contract. It does not
qualify the compiled operation. Runtime controls call the public aliases.
"""
import sys

import pytest


@pytest.fixture
def public_classes():
    # Delayed import allows the source-only selection to run without loading
    # unbuilt native modules. Run these controls after root's local compile.
    import mojolearn as ml
    from mojolearn import feature_selection
    assert ml.VarianceThreshold is feature_selection.VarianceThreshold
    assert ml.SelectKBest is feature_selection.SelectKBest
    return [feature_selection.VarianceThreshold, feature_selection.SelectKBest]


def _fitted(cls):
    model = cls()
    # get_support consumes only completed fitted metadata. No numerical fit
    # is necessary to exercise this complete public operation and its errors.
    model.n_features_in_ = 3
    model._mask = [True, False, True]
    return model


def test_public_support_preserves_list_values_and_ownership(public_classes):
    for cls in public_classes:
        model = _fitted(cls)
        assert model.get_support() == [True, False, True]
        assert model.get_support(indices=True) == [0, 2]
        result = model.get_support()
        assert type(result) is list
        assert result is not model._mask
        result[0] = False
        assert model._mask == [True, False, True]


def test_public_support_retains_unfitted_error(public_classes):
    for cls in public_classes:
        with pytest.raises(RuntimeError, match='not fitted yet'):
            cls().get_support()


def test_public_support_retains_missing_mask_error(public_classes):
    for cls in public_classes:
        model = cls()
        model.n_features_in_ = 3
        with pytest.raises(AttributeError) as error:
            model.get_support()
        assert str(error.value) == f"'{cls.__name__}' object has no attribute '_mask'"


def test_public_support_mask_copy_does_not_truth_test_items(public_classes):
    calls = []

    class Passive:
        def __bool__(self):
            calls.append('bool')
            raise AssertionError('passive support value was truth-tested')

    value = Passive()
    for cls in public_classes:
        model = _fitted(cls)
        model._mask = [value, False, True]
        result = model.get_support()
        assert result[0] is value
        assert calls == []


@pytest.mark.parametrize('location', ['indices', 'mask', 'item'])
def test_public_support_refuses_protocols_without_calling_them(public_classes, location):
    calls = []

    class Hook:
        def __bool__(self):
            calls.append('bool')
            return True

        def __iter__(self):
            calls.append('iter')
            return iter([True, False, True])

    for cls in public_classes:
        model = _fitted(cls)
        if location == 'mask':
            model._mask = Hook()
        elif location == 'item':
            model._mask = [Hook(), False, True]
        with pytest.raises(TypeError):
            model.get_support(indices=Hook() if location == 'indices' else True)
        assert calls == []


def test_public_support_has_no_python_preparation_or_result_frames(public_classes):
    for cls in public_classes:
        model = _fitted(cls)
        method = model.get_support
        calls = []

        def record(frame, event, arg):
            if event == 'call' and '/mojolearn/' in frame.f_code.co_filename:
                calls.append(frame.f_code)

        previous = sys.getprofile()
        sys.setprofile(record)
        try:
            result = method(indices=True)
        finally:
            sys.setprofile(previous)
        assert result == [0, 2]
        assert calls == [method.__func__.__code__]


def test_public_support_rejects_dictionary_property_before_hook(public_classes):
    calls = []
    for cls in public_classes:
        def dictionary(self):
            calls.append('__dict__')
            return {}

        # A heap type's existing __dict__ slot is not assignable. Define the
        # property on an ordinary subclass and bypass its constructor so the
        # probe reaches support admission, rather than constructor admission.
        changed = type('DictionaryHook', (cls,), {'__dict__': property(dictionary)})
        model = object.__new__(changed)
        method = cls.get_support.__get__(model, changed)
        with pytest.raises(TypeError):
            method()
        assert calls == []
